
"""
export_examples.py  — 导出评估样例（含原文、标签、预测与概率）
- 自动识别 checkpoint 类型：LoRA（adapter_config.json）或 Full-FT（config.json/model.safetensors）
- 国内镜像/缓存（hf-mirror）+ 自动禁用缺失的 hf_transfer
- 支持完全离线（MODEL_LOCAL_DIR/DATASET_LOCAL_DIR）
- 支持 GPU 推理、进度条、限制样本数与批量大小（BATCH / N）
"""

import os
import sys
import json
import importlib.util as _importlib_util

# ---------- 1) 国内镜像与缓存（必须在导入 transformers/datasets 之前设置） ----------
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("HF_HOME", hf_home)
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_home, "transformers"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(hf_home, "datasets"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
if not _importlib_util.find_spec("hf_transfer"):
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
else:
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch
import numpy as np

# ---------- 2) 基本配置（可按需修改） ----------
OUT_DIR = os.environ.get("OUT_DIR", "runs/lora_imdb_r4")
CKPT    = os.environ.get("CKPT", None)  # 如果不填，自动挑 OUT_DIR 里最后一个 checkpoint-*
MODEL   = os.environ.get("MODEL_NAME", "distilbert-base-uncased")

MODEL_LOCAL_DIR   = os.environ.get("MODEL_LOCAL_DIR")      # 完全离线模型目录（可选）
DATASET_LOCAL_DIR = os.environ.get("DATASET_LOCAL_DIR")    # 完全离线数据集目录（可选）

CACHE_DIR = os.environ.get("HF_HOME")
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", 256))
BATCH = int(os.environ.get("BATCH", 64))   # 批量大小，按显存调整
N = os.environ.get("N")                    # 只导出前 N 条，默认全部
N = int(N) if N is not None else None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

def _pick_latest_checkpoint(out_dir):
    if not os.path.isdir(out_dir):
        return None
    cands = [d for d in os.listdir(out_dir) if d.startswith("checkpoint-")]
    if not cands:
        return None
    cands.sort(key=lambda x: int(x.split("-")[-1]))
    return os.path.join(out_dir, cands[-1])

if CKPT is None:
    CKPT = _pick_latest_checkpoint(OUT_DIR)
    if CKPT is None:
        print(f"[ERROR] 没找到 checkpoint，检查 OUT_DIR={OUT_DIR}")
        sys.exit(1)

# ---------- 3) 识别 checkpoint 类型 ----------
def _is_lora_ckpt(path):
    return os.path.isfile(os.path.join(path, "adapter_config.json"))

def _is_full_ckpt(path):
    return os.path.isfile(os.path.join(path, "config.json")) or os.path.isfile(os.path.join(path, "model.safetensors"))

CKPT_IS_LORA = _is_lora_ckpt(CKPT)
CKPT_IS_FULL = _is_full_ckpt(CKPT)

print(f"[INFO] OUT_DIR={OUT_DIR}")
print(f"[INFO] CKPT={CKPT}")
print(f"[INFO] MODEL={MODEL}")
print(f"[INFO] DEVICE={device}")
print(f"[INFO] HF_ENDPOINT={os.environ.get('HF_ENDPOINT')}")
print(f"[INFO] HF_HOME={os.environ.get('HF_HOME')}")
print(f"[INFO] HF_HUB_ENABLE_HF_TRANSFER={os.environ.get('HF_HUB_ENABLE_HF_TRANSFER')}")
print(f"[INFO] CKPT type: {'LoRA' if CKPT_IS_LORA else ('Full-FT' if CKPT_IS_FULL else 'Unknown')}")

# ---------- 4) 加载分词器 ----------
def load_tokenizer(model_name, model_local_dir=None, ckpt_dir=None, cache_dir=None):
    # 优先用 checkpoint（full-FT 通常自带 tokenizer 文件）
    for name, local_only in [(ckpt_dir, True), (model_local_dir, True), (model_name, False)]:
        if name and (not local_only or os.path.isdir(name)):
            try:
                tok = AutoTokenizer.from_pretrained(
                    name, use_fast=True, cache_dir=cache_dir,
                    local_files_only=local_only
                )
                return tok
            except Exception as e:
                print(f"[WARN] Load tokenizer from {name} failed:", e)
    raise RuntimeError("无法加载分词器，请检查 MODEL/MODEL_LOCAL_DIR/CKPT 是否可用")

# ---------- 5) 加载模型（自动区分 LoRA 与 Full） ----------
def load_model(model_name, ckpt_dir, num_labels=2, model_local_dir=None, cache_dir=None, is_lora=False, is_full=False):
    if is_lora:
        # LoRA: 先加载基座，再注入适配器
        try_name = model_local_dir if (model_local_dir and os.path.isdir(model_local_dir)) else model_name
        try:
            base = AutoModelForSequenceClassification.from_pretrained(
                try_name, num_labels=num_labels, cache_dir=cache_dir,
                local_files_only=bool(model_local_dir and os.path.isdir(model_local_dir))
            ).to(device)
        except Exception as e:
            print("[WARN] Base model online load failed, use local_files_only=True:", e)
            base = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels, cache_dir=cache_dir, local_files_only=True
            ).to(device)
        model = PeftModel.from_pretrained(base, ckpt_dir).to(device)
        return model

    if is_full:
        # Full-FT: 直接从 checkpoint 目录加载完整权重
        model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir).to(device)
        return model

    raise RuntimeError("未知的 checkpoint 类型：既不是 LoRA（缺少 adapter_config.json），也不是 Full（缺少 config.json/model.safetensors）。")

tok = load_tokenizer(MODEL, MODEL_LOCAL_DIR, CKPT, CACHE_DIR)
model = load_model(MODEL, CKPT, 2, MODEL_LOCAL_DIR, CACHE_DIR, CKPT_IS_LORA, CKPT_IS_FULL)
model.eval()

# ---------- 6) 加载 IMDb 测试集（镜像 + 离线兜底；含原文 text） ----------
def load_imdb_test(dataset_local_dir=None, cache_dir=None):
    ds = None
    if dataset_local_dir and os.path.isdir(dataset_local_dir):
        try:
            ds = load_dataset(dataset_local_dir, keep_in_memory=False)
        except Exception as e:
            print("[WARN] Local dataset dir failed, fallback to hub name:", e)
    if ds is None:
        try:
            ds = load_dataset("imdb", cache_dir=cache_dir)
        except Exception as e:
            print("[WARN] IMDb online load failed, retry local_files_only=True:", e)
            ds = load_dataset("imdb", cache_dir=cache_dir, local_files_only=True)
    return ds["test"]

test = load_imdb_test(DATASET_LOCAL_DIR, CACHE_DIR)
raw_texts = test["text"]
labels = np.array(test["label"])

if N is not None:
    raw_texts = raw_texts[:N]
    labels = labels[:N]
    print(f"[INFO] Only exporting first N={N} examples.")

# ---------- 7) 推理并保存 CSV ----------
def softmax_np(logits):
    x = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

def batch(it, n=32):
    for i in range(0, len(it), n):
        yield it[i:i+n]

# 简易进度条
def pbar(total):
    count = 0
    last_print = -1
    while True:
        inc = yield
        count += inc
        pct = int(count * 100 / total)
        if pct != last_print:
            last_print = pct
            print(f"\r[EXPORT] {count}/{total} ({pct}%)", end="", flush=True)

probs_all, preds_all = [], []
prog = pbar(len(raw_texts)); next(prog)
with torch.no_grad():
    for chunk in batch(raw_texts, BATCH):
        enc = tok(chunk, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items() if k in ["input_ids","attention_mask"]}
        with torch.autocast(device_type=device.type if device.type != "cpu" else "cpu", enabled=(device.type=="cuda")):
            out = model(**enc)
            logits = out.logits.detach().float().cpu().numpy()
        probs = softmax_np(logits)
        preds = probs.argmax(axis=1)
        probs_all.append(probs)
        preds_all.append(preds)
        prog.send(len(chunk))
print()  # newline

probs = np.vstack(probs_all)
preds = np.concatenate(preds_all)

os.makedirs(OUT_DIR, exist_ok=True)
csv_path = os.path.join(OUT_DIR, "eval_examples.csv")
with open(csv_path, "w", encoding="utf-8") as f:
    f.write("idx,label,pred,prob_neg,prob_pos,correct,text\n")
    for i, (y, yhat, p) in enumerate(zip(labels, preds, probs)):
        text = raw_texts[i].replace('"', '""')
        f.write(f'{i},{y},{yhat},{p[0]:.6f},{p[1]:.6f},{int(y==yhat)},"{text}"\n')

print("[OK] Saved:", csv_path)
