import os
import sys
import json
import time
import numpy as np
import torch
import importlib.util as _importlib_util

# ---------- Hugging Face mirror & cache (set BEFORE importing transformers) ----------
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("HF_HOME", hf_home)
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_home, "transformers"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(hf_home, "datasets"))
os.environ.setdefault("HF_MODULES_CACHE", os.path.join(hf_home, "modules"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# If hf_transfer isn't installed, disable fast transfer to avoid hard failure
if not _importlib_util.find_spec("hf_transfer"):
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
else:
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from transformers import AutoTokenizer

from model_utils import build_model, count_params
from data_utils import load_text_classification
from train_utils import make_trainer, record_resource_stats


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main(cfg_path):
    cfg = load_config(cfg_path)

    # Allow overriding by environment variables for quick testing
    model_name = os.environ.get("MODEL_NAME", cfg.get("model_name", "distilbert-base-uncased"))
    dataset = os.environ.get("DATASET_NAME", cfg.get("dataset", "imdb"))
    max_length = int(os.environ.get("MAX_LENGTH", cfg.get("max_length", 256)))
    seed = int(cfg.get("seed", 42))
    method = cfg.get("method", "lora")  # "full" or "lora"
    lora_cfg = cfg.get("lora_cfg", None)

    # Optional fully-offline local dirs (no internet required)
    model_local_dir = os.environ.get("MODEL_LOCAL_DIR", cfg.get("model_local_dir", "")) or None
    dataset_local_dir = os.environ.get("DATASET_LOCAL_DIR", cfg.get("dataset_local_dir", "")) or None

    # Reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cache_dir = os.environ.get("TRANSFORMERS_CACHE")

    # ---------- Tokenizer with mirror & offline fallback ----------
    try:
        preload_name = model_local_dir if (model_local_dir and os.path.isdir(model_local_dir)) else model_name
        tokenizer = AutoTokenizer.from_pretrained(
            preload_name, use_fast=True, cache_dir=cache_dir,
            local_files_only=bool(model_local_dir and os.path.isdir(model_local_dir))
        )
    except Exception as e:
        print("[WARN] Tokenizer load failed, retry with local_files_only=True:", e)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, cache_dir=cache_dir, local_files_only=True
        )

    # ---------- Dataset with mirror & offline fallback ----------
    train_set, eval_set = load_text_classification(
        dataset, max_length, tokenizer, seed=seed, dataset_local_dir=dataset_local_dir
    )

    # ---------- Model (supports full / LoRA) ----------
    model = build_model(model_name, num_labels=2, method=method, lora_cfg=lora_cfg, model_local_dir=model_local_dir)
    n_trainable, n_total = count_params(model)
    print(f"[INFO] Params: trainable={n_trainable:,} / total={n_total:,}")

    # ---------- Trainer / Training ----------
    trainer, train_dataloader, eval_dataloader = make_trainer(cfg, model, tokenizer, train_set, eval_set)

    t0 = time.time()
    trainer.train()
    t1 = time.time()

    print(f"[INFO] Training finished in {t1 - t0:.2f}s")

    # ---------- Eval ----------
    metrics = trainer.evaluate()
    print("[INFO] Eval:", metrics)

    # ---------- Resource stats ----------
    record_resource_stats(metrics, cfg, n_trainable, n_total)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <config.json>\n"
              "Example: python main.py configs/lora.imdb.json")
        sys.exit(1)
    main(sys.argv[1])