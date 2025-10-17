import os
import torch
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
try:
    from peft import TaskType
except Exception:
    # Fallback: older peft may accept string "SEQ_CLS"
    TaskType = None


def _load_base_model(model_name, num_labels, model_local_dir=None):
    cache_dir = os.environ.get("HF_HOME")  # prefer HF_HOME cache for v5 readiness
    if model_local_dir and os.path.isdir(model_local_dir):
        # Fully offline local directory (already contains config.json, pytorch_model.bin, etc.)
        return AutoModelForSequenceClassification.from_pretrained(
            model_local_dir, num_labels=num_labels, cache_dir=cache_dir, local_files_only=True
        )
    try:
        return AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, cache_dir=cache_dir
        )
    except Exception as e:
        print("[WARN] Model online load failed, retry with local_files_only=True:", e)
        return AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, cache_dir=cache_dir, local_files_only=True
        )


def _default_lora_cfg():
    # Safe defaults for DistilBERT / BERT-like models
    base = {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "bias": "none",
        # For PEFT >=0.10 prefer enum; otherwise string is accepted by older versions
        "task_type": TaskType.SEQ_CLS if TaskType is not None else "SEQ_CLS",
        # DistilBERT attention projections
        "target_modules": ["q_lin", "k_lin", "v_lin", "out_lin"],
        # keep classifier in full precision (optional to save)
        "modules_to_save": ["classifier"],
    }
    return base


def build_model(model_name, num_labels, method, lora_cfg=None, model_local_dir=None):
    base = _load_base_model(model_name, num_labels, model_local_dir=model_local_dir)
    method = (method or "lora").lower()
    if method == "full":
        return base
    elif method == "lora":
        if lora_cfg is None:
            lora_cfg = _default_lora_cfg()
        else:
            # ensure required keys exist
            if "task_type" not in lora_cfg:
                lora_cfg["task_type"] = TaskType.SEQ_CLS if TaskType is not None else "SEQ_CLS"
            if "target_modules" not in lora_cfg:
                lora_cfg["target_modules"] = ["q_lin", "k_lin", "v_lin", "out_lin"]
        peft_model = get_peft_model(base, LoraConfig(**lora_cfg))
        return peft_model
    else:
        raise ValueError("method must be 'full' or 'lora'")


def count_params(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total