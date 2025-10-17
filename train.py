import os
import time
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro"),
    }


def _get(cfg: dict, key: str, default):
    return cfg[key] if key in cfg else default


def make_trainer(cfg, model, tokenizer, train_set, eval_set):
    """Create HF Trainer + dataloaders based on a simple cfg dict."""
    out_dir = _get(cfg, "out_dir", "runs/imdb")
    per_device_train_batch_size = _get(cfg, "train_bs", 16)
    per_device_eval_batch_size = _get(cfg, "eval_bs", 32)
    num_train_epochs = float(_get(cfg, "epochs", 2))
    learning_rate = float(_get(cfg, "lr", 5e-5))
    weight_decay = float(_get(cfg, "weight_decay", 0.0))
    warmup_ratio = float(_get(cfg, "warmup_ratio", 0.0))
    gradient_accumulation_steps = int(_get(cfg, "grad_accum", 1))
    logging_steps = int(_get(cfg, "logging_steps", 50))
    eval_strategy = _get(cfg, "eval_strategy", "epoch")  # "steps" or "epoch"
    save_strategy = _get(cfg, "save_strategy", "epoch")
    save_total_limit = int(_get(cfg, "save_total_limit", 1))
    fp16 = bool(_get(cfg, "fp16", torch.cuda.is_available()))
    bf16 = bool(_get(cfg, "bf16", False))

    args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=logging_steps,
        evaluation_strategy=eval_strategy,
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        fp16=fp16,
        bf16=bf16,
        report_to=[],  # disable wandb by default
        load_best_model_at_end=True if eval_strategy != "no" else False,
        metric_for_best_model="accuracy",
        greater_is_better=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # expose dataloaders for convenience
    train_loader = trainer.get_train_dataloader()
    eval_loader = trainer.get_eval_dataloader()

    return trainer, train_loader, eval_loader


def record_resource_stats(metrics: dict, cfg: dict, trainable_params: int, total_params: int):
    out_dir = _get(cfg, "out_dir", "runs/imdb")
    os.makedirs(out_dir, exist_ok=True)
    info = {
        "metrics": metrics,
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "gpu_mem_mb": int(torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else None,
    }
    with open(os.path.join(out_dir, "resource.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(info, f, indent=2, ensure_ascii=False)
    print("[INFO] Saved resource stats to", os.path.join(out_dir, "resource.json"))
    return info