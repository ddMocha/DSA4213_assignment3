# DSA4213 Assignment 3 — IMDB Sentiment Classification

This is the **third assignment** of the NUS course **DSA4213:  Natural Language Processing for Data Science**.  
The project compares **Full Fine-tuning** and **LoRA (Low-Rank Adaptation)** on the **IMDB movie reviews** dataset using the `DistilBERT` model.

---

## 1. Overview

**Task:** Binary sentiment classification (positive vs. negative reviews).  
**Goal:** Compare performance and efficiency between:
- **Full Fine-tuning:** all model weights are updated.
- **LoRA:** only small low-rank adapter matrices are trained while freezing the base model.

---

## 2. File Structure

```
.
├── main.py                  # main training script
├── export_examples.py       # export a few test predictions
├── configs/
│   ├── full_ft.imdb.json    # configuration for full fine-tuning
│   └── lora.imdb.json       # configuration for LoRA fine-tuning
├── runs/                    # training outputs (checkpoints, logs)
│   ├── full_ft_imdb/
│   └── lora_imdb/
├── figs/                    # optional plots and result figures
└── README.md                # this file
```

---

## 3. Environment Configuration

All experiments were run on **AutoDL Cloud Platform** with the following setup:

| Component | Specification |
|------------|----------------|
| **CUDA** | 12.4 |
| **Python** | 3.12 (Ubuntu 22.04) |
| **PyTorch** | 2.5.1 |

A ready-to-use environment can be created directly from the provided requirements file:

```bash
# create and activate environment
conda create -n dsahw python=3.12 -y
conda activate dsahw

# install all dependencies
pip install -r requirements.txt
```


---

## 4. How to Run

### (1) Full Fine-tuning
```bash
python main.py configs/full_ft.imdb.json
```

### (2) LoRA Fine-tuning
```bash
python main.py configs/lora.imdb.json
```

Logs and checkpoints will be automatically saved under `runs/`.

---

## 5. Generate Example Predictions

After training, export sample outputs for the report:

```bash
python export_examples.py
```

Example predictions will be written to `runs/lora_imdb/examples.csv` (or similar path).

---

## 6. Expected Results

| Method | Accuracy | GPU (MB) | Trainable Params |
|--------|-----------|----------|------------------|
| Full Fine-tuning | ~0.91 | 2200 | 66.96M |
| LoRA (r=8) | ~0.90 | 1600 | 0.887M |

LoRA achieves **~75× fewer parameters** with comparable accuracy.

---

**Author:** Yao Kaiwen  
**Course:** NUS DSA4213 
**Date:** October 2025

