import os
from datasets import load_dataset


def load_text_classification(dataset_name, max_length, tokenizer, seed=42, train_subset=None, dataset_local_dir=None):
    name = dataset_name.lower()
    cache_dir = os.environ.get("HF_DATASETS_CACHE")

    # If user provided a local dataset dir (Arrow cache or HF dataset folder), load from it
    if dataset_local_dir and os.path.isdir(dataset_local_dir):
        # Expect the directory to be a dataset script or cached Arrow dataset
        try:
            ds = load_dataset(dataset_local_dir, cache_dir=cache_dir, keep_in_memory=False)
        except Exception as e:
            print("[WARN] Local dataset dir load failed, falling back to name:", e)
            # continue to named dataset path

    if name == "imdb":
        try:
            ds = load_dataset("imdb", cache_dir=cache_dir)
        except Exception as e:
            print("[WARN] IMDb online load failed, retry with local_files_only=True:", e)
            ds = load_dataset("imdb", cache_dir=cache_dir, local_files_only=True)
        text_key, label_key = "text", "label"

    elif name in ["sst2", "glue/sst2", "glue"]:
        try:
            ds = load_dataset("glue", "sst2", cache_dir=cache_dir)
        except Exception as e:
            print("[WARN] GLUE/SST-2 online load failed, retry with local_files_only=True:", e)
            ds = load_dataset("glue", "sst2", cache_dir=cache_dir, local_files_only=True)
        ds = ds.rename_column("sentence", "text")
        text_key, label_key = "text", "label"

    else:
        raise ValueError("Unsupported dataset. Use 'imdb' or 'glue/sst2'.")

    def tok(batch):
        return tokenizer(
            batch[text_key],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

    ds = ds.map(tok, batched=True, remove_columns=[text_key])
    ds = ds.rename_column("label", "labels")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_set = ds["train"]
    eval_split = "test" if "test" in ds else "validation"
    eval_set = ds[eval_split]

    if train_subset is not None and train_subset > 0:
        train_set = train_set.select(range(min(train_subset, len(train_set))))

    return train_set, eval_set