import json
import os

# Build a stable path relative to this file
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets", "finetune")
DATASET_PATH = os.path.join(DATASET_DIR, "qa_dataset.jsonl")

def save_qa_pairs(pairs):
    os.makedirs(DATASET_DIR, exist_ok=True)
    with open(DATASET_PATH, "a", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return DATASET_PATH
