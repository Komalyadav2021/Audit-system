# src/utils/file_watcher.py
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from utils.logger import logger
from fine_tune.qagen import generate_qas_from_labeled
from fine_tune.lora_trainer import fine_tune_lora
import os
import json

class LabeledDataHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()

    def on_created(self, event):
        path = event.src_path
        if path.endswith(".csv") and "datasets/labeled_data" in path:
            logger.info(f"New labeled csv found: {path}")
            # Load CSV quickly, create qas, and (optionally) trigger fine-tune
            import pandas as pd
            df = pd.read_csv(path)
            labeled = df.to_dict(orient="records")
            qas = generate_qas_from_labeled(labeled, source_doc_id=None, n_questions=20)
            # optional: call fine_tune_lora(qas)
            # save qas to file for manual inspect
            out_json = path + ".qapairs.json"
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(qas, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved QAs to {out_json}")

def start_watcher(path="datasets/labeled_data"):
    event_handler = LabeledDataHandler()
    observer = Observer()
    observer.schedule(event_handler, path=path, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
