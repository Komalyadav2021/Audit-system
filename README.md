# Audit Intelligence System

## Overview
Local, modular Audit Assistant: parse bank statements, auto-label transactions (hybrid rules+LLM), RAG retrieval (FAISS), and local fine-tuning (LoRA/PEFT). Runs fully offline with MongoDB.

## Quickstart
1. Start MongoDB: `docker-compose up -d`
2. Create and activate venv:
3. Start Streamlit UI:

## Folder layout
Explain the structure...

## Models
Download embedding model and base LLM before offline usage: `all-MiniLM-L6-v2`, etc.

## License
MIT
# Runbook

## Start services
1. Start MongoDB: `docker-compose up -d`
2. Start file watcher (optional): `python -c "from utils.file_watcher import start_watcher; start_watcher()"`

## Upload file
Use Streamlit UI to upload a bank statement. The orchestrator will run parse -> label -> analyze -> generate.

## Fine-tune
Labeled CSVs appear under `datasets/labeled_data/`. Create Q&A pairs from there or let the watcher auto-generate them. Run `python src/fine_tune/lora_trainer.py` (adapt script to call function).

## Troubleshooting
- FAISS index not found: check `FAISS_INDEX_PATH` in .env.
- Model load errors: ensure local model directories exist and `local_files_only=True` is set.
