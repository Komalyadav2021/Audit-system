# src/agents/executor.py
import os
from db.mongo_client import insert_document, find_document
from parsers.bank_statement_parser import parse_bank_statement_file
from rag.faiss_indexer import FaissIndexer
from utils.logger import logger
from llm.answer_generator import generate_final_answer
import pandas as pd

faiss_index = FaissIndexer()

def run_executor(payload: dict) -> dict:
    task = payload.get("task", {})
    ttype = task.get("type")
    args = task.get("args", {})
    request_id = payload.get("context", {}).get("request_id")

    # ---------------------------------------
    # 1. PARSE DOCUMENT
    # ---------------------------------------
    if ttype == "parse":
        doc_id = args.get("doc_id")
        parsed = parse_bank_statement_file(doc_id)

        # Save parsed doc to MongoDB
        doc_record = {
            "filename": os.path.basename(doc_id),
            "text": parsed.get("text", ""),
            "chunks": parsed.get("chunks", []),
            "metadata": parsed.get("metadata", {}),
            "source": "upload",
            "version": 1
        }
        saved = insert_document(doc_record)

        # Save chunks to FAISS
        if parsed.get("chunks"):
            texts = [c["text"] for c in parsed["chunks"]]
            metas = [
                {"document_id": str(saved["_id"]), "chunk_id": i, "text": texts[i]}
                for i in range(len(texts))
            ]
            faiss_index.add(texts, metas)

        return {
            "status": "ok",
            "document_id": str(saved["_id"]),
            "parsed_rows": parsed.get("rows", [])
        }

    # ---------------------------------------
    # 2. ANALYSIS (TOTAL DEBIT / TOTAL CREDIT)
    # ---------------------------------------
    if ttype == "analysis":
        rows = args.get("rows", [])
        df = pd.DataFrame(rows)

        total_debit = df.get("debit", pd.Series(dtype=float)).fillna(0).astype(float).sum()
        total_credit = df.get("credit", pd.Series(dtype=float)).fillna(0).astype(float).sum()

        return {
            "analysis": {
                "total_debit": float(total_debit),
                "total_credit": float(total_credit)
            }
        }

    # ---------------------------------------
    # 3. GENERATE SUMMARY (PLACEHOLDER)
    # ---------------------------------------
    if ttype == "generate":
        return {"summary": "This is a placeholder summary. Add LLM later."}

    # ---------------------------------------
    # 4. RETRIEVAL FROM FAISS
    # ---------------------------------------
    if ttype == "retrieve":
        query = args.get("query", "")

        if faiss_index.index is None or faiss_index.index.ntotal == 0:
            return {"error": "No indexed documents found. Upload a statement first."}

        hits = faiss_index.search(query, k=5)
        if not hits:
            return {
                "results": [],
                "message": "No relevant chunks found."
            }

        out = []
        for hit in hits:
            doc = find_document(hit["document_id"])
            if not doc:
                continue

            chunk_id = hit["chunk_id"]
            chunk_text = doc["chunks"][chunk_id]["text"]

            out.append({
                "document_id": hit["document_id"],
                "chunk_id": chunk_id,
                "text": chunk_text
            })

        return {"retrieved_chunks": out}

        # ---------------------------------------
    # 5. ANSWER USING OLLAMA (LOCAL LLM)
    # ---------------------------------------
    if ttype == "answer":
        query = args.get("query") or ""

        # get memory (shared context)
        memory = payload.get("context", {}).get("memory", {})

        # retrieved chunks may be saved under different keys
        retrieved = (
            memory.get("retrieved") or
            memory.get("retrieved_chunks") or
            []
        ) or []

        # Safety: reduce number of chunks and truncate each chunk's text
        # to avoid huge prompts (we will rely on the llm.answer_generator to further protect)
        safe_chunks = []
        for c in retrieved:
            text = (c.get("text") or "")[:1200]  # keep up to 1200 chars per chunk at this stage
            safe_chunks.append({
                "document_id": c.get("document_id"),
                "chunk_id": c.get("chunk_id"),
                "text": text
            })

        # Optionally prefer fewer chunks for heavy models
        # If env var or memory indicates heavy model, reduce more (handled by answer_generator too)
        # Call the answer generator with limited chunks
        final = generate_final_answer(query, safe_chunks)

        return {"answer": final}


    # ---------------------------------------
    # UNKNOWN TASK
    # ---------------------------------------
    return {"status": "unknown_task"}
