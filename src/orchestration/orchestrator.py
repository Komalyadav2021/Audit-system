import os
import pandas as pd
import re
from agents.planner import run_planner
from agents.executor import run_executor
from agents.reviewer import run_reviewer
from agents.labeler import run_labeler
from db.mongo_client import insert_log
from utils.logger import logger

FAST_KEYWORDS = ["email", "mail", "emails", "mobile", "phone", "total", "sum", "count"]

TASK_MAP = {
    "parse": run_executor,
    "parse_if_needed": run_executor,
    "ensure_indexed": run_executor,
    "label": run_labeler,
    "analysis": run_executor,
    "retrieve": run_executor,
    "generate": run_executor,
    "answer": run_executor,
}

# ---------------------------------------------
# FAST MODE FOR CSV / XLSX QUERIES
# ---------------------------------------------
def fast_extract_csv(file_path, query):
    try:
        df = pd.read_csv(file_path)
    except:
        df = pd.read_excel(file_path)

    text = "\n".join(df.astype(str).apply(lambda x: " ".join(x), axis=1))

    # Extract emails
    if "email" in query.lower():
        emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
        return list(sorted(set(emails)))

    # Extract mobile numbers
    if "mobile" in query.lower() or "phone" in query.lower():
        mobiles = re.findall(r"\b[6-9]\d{9}\b", text)
        return list(sorted(set(mobiles)))

    # Compute total amount
    if "total" in query.lower() or "sum" in query.lower():
        for col in df.columns:
            try:
                numeric = pd.to_numeric(df[col], errors="ignore")
                if numeric.dtype in ["float64", "int64"]:
                    return float(numeric.sum())
            except:
                continue
        return "No numeric column found."

    # Generic fallback
    return df.head(20).to_dict()

# ---------------------------------------------
# ORCHESTRATOR
# ---------------------------------------------
def orchestrate(input_json: dict) -> dict:
    query = input_json.get("user_query", "").lower()
    doc = input_json.get("doc_id", "")

    # ---------------------------------------------
    # 🔥 FAST MODE: CSV / XLSX → instant answer
    # ---------------------------------------------
    if doc and any(doc.endswith(x) for x in [".csv", ".xlsx"]):
        if any(k in query for k in FAST_KEYWORDS):
            fast_answer = fast_extract_csv(doc, query)
            return {
                "request_id": "FAST-MODE",
                "final_answer": fast_answer,
                "chunks_used": [],
                "results": []
            }

    # --------------------------------------------------
    # OTHERWISE: normal multi-agent pipeline (RAG + LLM)
    # --------------------------------------------------
    planner_out = run_planner(input_json)
    request_id = planner_out["request_id"]

    shared_context = {
        "request_id": request_id,
        "input": input_json,
        "memory": {}
    }

    results = []

    for t in planner_out["tasks"]:
        func = TASK_MAP.get(t["type"])
        if not func:
            continue

        try:
            out = func({"task": t, "context": shared_context})

            # store outputs in memory
            if isinstance(out, dict):
                for k in (
                    "parsed_rows",
                    "labels",
                    "analysis",
                    "retrieved",
                    "retrieved_chunks",
                    "summary",
                    "answer"
                ):
                    if k in out:
                        shared_context["memory"][k] = out[k]

            # reviewer
            review = run_reviewer({"result": out, "context": shared_context})
            insert_log(func.__name__, request_id, t, {"result": out, "review": review})

            results.append({"task": t, "result": out, "review": review})

        except Exception as e:
            results.append({"task": t, "error": str(e)})

    mem = shared_context["memory"]

    final_answer = (
        mem.get("answer") or
        mem.get("summary") or
        mem.get("retrieved") or
        results
    )

    return {
        "request_id": request_id,
        "final_answer": final_answer,
        "chunks_used": mem.get("retrieved_chunks", []),
        "results": results
    }
