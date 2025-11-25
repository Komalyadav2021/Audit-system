# src/agents/planner.py
import uuid
from db.mongo_client import find_document

def run_planner(input_json: dict) -> dict:
    request_id = input_json.get("request_id", str(uuid.uuid4()))
    q = input_json.get("user_query", "").lower()
    doc_id = input_json.get("doc_id")

    tasks = []

    # If user asks a general question → no RAG, no analysis, just answer
    GENERAL_KEYWORDS = ["what is", "define", "explain", "meaning of"]
    if any(q.startswith(k) for k in GENERAL_KEYWORDS):
        tasks.append({
            "task_id": "direct_answer",
            "type": "answer",
            "args": {"query": input_json.get("user_query")}
        })
        return {
            "request_id": request_id,
            "tasks": tasks,
            "status": "ok",
            "confidence": 0.9
        }

    # Summary flow
    if "summary" in q or "summarize" in q:
        tasks += [
            {"task_id": "parse", "type": "parse", "args": {"doc_id": doc_id}},
            {"task_id": "analysis", "type": "analysis", "args": {"doc_id": doc_id}},
            {"task_id": "generate_summary", "type": "generate", "args": {"doc_id": doc_id}}
        ]
    
    # Analysis flow
    # elif ("debit" in q or "credit" in q) and "balance sheet" not in q:
    elif any(word in q for word in ["debit", "credit"]) and not ("balance sheet" in q):
        doc = find_document(doc_id)
        tasks.append({
            "task_id": "analysis",
            "type": "analysis",
            "args": {"rows": doc.get("rows", []) if doc else []}
        })

    # RAG retrieval for document-specific questions
    else:
        tasks.append({
            "task_id": "retrieve",
            "type": "retrieve",
            "args": {"query": input_json.get("user_query")}
        })
        tasks.append({
            "task_id": "answer_from_chunks",
            "type": "answer",
            "args": {"query": input_json.get("user_query")}
        })

    return {"request_id": request_id, "tasks": tasks, "status": "ok", "confidence": 0.9}
