# chunk-based Q/A generator — robust, deterministic prompts
from llm.answer_generator import _call_ollama, _pick_model
import re

def generate_qa_from_chunks(chunks, num_pairs=20):
    """
    chunks: list of {'text': '...', 'document_id': '...', 'chunk_id': n}
    Returns list of {"question":..., "answer": ...}
    """
    if not chunks:
        return []

    model = _pick_model()
    # prepare a single text sample by joining top chunks
    texts = [c.get("text","") for c in chunks[:5]]
    context = "\n\n---\n\n".join(texts)

    prompt = f"""
You are a helpful audit assistant. Using ONLY the context below, create {num_pairs} concise, high-quality Q&A pairs (question and answer).
- Questions should be factual, cover totals, specific transactions, dates, categories, and one general summary question.
- Output format (one pair per line in JSON): 
{{"question":"...","answer":"..."}}

Context:
{context}
"""
    raw = _call_ollama(prompt, model=model, max_tokens=1024)
    lines = raw.splitlines()
    qas = []
    import json
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # try parse JSON on the line
        try:
            obj = json.loads(line)
            if "question" in obj and "answer" in obj:
                qas.append({"question": obj["question"], "answer": obj["answer"]})
        except Exception:
            # fallback simple parse: split by "?" or "Q:" patterns
            if "Q:" in line and "A:" in line:
                try:
                    q,a = line.split("A:",1)
                    q = q.replace("Q:","").strip()
                    a = a.strip()
                    qas.append({"question": q, "answer": a})
                except:
                    continue
        if len(qas) >= num_pairs:
            break
    # last resort: heuristic small QAs if generation fails
    if not qas:
        qas = [{"question": "Provide a brief summary of the context", "answer": "Summary not available."}]
    return qas
