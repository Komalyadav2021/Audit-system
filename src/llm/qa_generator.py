import random
from llm.answer_generator import _call_ollama, _pick_model

def generate_qa_from_chunks(chunks, num_pairs=10):
    model = _pick_model()
    qa_pairs = []

    for _ in range(num_pairs):
        prompt = f"""
You are an audit assistant. Create one high-quality Q&A pair using ONLY this data:

{chunks[:3]}

Output format:
Q: <question>
A: <answer>
"""
        result = _call_ollama(prompt, model=model)
        try:
            q, a = result.split("A:", 1)
            q = q.replace("Q:", "").strip()
            a = a.strip()
            qa_pairs.append({"question": q, "answer": a})
        except:
            continue

    return qa_pairs
