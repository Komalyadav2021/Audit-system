# src/llm/answer_generator.py
"""
Robust local LLM caller for Ollama (with prompt-size protection and model fallback).

Behavior (hybrid):
 - If chunks provided -> try to answer using ONLY the chunks (RAG).
 - If no chunks -> fallback to general LLM response.
 - Protects against huge prompts by truncating and limiting chunks.
 - Prefers a fast CPU-friendly model (phi3) if available; falls back to OLLAMA_MODEL.
"""

import os
import requests
from typing import List, Dict
from utils.logger import logger

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3")   # default from .env
FALLBACK_MODEL = os.getenv("OLLAMA_FALLBACK_MODEL", "phi3")  # small / fast model

# timeout in seconds (increase if your machine is very slow)
REQUEST_TIMEOUT = int(os.getenv("OLLAMA_REQUEST_TIMEOUT", "600"))  # 10 minutes

# safe limits
MAX_CHUNKS = int(os.getenv("RAG_MAX_CHUNKS", "3"))        # max number of chunks to send
MAX_CHARS_PER_CHUNK = int(os.getenv("RAG_MAX_CHARS", "800"))  # truncate each chunk
MAX_PROMPT_CHARS = int(os.getenv("RAG_MAX_PROMPT_CHARS", "3000"))  # total allowed context chars

def _call_ollama(prompt: str, model: str, max_tokens: int = 512) -> str:
    """
    Calls Ollama /api/generate with stream=False (most robust).
    Returns generated string or raises Exception.
    """
    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "maxTokens": max_tokens,
        "stream": False
    }
    try:
        resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        # Try common shapes
        if isinstance(data, dict):
            if "response" in data:
                return data["response"]
            if "output" in data:
                return data["output"]
            if "result" in data:
                return data["result"]
            if "choices" in data and len(data["choices"]) > 0:
                ch0 = data["choices"][0]
                return ch0.get("text") or ch0.get("message") or str(ch0)
        return str(data)
    except Exception as e:
        logger.exception("Ollama request failed")
        raise

def _pick_model() -> str:
    """
    Return the model name to use:
      - prefer FALLBACK_MODEL (phi3) if present in server,
      - otherwise use OLLAMA_MODEL from env.
    """
    try:
        # query server for available models
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        r.raise_for_status()
        info = r.json()
        models = [m.get("name") for m in info.get("models", []) if isinstance(m, dict)]
        # prefer fallbacks for speed
        if FALLBACK_MODEL in models:
            return FALLBACK_MODEL
        if OLLAMA_MODEL in models:
            return OLLAMA_MODEL
        # choose first available model if neither present
        if models:
            return models[0]
    except Exception:
        # if anything fails, fallback to environment model
        return OLLAMA_MODEL
    return OLLAMA_MODEL

def _prepare_context_from_chunks(chunks: List[Dict], max_chunks=MAX_CHUNKS,
                                 max_chars_per_chunk=MAX_CHARS_PER_CHUNK,
                                 max_total_chars=MAX_PROMPT_CHARS) -> str:
    """
    Create a short, safe context string from a list of chunks.
    Each chunk is truncated; number of chunks limited.
    """
    selected = chunks[:max_chunks]
    parts = []
    total = 0
    for i, c in enumerate(selected):
        txt = (c.get("text") or c.get("chunk_text") or "")[:max_chars_per_chunk].strip()
        header = f"[Chunk {i+1} | doc={c.get('document_id','')[:8]} | chunk={c.get('chunk_id','')}]"
        block = header + "\n" + txt
        if total + len(block) > max_total_chars:
            # if adding this block would exceed total, cut the block shorter
            remaining = max_total_chars - total - 10
            if remaining > 0:
                parts.append(block[:remaining] + "...")
            break
        parts.append(block)
        total += len(block)
    return "\n\n---\n\n".join(parts)

def generate_final_answer(query: str, chunks: List[Dict] = None, top_k: int = MAX_CHUNKS) -> str:
    """
    Generate human-readable answer. Protects prompt size and chooses a fast model when possible.
    """
    chunks = chunks or []
    model_to_use = _pick_model()

    # If there is retrieved context - strict RAG
    if chunks:
        context = _prepare_context_from_chunks(chunks, max_chunks=top_k)
        prompt = f"""
You are an Audit Intelligence Assistant. Use ONLY the context below to answer the user's question.
Do NOT hallucinate or add facts not present in the context. If the answer is not present, reply: "Information not found in the provided documents."

Context:
{context}

Question:
{query}

Answer concisely in a natural, human-readable way.
Do NOT mention chunk numbers, chunk IDs, or document IDs in your answer.
Only use the content to answer the question.

"""
        try:
            return _call_ollama(prompt, model=model_to_use).strip()
        except Exception as e:
            # try fallback to a different, smaller model if possible
            try:
                fallback = FALLBACK_MODEL if model_to_use != FALLBACK_MODEL else OLLAMA_MODEL
                return _call_ollama(prompt, model=fallback).strip()
            except Exception as e2:
                logger.exception("Both primary and fallback LLM calls failed")
                return f"[LLM error] Could not generate answer locally: {e2}"

    # No context: hybrid fallback to general LLM
    else:
        prompt = f"""
You are an Audit Intelligence Assistant.

Question:
{query}

Provide a concise, helpful answer. If you are unsure, say you are unsure and advise the user to upload the relevant documents for a precise answer.
"""
        try:
            return _call_ollama(prompt, model=model_to_use).strip()
        except Exception as e:
            logger.exception("Fallback LLM call failed")
            return f"[LLM error] Could not generate answer locally: {e}"
