# src/rag/embedding_model.py
from sentence_transformers import SentenceTransformer
import os

MODEL_ID = os.getenv("EMBED_MODEL", "sentence-transformers_all-MiniLM-L6-v2")

# If you downloaded model archive into /models/..., point MODEL_ID to that path.
MODEL = SentenceTransformer("all-MiniLM-L6-v2")  # ensure model cached locally for offline

def get_embedding(text: str):
    emb = MODEL.encode(text, convert_to_numpy=True)
    return emb
