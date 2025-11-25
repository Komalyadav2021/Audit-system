# src/rag/faiss_indexer.py
import os
import numpy as np
import faiss
import pickle
from .embedding_model import get_embedding
from dotenv import load_dotenv
load_dotenv()
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./src/rag/faiss.index")
META_PATH = FAISS_INDEX_PATH + ".meta.pkl"

class FaissIndexer:
    def __init__(self, dim=384):
        self.dim = dim
        self.index = None
        self.metadata = []
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(META_PATH):
            self.load()
        else:
            self.index = faiss.IndexFlatIP(dim)
            self.metadata = []

    def add(self, texts, metas):
        embs = np.vstack([get_embedding(t) for t in texts])
        # normalize for inner product (cosine similarity)
        faiss.normalize_L2(embs)
        self.index.add(embs)
        self.metadata.extend(metas)
        self.save()

    def save(self):
        faiss.write_index(self.index, FAISS_INDEX_PATH)
        with open(META_PATH, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self):
        self.index = faiss.read_index(FAISS_INDEX_PATH)
        with open(META_PATH, "rb") as f:
            self.metadata = pickle.load(f)

    def search(self, query, k=5):
        q_emb = get_embedding(query).reshape(1, -1)
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, k)
        results = []
        for idx in I[0]:
            if idx < len(self.metadata):
                results.append(self.metadata[idx])
        return results
