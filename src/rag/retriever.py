# src/rag/retriever.py
from .faiss_indexer import FaissIndexer
from db.mongo_client import db
fi = FaissIndexer()

def retrieve(query: str, k=5):
    hits = fi.search(query, k=k)
    # return text + doc meta
    enriched = []
    for h in hits:
        doc = db.documents.find_one({"_id": h.get("document_id")}) if h.get("document_id") else None
        enriched.append({"meta": h, "document": doc})
    return enriched
