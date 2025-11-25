# src/db/mongo_client.py
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime
from bson import ObjectId

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "audit_ai")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

def insert_document(record: dict):
    record['uploaded_at'] = datetime.utcnow()
    res = db.documents.insert_one(record)
    return db.documents.find_one({"_id": res.inserted_id})

def insert_labeled_document(record: dict):
    record['generated_at'] = datetime.utcnow()
    res = db.labeled_documents.insert_one(record)
    return db.labeled_documents.find_one({"_id": res.inserted_id})

def insert_log(agent: str, request_id: str, input_payload: dict, output_payload: dict, confidence: float = None):
    log = {
        "agent": agent,
        "request_id": request_id,
        "input": input_payload,
        "output": output_payload,
        "confidence": confidence,
        "timestamp": datetime.utcnow()
    }
    db.agents_logs.insert_one(log)

def insert_qapairs(qapairs: list, source_doc_id=None):
    docs = []
    for qa in qapairs:
        qa_doc = {
            "question": qa["question"],
            "answer": qa["answer"],
            "source_doc_id": source_doc_id,
            "created_at": datetime.utcnow()
        }
        docs.append(qa_doc)
    if docs:
        db.qapairs.insert_many(docs)
    return len(docs)

def insert_finetune_record(record: dict):
    record['created_at'] = datetime.utcnow()
    res = db.fine_tunes.insert_one(record)
    return res.inserted_id

def find_document(doc_id):
    return db.documents.find_one({"_id": ObjectId(doc_id)})
