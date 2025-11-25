# src/fine_tune/qagen.py
import pandas as pd
from db.mongo_client import insert_qapairs
import random

def generate_qas_from_labeled(labeled_rows: list, source_doc_id=None, n_questions=20):
    df = pd.DataFrame(labeled_rows)
    qas = []
    # simple Q1: totals
    total_debit = df['debit'].dropna().astype(float).sum() if 'debit' in df.columns else 0.0
    total_credit = df['credit'].dropna().astype(float).sum() if 'credit' in df.columns else 0.0
    qas.append({"question": "What is the total debit in this statement?", "answer": str(total_debit)})
    qas.append({"question": "What is the total credit in this statement?", "answer": str(total_credit)})
    # top categories
    if 'category' in df.columns and not df['category'].isnull().all():
        cat_sum = df.groupby('category')['debit'].sum().sort_values(ascending=False)
        top_cat = cat_sum.index[0] if len(cat_sum) else "Unknown"
        qas.append({"question": "Which category has the highest debit?", "answer": str(top_cat)})
    # random line questions
    sample = df.sample(n=min(10, len(df))) if len(df) else []
    for _, row in sample.iterrows():
        amt = row['debit'] if pd.notna(row.get('debit')) else row.get('credit')
        q = {"question": f"What is the transaction amount on {row.get('date')} for '{row.get('description')}'?", "answer": str(amt)}
        qas.append(q)
    # pad until n_questions (rephrase or duplicate with slight change)
    while len(qas) < n_questions:
        qas.append(random.choice(qas))
    # store in DB
    insert_qapairs(qas, source_doc_id=source_doc_id)
    return qas
