# src/parsers/bank_statement_parser.py
import os
import re
import pandas as pd
from typing import Dict, Any, List
import pdfplumber
from utils.logger import logger

AMOUNT_RE = re.compile(r'-?\d{1,3}(?:,\d{3})*(?:\.\d+)?')

def parse_bank_statement_file(path: str) -> Dict[str, Any]:
    ext = os.path.splitext(path)[1].lower()
    rows = []
    text = ""
    metadata = {"filetype": ext}
    if ext == ".csv":
        df = pd.read_csv(path)
        rows = df.to_dict(orient="records")
        text = df.astype(str).to_string()
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(path)
        rows = df.to_dict(orient="records")
        text = df.astype(str).to_string()
    elif ext == ".pdf":
        try:
            with pdfplumber.open(path) as pdf:
                for p in pdf.pages:
                    # try table extraction
                    try:
                        table = p.extract_table()
                        if table and len(table) > 1:
                            header = table[0]
                            for r in table[1:]:
                                record = {header[i] if i < len(header) else f"c{i}": (r[i] if i < len(r) else None) for i in range(len(r))}
                                rows.append(record)
                        else:
                            text += p.extract_text() or ""
                    except Exception:
                        text += p.extract_text() or ""
        except Exception as e:
            logger.exception("pdf parsing error")
            text = ""
    else:
        # fallback: treat file as text
        with open(path, "r", errors="ignore") as f:
            text = f.read()
        for line in text.splitlines():
            if line.strip():
                rows.append({"line": line.strip()})
    # basic chunking of text for RAG (split into 500-char chunks)
    chunks = []
    for i in range(0, len(text), 500):
        chunks.append({"text": text[i:i+500], "start": i, "end": i+500})
    # try rule-based labeling to structure rows (best-effort)
    labeled_rows = rule_based_labeling(rows)
    return {"text": text, "rows": labeled_rows, "chunks": chunks, "metadata": metadata}

def rule_based_labeling(rows: List[dict]) -> List[dict]:
    labeled = []
    for idx, r in enumerate(rows):
        # If the row is dict with many columns, combine into a single line
        if isinstance(r, dict):
            line_text = " ".join(str(v) for v in r.values() if v is not None)
        else:
            line_text = str(r)
        date = extract_date(line_text)
        amounts = AMOUNT_RE.findall(line_text)
        debit = credit = None
        # heuristic: if negative sign or 'dr' present treat as debit
        if amounts:
            # take last two numbers, or last one
            if len(amounts) >= 2:
                # decide which is debit/credit by keywords
                if "dr" in line_text.lower() or "debit" in line_text.lower() or "-" in amounts[-1]:
                    debit = parse_amount(amounts[-1])
                else:
                    # fallback: assume last is balance, previous is amount
                    credit = parse_amount(amounts[-1])
            else:
                val = parse_amount(amounts[-1])
                if "withdraw" in line_text.lower() or "debit" in line_text.lower() or "-" in line_text:
                    debit = val
                else:
                    credit = val
        labeled.append({
            "line_id": idx,
            "raw": line_text,
            "date": date,
            "description": line_text,
            "debit": debit,
            "credit": credit,
            "balance": None,
            "category": None
        })
    return labeled

def extract_date(s: str):
    # mm/dd/yyyy, dd/mm/yyyy, yyyy-mm-dd common forms
    patterns = [
        r'\b\d{4}-\d{1,2}-\d{1,2}\b',
        r'\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b'
    ]
    for p in patterns:
        m = re.search(p, s)
        if m:
            return m.group(0)
    return None

def parse_amount(s: str):
    if s is None: return None
    return float(s.replace(',', '').replace('+', '').replace('(', '-').replace(')', ''))
