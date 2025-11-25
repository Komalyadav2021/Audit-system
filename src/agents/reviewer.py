# src/agents/reviewer.py
from utils.logger import logger

def run_reviewer(payload: dict) -> dict:
    """
    payload: { result: <agent output>, context: {...} }
    returns: { action: 'accept'|'retry'|'flag', confidence: 0-1, message: str }
    """
    result = payload.get("result", {})
    ctx = payload.get("context", {})
    # If parsed rows exist, check date coverage and numeric consistency
    rows = result.get("parsed_rows") or result.get("labels") or []
    if rows:
        missing_dates = sum(1 for r in rows if not r.get("date"))
        # if many missing dates, request retry or flag for human review
        if missing_dates > max(3, 0.1 * len(rows)):
            return {"action": "retry", "reason": "many rows missing dates", "missing": missing_dates, "confidence": 0.3}
        # check simple debit/credit numeric sanity (non-negative)
        bad_amounts = sum(1 for r in rows if (r.get("debit") is not None and float(r.get("debit") or 0) < 0) or (r.get("credit") is not None and float(r.get("credit") or 0) < 0))
        if bad_amounts > 0:
            return {"action": "flag", "reason": "negative values found", "bad_amounts": bad_amounts, "confidence": 0.5}
    # For analysis results, ensure values not absurd (NaN)
    if "analysis" in result:
        analysis = result["analysis"]
        # basic check
        if analysis and any(v is None for v in analysis.values()):
            return {"action": "flag", "reason": "analysis contains nulls", "confidence": 0.5}
    return {"action": "accept", "confidence": 0.95}
