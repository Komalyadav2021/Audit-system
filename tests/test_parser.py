# tests/test_parser.py
from parsers.bank_statement_parser import rule_based_labeling

def test_rule_labeling_simple():
    rows = [{"line": "2025-01-01 Salary 1,000.00"}, {"line": "2025-01-02 Grocery -50.00"}]
    labeled = rule_based_labeling([r["line"] for r in rows])
    assert len(labeled) == 2
    assert any(r.get("date") for r in labeled)
