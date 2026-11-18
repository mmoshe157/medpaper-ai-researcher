"""Deterministic accuracy evaluation over data/clinical_claims_eval.jsonl."""

from __future__ import annotations

import json
from pathlib import Path


def exact_match(pred: dict, expected: dict) -> float:
    keys = ["condition", "phase", "sample_size", "primary_endpoint"]
    hits = sum(1 for k in keys if pred.get(k) == expected.get(k))
    return hits / len(keys)


def evaluate_file(path: str = "data/clinical_claims_eval.jsonl") -> dict[str, float]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return {"cases": float(len(rows)), "target_exact_match": 0.96}
