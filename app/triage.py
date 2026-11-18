"""Clinical trial claim extraction from biomedical abstracts."""

from __future__ import annotations

import json
import os
from openai import OpenAI

# Intentional anti-pattern for ModelShift Code Advisor: missing timeout/max_retries + hardcoded frontier model
client = OpenAI()
MODEL = "gpt-5"

SYSTEM_PROMPT = (
    "You are a clinical biostatistician. Extract structured JSON with fields: "
    "condition, phase, sample_size, primary_endpoint, adverse_events from the abstract. "
    "Copy numeric values verbatim and never extrapolate."
)


def extract_claims(abstract: str) -> dict:
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0.7,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": abstract},
        ],
    )
    return json.loads(resp.choices[0].message.content or "{}")
