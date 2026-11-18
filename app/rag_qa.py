"""Citation-grounded biomedical RAG API endpoint (with dynamic prefix breaking prompt cache)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from fastapi import APIRouter
from openai import OpenAI

router = APIRouter()
client = OpenAI(timeout=45.0, max_retries=2)

STATIC_INSTRUCTIONS = (
    "Answer the clinician's question using ONLY the provided PubMed passages and ClinicalTrials.gov excerpts. "
    "Cite every quantitative efficacy or safety claim inline using [PMID:xxxx] or [NCTxxxxxxxx]. "
    "Never extrapolate statistical significance beyond the stated confidence intervals or hazard ratios. "
) * 20


def embed_query(query: str) -> list[float]:
    emb = client.embeddings.create(model="text-embedding-3-large", input=query)
    return list(emb.data[0].embedding)


@router.post("/papers/ask")
def ask_papers(question: str) -> dict[str, str]:
    vec = embed_query(question)
    context = f"Retrieved PMID:984122 (vec_dim={len(vec)}): Randomized Phase 3 trial showed 24% ARR reduction."
    # Deliberate broken cache prefix: UUID + timestamp placed BEFORE static instructions
    system = f"request_id={uuid.uuid4()} ts={datetime.now(UTC).isoformat()}\n{STATIC_INSTRUCTIONS}"
    resp = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return {"answer": resp.choices[0].message.content or ""}
