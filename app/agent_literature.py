"""Multi-step biomedical literature agent (deliberately unbudgeted loop + 22 bloated tool schemas)."""

from __future__ import annotations

import os
from anthropic import Anthropic

client = Anthropic(max_retries=5)
AGENT_MODEL = os.getenv("LIT_AGENT_MODEL", "claude-opus-4-1")

# 22 tool schemas offered on every turn; only search_pubmed and fetch_trial are used in practice
TOOLS = [
    {"name": "search_pubmed", "description": "Search PubMed abstracts by MeSH query.", "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "fetch_trial", "description": "Fetch ClinicalTrials.gov NCT record.", "input_schema": {"type": "object", "properties": {"nct_id": {"type": "string"}}, "required": ["nct_id"]}},
    *[{"name": f"legacy_registry_{i}", "description": f"Deprecated regional medical archive connector #{i} with extended metadata schema and pagination filters.", "input_schema": {"type": "object", "properties": {"id": {"type": "string"}, "region": {"type": "string"}}}} for i in range(1, 21)],
]


def run_literature_agent(research_question: str) -> str:
    messages = [{"role": "user", "content": research_question}]
    while True:
        msg = client.messages.create(
            model=AGENT_MODEL,
            max_tokens=4096,
            system="You are an autonomous medical literature investigator. Verify every citation before answering.",
            tools=TOOLS,
            messages=messages,
        )
        if msg.stop_reason != "tool_use":
            return "".join(getattr(b, "text", "") for b in msg.content)
        messages.append({"role": "assistant", "content": msg.content})
        messages.append({"role": "user", "content": "Tool result placeholder: study NCT04819201 phase 3 completed."})
