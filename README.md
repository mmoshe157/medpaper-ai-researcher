# MedPaper AI Researcher

Autonomous biomedical literature triage, clinical-trial claim extraction, and citation-grounded RAG assistant.

## Workflows
1. **Clinical Claim Extraction** (`app/triage.py`) - Extracts study phase, sample size, primary endpoint, and adverse events from abstracts.
2. **Literature Agent** (`app/agent_literature.py`) - Multi-step research agent querying PubMed and clinical trial registries.
3. **Citation RAG QA** (`app/rag_qa.py`) - Retrieves relevant study passages and generates evidence-backed answers.
