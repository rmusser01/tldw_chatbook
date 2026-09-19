---
id: TASK-32854
title: Collapse the eight local summarize_with handlers onto the local engine
status: To Do
assignee: []
created_date: '2026-09-19 08:24'
labels:
  - core-review
  - review-cascade
dependencies: []
parent_task_id: TASK-32850
references:
  - qa/cascade-review-2026-09-19/report.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`LLM_Calls/Local_Summarization_Lib.py` (~2,340 LOC) hand-rolls eight local handlers (llama :400, kobold :645, oobabooga :917, tabby :1150, vllm :1409, ollama :1673, custom-openai :1945, custom-openai-2 :2204 — ~2,085 LOC) even though `LLM_API_Calls_Local.py` proved the collapse for chat: every local chat provider routes through one `_chat_with_openai_compatible_local_server` (:372). 33 `create_default_session`/`HTTPAdapter` occurrences sit in the summarization file alone.

Known defects this absorbs: the kobold/tabby generator-function bug (task-17387 — bare `yield` makes every call return the error string), `summarize_with_local_llm:90` hardcoding `http://127.0.0.1:8080/v1/chat/completions` while its chat sibling reads config, `summarize_with_ollama:1756` building a second session it never uses, and the same close-on-abandon drift as the hosted side. Local servers are OpenAI-compatible or have small per-server payload quirks (kobold forces non-streaming) — one shared local transport with per-server profiles.

ADR required: no — the local seam already exists; this extends it to summarization. Ledger constraint as in TASK-32853.

Source: cascade review 2026-09-19 — `qa/cascade-review-2026-09-19/report.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The eight local handlers route through one shared local transport with per-server profiles; no per-handler session/adapter construction remains
- [ ] #2 `summarize_with_local_llm` reads its endpoint from config like its chat sibling
- [ ] #3 The kobold/tabby generator-function defect class is structurally impossible in the shared transport (nested-stream pattern) and pinned by a test
- [ ] #4 The diagnostic-privacy ledger is re-keyed coherently with its tests
- [ ] #5 Local summarization suites pass; error-string contract preserved
<!-- AC:END -->
