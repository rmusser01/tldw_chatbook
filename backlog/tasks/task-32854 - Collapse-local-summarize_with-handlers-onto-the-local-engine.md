---
id: TASK-32854
title: Collapse the eight local summarize_with handlers onto the local engine
status: In Progress
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reconnaissance + first safe step, 2026-09-20 (branch `fix/cascade-summarize-local`, commit `73b7c0af67`): **landed** — General's dead module-level `api_key = get_cli_setting(...)` read deleted (the 2026-09-17 P3: secret into an unused global, one admission handshake per import); this is the prerequisite for 32854 because the local handlers must LAZY-import `_post_with_retry` from General (General imports Local for its dispatch table — module-level would be circular) and a side-effectful General module body made that unsafe. **Attempted and reverted** — the llama migration itself (helper transport + stream close + body-leak redaction) worked code-wise and its own pins passed, but the three local config suites (test_llama_summarizer_config / test_kobold_tabby_config / test_custom_openai_credential_resolution) destabilized well beyond their single pre-existing admission failure each (13+ failures/errors in llama's file): their config-install and post-seam infrastructure needs its own reconciliation pass BEFORE the handler migrations land — including positional-vs-kwarg url expectations, keep-profile/admission interplay, and the kobold KeyError'url' seam assertions. Reconciliation + llama landed 2026-09-20 (`15ca8bd0f3`, 1/8): (1) the three local config suites joined keep_bootstrap_profile; (2) analyze()'s consume_generator now unwraps recovery_review's _OpenAIStream -- the wrapper passed through unconsumed since TASK-32628, so kobold/tabby generator bodies never ran (verified failing at pristine dev; the boundary test's KeyError 'url' was exactly this); (3) llama tests' session seams moved to Summarization_General_Lib alongside the shared transport (both modules patched during the remaining migrations). llama: helper transport (real retry set), stream close-on-abandon pinned, non-200 return no longer interpolates the body. 488 passed across the seven affected suites. Remaining: vllm -> ollama -> oobabooga -> tabby -> custom x2 (same pattern), kobold last (own wire + task-17387).
<!-- SECTION:NOTES:END -->
