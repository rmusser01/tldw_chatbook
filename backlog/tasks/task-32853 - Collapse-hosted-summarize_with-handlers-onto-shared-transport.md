---
id: TASK-32853
title: Collapse the nine hosted summarize_with handlers onto shared transport
status: In Progress
assignee: []
created_date: '2026-09-19 08:24'
labels:
  - core-review
  - review-cascade
dependencies:
  - TASK-19642.10
parent_task_id: TASK-32850
references:
  - qa/cascade-review-2026-09-19/report.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The summarization layer never received the transport collapse: `LLM_Calls/Summarization_General_Lib.py` carries nine hosted `summarize_with_*` handlers (openai :904, anthropic :1109, cohere :1381, groq :1631, openrouter :1813, huggingface :2046, deepseek :2236, mistral :2424, google :2646 — 1,936 LOC of handlers), each re-rolling "session + Retry adapter + post + stream_generator". The 2026-09-17 review documented the drift: only 2 of 16 total copies close the response on abandon; `summarize_with_deepseek` hardcodes `https://api.deepseek.com/chat/completions` (:2257, :2316) ignoring `api_base_url`; three handlers double-yield the accumulated text after the deltas; retry loops are hand-rolled per handler.

Summarization is a chat call + prompt template + `choices[0]` extraction — ~15-30 provider-specific LOC survive per provider. OpenAI-compatible providers should route through `hosted_chat.py`; the non-OpenAI wire shapes (anthropic, google, cohere) get one shared `_post_with_retry` helper with their payload assembly kept.

Constraint: `test_summarization_diagnostic_privacy.py` freezes per-function log call sites, so consolidation re-keys the diagnostic ledger — do it coherently per the task-17387 pattern, not by deleting assertions. Error-STRING return shape is a caller contract; preserve it. ADR required: no — ADR-062-consistent; the ledger re-key follows task-17387's precedent.

Source: cascade review 2026-09-19 — `qa/cascade-review-2026-09-17`-adjacent evidence in `slices/LLM.md`; cascade framing in `qa/cascade-review-2026-09-19/report.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The nine hosted handlers share one transport path (hosted_chat for OpenAI-compatible providers; one `_post_with_retry` for the anthropic/google/cohere wire shapes); each handler is a profile + prompt assembly
- [ ] #2 No hardcoded endpoint URLs remain; `api_base_url` is honored everywhere it exists for chat
- [ ] #3 Abandoned streaming consumers close the response in all nine; the double-yield defect class is gone and pinned
- [ ] #4 Retry-After honoring goes through the shared, bounded policy (no uncapped sleeps)
- [ ] #5 The diagnostic ledger is re-keyed coherently and its tests updated to the new call sites
- [ ] #6 Caller-visible error-string returns are unchanged; ingest analysis paths still work
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Phase A (safety-net restoration) landed 2026-09-19 via PR #2746 (commits on `fix/cascade-wave1`): the two summarization suites joined `keep_bootstrap_profile`, restoring 106 admission-red tests (the TASK-19642.10 signature -- every `summarize_with_*` reads provider settings through `get_cli_setting` and the broad excepts turn the admission failure into an error STRING); the ten stream-laziness tests unmasked by that fix now accept `recovery_review._OpenAIStream` (TASK-32628's retention wrapper); and the manifest boundary is re-pinned. Both suites are green for the first time since 2026-09-16 -- the collapse work can proceed against a live safety net. Phase B (the six OpenAI-compatible handlers onto `owned_json_post`) and Phase C (anthropic/google/cohere behind one `_post_with_retry`) are not started. Phase B 1/6 landed 2026-09-20 via PR #2753 (stacked on #2752): summarize_with_groq migrated onto owned_json_post as the exemplar -- config-read discipline holds the ledger's settings pin, the never-closing stream relay now closes exactly-once, non-200 failures return typed-redacted errors instead of the response body, and the ledger/inventory re-key procedure is proven end-to-end (site marked deleted with reason, counts 10->11, fake gains the iter_content seam). deepseek+mistral landed on the same branch (PR #2753 grew): deepseek drops its hardcoded URL (api_base_url honored) and the double-yield; mistral honors api_base_url; both frozen status logs preserved verbatim via a typed-error handler that binds the error as `response` (exact reviewed shape survives, same caller-visible status strings, body never leaks). Remaining, with reconnaissance recorded 2026-09-20: **openai** landed 2026-09-20 (`75138ba5c0`, PR #2753 grew to 4/6): transport-only swap, capability-gated payload untouched, the frozen 'Endpoint configured' log survived shape-identical so the ledger reconciled with zero fixture changes. **HuggingFace is RECLASSIFIED to Phase C**: it posts the legacy `inputs` API (api-inference.huggingface.co/models/{model}), not OpenAI chat-completions -- the engine's wire shape does not fit; its endpoint-modernization question (the review's unverified 404 claim) rides with Phase C. So Phase B's OpenAI-compatible set is COMPLETE at 4/4 once openrouter lands; remaining in B: openrouter only — **openrouter landed 2026-09-20 (`24d29c8b6a`), PHASE B COMPLETE at 5/5**: its consume-and-return-string contract, per-chunk 'Content received' log, and every frozen statement survived verbatim; both status-failure returns stopped leaking the response body; the stream response now closes exactly once; the metadata status logs keep their approved response.status_code shape via the typed-error response-binding (the guard rejects bare-local expressions). PR #2753 retitled to the full Phase B. Phase C opened 2026-09-20 (PR #2758, stacked on #2753): the shared `_post_with_retry` transport landed with **anthropic** as first consumer -- its real retry semantics (500 + network only; the urllib3 adapter it built was dead code) preserved exactly, per-attempt provider network log superseded (private site deleted with reason), stream close-on-abandon fixed and pinned, session pooling + default timeout gained. cohere + google landed 2026-09-20 (`13a9c28d21`, PR #2758 grew to 3/4): cohere kept its real retry set and stream-close; its status returns stopped leaking the response body. google got the review's hardcoded-URL fix (the old base path had no /chat/completions route -- the suspected live 404) + stream close-on-abandon; streaming failures keep the old raise surface so the frozen status statement stays single-occurrence. ALSO fixed the privacy suite's last pre-existing red: test_google_input_json_error patched loads on the SHARED json module, breaking the storage admission's own reads since TASK-32628 -- a module-local json proxy contains it; **the diagnostic-privacy suite is 257/257 green for the first time**. Remaining: huggingface only (legacy inputs API + endpoint-modernization decision) -- Phase C's final slice. **openrouter is the heaviest re-key of the set**: its summarizer CONSUMES the stream inside the function and returns the full string (not a generator), posts `data=json.dumps(...)` with `url=` as a kwarg (the engine posts the url positionally and sends `json=` -- ~8 tests pin these shapes and need re-keying), logs `OpenRouter Stream: Content received` PER CHUNK (info), and its non-200/except paths carry six provider-prefixed log statements whose ledger classifications must be checked for the freeze before any statement moves (use the response-binding trick from deepseek/mistral where shapes must survive verbatim). Do openrouter last, after the mechanical two. (deepseek also drops its hardcoded URL + double-yield), then Phase C. AC#4's engine side landed 2026-09-20 via PR #2752 (stacked on #2747): both uncapped provider-Retry-After sleeps (hosted_chat._retry_delay and qwencloud's copy) clamp to a named 60s constant, TDD-pinned — every future migrated summarizer inherits the bound for free.
<!-- SECTION:NOTES:END -->
