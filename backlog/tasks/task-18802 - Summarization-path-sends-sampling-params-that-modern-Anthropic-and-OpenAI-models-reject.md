---
id: TASK-18802
title: >-
  Summarization path sends sampling params that modern Anthropic and OpenAI
  models reject
status: Done
assignee:
  - '@claude'
created_date: '2026-08-18 23:55'
updated_date: '2026-08-20 15:18'
labels:
  - llm
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Summarization_General_Lib.py builds its own provider payloads and applies no per-model capability gate at all, so it reproduces exactly the defect TASK-18414 fixed on the chat path -- on a different, still-live code path.

summarize_with_anthropic (around lines 1048-1061) unconditionally sends temperature, top_k and top_p alongside get_cli_setting('anthropic_api', 'model', ...). PROBE-VERIFIED: the exact payload that function builds, sent to api.anthropic.com with a real key, returns

  HTTP 400 {"type":"error","error":{"type":"invalid_request_error","message":"`temperature` is deprecated for this model."},"request_id":"req_011CeB7m2VQbn2JXsd3LcQQy"}

on claude-opus-5. The same holds for Fable 5, Mythos 5, Opus 4.8, Opus 4.7 and Sonnet 5 -- and the shipped [api_settings.anthropic] model default is now claude-sonnet-5, one of them. summarize_with_openai (around lines 870-880) has the same shape, unconditionally sending max_tokens and temperature with openai_api.model (CODE-READ, not probe-verified: expected to 400 on gpt-5 / o-series / gpt-5.6).

This is reachable: analyze() in this module is imported by Web_Scraping/WebSearch_APIs.py, Web_Scraping/Article_Extractor_Lib.py, Local_Ingestion/{Book,Document,Image}_Processing_Lib.py and Local_Ingestion/XML_Ingestion.py, so ingestion and web-search summarization fail against the default configured model.

Found by a deliberate sweep for the same pattern while fixing TASK-18414, which added anthropic_model_rejects_sampling_params to model_capabilities.py; this path simply never consults it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Summarizing with a modern Anthropic model (Opus 5, Fable 5, Opus 4.8, Opus 4.7, Sonnet 5) completes instead of returning HTTP 400
- [x] #2 Summarizing with an OpenAI reasoning model completes, with the correct token-cap parameter name
- [x] #3 The summarization path reuses the model_capabilities predicates rather than adding its own model-name checks
- [x] #4 Models that still accept these parameters are unchanged, pinned by a regression test
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Probe api.openai.com with the exact summarize_with_openai payload shape (temperature+max_tokens) against modern reasoning-family models (gpt-5, gpt-5.6, o4-mini) and a legacy control (gpt-4o); capture verbatim 400 bodies; establish which params are rejected and the correct token-cap name\n2. Add immutable OpenAI predicates to model_capabilities.py (outside config-driven tables), family-matched per 18414 design\n3. Write RED payload pins for summarize_with_anthropic and summarize_with_openai (rejecting models omit params, legacy models unchanged)\n4. Rewire summarize_with_anthropic to consult anthropic_model_rejects_sampling_params; rewire summarize_with_openai to consult the new predicates\n5. Mutation-test predicate consultation; targeted suites\n6. Live-verify AC#1/#2 through the production payload builder with real keys + dev control on clean origin/dev\n7. Sweep other summarize_with_* functions, file follow-up; update 18803 with probe evidence
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Both halves fixed by consulting capability predicates in model_capabilities.py; no new model-name checks and no new logging calls (the diagnostic-privacy manifest hard-codes this module's diagnostic counts).

**Anthropic.** summarize_with_anthropic now gates temperature/top_k/top_p on the existing anthropic_model_rejects_sampling_params (TASK-18414); the function sends no thinking config, so the fixed-budget predicate has nothing to gate. Legacy models keep the exact prior payload (AC #4 pins).

**OpenAI, probe-first.** Standalone curl with the exact payload shape this function builds established: gpt-5/gpt-5.6/o3/o4-mini 400 on max_tokens (unsupported_parameter -> 'Use max_completion_tokens instead') and on temperature=0.7 (unsupported_value; temperature=1 accepted); max_completion_tokens + no sampling -> 200; gpt-4o/gpt-4.1 accept the old shape unchanged. Two new immutable predicates (openai_model_rejects_sampling_params, openai_model_requires_max_completion_tokens) over one (series, major) family table {(gpt,5),(o,1),(o,3),(o,4)}, boundary-safe (o365-copilot, olmo-7b, gpt-4o, gpt-oss never parse), outside the user-overridable capability tables per the 18414 design. summarize_with_openai consults both.

**Evidence.** Red-first 17 failed/37 passed -> 54 passed. Mutation-tested three ways: inverted consultation in the builders (17 pins die), narrowed table dropping (gpt,5) (17 die), widened table adding (gpt,4) (5 die incl. legacy over-match pins -- two-sided). Targeted gate 1048 passed 0 failed (whole Tests/LLM_Calls incl. the diagnostic-privacy manifest suite, model_capabilities consumers); collect-only 50946, 0 errors. Live via pytest driving the production analyze() seam with real keys (scratch bootstrap config): claude-sonnet-5 (the shipped default) and gpt-5 both return real summaries; gpt-4o unchanged; clean-origin/dev control worktree reproduces the 400 both ways (anthropic 'Failed to process summary; status_code=400', openai 'Error: OpenAI API request failed: 400 Client Error').

**Discovered, filed, not fixed:** TASK-19020 -- the legacy Anthropic payload sends temperature AND top_p together, which every served Claude 4.x rejects (probe req_011CeEDXPHNyF7apkaZepbTN), and the fallback default claude-3-haiku-20240307 is retired (404). TASK-19021 -- same unconditional-params shape in the six remaining summarize_with_* providers. TASK-18803 updated with the OpenAI probe bodies.

Files: tldw_chatbook/model_capabilities.py, tldw_chatbook/LLM_Calls/Summarization_General_Lib.py, Tests/LLM_Calls/test_summarization_model_capabilities.py (54 pins), Docs/superpowers/plans/2026-08-20-task-18802-report.md.
<!-- SECTION:NOTES:END -->
