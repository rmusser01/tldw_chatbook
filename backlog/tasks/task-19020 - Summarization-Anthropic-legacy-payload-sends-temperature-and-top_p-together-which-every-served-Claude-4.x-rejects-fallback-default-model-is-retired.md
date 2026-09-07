---
id: TASK-19020
title: >-
  Summarization Anthropic legacy payload sends temperature and top_p together,
  which every served Claude 4.x rejects; fallback default model is retired
status: Done
assignee:
  - '@claude'
created_date: '2026-08-20 15:15'
updated_date: '2026-08-20 15:48'
labels:
  - llm
  - bug
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during TASK-18802 live verification. summarize_with_anthropic's legacy-model payload (unchanged by 18802, which only gates the families that reject sampling params outright) sends temperature, top_k AND top_p together. PROBE-VERIFIED against api.anthropic.com on 2026-08-20: claude-haiku-4-5 and claude-sonnet-4-5 with that exact trio return HTTP 400 {"type":"invalid_request_error","message":"`temperature` and `top_p` cannot both be specified for this model. Please use only one."} (req_011CeEDXPHNyF7apkaZepbTN, req_011CeEDXa9V99yBoHN5vcjDG), while temperature alone and temperature+top_k both return 200 (req_011CeEDXQwqi7yXoozbdrXFX, req_011CeEDXVk4nXXCoBGdf9mFm). Separately, the function's own fallback default get_cli_setting('anthropic_api','model','claude-3-haiku-20240307') names a RETIRED model: the same probe run returns HTTP 404 not_found_error (req_011CeEDXZ8iS29MZCgyySwQa). Net effect: with an Anthropic model configured that still accepts sampling params (Haiku 4.5, Sonnet 4.5, Opus 4.6), or with no model configured at all, summarization still fails after 18802. Chat path is NOT affected (chat_with_anthropic sends at most one of temperature/top_p). Fix should probe-verify which families reject the combination, express it as a model_capabilities predicate per the 18414/18802 design, and refresh the stale default model id.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Summarizing with claude-haiku-4-5 or claude-sonnet-4-5 completes instead of returning HTTP 400
- [x] #2 The combination rule is expressed via model_capabilities rather than a name check in the request builder
- [x] #3 The summarization path's fallback default Anthropic model is a currently-served model
- [x] #4 Models unaffected by the combination rule are unchanged, pinned by a regression test
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Probe boundary: combo trio on claude-opus-4-6 / claude-sonnet-4-6 / claude-opus-4-5 (standalone curl), post-fix shape control on opus-4-6\n2. Add immutable predicate anthropic_model_rejects_temperature_top_p_combination to model_capabilities.py (18414/18802 design: parsed tier-first family, outside config tables)\n3. Red-first pins in Tests/LLM_Calls/test_summarization_model_capabilities.py: combo models send temperature+top_k without top_p; modern-family payloads unchanged; fallback default resolves to a served model; predicate unit pins\n4. Rewire summarize_with_anthropic: drop top_p when the predicate is true (temperature precedence, top_k kept); replace retired fallback default claude-3-haiku-20240307 with claude-haiku-4-5\n5. Mutation-test the predicate consultation; targeted suites incl. diagnostic-privacy manifest\n6. Live-verify via production analyze() seam: haiku-4-5 + sonnet-4-5 summaries, no-model-configured fallback, sonnet-5 modern control; dev control worktree reproduces the 400\n7. Sweep other retired-id sites, file follow-up; report + PR
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed both halves via a new immutable predicate anthropic_model_rejects_temperature_top_p_combination in model_capabilities.py (18414/18802 design: parsed tier-first family, major >= 4, outside the user-overridable capability tables). Probe-first: claude-opus-4-6 (req_011CeEFGsbHd7VCjcjz4etar), claude-sonnet-4-6 (req_011CeEFGuRfeCzC6PiLyDtFb) and claude-opus-4-5 (req_011CeEFGvySC6z61NDRH5uN5) all 400 on the temperature+top_p pair with the identical message, while temperature+top_k without top_p returns 200 (msg_011CeEFGzjeXQ6ftPf9KH45n) -- together with 18802's haiku-4-5/sonnet-4-5 probes, every served sampling-accepting model rejects the pair. summarize_with_anthropic now sends temperature+top_k and drops top_p for those families (chat-path precedence mirrored; no warning log, deliberately -- the diagnostic-privacy manifest freezes this module's reviewed diagnostic set, same call 18802 made). Fallback default claude-3-haiku-20240307 (retired, 404) replaced with claude-haiku-4-5, its served successor in the same haiku lineage. Evidence: red-first 6 failed/75 passed -> 81 passed; targeted gate 1075 passed incl. the diagnostic-privacy manifest; collect-only 51028/0 errors; three mutations (invert consultation, narrow to major>=6, widen to unparsed ids) kill 7/22/9 pins, Edit-restored with empty git diff. Live via the production analyze() seam with a passthrough wire spy: haiku-4-5, sonnet-4-5, the no-model-configured fallback (wire model claude-haiku-4-5) and sonnet-5 modern control all return real summaries on the fix; clean origin/dev control worktree reproduces HTTP 400 (haiku-4-5 trio) and HTTP 404 (retired fallback default). Files: tldw_chatbook/model_capabilities.py, tldw_chatbook/LLM_Calls/Summarization_General_Lib.py, Tests/LLM_Calls/test_summarization_model_capabilities.py, Docs/superpowers/plans/2026-08-20-task-19020-report.md. Filed TASK-19048 for the remaining retired-id sites (character_defaults template, code_audit_tool, deprecated Tools_Settings_Window).
<!-- SECTION:NOTES:END -->
