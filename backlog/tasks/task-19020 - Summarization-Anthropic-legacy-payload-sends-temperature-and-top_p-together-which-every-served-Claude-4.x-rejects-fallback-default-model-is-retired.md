---
id: TASK-19020
title: >-
  Summarization Anthropic legacy payload sends temperature and top_p together,
  which every served Claude 4.x rejects; fallback default model is retired
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-20 15:15'
updated_date: '2026-08-20 15:36'
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
- [ ] #1 Summarizing with claude-haiku-4-5 or claude-sonnet-4-5 completes instead of returning HTTP 400
- [ ] #2 The combination rule is expressed via model_capabilities rather than a name check in the request builder
- [ ] #3 The summarization path's fallback default Anthropic model is a currently-served model
- [ ] #4 Models unaffected by the combination rule are unchanged, pinned by a regression test
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Probe boundary: combo trio on claude-opus-4-6 / claude-sonnet-4-6 / claude-opus-4-5 (standalone curl), post-fix shape control on opus-4-6\n2. Add immutable predicate anthropic_model_rejects_temperature_top_p_combination to model_capabilities.py (18414/18802 design: parsed tier-first family, outside config tables)\n3. Red-first pins in Tests/LLM_Calls/test_summarization_model_capabilities.py: combo models send temperature+top_k without top_p; modern-family payloads unchanged; fallback default resolves to a served model; predicate unit pins\n4. Rewire summarize_with_anthropic: drop top_p when the predicate is true (temperature precedence, top_k kept); replace retired fallback default claude-3-haiku-20240307 with claude-haiku-4-5\n5. Mutation-test the predicate consultation; targeted suites incl. diagnostic-privacy manifest\n6. Live-verify via production analyze() seam: haiku-4-5 + sonnet-4-5 summaries, no-model-configured fallback, sonnet-5 modern control; dev control worktree reproduces the 400\n7. Sweep other retired-id sites, file follow-up; report + PR
<!-- SECTION:PLAN:END -->
