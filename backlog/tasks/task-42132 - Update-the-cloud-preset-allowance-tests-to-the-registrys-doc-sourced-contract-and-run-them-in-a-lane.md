---
id: TASK-42132
title: "Update the cloud preset allowance tests to the registry's doc-sourced contract and run them in a lane"
status: To Do
assignee: []
created_date: 2026-11-20 20:06
updated_date: 2026-11-20 20:45
labels:
- follow-up
- providers
- tests
priority: low
references:
- Tests/LLM_Calls/test_inference_cloud_presets.py
- tldw_chatbook/provider_registry.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the providers stream (round 18). At d85e90bc9e four nodes in `Tests/LLM_Calls/test_inference_cloud_presets.py` fail (`-n 4`, through the test wrapper):
- `test_allowances_ship_empty_pending_first_capture[together]`, `[fireworks]` and `[cerebras]` (line 105) still expect the provisional empty allowances;
- `test_registry_carries_the_provisional_allowance_comment` (line 120) looks for "PROVISIONAL PENDING FIRST LIVE CAPTURE" in `tldw_chatbook/provider_registry.py`.

The registry records the cause in its comment above `TOGETHER` (`provider_registry.py:319-330`): 3e618d7119 (2026-10-16) replaced the provisional comment with the TASK-34367.3-.5 reconciliation note and shipped allowances taken mostly from each provider's official schema (offline contracts; live qualification stays with TASK-33640). The shipped sets are:
- Together (`:379-381`): response `{prompt, warnings}`, choice `{logprobs, seed, top_logprobs, text}`, message `{reasoning}`;
- Fireworks (`:418-419`, `:423`): response `{perf_metrics, prompt_token_ids}`, choice `{logprobs, raw_output}`, tool call `{index, name}`;
- Cerebras (`:444-446`): response `{time_info, service_tier, service_tier_used}`, choice `{logprobs, reasoning_logprobs}`, message `{reasoning}`.

Two of these sets are proved by live capture rather than documentation, and the test already pins them with fixture citations through `_CAPTURED_CHOICE_ALLOWANCES` and `_CAPTURED_TOOL_CALL_ALLOWANCES`: Together's choice `logprobs` (`Tests/fixtures/cloud_live/together.json`, TASK-34362) and Fireworks' tool call `{index, name}` (`Tests/fixtures/cloud_live/fireworks.json`, b2ca240289, TASK-34364).

**Overlap.** The open tasks TASK-34367.3, TASK-34367.4 and TASK-34367.5 each have this test update as their AC#3 ("targeted regressions replace provisional empty-allowance assertions"). This task is only the test update to the shipped contract (doc-sourced, apart from the two captured sets) plus a lane; the reconciliation itself stays with those tasks. No PR lane runs this file, so the reds went unnoticed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The tests assert the contract the registry ships, for all four allowance families of each preset: Together's response `{prompt, warnings}`, choice `{logprobs, seed, top_logprobs, text}` and message `{reasoning}`; Fireworks' response `{perf_metrics, prompt_token_ids}`, choice `{logprobs, raw_output}` and tool call `{index, name}`; Cerebras' response `{time_info, service_tier, service_tier_used}`, choice `{logprobs, reasoning_logprobs}` and message `{reasoning}`; every other family empty. Together's choice `logprobs` and Fireworks' tool call `{index, name}` keep their live-capture fixture citations (TASK-34362, TASK-34364) and are not folded into the doc-sourced sets
- [ ] #2 The provisional-comment test is replaced by one that pins the reconciliation note's source per preset, and the four nodes (or their replacements) pass
- [ ] #3 The file runs in a PR lane, so a later drift fails CI
- [ ] #4 The notes say which part of TASK-34367.3-.5's AC#3 this covers; those task files are not edited here
<!-- AC:END -->
