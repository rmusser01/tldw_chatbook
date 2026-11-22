---
id: TASK-42132
title: "Update the cloud preset allowance tests to the registry's doc-sourced contract and run them in a lane"
status: Done
assignee: ['@claude']
created_date: 2026-11-20 20:06
updated_date: 2026-11-21 18:30
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
- [x] #1 The tests assert the contract the registry ships, for all four allowance families of each preset: Together's response `{prompt, warnings}`, choice `{logprobs, seed, top_logprobs, text}` and message `{reasoning}`; Fireworks' response `{perf_metrics, prompt_token_ids}`, choice `{logprobs, raw_output}` and tool call `{index, name}`; Cerebras' response `{time_info, service_tier, service_tier_used}`, choice `{logprobs, reasoning_logprobs}` and message `{reasoning}`; every other family empty. Together's choice `logprobs` and Fireworks' tool call `{index, name}` keep their live-capture fixture citations (TASK-34362, TASK-34364) and are not folded into the doc-sourced sets
- [x] #2 The provisional-comment test is replaced by one that pins the reconciliation note's source per preset, and the four nodes (or their replacements) pass
- [x] #3 The file runs in a PR lane, so a later drift fails CI
- [x] #4 The notes say which part of TASK-34367.3-.5's AC#3 this covers; those task files are not edited here
<!-- AC:END -->

## Implementation Plan

1. Replace `test_allowances_ship_empty_pending_first_capture` with a test that asserts each preset's four allowance families equal a doc-sourced table plus the two captured sets, which keep their own tables and fixture citations; every other family empty.
2. Replace `test_registry_carries_the_provisional_allowance_comment` with a test that pins each preset's schema source inside the registry's TASK-34367.3-.5 reconciliation note.
3. Show both red under temporary registry edits, then green.
4. Lane the file in `pr-fast-lane` ("Run fast PR contract") and add it to `FAST_LANE_TARGETS` in `Tests/CI/test_ci_queue_pressure_contract.py`.

## Implementation Notes

`Tests/LLM_Calls/test_inference_cloud_presets.py` now asserts the contract the registry ships.
- `test_allowances_are_the_documented_contract_plus_captured_sets[together|fireworks|cerebras]`: for each of the response, choice, message and tool-call families, the record's set equals `_DOCUMENTED_ALLOWANCES` (the doc-sourced sets from the task) plus the captured set, and any family neither names is empty. Together's choice `logprobs` stays in `_CAPTURED_CHOICE_ALLOWANCES` (cloud_live/together.json, TASK-34362), and Fireworks' tool call `{index, name}` stays in `_CAPTURED_TOOL_CALL_ALLOWANCES` (cloud_live/fireworks.json, TASK-34364). The test asserts a captured set never overlaps the documented one, so it cannot be folded in. Together's documented choice set is therefore `{seed, top_logprobs, text}`.
- `test_registry_reconciliation_note_names_each_presets_schema_source[...]` replaces the provisional-comment test. It reads the registry's `# TASK-34367.3-.5:` note up to its first blank comment line and asserts it names each preset's schema page (`docs.together.ai/reference/chat-completions`, `docs.fireworks.ai/api-reference/post-chatcompletions`, `inference-docs.cerebras.ai/api-reference/chat-completions`). The note is scoped to that block because a later paragraph cites the Cerebras page for another fact; the first version, scoped to the whole comment, stayed green with the Cerebras source removed from the note.
- The module docstring no longer says the presets ship empty, provisional sets.

**Red/green.** At base (dc371f0842) the four old nodes failed (`-n 4`: 4 failed, 20 passed). The new tests keep an invariant the registry already has, so red was shown under temporary registry edits, then reverted: dropping Together's response `warnings`, Fireworks' tool call `name`, and adding a Cerebras message field failed the three allowance nodes; removing the Together and Cerebras sources from the note failed those two source nodes. After: 26 passed.

**Lane (AC#3).** The file is in `pr-fast-lane`'s "Run fast PR contract" step (`.github/workflows/derived-artifacts.yml`) and in `FAST_LANE_TARGETS` (`Tests/CI/test_ci_queue_pressure_contract.py`), next to the other `Tests/LLM_Calls` entries. Serial cost with the lane's `--timeout=180`: 0.91 s, 26 passed (load average 15.5). `Tests/CI` at `-n 4`: 1811 passed, 2 skipped.

**Overlap with TASK-34367.3-.5 (AC#4).** Each of those tasks' AC#3 has two halves: "targeted regression fixtures replace provisional empty-allowance assertions" and "unknown fields still fail closed" (TASK-34367.4: "retain unknown-field/required-field rejection"). This task covers only the first half's test update: the provisional empty-allowance assertions are replaced by assertions of the shipped sets. It adds no regression fixtures and no fail-closed check for unknown fields, and it does not reconcile the allowances. Those stay with TASK-34367.3-.5, which are not edited here.

Files: `Tests/LLM_Calls/test_inference_cloud_presets.py`, `Tests/CI/test_ci_queue_pressure_contract.py`, `.github/workflows/derived-artifacts.yml`.
