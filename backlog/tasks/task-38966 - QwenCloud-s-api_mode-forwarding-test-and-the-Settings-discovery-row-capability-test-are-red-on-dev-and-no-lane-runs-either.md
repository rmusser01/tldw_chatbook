---
id: TASK-38966
title: "QwenCloud's api_mode forwarding test and the Settings discovery-row capability test are red on dev, and no lane runs either"
status: Done
assignee:
- '@claude'
created_date: 2026-11-06 01:21
updated_date: '2026-11-06 04:52'
labels:
- follow-up
- tests
- providers
- settings
priority: low
references:
- Tests/Chat/test_qwencloud_provider_contract.py
- tldw_chatbook/Chat/Chat_Functions.py
- Tests/UI/test_settings_provider_test_draft.py
- tldw_chatbook/UI/Screens/settings_screen.py
- scripts/ui_pr_gate_census.txt
- .github/workflows/derived-artifacts.yml
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05. Both tests failed 3 of 3 runs alone at load 12-14. No branch due to merge before this one touches either test file or its source. Both reds look like test drift.

**QwenCloud api_mode forwarding.**
- `Tests/Chat/test_qwencloud_provider_contract.py::test_chat_api_call_forwards_qwencloud_mode_base_and_tools` fails at `:257` with `assert 'api_mode' not in {'api_mode': 'responses', 'input_data': [...], 'model': 'representative-model', 'streaming': False}`.
- Its loop (`:238-257`) treats openai and deepseek as providers that must not receive `api_mode`. But f5a63125ff (2026-10-12, "explicit DeepSeek mode", ADR 064) added `"api_mode": "api_mode"` to `PROVIDER_PARAM_MAP["deepseek"]` (`tldw_chatbook/Chat/Chat_Functions.py:424`).
- The file has 7 tests (6 pass) and takes 2.7 s at `-n 4`. No lane runs it.

**Settings discovery-row capability.**
- `Tests/UI/test_settings_provider_test_draft.py::test_discovery_row_reads_a_fresh_models_capability_from_its_listing` (`:1976`) fails with `AttributeError: 'SettingsScreen' object has no attribute '_nodes'`.
- The test builds a screen with no DOM (`_bare_settings_screen`, `:717-722`, via `SettingsScreen.__new__`). `_model_discovery_selection_options` (`settings_screen.py:16288`) now calls `_provider_widget_value()`, whose `query_one` (`:15528`) raises AttributeError on such a screen. The method only catches `QueryError` (`:15542`).
- The test came in with TASK-34402 (fda5d3bd30), when this method read no widgets. The widget read came later, in TASK-33007.6 (0d2aca9cfc).
- The census lists other functions from this file, not this one.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both tests pass 10 of 10 runs alone on dev
- [x] #2 Each still fails on its assertion when the behaviour it pins is removed, shown under temporary source edits: QwenCloud's api_mode is no longer forwarded or a provider without an api_mode param starts receiving it; the listing's vision hint is ignored so the row reads "capabilities unknown"
- [x] #3 The QwenCloud file joins a PR lane with its measured time recorded (2.7 s at -n 4 on dev) and the Settings test gets a census row, or the notes say why not
<!-- AC:END -->

## Implementation Plan

1. QwenCloud: the "never receives api_mode" loop drops deepseek, which has forwarded `api_mode` since f5a63125ff (ADR 064), for anthropic; openai stays. Both have no `api_mode` param.
2. Settings: the bare screen in the capability test gets the provider its form would show (`_provider_widget_value`), since the method under test now reads that widget (TASK-33007.6). The product is unchanged: the red is test drift.
3. Show each red on dev, then 10 of 10 green alone; negative controls under temporary source edits (api_mode dropped from QwenCloud's map / added to openai's; the listing's capability hint ignored).
4. Lanes: the QwenCloud file is appended at the END of dreams-boundary's "Run Dreams and Guardian boundary contracts" step (pr-fast-lane is at its cap), with the `Tests/CI` lane pins updated, after reading recent job times. The Settings test gets a census row at the tail.

## Implementation Notes

Both reds were test drift; the product code is unchanged.
- QwenCloud (`Tests/Chat/test_qwencloud_provider_contract.py`): the loop that pins "a provider without an `api_mode` param never receives it" listed deepseek, which forwards `api_mode` since f5a63125ff (ADR 064). Only deepseek and qwencloud have the param in `PROVIDER_PARAM_MAP`, so the loop now checks openai and anthropic, with a comment saying why.
- Settings (`Tests/UI/test_settings_provider_test_draft.py::test_discovery_row_reads_a_fresh_models_capability_from_its_listing`): two drifts since TASK-33007.6. The method reads the provider from its widget to mark saved models, so the bare screen gets `_provider_widget_value`. And a row is now `Text` that starts with its selected word, so the test reads each id from its option tuple. It still checks the order of the ids, each id inside its row, and each row's capability word, now as the row's last word.
- AC#1: red at base. The QwenCloud test failed in the covering run at `fu-base-r10` (`assert 'api_mode' not in {...}`), and the Settings test failed in the UI covering run at base. After the fix, each passed 10 of 10 runs alone (run.sh, `-p no:xdist`): QwenCloud 0.48-0.78 s, Settings 2.56-3.61 s.
- AC#2, temporary source edits, each reverted:
  - `"api_mode"` removed from `PROVIDER_PARAM_MAP["qwencloud"]`: failed at the captured-call `==`.
  - `"api_mode"` added to `PROVIDER_PARAM_MAP["openai"]`: failed with `assert 'api_mode' not in {'api_mode': 'responses', ...}`.
  - `_model_discovery_selection_options` skipped the listing's capability hint: failed at `rows[0][1].endswith("capabilities inferred")`.
- AC#3, lanes:
  - The QwenCloud file is appended at the END of dreams-boundary's "Run Dreams and Guardian boundary contracts" step (`.github/workflows/derived-artifacts.yml`), with `DREAMS_BOUNDARY_TARGETS` and the step argv pin in `Tests/CI/test_response_rules_critical_gate.py` updated. It pauses or drains no storage, so it does not go first.
  - Job times before the change (read-only `gh run view --json jobs`, six successful PR runs, 2026-11-04 19:44Z to 11-05 16:00Z): the step took 122-141 s and the job 6-7 min of its 20.
  - The file with the step's command (`pytest -p no:xdist Tests/Chat/test_qwencloud_provider_contract.py --timeout=180 --tb=short`): 7 passed in 2.14, 2.11 and 1.79 s, at load 36-38. It took 2.7 s at `-n 4` on dev.
  - The whole step in workflow order (`dreamstep.sh`, 27 targets, last the QwenCloud file): 529 passed in 383 s, serial and loaded.
  - The Settings test gets a node-id row at the tail of `scripts/ui_pr_gate_census.txt`, and `MINIMUM_FILES` is now the measured 1,701 entries. `check_ui_pr_gate_census.py` passes, and so does `Tests/CI` (1211 passed, 1 skipped).

Files: `Tests/Chat/test_qwencloud_provider_contract.py`, `Tests/UI/test_settings_provider_test_draft.py`, `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`, `Tests/CI/test_response_rules_critical_gate.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.
