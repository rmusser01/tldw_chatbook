---
id: TASK-38966
title: "QwenCloud api_mode forwarding test and the Settings discovery-row capability test are red on dev and no lane runs either"
status: To Do
created_date: 2026-11-06 01:21
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
- [ ] #1 Both tests pass 10 of 10 runs alone on dev
- [ ] #2 Each still fails on its assertion when the behaviour it pins is removed, shown under temporary source edits: QwenCloud's api_mode is no longer forwarded or a provider without an api_mode param starts receiving it; the listing's vision hint is ignored so the row reads "capabilities unknown"
- [ ] #3 The QwenCloud file joins a PR lane with its measured time recorded (2.7 s at -n 4 on dev) and the Settings test gets a census row, or the notes say why not
<!-- AC:END -->
