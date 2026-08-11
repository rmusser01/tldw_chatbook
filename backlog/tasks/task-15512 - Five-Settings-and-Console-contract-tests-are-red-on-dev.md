---
id: task-15512
title: Five Settings and Console contract tests are red on dev
status: To Do
assignee: []
labels:
  - test-health
  - settings
priority: medium
---

## Description

Five tests fail on `origin/dev` with no local changes. They were found while
baselining task-15270 (running the same modules against the branch and against
dev to tell genuinely-new failures from inherited ones), and they are not caused
by that branch. Filed so they are not silently re-attributed to whatever change
happens to run next to them.

Measured on `origin/dev` at `d85e6cff1`:

- `test_console_workbench_contract.py::test_console_registers_footer_workbench_shortcuts`
  -- footer hint text drift: expected `... nter send | Ctrl+K ...`, actual
  `... nter send / queue | Ctrl+K ...`. A `/ queue` affordance was added to the
  composer hint without updating the contract test.
- `test_settings_configuration_hub.py::test_settings_ownership_records_cover_categories_and_runtime_boundaries`
  -- ownership records differ by one entry:
  `model_capabilities.models.<model>.context_window` is present where the
  expected tuple does not carry it.
- `test_settings_configuration_hub.py::test_settings_console_behavior_saves_display_name_exactly`
  -- times out waiting for the toast `Console behavior settings saved.`
- `test_settings_configuration_hub.py::test_settings_provider_category_saves_provider_defaults_without_sampling`
  -- saved-settings list comes back empty where
  `('chat_defaults', 'provider', 'llama_cpp')` and `('chat_defaults', 'model', 'qwen')`
  were expected.
- `test_settings_configuration_hub.py::test_settings_provider_switch_does_not_save_stale_endpoint`
  -- same shape: empty saved list where a `chat_defaults` provider entry was expected.

The last three are the interesting cluster: two independent tests observe that a
Settings save produced no persisted rows, and a third observes that a save toast
never arrives. That is consistent with the Settings save path not completing,
which would be user-visible, so triage should establish whether these are stale
test contracts or a broken save before any of them is adjusted to match current
behaviour.

## Acceptance Criteria

- [ ] Each of the five failures is attributed to its causing change, with the commit identified
- [ ] It is established whether the three save-related failures are stale contracts or a genuine break in the Settings save path, with evidence either way
- [ ] Any genuine product break found is fixed rather than absorbed into the tests' expectations
- [ ] All five pass on dev
