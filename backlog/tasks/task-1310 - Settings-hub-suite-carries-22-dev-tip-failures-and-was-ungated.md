---
id: TASK-1310
title: Settings-hub suite carries 22 dev-tip failures and was ungated
status: Done
assignee:
  - '@claude'
created_date: '2026-07-28 13:30'
updated_date: '2026-07-29 01:10'
labels:
  - settings
  - tests
  - regressions
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
During TASK-1234's review, the first settings-hub gate on the fleet-findings branch surfaced 22 failing tests in Tests/UI/test_settings_configuration_hub.py present at the dev-tip base (byte-identical name sets at base 93bf5518c and branch HEAD — none caused by the branch): a provider/model-resolution TypeError family, a save_setting_to_cli_config/save_settings_to_cli_config naming-drift family, and a PrivatePathError. The suite was last known green in this program pre-#1050 and none of the recent trains gated it. Fix the regressions and keep the hub in routine verification gates.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All 22 named failures (see TASK-1234 review) pass or are individually dispositioned with root-cause notes.
- [x] #2 The originating dev commits are identified (naming drift especially).
- [x] #3 The hub suite is listed in the standard Console-area verification gates going forward.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the 22 failures locally (hub-baseline.log) and bucket by traceback shape: TypeError persisted_defaults (8), module AttributeError save_setting_to_cli_config (12), TldwCli.chat_api_provider_value AttributeError (1), PrivatePathError missing_parent (1).
2. git blame/git log -S each removed symbol to find the originating commit and read its diff to determine whether the change was deliberate (own tests, docstring intent, all other call sites updated) or accidental drift (dangling stale caller left in production).
3. For each family, fix the RIGHT side: update stale test expectations to the new deliberate contract, or fix production if a caller was missed.
4. Grep tldw_chatbook/ for any other stray references to removed symbols to rule out a live production bug (broken real save path).
5. Run the full hub suite (target 253/0) plus the parallel-runs/per-session regression backstop suites in blocking foreground calls.
6. Document per-failure dispositions and originating commits in Implementation Notes; append a lessons-testing-evidence.md entry for AC#3.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed all 22 dev-tip failures by tracing each to its originating commit and confirming the intent
was deliberate on the production side; no production drift remained live for real users (verified
below). All fixes landed in the test file. `Tests/UI/test_settings_configuration_hub.py` now passes
253/253; the regression backstop (`Tests/UI/test_console_parallel_runs.py` +
`Tests/Chat/test_console_run_state_per_session.py`, 46 tests) is green.

**Two originating commits account for all 22 failures:**

- `d15882398` "refactor(console): own provider selection by lifetime (task-648)" — changed
  `resolve_effective_provider_model(app_instance, ...)` to `resolve_effective_provider_model(persisted_defaults: Mapping, ...)`,
  dropped the app-reactive fallback branch (and its `console_control` source label, now
  `console_session`), and deleted the `chat_api_provider_value`/`chat_api_model_value`/`chat_model_value`
  reactive attributes from `TldwCli` entirely. Deliberate: shipped its own comprehensive test file
  (`Tests/Provider/test_provider_model_resolution.py`, including a guard test —
  `test_explicit_api_has_no_application_parameter` — that asserts no `app_instance` parameter exists),
  and every production caller (`chat_screen.py`, `settings_screen.py`) was updated to the new
  signature. `grep -rn "chat_api_provider_value|chat_api_model_value|chat_model_value" tldw_chatbook/`
  returns nothing — no live production bug.
- `1df0c4cb4` "fix: reconcile privacy lifecycle eval and packaging hardening" — removed
  `settings_screen.py`'s import/use of the singular `save_setting_to_cli_config` helper (Console
  Behavior category now saves exclusively through the atomic, batched
  `SettingsConfigAdapter.save_sections()` -> `save_settings_to_cli_config()`), and made
  `application_owned_config_directory()` return `None` whenever `TLDW_CONFIG_PATH` is set (custom
  config parents are fail-closed and never auto-created — "never a custom parent"). Deliberate
  security/atomicity hardening: `grep -n "save_setting_to_cli_config(\|save_settings_to_cli_config(" tldw_chatbook/UI/Screens/settings_screen.py`
  shows only the plural, batched call at both real call sites — no live production bug.

**Per-failure disposition (test-file fix in every case; production was correct):**

1. `test_effective_provider_model_prefers_console_overrides` — stale app-instance call shape; updated to pass `persisted_defaults` mapping directly, `provider_source` expectation updated `console_control` → `console_session`. [d15882398]
2. `test_effective_provider_model_preserves_configured_provider_when_reactive_is_default_openai` — premise (app-reactive OpenAI-default special case) was deliberately removed; rewrote to assert the surviving plain fallback (persisted_defaults wins with no overrides). [d15882398]
3. `test_effective_provider_model_prefers_settings_draft_values` — stale call shape only; passes `persisted_defaults` mapping directly, assertions unchanged. [d15882398]
4. `test_effective_provider_model_ignores_blank_provider_overrides_for_default_fallback` — stale call shape only; passes `persisted_defaults` mapping directly. [d15882398]
5. `test_effective_provider_model_ignores_blank_reactive_provider_for_default_fallback` — app-reactive input no longer exists; repurposed to cover a genuinely new edge case (blank/whitespace *configured* provider values pass through unsanitized as the final fallback, unlike the override paths which do filter blank text). [d15882398]
6. `test_effective_provider_model_ignores_textual_blank_select_provider_for_default_fallback` — stale call shape only; `Select.BLANK` handling unaffected. [d15882398]
7. `test_effective_provider_model_ignores_blank_model_overrides_for_default_fallback` — stale call shape only. [d15882398]
8. `test_effective_provider_model_handles_non_mapping_app_config` — old behavior (tolerate non-mapping, return None) was deliberately replaced by fail-fast `TypeError`; rewrote to assert `pytest.raises(TypeError, match="mapping")`, matching `Tests/Provider/test_provider_model_resolution.py::test_effective_resolver_rejects_non_mapping_defaults`. [d15882398]
9. `test_settings_console_behavior_rejects_invalid_global_defaults` (6 parametrized cases: max-tokens, min-p, streaming, temperature, thinking-budget-tokens, top-p) — monkeypatch retargeted from the removed `settings_screen.save_setting_to_cli_config` to `settings_config_adapter.save_settings_to_cli_config` (the actual batched call Console Behavior now uses), lambda signature updated to the plural function's single-mapping-arg shape. [1df0c4cb4]
10. `test_settings_console_behavior_revert_button_works_with_input_focus` — same retarget as #9. [1df0c4cb4]
11. `test_settings_console_behavior_revert_discards_draft` — same retarget as #9. [1df0c4cb4]
12. `test_settings_console_behavior_revert_restores_global_defaults` — same retarget as #9. [1df0c4cb4]
13. `test_settings_console_behavior_uses_batched_save_adapter` — the `legacy_calls` monkeypatch target (`settings_screen_module.save_setting_to_cli_config`) no longer exists as an attribute to patch; removed the dead monkeypatch and the `legacy_calls` tracking/assertion (a NameError would now fire on any attempt to reintroduce a per-key call from that module — a stronger guarantee than the old runtime mock). `batched_calls` assertions (the test's actual purpose) are untouched. [1df0c4cb4]
14. `test_settings_overview_paste_summary_updates_after_toggle` — monkeypatch retargeted from `settings_screen.save_setting_to_cli_config` to `settings_config_adapter.save_settings_to_cli_config`. [1df0c4cb4]
15. `test_settings_paste_toggle_keeps_keyboard_focus_after_refresh` — same retarget as #14. [1df0c4cb4]
16. `test_settings_provider_category_saves_provider_defaults_without_sampling` — asserted on the removed `app.chat_api_provider_value` reactive before and after save; both assertions rewritten to read `app.app_config["chat_defaults"]["provider"]` (the persisted source of truth this reactive used to mirror). [d15882398]
17. `test_settings_storage_test_shortcut_runs_safety_check` — test-only bug, unrelated to any rename: it set `TLDW_CONFIG_PATH` to `tmp_path/"config"/"config.toml"` but only ever created `tmp_path/"data"`, never `tmp_path/"config"`. Every sibling test in this file uses `tmp_path/"config.toml"` directly (parent = `tmp_path`, always present). Before 1df0c4cb4 a missing parent for the *default* config path was silently recovered by `_load_cli_config_bootstrap_unlocked`'s `missing_parent` handler; that recovery is deliberately gated to `application_owned_config_directory(config_path) is not None`, which returns `None` whenever `TLDW_CONFIG_PATH` is set — custom config parents are never auto-created (fail-closed security hardening). Fixed by adding `config_path.parent.mkdir(parents=True, exist_ok=True)` before setting the env var, matching real deployments where the parent directory always exists first. [1df0c4cb4, test bug predates it by commit 0d314d29f]

**Files modified:**
- `Tests/UI/test_settings_configuration_hub.py` — all 22 fixes (test-only; no production changes were needed).
- `backlog/docs/lessons-testing-evidence.md` — new entry: "A suite that no gate runs can rot invisibly for days", citing this task's evidence for AC#3 (route the hub suite through Console-area verification gates going forward).

**Verification:**
- `Tests/UI/test_settings_configuration_hub.py`: 253 passed, 0 failed (was 231 passed / 22 failed).
- Regression backstop `Tests/UI/test_console_parallel_runs.py` + `Tests/Chat/test_console_run_state_per_session.py`: 46 passed, 0 failed.
<!-- SECTION:NOTES:END -->
