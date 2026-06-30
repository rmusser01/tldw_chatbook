---
id: TASK-145
title: Restore provider credential onboarding and polish Console setup UX
status: Done
assignee: []
created_date: '2026-06-30 02:42'
updated_date: '2026-06-30 03:32'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make provider API-key setup discoverable and usable from the app, then polish the Console blocked-setup surface so first-time and regular users can recover without hidden command-palette knowledge or obscured composer controls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 User can add and clear a local provider API key from Settings > Providers & Models without editing TOML by hand.
- [x] #2 Env-var credential setup remains visible as the safer/power-user path.
- [x] #3 Console missing-key recovery opens Settings on the exact provider credential controls.
- [x] #4 Provider secrets are masked or redacted in all visible UI and tests.
- [x] #5 Console blocked-state UI does not obscure composer input or rely on same-color separators.
- [x] #6 Console has no blank structural bars that contain no actionable content.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/012-provider-credential-settings-boundary.md
Reason: The task exposes direct in-app mutation of local provider API keys, which is a credential and privacy boundary even though config fallback support already exists.

1. Follow the approved design spec in Docs/superpowers/specs/2026-06-30-provider-credentials-console-setup-polish-design.md.
2. Write failing Settings tests for visible API-key controls, masked status, save, and clear behavior.
3. Write failing Console tests for missing-key recovery context and composer/readability polish.
4. Implement Settings provider credential controls using the existing SettingsConfigAdapter and provider readiness helpers.
5. Implement Console recovery routing and blocked-state/composer layout polish.
6. Run focused tests, then broader Settings and Console regression tests.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a provider credential block in Settings > Providers & Models with a masked local API-key input, explicit clear action, env-var-name input, source status copy, and redacted save/test behavior. Console missing-key and endpoint recovery now include a field target so Settings focuses the API-key or endpoint control instead of only opening the category. Console blocked-state copy was moved out of the composer row; compatibility widgets remain mounted but hidden, and stronger semantic separator tokens are used for the transcript/composer structure.

Added Textual-specific regression coverage for the provider settings scroll container using the real app stylesheet, plus visual proof captures for the top, credential, and provider-action scroll positions. Rebuilt the modular TCSS bundle after component CSS changes.

Addressed PR review feedback by redacting sensitive values in the central single-setting config save log, moving provider API-key validation into the shared input validation module, exposing provider readiness placeholder-key validation as a public helper, rejecting placeholder keys before Settings saves them, adding docstrings for the new provider API-key handlers, and simplifying hidden Console composer compatibility widgets so state sync no longer recalculates their hidden styles.

Verification:

- `PATH=.venv/bin:$PATH pytest Tests/UI/test_settings_configuration_hub.py::test_settings_long_detail_and_inspector_panes_are_scrollable_containers -q`
- `PATH=.venv/bin:$PATH pytest Tests/UI/test_settings_configuration_hub.py::test_settings_ownership_records_cover_categories_and_runtime_boundaries Tests/UI/test_settings_configuration_hub.py::test_settings_provider_test_redacts_secrets Tests/UI/test_settings_configuration_hub.py::test_settings_provider_test_blocks_unknown_provider Tests/UI/test_settings_configuration_hub.py::test_settings_provider_test_uses_api_settings_env_var_without_secret_leak Tests/UI/test_settings_configuration_hub.py::test_settings_provider_model_discovery_saves_selected_runtime_models Tests/UI/test_settings_configuration_hub.py::test_settings_provider_model_discovery_shows_ambiguous_provider_recovery Tests/UI/test_settings_configuration_hub.py::test_settings_provider_test_does_not_depend_on_console_sampling_defaults Tests/UI/test_console_native_chat_flow.py::test_console_native_enter_on_setup_blocked_send_shows_recovery_feedback Tests/UI/test_console_native_chat_flow.py::test_console_setup_blocked_send_adds_durable_transcript_recovery_feedback Tests/UI/test_console_internals_decomposition.py::test_console_enter_sends_native_composer_draft -q`
- `PATH=.venv/bin:$PATH pytest Tests/UI/test_settings_configuration_hub.py Tests/UI/test_settings_privacy_security.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_session_settings.py Tests/UI/test_console_workbench_contract.py Tests/UI/test_console_internals_decomposition.py Tests/UI/test_non_obscuring_focus_contract.py Tests/UI/test_workbench_visual_snapshots.py -q`
- `PATH=.venv/bin:$PATH pytest Tests/Chat/test_provider_readiness.py Tests/test_config_console_defaults.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_settings_privacy_security.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_session_settings.py Tests/UI/test_console_workbench_contract.py Tests/UI/test_console_internals_decomposition.py Tests/UI/test_non_obscuring_focus_contract.py Tests/UI/test_workbench_visual_snapshots.py -q`
- `PATH=.venv/bin:$PATH python3 tldw_chatbook/css/build_css.py`
- `git diff --check`

ADR check: ADR required and completed in `backlog/decisions/012-provider-credential-settings-boundary.md`.
<!-- SECTION:NOTES:END -->
