---
id: TASK-34201
title: Settings control to sign in to Anthropic with the Claude subscription
status: Done
assignee:
  - '@claude'
created_date: '2026-10-03 21:00'
updated_date: '2026-10-03 23:31'
labels:
  - providers
  - settings
  - auth
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-26022 lets Anthropic requests use the Claude subscription credential that Claude Code already holds, but the only way to turn it on is hand-editing `auth_source = "claude_subscription"` into `[api_settings.anthropic]` in config.toml. Settings > Providers > Anthropic should offer the choice directly, using the owner-approved design (2026-10-03): a "Sign in with" select (API key | Claude subscription) at the top of the Credentials section, following the existing Qwen Cloud "API mode" precedent; choosing the subscription disables, but keeps visible, the API key and Env var rows; a guidance line instead of a confirmation dialog.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Settings > Providers > Anthropic shows a "Sign in with" select offering API key and Claude subscription; no other provider shows it
- [x] #2 The select reflects the saved auth_source, and saving writes it to [api_settings.anthropic], so the next send uses that credential source
- [x] #3 With Claude subscription chosen, the API key and Env var rows are disabled but visible, and switching back restores them without losing a saved key
- [x] #4 A guidance line states that the credential is borrowed from Claude Code (read, never stored or refreshed) and bills the Claude plan
- [x] #5 The change is an ordinary unsaved edit: revert discards it, and switching providers keeps it until saved or reverted
- [x] #6 Readiness and the credential status row show the subscription state (ready, or refresh in Claude Code)
- [x] #7 With the select left at API key, saved configs and behavior are exactly as today
- [x] #8 The User Guide's Settings page describes the control
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Persistence: ProviderSetupDraft carries an optional Anthropic auth_source (api_key | claude_subscription); the setup mutation writes it into [api_settings.anthropic] and validates it; the existing preserve_credentials subscription path stays consistent.
2. Settings: mirror the Qwen Cloud api_mode wiring (provider-scoped draft key, staging, snapshot/restore across provider switches, display sync, dirty label, save path, compose, change handler) for auth_source, Anthropic only; disable the API key/env-var/clear controls while the subscription is selected.
3. Readiness/status row refresh after the change (the subscription readiness poller already exists).
4. Tests: persistence unit tests (local); a Settings UI test file mirroring test_settings_qwencloud_api_mode.py, added to scripts/ui_pr_gate_census.txt so CI's UI Fast Lane runs it (mounted UI tests fail locally with RecoveryRequired on this machine).
5. User Guide: Docs/User_Guide/settings.md, Anthropic credentials.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Adds a "Sign in with" select (API key | Claude subscription) at the top of Settings > Providers > Anthropic > Credentials, Anthropic only. Choosing the subscription disables, but keeps visible, the API key, Env var and Clear controls, and shows a guidance line; no confirm (owner's design, 2026-10-03).

Approach: mirrors the Qwen Cloud api_mode wiring in UI/Screens/settings_screen.py (provider-scoped draft key provider_auth_source:anthropic, staging, snapshot across provider switches and before save, display sync after the registry lock, save via section_values). The allowed values come from LLM_Calls/anthropic_subscription.py (imported lazily, so the UI-ready census does not grow). Readiness, the credential status row, the key placeholder and the poller read the draft choice through one helper, _provider_auth_readiness_config, so the status row follows an unsaved choice and a saved subscription does not hide a stored key after switching back (both Qodo findings).

Deviation from plan step 1: no ProviderSetupDraft field. The choice is written as an auth_source-only combined mutation, and Chat/provider_setup_persistence.py's combined validator now admits auth_source for Anthropic only, limited to the two known values (it previously raised ValueError, which Settings reported as "the file was not written").

Tests: Tests/UI/test_settings_anthropic_auth_source.py (9 mounted + 1 pure; in scripts/ui_pr_gate_census.txt; marked bootstrap_profile because the per-test sandbox refuses Settings' config-participant admission, as in test_console_fork_fresh_lineage_flow.py) and 5 boundary tests in Tests/Chat/test_provider_setup_persistence.py. Saves are captured at the atomic writer, never a real config. The 46 local failures in test_provider_setup_persistence.py + test_settings_qwencloud_api_mode.py (RecoveryRequired raw_source_selection_changed) reproduce identically on a clean origin/dev worktree.

Not done: a live TUI run of the select itself. The end-to-end subscription send was live-verified under TASK-26022 with auth_source set in a scratch config.

Docs: Docs/User_Guide/settings.md Credentials row.
<!-- SECTION:NOTES:END -->
