---
id: TASK-34201
title: Settings control to sign in to Anthropic with the Claude subscription
status: In Progress
assignee:
  - '@claude'
created_date: '2026-10-03 21:00'
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
- [ ] #1 Settings > Providers > Anthropic shows a "Sign in with" select offering API key and Claude subscription; no other provider shows it
- [ ] #2 The select reflects the saved auth_source, and saving writes it to [api_settings.anthropic], so the next send uses that credential source
- [ ] #3 With Claude subscription chosen, the API key and Env var rows are disabled but visible, and switching back restores them without losing a saved key
- [ ] #4 A guidance line states that the credential is borrowed from Claude Code (read, never stored or refreshed) and bills the Claude plan
- [ ] #5 The change is an ordinary unsaved edit: revert discards it, and switching providers keeps it until saved or reverted
- [ ] #6 Readiness and the credential status row show the subscription state (ready, or refresh in Claude Code)
- [ ] #7 With the select left at API key, saved configs and behavior are exactly as today
- [ ] #8 The User Guide's Settings page describes the control
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Persistence: ProviderSetupDraft carries an optional Anthropic auth_source (api_key | claude_subscription); the setup mutation writes it into [api_settings.anthropic] and validates it; the existing preserve_credentials subscription path stays consistent.
2. Settings: mirror the Qwen Cloud api_mode wiring (provider-scoped draft key, staging, snapshot/restore across provider switches, display sync, dirty label, save path, compose, change handler) for auth_source, Anthropic only; disable the API key/env-var/clear controls while the subscription is selected.
3. Readiness/status row refresh after the change (the subscription readiness poller already exists).
4. Tests: persistence unit tests (local); a Settings UI test file mirroring test_settings_qwencloud_api_mode.py, added to scripts/ui_pr_gate_census.txt so CI's UI Fast Lane runs it (mounted UI tests fail locally with RecoveryRequired on this machine).
5. User Guide: Docs/User_Guide/settings.md, Anthropic credentials.
<!-- SECTION:PLAN:END -->
