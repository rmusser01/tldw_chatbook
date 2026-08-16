---
id: TASK-16475
title: Console surface stale-default provider swaps to the user
status: Done
assignee:
  - '@Robert'
created_date: '2026-08-15 15:10'
labels: []
dependencies: []
---
## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User report (2026-08-15): the Provider chip flips to a provider the user never chose. `_maybe_refresh_stale_default_console_settings` (`UI/Console_Modules/session.py`, from task-177) replaces an untouched blocked session's settings with re-derived defaults whenever the fresh defaults are send-capable — including when they name a DIFFERENT provider than the session had. The swap itself is task-177's intended convergence (a Settings fix must reach never-used blocked sessions without a restart) and is protected by `test_console_stale_default_refresh_respects_user_marked_settings`; what is missing is any signal. From the user's seat the Provider chip silently changes identity, which reads as "random". This task makes the swap observable; it must NOT block or narrow the task-177 convergence (provider-equality gating is explicitly out of scope unless separately decided).

ADR required: no — observability addition on an existing behavior; the refresh policy itself is unchanged (task-177 lineage).
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan (the how)

1. In `_maybe_refresh_stale_default_console_settings`, when the replacing defaults change the provider key, emit a one-time warning notify (previous -> new provider, pointing at Settings/chat_defaults) plus a log record with both keys
2. Red test exists (`test_stale_default_refresh_swap_is_visible`); equality gates already prevent repeat notices
3. Keep task-177 convergence untouched (same-provider refresh stays silent; user-marked sessions untouched)
4. Run the task-177 tests + session settings suite

ADR required: no
ADR path: N/A
Reason: observability only; refresh policy unchanged (task-177 lineage)

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 When the stale-default refresh replaces a session's settings with defaults whose provider (`provider_config_key`) differs from the session's previous provider, the Console surfaces a one-time notice naming the change (previous provider -> new provider) and pointing at Settings/`chat_defaults` as the source of the new default
- [x] #2 Same-provider refreshes (credential/endpoint/model converging after a Settings fix) produce no notice; user-marked sessions are never refreshed (existing behavior, stays covered)
- [x] #3 The notice fires once per actual replacement (the ensure path runs on many sync ticks; equality gates must keep repeat notices away)
- [x] #4 Regression test `test_stale_default_refresh_swap_is_visible` (added red in `Tests/UI/test_console_provider_persistence_regressions.py`) passes; the task-177 tests in `Tests/UI/test_console_session_settings.py` stay green
- [x] #5 The notice is a warning-severity notification (or transcript-level notice) and is also logged with both provider keys for diagnostics
<!-- AC:END -->
## Implementation Notes

- `_maybe_refresh_stale_default_console_settings` (`UI/Console_Modules/session.py`) now detects when the replacing defaults change the provider key and calls `_notify_stale_default_provider_swap`: one warning-severity notify ("Console provider changed X -> Y: this unused session now follows your saved defaults (Settings > Providers & Models).") plus an info log with both keys. Best-effort -- notification failure never breaks the ensure path.
- task-177 convergence untouched: same-provider refresh stays silent, user-marked sessions never refreshed (both existing tests still green).
- Modified files: `UI/Console_Modules/session.py`, `Tests/UI/test_console_provider_persistence_regressions.py` (regression test with a sanity assert that the swap itself still happens).
