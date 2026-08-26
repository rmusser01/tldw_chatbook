---
id: TASK-16310
title: >-
  Fix credential/URL persistence: silent keyring gap, false-success config
  saves, and encryption fallbacks
status: Done
assignee:
  - '@Robert'
created_date: '2026-08-14 20:06'
updated_date: '2026-08-14 21:35'
labels:
  - bug
  - config
  - security
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Users report that credentials/URLs entered in the UI are not saved across app restarts and never reach the OS secret store. Investigation found four related defects in the save/load path for server and provider credentials.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `save_settings_to_cli_config` no longer returns `True` for refused/no-op mutations; an identity-conflict result is reported as a failure to the caller (regression test included)
- [x] #2 Server auth token is written to the OS credential store eagerly at save time (not only via the lazy legacy import), and keyring write failures are surfaced (logged with context / user-visible) instead of silently swallowed
- [x] #3 Unavailable/insecure keyring backends are reported to the user at startup or save time instead of silently substituting `UnavailableServerCredentialStore`
- [x] #4 With config encryption enabled and no password available, save fails with a clear user-facing error and never silently downgrades a secret to plaintext (silent plaintext fallback in `_maybe_encrypt_setting_value` removed or made explicit)
- [x] #5 Stale `Widgets/Media_Ingest/__pycache__` `.pyc` files for the deleted `IngestTldwApi*` windows are removed and prevented from shipping
- [x] #6 Sign-out/re-login flow: a re-entered token saved via the server-switch modal resolves after restart (regression test)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See docs/superpowers/plans/2026-08-14-credential-persistence-fixes.md - five tasks: (1) fix false-success conflict return in save_settings_to_cli_config, (2) hard-fail encryption errors instead of plaintext fallback, (3) add store_static_server_credential eager keyring write on RuntimeServerContextProvider, (4) wire eager write into _perform_runtime_source_switch + surface unavailable keyring, (5) remove stale Media_Ingest pycache dir + task closeout. ADR: not required (bug fixes within existing designed boundaries).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented on fix/credential-persistence: conflict-aware config save semantics, encryption hard-fail, eager keyring write (store_static_server_credential) wired into server-switch save with surfaced failures, stale Media_Ingest pycache removed. 111 tests green across affected suites.
<!-- SECTION:NOTES:END -->

## Implementation Plan (research notes from investigation)

Defects and their locations:

1. **False-success config saves** — `config.py:5470-5476` (`save_settings_to_cli_config`): a mutation refused due to identity conflict (`config.py:5410-5416`, `failure_phase=None`, `file_replaced=False`, `conflict=True`) returns `True`, so callers like `_perform_runtime_source_switch` (`UI/Screens/settings_screen.py:16347`) report success while nothing was written. Fix the success check (or return `fully_applied` / surface `conflict`).
2. **Keyring never written at save time** — token is only written plaintext to `[tldw_api].auth_token` in config.toml. The keyring is populated lazily by `_import_legacy_token` (`runtime_policy/server_context.py:741-753`), which swallows all exceptions. Write to the credential store eagerly in `_perform_runtime_source_switch` and log/surface failures.
3. **Silent keyring-backend substitution** — `build_default_server_credential_store` (`runtime_policy/server_credentials.py:292-312`) rejects non-SecretService/KWallet Linux backends; `app.py:5818-5822` silently substitutes `UnavailableServerCredentialStore`. Notify the user.
4. **Encryption edge cases** — `config.py:5043-5047` raises with `[encryption] enabled` and no password (save fails); `_maybe_encrypt_setting_value` (`config.py:4554-4559`) swallows encryption errors and stores secrets plaintext. Make failures explicit.
5. **Sign-out hides re-entered token** — `_mark_legacy_server_id_cleared` (`runtime_policy/server_context.py:723`) permanently ignores the config token for that server ID; verify a re-saved token via the modal resolves after restart.
6. **Stale artifacts** — `Widgets/Media_Ingest/__pycache__/` still contains `.pyc` for the deleted `IngestTldwApi*` windows (removed in commit `0d45bf802`); the last source version never persisted URL/token at all. Delete and ensure packaging excludes stale `.pyc`.

Note: the LLM provider key path (`Chat/provider_setup_persistence.py`) was audited and found correct; no changes needed there.
