---
id: TASK-42482
title: "A moved-HOME user who chose the Claude subscription can opt in to the Keychain read, and no test can reach that lookup"
status: Done
assignee:
- '@claude'
created_date: 2026-11-23 16:44
updated_date: 2026-11-23 17:11
labels:
- follow-up
- security
- privacy
- tests
priority: high
references:
- tldw_chatbook/LLM_Calls/anthropic_subscription.py
- Tests/real_profile_guard.py
- Tests/conftest.py
- Tests/test_network_guard.py
- Docs/User_Guide/settings.md
dependencies:
- TASK-42478
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Round 23 of the follow-up burn-down (2026-11-23). Owner decision, 2026-11-23: "give a moved-HOME user who explicitly chose the subscription a way to still use the Keychain, without tests ever reaching it".

**Why.** Since TASK-42477 (#3730), `LLM_Calls/anthropic_subscription.py::_keychain_credential_raw` returns `None` unless HOME is the login account's own home (`_home_is_account_home`). That keeps every test sandbox, and every child process it starts, away from the real macOS Keychain service. It also takes the Keychain away from a real user who runs Chatbook with a moved HOME and borrows the Claude subscription, and that user has no way to get it back. Such a user needs an explicit switch. The suite then needs a backstop that holds even if a test turns the switch on, because the HOME check alone would no longer protect it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A documented config setting, off by default, lets the Keychain read happen with a moved HOME, and only while Anthropic's `auth_source` is `claude_subscription`; with it off, or with another auth source, a moved HOME reads no Keychain item
- [x] #2 The User Guide's moved-HOME sentence and the missing-credential message name the setting
- [x] #3 Every pytest process refuses, and fails the test for, any spawn of `security find-generic-password` for the `Claude Code-credentials` service, even when a test turns the setting on; other `security` commands (the native-credential harness's `list-keychains`) still run. The guard is shown red without it and green with it
- [x] #4 Child processes that load no conftest stay covered by the HOME check: the setting has no environment form, so a child never inherits it
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Product: `[api_settings.anthropic] allow_keychain_with_moved_home`, read through `load_settings()` inside `_keychain_credential_raw`'s HOME check, and honoured only with `auth_source = "claude_subscription"`. Mention it in `MISSING_CREDENTIAL_MESSAGE` and `Docs/User_Guide/settings.md`. No UI switch: the Sign in with row would need a new row, a draft key, save logic and a new Tab stop inside its five-press budget.
2. Backstop: extend `Tests/real_profile_guard.py`'s audit hook, which `Tests/conftest.py` installs in every pytest process and whose refusals already fail the test (autouse fixture) and the run (session end, xdist carried). Match only that lookup.
3. Tests in `Tests/test_network_guard.py` (Architecture guard contracts lane): the opt-in matrix with a fake `run`; the matcher's scope; a real-shaped spawn of a stand-in `security` script, red without the hook and green with it.
4. Runs, ratchets, preflight, ruff; notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Setting.** `[api_settings.anthropic] allow_keychain_with_moved_home = true` (`KEYCHAIN_WITH_MOVED_HOME_SETTING`), off by default. `_keychain_credential_raw` now runs `security` only when HOME is the account's own home, or when `_moved_home_keychain_allowed()` is true. That function reads `load_settings()["api_settings"]["anthropic"]`, the table the send path reads `auth_source` from, and is true only when the value is the boolean `true` and `auth_source` is `claude_subscription`. A config that cannot be read counts as off. The check runs before the 5 s memo, so turning the setting off takes effect at once, and an account-home user never pays the config read. A scratch-profile probe through the real loader showed `load_settings` and `get_cli_setting` both return the key from `config.toml`.

**No UI switch.** The Sign in with row is the obvious surface, but a switch there needs a new row, a draft key, save handling, a Source word, guidance copy and a Tab stop, and Model must stay within five presses of Provider. That is not trivial, so the setting is config-only. Documented in `Docs/User_Guide/settings.md`, beside the moved-HOME sentence. `MISSING_CREDENTIAL_MESSAGE` names it in its moved-HOME clause.

**Suite backstop.** `Tests/real_profile_guard.py`'s audit hook (which `Tests/conftest.py` installs in every pytest process, xdist workers and private-profile children included) now also matches the spawn events `subprocess.Popen`, `os.posix_spawn`, `os.exec`, `os.spawn` and `os.system`. Any command line matching `security find-generic-password … Claude Code-credentials` is refused with `RealKeychainLookupError` and recorded. That reuses the guard's whole pipeline: the autouse `refuse_real_profile_writes` fixture fails the test even when the product swallows the refusal (it catches `OSError`), the session-end check fails the run, and xdist workers carry their refusals to the controller. Only that lookup is matched, so `security list-keychains`, as the GitHub-hosted native-credential harness runs it, and other services still run. The fixture's and the session-end messages now name the Keychain too.

**Children (AC#4).** A child that loads no conftest gets no hook. It stays covered by the HOME check, which this task keeps: under a moved HOME the child reads the Keychain only if its own config sets the opt-in. The setting has no environment form, so nothing passes it to a child. No test or harness writes it to a config file; grep finds it only in the product, its tests (in-process `load_settings` fakes) and the docs. TASK-42477's two guards still pass: the in-process sandbox read and the plain Python child.

**Tests** (`Tests/test_network_guard.py`, in the "Run Architecture guard contracts" PR step):
- `test_a_moved_home_reads_the_keychain_only_when_the_subscription_user_opted_in[opted-in|api-key|not-opted-in]`: darwin forced, HOME moved, `load_settings` faked, a fake `run` set first. Red before the change: `[opted-in]` failed with `assert [] == [['/usr/bin/security', 'find-generic-password', '-s', 'Claude Code-credentials', '-w']]`. Green after. Negative controls, temporary source edits that were reverted: dropping the `auth_source` condition turns `[api-key]` red, and dropping the flag check turns `[not-opted-in]` red.
- `test_the_suite_guard_matches_only_the_claude_code_keychain_lookup[...]`: the matcher on argv, `sh -c`, `posix_spawn` and `os.system` shapes, and not on `list-keychains`, another service, or a non-`security` command. With the pattern widened to any `security` command, `[list-keychains]` and `[other-service]` go red.
- `test_the_suite_refuses_the_claude_code_keychain_lookup[argv|shell]`: a real-shaped spawn of a stand-in `security` script in `tmp_path`; `/usr/bin/security` is never named. With the hook's Keychain branch removed, both cases fail with `DID NOT RAISE <class 'Tests.real_profile_guard.RealKeychainLookupError'>` and the stand-in runs. With it, both pass.
- `test_a_test_that_opts_in_without_faking_run_is_refused_and_recorded`: a test that turns the opt-in on, with `run` only redirected to the stand-in, is refused at the product's own spawn and recorded. Without the branch it fails with `assert not True` (the stand-in ran). A temporary test file (deleted) that did not take the refusal failed at teardown with "Wrote to the real tldw profile or spawned the real Keychain lookup … subprocess.Popen of the Keychain lookup '…/security find-generic-password -s Claude Code-credentials -w'".

The last three must run without a refusing probe on PYTHONPATH: the probe's hook is added first, refuses any `security` spawn itself, and the suite hook never sees it (TASK-42477's lesson). They spawn only the stand-in.

**Runs** (scratch wrapper; refusing probe on unless noted; load 8-11; 0 `security` spawns in every run):
- `Tests/test_network_guard.py`, `Tests/test_real_profile_guard.py`, `Tests/LLM_Calls`, `Tests/test_keyring_isolation.py` and `Tests/Chat/test_subscription_background_readiness.py` at `-n 4`, with the three no-probe tests deselected: 2395 passed, 50 skipped, 8 failed. All 8 fail at base 82055f4750 with the same command: TASK-42477's 7, and `test_a_constructed_app_records_the_store_as_unavailable` (`RecoveryRequired: raw_source_selection_changed`).
- The three no-probe tests, probe off: 3 passed.
- `Tests/Backup_Recovery/test_native_credential_runner.py`: 145 passed. `Tests/ProductionApp/test_native_credential_recovery.py`: 2 skipped (they run only in the macOS native-credential workflow).
- Console/Settings/Personas sample at `-n 4` (10 files: the three subscription readiness files, `test_settings_anthropic_auth_source.py`, both poll-gate files, `test_console_rail_reconciliation.py`, `test_console_session_tab_close.py`, `test_console_model_switcher.py`, `test_personas_character_generation_ui.py`): 239 passed.
- `Tests/CI` at `-n 4`: 1823 passed, 2 skipped.
- `Tests/Architecture` serially: 1492 passed, 5 skipped.
- Preflight passes. `ruff check tldw_chatbook --select F` is clean, `Tests/lint/ruff_gate.py` exits 0, and the touched test files are F-clean. No size ratchet lists the touched files, and no census or lane changed.

**ID check.** 42482 (and 42483) were unused after a fetch: 1099 refs, no ref name, no object name in any history, no file in any worktree's `backlog/`, and none in the 98,860 distinct blobs at every ref tip (`$FU/sp23-idcheck.sh`, adapted from `kc22-idcheck2.sh`).

**Files.**
- Product: `tldw_chatbook/LLM_Calls/anthropic_subscription.py`.
- Tests: `Tests/real_profile_guard.py`, `Tests/conftest.py`, `Tests/test_network_guard.py`.
- Docs: `Docs/User_Guide/settings.md`.

No lesson: the probe-pre-emption trap is TASK-42477's.
<!-- SECTION:NOTES:END -->
