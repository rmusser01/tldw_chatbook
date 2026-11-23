---
id: TASK-42482
title: "A moved-HOME user who chose the Claude subscription can opt in to the Keychain read, and no test can reach that lookup"
status: Done
assignee:
- '@claude'
created_date: 2026-11-23 16:44
updated_date: 2026-11-23 17:54
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
- [x] #1 A documented config setting, off by default, lets the Keychain read happen with a moved HOME, and only while Anthropic's saved `auth_source` is `claude_subscription`; with it off, or with another auth source, a moved HOME reads no Keychain item
- [x] #2 The User Guide's moved-HOME sentence and the missing-credential message name the setting
- [x] #3 Every pytest process refuses, and fails the test for, a spawn of `security find-generic-password` for the `Claude Code-credentials` service in the shapes its audit hook sees (subprocess, including `shell=True`, `os.popen` and asyncio; `os.posix_spawn` and `posix_spawnp`; `os.exec*`; `os.system`; with global options, quoting, an escaped space, any capitalisation, or a renamed or `executable=` binary), even when a test turns the setting on; `os.spawn*` on POSIX is refused in its forked child, so nothing runs, but is recorded only there. Other `security` commands (the native-credential harness's `list-keychains`) still run. The guard is shown red without it and green with it
- [x] #4 Child processes that load no conftest have no guard; they inherit HOME, `XDG_*` and `TLDW_CONFIG_PATH` and so read the same config file as their parent. They stay off the Keychain through the moved HOME and because no sandbox or test config carries the setting, which a test pins (no file under `Tests/` but that test's own names the setting, and the shipped default config does not set it)
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

**Saved, not draft.** The opt-in checks the SAVED `auth_source`. The Providers card reads the credential as soon as the draft Sign in with is Claude subscription, so an opted-in user with a moved HOME who switches the draft sees "No Claude subscription credential found" until they save. The User Guide says so.

**Which table.** `_moved_home_keychain_allowed` reads `load_settings()["api_settings"]["anthropic"]` by its exact key. Readiness resolves that table with `provider_settings_for_key`, and the send path falls back to the legacy `anthropic_api` mapping. Wherever these differ (an alias-named table, or a legacy-only `auth_source`), the opt-in reads as off, so every divergence fails closed.

**No UI switch.** The Sign in with row is the obvious surface, but a switch there needs a new row, a draft key, save handling, a Source word, guidance copy and a Tab stop, and Model must stay within five presses of Provider. That is not trivial, so the setting is config-only. Documented in `Docs/User_Guide/settings.md`, beside the moved-HOME sentence. `MISSING_CREDENTIAL_MESSAGE` names it in its moved-HOME clause.

**Suite backstop.** `Tests/real_profile_guard.py`'s audit hook (which `Tests/conftest.py` installs in every pytest process, xdist workers and private-profile children included) also matches the spawn events `subprocess.Popen` (so `shell=True`, `os.popen` and asyncio subprocesses too), `os.posix_spawn` (and `posix_spawnp`), `os.exec*` and `os.system`. It joins the program and argv into one command line and refuses, with `RealKeychainLookupError`, any that contains both `find-generic-password` and `Code-credentials`, compared case-insensitively. The refusal is recorded too. That reuses the guard's whole pipeline: the autouse `refuse_real_profile_writes` fixture fails the test even when the product swallows the refusal (it catches `OSError`), the session-end check fails the run, and xdist workers carry their refusals to the controller. `security list-keychains`, as the GitHub-hosted native-credential harness runs it, and other services still run. The fixture's and the session-end messages now name the Keychain too. Known limit: `os.spawn*` on POSIX forks first and calls `exec` in the child. The child refuses it, so nothing runs (the reviewer's probe: `spawnv` returned 127 and the stand-in did not run), but the refusal is recorded only in the child, so the parent's test does not fail.

**Children (AC#4).** A child that loads no conftest gets no hook. It inherits HOME, `XDG_*` and `TLDW_CONFIG_PATH`, so it reads the same config file as its parent. If that file set the opt-in, a conftest-less child, such as a `Tests/Performance` plain child that mounts a Console, would pass both gates and spawn the real `security` with nothing in the way. What keeps children off the Keychain is the moved HOME, together with no sandbox or test config that carries the key. `test_no_test_or_default_config_turns_on_the_keychain_opt_in` pins this: no file under `Tests/` but its own names the key, and the parsed default config (`DEFAULT_CONFIG_FROM_TOML`, the baseline every load merges onto) does not set it. TASK-42477's two guards still pass: the in-process sandbox read and the plain Python child. (The first version of this paragraph claimed that a child never inherits the setting. That was wrong, and review found it.)

**Tests** (`Tests/test_network_guard.py`, in the "Run Architecture guard contracts" PR step):
- `test_a_moved_home_reads_the_keychain_only_when_the_subscription_user_opted_in[opted-in|api-key|not-opted-in]`: darwin forced, HOME moved, `load_settings` faked, a fake `run` set first. Red before the change: `[opted-in]` failed with `assert [] == [['/usr/bin/security', 'find-generic-password', '-s', 'Claude Code-credentials', '-w']]`. Green after. Negative controls, temporary source edits that were reverted: dropping the `auth_source` condition turns `[api-key]` red, and dropping the flag check turns `[not-opted-in]` red.
- `test_the_suite_guard_matches_only_the_claude_code_keychain_lookup[...]`: the matcher on argv, `sh -c`, `posix_spawn` and `os.system` shapes, and not on `list-keychains`, another service, or a non-`security` command. With the pattern widened to any `security` command, `[list-keychains]` and `[other-service]` go red.
- `test_the_suite_refuses_the_claude_code_keychain_lookup[argv|shell]`: a real-shaped spawn of a stand-in `security` script in `tmp_path`; `/usr/bin/security` is never named. With the hook's Keychain branch removed, both cases fail with `DID NOT RAISE <class 'Tests.real_profile_guard.RealKeychainLookupError'>` and the stand-in runs. With it, both pass.
- `test_a_test_that_opts_in_without_faking_run_is_refused_and_recorded`: a test that turns the opt-in on, with `run` only redirected to the stand-in, is refused at the product's own spawn and recorded. Without the branch it fails with `assert not True` (the stand-in ran). A temporary test file (deleted) that did not take the refusal failed at teardown with "Wrote to the real tldw profile or spawned the real Keychain lookup … subprocess.Popen of the Keychain lookup '…/security find-generic-password -s Claude Code-credentials -w'".

The last three must run without a refusing probe on PYTHONPATH: the probe's hook is added first, refuses any `security` spawn itself, and the suite hook never sees it (TASK-42477's lesson). They spawn only the stand-in.

**Runs** (scratch wrapper; refusing probe on unless noted; load 8-11; 0 `security` spawns in every run):
- `Tests/test_network_guard.py`, `Tests/test_real_profile_guard.py`, `Tests/LLM_Calls`, `Tests/test_keyring_isolation.py` and `Tests/Chat/test_subscription_background_readiness.py` at `-n 4`, with the three no-probe tests deselected: 2395 passed, 50 skipped, 8 failed. All 8 fail at base 82055f4750 with the same command:
  - the local `RecoveryRequired: raw_source_selection_changed` trap: `test_chat_with_anthropic_refuses_cross_origin_redirect`, the two `test_realtime_protocol.py` connect tests, and `test_keyring_isolation.py::test_a_constructed_app_records_the_store_as_unavailable`;
  - the preset set: `test_only_fireworks_restricts_reasoning_effort_levels` (`{'deepseek', 'fireworks'} == {'fireworks'}`);
  - the summarization manifest inventory: the three `test_summarization_diagnostic_privacy.py` manifest-boundary tests ("checked inventory changed outside the live summarization owners").
- The three no-probe tests, probe off: 3 passed.
- `Tests/Backup_Recovery/test_native_credential_runner.py`: 145 passed. `Tests/ProductionApp/test_native_credential_recovery.py`: 2 skipped (they run only in the macOS native-credential workflow).
- Console/Settings/Personas sample at `-n 4` (10 files: the three subscription readiness files, `test_settings_anthropic_auth_source.py`, both poll-gate files, `test_console_rail_reconciliation.py`, `test_console_session_tab_close.py`, `test_console_model_switcher.py`, `test_personas_character_generation_ui.py`): 239 passed.
- `Tests/CI` at `-n 4`: 1823 passed, 2 skipped.
- `Tests/Architecture` serially: 1492 passed, 5 skipped.
- Preflight passes. `ruff check tldw_chatbook --select F` is clean, `Tests/lint/ruff_gate.py` exits 0, and the touched test files are F-clean. No size ratchet lists the touched files, and no census or lane changed.

**Review round 1 (2026-11-23, "with fixes").** The review confirmed the opt-in logic, its lock safety and the strict `is True` check. It found the following, all fixed in new commits:
1. **Children.** AC#4's claim that a child never inherits the setting was false (see "Children" above). AC#4, the `real_profile_guard.py` docstring and the `KEYCHAIN_WITH_MOVED_HOME_SETTING` comment now say what does protect children, and `test_no_test_or_default_config_turns_on_the_keychain_opt_in` pins it.
   - Red before: the guard's own docstring named the key.
   - With a planted `Tests/fixtures/*.toml` setting the key: `assert ['Tests/fixtu...planted.toml'] == []`.
   - With the default template's line uncommented to `true`: `assert 'allow_keychain_with_moved_home' not in {...}`.
   - The pin also caught my own first re-pin comment in `test_config_load_settings_table_guard.py`, which named the key; it is reworded.
2. **Matcher gaps.** These shapes went unmatched while the stand-in ran: global options (`-q`, `-v`), `executable=`, a quoted command word, a quoted path with a space, an escaped space, and `Security`. The matcher now joins the program and argv and matches `find-generic-password` and `Code-credentials` case-insensitively; it no longer requires `security` next to the subcommand.
   - The matcher test gained those seven shapes plus `exec` and `spawn`. All seven were red before (`assert (None is not None) is True`) and pass after.
   - The spawn test gained `executable-override` and `global-option`. Both were red before (`DID NOT RAISE`, the stand-in ran) and pass after.
   - The reviewer's `bypass_probe.py` refuses all 15 matcher shapes and all 6 real spawns.
   - `os.spawn*` on POSIX is the stated limit above; AC#3 and the docstrings now list the shapes covered instead of "any spawn".
3. **Draft vs saved.** See "Saved, not draft" above; the User Guide now says "the saved Sign in with".
4. **Discoverability.** `default_config.toml` has a commented `# allow_keychain_with_moved_home = false` line with a one-line explanation under the `auth_source` comment. `test_default_config_template_bytes_are_unchanged` is re-pinned on purpose.
5. **Pins.** The matrix gained `[string-true]` and `[unreadable-config]`, both 0 reads. Each turns red under a negative control: truthiness instead of `is True` for the first, and the `except` narrowed to an unrelated type for the second.
   - The coordinator's decision: a non-boolean value logs ONE loguru warning per value, naming only the key and the value's type, never the value or anything credential-related. `test_an_opt_in_that_is_not_a_boolean_logs_one_warning_per_value` was red before (`assert 0 == 2`).
   - Deviation: an explicit `false` stays silent. It is a valid off, and the default config now suggests it, so a warning would be noise.
   - The warning is the module's first diagnostic, so `Docs/security/production-diagnostic-inventory.json` gains its row (owner_files 749 -> 750). Before running `--write` I reviewed the added statement with `--statements`: it interpolates only the key's name (a constant) and `type(value).__name__`, with no user content, secret, path or URL.
6. and 7. The "Runs" failure list now names its causes, and "Which table" covers the lookup divergence.

After the fixes (probe on unless noted; 0 `security` spawns):
- The no-probe guard tests, probe off: 5 passed.
- The covering set at `-n 4` (`Tests/test_network_guard.py`, `Tests/test_real_profile_guard.py`, `Tests/LLM_Calls`, keyring isolation, the background-readiness file, the native-credential runner, and five `Tests/test_config_*` and registry files that read the default template): 2740 passed, 50 skipped, 9 failed. Eight are the pre-existing failures named under "Runs". The ninth was the template byte pin, now re-pinned; that file plus the new pin then gave 13 passed.
- UI sample (the subscription readiness files, `test_settings_anthropic_auth_source.py`, both poll-gate files, Home's readiness file): 46 passed.
- `Tests/CI`: 1823 passed, 2 skipped.
- `Tests/Architecture` serially: 1492 passed, 5 skipped.
- Preflight passes, after the inventory `--write` above; `Tests/test_persistent_diagnostic_boundary.py` passes.
- `ruff check tldw_chatbook --select F` is clean, the ruff gate exits 0, and the touched test files are F-clean.

**ID check.** 42482 (and 42483) were unused after a fetch: 1099 refs, no ref name, no object name in any history, no file in any worktree's `backlog/`, and none in the 98,860 distinct blobs at every ref tip (`$FU/sp23-idcheck.sh`, adapted from `kc22-idcheck2.sh`).

**Files.**
- Product: `tldw_chatbook/LLM_Calls/anthropic_subscription.py`.
- Tests: `Tests/real_profile_guard.py`, `Tests/conftest.py`, `Tests/test_network_guard.py`.
- Config template: `tldw_chatbook/Config_Files/default_config.toml` (pin in `Tests/test_config_load_settings_table_guard.py`).
- Inventory: `Docs/security/production-diagnostic-inventory.json`.
- Docs: `Docs/User_Guide/settings.md`.

No lesson: the probe-pre-emption trap is TASK-42477's.
<!-- SECTION:NOTES:END -->
