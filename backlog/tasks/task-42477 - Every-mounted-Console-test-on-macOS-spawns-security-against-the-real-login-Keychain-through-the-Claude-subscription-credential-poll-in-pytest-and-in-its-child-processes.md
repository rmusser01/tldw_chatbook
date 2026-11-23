---
id: TASK-42477
title: "Every mounted Console test on macOS spawns security against the real login Keychain through the Claude subscription credential poll, in pytest and in its child processes"
status: Done
assignee:
- '@claude'
created_date: 2026-11-23 10:39
updated_date: 2026-11-23 11:21
labels:
- follow-up
- tests
- console
- security
- hardening
priority: high
references:
- tldw_chatbook/LLM_Calls/anthropic_subscription.py
- Tests/test_network_guard.py
- Tests/LLM_Calls/test_anthropic_subscription.py
- Tests/LLM_Calls/test_subscription_credential_cache.py
- Tests/UI/test_console_subscription_readiness.py
- Tests/UI/test_personas_subscription_readiness.py
- Docs/User_Guide/settings.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 22 of the follow-up burn-down (2026-11-23). Checked on `fu/r22-keychain` at 62531d037a (the head of `fu/r22-railrecon`); paths are under `tldw_chatbook/` unless they start with `Tests/` or `Docs/`.

**Why.** On macOS, when `~/.claude/.credentials.json` is absent, `LLM_Calls/anthropic_subscription.py::_keychain_credential_raw` runs `/usr/bin/security find-generic-password -s "Claude Code-credentials" -w`, a query of the real OS Keychain service. Moving HOME does not stop the spawn. What `security` returns under a moved HOME (the token, nothing, a block or a prompt) was not established, because every measured spawn was refused. Earlier lessons point away from a returned token: TASK-31450, TASK-32344 and TASK-33020 in `backlog/docs/lessons-live-verification.md` saw a 401, "no login keychain" and an authorization block. The Console's credential poll (`ChatScreen._poll_console_credential_readiness`) calls `subscription_credential_status(background=True)` on every tick, whether or not Anthropic is selected, and the first background read reaches that spawn. The Personas and Settings screens make the same call.

The suite's isolation does not cover it. `Tests/conftest.py` redirects HOME, XDG and the config path and pins the null `keyring` backend; none of that stopped the `security` spawn. No conftest stubs the read, and neither does the scratch wrapper. Only four test files fake it, and only in their own Keychain tests. Many tests also run the app in child processes (the `Tests/Performance` profilers and the private-profile harness), which no conftest monkeypatch reaches. The owner's rule is that tests never touch the real OS keyring.

**Measured** with a scratch probe, never with the real command: a PYTHONPATH `sitecustomize` whose audit hook logs and refuses (`PermissionError`) every `security` spawn, and adds itself to a child's explicit PYTHONPATH.
- A 40-file sample of `Tests/UI/test_console_*.py` at `-n 4` (464 tests ran): 71 refused spawns from 58 tests in 17 files. 48 came from the 4 xdist workers and 23 from 12 private-profile pytest children. All 71 came from the `claude-credential-readiness` thread (`_SubscriptionReadinessCache._refresh > read_claude_code_credential > _keychain_credential_raw`), with argv `['/usr/bin/security', 'find-generic-password', '-s', 'Claude Code-credentials', '-w']`. The read is memoized per process for 5 s, so this is a floor. (In that sample, 76 failures were the known local `RecoveryRequired: raw_source_selection_changed` trap; those tests never mounted a Console.)
- Child path: both children of `Tests/Performance/test_console_mount_profile.py` (`run_console_mount_profile.py`, plain Python, no conftest, HOME moved to their own scratch dir) made the same spawn from the same thread.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No test spawns `security` against the real login Keychain: with the fix, the same Console sample and the same Performance child file make zero `security` spawns under the refusing probe
- [x] #2 The fix also covers child processes that load no conftest (a plain Python child that inherits the test environment), with no per-harness plumbing
- [x] #3 A guard test, in a file a pull-request lane already runs, fails if the credential read would run `security` in-process or in a child. It is shown red before the fix (refusing the spawn, never running it) and green after, and it bites on Linux CI too
- [x] #4 The four test files that test the Keychain path still exercise it with their own fakes and pass; a negative control shows those cases need the Keychain branch
- [x] #5 A real user whose HOME is the account's own home still reads the Keychain (pinned by a unit test), and the missing-credential copy and the User Guide say when the Keychain is not read
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure with a refusing scratch probe (audit hook in a PYTHONPATH `sitecustomize`, which also reaches children): a Console UI sample and one Performance child file.
2. Fix at the one spawn site, `_keychain_credential_raw`: read the Keychain only while HOME is the login account's own home. Every test sandbox already moves HOME (conftest, the scratch wrapper, the Performance runners, the private-profile children), so in-process and child processes are covered with nothing to plumb.
3. Guard tests in `Tests/test_network_guard.py` (run by the "Run Architecture guard contracts" step): in-process (`subprocess.run` recorder) and in a plain Python child (its own audit hook), darwin forced, every spawn refused. Show them red at base and green after.
4. The four Keychain test files opt in to the account-home branch. Add a unit test for the helper's both branches, and run negative controls.
5. Update the missing-credential message and the User Guide sentence. Add a lesson. Run the four files, `Tests/LLM_Calls`, the Console sample, a Performance child file, `Tests/CI`, preflight, ruff and the ruff gate.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Fix (product, one spawn site).** `_keychain_credential_raw` now returns `None` unless `_home_is_account_home()`: `os.path.samefile(Path.home(), pwd.getpwuid(os.getuid()).pw_dir)`, so a symlinked home still counts. Any failure, including no `pwd` module, a missing HOME directory or an unresolvable home, reads as "not the account home". `_keychain_credential_raw` is the only `/usr/bin/security` call in the product, and every caller (the Console, Personas and Settings polls, provider readiness, the send path) reaches it through `read_claude_code_credential`, so one guard covers them all. The check runs before the 5 s memo, so a memo never serves a sandboxed process. `MISSING_CREDENTIAL_MESSAGE` and the Settings User Guide sentence now say the Keychain is read only while HOME is the account's own home folder.

**Why this option.** Every test sandbox already moves HOME: the conftest bootstrap, the scratch wrapper, the `Tests/Performance` runners (each sets its own scratch HOME), the private-profile pytest children, and live scratch launches. Keying the read on HOME therefore covers in-process tests and every child with no plumbing. The candidates were weaker:
- A conftest autouse patch does not reach children.
- A new env switch would have to be added to every child-environment builder. `run_console_three_turn_profile.build_child_environment` is a strict allowlist that already drops `PYTHON_KEYRING_BACKEND`, and the TASK-31210 harnesses rebuild the environment as well. The switch would also be a test-only product knob.
- A `sitecustomize` misses children whose PYTHONPATH is rebuilt (the mount profiler sets `PYTHONPATH=<repo root>`).

The possible cost: a real user who runs Chatbook with a moved HOME no longer borrows the Keychain credential, if `security` would have returned it there. If `security` finds nothing under a moved HOME (as the TASK-31450, TASK-32344 and TASK-33020 lessons suggest), users lose nothing. Either way the trade is deliberate: a moved HOME is a sandboxed profile. The credential file under that HOME still works, and the missing-credential message says why the Keychain was skipped.

**Guard tests** (`Tests/test_network_guard.py`, a whole-file target of the "Run Architecture guard contracts" PR step). Both force the module's darwin branch, so they bite on Linux CI too:
- In-process: a `subprocess.run` recorder refuses `security`, and the test drives a sync read plus a fresh readiness cache's background read (the Console path).
- A plain Python child: inherits the test environment, as `Tests/Performance` children do, and installs its own audit hook that refuses the spawn.

Red at the base, with the spawn refused, never run: both failed with `assert [['/usr/bin/security', 'find-generic-password', '-s', 'Claude Code-credentials', '-w']] == []`. Green after. A probe's audit hook pre-empts the child's own hook, so the child test's red was taken without the probe (its own hook refused the spawn). A sanity run against a fake `security` binary first confirmed that the hook refuses.

**Helper pin.** `test_keychain_is_read_only_while_home_is_the_accounts_own[account|link|moved|missing]` (`Tests/LLM_Calls/test_anthropic_subscription.py`) fakes the password database and `run`. Negative controls under temporary source edits, reverted:
- Helper forced `False`: `[account]` and `[link]` red.
- Helper forced `True`: `[moved]`, `[missing]` and the in-process guard red.

**The four Keychain files** opt in with `monkeypatch.setattr(subscription, "_home_is_account_home", lambda: True)` beside their existing darwin and fake-`run` setup, so they still run the real reader against their fakes. Negative control: without those lines, 5 cases go red (`test_keychain_reader_is_memoized`, both credential-cache tests, Console `[keychain]`, Personas `[keychain-ready]`). With them, all 60 pass.

**Measurement after the fix** (same refusing probe, load 4-5): the same 40-file Console sample still made 71 calls to `_keychain_credential_raw` from 58 tests, and 0 spawns. Its 81 failures are the same set as before the fix: 76 `RecoveryRequired: raw_source_selection_changed`, plus 5 in `local_server_discovery_card`, `transcript_sync_gate` and `workspace_tree_performance`, which also failed under the probe at base. That base run answered "missing", like the fix. They cannot be re-run unprobed at base, because that run would read the Keychain. In `Tests/Performance/test_console_mount_profile.py`, both children made 2 calls and 0 spawns; 4 passed.

**Runs** (through the scratch wrapper, refusing probe on, load 4-5):
- The four Keychain files at `-n 4`: 60 passed.
- `Tests/LLM_Calls` and `Tests/UI/test_console_subscription_readiness.py` at `-n 4`: 2280 passed, 47 skipped, 7 failed, 0 spawns. The 7 are red at the base with the same command (patch set aside and re-applied): 3 `test_summarization_diagnostic_privacy` digest pins, 2 `test_realtime_protocol`, `test_only_fireworks_restricts_reasoning_effort_levels` (deepseek also restricts), and `test_chat_with_anthropic_refuses_cross_origin_redirect` (`RecoveryRequired` in `load_settings`).
- `Tests/test_network_guard.py` serially: 32 passed, 1 skipped, in 1.06 s.
- `Tests/CI` at `-n 4`: 1823 passed, 2 skipped.
- Lint: `ruff check --select F` is clean on the product file. Its two findings in touched test files predate this task (`test_anthropic_subscription.py:149` F841 and `test_personas_subscription_readiness.py:17` F401; both are at the base, and CI's F gate covers only `tldw_chatbook`). The review round removed both. `Tests/lint/ruff_gate.py` exits 0.
- Preflight: passed (`scripts/preflight.sh` through the wrapper, all derived-artifact checks).

No module-size row lists `anthropic_subscription.py`, and no census or lane changed.

**Files.**
- Product: `tldw_chatbook/LLM_Calls/anthropic_subscription.py`.
- Tests: `Tests/test_network_guard.py`, `Tests/LLM_Calls/test_anthropic_subscription.py`, `Tests/LLM_Calls/test_subscription_credential_cache.py`, `Tests/UI/test_console_subscription_readiness.py`, `Tests/UI/test_personas_subscription_readiness.py`.
- Docs: `Docs/User_Guide/settings.md`, and a lesson in `backlog/docs/lessons-testing-evidence.md`.

**ID check.** 42477-42480 were unused after a fetch: 1130 refs, all history names, every other worktree's backlog, and all 98,896 distinct blobs at every ref tip.

**Review round (2026-11-23).** The review of b97db969fd ("ready with fixes, wording") confirmed:
- `_keychain_credential_raw` is the only `security` spawn site;
- the HOME check is sound on every edge case (4.6 µs per call);
- the guards run on a PR lane and fail without the fix;
- the opt-in files never run the real command.

Its findings are fixed in new commits:
- **Unmeasured claim.** The first version said a moved HOME "would still read the real account's token". Every spawn was refused, so only the spawn against the real OS Keychain service was measured; whether it would have returned the token, found nothing, blocked or prompted was not. The lesson, `_home_is_account_home`'s docstring, the guard comment in `Tests/test_network_guard.py`, this task's title and description, and AC #1 now say only that. The lesson cites the three `lessons-live-verification.md` entries that point the other way (TASK-31450, TASK-32344, TASK-33020). The cost sentence above is now conditional. The task file was renamed to match the new title. The first commit's message was reworded to the measured claim before push (now 876fc9b1d7; the tree is unchanged).
- **Opt-in order.** In the four files, the `_home_is_account_home` opt-in is now set after the fake `run`, and in the helper test HOME and the password database are set after it. monkeypatch's reverse undo therefore never leaves the real `run` in place while the HOME check says "home".
- **The returncode test was vacuous.** `test_keychain_reader_nonzero_returncode_is_none` now counts calls to its fake `run`. With its opt-in removed it fails with `assert 0 == 1`; before, it passed either way.
- **The fix's limit is in the lesson:** a child started with HOME removed counts as the account's home, because `Path.home()` falls back to the password database. An example is `Tests/Performance/test_console_three_turn_profile.py:3991`, whose child environment holds only PATH; it only imports modules, so it never reaches the read.

Filed TASK-42478 (high: the polls spawn the Keychain read whatever the auth source), TASK-42479 (low: the three-turn profiler's child gets no null keyring backend) and TASK-42480 (low: lane `Tests/test_keyring_isolation.py`).

After the review fixes, all with the refusing probe on (0 spawns in every run):
- the four Keychain files plus `Tests/test_network_guard.py` at `-n 4`: 92 passed, 1 skipped;
- `Tests/CI` at `-n 4`: 1823 passed, 2 skipped;
- preflight: passed;
- `ruff check --select F` on the touched files: clean, after removing the two findings that predate this task (an unused `result =` in `test_anthropic_subscription.py` and an unused `character_handler_module` import in `test_personas_subscription_readiness.py`); `Tests/lint/ruff_gate.py` exits 0.
<!-- SECTION:NOTES:END -->
