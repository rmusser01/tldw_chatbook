# PR #3155 notification review repair evidence

Date: 2026-10-31
Repair parent: `3c011a3f0d5ea7c3e6aa7392545ab78767c749ab`
Current-dev base for this repair: `57e330e735fff25515778e839f31ed85602c34f2`
Host: Windows, Python 3.12.10; existing shared venv, default pytest importlib configuration, fresh private profile for each run.

## Scope and decisions

The repair retains the existing notification cadence and Settings persistence.
It changes the one-shot thread to non-daemon so orderly interpreter shutdown
retains subprocess timeout/kill/reap ownership, escapes Linux body XML through
`html.escape(..., quote=False)`, returns False for nonzero notifier exits, and
logs only category plus exit code/exception type. Exception text, output,
argv, and session identity are excluded from failure logs.

[ADR-282](../../../backlog/decisions/282-console-os-notification-dispatch.md)
records platform support, privacy, separation from the inbox, and the bounded
orderly-exit wait. The ADR number was allocated after scanning all reachable
refs/history and live worktree decision filenames (prior maximum 281); it stays
provisional until merge-time collision checking. The user explicitly waived
size limits and subsequently instructed the parent to skip CI for this pass.
No remote action or release was performed by this implementer.

## Verification receipts

These are local receipts, not a claim that the whole suite or native desktop
delivery passed. Raw log/XML files remain in the worktree as untracked evidence;
this committed table records their results and the commands below reproduce them.

| Run | Result | JUnit time | Local receipt |
| --- | --- | --- | --- |
| Pre-repair baseline (parent) | 32 passed, 0 failed, 0 errors | 252.26 s | `.tmp-pr3155-core.xml` |
| New regressions RED | 0 passed, 5 failed, 0 errors | 15.04 s | `.tmp-pr3155-red.xml` |
| Dispatcher GREEN | 28 passed, 0 failed, 0 errors | 9.32 s | `.tmp-pr3155-green-notifier.xml` |
| Repaired feature selection | 35 passed, 2 failed, 0 errors | 542.25 s | `.tmp-pr3155-repair-core.xml` |
| Settings grammar | 8 passed, 0 failed, 0 errors | 540.02 s | `.tmp-pr3155-repair-grammar.xml` |
| Console serial replay | 4 passed, 2 failed, 0 errors | 300.81 s | `.tmp-pr3155-repair-console-serial.xml` |
| ChatScreen/Settings pins | 4 passed, 0 failed, 0 errors | 2.35 s | `.tmp-pr3155-repair-pins.xml` |
| Config pin | 2 passed, 0 failed, 0 errors | 0.97 s | `.tmp-pr3155-repair-config-pin.xml` |

The RED run failed all five new cases for the intended reason: raw Linux tags,
True after nonzero exit, two exception-text leaks, and a missing completion
receipt after parent exit. The subsequent dispatcher run passed all 28 cases.
The real slow-child test imports the real screen helper and sender in a fresh
interpreter; only the external notifier executable is replaced by a Python child
that sleeps six seconds. It proves off-thread execution, an immediate caller
return, timeout cleanup, and a False result retained through orderly process exit.
There is no explicit thread join in the parent fixture to hide the daemon defect.

### Open Console teardown qualification

The repaired combined run reported 35 passed / 2 failed. The failing parent nodes were:

- `Tests/UI/test_console_os_notifications.py::test_viewed_failure_os_notification_uses_title_not_error_copy`
- `Tests/UI/test_console_os_notifications.py::test_park_approval_os_notification_dedupes_with_the_toast`

A second, serial six-case file run reported 4 passed / 2 failed, in different nodes:

- `Tests/UI/test_console_os_notifications.py::test_background_run_outcome_fires_one_os_notification`
- `Tests/UI/test_console_os_notifications.py::test_per_type_toggle_blocks_its_kind`

Each failing private child reported **1 passed, 1 teardown error**. Its exception
group includes `RuntimeError: Native app workers remain uncertain; config retirement refused`
and failure to complete `shutdown_default_executor`. The two first-run failures
passed in the serial replay. No third suite was run and no application-cleanup
behavior was changed to make these receipts green. The underlying teardown
cause was not independently established; moving failures are not a stable green.

All four named tests call `_wire_os_notifications` before constructing their app.
That helper, at `Tests/UI/test_console_os_notifications.py:44`, replaces
`send_os_notification` with `_RecordingSender` and, at line 58, replaces
`_spawn_os_notification_thread` with `lambda fn: fn()`. Thus these cases cannot
execute the changed production thread helper or dispatcher during their assertions;
the actual notifier lifecycle is exercised separately by the green slow-child test.
This establishes the test boundary, not a general claim that all shutdown behavior
is correct.

## Census, diagnostics, and static checks

The census checker documents a **verified-green subset** at
`scripts/check_ui_pr_gate_census.py:31`. Only the successfully qualified Settings
journey was appended; its latest run passed. The Console file is retained in the
repository and explicitly **unqualified**, not silently dropped from an existing
census contract. No previous census entry was removed, reordered, or substituted.
The existing 1,173-entry prefix is unchanged, and the new floor is 1,174.

Normalized original-prefix SHA-256: `0924d49048ba3b295cd000d24be335634df263dc0d30207ccbfe56eecc200631`.

Diagnostic statement review found exactly three new metadata-only debug calls
in `Utils/os_notifications.py`; ChatScreen and Settings statements were unchanged.
The full inventory generator completed with `--write`. A postcheck rescanned the
new module (three calls, digest `b1687c2331944c91ff78`, no sink or path candidate),
removed its row in memory, and confirmed the remainder equals the base inventory
exactly. Final totals: 709 owners, 1,058 TASK-492 calls, 56 TASK-31551 calls,
7,775 TASK-494 calls, 16 sink files. No production source changed after generation.

The user's size waiver is recorded beside exact pins: ChatScreen **24,420 lines /
739 methods**, Settings **35,758 lines**, config **12,245 lines**. All six targeted
ceiling/slack checks passed. No code was moved or compressed to fit a former pin.

Ruff check and formatter checks passed for the new notifier/test module. Comparing
all changed existing Python files against the base showed no new Ruff findings;
existing findings remain (ChatScreen 17, Settings 13, config 63, module ratchet 1).
`git diff --check` is clean. No full test sweep was requested or performed.

## Reproduction commands

Use the shared Python 3.12 venv with TEMP/TMP under a private runtime directory,
and create a fresh `TLDW_TEST_CONFIG_ROOT` with the repository's private-directory
helper before each invocation. Retain the repository's default pytest addopts.

```text
python -m pytest Tests/Utils/test_os_notifications.py -q
python -m pytest Tests/Utils/test_os_notifications.py Tests/UI/test_console_os_notifications.py Tests/UI/test_settings_console_os_notifications_journeys.py Tests/test_config_console_defaults.py -k os_notifications -q
python -m pytest Tests/UI/test_settings_console_behavior_grammar.py -q
python -m pytest Tests/UI/test_console_os_notifications.py -q
python -m pytest Tests/Architecture/test_screen_size_ratchet.py Tests/Architecture/test_module_size_ratchet.py -k "chat_screen or settings_screen" -q
python -m pytest Tests/Architecture/test_module_size_ratchet.py::test_module_does_not_grow_past_its_budget[tldw_chatbook/config.py] Tests/Architecture/test_module_size_ratchet.py::test_budget_is_not_left_slack[tldw_chatbook/config.py] -q
python scripts/check_persistent_diagnostic_inventory.py --statements tldw_chatbook/Utils/os_notifications.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/UI/Screens/settings_screen.py --since 57e330e735fff25515778e839f31ed85602c34f2
python scripts/check_persistent_diagnostic_inventory.py --write
python scripts/check_ui_pr_gate_census.py
```

## Remaining limits

Native macOS/Linux desktop display, OS notification permissions, daemon-specific
rendering, and locale behavior were not exercised on this Windows host. Windows
native delivery remains unsupported by design. Forced process termination and
uninterruptible platform process creation are outside the orderly-shutdown timeout
guarantee. The task remains In Progress pending independent review and disposition
of the Console teardown qualification; no passing-suite or merge-readiness claim
is made for that file.
