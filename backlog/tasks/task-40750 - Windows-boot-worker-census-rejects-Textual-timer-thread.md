---
id: TASK-40750
title: Windows boot worker census rejects the Textual timer thread
status: To Do
type: bug
created_date: '2026-11-13 08:05'
labels:
- windows
- performance
- ci
priority: medium
references:
- https://github.com/rmusser01/tldw_chatbook/actions/runs/41289667965/job/124798084235
- https://github.com/rmusser01/tldw_chatbook/commit/72e0fdb561
- Tests/Performance/test_boot_worker_census.py
- tldw_chatbook/Utils/text_selection_crash_guard.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Windows boot census rejects a thread from Textual's dedicated timer executor. Establish which timers start that pool before UI readiness and keep the boot contract green while preserving TASK-38462's separation of short timer sleeps from the default executor.

### Reproduce

Use a clean Windows checkout of dev at `beab5c1e4e1128086ac518e525a9274dff4baf5f` with the repository's dev dependencies installed. The audit used Python 3.12.10, Textual 8.2.8 and pytest 8.3.5. From the repository root in PowerShell, activate the dev virtual environment and use a new private test root under a private parent such as `$env:USERPROFILE/.codex`.

```powershell
$testRoot = Join-Path "$env:USERPROFILE/.codex" ('ci-repro-' + [guid]::NewGuid().ToString('N'))
python -c "import sys; from tldw_chatbook.Utils.platform_files import os; os.mkdir(sys.argv[1], 0o700)" "$testRoot"
$env:TLDW_TEST_CONFIG_ROOT = $testRoot
$env:TMP = $testRoot
$env:TEMP = $testRoot
python -m pytest 'Tests/Performance/test_boot_worker_census.py::test_boot_worker_and_thread_starts_stay_within_the_allowlist' -q --tb=short --no-cov
```

### Findings

The failing entry is `('textual-timer_#', 'concurrent.futures.thread', '_worker')` at `test_boot_worker_census.py:484`. Commit `72e0fdb561` (TASK-38462) introduced the `textual-timer` pool for short Windows timer waits in `install_windows_timer_sleep()`. The historical Linux CI failure for this test named the `_reconcile_console_after_attach` worker instead. That worker was already present on the merge parent, so the current Windows failure has a different signature.

The audit also tested a detached checkout of the same dev commit after reverse-applying the first-parent diff of merge commit `cf5fda70d3` for `tldw_chatbook/UI/Console_Modules/session.py` and `tldw_chatbook/UI/Console_Modules/wiring.py`, the two production files changed by #3306. The boot census rejected the same `('textual-timer_#', 'concurrent.futures.thread', '_worker')` entry there.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 The boot-worker census passes on Windows with the actual timer thread start accounted for.
- [ ] #2 Short Windows timer waits stay off the default executor, and the census continues to reject unexpected boot threads.
<!-- AC:END -->
