---
id: TASK-38973
title: "Twenty-six stub-screen tests in four Backup_Recovery and Console config files are red on dev, because their screen doubles predate new attributes"
status: To Do
created_date: 2026-11-06 01:21
labels:
- follow-up
- tests
- console
- backup-recovery
priority: medium
references:
- Tests/Backup_Recovery/test_console_config_sync_lifetime.py
- Tests/UI/test_console_full_config_preparation.py
- Tests/Backup_Recovery/test_console_sync_maintenance.py
- Tests/Backup_Recovery/test_home_citation_retirement.py
- tldw_chatbook/UI/Screens/chat_screen.py
- tldw_chatbook/UI/Screens/home_screen.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05. One whole run of the four files at `-n 4` gave 26 failed and 38 passed in 149 s, at load 13. None failed with RecoveryRequired. Each red comes from a screen double that is missing an attribute the code now reads:

| File | Failed | Failure and cause |
|---|---|---|
| `Tests/Backup_Recovery/test_console_config_sync_lifetime.py` | 4 (native-worker, -external, -real-pause, -inprogress) | The child process raises `type object 'ChatScreen' has no attribute 'app_instance'` at `chat_screen.py:17891`. ab43a0303b (2026-11-02) added that read. |
| `Tests/UI/test_console_full_config_preparation.py` | 7 | `'types.SimpleNamespace' object has no attribute '_sync_console_temporary_chip'` at `chat_screen.py:17780`. 6ad513fad6 (TASK-37840, 2026-10-30) added that read. |
| `Tests/Backup_Recovery/test_console_sync_maintenance.py` | 3 (active, timeout, error) | Hits the 45 s subprocess timeout. The same missing `_sync_console_temporary_chip` raises at `:17780` before the sync reaches the held `_warm_console_effective_scope_cache_if_stale` (`:17791`), so the child waits on `entered.wait()` forever. |
| `Tests/Backup_Recovery/test_home_citation_retirement.py` | 12 | `'types.SimpleNamespace' object has no attribute 'is_mounted'`. `HomeScreen._home_content_seam_call` reads `self.is_mounted` (8e8af97b91, 2026-10-19). The three cancel cases time out waiting for `entered`. |

None of the four files is on a PR lane. `test.yml:133` runs `Tests --ignore=Tests/UI` only on pushes to main and on manual dispatch. None of the files changes on the branches due to merge before this one.

This has happened before. TASK-33620.15.1.2 (Done 2026-10-27) fixed a missing `_fleet` on the same double, and TASK-36518's notes recorded 14 of these reds as pre-existing without filing them. Because a missing attribute surfaces as a subprocess hang rather than a named failure, the drift stays hidden.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All 26 nodes pass 3 of 3 whole-file -n 4 runs, and none reaches its 45 s subprocess timeout
- [ ] #2 Each still fails on its own assertion under a temporary source edit that removes the behaviour it pins (the maintenance drain ends early, the native worker does not defer, a Home worker keeps a lease, FULL enters native work on the main thread)
- [ ] #3 A missing double attribute now fails fast and names the attribute instead of hanging (for example, doubles built from the real class's attribute set, or the files joining a lane with measured time); the notes say which approach and why
<!-- AC:END -->
