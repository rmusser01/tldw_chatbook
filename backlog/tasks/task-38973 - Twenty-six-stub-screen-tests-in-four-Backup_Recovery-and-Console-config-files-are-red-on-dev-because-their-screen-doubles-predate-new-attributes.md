---
id: TASK-38973
title: "Twenty-six stub-screen tests in four Backup_Recovery and Console config files are red on dev, because their screen doubles predate new attributes"
status: Done
assignee:
- '@claude'
created_date: 2026-11-06 01:21
updated_date: 2026-11-06 04:58
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
- [x] #1 All 26 nodes pass 3 of 3 whole-file -n 4 runs, and none reaches its 45 s subprocess timeout
- [x] #2 Each still fails on its own assertion under a temporary source edit that removes the behaviour it pins (the maintenance drain ends early, the native worker does not defer, a Home worker keeps a lease, FULL enters native work on the main thread)
- [x] #3 A missing double attribute now fails fast and names the attribute instead of hanging (for example, doubles built from the real class's attribute set, or the files joining a lane with measured time); the notes say which approach and why
<!-- AC:END -->

## Implementation Plan

1. Re-run the four files at base e8eeae7db9 (-n 4) to confirm the 26 reds.
2. Make the two waits that hang (the maintenance `entered.wait()`, the Home cancel `entered` poll) race the task they wait on, so a sync that dies early re-raises its own AttributeError at once; make the `Surface` double name a class attribute it cannot bind.
3. Give each double the answers the current code reads (attach visit, temporary chip, tabs-sync count, replay request, app/UI readiness, archive recovery view, the terminal-return class default, `is_mounted`), updating expectations only where the product contract changed on purpose.
4. Run the four files 3x whole at -n 4; check no subprocess timeout.
5. Negative controls, one temporary source edit per pinned behaviour.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Checked at base e8eeae7db9 (2026-11-06): the four files at `-n 4` gave 26 failed, 38 passed in 144 s (load 24), the same 26 nodes as filed.

**AC#3 approach: fail fast in place, not a lane.** A plain `SimpleNamespace` double already raises an AttributeError naming the attribute; the drift hid because two waits outlived the error. The maintenance script awaited `entered.wait()` and the Home cancel route polled `entered`, while the sync or worker had already died. Both waits now race the task they wait on and re-raise its own error at once. The `Surface` double in the config-sync lifetime file bound every unknown class attribute with `MethodType`, so a property or class default failed as "first argument must be callable"; it now raises `TypeError: Surface has no instance answer for ChatScreen.<name>`. I chose this over laning the files because the PR lanes run near their caps (TASK-38716), and these four files take 80-90 s together. The change makes the next drift name itself in any local run. That it paid off at once: with the waits racing, the maintenance cases named `_current_console_attach_visit`, not the `_sync_console_temporary_chip` the filing inferred (63cbf3177a added that read first), and the improved `Surface` message named `_console_terminal_return_reading`, a class default added after the filing.

**Double answers added**, each a read the current code makes:
- config-sync lifetime: `app_instance`/`app` (a started app with `_ui_ready=True`, empty config), a no-op archive recovery view (ab43a0303b), and the `_console_terminal_return_reading` class default. Answering `app` was needed because `app_instance` now exists, so `_screen_readiness_config` reads `screen.app`.
- full-config preparation: `_sync_console_temporary_chip` (TASK-37840), the archive recovery view, and the terminal-return default.
- sync maintenance: `_sync_console_temporary_chip`, an attach visit (a real `ConsoleAttachVisit`), `_console_session_tabs_sync_calls=0` (the drain reads it), and a recorder for `_request_console_control_bar_sync`.
- Home: `is_mounted=False` on the three Home doubles (the hidden-Home guard from 8e8af97b91).

**Two maintenance expectations follow product changes made on purpose.** `_sync_native_console_chat_ui` has returned a bool since 3e618d7119, so a pass that meets the maintenance pause returns False, not None. Since TASK-38696, such a pass hands its whole sync to the replay timer, and resume cuts that timer short (`_request_console_control_bar_sync()`) instead of starting a worker. The `error` case still resumes through one worker. Both expectations are exact, and neither is looser than the old one.

**AC#1.** Three whole-file runs of the four files at `-n 4`: 64 passed each, in 79-89 s, at load 38-43. There were no `TimeoutExpired`, and the slowest node took 10.2 s against the 45 s budget.

**AC#2.** Each node failed on its own assertion under a temporary source edit, reverted after each run:
- `_console_sync_maintenance_drain` returns True at once: the 3 maintenance cases failed at `assert not await ..._console_sync_maintenance_drain(...)`.
- `project_console_tick` ignores the rail deferral (no `replay_whole_console_sync`): all 5 native-worker cases failed at `assert native_requested and not rendered and not operations`.
- the worker `close_connection()` in `NotesScopeService.list_notes`/`count_notes` and Home's media worker removed: the 9 list/count/media retirement cases failed with "worker retained lease".
- `list_notes` acquires the database before its policy check: both rejection cases failed with "rejected request acquired database".
- Home's worker closes the app's `chachanotes_db`: the preserve case failed (the seam logged "custom owner must not be closed").
- FULL skips its off-thread warm-up: the 6 held-read cases failed with "FULL completed before its checked config read retired".
- FULL presentation renders live, without the checked display: the expired case failed with `main_scopes == 2`, `main_opens == 64`.

Files: `Tests/Backup_Recovery/test_console_config_sync_lifetime.py`, `Tests/Backup_Recovery/test_console_sync_maintenance.py`, `Tests/Backup_Recovery/test_home_citation_retirement.py`, `Tests/UI/test_console_full_config_preparation.py`. No product code changed. None of the four files is on a PR lane; laning them remains a proposed follow-up.
<!-- SECTION:NOTES:END -->
