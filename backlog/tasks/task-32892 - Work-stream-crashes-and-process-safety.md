---
id: TASK-32892
title: "Work stream: crashes and process safety"
status: To Do
assignee: []
created_date: '2026-09-21 23:05'
labels:
  - tier2-review
  - review-crashes
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Seven paths where the app dies, hangs, or kills itself. One is a P0: `SimpleAudioPlayer.stop()` sends
`SIGKILL` to `os.getpgid(...)` of a child spawned **without** `start_new_session`, so the child shares the
app's own process group and the app kills itself. The rest are Textual crashes -- a modal that cannot
compose, a non-ASCII chat topic that kills the Stats screen at mount, workers with the default
`exit_on_error=True`, and a DOM lookup in a `finally:` body.

All small diffs; all "the app dies". One PR.

Source: tier-2 code review 2026-09-21 -- `qa/tier2-code-review-2026-09-21/report.md` (26 slices, 890,356 lines: the surface tier 1 never reached). Per-slice evidence in `qa/tier2-code-review-2026-09-21/slices/`, reproductions in `phase4-verification.md`, and per-finding re-validation against `origin/dev d0face3ebe` in `validation/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The three P0s are closed and each has a test that is born red
- [ ] #2 No `Popen` in TTS spawns into the app's own process group
- [ ] #3 Every crash site in the children has a pinning test that fails before the fix
- [ ] #4 `./scripts/preflight.sh` is green and no size ratchet moves
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented on `fix/tier2-crashes` as `6ee291ee8c`. Not pushed. All seven confirmed and fixed; 27 tests, every
one observed red before its fix.

**P0-1's root cause is duplication, not a stray keyword -- which makes it the review's own thesis in
miniature.** `UI/Console_Modules/library_activity.py:92 ConsoleLibraryActivityController.build_provider` was
a byte-for-byte copy of `console_runtime._library_provider_for_app`. Commit `5dd1077df6` retired
`LocalLibraryToolService`'s `collections_service` parameter and updated **only the copy**, leaving the
production call site passing an argument the constructor no longer accepts. The collections service is
genuinely not needed -- that commit retired the generic-container surfaces and the service has no
`_collections` at all. Fixed by collapsing to one builder (the runtime's, which gained `**activity_kwargs`),
with the controller delegating.

There was a **third** copy of the retired parameter: `Tests/UI/test_console_library_tool_setting.py:229`
`assert service._collections is app.local_library_collections_service`, which cannot pass. It has been in the
tree since `0577884cf2` without failing anything -- see **TASK-32908**, which is the larger finding this
exposed: `Tests/UI/` (1,172 files, 19,182 test functions) is excluded from the PR gate by name and errors
locally on the ADR-126 gate. A P0 shipped, and the test that would have caught it was somewhere nothing runs.

**P0-2 was fixed twice; the first attempt was wrong and measuring caught it.** A fail-closed guard inside
`SyncEngine` broke 10 legitimate tests -- the engine's `owner_id` means "the scope to sync", and its *push*
half legitimately operates on locally-owned rows. Shipped instead:
`SchedulingService.sync_target_owner_id()`, applying the rule `_active_server_owner_id`'s docstring already
states, used by both pull entry points -- `sync_now` and the workbench's notification-results pull at
`schedules_workbench.py:1102`, which had the same defect for results and whose own file already contained the
correct precedent at `:4686`. `SyncEngine` gained documentation only, zero behaviour change.
Of the four candidate sites, `:1936 _server_id_for_local` (a mapping lookup -- a miss never arms anything) and
`:1950 _record_sync_error` (appends an error string) were **deliberately left**: every caller passes an
explicit owner, so the `None` default is unreachable in production.

**P0-3** took `start_new_session=(os.name == "posix")` on all three `Popen` calls plus
`audio_cpp_supervisor.py` and `backends/chatterbox.py`, matching `server_lifecycle.run_server_subprocess`
(TASK-32806.5), and added `_kill_player_process_group` refusing a pgid equal to `os.getpgrp()`. Where
`test_spawn_uses_exact_argv_cwd_stdin_and_environment` pins exact kwargs, the flag was **added to the
assertion with a rationale** rather than the assertion loosened. The behavioural test literally reproduces the
suicide: `stop() SIGKILLed the app's own process group: [(25017, SIGKILL)]`.

Two design calls worth recording:
- **Item 5**: the Stats widget id was **dropped**, not slugified. Slugifying merely trades `BadIdentifier`
  for a duplicate-id crash on two topics that slugify alike.
- **Item 4**: replaced with `Widgets/confirmation_dialog.py`, which also renders markup-off, so a profile
  named `[old] voice` names itself instead of vanishing. An AST sweep for the same defect class across the
  repo found exactly one real instance.

Also corrected inside the diff: one `logger.error(..., exc_info=True)` became `logger.opt(exception=True)` --
loguru ignores the stdlib kwarg, which is the review's own repo-wide finding.

Gates: preflight GREEN; size ratchet exactly the 5 baseline reds by name; regressions diffed as **name sets**
against a pristine-dev baseline -- the 12 that surfaced are the ones described above and all re-run green.
`Tests/UI/test_schedules_sync_surface.py`, `test_schedules_workbench.py` and `test_console_library_tool_setting.py`
error at fixture setup on the ADR-126 gate and are CI-only; every test written here is gate-free.

Deliberately **not** shipped: the `kwargs.update` at ~`:3661` is a second, distinct defect. The overwrite is
pinned as deliberate by `_VIEW_HOOK_OWNERSHIP` ("app-owned-domain"), so `setdefault` would be wrong; its real
consequence is that Console Library-activity capture records nothing in production. Filed as **TASK-32909**.
<!-- SECTION:NOTES:END -->
