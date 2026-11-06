---
id: TASK-38979
title: "Retiring the portrait region editor before it mounts leaves it open, and the Personas performance controller's close never returns"
status: Done
assignee: ['@claude']
created_date: 2026-11-06 01:42
updated_date: '2026-11-06 17:26'
labels:
- follow-up
- bug
- personas
- tests
- flaky
priority: medium
references:
- tldw_chatbook/Widgets/Persona_Widgets/portrait_region_editor.py
- tldw_chatbook/UI/Persona_Modules/portrait_regions.py
- tldw_chatbook/UI/Persona_Modules/portrait_performance.py
- tldw_chatbook/UI/Screens/personas_screen.py
- Tests/UI/test_portrait_region_editor.py
dependencies:
- TASK-38747
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05. TASK-38747 recorded that `Tests/UI/test_portrait_region_editor.py::test_dialog_validation_is_coalesced_owned_and_drained` fails, then hangs past `--timeout`, in 2 of 3 `-n 4` runs. It did not own the failure. No lane runs the file.

**Measured on dev:**
- `[close]` alone, in a serial probe at load 5.5-7.4, hung in 4 of 19 runs; passing runs took 3.3-4.1 s.
- Alone at `--timeout=300` (load 12.3), `[replacement]` passed. `[close]` hit the timeout and then hung in teardown until it was killed at 1586 s.
- The whole file at `-n 4 --timeout=120` gave 23 passed and 1 failed on the timeout, then hung for more than 20 minutes.

**Mechanism, from faulthandler and asyncio task dumps:**
1. `PortraitPerformanceController.close()` waits on `self._task` (`tldw_chatbook/UI/Persona_Modules/portrait_performance.py:803`).
2. That task is the `handle("regions")` worker, parked in `edit_regions` on `screen.app.push_screen_wait(modal)` (`portrait_regions.py:126`).
3. `discard()` (`portrait_performance.py:246-252`) calls `PortraitRegionEditor.retire()` (`tldw_chatbook/Widgets/Persona_Widgets/portrait_region_editor.py:463-472`).
4. In every hung run, `retire()` ran while the editor was already `app.screen` but `is_mounted` was still False. It set `_retired` and skipped `dismiss_safe_once_when_on_top`.
5. The editor then finished mounting and stayed on top while retired, with the stack `['Screen','PersonasScreen','PortraitRegionEditor']`. Nothing dismisses it, so `push_screen_wait` never resolves.
6. In passing runs, `retire()` ran after mount and the dismiss returned True.
7. After pytest-timeout fails the test, its `finally: await controller.close()` leaves the pytest-asyncio loop finalizer idle in `select` for good, so `--timeout` never ends the process.

**Production reach.** PersonasScreen awaits `controller.close()` on a runtime-backend change (`personas_screen.py:1917`) and on unmount (`:1971`). A discard or close that lands between push and mount would hang those awaits. The window is narrow.

The TASK-38747 branch adds only `PortraitRegionEditor.confirm_quit`; `retire`, `edit_regions` and `close` are unchanged there.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Discarding the draft or closing the controller after the region editor is pushed but before it mounts closes the editor and close() returns; a deterministic test that holds the mount pins this and is red on dev
- [x] #2 The [close] case passes 20 of 20 serial runs and the file passes 3 of 3 runs at -n 4, with the load recorded (dev: [close] hung in 4 of 19 serial runs)
- [x] #3 When a case in this file fails, the run still ends: no case holds pytest past --timeout in teardown
- [x] #4 A Personas runtime-backend switch and a screen unmount made while a region editor is opening both complete
- [x] #5 Once green, the file joins the UI census with its measured serial time, and its shard stays at least 5 minutes under its cap
<!-- AC:END -->

## Implementation Plan

1. Reproduce deterministically: a mounted test holds `PortraitRegionEditor.on_mount` (class-level async stand-in that awaits an event, then runs the real one), so the editor is `app.screen` with `is_mounted` False. While held, retire it four ways: `controller.discard()`, `controller.close()`, `PersonasScreen.handle_runtime_backend_changed(...)` and `PersonasScreen.on_unmount()`. Release the mount, then wait at most 5 s (plain `asyncio.wait`, no Pilot) for the worker and the retiring call. Red on dev: the editor stays on top and the wait times out; the test's cleanup dismisses the editor so the case still ends.
2. Fix at the editor: `retire()` before mount records the retirement and queues itself (`call_later`) so it closes the editor once mounted, through the same `dismiss_safe_once_when_on_top` path (a covering screen still keeps the close for resume). (`PortraitCompositionReview.retire()` has the same gate; it is outside these criteria and goes to the report as a follow-up.)
3. Bound the file's cleanups: the `finally` blocks that await `controller.close()` first dismiss a still-open editor and bound the close, so a failing case cannot hold pytest in teardown.
4. Measure: `[close]` 20 serial runs, the whole file 3 runs at `-n 4`, with load; then add the file to the UI census tail with its measured serial time, check the shard against its cap.

## Implementation Notes

**Approach.** `PortraitRegionEditor.retire()` gated its dismissal on `is_mounted`. `push_screen` puts the editor on top at once, but its Compose and Mount run later on its own task, so a retire in that window (`discard()`, `close()`, and through them a runtime-backend switch or Personas unmount) set `_retired`, skipped the dismiss, and left the editor open; the regions worker stayed in `push_screen_wait` and `close()` waited on it forever. A retire before mount now queues itself (`self.call_later(self.retire)`); the Callback is handled after `_pre_process` sets `_is_mounted`, so the editor closes through the usual `dismiss_safe_once_when_on_top` path (a covering screen still keeps the close for its resume).

**Tests.** `test_retiring_the_editor_before_it_mounts_still_closes_it[discard|close|backend|unmount]` holds the real editor in the window with a class-level async `on_mount` stand-in, retires it four ways (`discard()`, `close()`, `handle_runtime_backend_changed("local")`, and `PersonasScreen.on_unmount()`, the unmount handler that awaits `close()`), and waits at most 5 s with plain `asyncio.wait`. The file's cleanups go through `close_bounded` (dismiss a leftover editor, give `close()` 10 s, cancel and name stragglers), and the drained test's body awaits use `settled(...)` (10 s, names the task).

**Red/green.** On the base source the new test was 4 of 4 red in about 5 s each ("retired before mount, the editor stayed open (stack ['Screen', 'PersonasScreen', 'PortraitRegionEditor'], retired=True)"), and the run ended (4 failed in 42 s). With the fix, 4 passed.
- AC#2: `[close]` 20 of 20 serial (3.3-5.1 s each, load 8.2-12.3); the file 3 of 3 at `-n 4` (29 passed, 25-30 s, load 8.0-9.9).
- AC#3: with the fix temporarily reverted, the whole file serially at `--timeout=60` ended in 132 s: 5 failed (the four new cases, and `[replacement]`, which hit the window by itself and failed at its 10 s bound) and 24 passed, with no hang.
- AC#5: serial, the file is 29 passed in 98.6 s at load about 9 (`--timeout=180`, the lane's flags). It joins the census tail as one whole-file entry, which deals it to UI Fast Lane shard 22 of 32 on this base. That shard's slowest job over the 7 completed shard runs read from 8 recent PR runs (42150847285 to 42191565464, read-only) was 12.3 min; a census file measured both ways ran 12.1 s on CI against 10.5 s here, so about 1.9 min CI is expected, about 14.2 min in all, which is 5.8 min under the 20-minute cap. The whole shard, run locally serially with the lane's flags, passed: 169 passed in 872 s at load 32. A tail rebase changes the shard; the coordinator should re-check it. The whole-file entry is longer than the lane's usual 30 s: 16 function entries would have dealt one onto shard 3, whose slowest run (16.9 min) is already less than 5 minutes under the cap.

**Not done here.** `PortraitCompositionReview.retire()` has the same `is_mounted` gate; it is outside these criteria and is reported as a follow-up.

**Files.** `tldw_chatbook/Widgets/Persona_Widgets/portrait_region_editor.py`, `Tests/UI/test_portrait_region_editor.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, `backlog/docs/lessons-textual.md`.
