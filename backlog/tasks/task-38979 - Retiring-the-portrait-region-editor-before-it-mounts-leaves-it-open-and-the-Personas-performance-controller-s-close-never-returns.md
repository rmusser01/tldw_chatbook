---
id: TASK-38979
title: "Retiring the portrait region editor before it mounts leaves it open and the Personas performance controller's close never returns"
status: To Do
created_date: 2026-11-06 01:42
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
- [ ] #1 Discarding the draft or closing the controller after the region editor is pushed but before it mounts closes the editor and close() returns; a deterministic test that holds the mount pins this and is red on dev
- [ ] #2 The [close] case passes 20 of 20 serial runs and the file passes 3 of 3 runs at -n 4, with the load recorded (dev: [close] hung in 4 of 19 serial runs)
- [ ] #3 When a case in this file fails, the run still ends: no case holds pytest past --timeout in teardown
- [ ] #4 A Personas runtime-backend switch and a screen unmount made while a region editor is opening both complete
- [ ] #5 Once green, the file joins the UI census with its measured serial time, and its shard stays at least 5 minutes under its cap
<!-- AC:END -->
