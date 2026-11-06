---
id: TASK-38426
title: >-
  Console: a refused Enter leaves 'Sending…' under the turn and paints no System
  row
status: To Do
assignee: []
created_date: '2026-10-31 03:50'
updated_date: '2026-11-06 13:11'
labels:
  - console
  - errors-recovery
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found live while verifying TASK-33621.11's review fixes, and identical on dev 7536ac3b87 (evidence U09/30). With a PNG staged on claude-haiku-4-5 (a model the capability table says cannot take images), typing a message and pressing Enter shows the user row with 'Sending…' under it. The send gate refused the turn in 85 ms (console_send_stage phase=ui_dispatch status=refused), but the 'Sending…' acknowledgement stays and the refusal's System row is not painted. It appears only after a later sync, and then above the stale 'Sending…' turn. The user sees a send that looks in flight and never learns why it stopped.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A refused Enter removes or resolves its 'Sending…' acknowledgement at once
- [ ] #2 The refusal's System row is painted when the refusal happens, below the refused turn
- [ ] #3 A mounted test drives a refused Enter through the real keyboard path and asserts both
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verification, round 10 of the 2026-10-23 burn-down (2026-11-06). Nothing was implemented here. Checked on dev b2c618185f and on send PR #3433 (head e3106a313c), which lands TASK-38994's resync fix; the runs used the round-10 PRs stacked through storage #3435.

- **The fix.** 14cd6d2ff4 (TASK-33620.1, "repaint the transcript when a refused send's Sending row is released") is on dev, but not in 7536ac3b87, where this task was observed. TASK-38994 generalizes the repaint to a refusal that writes no store row.
- **The probe.** It drives this task's path: a PNG staged on the harness's non-vision `test-model`, then a driver-delivered Enter. Six runs on the stacked PRs (load 25-31):
  - "Sending…" appears at 17-28 ms and the user row at 58-97 ms.
  - Both are gone when the System row lands, at 0.21-0.47 s: "Console send blocked: test-model can't accept images. Remove the attachment, …".
  - There are 0 provider calls, and the draft is kept.
  - Dev gives the same result (System row at 0.21-0.25 s).
- **AC#1 holds.** A refused Enter releases its Sending… row.
  - Shown by the probe, and on a lane by `Tests/UI/test_console_send_acknowledgement.py::test_a_refused_acknowledged_enter_repaints_its_refusal` (dev :807).
  - `test_a_release_reopens_the_transcript_gate_the_paint_went_past` in the same file covers the release too. It is new in send PR #3433 (#3433 :617) and not yet on dev.
  - That file is a UI census entry, `scripts/ui_pr_gate_census.txt:807`.
- **AC#2 holds in substance but not as worded.** The refusal's System row is painted when the refusal happens, and it is the last row. But the refused turn's own row is rolled back, as `Docs/User_Guide/console/chat-basics.md:504-511` documents ("the **Sending…** row disappears and the draft is kept"), so there is no turn for it to sit below. Needs an owner ruling to reword it, e.g. "is the last transcript row".
- **AC#3 is partial.**
  - The laned test drives a real `pilot.press("enter")` through the same send gate (`chat_screen.py` `_console_send_gate`, then `prompt_queue.py` `_append_owned_dispatch_message`). But it uses the archive-in-progress refusal, not the image refusal, and it does not assert that the System row is last.
  - What remains: one mounted test of the image path that asserts both (the probe is a template), and the AC#2 wording. Red-before would need a tree from before 14cd6d2ff4.
<!-- SECTION:NOTES:END -->
