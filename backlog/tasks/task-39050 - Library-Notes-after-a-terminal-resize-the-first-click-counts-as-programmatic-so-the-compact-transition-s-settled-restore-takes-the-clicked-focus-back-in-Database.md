---
id: TASK-39050
title: "Library Notes: after a terminal resize the first click counts as programmatic, so the compact transition's settled restore takes the clicked focus back in Database Notes and Folder files"
status: To Do
created_date: 2026-11-06 10:09
labels:
- follow-up
- library
- notes
- focus
priority: low
references:
- tldw_chatbook/UI/Screens/library_screen.py
- tldw_chatbook/UI/Library_Modules/library_notes_controller.py
- Tests/UI/test_library_file_notes_workspace.py
- Tests/UI/test_library_shell.py
dependencies:
- TASK-38957
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 10 of the 2026-10-23 burn-down; checked on dev b2c618185f, 2026-11-06, and on the Library branch that lands TASK-38957 (the state after it merges). TASK-38957's notes record the click case as a follow-up. Line numbers are from that branch; dev's are the same up to `:11241`.

**Why a click is read as programmatic.**
- `on_resize` sets `resize_settling = True` (`library_screen.py:8452`).
- Only real input clears it: `_mark_library_notes_user_interaction` (`library_notes_controller.py:3018-3020`). That runs from `on_mouse_down` (`library_screen.py:8560`), `on_key`, the wheel, and `action_focus_next`/`previous` (`:9420-9450`, TASK-32106's keyboard fix).
- `on_descendant_focus` counts `resize_settling` as programmatic (`:11203-11208`). The focus generation only advances when `relevant and not programmatic` (`:11254-11258`).
- Textual moves focus before it forwards the MouseDown. So for a click the screen sees DescendantFocus first, while `resize_settling` is still True, and the generation stays put.
- The settled restore's guard (`library_notes_controller.py:2768-2777`, `library_screen.py:5717-5735`) then still passes, and the restore moves focus back.
- TASK-38957 only widened `relevant` to Folder files (`:11242-11244`). The `programmatic` term is shared by every Notes mode and is unchanged.

**Probe.** The probe held `_restore_library_notes_settled_focus`, resized across the compact breakpoint, acted, then ran the held restore. `resize_settling` was True before every action. The holds make it deterministic: 3 runs at `-n 4` gave the same result (load 18-20).

| Mode | Action | Dev | After TASK-38957 |
|---|---|---|---|
| Folder files (160x45 -> 79x30) | Tab | focus taken back (TASK-38957's bug) | kept; generation 1 -> 5 |
| Folder files | click the editor | taken back to `library-file-notes-items-grip` | taken back; generation 1 -> 1 |
| Database Notes (170x48 -> 100x30) | Shift+Tab | kept | kept |
| Database Notes | click the title | taken back to `library-note-body` | taken back; generation 1 -> 1 |

- No test covers a click. Tests that simulate a user focus call `_mark_library_notes_user_interaction()` before `.focus()` (for example `Tests/UI/test_library_shell.py:32410`), which skips this path.
- Media's resize focus code reads the same generation and settling flag (`library_screen.py:7719-7723`, `:7905-7915`). It was not probed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After a resize across the compact breakpoint, a `pilot.click` that moves focus between the transition's two restores advances the focus generation and keeps the clicked control, in Database Notes and in Folder files; one test per mode, each red before the fix (generation 1 -> 1, focus back on `library-note-body` / `library-file-notes-items-grip`)
- [ ] #2 Programmatic landings during settling are still classified programmatic (the restore's own target, the restoring flag, Textual's focus reset after a prune): TASK-38957's Tab test and the Notes focus and canvas-sync pins stay green, named in the notes
- [ ] #3 Media's and Conversations' resize-focus pins stay green; if Media shows the same click case, it is fixed and tested too, or recorded as out of scope with the probe result
- [ ] #4 The new tests run on a PR lane (a census tail entry) and pass 20 of 20 at `-n 4`, with the load recorded
<!-- AC:END -->
