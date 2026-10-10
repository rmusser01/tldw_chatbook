---
id: TASK-34720
title: >-
  Console: a burst of Library chip activations stacks Library access dialogs and
  crashes the app with RecursionError
status: In Progress
assignee: []
created_date: '2026-10-10 16:10'
updated_date: '2026-10-10 17:26'
labels:
  - console
  - crash
  - p0
  - library
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Impact (P0 crash).** Live on dev 75159f8843: open the Library access dialog from the Console status-strip Library chip, close it (Save then Cancel, or Esc), and keep typing. Closing the dialog returns focus to the chip (TASK-16211), so typed text lands on the chip instead of the composer (TASK-33622.3). Every Space or Enter in that text activates the chip, and each activation pushed another Library access dialog: the activations are all queued before the first dialog is on screen, so none of them saw it. A typed sentence stacked about twenty identical dialogs. Textual draws every translucent modal over the one beneath it, one nested render per stacked screen, so the stack exceeded Python's recursion limit and the whole app exited with RecursionError in textual._compositor.render_strips (diagnostics: event=unhandled_exception widget_type=ConsoleLibraryAccessModal). A smaller burst leaves duplicate dialogs stacked: Save and Cancel on the top one reveal the next, which reads as 'the dialog stays open after Save'.

**Repro (live, tmux, 160x45).** Click the Library chip, choose Automatic, Save, press Enter on Cancel (focus returns to the chip), then type 'What do my notes say about the project plan and the next steps for it? Reply in one short sentence please.' as one burst. The app exits with RecursionError: maximum recursion depth exceeded. Evidence: uxrev/evidence/bd9/libmodal/12-dev-long-typing-after-dismiss.txt.

**Scope.** Only the duplicate-dialog guard (TASK-33622.3 AC #5) and the crash it causes. Type-to-compose, the bare 'y' Trace key and focus return stay with TASK-33622.3.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 However many Library chip activations arrive together, at most one Library access dialog is open at a time
- [x] #2 Typing a sentence while the Library chip has focus no longer crashes the app
- [x] #3 Closing the Library access dialog (Cancel, or Esc when clean) returns to the Console, with no second dialog left beneath it
- [x] #4 Activating the Library chip again after the dialog closes still opens the dialog
- [x] #5 Pilot tests on the real ChatScreen send a burst of activations and a typed sentence to the focused chip, and fail on the pre-fix code by stacking dialogs
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce live on dev (tmux harness) and capture the RecursionError.
2. Root-cause: count stacked dialogs per typed burst; confirm render depth grows per stacked translucent modal.
3. Pilot tests on the real ChatScreen (burst of activations; typed sentence) that fail on dev by stacking dialogs.
4. Guard ConsoleLibraryPolicyController.open_access so it opens only while the Console owns the top of the screen stack.
5. Green tests, compare touched test files with dev by test id, preflight, live re-verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reproduced live on dev 75159f8843 (tmux, 160x45, isolated profile): after closing the Library access dialog, focus returns to the Library chip; a typed sentence then stacked one dialog per Space/Enter and the app exited with RecursionError (Compositor.__rich__/render_strips; log event=unhandled_exception widget_type=ConsoleLibraryAccessModal). The reported 'dialog stays open after Save' is the same bug at a smaller burst: Save/Cancel on the top dialog reveal an identical one beneath. (Staying open after Save with the 'Applied/Saved' copy is the dialog's designed behaviour; Cancel then closes it.)

Root cause: ConsoleLibraryPolicyController.open_access (UI/Console_Modules/library_policy.py) pushed unconditionally. ConsoleLibraryChip posts one OpenRequested per Enter/Space, and a typed burst queues every key on the focused chip before ChatScreen handles the first request, so N activations pushed N dialogs. Textual paints each translucent ModalScreen over the one beneath it via BackgroundScreen -> Compositor.render_strips, nesting one render (~20 Python frames, measured: 71 frames at 1 dialog, 134 at 4, 491 at 21 in the headless harness) per stacked screen until the live app passed the recursion limit.

Fix: open_access now returns early unless the Console owns the top of the screen stack (new required console_owns_screen_stack callable, wired to ChatScreen._owns_console_screen_stack in UI/Console_Modules/wiring.py). push_screen appends synchronously, so the first request's dialog covers the Console and every queued request after it is refused; once the dialog closes the chip opens it again. The stack is the authority, so no open/closed flag can go stale.

Tests: two Pilot tests on the real ChatScreen in Tests/UI/test_console_library_controls_workflow.py (marked bootstrap_profile, the documented opt-in for real-app mounts). On dev they fail stacking 4 and 21 dialogs; green with the fix. The headless harness renders from a shallower stack and survives 21 dialogs, so the tests pin the stack, not the RecursionError itself; the crash is verified live. Controller unit fixture updated for the new kwarg.

Live after the fix (bd9lm2, 160x45 then 100x30): Save, Cancel, typed sentence + Enter -> one dialog, app alive, one Esc returns to the Console; a 60-space burst + 3 Enters at 100x30 -> one dialog, Esc returns to the Console, no RecursionError in the log.

Out of scope, still with TASK-33622.3: typed text landing on the chip instead of the composer, and the bare 'y' Trace key. This task delivers 33622.3's AC #5. Sibling chips (model, assistant, system prompt) push without the same guard and may stack under a burst too; not changed here.

Files: tldw_chatbook/UI/Console_Modules/library_policy.py, tldw_chatbook/UI/Console_Modules/wiring.py, Tests/UI/test_console_library_controls_workflow.py, Tests/UI/test_console_library_policy_controller.py.
<!-- SECTION:NOTES:END -->
