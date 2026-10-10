---
id: TASK-34720
title: >-
  Console: a burst of chip activations stacks identical dialogs and crashes the
  app with RecursionError
status: In Progress
assignee: []
created_date: '2026-10-10 16:10'
updated_date: '2026-10-10 18:56'
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

**The other dialog chips stack the same way.** The Provider and Model chips (Switch model), System Prompt chip (system prompt editor), Assistant chip (character picker), Scope chip (retrieval scope picker) and Cost chip (Conversation Inspector) also open their dialog once per Enter/Space/click. A burst stacks identical dialogs: live on dev, sixteen keys typed on the focused Model chip needed fifteen Esc presses to get back to the Console. (Whether a longer burst on these chips also reaches the RecursionError was not measured; the stacking is the defect either way.)

**Scope.** The duplicate-dialog guard for every status-strip chip that opens a dialog (TASK-33622.3 AC #5 and its siblings) and the crash it causes. The Sources, Tools and Run chips only reveal the Inspector rail, the Approvals chip only moves focus and the Temporary chip runs an exclusive save worker, so none of them pushes a dialog. Type-to-compose, the bare 'y' Trace key and focus return stay with TASK-33622.3.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 However many Library chip activations arrive together, at most one Library access dialog is open at a time
- [x] #2 Typing a sentence while the Library chip has focus no longer crashes the app
- [x] #3 Closing the Library access dialog (Cancel, or Esc when clean) returns to the Console, with no second dialog left beneath it
- [x] #4 Activating the Library chip again after the dialog closes still opens the dialog
- [x] #5 Pilot tests on the real ChatScreen send a burst of activations and a typed sentence to the focused chip, and fail on the pre-fix code by stacking dialogs
- [x] #6 However many activations of the Provider, Model, System Prompt, Assistant, Scope or Cost chip arrive together, at most one of that chip's dialogs opens, and the chip opens it again once it closes
- [x] #7 A parametrized Pilot test on the real ChatScreen covers every dialog-opening status chip and fails on the pre-fix code by stacking dialogs
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce live on dev (tmux harness) and capture the RecursionError.
2. Root-cause: count stacked dialogs per typed burst; confirm render depth grows per stacked translucent modal.
3. Pilot tests on the real ChatScreen (burst of activations; typed sentence) that fail on dev by stacking dialogs.
4. Guard ConsoleLibraryPolicyController.open_access so it opens only while the Console owns the top of the screen stack.
5. Extend to every dialog-opening status chip: a parametrized burst test (red on dev), then the same rule checked at each dialog's single push site.
6. Green tests, compare touched-area test files with dev by test id, preflight, live re-verification (Library and Model chips), rebase onto origin/dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reproduced live on dev 75159f8843 (tmux, 160x45, isolated profile): after closing the Library access dialog, focus returns to the Library chip; a typed sentence then stacked one dialog per Space/Enter and the app exited with RecursionError (Compositor.__rich__/render_strips; log event=unhandled_exception widget_type=ConsoleLibraryAccessModal). The reported 'dialog stays open after Save' is the same bug at a smaller burst: Save/Cancel on the top dialog reveal an identical one beneath. (Staying open after Save with the 'Applied/Saved' copy is the dialog's designed behaviour; Cancel then closes it.)

Root cause: ConsoleLibraryPolicyController.open_access (UI/Console_Modules/library_policy.py) pushed unconditionally. ConsoleLibraryChip posts one OpenRequested per Enter/Space, and a typed burst queues every key on the focused chip before ChatScreen handles the first request, so N activations pushed N dialogs. Textual paints each translucent ModalScreen over the one beneath it via BackgroundScreen -> Compositor.render_strips, nesting one render (~20 Python frames, measured: 71 frames at 1 dialog, 134 at 4, 491 at 21 in the headless harness) per stacked screen until the live app passed the recursion limit.

Fix: open_access now returns early unless the Console owns the top of the screen stack (new required console_owns_screen_stack callable, wired to ChatScreen._owns_console_screen_stack in UI/Console_Modules/wiring.py). push_screen appends synchronously, so the first request's dialog covers the Console and every queued request after it is refused; once the dialog closes the chip opens it again. The stack is the authority, so no open/closed flag can go stale.

Tests: two Pilot tests on the real ChatScreen in Tests/UI/test_console_library_controls_workflow.py (marked bootstrap_profile, the documented opt-in for real-app mounts). On dev they fail stacking 4 and 21 dialogs; green with the fix. The headless harness renders from a shallower stack and survives 21 dialogs, so the tests pin the stack, not the RecursionError itself; the crash is verified live. Controller unit fixture updated for the new kwarg.

Live after the fix (160x45 then 100x30): Save, Cancel, typed sentence + Enter -> one dialog, app alive, one Esc returns to the Console; a 60-space burst + 3 Enters at 100x30 -> one dialog, Esc returns to the Console, no RecursionError in the log.

Extension to every dialog-opening chip (AC #6, #7). A parametrized Pilot test posts four activation requests from each status-strip chip that opens a dialog -- what four queued Enter/Space keys post. On dev all seven stack four identical dialogs (ConsoleLibraryAccessModal, ConsoleModelPopover from both the Provider and Model chips, ConsoleSystemPromptModal, ConsoleCharacterPickerModal, ConsoleScopePickerModal, ConsoleConversationInspector). In that dev run the typed-sentence test also hit the RecursionError itself headless. With the guards, each chip opens exactly one dialog, and opens it again once that dialog is gone. Each dialog's single opener checks ChatScreen._owns_console_screen_stack() immediately before its push: open_model_switcher (Console_Modules/model_switcher.py; chips, Alt+M, /model, palette), ConsolePromptsController._open_console_system_prompt_editor (Console_Modules/prompts.py; chip, rail line, /system, palette), ChatScreen._open_console_character_picker, _open_console_retrieval_scope_picker and _push_console_inspector (cost chip and Ctrl+Shift+P). Checking at the push rather than in each chip handler is what covers openers that run as workers: the system prompt editor is opened via run_worker, so a handler-time check would let every queued request start a worker before the first one pushes. It also covers every other entry point into those dialogs with no per-dialog flag. The Library controller keeps its injected form of the same rule.

Live, model chip: on dev, sixteen keys typed on the focused Model chip stacked Switch model popovers that took fifteen Esc presses to clear. On the fix the same burst opened one popover, and one Esc returned to the Console.

Two unit tests drive these openers with a SimpleNamespace screen (test_console_model_apply_chips, test_chat_screen_console_inspector_loader). Their fakes now provide _owns_console_screen_stack, so they keep testing what they tested before. Regression check, by test id: 41 test files that exercise the touched openers ran on dev and on the fix. Locally, many of them fail at setup with RecoveryRequired('raw_source_selection_changed'), so the 25 such files were re-run with the bootstrap profile forced on, on both sides. The only differences were the two fakes above, now fixed, and one load-flaky settings-modal test that passes 3/3 on both sides in isolation.

Out of scope, still with TASK-33622.3: typed text landing on the chip instead of the composer, and the bare 'y' Trace key. This task delivers 33622.3's AC #5 and extends it to the sibling chips. Console Buttons (header actions, rail buttons) bind only Enter, so typed text does not press them repeatedly; they were not changed.

Files: tldw_chatbook/UI/Console_Modules/library_policy.py, tldw_chatbook/UI/Console_Modules/wiring.py, tldw_chatbook/UI/Console_Modules/model_switcher.py, tldw_chatbook/UI/Console_Modules/prompts.py, tldw_chatbook/UI/Screens/chat_screen.py, Tests/UI/test_console_library_controls_workflow.py, Tests/UI/test_console_library_policy_controller.py.
<!-- SECTION:NOTES:END -->
