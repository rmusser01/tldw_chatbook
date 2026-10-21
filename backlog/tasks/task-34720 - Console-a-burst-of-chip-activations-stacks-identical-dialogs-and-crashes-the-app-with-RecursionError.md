---
id: TASK-34720
title: >-
  Console: a burst of chip activations stacks identical dialogs and crashes the
  app with RecursionError
status: In Progress
assignee: []
created_date: '2026-10-10 16:10'
updated_date: '2026-10-19 12:40'
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

**On dev after TASK-35160 (rebased onto 4de437a5c5, 2026-10-19).** The status strip is now one focusable text line of segments: Enter or Space activates the cursored segment, and closing a dialog hands focus back to the line with its cursor still on that segment. The Library chip left the line; its dialog opens from the composer menu's "Library access..." entry. The crash path is unchanged for the segments whose dialog openers had no guard: on 4de437a5c5 a sentence typed on the focused line with the cursor on System prompt stacks 21 system prompt editors, and a burst on System prompt, Assistant, Scope or Cost stacks one dialog per activation. Switch model (Provider and Model segments) gained the same stack check in TASK-33008.3. The acceptance criteria below were re-worded from chips to segments and to Library access requests; the outcomes they require are unchanged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 However many Library access requests arrive together, at most one Library access dialog is open at a time
- [x] #2 Typing a sentence while the status line has focus no longer crashes the app
- [x] #3 Closing the Library access dialog (Cancel, or Esc when clean) returns to the Console, with no second dialog left beneath it
- [x] #4 Requesting Library access again after the dialog closes still opens the dialog
- [x] #5 Pilot tests on the real ChatScreen send a burst of Library access requests and a typed sentence to the focused status line, and fail on the pre-fix code by stacking dialogs
- [x] #6 However many activations of the Provider, Model, System prompt, Assistant, Scope or Cost segment arrive together, at most one of that segment's dialogs opens, and the segment opens it again once it closes
- [x] #7 A parametrized Pilot test on the real ChatScreen covers every dialog-opening status-line segment and fails on the pre-fix code by stacking dialogs
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce live on dev (tmux harness) and capture the RecursionError.
2. Root-cause: count stacked dialogs per typed burst; confirm render depth grows per stacked translucent modal.
3. Pilot tests on the real ChatScreen (burst of activations; typed sentence) that fail on dev by stacking dialogs.
4. Guard ConsoleLibraryPolicyController.open_access so it opens only while the Console owns the top of the screen stack.
5. Extend to every dialog-opening status chip: a parametrized burst test (red on dev), then the same rule checked at each dialog's single push site.
6. Green tests, compare touched-area test files with dev by test id, preflight, live re-verification (Library and Model chips), rebase onto origin/dev.
7. Rebase onto dev after TASK-35160 (one status line, Library access in the composer menu): retarget the tests at the status-line segments and the composer-menu route, re-prove them red on dev, re-run the gates and one live check.
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

**Rebase onto dev 4de437a5c5 (2026-10-19, about 1,500 commits).** Conflicts, resolved by keeping dev's content plus this branch's: the workflow test's imports (dev had added a module-level bootstrap_profile pytestmark, which replaces this branch's per-test markers); model_switcher.py, where dev's TASK-33008.3 final fix wave had already put the same stack check at the top of `_open_switcher`, which `open_model_switcher` calls (dev's check kept; this branch's copy also called `_console_setup_modal_blocking()`, which would have shut the Get started card's switcher there, so it was dropped and only a comment line remains); prompts.py, where dev made the modal import lazy (the check now sits before it); test_console_model_apply_chips.py, whose fake dev had already given `_owns_console_screen_stack` (dev's taken, so commit c538c2dd53, "two screen fakes", now changes only one). The two original test commits (a5d84b00ef, f15bcb5a33) import `ConsoleLibraryChip`, which TASK-35160 removed, so they no longer collect on their own; 58c773440c retargets them.

**Re-rebase onto dev ce4aa4a231 (2026-10-19 17:00).** Four new dev commits (private SQLite write-only imports, backlog status hygiene), none touching this branch's files: clean, no conflicts. Commit ids that name this branch's commits are those of the final rebase onto 0fef489ece (below); a measurement keeps the head it ran on.

TASK-35160 changed the surface under this task (see the Description addendum): one focusable status line whose Enter/Space activate the cursored segment, and Library access in the composer menu. The tests were retargeted to it (Tests/UI/test_console_library_controls_workflow.py): four Library access requests through the composer-menu route (`_handle_console_composer_menu_choice(ACTION_LIBRARY_ACCESS)`), Cancel, and one more request; the sentence typed as one key burst on the focused line with its cursor moved onto System prompt, plus Enter; and, per dialog-opening segment, four requests posted from the line (`ConsoleStatusText.SEGMENT_MESSAGES`), a pop, then one more request. Plain command (`pytest <file> -q`, no plugin), dev 4de437a5c5 with the new file copied in: 6 failed for the right reason -- 4 Library access dialogs; 21 ConsoleSystemPromptModal from the sentence; 4 each for system-prompt, assistant, scope, cost -- and provider/model passed, because dev's Switch model opener already refuses a covered Console. Negative control for those two: with dev's check in `_open_switcher` replaced by `if False:`, both stacked 4 ConsoleModelPopover (restored by edit afterwards). Head: the 8 new cases pass under the plain command. The file's 4 `test_four_policy_states_render_exactly_and_open_independent_axes` cases fail identically on dev's own copy of the file (NoMatches '#console-status-chips': dev's `_ChipApp` composes ConsoleStatusChips without that id, since d102ce9cb5) and are not this task's. Re-run after the re-rebase, plain command: ce4aa4a231 with the new file copied in, 10 failed (the same 6 with the same counts, plus those 4) and 3 passed; head 6e0df05b66, 9 passed and only those 4 failed, with the same NoMatches. The file is not in scripts/ui_pr_gate_census.txt and takes about 80 s, too slow to add, so CI runs it only in the full UI shards.

Comments and docs: the guard comments and `open_access`'s docstring now name the status line and the composer menu's Library access entry; the expanded `_owns_console_screen_stack` docstring went back to dev's one line. Docs/User_Guide/console.md's "Status chips" section gained one paragraph: an entry that opens a dialog opens one however fast it is pressed, and opens it again once the dialog closes.

Ratchets: chat_screen.py grows by six lines (three two-line guards). Tests/Architecture/test_screen_size_ratchet.py's chat_screen row rises 24369 -> 24375 with a dated owner-decision comment; dev 4de437a5c5 already measured 24378 (+9) before this change, so the row fails on head with exactly dev's +9 message, the overage recorded and not absorbed. Unchanged after the re-rebase: ce4aa4a231 measures 24378 against 24369, head 24384 against 24375, both +9. Dev has been over this ceiling since 8d58322b51 (the single-line status strip's review round 1, 2026-10-16), the day after the row was lowered to 24369. test_module_size_ratchet.py: 7 failures, identical on dev, none on a touched file. test_console_wave6_inventory.py: the same 3 failures on dev and head (chat_screen.py line counts already about 2,200 over those wave-6 limits on dev; the messages differ only by this fix's six lines, 24378 -> 24384). No module or CSS added, so the ADR-097 boot ratchets were not re-measured.

Live (R67, 2026-10-19, 160x45, isolated profile and null keyring via launch.sh, real Anthropic claude-haiku-4-5 through the review proxy). Both builds: dismiss the model-list consent dialog, click the System Prompt segment (editor opens), Esc (focus back on the line, cursor on Provider), Right twice (cursor on System Prompt), then the sentence plus a carriage return as one `send-keys -l` burst. Fix (this branch at dd5d02f8cb, before the TASK-35160 rebase; now f5791c3000): one system prompt editor, its text area empty, app alive; one Esc returned to the Console; no RecursionError or unhandled_exception in the app log. Composer Menu -> Library access... opened the dialog and one Esc returned. Dev 4de437a5c5: the same burst stacked 21 editors and took 21 Esc presses to clear; this run did not reach RecursionError. Both builds also showed the bare 'y' Trace key's toast ("no persisted trace yet"), which stays with TASK-33622.3. Evidence: scratchpad uxrev/evidence/bd5/R67/ (README.txt, 01-25).

Live re-check on the head rebased onto ce4aa4a231 (6e0df05b66, 2026-10-19 17:05, same harness and steps, launch r67fix). Before the burst the line's cursor sat on "System Prompt: off" (reverse video in the SGR capture). The burst left one system prompt editor with an empty text area, and the app stayed alive. One Esc closed it and showed the Console's ready line. The app log has no RecursionError or unhandled_exception. Evidence: uxrev/evidence/bd5/R67/final/ (README.txt, 01-07).

Files: tldw_chatbook/UI/Console_Modules/library_policy.py, tldw_chatbook/UI/Console_Modules/wiring.py, tldw_chatbook/UI/Console_Modules/model_switcher.py (comment only; dev holds the check), tldw_chatbook/UI/Console_Modules/prompts.py, tldw_chatbook/UI/Screens/chat_screen.py, Tests/UI/test_console_library_controls_workflow.py, Tests/UI/test_console_library_policy_controller.py, Tests/UI/test_chat_screen_console_inspector_loader.py, Tests/Architecture/test_screen_size_ratchet.py, Docs/User_Guide/console.md.
<!-- SECTION:NOTES:END -->
