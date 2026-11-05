---
id: TASK-38748
title: "Console mix-in Discard or reopen during a threaded Library save cancels it after the write lands"
status: Done
assignee:
- '@claude'
created_date: 2026-11-04 06:34
updated_date: '2026-11-05 21:39'
labels:
- follow-up
- console
- prompts
- bug
priority: medium
references:
- tldw_chatbook/Widgets/Console/console_prompts_modal.py
- tldw_chatbook/UI/Console_Modules/prompts.py
- tldw_chatbook/Prompt_Management/prompt_scope_service.py
- Tests/UI/test_console_prompt_mixins.py
dependencies:
- TASK-38686
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the review of PR #3402 (TASK-38686), which is not on dev yet. Checked on dev bb80905eb3, 2026-11-04; the line numbers below are dev's.

**On dev.** `PromptScopeService.save_prompt` runs a local create or update on the event loop (`_maybe_await`, `Prompt_Management/prompt_scope_service.py:2092-2097`). While a Console mix-in Save writes, no other press is handled, so nothing can cancel the save part-way.

**With #3402.** The write moves to a worker thread (`asyncio.to_thread(run_finite_local_worker, ...)`), because a contended write now waits up to SQLite's 10 s busy timeout. The save worker can now suspend mid-write. If it is cancelled after the write lands, everything after `await self.scope.save_prompt(...)` in `_ConsolePromptMixinsFlow.save` (`UI/Console_Modules/prompts.py:1189`) is skipped: the preferences reconcile (`reconcile_mixin_placements`), and, in the modal's `_save_mixin_editor` (`Widgets/Console/console_prompts_modal.py:535`), the editor's saved record and revision. #3402's review probe showed the result for a Save or Preview press: the DB at version 2, the editor still on version 1 and unsaved, and the next Save refused.

#3402 guards those two presses: while the save worker is unfinished, Save and Preview are ignored and show "Saving Library changes…". Two paths remain, in `console_prompts_modal.py`:
- **Discard.** The dirty guard's Discard (:2823-2827) calls `_back_internal(discard=True)`. When Mix-ins is the modal's only mode (`initial_mode="mixins"`, :236), that dismisses the modal (:742), and dismissing it cancels the modal's workers, the save among them. The view stays dirty while the save runs, so Escape, Back and Close all lead to that guard.
- **Reopening Mix-ins.** When Discard goes back to Browse instead, opening Mix-ins again starts an exclusive `console-mixins` worker (`_open_mixins`, :428-432), which cancels a save still running.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A Discard (from Escape, Back or Close), or reopening Mix-ins, while a mix-in save runs never cancels that save: once its write lands, the mix-in preferences are reconciled to the saved record
- [x] #2 After such a save, the next Save of the same record is accepted; the user is never left on a stale version that the next Save is refused for
- [x] #3 A test holds the write behind a contended lock (a raw connection in `BEGIN IMMEDIATE`), presses Discard in a modal opened on Mix-ins, and separately reopens Mix-ins from Browse; it is red before the fix
- [x] #4 If a press is ignored or deferred while the save runs, it shows the same "Saving Library changes…" status #3402 uses
<!-- AC:END -->

## Implementation Plan

1. The save is cancelled only when the modal is dismissed (it owns the save worker) or when an exclusive `console-mixins` worker starts. Reaching Browse during a save is what makes the reopen path possible, so both paths close if no press can leave the Mix-ins view while the save runs.
2. Defer, rather than drop, a leave pressed during the save: `_back_internal` (Discard from Escape, Back or Close, and a clean Back), `_request_close` (a clean Close or Escape) and the view's Browse button each record themselves as the pending leave and show "Saving Library changes…" (TASK-38686's status). When the save worker finishes, the modal runs the pending leave. "Keep editing" drops it.
3. Test: extend the lane-run `test_press_during_contended_mixin_save_cannot_cancel_it` with Discard cases from Escape, Back and Close in a modal opened on Mix-ins, plus a reopen case from Browse. A raw connection holds `BEGIN IMMEDIATE`. Each case pins the Saving copy, the reconciled preferences once the write lands, and a next Save accepted as version 3. Show it red before the fix, then green.
4. Update the Console User Guide's mix-ins saving note.

## Implementation Notes

**Approach: a leave pressed during a mix-in save waits for it.** The save can be cancelled two ways: dismissing the modal cancels its workers, and an exclusive `console-mixins` worker (reopening Mix-ins) cancels the group. Reopening is only reachable through Browse, so if nothing leaves the Mix-ins view mid-save, both paths are closed.
- `ConsolePromptsModal._defer_mixin_leave` records the leave and shows "Saving Library changes…", TASK-38686's copy (AC#4). It is consulted where each leave actually happens, after the dirty-guard check:
  - `_back_internal`: Discard from Escape, Back or Close, and a clean Back;
  - `_request_close`: a clean Close or Escape;
  - `_mixins_browse`: the view's "Browse Prompts / Recipes" button.
- `on_worker_state_changed` runs the pending leave once the save worker is finished, so the save, its preference reconcile and the editor's saved state complete first (AC#1). A failed save still leaves on a Discard, because Discard means discard.
- "Keep editing" drops a pending leave, so a user who changes their mind is not closed out after the save.
- I chose to defer rather than ignore the press. The user confirmed Discard, so dropping it silently would leave them in the window with no sign their choice was lost. Moving the save into its own worker group would not have been enough: a reopened view could still read version 1 before the write landed, and the next Save would be refused (AC#2).

**Test (AC#3).** `Tests/UI/test_console_prompt_mixins.py::test_press_during_contended_mixin_save_cannot_cancel_it` is the lane-run node TASK-38686 added; it gains five cases, so the census needs no edit. In every case a raw connection holds `BEGIN IMMEDIATE` and a spy shows the save has reached its write.
- `escape`, `back` and `close`: Discard in a modal opened on Mix-ins.
- `reopen`: Back, then Discard, in a modal opened on Browse; then reopen Mix-ins.
- `keep`: Discard, then Escape and Keep editing.

Each case:
- clears the status before the second press and pins the "Saving" copy, with the modal still on Mix-ins;
- after release, waits for the stored version 2, then checks the preferences reconcile before any reopen (a reopened projection reconciles by itself, which at first masked the dismiss-path cancel in a probe);
- requires the next Save to land as version 3 (AC#2).

**Red/green.**
- At base 29592ecc9b's modal source, the five new cases failed on "the leave did not wait for the save".
- With that assert and the Saving assert removed, and Mix-ins reopened while the write still waited, `escape`/`back`/`close`/`reopen` failed 3 of 3 runs on `assert 'Style' == 'Style saved'`: the save was cancelled after its write landed.
- Without the Keep editing line, `[keep]` was red ("Mix-ins did not load": the modal closed anyway).
- With the fix: 7 of 7 cases, 5 of 5 runs (`-n 4`, load 34-56).

**Covering runs** (`-n 4`, scratch wrapper): the prompts modal, mix-ins, controller, workbench contract, modal dismissal and quit-guard files, the Library mix-ins file, the Console prompt draft workbench and the Chat mix-in files gave 22 failed, 814 passed. The same five files that hold those failures, run with base sources, fail on the same nodes plus one load flake (`test_console_active_stream_sync_skips_unchanged_chrome_and_inspector`). They are pre-existing: `test_guarded_modal_quit_census` names five other modals, and the rest are Console/Improve UI tests that are red locally on dev.

**Docs.** `Docs/User_Guide/console.md` (Prompt mix-ins) says that leaving during a save waits for it, and that Keep editing cancels that.

**Files.** `tldw_chatbook/Widgets/Console/console_prompts_modal.py`, `Tests/UI/test_console_prompt_mixins.py`, `Docs/User_Guide/console.md`.
