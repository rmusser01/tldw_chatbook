---
id: TASK-38748
title: "Console mix-in Discard or reopen during a threaded Library save cancels it after the write lands"
status: Done
assignee:
- '@claude'
created_date: 2026-11-04 06:34
updated_date: '2026-11-05 22:33'
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
- [x] #5 A deferred leave runs only after a successful save; when the save fails, the modal stays on Mix-ins with the failure copy and the edit kept (review ruling, 2026-11-05)
- [x] #6 A deferred leave that becomes due while another screen covers the modal (Ctrl+Q's question) runs once the modal is on top again
- [x] #7 Ctrl+Q during a mix-in save refuses with "Still saving Library changes." instead of quitting or asking about discarding
- [x] #8 Refresh (Search or paging) and an on/off toggle from a clean view during the save are refused with the Saving copy and cannot cancel it; a refused toggle restores the switches
<!-- AC:END -->

## Implementation Plan

1. The save is cancelled only when the modal is dismissed (it owns the save worker) or when an exclusive `console-mixins` worker starts. Reaching Browse during a save is what makes the reopen path possible, so both paths close if no press can leave the Mix-ins view while the save runs.
2. Defer, rather than drop, a leave pressed during the save: `_back_internal` (Discard from Escape, Back or Close, and a clean Back), `_request_close` (a clean Close or Escape) and the view's Browse button each record themselves as the pending leave and show "Saving Library changes…" (TASK-38686's status). When the save worker finishes, the modal runs the pending leave. "Keep editing" drops it.
3. Test: extend the lane-run `test_press_during_contended_mixin_save_cannot_cancel_it` with Discard cases from Escape, Back and Close in a modal opened on Mix-ins, plus a reopen case from Browse. A raw connection holds `BEGIN IMMEDIATE`. Each case pins the Saving copy, the reconciled preferences once the write lands, and a next Save accepted as version 3. Show it red before the fix, then green.
4. Update the Console User Guide's mix-ins saving note.
5. Review r8p-rev1 (2026-11-05): run the leave only when the save worker succeeded with True; keep a due leave until the modal is on top (`on_screen_resume`); refuse Ctrl+Q, Refresh and toggles during the save; add cases for each, plus unchanged-editor Close and Browse, each shown red with its line removed.

## Implementation Notes

**Approach: a leave pressed during a mix-in save waits for it.** The save can be cancelled two ways: dismissing the modal cancels its workers, and an exclusive `console-mixins` worker (reopening Mix-ins, Refresh, a toggle) cancels the group. Reopening is only reachable through Browse, so once nothing leaves the Mix-ins view mid-save and nothing in it starts such a worker, both paths are closed.

**The leave is deferred.** `ConsolePromptsModal._defer_mixin_leave` records the leave and shows "Saving Library changes…", TASK-38686's copy (AC#4). It is consulted where each leave actually happens, after the dirty-guard check:
- `_back_internal`: Discard from Escape, Back or Close, and a clean Back;
- `_request_close`: a clean Close or Escape;
- `_mixins_browse`: the view's "Browse Prompts / Recipes" button.

While the edit is unsaved, Back, Close, Escape and Browse still stop at the dirty guard, so only the guard's Discard reaches the deferral. A clean view (an unchanged editor saved) reaches it directly.

**The deferred leave runs only after a successful save** (review ruling, AC#5). `_save_mixin_editor` returns True once the editor holds the saved record, including the "Earlier Library edit saved" path. `on_worker_state_changed` runs the leave only when the worker ended in `SUCCESS` with result `True`; otherwise it drops the leave, so the "Library changes were not saved… Your edit is kept." copy and the edit stay on screen.

**A due leave waits for the modal to be on top** (AC#6). `_finish_mixin_leave` keeps the leave while another screen, such as Ctrl+Q's "Quit while still working?", covers the modal. `on_screen_resume` runs it after Wait. The SafeModalDismissMixin handler still runs too, because Textual dispatches along the MRO.

**"Keep editing" drops a pending leave.** I chose to defer rather than ignore the press: the user confirmed Discard, and dropping it silently would lose that choice.

**Also folded in from the review:**
- `confirm_quit` refuses with `refuse_quit_while_working(self, "Still saving Library changes.")` while the save runs (AC#7). The modal's existing CONFIRM_QUIT census row covers it.
- `_mixins_refresh` and `_mixins_intent` refuse with the Saving copy while the save runs (AC#8); a refused toggle calls `_restore_mixin_enabled`.
- `Tests/UI/test_modal_quit_discard_guards.py`'s bare `__new__` modal helper now sets `_mixin_save_worker = None`, the one attribute `confirm_quit` gained.

**Test (AC#3).** `Tests/UI/test_console_prompt_mixins.py::test_press_during_contended_mixin_save_cannot_cancel_it` is the lane-run node TASK-38686 added; it now has 14 cases, so the census needs no edit. In every case a raw connection holds `BEGIN IMMEDIATE` and a spy shows the save has reached its write. The cases:
- `save` and `preview`, from TASK-38686.
- `escape`, `back` and `close`: Discard in a modal opened on Mix-ins.
- `reopen`: Discard from Back in a modal opened on Browse, then Mix-ins reopened.
- `keep`: Discard, then Escape and Keep editing.
- `fail`: the spy raises a version conflict once the lock is released.
- `covered`: Discard, then the quit-anyway question; the save finishes under it.
- `quit`: Ctrl+Q's `confirm_quit`.
- `clean-close` and `clean-browse`: an unchanged editor is saved.
- `reverted-refresh` and `reverted-toggle`: the edit is undone after Save, so the view is clean while the changed name saves. This is the realistic clean-view mid-save case.

Each case:
- clears the status before the press and pins the Saving copy (`quit` checks its notice instead);
- after release, waits for the stored version, then checks the preferences reconcile before any reopen (a reopened projection reconciles by itself);
- requires the next Save to land as version 3 (AC#2). `fail` keeps the editor, version 1, the old preference name and its copy; a retried Save then lands.

**Red/green.**
- At base 29592ecc9b's modal source, the first five new cases failed on "the leave did not wait for the save".
- With that assert and the Saving assert removed, and Mix-ins reopened while the write still waited, `escape`/`back`/`close`/`reopen` failed 3 of 3 runs on `assert 'Style' == 'Style saved'`, the actual cancel.
- Review fixes, each case red with only its own line removed (temporary edits, `-n 2`):

| Line removed | Case | Red on |
|---|---|---|
| the SUCCESS/True check | `fail` | "the failure copy" |
| the `_request_close` deferral | `clean-close` | "the press left Mix-ins while the save ran" |
| the `_mixins_browse` deferral | `clean-browse` | "the press left Mix-ins while the save ran" |
| the on-top check | `covered` | "the leave did not run after the save" |
| the `confirm_quit` refusal | `quit` | "Ctrl+Q asked instead of refusing" |
| the Refresh guard | `reverted-refresh` | `'Saving' in ''` |
| the toggle guard | `reverted-toggle` | `'Saving' in ''` |

- With the Saving assert also removed, the save's own outcome is lost: `'Earlier Library edit saved' in ''` for Refresh, and the toggle applied (`False is True`) for the toggle.
- The Keep-editing line removed earlier: `[keep]` red. `_toggle` already reverts the control itself, so the refused toggle's `_restore_mixin_enabled` has no separate red.
- Green: 14 of 14 cases, 5 of 5 runs (`-n 4`, load 14-17).

**Covering runs** (`-n 4`, scratch wrapper): the prompts modal, mix-ins, controller, workbench contract, modal dismissal and quit-guard files, `test_close_under_quit_question.py`, `test_app_quit_in_flight_modals.py`, the UI-ready census, the Library mix-ins file, the Console prompt draft workbench and the Chat mix-in files. Result: 30 failed, 831 passed.
- 4 of the failures were `test_modal_quit_discard_guards.py` nodes for `ConsolePromptsModal`; they were fixed by the helper edit above.
- 6 were load flakes (literal history restore ×3, the system-replacement chip, stream sync, composer impersonate); they passed in a rerun (35 passed with the quit-guard file).
- The other 20 are the set that is red with base sources: `test_guarded_modal_quit_census` names five other modals (TASK-38747), and the rest are Console/Improve UI tests that are red locally on dev.

**Docs.** `Docs/User_Guide/console.md` (Prompt mix-ins):
- while an edited mix-in saves, Back, Close, Escape and Browse open the "Unsaved changes would be lost" check, and only its Discard waits;
- a failed save keeps the edit and drops the Discard;
- an unchanged editor's leave waits without asking;
- Keep editing cancels a waiting Discard; Ctrl+Q waits too.

**Files.** `tldw_chatbook/Widgets/Console/console_prompts_modal.py`, `Tests/UI/test_console_prompt_mixins.py`, `Tests/UI/test_modal_quit_discard_guards.py`, `Docs/User_Guide/console.md`.
