---
id: TASK-38748
title: "Console mix-in Discard or reopen during a threaded Library save cancels it after the write lands"
status: To Do
assignee: []
created_date: 2026-11-04 06:34
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
- [ ] #1 A Discard (from Escape, Back or Close), or reopening Mix-ins, while a mix-in save runs never cancels that save: once its write lands, the mix-in preferences are reconciled to the saved record
- [ ] #2 After such a save, the next Save of the same record is accepted; the user is never left on a stale version that the next Save is refused for
- [ ] #3 A test holds the write behind a contended lock (a raw connection in `BEGIN IMMEDIATE`), presses Discard in a modal opened on Mix-ins, and separately reopens Mix-ins from Browse; it is red before the fix
- [ ] #4 If a press is ignored or deferred while the save runs, it shows the same "Saving Library changes…" status #3402 uses
<!-- AC:END -->
