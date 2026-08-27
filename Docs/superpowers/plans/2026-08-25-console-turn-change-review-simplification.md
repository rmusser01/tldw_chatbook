# TASK-22305 — Console per-turn change review simplification plan

> Execute in the isolated `codex/console-turn-change-review-simplification`
> worktree. Preserve per-turn changed-file cards and the full Change Review
> screen while deleting the retired Inspector projection.

ADR required: yes
ADR path: `backlog/decisions/089-console-per-turn-change-review-ownership.md`
Reason: this changes long-lived Console ownership and removes an Inspector
information architecture surface in favor of turn-owned review.

## 1. Pin the approved card contract red-first

Modify:

- `Tests/UI/test_console_turn_file_card.py`
- `Tests/UI/test_console_turn_file_card_factory.py`

Add assertions that a loaded change card exposes **Undo All** beside **Review**,
starts disabled while rows are unresolved, prevents a second press while busy,
can be marked **Undone** without losing file rows/Review, and leaves live/resume
factory selection unchanged. Run the focused tests and preserve the RED output.

## 2. Pin safe turn-level planning and execution red-first

Add a focused test module for the screen-owned orchestration. Cover:

- one snapshot row;
- one row per root in a multi-root turn;
- duplicate canonical root rows refusing before preflight/revert and requesting
  Review for the exact run;
- edited-since warning labels that identify the root in multi-root turns;
- no provider, tracking error/pruned/no-file turns;
- confirmation cancellation;
- active-run refusal;
- full success, partial failure, and raised-provider error;
- all provider/git work occurring outside the UI thread; and
- duplicate dispatch suppression.

Use the existing provider and `ChangeRevertConfirmModal` contracts. Do not add a
second filesystem restore engine.

## 3. Implement card and screen orchestration

Modify:

- `tldw_chatbook/Widgets/Console/console_turn_file_card.py`
- `tldw_chatbook/UI/Screens/chat_screen.py`

Add `UndoAllRequested`, a compact card button, and minimal card state methods.
The screen captures the provider for the card's conversation, prepares the
turn with `asyncio.to_thread`, pushes the existing confirmation modal, and runs
reverts off-thread after confirmation. Keep mutation-time active-run refusal
authoritative. Refuse duplicate-root rows before preflight or mutation, notify
plainly, and open Review for that run. On full success mark the card **Undone**;
otherwise restore **Undo All** for retry and report named failures.

## 4. Retire Inspector ownership and sync work

Modify:

- `tldw_chatbook/UI/Console_Modules/right_rail.py`
- `tldw_chatbook/UI/Screens/chat_screen.py`
- `Tests/UI/test_console_right_rail.py`
- `Tests/UI/test_console_rail_reconciliation.py`
- `Tests/Architecture/test_console_wave6_closeout_inventory.py`
- other tests whose shared constructors still pass `changed_files_state`

Remove the changed-files constructor state, composition block, screen caches,
scope guard, background worker, sync calls, rail click handler, note-driven
invalidation, and Review dismissal callback. Assert the Inspector no longer
mounts the section and a Console sync does not dispatch a changed-files worker.

## 5. Delete rail-only domain and store code

Delete or modify:

- `tldw_chatbook/Widgets/Console/console_changed_files_section.py`
- `tldw_chatbook/Widgets/Console/__init__.py`
- `tldw_chatbook/Chat/console_display_state.py`
- `tldw_chatbook/UI/Screens/change_review_screen.py`
- `tldw_chatbook/Chat/console_chat_store.py`
- `Tests/UI/test_console_changed_files_section.py`
- `Tests/UI/test_console_changed_files_wiring.py`
- `Tests/UI/test_console_changed_files_scope_memo.py`
- `Tests/Chat/test_console_conversation_files.py`

Remove only symbols whose production owner was the retired Inspector
projection: `ConversationFileEntry`, `conversation_file_summary`, provider
`conversation_changed_files`, and store `newest_change_review_run_id` plus its
memo/drop hooks. Preserve snapshot capture, per-turn provider methods, card
notes, Review comments, diff rendering, and revert engines.

## 6. Reconcile generated assets and documentation

Modify:

- `Docs/User_Guide/console/agent-runs-and-tools.md`
- generated Console/widget CSS outputs
- diagnostic inventories that still name the removed widget

Remove `[console] changed_files_section` guidance and describe the turn card's
**Undo All** behavior, edited-since confirmation, and duplicate-root Review
fallback. Rebuild CSS with the repository generator; do not hand-edit generated
styles. Update inventories through their owning generator/check where available.

## 7. Verify behavior and presentation

Run focused card/orchestration/Review/right-rail tests, then the complete prior
417-node Console change-review baseline. Run Ruff check/format, CSS integrity,
diagnostic inventory checks, and `git diff --check`.

Use the production consolidated CSS stack to render a card with several files at
representative and narrow Console widths. Verify the header actions are visible,
focusable, and do not obscure counts; expand a file; cancel Undo All; complete a
safe temporary-root undo; and verify Inspector has no Changed Files section.

## 8. Self-review and closeout

Review the branch for destructive-action safety, stale async callbacks,
cross-conversation retargeting, UI-thread I/O, dead rail code, documentation
drift, and unrelated changes. Record exact verification evidence, check every
acceptance criterion, add concise Implementation Notes, and mark TASK-22305 Done
only if every Definition-of-Done requirement is satisfied.
