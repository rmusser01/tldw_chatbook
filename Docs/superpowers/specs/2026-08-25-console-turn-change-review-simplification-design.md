# Console per-turn change review simplification — Design

Date: 2026-08-25
Status: Approved
Task: TASK-22305
ADR: ADR-089

## Problem

Console currently presents the same file-change history twice:

1. the agent turn already owns a changed-file card with file rows, expandable
   diffs, notes, and a direct route to the full Change Review screen; and
2. Inspector independently builds a cross-turn latest-file projection.

The Inspector projection is costly for the value it adds. It owns a store memo,
screen guard, per-row git cache, background worker, note-invalidation paths,
rail widget, configuration switch, and substantial reconciliation/test surface.
It also compresses distinct turn windows into a latest-per-path view, which can
make a note badge and its click-through refer to different snapshot windows.

Users still need file-change review. The simplification must remove the
duplicate ownership, not the capability.

## Core purpose

Make it fast and obvious to review or undo the files changed by one agent turn.

The turn is the natural owner: it is where attribution is meaningful, where the
user encounters the result, and where the existing snapshot run id already
provides an exact route to detailed Review.

## Decision

### Keep the per-turn card and full Review screen

Every live or resumed change marker continues to render
`ConsoleTurnFileCard` when turn-file cards are enabled. The card retains:

- the `Edited N files +A −D` summary;
- one row per changed file;
- expandable, bounded diffs;
- hunk notes;
- expand/collapse all; and
- the per-turn **Review** action.

The full Change Review screen remains the authoritative surface for browsing
turn history, viewing complete diffs, reverting one file, commenting, and git
actions. The existing selected-row `v` action and run-Inspector Review action
remain available.

### Add direct Undo All to the turn card

The card header adds a visible **Undo All** button beside **Review**. It is a
shortcut to the existing guarded revert engine, not a second revert
implementation.

The interaction is:

1. The button is disabled until the card's snapshot rows have loaded.
2. A press disables repeat dispatch and resolves the card's run through the
   same run-scoped provider used by the card and Review opener.
3. Snapshot reads, changed-file enumeration, edited-since preflight, and revert
   operations run off the Textual UI thread.
4. The existing `ChangeRevertConfirmModal` names any files whose current disk
   state differs from the turn's end snapshot. No mutation occurs before the
   user confirms.
5. The existing engine rechecks active-run refusal at mutation time and returns
   per-path outcomes. The card reports named failures; full success changes the
   button to **Undone** while leaving the historical card reviewable.
6. Cancellation or failure restores the button so the user can retry.

For multi-root turns, edited-since warning labels include the root name so two
equal relative paths are distinguishable.

### Refuse ambiguous same-root multi-window turns inline

A run can contain multiple snapshot windows for the same canonical root (for
example, the turn window plus a surviving sub-agent post-turn window). Applying
those windows sequentially is order-sensitive and the compact card deliberately
does not expose their baseline relationship.

If a turn contains more than one snapshot row for the same root, card-level
**Undo All** refuses before preflight or mutation, explains that the run has
multiple change windows, and opens that run in the full Review screen. Ordinary
multi-root turns with exactly one row per root remain supported.

### Remove the cross-turn Inspector projection completely

Inspector no longer mounts a Changed Files section. Remove the rail-only:

- `ConsoleChangedFilesSection` widget and generated CSS;
- `ConsoleChangedFilesState`, `ConversationFileEntry`, and
  `conversation_file_summary` projection;
- `AgentRunsChangeReviewProvider.conversation_changed_files` aggregation;
- `ChatScreen` guard, worker, result cache, row cache, invalidation handlers,
  config gate, and sync calls;
- `ConsoleChatStore.newest_change_review_run_id` memo used only by that guard;
- `[console] changed_files_section` documentation; and
- tests whose only contract is the retired projection.

Review comments and hunk notes remain. Removing the cross-turn badge removes the
need to recompute note counts after card edits or Review dismissal, so the card's
rail-specific `NotesChanged` event and Review dismissal callback are retired.

## Failure and concurrency posture

- No provider, missing/pruned snapshots, malformed rows, preflight failure, or
  active-run refusal touches disk.
- A second card-level undo cannot dispatch while one is already in flight.
- The provider is captured for the card's conversation before asynchronous work;
  a later session switch cannot retarget the action to another conversation.
- The mutation-time engine remains authoritative for active-run refusal because
  run state can change after preflight.
- Partial failures are named and keep **Undo All** available for retry.
- A full success is historical state, not deletion: the changed-file card and
  Review history remain visible.

## Visual hierarchy

The card header has one informational summary and three compact actions:

`Edited N files +A −D    ▸ All    Undo All    Review`

`Review` is the primary detailed path. `Undo All` is explicit text, visually
secondary, and confirmation-gated. `▸ All` remains a presentation-only control.
No new icon language, nested card, or modal family is introduced.

## Testing and verification

- Pure planning tests cover one row, ordinary multi-root rows, duplicate-root
  refusal, edited-since root labels, tracking errors, and no-change turns.
- Mounted card tests cover button presence, disabled/loading state, duplicate
  press suppression, cancellation, success/Undone state, partial failure, and
  Review routing for ambiguous runs.
- Integration tests prove live and resumed markers still create cards and the
  Review action still opens the exact run.
- Inspector tests assert that no Changed Files section is mounted and that no
  cross-turn worker dispatches during sync.
- A production-CSS-stack render verifies header fit and action reachability at
  representative and narrow Console widths.

## Alternatives rejected

| Alternative | Reason |
| --- | --- |
| Keep both surfaces and optimize the worker again | Retains duplicate ownership and snapshot-window ambiguity for a secondary view. |
| Remove all changed-file UI | Breaks the core user need to inspect an agent turn's disk effects. |
| Put Undo All only in Review | Safer but misses the requested direct turn-level recovery path. |
| Inline-undo duplicate-root windows in row order | Order-sensitive and not explainable from the compact card. |
| Add a new revert service/controller abstraction | The existing provider and revert engine already supply the required domain seams. |
