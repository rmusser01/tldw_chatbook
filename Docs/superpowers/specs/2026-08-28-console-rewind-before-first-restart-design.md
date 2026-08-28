# Console `/rewind` before-first restart durability — Design

**Date:** 2026-08-28

**Status:** Reviewer-approved; awaiting user written-spec approval

**Task:** TASK-574

**ADR:** [ADR-100](../../../backlog/decisions/100-console-active-path-before-first-cursor.md)

## Purpose

Make `/rewind` to the first prompt survive an application restart. The restored
conversation must reopen immediately before that prompt with an empty active path,
retain every message branch, and refill the composer with the selected prompt's
original text. Conversations whose cursor was never set must keep today's
most-recent-leaf fallback.

## Approved behavior

1. Rewinding a mid-path prompt is unchanged: its predecessor becomes the active
   leaf and the prompt text replaces the current composer draft.
2. Rewinding the active path's first prompt records a durable
   "before this root prompt" cursor, leaves the active path empty, and puts the
   full original prompt into the composer.
3. Closing and restarting the app reconstructs that empty path and original
   composer text from durable message state.
4. Edits or clearing performed after the composer is reconstructed remain
   session-only. If the app closes before send, the next restart reconstructs the
   original prompt again.
5. Sending the restored or edited text creates the normal new root branch. Once
   its durable message becomes the active leaf, the before-first marker is cleared.
6. Existing branches are never deleted. The old root branch remains in the full
   tree and is reachable through the existing branch-navigation behavior after a
   new root is created.

## Durable representation

Schema v54 adds nullable `conversations.active_leaf_before_message_id TEXT`.
Together with `active_leaf_message_id`, it forms one local tri-state cursor:

| Active leaf | Before message | Meaning on resume |
| --- | --- | --- |
| message ID | `NULL` | Restore that branch leaf |
| `NULL` | `NULL` | Unset/legacy; fall back to newest leaf |
| `NULL` | root user message ID | Restore immediately before that prompt |

Both non-null is non-canonical. The active leaf wins if valid and the companion
marker is cleared. A dangling active leaf retains the existing newest-leaf
fallback; the marker must not rescue or override it.

The companion column intentionally stores an ID instead of prompt text. It
distinguishes multiple root branches, reuses the message row as the content owner,
and avoids a second durable copy of potentially sensitive user text.

## Persistence API

The database layer exposes an atomic local cursor read/write over both columns.
The existing active-leaf setter remains source-compatible and delegates to the
atomic writer with `before_message_id=None`. The existing scalar
`get_conversation_active_leaf()` contract also remains source-compatible; the
conversation hydration path opts into a new two-component cursor reader. A
dedicated before-message writer sets the leaf to `NULL` and the prompt ID in the
same transaction.

These updates remain bare local writes:

- no `version` or `last_modified` bump;
- no Sync trigger or `sync_log` row;
- no field in conversation Sync payloads;
- no optimistic locking, because this remains a per-client view cursor.

The v53→v54 migration adds only the nullable column and a guarded version update.
It follows the existing interrupted-migration-safe column-presence check. No data
backfill is needed: every existing row has `NULL`, which preserves unset fallback.

## Store and resume flow

`ConsoleChatStore` gains a dedicated `set_active_path_before(session_id,
message_id)` operation. It verifies that the target is a root `USER` node, clears
the in-memory active leaf, recomputes the empty active path, and writes both durable
cursor components. Existing `set_active_leaf` continues to select a real node or
perform a generic unset; it never implicitly invents a before-message marker.

Conversation hydration reads both cursor components and passes both persisted IDs
to `restore_persisted_session`. Full-tree ingestion maps them to the new store's
native IDs after rebuilding every node:

- valid active leaf: restore its branch and clear a contradictory companion;
- no pointers: use the current newest-leaf fallback and repair the leaf;
- valid before-message root user: retain `active_leaf=None`, materialize an empty
  path, and set `session.draft` to that node's original content;
- invalid companion: use the newest-leaf fallback and repair both columns.

If the tree is empty, invalid or unset state resolves to an empty session without
inventing a draft. Durable write failure remains best-effort and logged, matching
the store's established local pointer convention.

When a newly persisted message is the in-memory active leaf, the existing
post-persistence pointer write becomes an atomic cursor write that also clears the
before-message marker. The same rule applies to branch selection, sibling creation,
and any other durable active-leaf advance.

## UI integration

`ChatScreen._apply_console_rewind_choice` keeps its existing ID-based path lookup.
For a first-path prompt it calls the dedicated before-message operation; otherwise
it calls `set_active_leaf` with the predecessor. The current
`_insert_prompt_text_into_composer(..., replace=True)` behavior remains responsible
for immediate refill in the running screen.

After restart, the normal session-draft synchronization loads the draft hydrated by
the store. No second UI-only pending handoff or draft database is introduced.

## Invalid-state and compatibility policy

- A before-message ID must map to a root `USER` node. Anything else is treated as
  corrupt/dangling local cursor state.
- A valid active leaf is authoritative if both columns are populated.
- Invalid state falls back to the newest leaf and is repaired immediately when the
  persistence seam is available.
- Legacy databases migrate with a null companion, so their behavior is byte-for-
  byte equivalent at the cursor boundary.
- Existing callers and tests that use `set_active_leaf(..., None)` continue to mean
  genuinely unset; this is required to prove acceptance criterion 2.

## Portability boundary

The companion marker is local restart state only. This task does not change Sync,
trajectory export/import, Chatbook export/import, or fork snapshot schemas. Those
formats continue to carry their existing active-leaf representation; an exported
explicit-before-first conversation therefore imports with an unset pointer and
uses the newest-leaf fallback. Portable before-first state can be designed later if
there is a demonstrated user need.

## Verification

Tests will be written before implementation and cover:

1. v53→v54 migration, null defaults, idempotent column guarding, and schema version.
2. Atomic database read/write for selected, unset, and before-first cursor states.
3. No version, `last_modified`, or `sync_log` change from either local cursor field.
4. Store validation of root-user targets and canonical clearing on later leaf writes.
5. Full-tree resume for all three states, including original composer hydration.
6. Invalid, dangling, contradictory, non-root, and non-user state fallback/repair.
7. Screen routing: first prompt uses before-message; mid-path rewind remains unchanged.
8. End-to-end persist/drop/resume: empty active path, original prompt draft, old tree
   retained, edited resend creates a root sibling, and a following restart selects
   the new branch with no stale marker.

The current `dev` baseline has one reproducible test-harness failure in
`Tests/integration/test_console_rewind_e2e.py`: its raw store/session fixture does
not hydrate durable Console Library-policy authority, so the first turn creates a
policy row and the second turn is refused with `authority.source == "unavailable"`.
With user approval, this PR will add the minimal fixture hydration needed to make
the pre-existing integration scenario exercise current production policy gates.
No product policy behavior changes.

## Alternatives rejected

- **Sentinel in `active_leaf_message_id`:** collides with arbitrary string IDs and
  forces parsing into unrelated consumers.
- **Boolean/enum plus prompt ID:** redundant state with more invalid combinations.
- **Persist full composer drafts:** expands the task into autosave, retention,
  conflict, and privacy policy not required by the restart bug.
- **Export the new marker:** broadens portable formats and copy semantics beyond
  this local restart fix.

## Scope limits

No general composer autosave; no Sync or remote state; no export/import schema
change; no new branch browser; no deletion or pruning; no changes to mid-path
rewind, summarization, provider payloads, or fork semantics.
