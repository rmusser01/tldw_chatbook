# Console `/rewind` before-first restart durability — Design

**Date:** 2026-08-28

**Status:** Revised after second code-grounded review; pending independent re-review

**Task:** TASK-574

**ADR:** [ADR-100](../../../backlog/decisions/100-console-active-path-before-first-cursor.md)

## Purpose

Make `/rewind` to the first prompt survive an application restart. The restored
conversation must reopen immediately before that prompt with an empty active path,
retain every message branch, and refill the composer with the selected prompt
row's current durable text. Conversations whose cursor was never set must keep
today's most-recent-leaf fallback.

## Approved behavior

1. Rewinding a mid-path prompt is unchanged: its predecessor becomes the active
   leaf and the prompt text replaces the current composer draft.
2. Rewinding the active path's first prompt records a durable
   "before this root prompt" cursor, leaves the active path empty, and puts the
   full current prompt text into the composer.
3. Closing and restarting the app reconstructs that empty path and the target
   message row's current durable text. This is normally the original prompt text,
   because edit/resend creates a sibling rather than rewriting the selected row;
   the cursor does not store a text snapshot.
4. Edits or clearing performed after the composer is reconstructed remain
   session-only. If the app closes before send, the next restart reconstructs the
   referenced durable prompt text again.
5. Sending the restored or edited text creates the normal new root branch. Once
   its durable message becomes the active leaf, the before-first marker is cleared.
   Canonical modern root-fork trees retain their existing sibling navigation.
6. Existing messages are never deleted or rewritten. While the active path is
   empty there is no visible message to swipe and `/rewind` has no row to offer.
   Legacy flat mixed-role roots keep the existing `_chain_legacy_flat_roots`
   compatibility repair, which may present them as a linear chain after reload;
   this task does not promise a new branch shape for those ambiguous trees.
7. If a persisted conversation rewinds in memory but its durable cursor write
   fails, the rewind stays in effect for the running session and the screen warns
   that the restart position could not be saved.

## Durable representation

The next schema revision adds nullable
`conversations.active_leaf_before_message_id TEXT`. The rebased 2026-08-28 `dev`
base remains at v53, so v54 is available for this migration. Together with
`active_leaf_message_id`, it forms one local tri-state cursor:

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

The database layer exposes a deliberately small atomic local cursor API over both
columns:

```python
get_conversation_active_cursor(
    conversation_id: str,
) -> tuple[str | None, str | None]

set_conversation_active_cursor(
    conversation_id: str,
    *,
    active_leaf_message_id: str | None,
    before_message_id: str | None,
) -> bool
```

The writer returns whether the requested durable row was updated. No cursor
dataclass or enum is introduced.

The existing active-leaf setter remains source-compatible and delegates to the
atomic writer with `before_message_id=None`. The existing scalar
`get_conversation_active_leaf()` contract also remains source-compatible; the
conversation hydration path opts into a new two-component cursor reader. A
dedicated before-message writer sets the leaf to `NULL` and the prompt ID in the
same transaction. Existing best-effort scalar-setter callers may ignore the
writer result.

The existing-conversation acceptance transaction in
`ConsoleDispatchRepository.insert_with_messages` is also a cursor writer. Its
direct `UPDATE conversations` statement must set the accepted leaf and clear
`active_leaf_before_message_id` in that same transaction. The later best-effort
store pointer write is not the correctness boundary for clearing a stale marker.

These updates remain bare local writes:

- no `version` or `last_modified` bump;
- no Sync trigger or `sync_log` row;
- no field in conversation Sync payloads;
- no optimistic locking, because this remains a per-client view cursor.

The v53→v54 migration adds only the nullable column and a guarded
version update. It follows the existing interrupted-migration-safe
column-presence check. No data backfill is needed: every existing row has `NULL`,
which preserves unset fallback.

## Store and resume flow

`ConsoleChatStore` gains a dedicated `set_active_path_before(session_id,
message_id) -> bool` operation. It verifies that the target is a root `USER` node,
clears the in-memory active leaf, recomputes the empty active path, and writes both
durable cursor components. Existing `set_active_leaf` continues to select a real
node or perform a generic unset; it never implicitly invents a before-message
marker.

For this validation, "root" means `message.parent_message_id is None` on the node
entering `_ingest_full_tree`, before `_chain_legacy_flat_roots` changes the native
parent map. This is the pre-repair imported-tree parent, not necessarily the raw
database column because conversion may transparently remove empty rows and
reparent their children. A node is not rejected merely because the legacy
compatibility repair subsequently gives it a native parent. The screen also
requires that the chosen prompt is index zero in the current active path.

A temporary session or message with no durable identity keeps the existing
in-memory first-prompt rewind behavior: validate its in-memory root-user shape,
clear the active path, and refill the draft, but skip the companion write because
there is no restartable cursor yet. Persisted-root validation applies once both the
conversation and target message have durable IDs. The operation returns `True`
when no durable write is required or when that write succeeds. For a persisted
conversation it returns `False` if the persistence seam is unavailable, the row
is missing, or the write fails; the in-memory rewind is deliberately not rolled
back, and the failure is logged.

Conversation hydration reads both cursor components and passes both persisted IDs
to `restore_persisted_session`. Full-tree ingestion maps them to the new store's
native IDs after rebuilding every node:

- valid active leaf: restore its branch and clear a contradictory companion;
- no pointers: use the current newest-leaf fallback and repair the leaf;
- valid before-message root user: retain `active_leaf=None`, materialize an empty
  path, and call `set_session_draft()` with that node's current durable content
  so the restored composer is represented as user work;
- invalid companion: use the newest-leaf fallback and repair both columns.

If the tree is empty, both-null unset state resolves to an empty session without
inventing a draft. A non-null companion in an empty tree is invalid and triggers
a best-effort clear of both cursor columns before returning that same empty
session.

When a newly persisted message is the in-memory active leaf, durable acceptance
atomically clears the before-message marker with the accepted leaf. The existing
post-persistence pointer write remains a consistent best-effort reinforcement of
that cursor. The same atomic-clear invariant applies to branch selection, sibling
creation, and any other durable active-leaf advance.

## UI integration

`ChatScreen._apply_console_rewind_choice` keeps its existing ID-based path lookup.
For a first-path prompt it calls the dedicated before-message operation; otherwise
it calls `set_active_leaf` with the predecessor. The current
`_insert_prompt_text_into_composer(..., replace=True)` behavior remains responsible
for immediate refill in the running screen.

If the dedicated operation returns `False`, the screen preserves the successful
in-memory rewind and shows: "Rewound for this session, but the restart position
could not be saved." It does not fail closed or restore the old active path.

After restart, the normal session-draft synchronization loads the draft hydrated by
the store. No second UI-only pending handoff or draft database is introduced.

## Invalid-state and compatibility policy

- A before-message ID must map to a root `USER` node in the pre-repair imported
  tree. Anything else is treated as corrupt/dangling local cursor state. Later
  legacy native re-parenting does not invalidate an otherwise eligible target.
- A valid active leaf is authoritative if both columns are populated.
- Invalid state falls back to the newest leaf and is repaired immediately when the
  persistence seam is available.
- Legacy databases migrate with a null companion, so their behavior is byte-for-
  byte equivalent at the cursor boundary.
- Existing callers and tests that use `set_active_leaf(..., None)` continue to mean
  genuinely unset; this is required to prove acceptance criterion 2.
- Restoring the referenced durable prompt text uses `set_session_draft()`,
  preserving `has_user_work` safeguards against later hydration overwriting the
  composer.

## Portability boundary

The companion marker is local restart state only. This task does not change Sync,
trajectory export/import, Chatbook export/import, or fork snapshot schemas. Those
formats continue to carry their existing active-leaf representation; an exported
explicit-before-first conversation therefore imports with an unset pointer and
uses the newest-leaf fallback. Portable before-first state can be designed later if
there is a demonstrated user need.

## Verification

Tests will be written before implementation and cover:

1. The rebased current→next migration, null defaults, idempotent column guarding,
   and schema version.
2. Atomic database read/write for selected, unset, and before-first cursor states.
3. No version, `last_modified`, or `sync_log` change from either local cursor field.
4. Direct durable message acceptance clears a stale before-message marker in the
   same transaction, independently of the later best-effort pointer write.
5. Store validation of persisted root-user targets, including a legacy tree whose
   pre-repair parent is authoritative even when native legacy repair changes it,
   and canonical clearing on later leaf writes.
6. Full-tree resume for all three states, including durable-text composer
   hydration through `set_session_draft()` and `has_user_work=True`.
7. Invalid, dangling, contradictory, non-root, and non-user state fallback/repair,
   including clearing a non-null marker for an empty tree.
8. Screen routing: first prompt uses before-message; mid-path rewind remains unchanged.
9. A failed durable before-message write leaves the in-memory rewind intact and
   produces the restart-durability warning.
10. End-to-end persist/drop/resume: empty active path, the target row's current
    durable prompt text, old tree retained, edited resend creates a root sibling,
    and a following restart selects the new branch with no stale marker.
11. Canonical root-fork trees preserve existing branch recovery after resend;
    legacy flat mixed-role roots preserve every durable row without asserting a
    new branch-navigation shape.
12. Temporary/unpersisted sessions still rewind before their first prompt in memory
    without attempting to persist a companion ID.

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
rewind, summarization, provider payloads, or fork semantics. Composer restoration
is text-only, matching current `/rewind`; attachments from the selected prompt are
not re-staged. The existing legacy flat-root repair is also unchanged.

This task also does not add an immediate undo control for the empty-path state.
Until a root prompt is sent, `/rewind` has no active-path row and swipe navigation
has no visible message anchor. The durable tree remains untouched. Canonical root
forks become navigable through their existing controls after resend; ambiguous
legacy flat trees retain their existing repaired presentation. A pre-send recovery
affordance requires its own UX design.
