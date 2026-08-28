# ADR-100: Represent the Console cursor before the first prompt explicitly

Status: Accepted

Date: 2026-08-28

Related Task: [TASK-574](../tasks/task-574%20-%20Console-rewind-restore-to-before-first-message-does-not-survive-restart.md)

Related Spec: [Console rewind restart durability](../../Docs/superpowers/specs/2026-08-28-console-rewind-before-first-restart-design.md)

## Context

Console conversations persist the selected branch in the local-only nullable
`conversations.active_leaf_message_id` column. On resume, a stored message ID
selects that leaf, while `NULL` means no cursor has ever been stored and falls
back to the most-recent leaf.

`/rewind` can also position the conversation immediately before its first user
prompt. That position has no active leaf, so the original implementation writes
`NULL`. Restart then misreads a deliberate empty path as an unset cursor and
silently returns to the newest leaf. The running session behaves correctly and
the complete message tree remains durable; only the cursor meaning is lost.

The user also expects restart to refill the composer with the original selected
prompt. A conversation may contain multiple root user branches, so a boolean
"explicitly empty" flag cannot identify which prompt supplies that text.

## Decision

1. Add nullable local-only `conversations.active_leaf_before_message_id`. It
   stores the persisted root user message immediately after an explicitly empty
   cursor. It deliberately has no foreign key, matching the existing active-leaf
   pointer's fail-open handling of dangling local state.
2. Interpret the two nullable cursor columns as one three-state value:
   - `active_leaf_message_id != NULL`, `active_leaf_before_message_id = NULL`:
     select the stored active leaf.
   - both columns `NULL`: the cursor is genuinely unset; resume uses and repairs
     the existing most-recent-leaf fallback.
   - `active_leaf_message_id = NULL`,
     `active_leaf_before_message_id != NULL`: resume immediately before that root
     user message, with an empty active path.
3. Persist both columns atomically through one local cursor write. Selecting a
   leaf clears the before-message marker. Explicitly positioning before a root
   prompt clears the leaf and stores that prompt. A generic clear leaves both
   columns `NULL` and therefore retains the historical unset semantics.
   Preserve the existing scalar active-leaf getter for compatibility; only
   cursor-aware hydration opts into the two-component read.
   The existing-conversation acceptance SQL in
   `ConsoleDispatchRepository.insert_with_messages` must also clear the companion
   marker in the same transaction that accepts the new active leaf; a later
   best-effort pointer write is not the correctness boundary.
4. Expose a dedicated store operation for positioning before a message instead
   of changing the meaning of every existing `set_active_leaf(..., None)` call.
   The operation accepts only a root user node in the session tree. Root status is
   determined by the persisted `parent_message_id`, captured before any
   `_chain_legacy_flat_roots` compatibility re-parenting.
5. On resume, a non-null active-leaf pointer takes precedence. If it resolves,
   any contradictory before-message marker is cleared. If it dangles, resume
   uses the existing newest-leaf fallback and repairs both columns.
6. When the active leaf is null and the before-message pointer resolves to a
   root user node, keep the active path empty and load that node's original text
   into the restored session's in-memory composer draft through
   `set_session_draft()`. This marks the hydrated composer as user work so normal
   hydration safeguards do not overwrite it. Unsent edits after hydration remain
   session-only; another restart restores the original message text again.
7. A dangling, non-user, or non-root before-message pointer is invalid local
   state. Resume falls back to the newest leaf and atomically repairs the cursor.
8. The migration and cursor writes do not bump conversation version or
   `last_modified`, do not enter Sync payloads, and do not emit `sync_log` rows.
9. Trajectory and Chatbook import/export formats, remote sync, and fork-copy
   contracts do not gain this marker. Moving a conversation through those
   boundaries treats the explicit-before-first cursor as unset on the target.

## Consequences

- Restart can distinguish a deliberate empty active path from legacy or never-set
  state without reserving values in the arbitrary string message-ID namespace.
- The stored prompt ID both identifies the cursor boundary and reconstructs the
  requested composer text without duplicating prompt content in conversation
  metadata.
- Cursor mutations must keep two columns canonical and atomic. Resume includes
  defensive repair for partial, contradictory, or dangling local state.
- Ordinary composer drafts remain non-durable. This decision creates no general
  draft persistence, autosave, conflict, sync, or privacy surface.
- Explicit-before-first state is device-local and restart-durable, but not
  portable through exports or conversation-copy formats.
- While the active path is empty, the current `/rewind` and swipe controls have no
  visible message anchor. The old branch remains durable and becomes navigable
  after a new root is sent. An immediate pre-send undo/recovery affordance is a
  separate UX decision, not part of this storage fix.

## Alternatives considered

### Encode a sentinel in `active_leaf_message_id`

Rejected. Message IDs are arbitrary strings, so a reserved sentinel can collide.
Encoding a prompt ID inside the sentinel also makes every active-leaf consumer
parse a second wire format and risks leaking the marker into branch, dispatch,
trajectory, and fork code.

### Add a boolean or enum plus a prompt pointer

Rejected. Composer restoration still requires the prompt ID, so a separate state
column would duplicate information and permit more contradictory combinations.
The companion pointer alone represents the needed third state.

### Persist the full composer draft

Rejected. It would broaden a cursor bug into general draft autosave, including
edit/clear semantics, privacy and retention rules, and conflict behavior. The
durable message already owns the exact original prompt required here.

## Links

- [Approved design](../../Docs/superpowers/specs/2026-08-28-console-rewind-before-first-restart-design.md)
- [TASK-574](../tasks/task-574%20-%20Console-rewind-restore-to-before-first-message-does-not-survive-restart.md)
- [Original `/rewind` design](../../Docs/superpowers/specs/2026-07-24-console-rewind-menu-design.md)
