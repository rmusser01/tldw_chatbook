# Discover older local character conversations in Roleplay

- **Date:** 2026-08-27
- **Status:** Approved
- **Task:** [TASK-22453](../../../backlog/tasks/task-22453%20-%20Make-older-local-character-conversations-discoverable-in-Roleplay.md)
- **Predecessor:** [Resume prior character chats from Roleplay](2026-08-26-roleplay-resume-prior-character-chat-design.md)
- **Visitor mode:** Operate
- **ADR required:** No
- **ADR path:** N/A
- **ADR reason:** This is a routine extension of the existing local Roleplay discovery query and inspector list. It does not change storage, conversation ownership, resume ownership, synchronization policy, or a service boundary.

## Problem

Roleplay lists only the 20 most recently modified saved conversations for a selected
local character. A user with a longer history must leave Roleplay and search Library
to find an older chat, even though Roleplay already owns local character conversation
discovery and the read-only preview.

The current listing path also converts database failures into an empty result. Any
pagination design built directly on that behavior would mislabel an unavailable list
as **No saved conversations** and could not offer an honest retry state.

## Goals

1. Let users browse older saved conversations without leaving Roleplay.
2. Preserve a stable, duplicate-free traversal order for conversations whose
   ordering keys remain unchanged during the active browse session.
3. Route every discovered row through the existing read-only preview and its
   **Resume chat**, **Send to Console draft**, and **Open in Library** actions.
4. Make initial loading, append loading, empty, exhausted, and retryable failure
   states explicit and keyboard accessible.
5. Keep the feature local-only, bounded, and independent of transcript loading or
   provider readiness.

## Non-goals

- Server-backed character conversations.
- Search, filtering, total counts, or a configurable page size.
- Automatically reconciling conversation changes while the list is open.
- Prefetching every conversation or persisting pagination state.
- Adding a schema migration or speculative pagination index.
- Changing read-only preview, Resume, Console draft handoff, or Library navigation
  behavior.
- Adding a shortcut or changing global keybinding conventions.

## User experience

### Initial list

Selecting a local character resets the prior character's pagination state and shows
one readable **Loading conversations...** status row. The first successful read shows
up to 20 conversations in the existing inspector list.

- Zero results show **No saved conversations.**
- One through 20 results with nothing older show **All conversations shown.**
- When older results exist, the final selectable row is **Load 20 older
  conversations**.

The existing list remains capped at ten visible rows and scrolls normally. The tail
row is part of the ListView's keyboard order; Enter activates Load without a new
screen binding.

### Loading older conversations

Activating Load keeps every existing row, the current preview, and its actions in
place. The tail row changes to **Loading older conversations...** and remains
highlighted while the ListView owns focus, but is inert, so ListView does not move
selection unexpectedly and repeated Enter cannot start a second read.

On success, new conversations append below the existing rows in their returned order.
Only the tail row is replaced; existing conversation widgets are not rebuilt. The
ListView highlight moves to the first newly appended conversation only when the
ListView still owns focus and its highlighted child is the exact Load/Loading tail.
If the user moved into the preview, another control, or another conversation row while
the read was running, completion changes neither focus nor highlight.

### Failure and exhaustion

An initial failure renders a two-line selectable tail state:

```text
Load failed.
Retry conversations
```

An additional-page failure preserves all existing rows and the cursor, then renders:

```text
Load failed.
Retry older conversations
```

The special row wraps rather than clipping the recovery action in the narrow
inspector. Enter retries the exact failed boundary. Inline recovery owns the error;
the flow does not also emit a duplicate toast.

When a successful page proves there is nothing older, the tail becomes the readable,
non-action status **All conversations shown.** Empty and exhausted states are
distinct.

### Preview and action parity

Every appended conversation is an ordinary conversation row. Selecting it posts the
same `ConversationRowSelected` message as an initial row, opens the same read-only
preview, and exposes the same actions. Load and Retry use dedicated inspector messages
and never masquerade as conversation IDs, so they cannot collide with a durable ID or
enter the preview/Resume path.

## Architecture and ownership

Roleplay continues to own local character conversation discovery and the read-only
preview. Console remains the sole writable owner of restored live sessions, consistent
with [ADR-026](../../../backlog/decisions/026-retire-chat-tab-conversation-entry-chain.md).
The persisted conversation remains authoritative for resumed behavior, consistent
with [ADR-046](../../../backlog/decisions/046-roleplay-chat-display-identity-and-template-provenance.md).

`PersonasConversationsController` owns one memory-only browse session:

- selected local character ID;
- ordered loaded conversation records and their ID set;
- next seek cursor;
- whether another page exists;
- current list phase; and
- exact request-attempt token.

`PersonasInspectorPane` remains presentation-only. It renders/appends conversation
rows, replaces the trailing state row, exposes dedicated Load/Retry messages, and
performs the conditional highlight move requested by the controller.

The pagination path extends and calls
`CharactersRAGDB.get_conversations_for_character()` directly with an optional seek
cursor and deterministic ordering. That database seam propagates failures. It must not
consume the legacy `list_character_conversations()` fallback that turns exceptions
into an empty list. A failed read reaches the controller as failure, not as valid
empty data; existing offset callers retain their current behavior. Seek mode and a
nonzero legacy offset are mutually exclusive and fail fast if combined. The seek
cursor is exposed as two keyword-only values after the existing positional parameters;
callers supply both `last_modified` and ID or neither. A partial cursor fails before
SQL executes.

The existing character-ID index remains the starting point. This task does not add an
index or bump the schema without measured evidence that realistic long histories need
one; any such finding is filed as separate database work.

## Stable seek pagination

The page size is the existing 20 conversations. Each read requests 21 records; the
21st is a sentinel proving another page exists and is not consumed. The first read has
no cursor. Later reads use the `(last_modified, id)` values from the last accepted
visible row.

The database order is total and deterministic:

```sql
ORDER BY last_modified DESC, id DESC
```

The next-page predicate is strict:

```sql
last_modified < :cursor_last_modified
OR (last_modified = :cursor_last_modified AND id < :cursor_id)
```

The existing schema guarantees a non-null `last_modified`; conversation ID completes
the ordering when timestamps tie. This seek boundary avoids offset shifts: newer
inserts or updates do not push an already-traversed row into a later page, and deletion
of an earlier row does not pull a later row across a numeric offset. The controller
also rejects any ID it has already accepted as a fail-safe against malformed results.

Loaded rows never reorder during one browse session. The no-skip/no-repeat guarantee
applies to conversations whose `(last_modified, id)` ordering key remains unchanged
while that session is active. Conversations created or whose ordering key is modified
after browsing begins take effect after the user reselects the character; Roleplay
does not attempt live reconciliation. A durable row deleted before a later page read
may disappear immediately, while the remaining unchanged older rows continue from the
seek boundary.

## Concurrency and stale-result ownership

Only one page read may be active. Each worker captures the selected character ID, the
seek cursor, and a unique attempt token. A UI continuation applies only when:

1. Roleplay is still mounted in Characters mode;
2. the same local character remains selected;
3. the cursor still matches the requested boundary; and
4. the controller still owns the exact attempt token.

The token check is required even with an exclusive worker group because work already
handed to a thread cannot be assumed cancelled. A character switch, mode switch,
reset, retry, or newer request invalidates the prior attempt before any late
continuation can mutate the visible list.

## Data flow

```text
Select local character
  -> reset browse session
  -> show initial loading row
  -> read 21 records, newest first
       -> failure: initial Retry row
       -> empty: No saved conversations
       -> <=20: rows + All conversations shown
       -> 21: first 20 rows + Load 20 older conversations

Load 20 older conversations
  -> capture character + seek cursor + attempt token
  -> tail becomes Loading older conversations
  -> read 21 records strictly older than cursor
       -> stale completion: ignore
       -> failure: preserve rows/cursor + Retry older conversations
       -> success: append up to 20 new rows
            -> sentinel present: replace tail with Load
            -> no sentinel: replace tail with All conversations shown
            -> highlight first new row only if focused ListView still highlights tail

Select any conversation row
  -> existing read-only preview
  -> existing Resume / draft / Library actions
```

## Accessibility and layout

- Load and Retry are text-labeled, highlightable and keyboard-selectable ListView rows
  activated by Enter.
- Loading, empty, failure, and exhausted states remain observable as text; color is
  never their only signal.
- Long failure/recovery copy uses a dedicated wrapping tail-row style so **Retry** is
  not clipped at supported narrow widths.
- The busy row remains highlighted while the ListView owns focus but ignores
  activation, preventing surprise selection movement.
- Async completion never steals focus from another region.
- No terminal-convention or global shortcut is added.
- Existing conversation-title ellipsis and ten-row inspector scrolling remain
  unchanged.

## Error handling

- Database exceptions are logged with context but rendered as generic recovery copy;
  raw exception text does not enter the UI.
- Initial failure does not claim the character has no conversations.
- Append failure does not clear rows, change the cursor, close the preview, or alter
  preview actions.
- Retry replaces the attempt token but reuses the failed seek boundary.
- Preview loading and Resume failures retain their existing, separately owned behavior.

## Verification strategy

Only targeted verification is required unless a full sweep is separately requested.

### Database tests

- First and subsequent pages use `last_modified DESC, id DESC`.
- Equal timestamps are ordered by ID and split across a seek boundary without overlap.
- The 21st record acts only as a sentinel and becomes the first record of the next
  page.
- A newer insertion between reads does not duplicate or displace older traversal.
- Deleting an earlier row between reads does not skip a remaining older row.
- Character, global-scope, and non-deleted filters remain enforced.
- Read failures remain distinguishable from a successful empty page.
- Existing positional offset calls remain compatible; partial cursors and a cursor
  combined with nonzero offset fail before querying.

### Controller and Pilot tests

- Initial loading, empty, Load, append-loading, Retry, and exhausted states render.
- Enter activates Load and Retry; repeated Enter during loading dispatches once.
- Retry preserves existing rows and the seek cursor.
- Stale attempts after character or mode changes cannot append.
- Appending does not rebuild existing rows or change the open preview.
- Highlight advances only while the focused ListView still highlights the exact tail.
- A discovered older row opens the ordinary preview and exposes Resume, Send to
  Console draft, and Open in Library.
- Production stylesheet checks at narrow and standard sizes prove failure copy wraps
  and the Load/Retry action remains readable.

### Isolated live acceptance

Use a scratch profile and database seeded with more than 40 conversations for one
local character. Traverse both additional batches with the keyboard, confirm the
exhausted state, open the oldest row, and verify its preview actions. Fingerprint the
working tree and generated stylesheet before and after the launch so runtime rebuilds
cannot enter the implementation diff unnoticed.

## Alternatives considered

### Offset pagination

Rejected after design review. A deterministic tie-breaker fixes equal timestamps but
does not prevent insertion, deletion, or update between reads from shifting a later
numeric offset and causing a duplicate or omission.

### Automatic loading at the bottom

Rejected because it hides the loading trigger, complicates keyboard focus, and makes
retry recovery less explicit.

### Numbered Previous/Next pages

Rejected because replacing the visible page interrupts continuous browsing and makes
it harder to compare or revisit conversations while keeping a preview open.

### Materialize every conversation ID up front

Rejected because it front-loads unbounded work and invents a snapshot mechanism when
bounded seek reads satisfy the local workflow.

## Acceptance-criteria mapping

1. The explicit Load tail appends further batches without leaving Roleplay.
2. Composite seek ordering and stale-attempt ownership prevent overlap, offset shifts,
   and late-result corruption for unchanged conversations. New or reordered rows take
   effect after character reselection; deletions may be reflected by the next read.
3. Appended rows reuse `ConversationRowSelected` and the existing preview action path.
4. Every loading, empty, exhausted, and failure state has explicit text; Load and
   Retry are keyboard actions with bounded, recoverable behavior.
