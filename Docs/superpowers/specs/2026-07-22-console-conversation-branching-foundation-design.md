# Console Conversation Branching Foundation — Design

**Date:** 2026-07-22
**Status:** Approved (design) — pending spec review
**Scope:** Sub-project 1 of 2 in the "Console `/rewind`" program. This spec covers only the
branching foundation. `/rewind` (undo + summarize) is a separate follow-up spec that builds on it.

## Why

Users expect the native Console to behave like a conversation *tree*: fork a message, revisit
different nodes, and continue / swipe / retry from a specific node — the way tldw_server models a
conversation. Today the native Console cannot do this. It is a strictly linear thread, and its
"swipe" affordance is a throwaway in-memory feature that silently disappears on resume.

This gap also blocks a good `/rewind`: without a real tree, "restore to an earlier point" can only
mean *delete the tail*. With a tree, "restore to here" becomes *fork a new branch from this point*,
non-destructively — which is the behavior we ultimately want. So branching is built first.

## Current state (evidence)

**The schema is already tree-capable, but the Console never uses it as a tree.**

- `messages.parent_message_id TEXT REFERENCES messages(id)` exists and is indexed
  (`idx_msgs_parent`) — `DB/ChaChaNotes_DB.py:324,338`.
- A real tree builder exists: `Chat/chat_conversation_service.py:_build_message_tree` produces
  nested `children[]` arrays; `get_conversation_tree` exposes it (also proxied to tldw_server via
  `get_chat_conversation_tree`).
- `conversations` already carries `forked_from_message_id`, `root_id`, `parent_conversation_id`
  — but that is *conversation-level* forking (a whole new conversation forked off a message), a
  different axis from the in-conversation message tree this spec targets.

**What the native Console actually does:**

1. **Never creates branches.** Every new message is persisted with its parent set to the previous
   persisted message — a strictly linear chain (`Chat/console_chat_store.py:1120`,
   `_previous_persisted_message_id`).
2. **"Swipe" is not the tree.** Regenerate builds an in-memory `ConsoleVariantSet` — a flat list of
   alternate texts for one assistant turn (`Chat/console_chat_models.py:224`). `<` / `>` navigates
   that in-memory list only; nothing is persisted as tree siblings.
3. **Resume collapses branches.** Restoring a persisted conversation walks the tree following only
   the **last child at each node** (`UI/Screens/chat_screen.py:_iter_console_tree_messages`,
   `_visit → children[-1]`). Any branches that exist in the DB — from tldw_server or another client
   — are silently dropped to a single path.
4. **No node navigation UI** — nothing lets a user fork a message or view/visit siblings.

**How tldw_server models branching (the parity target):**

- A branch is created by a continuation spec on the send: `{from_message_id, mode}` with
  `mode ∈ {branch, append}` (`tldw_Server_API/.../Chat/chat_service.py:2841`). `branch` parents the
  new message at *any* anchor node; `append` requires the anchor to be the latest message.
- History for a send is rebuilt by walking `parent_message_id` ancestry upward from the anchor to
  the root, then reversing (`_resolve_tldw_continuation_history`).
- **There is no `active_branch` / `selected_child` column.** The server never persists which branch
  is "active." The active path is derived: whoever reads the tree picks a leaf, and its ancestry
  *is* the path.

**Takeaway:** the tree (via `parent_message_id`) is the source of truth. "Active path" is a
client-side pointer. We adopt exactly that model.

## Goals

- Regenerate and edit-and-resend create **real, persisted sibling nodes** in the message tree.
- `<` / `>` swipe navigates **persisted siblings** at any branch point (assistant *and* user),
  survives resume, and shows a `n/m` sibling counter.
- Resume reconstructs the **active path** (not `children[-1]`), so the conversation you left is the
  conversation you return to.
- No divergence from the server's tree semantics: branches are ordinary messages with a
  `parent_message_id`; the only chatbook-local addition is the active-leaf pointer.

## Non-goals (explicit follow-up specs)

- **Explicit "Fork from here"** on any node (server `branch` mode against an arbitrary anchor) — B.
- **Tree / branch browser** to visualize the whole tree and jump to any node — C.
- **`/rewind` menu** (undo restore + summarize / free-context) — sub-project 2.
- Any change to *conversation-level* forking (`forked_from_message_id`) — untouched.

## Model

**The tree is truth; the active path is the ancestry of one remembered active-leaf pointer per
conversation.** One pointer fully determines the visible path: walk `parent_message_id` from the
leaf to the root, reverse, and that is the transcript. Regenerate / edit-resend move the pointer;
resume reads it.

### Decision record

- **D1 — Edit semantics: in-place stays; branching is an explicit "Edit & resend".** The existing
  "Edit" action keeps updating message content in place (typo fixes, no re-run). A new **"Edit &
  resend"** path in the edit modal creates a sibling user node under the same parent and re-sends,
  preserving the old subtree. This avoids surprising users whose muscle memory is in-place edit.
- **D2 — Active-path model: single active-leaf pointer** (not per-parent `selected_child`
  markers). Minimal, server-aligned, and sufficient because a path is uniquely defined by its leaf.

## Data flow

### Branch creation

- **Regenerate (assistant):** the regenerated turn's parent = the parent of the current assistant
  node. The new assistant reply is persisted as a **sibling** under that parent. Active-leaf → the
  new assistant node.
- **Edit & resend (user):** the edited user message's parent = the parent of the original user
  node. A new user sibling is persisted, then the model reply streams under it. Active-leaf → the
  new reply once complete (the new user node while pending).

In both cases the pre-existing sibling(s) and their entire subtrees remain in the DB untouched.

### Swipe navigation

- `<` / `>` on a message move the active-leaf pointer to the corresponding sibling's subtree leaf
  and re-render the transcript from the new active path.
- Choosing a sibling with children re-enters that sibling's own most-recent descendant path
  (recursively pick the latest child) to land on a concrete leaf.
- The `n/m` counter reflects sibling index / sibling count at that branch point. Branch points with
  a single child show no counter (unchanged from a linear thread's appearance).

### Resume

Replace the `children[-1]` walk in `_iter_console_tree_messages`. On restore:

1. Read `conversations.active_leaf_message_id`.
2. If present, walk `parent_message_id` from that leaf to the root and reverse → active path.
3. If absent or dangling (e.g., leaf soft-deleted), fall back to the current "latest child at each
   node" heuristic and repair the pointer to that path's leaf. This keeps pre-migration
   conversations and externally-authored trees working.

## Persistence & schema

- **New column:** `conversations.active_leaf_message_id TEXT REFERENCES messages(id) ON DELETE SET NULL`.
  Migration **v21 → v22** (mirroring the existing conversation-metadata migration files under
  `DB/migrations/`). `ON DELETE SET NULL` means a soft/hard-deleted leaf degrades gracefully to the
  resume fallback above.
- **Branch messages** persist through the existing message path with `parent_message_id` set to the
  branch anchor's parent — no new message columns. The sync layer already carries
  `parent_message_id` (`console_chat_store._enqueue_sync_v2_message_if_ready`), so **branches sync
  to tldw_server as ordinary messages with no new sync work.** The active-leaf pointer is
  chatbook-local UX state; the server derives active-path independently, so it does not sync.
- **Optimistic-locking / soft-delete** conventions are unchanged: siblings are normal rows.

## Store & controller changes

`ConsoleChatStore` gains a small, well-bounded branch API (names indicative):

- `create_sibling(anchor_message_id, ...) -> ConsoleChatMessage` — persist a new message parented
  at the anchor's parent.
- `siblings_at(message_id) -> list[...]` — ordered siblings sharing a parent, with index/count.
- `set_active_leaf(session_id, message_id)` / `active_leaf(session_id)` — pointer accessors,
  write-through to `conversations.active_leaf_message_id`.
- `active_path_messages(session_id) -> list[ConsoleChatMessage]` — rebuild the transcript from the
  active leaf's ancestry (the resume + swipe primitive).

`ConsoleVariantSet` is **refactored into a view over persisted siblings** rather than an
independent in-memory store, so `select_variant` / `<` / `>` and regenerate all read and move the
same tree state. Existing variant call sites are re-pointed at the sibling API; the dataclass may
remain as a read model but is no longer the source of truth.

The controller's regenerate and edit-resend paths write siblings + update the pointer instead of
appending in-memory variants or overwriting content.

## UI

- `<` / `>` and the `n/m` counter render at **any** branch point — including user messages, which
  can now have siblings. Reuse the existing selected-message action row and transcript rendering;
  add sibling awareness to user rows (today only assistant rows carry variants).
- The edit modal gains an **"Edit & resend"** affordance alongside the existing in-place save
  (D1). In-place save is unchanged.
- No new screens. This is a foundation; the tree browser (C) and explicit Fork-from-here (B) are
  where richer UI lands.

## Edge cases

- **Dangling / soft-deleted active leaf** → resume fallback to latest-child path, pointer repaired.
- **Externally-authored branches** (tldw_server, other clients) with no chatbook pointer → resume
  fallback, then pointer set on first swipe/regenerate.
- **Pending/streaming message** while swiping → swipe is blocked during an active run (consistent
  with existing message-action gating in `console_message_actions`).
- **Single-child nodes** → identical to a linear thread; no counter, no behavior change.
- **Cycle protection** — ancestry walks bound by a visited-set (the server does this in
  `_resolve_tldw_continuation_history`; mirror it).

## Testing

- **Persistence round-trip:** regenerate → two assistant siblings exist in DB under one parent;
  active-leaf points at the second.
- **Resume follows active path:** persist a branched tree, set active-leaf to a non-latest branch,
  restore → transcript matches that branch (not `children[-1]`).
- **Swipe moves the pointer and survives resume:** swipe to sibling 1/2, resume → still on 1/2.
- **Edit & resend branches; in-place edit does not:** in-place edit leaves sibling count at 1;
  Edit & resend raises it to 2 and preserves the old subtree.
- **Fallback paths:** missing pointer, dangling pointer, externally-authored tree.
- **User-message sibling navigation** counter and rendering.
- Real in-memory SQLite for DB tests (house convention).

## Rollout / risk

- Behind the existing native Console; no change to the deprecated enhanced-chat path.
- Pre-migration conversations and server-authored trees are handled by the resume fallback, so the
  change is backward-compatible on first open.
- The `ConsoleVariantSet` refactor is the highest-touch area; keeping the dataclass as a read model
  over siblings limits blast radius at call sites.
