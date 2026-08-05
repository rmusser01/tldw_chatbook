# Console Conversation Branching Foundation — Design

**Date:** 2026-07-22
**Status:** Approved (design, revised after code-grounded review) — pending final spec review
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
  nested `children[]` arrays (fetching children in batches by parent id); `get_conversation_tree`
  exposes it (also proxied to tldw_server via `get_chat_conversation_tree`).
- `conversations` already carries `forked_from_message_id`, `root_id`, `parent_conversation_id`
  — but that is *conversation-level* forking (a whole new conversation forked off a message), a
  different axis from the in-conversation message tree this spec targets.

**What the native Console actually does:**

1. **Never creates branches — and the local DB has *no* parent links at all.** Console's local
   persistence writes `parent_message_id=None` for every message (`_persist_new_message` /
   `_persist_existing_message` hardcode `None`, even though the `ConsoleChatPersistence`
   contract already accepts `parent_message_id`). The only place a linear parent is *computed*,
   `_previous_persisted_message_id`, feeds **solely the Sync v2 outbox envelope** (parent +
   sequence), never the local row. So existing Console conversations round-trip as a **flat set of
   rootless messages** — which is exactly why resume's `children[-1]` walk has been harmless so far.
   Introducing branching means we start writing real `parent_message_id` to the local DB for the
   first time; existing flat conversations become the resume-fallback case.
2. **"Swipe" is not the tree.** Regenerate builds an in-memory `ConsoleVariantSet` — a flat list of
   alternate texts for one assistant turn (`Chat/console_chat_models.py:224`). `<` / `>` navigates
   that in-memory list only; nothing is persisted as tree siblings.
3. **Resume collapses branches.** Restoring a persisted conversation walks the tree following only
   the **last child at each node** (`UI/Screens/chat_screen.py:_iter_console_tree_messages`,
   `_visit → children[-1]`). Any branches that exist in the DB — from tldw_server or another client
   — are silently dropped to a single path.
4. **No node navigation UI** — nothing lets a user fork a message or view/visit siblings.

**Two facts from the review that shape the design:**

- **Persistence is deferred and optional.** `ConsoleChatStore` runs live entirely in memory and
  flushes lazily (`_persist_new_message_or_defer`, `persist_session_if_needed`); with
  `persistence is None` it is a pure in-memory store (tests, and any no-backend mode). **The live
  transcript is in-memory-first; the DB is write-through.** The tree model must therefore live in
  memory and *mirror* to the DB, not the other way around.
- **The system prompt is a conversation field, not a message row** (`update_conversation_system_prompt`
  / `set_session_system_prompt`). So it is never a tree root and needs no special path handling.

**How tldw_server models branching (the parity target):**

- A branch is created by a continuation spec on the send: `{from_message_id, mode}` with
  `mode ∈ {branch, append}` (`tldw_Server_API/.../Chat/chat_service.py:2841`). `branch` parents the
  new message at *any* anchor node; `append` requires the anchor to be the latest message.
- History for a send is rebuilt by walking `parent_message_id` ancestry upward from the anchor to
  the root, then reversing (`_resolve_tldw_continuation_history`), bounded by a visited-set.
- **There is no `active_branch` / `selected_child` column.** The server never persists which branch
  is "active." The active path is derived: whoever reads the tree picks a leaf, and its ancestry
  *is* the path.

**Takeaway:** the tree (via `parent_message_id`) is the source of truth. "Active path" is a
client-side pointer. We adopt exactly that model, held in memory and mirrored to the DB.

## Goals

- Regenerate and edit-and-resend, **at any message**, create **real, persisted sibling nodes** in
  the message tree — a new branch from that point.
- `<` / `>` swipe navigates **persisted siblings** at any branch point (assistant *and* user),
  survives resume, and shows a `n/m` sibling counter.
- Resume reconstructs the **active path** (not `children[-1]`), so the conversation you left is the
  conversation you return to.
- No divergence from the server's tree semantics: branches are ordinary messages with a
  `parent_message_id`; the only chatbook-local addition is the active-leaf pointer.

## Non-goals (explicit follow-up specs)

- **Explicit "Fork from here"** as a first-class deliberate action + affordance on any node — B.
  (Branch *creation* via regenerate/edit-resend at any node IS in scope here; a dedicated
  fork-without-editing action and its UI are the follow-up.)
- **Tree / branch browser** to visualize the whole tree and jump to any node — C.
- **`/rewind` menu** (undo restore + summarize / free-context) — sub-project 2.
- Any change to *conversation-level* forking (`forked_from_message_id`) — untouched.

## Phasing (three independently-shippable plans)

The foundation is too large for one implementation plan. It splits into three phases, each a
complete, testable, independently-reviewable deliverable with its own plan document:

- **Phase A — Persisted assistant branching + active-path resume.** The ChaChaNotes v22→v23
  active-leaf column; the in-memory tree + real `parent_message_id` writes; regenerate creating a
  persisted sibling assistant node; swipe navigating siblings; the active-leaf pointer; resume
  reconstruction (active-leaf ancestry, fallback to `children[-1]`); sync-sequencing rework.
  *Ships:* "regenerate alternates persist as real branches and survive resume." This is the first
  plan written.
- **Phase B — User-message branching via "Edit & resend."** Widen the edit modal's return
  contract; user-side branch creation under the shared parent; user-row swipe UI. Builds on A.
- **Phase C — Agent-marker anchoring under branching.** The `agent_runs` v1→v2 migration + threading
  `assistant_message_id` + rewriting marker placement to anchor on the active path. Only matters for
  agent-runtime conversations that also branch; A/B keep ordinal placement until this lands.

## Model

**The in-memory tree is the source of truth for a live session; the DB is a write-through mirror
for durability and resume.** The active path you see = the ancestry of one **active-leaf pointer**,
held in session state and mirrored to `conversations.active_leaf_message_id`. One pointer fully
determines the visible path: walk parents from the leaf to the root, reverse — that is the
transcript. Regenerate / edit-resend move the pointer; resume reads it (or falls back).

`ConsoleChatStore` today keeps a **flat** `_messages_by_session` list. Under this design that flat
list becomes a **rendered view = the active path**, derived from an in-memory node set that holds
*all* branches (parent links + children). Swipe recomputes the view from memory with **no DB
access**; only resume touches the DB.

### Decision record

- **D1 — Edit semantics: in-place stays; branching is an explicit "Edit & resend".** The existing
  "Edit" action keeps updating message content in place (typo fixes, no re-run). A new **"Edit &
  resend"** path in the edit modal creates a sibling user node and re-sends through the normal send
  pipeline, preserving the old subtree. Avoids surprising the existing in-place-edit muscle memory.
- **D2 — Active-path model: single active-leaf pointer** (not per-parent `selected_child`
  markers). Minimal, server-aligned, sufficient because a path is uniquely defined by its leaf.
- **D3 — Branch creation is allowed at *any* message (full mid-conversation branching).**
  Regenerating an assistant reply or editing-and-resending a user message that sits mid-conversation
  creates a **new branch rooted at that point**: the new node becomes a sibling under the anchor's
  parent, the active-leaf moves onto the new subtree, and the previously-visible tail is **truncated
  from view but preserved off-path** (reachable again by swiping back). This is standard tree
  behavior (ChatGPT/tldw_server) and is the explicit intent.

## Data flow

### Branch creation (any node)

- **Regenerate (assistant node `A`, child of parent `P`):** persist a new assistant sibling `A'`
  under `P`. Active-leaf → `A'`. If `A` had descendants (a mid-conversation regenerate), that whole
  subtree stays under `A`, off the active path; the visible transcript now ends at `A'`.
- **Edit & resend (user node `U`, child of parent `P`):** persist a new user sibling `U'` under
  `P`, then stream the model reply `R` under `U'` via the normal send pipeline (readiness gating,
  provider selection, dictionaries, streaming). Active-leaf → `U'` while pending, → `R` on
  completion. `U`'s old subtree is preserved off-path.

In every case the pre-existing sibling(s) and their entire subtrees remain untouched in memory and
in the DB.

### Swipe navigation (pure in-memory)

- `<` / `>` at a branch point move the active-leaf to the chosen sibling's subtree, then descend
  that subtree by **most-recent child at each step** to land on a concrete leaf, and re-render the
  view. No DB reads.
- Sibling order is **deterministic**: `ORDER BY timestamp, id` (matching the server), so `<` / `>`
  and the `n/m` counter are stable across a session and across resume.
- The `n/m` counter shows sibling index / count at that branch point. Single-child nodes show no
  counter and behave exactly like a linear thread.
- Swipe is **blocked while a run is pending/streaming** (consistent with existing message-action
  gating in `console_message_actions`).

### Tool / agent markers under branching

Resume re-derives non-persisted TOOL markers from `AgentRunsDB` and interleaves them
(`inject_resume_agent_markers`). Today placement is purely **ordinal** — the Nth run matched to the
Nth assistant message — which is *wrong once branches exist*, because off-path branches also
produced runs.

Fix: **anchor each run's marker block to the `assistant_message_id` it produced.** Concretely,
`AgentRunsDB` does **not** store a message id today (`agent_runs` is at schema v1 with no message
column) and `run_reply` receives `assistant_message_id` but drops it before `create_run`. So the
fix requires: an **`agent_runs` v1→v2 migration** adding an `assistant_message_id` column;
threading that id through `run_turn` → `AgentService._run_one` → `create_run`;
`resume_marker_messages` returning `(assistant_message_id, block)` pairs; and rewriting
`inject_resume_agent_markers` to place each block after the transcript message whose
`persisted_message_id == assistant_message_id`, hiding blocks whose anchor is off the active path.
The rail's per-conversation agent summary (`historical_snapshot`) is unchanged — it is a
conversation-wide rollup, not a per-path view. Because this is a self-contained surface that only
affects agent-runtime conversations that also have branches, it is deferred to **Phase C** (see
Phasing); Phases A/B keep the existing ordinal placement.

### Resume

Replace the `children[-1]` walk. On restore, fetch the conversation's messages once (reuse the
batched `get_conversation_tree` fetch — no per-ancestor N+1), then:

1. Read `conversations.active_leaf_message_id`.
2. If it names a live (non-deleted) message in this conversation, walk parents from it to the root,
   reverse → active path.
3. Otherwise (missing, soft-deleted, or foreign) **fall back** to the current "most-recent child at
   each node" heuristic and repair the pointer to that path's leaf.
4. Include any **root-level thread that is not the active leaf's ancestor** only if required for a
   valid transcript; in practice the system prompt is a conversation field (not a root message), so
   the active path is normally the whole visible transcript. Multiple `root_threads` are handled by
   selecting the root that is the active leaf's ancestor.

## Persistence & schema

- **New column:** `conversations.active_leaf_message_id TEXT` — a **plain nullable pointer**. This
  codebase's canonical `conversations` CREATE TABLE is **frozen at the v4 schema**; every later
  column is added *only* by running migrations, and a fresh DB runs 4→N on first open. So this is
  **migration-only — do not edit the CREATE TABLE.** Add a **v22 → v23** migration
  (`_CURRENT_SCHEMA_VERSION` is already **22** on `origin/dev`): a guarded `_migrate_from_v22_to_v23`
  method (PRAGMA-check then `ALTER TABLE conversations ADD COLUMN active_leaf_message_id TEXT`) that
  `executescript`s a `_MIGRATE_V22_TO_V23_SQL` constant containing **only the
  `db_schema_version` bump** (no trigger DDL — see the next bullet), a
  `DB/migrations/chachanotes_v22_to_v23_*.sql` reference file, registering
  `22: self._migrate_from_v22_to_v23` in the `migration_steps` dict, and bumping
  `_CURRENT_SCHEMA_VERSION` to 23.
- **No trigger redefinition; the pointer is local-only and never syncs.** The active-leaf pointer
  is chatbook-local UX state — the server derives active-path and has no column to receive it, so
  syncing it is pointless. Therefore the column is written by a **dedicated setter,
  `set_conversation_active_leaf(conversation_id, message_id)`, that does a bare
  `UPDATE conversations SET active_leaf_message_id = ? WHERE id = ? AND deleted = 0`** — crucially
  **not** bumping `version`/`last_modified` and **not** touching any column named in the
  `conversations_sync_update` trigger's `WHEN` clause. Because that `WHEN` only fires on
  `version`/`last_modified`/other tracked-field changes, the setter produces **no `sync_log` row**
  and swipes cause no sync churn. Consequences: the migration does **not** redefine the
  `conversations_sync_*` triggers (they never carry the column), and the pointer is **not** routed
  through the optimistic-lock `update_conversation` path (which would bump `version` and sync). A
  matching reader `get_conversation_active_leaf(conversation_id) -> str | None` does a plain
  `SELECT`. This deliberately trades optimistic-lock safety (last-write-wins is fine for a
  per-client view pointer) for zero sync/version churn.
- **Do not rely on FK cascade for correctness.** The DB uses **soft-delete**, so an
  `ON DELETE SET NULL` FK would never fire on a soft-deleted leaf, and FK enforcement
  (`PRAGMA foreign_keys`) is not guaranteed on. Correctness comes from the **resume fallback +
  pointer validation** above (treat missing / soft-deleted / foreign leaves as dangling). A FK
  clause, if added, is belt-and-suspenders only.
- **Branch messages** persist through the existing message path, now passing a **real
  `parent_message_id`** (the in-memory node's actual tree parent) into `create_message` /
  `update_message_content` — which the persistence contract already accepts but the store has always
  passed as `None`. This is net-new local parent-linking, not a change to existing linear parents.
- **Sync sequencing must be revisited.** `_previous_persisted_message_id` also feeds sync sequence
  numbers, which today assume linearity. The plan must re-derive sequence/parent from the tree so
  siblings sync coherently; branch messages otherwise sync as ordinary messages (the sync layer
  already carries `parent_message_id`). The active-leaf pointer is chatbook-local UX state and does
  not sync (the server derives active-path independently).
- **Optimistic-locking / soft-delete** conventions are unchanged: siblings are normal rows; the new
  pointer column participates in the conversation row's normal versioning.

## Store & controller changes

`ConsoleChatStore` gains an in-memory branch model + a small, well-bounded API (names indicative):

- Per-session in-memory node set: each message carries its `parent_id` and the store can enumerate
  children; the rendered `_messages_by_session` view is the active path.
- `create_sibling(anchor_message_id, ...) -> ConsoleChatMessage` — new node under the anchor's
  parent, in memory + write-through.
- `siblings_at(message_id) -> (list, index, count)` — ordered siblings (timestamp, id).
- `set_active_leaf(session_id, message_id)` / `active_leaf(session_id)` — pointer accessors,
  write-through to `conversations.active_leaf_message_id` when a backend exists.
- `active_path_messages(session_id) -> list[ConsoleChatMessage]` — the active leaf's ancestry; the
  swipe (in-memory) and resume (post-fetch) primitive.

`ConsoleVariantSet` is **refactored into a read view over sibling nodes** rather than an
independent in-memory store, so `select_variant` / `<` / `>` / regenerate all move the same tree
state. Existing variant call sites are re-pointed at the sibling API; the dataclass may remain as a
render model but is no longer the source of truth. Assistant *and* user rows can now carry sibling
nav.

The controller's regenerate and edit-resend paths write siblings + move the pointer instead of
appending in-memory variants or overwriting content. Edit & resend reuses the existing send
pipeline.

## UI

- `<` / `>` and the `n/m` counter render at **any** branch point — including user messages, which
  can now have siblings. Reuse the existing selected-message action row and transcript rendering;
  add sibling awareness to user rows (today only assistant rows carry variants).
- The edit modal gains an **"Edit & resend"** affordance alongside the existing in-place save (D1).
- No new screens. The tree browser (C) and a first-class "Fork from here" action (B) are where
  richer UI lands.

## Edge cases

- **Dangling / soft-deleted / foreign active leaf** → resume fallback to most-recent-child path,
  pointer repaired.
- **`persistence is None` (pure in-memory session)** → full branching works in memory; the pointer
  is session state only; nothing is written.
- **Externally-authored branches** (tldw_server, other clients) with no chatbook pointer → resume
  fallback, then pointer set on first swipe/regenerate.
- **Mid-conversation regenerate/edit** truncates the visible tail onto the new branch (D3);
  off-path subtree preserved and reachable by swiping back.
- **Tool markers on off-path branches** are hidden; on-path markers render after their anchor
  message.
- **Pending/streaming** → swipe and branch-navigation blocked.
- **Multiple `root_threads`** → select the root that is the active leaf's ancestor.
- **Cycle protection** → ancestry walks bounded by a visited-set (mirror the server).

## Testing

- **Persistence round-trip:** regenerate → two assistant siblings under one parent; active-leaf =
  the second.
- **Branch at a mid-conversation node:** regenerate an old assistant reply → new sibling, visible
  tail truncates, old subtree still present in the tree; swipe back restores it.
- **Resume follows active path:** persist a branched tree, set active-leaf to a non-latest branch,
  restore → transcript matches that branch (not `children[-1]`).
- **Swipe survives resume:** swipe to sibling 1/2, resume → still 1/2.
- **Edit & resend branches; in-place edit does not:** in-place keeps sibling count at 1; Edit &
  resend raises it to 2 and preserves the old subtree.
- **In-memory branching with `persistence is None`.**
- **Tool markers:** a run on an off-path branch is hidden; an on-path run renders after its anchor;
  mixed branches don't cross-place markers.
- **Fallbacks:** missing pointer, dangling pointer, externally-authored tree, multiple roots.
- **Migration:** a fresh DB (runs 4→23) has the column; an existing v22 DB migrates to v23; the
  guarded ALTER is idempotent on replay; the redefined `conversations_sync_*` triggers emit the new
  column in their payloads.
- Real in-memory SQLite for DB tests (house convention).

## Rollout / risk

- Behind the existing native Console; no change to the deprecated enhanced-chat path.
- Pre-migration conversations and server-authored trees are handled by the resume fallback, so the
  change is backward-compatible on first open.
- Highest-touch areas: the `ConsoleVariantSet` → sibling-view refactor, and replacing the
  linear-`_previous_persisted_message_id` assumption in persist + sync sequencing. The in-memory
  read-view boundary limits blast radius at render call sites; the sync-sequencing change is the
  one to review carefully.
- The tool-marker anchoring change depends on `AgentRunsDB` carrying `assistant_message_id`; if it
  does not today, that becomes a small additive schema/plumbing task inside this project.
