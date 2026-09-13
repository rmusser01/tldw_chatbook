# Agent Chat Fork & Spawn — Design Spec

Date: 2026-09-11
Status: Approved design (pre-plan)
Governance: ADR required (conversation-lineage semantics are a data-ownership
decision; an agent-initiated, confirmation-gated chat-creation channel is a
cross-module interface with a security policy). New ADR, not amending ADR-146.
Backlog task to be created at plan time; follow-up task for sub-agent support
already filed (`backlog/tasks/task-32480 - Extend-fork_chat-new_chat-tools-to-sub-agents.md`).

```text
ADR required: yes
ADR path: backlog/decisions/NNN-agent-chat-fork-and-spawn.md (number assigned at plan time)
Reason: first writer of the conversation-lineage columns (data ownership) plus a new
agent-initiated UI-mutation channel gated by a confirmation policy (cross-module
interface + security); fork copy semantics will be reused by the sub-agent and
preset-routing follow-ups.
```

## Problem

A Console agent can suggest parallel workstreams ("we could chase A, B, or C"),
but the user must manually create each chat (Ctrl+T), re-title it, re-paste
context, and re-state the goal. There is no agent-reachable way to prepare those
chats, and no fork primitive at all: the `parent_conversation_id` /
`forked_from_message_id` columns added in the V11→V12 migration "for
conversation forking" (`DB/ChaChaNotes_DB.py:468-469`) have never been written
by Console code, and `Chat/chat_persistence_service.py:333`
`fork_conversation_into_workspace` is a misleading name — it links workspace
membership for the *same* conversation and copies nothing.

This spec adds two runtime tools, `fork_chat` and `new_chat`, that let the
primary agent prepare follow-through chats for the user, always confirmed by
default.

## Locked decisions (from brainstorming)

1. **Fork copies the active path as of the tool call.** Root → active leaf,
   verbatim: roles, content, tool-call/result markers, tree parents remapped.
   Mid-conversation forks compose with existing rewind (rewind first, then
   fork). No agent-facing message-addressing scheme.
2. **Agent-settable args: `title`, `opening_prompt`, `instructions`** — and
   nothing else. No provider/preset/model args in v1 (deferred, decision 7).
3. **Two runtime tools**, not one with a mode enum: `fork_chat` (copy path)
   and `new_chat` (fresh chat). Distinct schemas, descriptions, and card
   headers. Names deliberately avoid "spawn" (`spawn_subagent` exists).
4. **The opening prompt is a draft, never sent.** It lands in the new chat's
   composer via `set_session_draft` ("Open in Console" precedent). The user
   reviews, edits, and sends. One approval covers one creation.
5. **Confirmation: per-call card, Allow / Allow for this session / Deny**,
   modeled on `request_skill_script_confirm` (`Chat/console_chat_controller.py:6253`).
   Fail-closed on no-UI/cancel/timeout; parks a badge for background sessions.
   Remember is per-tool and Console-session-scoped — no persisted bypass, so
   every new conversation confirms by default.
6. **Landing: background + toast.** The new chat is created in the same
   workspace, not activated; a toast announces it by title; the user stays in
   the current chat while the agent finishes its turn.
7. **Provider/preset routing for new chats is out of scope for v1** and lands
   as a dedicated integration PR after the agent provider-routing work
   (presets + gated spawn overrides) merges to dev.

## Goals

* An agent can turn "here are three parallel workstreams" into three prepared
  chats — titled, optionally forked from current context, each with a draft
  opening prompt — in one turn, each behind an explicit user approval.
* Fork semantics are exact and durable: verbatim active-path copy, lineage
  columns written, divergence afterwards is total (new messages never appear
  in the source and vice versa).
* Confirmation is the default state; opt-out is bounded to the current
  Console session and per tool.
* Zero DB schema migration in either database.

## Non-goals

* Auto-sending the opening prompt, or the agent chatting in the new chat.
* Agent-chosen fork points, batch-approval of multiple creations in one card,
  full-tree copies (variants/inactive branches).
* Provider/model/preset selection on the created chats (decision 7).
* Sub-agent access to the tools — filed as `task-32480`.
* Any change to rewind, branching, or workspace-membership semantics beyond
  what creation needs.

## Architecture

Both tools follow the `install_skill` / `run_skill_script` **runtime-tool**
pattern — not local/builtin catalog tools — because they need conversation
context, a blocking confirm round-trip, and chat-store access from the agent's
worker thread. Per tool:

* `RUNTIME_TOOL_NAMES` entry (`Agents/agent_models.py:106`) and a
  `ToolSchema` constant in `Agents/tool_catalog.py` (style of
  `INSTALL_SKILL_TOOL_SCHEMA` :258). IDs: `runtime:fork_chat`,
  `runtime:new_chat`. Descriptions document the confirm contract, the
  draft-not-send behavior, the mid-turn snapshot boundary (see Fork
  primitive), and guidance to create sparingly (one card per call; a turn
  that needs three chats gets three sequential confirms).
* Constructor-injected callable on `AgentService.__init__` (pattern:
  `run_skill_script_tool`, `agent_service.py:1045`), threaded in from the
  bridge's `run_reply` composition (`Chat/console_agent_bridge.py:3385-3433`,
  closure pattern :3678-3865).
* Schema pin in `_run_one`, gated on
  `agent_kind == AGENT_KIND_PRIMARY and self._<callable> is not None`,
  **appended after the run-log block (post-`agent_service.py:2597`)** so the
  diff inside `agent_service.py` is new lines only — the in-flight
  provider-routing PR rewrites the spawn closure (:3115-3442) and the
  `runtime_schemas` block around :2513-2597.
* Dispatch branch in `Agents/agent_runtime.py` `run_agent_loop` (spawn branch
  :1450-1490 is the model; `LoopDeps` :327-476).

Transcript rendering is free: the tools emit normal `STEP_TOOL_CALL` /
`STEP_TOOL_RESULT` steps and render via `format_agent_step_marker`
(`console_agent_bridge.py:865-913`) / `_append_marker` (:6180).

## Tool contracts

### `fork_chat`

Args (all optional strings): `title`, `opening_prompt`, `instructions`.

Behavior: snapshot the **active path of the running agent's own conversation**
(not whatever session is visible on screen) and copy it verbatim into a new
conversation:

* Fields preserved per message: sender/role, content, `metadata_json`,
  `usage_json`, `provider_continuation_json`, image fields, timestamps. IDs
  regenerate; `parent_message_id` remaps old→new.
* `parent_conversation_id` = source conversation id;
  `forked_from_message_id` = source active-leaf message id.
* Inherits from the source: workspace (`scope_type`/`workspace_id`),
  assistant identity / character binding, system prompt, speech preferences.
  `instructions`, when given, **replaces** the system prompt — except on
  character-bound sources, where `instructions` is refused (see Errors): a
  model-authored prompt must not silently override a persona.
* Project-instruction bindings are **not** copied: the new chat starts with
  its own default `ProjectInstructionControlState` (per-session binding is a
  deliberate user act under ADR-069 governance).

### `new_chat`

Same args. Creates a fresh, empty conversation in the same workspace with
Console defaults plus whatever the agent supplied. Never inherits another
chat's identity or context.

### Shared semantics

* `title` defaults: `"Fork of <source title>"` / `"New Chat"`. Clamped and
  sanitized (control chars stripped, length cap) via `input_validation`.
* `opening_prompt` (max length capped, excess rejected with a clear error)
  becomes the new session's composer draft via `set_session_draft`.
* Result JSON to the agent: `{ok, title, conversation_id, workspace_id,
  copied_messages (fork only), draft_set: true, note}` where `note` states
  that the chat opened in the background and the user must send the draft
  themselves.
* Both tools may only run when the source session is durable. An ephemeral
  source (`ConsoleChatSession.ephemeral`) is a tool error for `fork_chat`
  (nothing persisted to copy); `new_chat` always creates a durable chat.

### Mid-turn snapshot boundary

The copy is taken at tool execution. Messages appended to the source after
that point — including this tool call's own result marker and the remainder
of the current assistant turn (the very reply suggesting the workstreams) —
are **not** in the fork. This is deliberate and must be taught in the tool
description: the agent carries workstream framing in `opening_prompt`, never
relies on its in-flight reply being present in the copy.

## Fork primitive

New helper `fork_conversation_history` in `Chat/chat_conversation_service.py`
(which already owns `create_conversation` :370; the name avoids collision
with the misnomer `fork_conversation_into_workspace`):

1. Read the source active path via the existing tree walk
   (`Chat/chat_conversation_scope_service.get_conversation_tree` :364, the
   same raised-caps read `Chat/console_conversation_hydration.py:250`
   `load_console_conversation_tree` uses). **The read must be uncapped —
   silent truncation of long chats is a bug, and a >cap-length fork is a
   required test.**
2. Open ONE `db.transaction()` and create the target inside it through
   `Chat/chat_persistence_service.create_conversation` (:216) — carries
   title, workspace scope, identity, system prompt (possibly overridden by
   `instructions`), speech prefs, metadata — passing the two lineage columns
   through the service's explicit validated `parent_conversation_id` /
   `forked_from_message_id` parameters. Creation shares the copy's
   transaction (the DB's nested `transaction()` levels make the outer one
   authoritative), so a mid-copy failure rolls the target back too: a
   failed fork creates nothing.
3. Still inside that transaction, loop `db.add_message` (:9524) remapping
   parents; chunked inserts so a large copy does not hold an oversized
   single statement batch (FTS triggers fire per row; cost is bounded and
   off the UI thread).
4. `set_conversation_active_leaf` (:9096) to the copied leaf, then commit.

No schema migration: both lineage columns already exist (index included,
:482) and are simply never written by the Console today.

### Draft durability

The composer draft is session state and would vanish on restart. To make the
handoff durable, the creation writes the draft into the conversation's
local-only metadata (key `console_agent_handoff`: `{draft, created_via,
source_run_id}`), the same persistence philosophy as per-conversation
project-instruction state (`set_conversation_console_project_context`,
`DB/ChaChaNotes_DB.py:9165`). `restore_persisted_session`
(`Chat/console_chat_store.py:1210`) rehydrates it into the session draft; it
is consumed by ordinary composer semantics (sent, edited, or replaced).

## Session creation & UI surfacing

After an allow, on the agent worker thread: run the fork copy (fork_chat) or
nothing (new_chat) — note `new_chat` also creates its conversation row
immediately (not lazily) so both tools produce an equal, durable artifact.
Then marshal to the UI thread (`app.call_from_thread`, pattern
`UI/Console_Modules/agent.py:1121`):

1. Build a **non-activated** session over the new conversation —
   `restore_persisted_session` activates today, so it gains an
   `activate=False` variant (`create_session` :883 already supports the
   flag; `Chat/console_launch_wake.py:235` proves no-screen hydration).
2. `set_session_draft(opening_prompt)`.
3. Workspace listing: verify at implementation time whether the workspace
   thread list keys off `conversations.scope_type/workspace_id` or
   `workspace_registry` membership; if the latter, apply the same
   membership link `fork_conversation_into_workspace` uses. The new chat
   must actually appear in its workspace list — invisibility here would be
   a silent failure.
4. `_invalidate_console_persisted_rows_cache()` and
   `run_worker(self._sync_native_console_chat_ui, exclusive=True,
   group="console-sync")` (`UI/Screens/chat_screen.py:12159`).
5. Toast: `app.notify` — "Forked chat created: <title>" / "New chat created:
   <title>".

The user is never switched away mid-turn (locked decision 6).

## Confirmation flow

Modeled on `request_skill_script_confirm` (`console_chat_controller.py:6253-6438`)
with `resolve_pending_*` (:6472) and a `set_pending_*` card seam:

* Card: verb header ("Fork this chat" / "Create new chat"), proposed title,
  **full bodies** of `opening_prompt` and `instructions` (a model-authored
  system prompt is a persistent injection surface and is always shown in
  full on the card), fork summary (N messages from "<source title>"),
  requesting run/agent identity.
* Buttons: **Allow / Allow for this session / Deny**. Remember is per tool
  name, scoped to the Console session (cleared when the session closes), and
  covers only the confirm card — creation still executes fresh each call.
* No UI / cancel / timeout / revoked round ⇒ deny-equivalent error result to
  the agent. Parks a badge for background sessions (pattern:
  `_park_console_approval`, `UI/Screens/chat_screen.py:16346`).
* Honors the run-cancel signal and the approval kill-switch semantics like
  the skill-script confirm does.
* **Denial guard:** denials are counted per tool per run; after 2 denials,
  further calls of that tool fail fast with a terminal error for the rest of
  the run (no card). Deny results and tool descriptions both instruct the
  model not to retry within the turn.

Residual, accepted risk (documented in the ADR): with session-remember
active, a later call's `instructions` payload is not card-reviewed; exposure
is bounded to the Console session and the resulting system prompt remains
visible/editable in the new chat's settings.

## Error handling

All errors surface as the tool's error result (never exceptions across the
seam):

* `source_not_persisted` — fork on an ephemeral session.
* `empty_history` — fork of a conversation with zero messages ("use
  new_chat").
* `character_conflict` — `instructions` on a character-bound fork source.
* `payload_too_large` — `opening_prompt`/`instructions` over the length cap.
* `user_denied` / `confirm_timeout` / `run_cancelled` — fail-closed outcomes.
* `denied_repeatedly` — terminal after the denial guard trips.
* `copy_failed` — transaction rolled back; nothing created.

## Sub-agent extension (deferred — task-32480)

v1 pins both tools to `AGENT_KIND_PRIMARY`. Sub-agents are scoped to the
parent conversation, so extending means: pin schemas for child kinds, fork
from the parent conversation's active path, identify the requesting agent
and parent run on the card, and scope denial/remember to the requesting run.

## Provider/preset integration (deferred — decision 7)

After the agent provider-routing PR merges to dev, a follow-up PR adds
optional preset/provider selection to the created chats, reusing that work's
`AgentDefinition` routing vocabulary and `SpawnTarget` resolution. The v1
schemas intentionally omit those args so the later change is additive.

## Testing

* **Unit (fork helper):** active-path selection; parent remap; leaf set;
  lineage columns written; field preservation (images, usage,
  provider_continuation, metadata); TOOL-role markers copied; **>
  read-cap message count copies without truncation**; mid-copy failure rolls
  back atomically; ephemeral and empty-history refusal; character-bound
  `instructions` refusal; `new_chat` immediate persistence.
* **Unit (confirm):** allow / deny / remember-scoping (per tool, per
  session) / no-UI fail-closed / cancel / deadline / park-for-background;
  denial-guard terminal behavior.
* **Integration:** bridge → controller → store end to end on a real
  in-memory SQLite DB; schema pinned for primary runs and absent for
  subagent kinds; dispatch branch reached; result JSON shape; STEP markers
  render via `format_agent_step_marker`; workspace listing shows the new
  chat (whichever listing mechanism step 3 of surfacing confirmed);
  `activate=False` does not switch the active session.
* **Draft durability:** conversation + `console_agent_handoff` metadata
  survive a store rebuild; draft rehydrates into the composer.
* **Live verification** (per `backlog/docs/lessons-live-verification.md`):
  real app run — agent proposes two workstreams, forks one chat and creates
  one fresh; confirm card, toast, switcher entries, drafts in composers;
  send in the fork and verify the source chat does not receive it; restart
  and verify both chats and drafts persist.

## Expected touch list

* New: `fork_conversation_history` in `Chat/chat_conversation_service.py`,
  spec (this doc), ADR.
* Modified: `Agents/agent_models.py` (two `RUNTIME_TOOL_NAMES` entries),
  `Agents/tool_catalog.py` (two schema constants),
  `Agents/agent_service.py` (two injected callables + post-:2597 pins — new
  lines only), `Agents/agent_runtime.py` (dispatch branches),
  `Chat/console_agent_bridge.py` (two closures in the `run_reply`
  composition region — accepted overlap with the routing PR),
  `Chat/console_chat_controller.py` (confirm rounds + resolvers + card
  seam), `Chat/console_chat_store.py` (`restore_persisted_session`
  `activate=False` variant, draft-rehydration hook),
  `Chat/chat_persistence_service.py` (lineage passthrough), plus the
  workspace-listing wiring whichever mechanism verification requires
  (`UI/Console_Modules/workspace.py` if membership links are needed).
* Docs: `Docs/User_Guide/console/agent-runs-and-tools.md`, a workstream
  blurb in `Docs/User_Guide/console/`.
* Tests: new `Tests/Chat/test_console_chat_fork_spawn.py` +
  fork-helper unit tests alongside the conversation-service suite.

## Rollout

1. Plan time: create ADR (`backlog/decisions/NNN-agent-chat-fork-and-spawn.md`)
   and the v1 backlog task; link both ways and to `task-32480`.
2. Sequence against the provider-routing PR: implement on top of it (or
   rebase before opening) — `agent_service.py`, `agent_runtime.py`, and
   `console_agent_bridge.py` are shared files; our edits are additive but
   the routing PR rewrites adjacent regions.
3. Implement in order: fork helper → runtime-tool seam (schema + pin +
   dispatch + injected callable) → confirm card & rounds → session
   surfacing & draft durability → docs.
4. Live verification before marking done; lessons doc updated only if the
   task surfaces something generalizable.
