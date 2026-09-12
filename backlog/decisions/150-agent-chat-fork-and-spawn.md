# ADR-150: Agent chat fork & spawn — confirmation-gated workstream chats

Status: Accepted
Date: 2026-09-11
Related Task: [TASK-32482](../tasks/task-32482%20-%20Agent-chat-fork-and-spawn-tools-fork_chat-new_chat.md)
Related Spec: [Agent chat fork & spawn design](../../Docs/superpowers/specs/2026-09-11-agent-chat-fork-spawn-design.md)
Coordinates with: [ADR-147](147-agent-provider-routing.md) (preset/provider args for created chats are a deferred follow-up PR); [ADR-069](069-console-project-instruction-local-state-and-preflight.md) (project-instruction bindings are never copied)
Follow-ups: TASK-32480 (sub-agent access), preset/provider integration PR (after ADR-147 lands on dev)

## Context

A Console agent can propose parallel workstreams but the user performs all the
follow-through manually: create tabs, re-title, re-paste context, re-state the
goal. Two gaps block automating the preparation. First, there is no
agent-reachable chat-creation path at all — the tool executors are
deliberately dependency-light (no store, no UI access), and reaching the
Console from the agent worker thread happens only through callables injected
at `run_reply` composition (the `install_skill` / `run_skill_script` seam).
Second, there is no fork primitive: the `parent_conversation_id` /
`forked_from_message_id` columns added in the V11→V12 migration "for
conversation forking" have never been written by Console code, and
`fork_conversation_into_workspace` is a misnomer that links workspace
membership for the same conversation without copying anything. Any
agent-initiated creation is also a new mutation channel from model-generated
input, so the confirmation policy is part of the architecture, not an
afterthought.

## Decision

- **Two runtime tools on the injected-callable seam: `fork_chat` and
  `new_chat`**, not local/builtin catalog tools. Both need conversation
  context, a blocking confirm round-trip, and chat-store access from the
  agent worker thread — exactly what the `run_skill_script` pattern provides.
  Pinned for `AGENT_KIND_PRIMARY` runs only in v1; sub-agents are a follow-up
  (TASK-32480). Names avoid "spawn" (`spawn_subagent` exists).
- **Fork = verbatim active-path snapshot at tool execution.** Root → active
  leaf of the running agent's own conversation: roles, content, tool markers,
  parents remapped, full field preservation (images, usage,
  provider-continuation JSON, metadata). Messages appended after the snapshot
  — including the tool's own result marker and the rest of the in-flight turn
  — are excluded, and the tool description teaches the agent to carry
  workstream framing in `opening_prompt` instead. Mid-conversation forks
  compose with existing rewind rather than inventing a message-addressing
  scheme the model could not use reliably. The lineage columns are finally
  written; no schema migration in either database.
- **Args are `title`, `opening_prompt`, `instructions` — and nothing else.**
  No provider/preset/model args in v1: those land in a dedicated integration
  PR after ADR-147 merges to dev. `instructions` replace the source's system
  prompt except on character-bound sources, which refuse them (a model-authored
  prompt must not silently override a persona). Project-instruction bindings
  are never copied — selecting a binding is a deliberate user act (ADR-069).
- **The opening prompt is a draft, never sent.** It lands in the new chat's
  composer via `set_session_draft`; the user reviews, edits, and sends. To
  survive restarts, the draft is persisted in the conversation's local-only
  metadata (`console_agent_handoff`) and rehydrated on session restore.
- **Confirmation is a per-call card: Allow / Allow for this session / Deny**
  on the `request_skill_script_confirm` pattern — fail-closed on no-UI,
  cancel, or timeout; parks a badge for background sessions; full bodies of
  `opening_prompt` and `instructions` always shown on the card (a
  model-authored system prompt is a persistent injection surface). Remember is
  per tool and Console-session-scoped: no persisted bypass, so every new
  conversation confirms by default. A per-run denial guard makes the tool
  terminal after two denials. Accepted residual risk, documented: a remembered
  session skips card review of later `instructions` payloads; exposure is
  bounded to the session and the prompt stays visible/editable in the new
  chat's settings.
- **Creation is durable, background, and never steals focus.** Both tools
  persist the conversation row immediately (no lazy minting), create a
  non-activated session in the same workspace, invalidate the persisted-rows
  cache, trigger the console-sync worker, and announce with a toast. The
  workspace listing mechanism (scope columns vs registry membership) is
  verified during implementation so the chat is guaranteed visible.

## Alternatives

- **Local/builtin catalog tool** was rejected: the catalog executor is
  dependency-light by design — no conversation context, no store access — and
  permission-store gating (allow/ask/deny states) cannot render the
  payload-review card these tools need.
- **One `create_chat` tool with a mode enum** was rejected: two verbs get
  distinct schemas, descriptions, and card headers, which is what teaches the
  model when to use which; the wiring cost is two small copies of an existing
  pattern.
- **Agent-chosen fork points** (e.g. "fork from message N") were rejected: the
  model has no usable message addressing in its context; rewind-then-fork
  composes to the same outcome.
- **Full-tree copies** (variants and inactive branches) were rejected: users
  act on the path they can see; the heavier copy buys nothing for v1.
- **Auto-sending the opening prompt** was rejected: the agent must not act as
  the user; drafts keep one approval per creation and keep the user in the
  send loop.
- **Permission-store persisted allow** was rejected: it would remove by
  configuration the default safety rail the requirement names; session-scoped
  remember is the only opt-out.
- **Batch approval of multiple creations in one card** was rejected as YAGNI:
  sequential cards are correct, rare, and each approval stays atomic.

## Consequences

- `fork_conversation_history` (new, `Chat/chat_conversation_service.py`) is
  the first Console writer of the lineage columns; the name avoids collision
  with the `fork_conversation_into_workspace` misnomer. The copy must use the
  raised-caps/uncapped tree read — silent truncation of long chats is a bug
  and an over-cap fork is a tested case.
- `restore_persisted_session` gains an `activate=False` variant
  (`create_session` already supports the flag; no-screen hydration is proven
  by `console_launch_wake`). Draft rehydration hooks into session restore.
- Shared files with the ADR-147 PR (`agent_service.py` — additive pins after
  the runtime-schema block, `agent_runtime.py`, `console_agent_bridge.py`):
  implement on top of it or rebase before opening; the routing PR rewrites
  adjacent regions.
- No new config keys: confirmation defaults are the tools' built-in policy.
- Docs: `agent-runs-and-tools.md` and a workstream blurb in the Console user
  guide document both tools, the draft-not-send contract, and the mid-turn
  snapshot boundary.
