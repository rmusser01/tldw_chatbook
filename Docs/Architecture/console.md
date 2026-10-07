# The Console: live agent chat architecture

This document describes the Console (`chat` destination): the screen layout, the send lifecycle from composer to durable turn, approvals, run states, branching, compaction, and crash recovery. The Console is both the app's default first screen and the only surface that owns streaming provider traffic.

## Authoritative files

| Layer | File | Key symbols |
| --- | --- | --- |
| Screen (view) | `UI/Screens/chat_screen.py` | `ChatScreen`, `compose_content()`, `_build_console_center()`, the 0.2 s transcript sync timer |
| Runtime holder | `Chat/console_runtime.py` | `ConsoleRuntime` — app-owned, idempotent `ensure_*` factories for store/controller/bridge/hooks; outlives every `ChatScreen` |
| Controller (turn orchestration) | `Chat/console_chat_controller.py` | `ConsoleChatController`: `submit_draft()`, `_run_agent_reply()`, `retry_message()`, `request_mcp_approvals()`, `_set_run_state()` |
| Data models | `Chat/console_chat_models.py` | `ConsoleMessageRole`, `ConsoleRunStatus`, `ConsoleRunState`, `ConsoleChatMessage`, `ConsoleVariantSet`, `ConsoleDispatchRecoveryState` |
| Store (state + persistence) | `Chat/console_chat_store.py` | `ConsoleChatStore`, `ConsoleChatSession`, `append_message()`, `append_stream_chunk()`, `persist_message_if_needed()` |
| Agent bridge | `Chat/console_agent_bridge.py` | `ConsoleAgentBridge.run_reply()`, the `_StreamingModelAdapter`, fleet coordination, tool marker projection |
| Persistence service | `Chat/chat_persistence_service.py` | `ChatPersistenceService.commit_durable_turn()` — the durable turn boundary |
| Transcript grouping | `Chat/console_turn_grouping.py` | `group_console_transcript_messages()`, `ConsoleAssistantTurn` |
| Compaction | `Chat/console_context_compaction.py`, `console_context_policy.py`, `console_context_repository.py` | `CompactionPlan`, `ResolvedConsoleContextPolicy`, memory records |

Supporting modules: `console_chat_fork.py` (fork/branch), `console_prompt_queue_coordinator.py` (bounded prompt queue, ADR-098), `console_fleet_wake.py`, `console_interrupt_rounds.py`, `console_dispatch_checkpoint.py` (crash recovery).

## Screen layout

Top-to-bottom inside `#console-shell`: a `DestinationHeader` with speech controls; hidden legacy compat statics kept for contract tests; a dispatch-recovery callout; the dense `ConsoleControlBar`; then the main grid — left `ConsoleRailHandle` → `ConsoleLeftRail` (3fr) → center column (13fr, hosting either the transcript region or the embedded terminal workspace) → `ConsoleInspectorRail` (4fr) → right handle. Below the grid: status chips (position from `[console] status_chips_position`), staged evidence strip, dispatch-recovery region, prompt-queue region, the `ConsoleComposerBar`, command popup, and setup modal.

Multiple chats are store-level `ConsoleChatSession`s, one per tab on the session surface's tab strip. Sessions bind to workspaces via `session.workspace_id` (default the global workspace). The screen refreshes through a `set_interval(0.2 s)` poll (`_sync_native_console_chat_ui`) — UI updates are **poll-based, not event-based** — and the timer deliberately does not stop while any session still has an in-flight run.

## Send lifecycle (dataflow)

1. The composer posts the draft; the screen calls `controller.submit_draft(draft, session_id, origin=MANUAL)`.
2. `submit_draft` fences shutdown/capture-quiescence, registers the submit task, and appends an optimistic user echo with `persist=False` — a blocked send leaves no durable row.
3. If the origin is manual, the controller fires the `UserPromptSubmit` hook (see [console-run-hooks.md](./console-run-hooks.md)). A blocking hook refuses the send (run state `BLOCKED`, hook-origin SYSTEM row); captured stdout becomes a persisted hook-context row.
4. On proceed, the store persists the echo (creating the durable conversation), appends the hook context row, and appends an empty assistant placeholder.
5. Turn acceptance notifies the prompt queue coordinator, arms the one-shot `Stop` hook, and clears the composer.
6. `_run_agent_reply` freezes workspace/scratch/project state, composes the tool registry and review hooks, and executes `bridge.run_reply(...)` on a worker thread via `asyncio.to_thread`.
7. The bridge builds an `AgentService` with `review_tool_calls`, `guard_tool_calls` (hooks), `post_tool_call`, and an `on_step` callback. The agent loop calls the bridge's streaming adapter per model round; text deltas stream straight into the store from the worker thread; tool executions surface as TOOL markers via `on_step`.
8. **Approvals**: before dispatch the review hook collects pending calls and makes one `request_mcp_approvals` round trip — it mints a round id, mounts a `ChatApprovalCard` on the UI thread (or parks it for background sessions), and polls the event. Unanswered keys fail closed to deny/timeout so audit rows never claim a human Deny. Despite the "mcp" name, builtin tools use the same path.
9. Turn terminal: the controller maps the agent outcome (`RUN_DONE`/`RUN_CANCELLED`/`RUN_ERROR`/`RUN_STUCK`) to final content and stamps the run state — which discards pending-approval flags, retries deferred wakes, fires the armed `Stop` hook, and schedules micro-compaction on a fresh COMPLETED transition.
10. Persistence: the store flushes terminal content/thinking/usage through `ChatPersistenceService.commit_durable_turn` — the sole outer `BEGIN IMMEDIATE` that creates the conversation, library policy, and context policy atomically, returning a dispatch checkpoint.
11. Transcript render: `group_console_transcript_messages` folds each ASSISTANT message with its immediately following agent-owned TOOL markers into a `ConsoleAssistantTurn`; user-owned markers stay standalone and close the pending turn.

Streaming details, transforms (skills → chat dictionaries → world info), swipes/branching, and the non-agent plain path are covered in [chat-pipeline.md](./chat-pipeline.md).

## Run states and gating

`ConsoleRunStatus`: idle / validating / streaming / checking_citations / completed / blocked / stopped / failed / retrying. New sends are only allowed from idle/blocked/completed/failed/stopped; Stop is only allowed during streaming/citation-checking. `_set_run_state` is the single writer of run state.

## Approvals

Approval requests surface as `ChatApprovalCard` widgets in the session surface's task-card region. Card decisions (`approve_once`, `approve_session`, `always_allow`, deny, timeout) resolve the pending round; permission persistence itself flows through the MCP permission store (see [mcp-hub.md](./mcp-hub.md)). Approval waits are indefinite by default (`[mcp] approval_timeout_seconds = 0`, ADR-067); headless runs return all keys unresolved-deny without a card.

## Branching, rewind, and variants

- **Regenerate/swipes** fork a sibling node under the same parent; the anchor branch stays stored and resumable. See [chat-pipeline.md](./chat-pipeline.md) for the full mechanism.
- **Fork/branch of conversations**: `console_chat_fork.py` fences whole-conversation copies (`ChatPersistenceService.fork_console_conversation_bundle`); authority boundaries are ADR-092.
- **Image variants** remain a per-message variant set (`ConsoleVariantSet`, `store.add_variant`) browsed with variant-previous/next.

## Conversation memory and compaction (ADR-052)

Four separately-owned concepts: the model context window (server/model data with a 32,000-token fallback), the request builder's reservation/safety windowing, global defaults in Settings, and per-conversation overrides plus memory records. Compaction is tri-state (`[console] compaction_mode` = ask / automatic / off): trigger at token ratio 0.80, target 0.55; the summarizer runs as a content-free-ledger auxiliary call and is never inserted as a chat message; failure behavior is `stop_and_ask` or `omit_older_context`. Manual `/rewind` summarize actions create immutable prefix/range memory records. Micro-compaction runs fire-and-forget on a per-session cadence (`micro_compaction_every_turns`, 0 = off) with an in-flight guard.

## Crash and dispatch recovery

Interrupted turns persist a `ProviderContinuationCheckpoint`; the resume path validates the target/settings and stamps visible warnings, offering retry or discard. `ConsoleDispatchRecoveryState` plus the recovery region present retry/discard from the durable `ConsoleDispatchCheckpoint`. The 0.2 s transcript sync timer plus a 1 s "survivor tick" keep sub-agent activity visible after the primary turn settles.

## Config keys (`[console]`, parsed in `config.py`)

Notable keys: `workspace_root` (compatibility-only for Console file authority — see [console-file-authority.md](./console-file-authority.md)), `compaction_mode`, `compaction_trigger_ratio`, `compaction_target_ratio`, `compaction_summary_max_tokens`, `compaction_auxiliary_timeout_seconds`, `micro_compaction_every_turns`, `compaction_failure_behavior`, `agent_max_model_turns` (2000), `agent_max_steps` (25000), `agent_max_wall_seconds` (86400), `agent_max_tool_call_seconds`, `raw_cli_permitted`, `status_chips_position`, `conversation_budget_mode`, paste-collapse keys.

## Boundaries

- **Screen** — layout, widgets, focus, the repaint tick, card mounting. Never runs agent work.
- **Controller** — the send lifecycle: origin semantics, hooks fire sites, approvals bridging, run-state map (single writer), retry/recovery, queue acceptance.
- **Store** — session/message state (thread-safe; written from both the UI loop and the bridge worker), the fork/branch tree, draft custody, and the `ConsoleChatPersistence` Protocol seam so `Chat/` never imports the DB layer.
- **Bridge** — one `run_reply` execution: `AgentService` construction, tool provider composition, live step feeds, fleet coordination, marker projection.

## Failure behaviors

| Case | Behavior |
| --- | --- |
| Hook blocks a manual send | Run state BLOCKED; no durable user row; hook-origin SYSTEM row explains why |
| Approval card unanswered + timeout 0 | Waits indefinitely; cancel still works; headless runs fail closed per key |
| Approval round trip raises | Per-run decision stamps were cleared at entry — no stale prior-turn `approve_once` can leak |
| Provider continuation conflict | Send blocked with a recovery-required notice offering retry/discard |
| Stop during stream | Cooperative cancel via cancel events (the worker thread is never killed); late chunks after stop are dropped silently |
| Micro-compaction failure | Fire-and-forget; logged at debug only; never surfaces as an error |
| Stop/SubagentStop hook observer failure | Swallowed with a logged warning; cannot block settlement |

## Governing decisions

ADR-052 (memory and compaction policy), ADR-067 (indefinite approval waits), ADR-082 (per-chat scratch), ADR-092 (fork and authority boundary), ADR-094 (turn lifetime), ADR-098 (bounded prompt queue), ADR-100 (active path), ADR-134/135 (fleet budgets/recovery), ADR-145 (live thinking), ADR-150 (fork & spawn), ADR-211 (destinations and bounded starts). Spec: `Docs/superpowers/specs/2026-10-02-console-chat-destinations-and-starts-design.md`. User guide: `Docs/User_Guide/console.md` and `Docs/User_Guide/console/`.

## Verified gotchas

1. UI updates are poll-based (0.2 s `set_interval`), not message-driven — there is no streaming event class to subscribe to.
2. `asyncio.to_thread` survives task cancellation, so Stop is cooperative (cancel events), not a thread kill.
3. Session-only TOOL-marker fields (`tool_output_full`, `tool_diff`, `raw_cli_presentation`, …) are never persisted or sent to providers.
4. `request_mcp_approvals` is owner-agnostic — builtin tools ride the same approval card path as MCP tools.
5. The review hook clears per-run decision stamps at entry so a raising approval round can never leak a prior turn's approval.
