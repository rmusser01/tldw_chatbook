# The Console: live agent chat architecture

This document describes the Console (`chat` destination): the screen layout, the send lifecycle from composer to durable turn, approvals, run states, branching, compaction, and crash recovery. The Console is both the app's default first screen and the only surface that owns streaming provider traffic.

## Authoritative files

| Layer | File | Key symbols |
| --- | --- | --- |
| Screen (view) | `UI/Screens/chat_screen.py` | `ChatScreen`, `compose_content()`, `_build_console_center()`, the 0.2 s transcript sync timer |
| Runtime holder | `Chat/console_runtime.py` | `ConsoleRuntime` — app-owned, idempotent `ensure_*` factories for store/controller/bridge/hooks; outlives every `ChatScreen` |
| Received input/admission | `Chat/console_received_intent.py`, `console_received_turn.py`, `console_received_dispatch.py` | Detached input revisions, atomic session claim/promotion, runtime-owned received Send task |
| Checked configuration | `Chat/console_configuration_capture.py`, `console_configuration_preparation.py` | Current source-owned configuration capture and finite checked preparation before claim promotion |
| Finite native work | `Chat/console_preparation_reads.py`, `console_native_commit.py` | Exact callback/handle custody through cancellation, retained save outcomes and terminal settlement |
| Controller (turn orchestration) | `Chat/console_chat_controller.py` | `ConsoleChatController`: `submit_draft()`, `_run_agent_reply()`, `retry_message()`, `request_mcp_approvals()`, `_set_run_state()` |
| Data models | `Chat/console_chat_models.py` | `ConsoleMessageRole`, `ConsoleRunStatus`, `ConsoleRunState`, `ConsoleChatMessage`, `ConsoleVariantSet`, `ConsoleDispatchRecoveryState` |
| Store (state + persistence) | `Chat/console_chat_store.py` | `ConsoleChatStore`, `ConsoleChatSession`, `append_message()`, `append_stream_chunk()`, `persist_message_if_needed()` |
| Agent bridge | `Chat/console_agent_bridge.py` | `ConsoleAgentBridge.run_reply()`, the `_StreamingModelAdapter`, fleet coordination, tool marker projection |
| Persistence service | `Chat/chat_persistence_service.py` | `ChatPersistenceService.commit_durable_turn()` — the durable turn boundary |
| Transcript grouping | `Chat/console_turn_grouping.py` | `group_console_transcript_messages()`, `ConsoleAssistantTurn` |
| Compaction | `Chat/console_context_compaction.py`, `console_context_policy.py`, `console_context_repository.py` | `CompactionPlan`, `ResolvedConsoleContextPolicy`, memory records |

Supporting modules: `console_chat_fork.py` (fork/branch), `console_prompt_queue_coordinator.py` (bounded prompt queue, ADR-098), `console_fleet_wake.py`, `console_interrupt_rounds.py`, `console_dispatch_checkpoint.py` (crash recovery).

Character presentation uses `UI/Console_Modules/character_context.py`. Stock active displays carry the original current-owner closure through the one-second run scope check into the same finite inner batch as idle display; direct/custom actions retain fresh no-argument dispatch. A stock display whose previous exact scope fingerprint names a different open conversation can skip the preliminary change-detection read: the existing stock reader selector must qualify first, and the unchanged refresh still validates fresh pair/groups/pair metadata and current ownership before publication. Missing/invalidated fingerprints, unchanged conversations, forced/direct and custom routes keep their original change detection. The existing two-second display memo, invalidation, source checks and physical callback retirement remain separate contracts (ADR126/230).

Finite preparation callbacks temporarily use at most1ms interpreter thread handoffs to reduce scheduling convoys across fresh native reads (TASK-34406.11, ADR230). The existing bookkeeping lock counts overlapping physical callbacks; last exit restores the prior value if still owned. Lower prior or later different external settings remain. This affects interpreter scheduling only: original source, permission, native ownership and cancellation checks are unchanged.

## Screen layout

Top-to-bottom inside `#console-shell`: a `DestinationHeader` with speech controls; hidden legacy compat statics kept for contract tests; a dispatch-recovery callout; the dense `ConsoleControlBar`; then the main grid — left `ConsoleRailHandle` → `ConsoleLeftRail` (3fr) → center column (13fr, hosting either the transcript region or the embedded terminal workspace) → `ConsoleInspectorRail` (4fr) → right handle. Below the grid: status chips (position from `[console] status_chips_position`), staged evidence strip, dispatch-recovery region, prompt-queue region, the `ConsoleComposerBar`, command popup, and setup modal.

Multiple chats are store-level `ConsoleChatSession`s, one per tab on the session surface's tab strip. Sessions bind to workspaces via `session.workspace_id` (default the global workspace). Streamed transcript and background-run activity are projected by `_sync_native_console_chat_ui`, including its `set_interval(0.2 s)` poll; the timer deliberately does not stop while any session still has an in-flight run. Session switches call `UI/Console_Modules/session.py::_bind_composer_to_active_session()` before the first await: they save the outgoing draft, load the selected chat's draft, and update the active-tab highlight and title immediately. The shared tab/switcher activation path also focuses the composer before awaiting; background switches preserve keyboard focus.

## Send lifecycle (dataflow)

1. The stock composer enters `UI/Console_Modules/composer_run_controls.py::_dispatch_console_draft_send`. `wiring.receive_console_visible_intent` captures detached selection and authored input revisions, then calls `ConsoleRuntime.accept_received_intent`, which claims the existing per-session slot before checked configuration preparation. Receipt grants no execution or durable acceptance. The view can show Preparing while the runtime owns the task; repeated presses cannot acquire a second claim.
2. The runtime checks the captured store/session/incarnation/binding and draft, attachment, prefill and settings revisions. Checked preparation then promotes that exact claim atomically into the existing controller submission, with no release/reacquire gap. `submit_draft` fences shutdown/capture-quiescence, registers the submit task, and appends an optimistic user echo with `persist=False`; a blocked send leaves no durable row. Custom/legacy complete-request entry retains its original fallback and callback affinity.
3. If the origin is manual, the controller fires the `UserPromptSubmit` hook (see [console-run-hooks.md](./console-run-hooks.md)). A blocking hook refuses the send (run state `BLOCKED`, hook-origin SYSTEM row); captured stdout becomes a persisted hook-context row.
4. On proceed, the store persists the echo (creating the durable conversation), appends the hook context row, and appends an empty assistant placeholder.
5. Turn acceptance notifies the prompt queue coordinator, arms the one-shot `Stop` hook, and clears the composer.
6. `_run_agent_reply` freezes workspace/scratch/project state, composes the tool registry and review hooks, and executes `bridge.run_reply(...)` on a worker thread via `asyncio.to_thread`.
7. The bridge builds an `AgentService` with `review_tool_calls`, `guard_tool_calls` (hooks), `post_tool_call`, and an `on_step` callback. The agent loop calls the bridge's streaming adapter per model round; text deltas stream straight into the store from the worker thread; tool executions surface as TOOL markers via `on_step`.
8. **Approvals**: before dispatch the review hook collects pending calls and makes one `request_mcp_approvals` round trip — it mints a round id, mounts a `ChatApprovalCard` on the UI thread (or parks it for background sessions), and polls the event. Unanswered keys fail closed to deny/timeout so audit rows never claim a human Deny. Despite the "mcp" name, builtin tools use the same path.
9. Turn terminal: the controller maps the agent outcome (`RUN_DONE`/`RUN_CANCELLED`/`RUN_ERROR`/`RUN_STUCK`) to final content and stamps the run state — which discards pending-approval flags, retries deferred wakes, fires the armed `Stop` hook, and schedules micro-compaction on a fresh COMPLETED transition.
10. Persistence: the store flushes terminal content/thinking/usage through `ChatPersistenceService.commit_durable_turn` — the sole outer `BEGIN IMMEDIATE` that creates the conversation, library policy, and context policy atomically, returning a dispatch checkpoint. The finite native owner retains the original callback, physical handle and actual save outcome through repeated cancellation; shutdown drains issued work before creator close. A canceled accepted save settles through its retained terminal/recovery owner, without replaying an uncertain write.
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

The recovery subtree in `UI/Console_Modules/provider_continuation_recovery.py` defers `ProviderContinuationTranscriptRegion.sync_recovery()` until mounted. Its `on_mount` queues a fresh builder read after Textual sets `is_mounted`, so a partial-mount poll cannot query pending action buttons or leave an old recovery state behind.

## Config keys (`[console]`, parsed in `config.py`)

Notable keys: `workspace_root` (compatibility-only for Console file authority — see [console-file-authority.md](./console-file-authority.md)), `compaction_mode`, `compaction_trigger_ratio`, `compaction_target_ratio`, `compaction_summary_max_tokens`, `compaction_auxiliary_timeout_seconds`, `micro_compaction_every_turns`, `compaction_failure_behavior`, `agent_max_model_turns` (2000), `agent_max_steps` (25000), `agent_max_wall_seconds` (86400), `agent_max_tool_call_seconds`, `raw_cli_permitted`, `status_chips_position`, `conversation_budget_mode`, paste-collapse keys.

## Boundaries

- **Screen** — layout, widgets, focus, the repaint tick, card mounting and disposable received-input feedback. The accepted sole stock unpromoted manual Preparing path can narrow presentation only with exact receipt/source/visit and completed-FULL witnesses; changed/custom/queued/background paths retain FULL work and trailing FULL replay. General polling remains Proposed (ADR-231). Never owns accepted agent work.
- **Runtime and received input owners** — one existing per-session admission slot, immutable captured inputs, checked preparation, atomic promotion and retained cancellation outcomes. Views hold no execution authority.
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

ADR-052 (memory and compaction policy), ADR-067 (indefinite approval waits), ADR-082 (per-chat scratch), ADR-092 (fork and authority boundary), ADR-094 (turn lifetime), ADR-098 (bounded prompt queue), ADR-100 (active path), ADR-134/135 (fleet budgets/recovery), ADR-145 (live thinking), ADR-150 (fork & spawn), ADR-211 (destinations and bounded starts), [ADR-230](../../backlog/decisions/230-console-send-preparation-and-io-ownership.md) (received intent and finite preparation), and [ADR-231](../../backlog/decisions/231-console-polling-and-full-state-reconciliation.md) (accepted bounded manual Preparing amendment; general polling Proposed). Numerical latency, visible paint and remaining platform qualification stay open. Spec: `Docs/superpowers/specs/2026-10-02-console-chat-destinations-and-starts-design.md`. User guide: `Docs/User_Guide/console.md` and `Docs/User_Guide/console/`.

## Verified gotchas

1. Streaming deltas reach the transcript through `_sync_native_console_chat_ui` (the 0.2 s timer plus explicit sync calls); there is no streaming event class to subscribe to. Session-switch composer binding is synchronous.
2. `asyncio.to_thread` survives task cancellation, so Stop is cooperative (cancel events), not a thread kill.
3. Session-only TOOL-marker fields (`tool_output_full`, `tool_diff`, `raw_cli_presentation`, …) are never persisted or sent to providers.
4. `request_mcp_approvals` is owner-agnostic — builtin tools ride the same approval card path as MCP tools.
5. The review hook clears per-run decision stamps at entry so a raising approval round can never leak a prior turn's approval.


### Learned response rules (ADR232)

The app-owned `ConsoleRuntime.ensure_response_rules` lazily binds the shared rule repository, helper pool, queue and controller on first native generation or manager access. `set_chat_controller` installs the lazy factory even for non-screen assembly. `response_rules/generation.py` holds successful completion until required lifecycle settlement and assessment; app shutdown drains retained preparation, transport and finite SQLite callbacks before retiring their owners. `response_rules/chat_state.py` stores native source fences and body-free tool evidence through `ConsoleChatStore` facades. `console_tool_result_observation.py` composes the existing Watchlists receipt observer with definitive rule facts.

`response_rules/goal_input.py` validates a real, accepted coordinator capability and the exact runtime custody handoff/parent version; ordinary SYSTEM content is not rule authority. A settled Goal's assessment is retained, but automatic repair is explicitly refused because ADR227 has no second-run continuation authority. Native/manual and genuine external-hook feedback otherwise share `ConsolePromptQueueCoordinator.admit_machine_followup` and one atomic receipt/checkpoint contribution.

The same scoped manager is used by `/rules`, Chat settings and canonical F4 Settings. It loads before or after a Chat exists; source/Workspace/profile changes revoke current tests, and Save uses the exact tested revision and binding CAS. Source CSS is `_console_response_rules.tcss`, included by the existing CSS builder. Successful Save refreshes the selected row, details and tested definition from the exact committed revision, which remains the input to subsequent promotion or Disable/Enable.

Rule phase events refresh the visible run chip directly from current in-memory state. They keep the control bar's nonblocking configuration deferral; rebuilding the hidden mode bar would re-enter configuration during a pause. Stop uses the already-bound controller for the viewed Chat and, for drafting/testing/checking, relies on that same rule event to repaint. It does not construct a controller or refresh provider selection to cancel admitted work.

Disable, Exclude and Delete first revoke affected assessment authority under a short current-user-scope fence, even if later persistence fails. That fence is released before writer admission. The mutation then rechecks the user scope and binding CAS, and invalidates again to fence work started during the wait. Rule writes acquire SQLite writer admission before the store/currentness mutation locks, recheck source and binding CAS after admission, and retain the short fences through commit or rollback. Temporary publication follows successful settlement; transcript adoption still uses its caller-owned transaction and completion observer. Waiting for another SQLite writer therefore cannot hold the synchronous Stop/source-revocation locks.

Recovery-card synchronization runs before the transcript message-token gate: pre-send holds and continuation changes must remain visible even when message projection is unchanged. The expensive message pipeline retains its existing gate.

## Physical execution owner (ADR-244)

`Chat/console_execution_owner.py` holds a nonblocking OS lease for the secured
physical run-store parent and leaf. `ConsoleRuntime` acquires it lazily before
constructing the sibling `agent_runs.db`; different chat filenames in that
parent share this resource. The empty runtime holder and saved-history readers
do not acquire it. Settings and retention writers share the App's owner; manual
worktree recovery shares its bridge's owner. A lost/unsafe lease refuses writes.

The executing App/child retains the lease through physical SQLite operations.
Disposal releases it only after the runtime drains, change-review publisher and
cleanup prove retirement. Bounded or failed cleanup retains the owner until
process exit; cancelling a waiter or losing a parent is never takeover authority.
See [ADR-244](../../backlog/decisions/244-shared-console-ownership-and-mobile-supervision.md).


## Served lifecycle ownership (TASK-36761, ADR-244)

Served mode selects `Web_Server/console_web_driver.py` through `TldwCli` only for
`CHATBOOK_SERVED_CHILD`. The App, ConsoleRuntime, native stores and execution-owner
lease survive terminal disconnect/replacement; a generation-fenced full repaint
restores the terminal without another input thread. Parent host ownership and the
saved-conversation reservation seam are documented in [web-server.md](web-server.md).
Task9 connects remote saved-open/save callbacks; terminal native hydration and
persistence retain their existing paths.

`app_served.py` watches the child control connection independently of Canvas
exposure. Genuine parent/control loss closes boot admission and automatic wakes,
marks the Console as tearing down, awaits the existing runtime disposal/shutdown,
and exits. Runtime physical cleanup retains the lease on failure or cancellation;
the parent only permits a new host after observing actual process exit. Browser
revocation and Canvas disable do not enter this native shutdown path.

## Command-owned input and exact native controls (ADR-244)

`Chat/console_remote_commands.py::ConsoleRemoteCommands` is the runtime command
adapter for an existing `ConsoleHostIdentity` / `ConsoleTarget`. Its supported
actions are `send`, `steer`, `redirect`, `stop`, `subagent_guidance` and
`subagent_stop`, plus exact retained `decision` answers. Unsupported actions refuse. This seam does not select a desktop
tab or construct a second controller. Browser transport wiring and receipt
observation are separate owners.

Send requires text and all five last-confirmed target revisions:
`settings_revision`, `generation_settings_revision`, `context_policy_revision`,
`identity_revision` and `conversation_binding_revision`. It builds a
`ConsoleSessionInputSnapshot` with `input_origin="command"` and a command ID,
empty attachments/prefill, and no staged-RAG launch. Its native source witnesses
remain currentness requirements; desktop typing, prefill, attachment edits and
view changes cannot invalidate or clear this command. Composer snapshots retain
the original complete input comparisons. The same received-turn slot is claimed
with `draft_revision=None`, then the existing checked configuration, hook review,
permissions, native preparation and atomic promotion run under runtime custody.
The synchronous resident configuration proof precedes receipt; complete async
capture still follows the existing Preparing claim. Settings/binding changes
before promotion refuse stale input.

Command origin and its repr-hidden native input witness follow
`ConsoleTurnCustodyRequest` through the existing prompt queue; configuration
changes while queued are rechecked before its claim and native promotion. Command requests capture no desktop prefill and bypass only the queue's
composer rider checks; normal queue revision, context, permission, capacity and
continuation ownership still apply. Native `preserve_composer` behavior protects
drafts and acceptance callbacks through prepared continuations and durable
acceptance. Buddy's existing composer guards remain unchanged.

`send()` returns the custody turn ID while it is Preparing. `execute()` returns
body-free `state`, optional `native_id`, and `reason`: Send is `pending`, guidance
is `queued`, accepted Stop is `accepted`, and native refusal is `refused`.
Neither queued guidance nor accepted Stop claims mailbox consumption or physical
worker exit. Main controls use explicit-session `steer_run`, `redirect_run` and
`stop_run`; existing active-tab methods delegate to those native controls. A
remote Stop needs an exact existing turn or live run and cannot stop a successor.


## Retained remote projection and decisions (TASK-36763, ADR-244)

`Chat/console_remote_projection.py::ConsoleRemoteProjection` reads the existing
runtime/store/round owners without a ChatScreen, changing selection, hydration,
or provider construction. Discovery combines live identities with canonical saved
conversation listings; an unhydrated live persisted target pages native metadata
first, then canonical bounded history, including when fleet/decisions remain live.
The history cursor fences that metadata and canonical content revisions; a change
restarts from native metadata with reset. Saved-only history routing remains Task9.
Each non-null target is checked against host, session incarnation and supplied
turn/run/handle/round identities before reading and again before returning.

Snapshot sections are `chats`, `messages`, `fleet`, `decisions`, and `recoveries`,
with host identity, native `sequence`, `reset_required`, and `next_cursor`. Chat
rows include the five command `expected_revisions`. Pages contain at most 100
rows and 64 KiB; large message text continues at a character offset. Content/scope
revision changes reset continuation. System/automatic instruction bodies, hidden
thinking, quarantined generation text, provider credentials and full DB rows are
excluded. Recovery and unfaithful reviews expose only pending/Open terminal state.

`InterruptRoundHost.remote_rounds` snapshots native ownership under its lock and
formats outside that lock. Ordinary tool approval summaries share the native
pure `Chat/approval_review.py` destination-first, bounded redaction policy with
the terminal card; phone serialization/redaction failure hides the review and
refuses answers. Raw shell, hook/skill/worktree/chat-create and project instruction
confirmations remain unsupported pending terminal reviews. Ordinary approvals
and validated user questions resolve through their original kind-specific native
resolvers; exact kind/session/incarnation/run/round scope and bounded completed
outcomes preserve one original winner. No remote policy grants or hook bypasses
are added.

Answerability belongs to a registered live subscription and an exact rendered
round. Registration is capped at 64 clients and rendered-round claims at 100.
Snapshot reads grant no answerability. Disconnect releases only that client's
claims, so another phone's visible round keeps its native human-input clock
answerable. Resolution and runtime disposal remove claims; native clock refresh
matches the rendered round, not merely the session/kind. Task9 supplies the
parent-generated authenticated subscription IDs and disconnect/revocation wiring.
