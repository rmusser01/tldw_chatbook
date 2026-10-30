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

Frozen automatic Library preparation is delegated to `Chat/console_library_preparation.py` through the controller's unchanged entry point. Controller and runtime retain ownership of preparation transitions, timeout configuration and recovery.

Character presentation uses `UI/Console_Modules/character_context.py`. Stock active displays carry the original current-owner closure through the one-second run scope check into the same finite inner batch as idle display; direct/custom actions retain fresh no-argument dispatch. A stock display whose previous exact scope fingerprint names a different open conversation can skip the preliminary change-detection read: the existing stock reader selector must qualify first, and the unchanged refresh still validates fresh pair/groups/pair metadata and current ownership before publication. Otherwise a stock display check (active or idle) decides and reloads in that one finite inner batch (`_check_recent_batch`, TASK-33620.15.6): the batch ends after its first fresh pair when that still matches the displayed fingerprint, and otherwise loads and fence-reads as a refresh does, so a changed check opens one handle (one ADR-125 helper process) instead of two. Nothing publishes until the callback retires; a reload (or its load error) then advances the generation as `_begin` would and publishes once, with no intermediate Refreshing state. As in the original batch, an owner that moved on or a qualified body that changed during the read gets no publication, and a scope that moved during the load is reloaded by the original refresh. An unchanged check makes no more UI-loop handoffs than the original probe: one on-loop midpoint inside its pair, then the ambient scope confirmed on the loop after retirement, with no reader owner check. A check whose display shows a settled fingerprint (TASK-35961) qualifies on the loop only what runs there: the display entry's own checks, the two identity readers its selection calls (`_settled_character_refresh_readers`), `_await_database_work`, which then receives its finite callback on the loop, and, in the midpoint of each pair of its first scope read (every attempt), the readers that midpoint calls (`require_loop_readers`); the worker qualifies every reader before and after each read, as before, and a reader changed after the selection is refused there before anything is read (no handle opens) and the check takes the original route. Its UI-loop CPU is then no more than the original probe route's. For a display owner that is still current, a first read that did not settle, or a callback that failed (in that read or after it), is handed to the caller as the probe's outcome (`_CharacterScopeProbed`): the caller takes the original route's branch for it without probing the scope again on a handle of its own (TASK-35961). An unchanged read whose ambient scope or displayed fingerprint changed by its retirement, or a refresh dispatch customized during it, takes the original probe-then-refresh route; an owner that moved on gets nothing, as after the original probe. Every outcome that neither finds the scope unchanged nor publishes a reload drops the last-successful-refresh marker, a cancellation or a raised error included, as the original route's `refresh()` did on entry, so the idle two-second memo never serves a list the check did not reload. Forced/direct and custom routes keep their original change detection. The existing two-second display memo, invalidation, source checks and physical callback retirement remain separate contracts (ADR126/230).

Living character portraits reuse the disposable `CharacterExpressionAvatar` in the
Character rail. `UI/Console_Modules/character.py` resolves local performance
attachments off-thread on activation/source/invalidation changes; immutable spec
identity includes profile and attachment revision/tombstone/absence. Visible frames
read only the current owned speech snapshot, never streaming text or SQLite. Blink
and blinking obey existing motion preferences. Whole-portrait translation is
disabled by the zero idle-offset token (TASK-36699). Source/profile/
session/renderer changes fence old preparation; unchanged forced refresh preserves
the mounted animation. Live emote feed consumption is serialized independently of
ordinary source requests, preserving each explicit beat. Cheap actor/session/manual/
motion facts are checked before painting without copying the transcript. See
[portrait ownership and preparation](./character-chat.md#local-portrait-performance-attachments).

Finite preparation callbacks temporarily use at most1ms interpreter thread handoffs to reduce scheduling convoys across fresh native reads (TASK-34406.11, ADR230). The existing bookkeeping lock counts overlapping physical callbacks; last exit restores the prior value if still owned. Lower prior or later different external settings remain. This affects interpreter scheduling only: original source, permission, native ownership and cancellation checks are unchanged.

## Screen layout

Top-to-bottom inside `#console-shell`: a `DestinationHeader` with speech controls; hidden legacy compat statics kept for contract tests; a dispatch-recovery callout; the dense `ConsoleControlBar`; then the main grid — left `ConsoleRailHandle` → `ConsoleLeftRail` (3fr) → center column (13fr, hosting either the transcript region or the embedded terminal workspace) → `ConsoleInspectorRail` (4fr) → right handle. Below the grid: status chips (position from `[console] status_chips_position`), staged evidence strip, dispatch-recovery region, prompt-queue region, the `ConsoleComposerBar`, command popup, and setup modal.

Multiple chats are store-level `ConsoleChatSession`s, one per tab on the session surface's tab strip. Sessions bind to workspaces via `session.workspace_id` (default the global workspace). Streamed transcript and background-run activity are projected by `_sync_native_console_chat_ui`, including its `set_interval(0.2 s)` poll; the timer deliberately does not stop while any session still has an in-flight run. Session switches call `UI/Console_Modules/session.py::_bind_composer_to_active_session()` before the first await: they save the outgoing draft, load the selected chat's draft, and update the active-tab highlight and title immediately. The shared tab/switcher activation path also focuses the composer before awaiting; background switches preserve keyboard focus.

Character-picker creation and character/persona Start Chat handoffs align the
workspace registry with the selected conversation through
`ConsoleWorkspaceController._set_active_workspace_for_console_session` before
publishing the activated chat (TASK-37620). A newly created global chat therefore
selects Default; a repurposed pristine tab keeps its existing workspace. This
prevents model-picker Apply/Cancel from resuming into the previous workspace's
chat. Resume reconciliation still honors genuine cross-screen workspace changes.

`ConsoleTranscript.refresh_messages()` keeps its awaited row reconciliation inside
the existing refresh lock and Textual's `App.batch_update()` (TASK-37145). Native
remove-before-mount teardown can span several repaint opportunities; the batch
retains the last committed compositor frame until the replacement can commit.
This closes TASK-33620.5's documented acknowledgement-to-echo residue without
changing the synthetic acknowledgement or exact store echo identities. The batch
unwinds on caller cancellation; TASK-34982's `finish_removal` still lets started
native removal finish independently. Row reuse, ordering and attachment checks
retain their existing owners. The held-Unmount regression in
`Tests/UI/test_console_transcript_ack_repaint.py` observes committed headless
compositor frames and repaint resumption after interval cancellation; it makes
no terminal-write or natural elapsed-blank-time claim. Runtime/store authority
remains under ADR-094 and accepted ADR-230; general polling is unchanged.

Project-instruction authority callbacks run directly when already on Textual's owning thread; worker callbacks use `call_from_thread` synchronously. A dispatch confirmation that never becomes answerable is cancelled after its finite notice deadline, preserving the chat's project-instruction preference. Once-mounted decisions retain the existing detached-view pause contract. Unsent-turn Restore/Discard releases only the exact refused turn's paused preparation -- a Library (retrieval) pause, or the destination/persistence pause a card Retry or Send once re-paused into (`SHELF_RELEASABLE_PAUSES`, TASK-33621.20) -- keeping the composer's current draft; newer preparations and authored input remain owned by their own flows. While the turn's transcript card is acting on it, the shelf does not offer it; teardown discards without releasing, because close and shutdown abandon every live preparation. Refusal and unexpected settlement diagnostics record status/error categories without draft, instruction, provider-error or endpoint bodies.

Workspace project guidance (ADR-241) uses the owning run's admitted roots and
final advertised readers. It prompts exploration for natural project questions
without scanning folders on attachment or changing Library settings. Initial and
retry dispatch retain their owning session's file aliases and Library scope when
another tab becomes visible, in both native and fenced tool protocols.

Console evidence formatting permits authorized file evidence alongside Library
references. File claims name their paths; `[S#]` labels refer only to the supplied
Library bundle. A mixed answer can omit Library labels when it uses no Library
evidence: canonical Console citation repair does not manufacture attribution.
Unknown and malformed labels still require repair. Dedicated Library answers and
generic local capture retain their stricter supplied-evidence/marker defaults.
Saved Never/Automatic and Blocked/Allowed values, manual staging, evidence capture
identity, and conversation/workspace scope intersection retain ADR-079 semantics.
This guidance release implements no project index; the optional folder-bound
cache described by ADR-241 requires separate implementation and qualification.
The [guidance qualification receipts](../superpowers/qa/2026-10-20-workspace-project-context/guidance/README.md)
record actual model/tool outcomes, saved policies and full Console snapshots,
including platform and harness limits.

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

Native handoff freezes two identities before admission (`console_library_destination.resolve_console_destination`): the exact selected provider and the canonical execution provider/protocol/full credential-free API base. The latter tuple lives in destination JSON V2 inside the existing dispatch column; the display origin stays origin-only. Ephemeral session-only bases never enter that tuple. `ConsoleTurnContext` copies it, and dispatch retry compares the entire destination. The 2048-byte destination limit is unchanged.

The bridge binds each provider candidate to its actual resolution as continuation JSON V3 before the runtime retains it. A child uses its own effective resolution. Store and repository validate the run target and accepted destination before signing or atomic handoff/CAS; after dispatch deletion the private assistant checkpoint retains the exact selection. Complete private history and explicit Resume therefore reject a different custom slug even at the same URL/model. V1/V2 bytes remain readable; legacy unbound virtual destinations cannot prove the original selection and refuse host recovery. Direct built-in legacy recovery remains supported, and a request target without `selected_provider` is an unbound legacy target that only a custom-hosted selection refuses (ADRs 063, 147, 179). A destination or dispatch checkpoint without a continuation tuple never fails the send: the run's continuation is ephemeral (see `console_continuation_binding`).

`AgentService._make_call_model` attaches the full live continuation once, at its newest exact native batch in a copied send payload. Earlier uniquely matching batches for that same owner lose only their redundant marker. Assistant/tool rows, call IDs, argument strings, results, steering order, other owners and original runtime history remain intact; malformed, duplicate or foreign markers are not normalized away. The strict request-owner guard, private token accounting and verifier protection of all checkpoint call IDs remain unchanged.

Capture On uses `LiveNativeContinuationCustody` only for the current host invocation. After projected persistence succeeds, the optional runtime acknowledgment binds the exact runtime checkpoint and each capped model result to the store's projected, optionally sealed checkpoint. The current generation, active path, message version and checkpoint are checked under the generation scope and again in the trace transaction. The issued surface retains one current raw reference with diagnostic representations suppressed; cold trace bytes cannot recreate that authority.

For a later native batch, `console_trace_service` proves the unchanged raw prefix, exact new call/result suffix, other continuation groups and frozen policy. It atomically replaces one continuation node and appends the new messages. `console_trace_continuation` stores a `native-continuation-round-v1` artifact containing only the new projected round and bounded metadata; the existing replacement edge identifies its predecessor. Both durable comparisons and the native reader resolve that node-aware chain with checkpoint limits. The ordinary full base stays readable. This makes new trace artifact and metadata storage linear in unique captured rounds; current assistant checkpoints and reconstruction CPU retain their existing costs. Fork/GC reachability follows the existing edges. Final-owner sweeping uses an operation-owned savepoint and temporary FK deferral for the segment/event cycle, checks the foreign keys of every table that references a swept trace table (not the whole database) before release and restores the caller's mode on success or rollback (ADR097).

Explicit Resume projects only assistant text, exact native calls and existing results through `ConsoleProviderGateway.expand_provider_continuation`. After existing target, runtime and permission checks, `begin_generation_attempt` rechecks the active checkpoint and persisted owner version under the generation scope before resetting live attempt bookkeeping; it preserves partial text, private bytes and durable versions. Executing, foreign or stale owners refuse. It does not retry or activate automatically. A final-only response without prior native state still follows ordinary text settlement.

The recovery subtree in `UI/Console_Modules/provider_continuation_recovery.py` defers `ProviderContinuationTranscriptRegion.sync_recovery()` until mounted. Its `on_mount` queues a fresh builder read after Textual sets `is_mounted`, so a partial-mount poll cannot query pending action buttons or leave an old recovery state behind.

## Config keys (`[console]`, parsed in `config.py`)

Notable keys: `workspace_root` (compatibility-only for Console file authority — see [console-file-authority.md](./console-file-authority.md)), `compaction_mode`, `compaction_trigger_ratio`, `compaction_target_ratio`, `compaction_summary_max_tokens`, `compaction_auxiliary_timeout_seconds`, `micro_compaction_every_turns`, `compaction_failure_behavior`, `agent_max_model_turns` (2000), `agent_max_steps` (25000), `agent_max_wall_seconds` (86400), `agent_max_tool_call_seconds`, `raw_cli_permitted`, `status_chips_position`, `conversation_budget_mode`, paste-collapse keys.

## Boundaries

- **Screen** — layout, widgets, focus, the repaint tick, card mounting and disposable received-input feedback. The accepted sole stock unpromoted manual Preparing path can narrow presentation only with exact receipt/source/visit and completed-FULL witnesses; changed/custom/queued/background paths retain FULL work and trailing FULL replay. General polling remains Proposed (ADR-231). Never owns accepted agent work.
- **Runtime and received input owners** — one existing per-session admission slot, immutable captured inputs, checked preparation, atomic promotion and retained cancellation outcomes. Views hold no execution authority.
- **Controller** — the send lifecycle: origin semantics, hooks fire sites, approvals bridging, run-state map (single writer), retry/recovery, queue acceptance.
- **Store** — session/message state (thread-safe; written from both the UI loop and the bridge worker), the fork/branch tree, draft custody, and the `ConsoleChatPersistence` Protocol seam (`console_chat_store.py:822`) that the store's session/message persistence is written against. `Chat/` as a whole does import the DB layer: 42 files under `Chat/` import `tldw_chatbook.DB` (checked at `2b296e60b3`), among them `console_chat_store.py:290` (`InputError`, `TrajectoryRowWrite`), `chat_persistence_service.py:118`, `chat_conversation_service.py:17` and `Chat_Functions.py:50`. The Protocol keeps the store's *persistence calls* swappable; it is not an import boundary.
- **Bridge** — one `run_reply` execution: `AgentService` construction, tool provider composition, live step feeds, fleet coordination, marker projection.

## Failure behaviors

| Case | Behavior |
| --- | --- |
| Unsent-turn Restore/Discard cleanup fails | Safe warning; recovery stays available. Transient echo deletion precedes cancellation publication, so retry retires the echo and continuation before removing recovery. |
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
6. Approved Quit cancels the exact cache/flat browser workers before cleanup yields; the existing manager still drains their finite native reads. New read admission checks the same App's permanent shutdown flag as well as runtime disposal, because runtime detachment can precede the end of screen cleanup. Ordinary no-runtime/custom routes remain available before that terminal flag.


### Learned response rules (ADR232)

The app-owned `ConsoleRuntime.ensure_response_rules` lazily binds the shared rule repository, helper pool, queue and controller on first native generation or manager access. `set_chat_controller` installs the lazy factory even for non-screen assembly. `response_rules/generation.py` holds successful completion until required lifecycle settlement and assessment; app shutdown drains retained preparation, transport and finite SQLite callbacks before retiring their owners. `response_rules/chat_state.py` stores native source fences and body-free tool evidence through `ConsoleChatStore` facades. `console_tool_result_observation.py` composes the existing Watchlists receipt observer with definitive rule facts.

`response_rules/goal_input.py` validates a real, accepted coordinator capability and the exact runtime custody handoff/parent version; ordinary SYSTEM content is not rule authority. A settled Goal's assessment is retained, but automatic repair is explicitly refused because ADR227 has no second-run continuation authority. Native/manual and genuine external-hook feedback otherwise share `ConsolePromptQueueCoordinator.admit_machine_followup` and one atomic receipt/checkpoint contribution.

The same scoped manager is used by `/rules`, Chat settings and canonical F4 Settings. It loads before or after a Chat exists; source/Workspace/profile changes revoke current tests, and Save uses the exact tested revision and binding CAS. Source CSS is `_console_response_rules.tcss`, included by the existing CSS builder. Successful Save refreshes the selected row, details and tested definition from the exact committed revision, which remains the input to subsequent promotion or Disable/Enable.

Rule phase events refresh the visible run chip directly from current in-memory state. They keep the control bar's nonblocking configuration deferral; rebuilding the hidden mode bar would re-enter configuration during a pause. Stop uses the already-bound controller for the viewed Chat and, for drafting/testing/checking, relies on that same rule event to repaint. It does not construct a controller or refresh provider selection to cancel admitted work.

Disable, Exclude and Delete first revoke affected assessment authority under a short current-user-scope fence, even if later persistence fails. That fence is released before writer admission. The mutation then rechecks the user scope and binding CAS, and invalidates again to fence work started during the wait. Rule writes acquire SQLite writer admission before the store/currentness mutation locks, recheck source and binding CAS after admission, and retain the short fences through commit or rollback. Temporary publication follows successful settlement; transcript adoption still uses its caller-owned transaction and completion observer. Waiting for another SQLite writer therefore cannot hold the synchronous Stop/source-revocation locks.

Recovery-card synchronization runs before the transcript message-token gate: pre-send holds and continuation changes must remain visible even when message projection is unchanged. The expensive message pipeline retains its existing gate.


### Native multi-model review (ADR281)

`console_code_review.py::ConsoleCodeReviewCoordinator` binds accepted USER intent,
narrows exact request grants, owns retained-review offers and binds actual later
primary actors. `_run_agent_reply` composes its authenticated provider only from
native acceptance or an accepted `AgentWakeAuthorization.review_boundary`.
Native accepted requests include “do multi-model code review” and “Can you review
my changes?”. A leading `$multi-model-review` alias consumes only original USER
arguments after the real skill substitution owner admitted its trusted binding;
expanded skill/source text cannot grant stages. Qualified verification still needs
later explicit native acceptance.
Local source reads reuse the composed permission owner; selected destinations reuse
project-instruction dispatch confirmation. Settlement hands an off-loop, bounded
original page to a mounted `ConsoleCodeReviewModal` for the exact active session;
only successful display acknowledges presentation before an authorized verifier.
The modal is presentation-layer code, so the Chat-layer controller reaches it only
through `review_report_presenter`, which `UI/Console_Modules/review_reports.attach_app`
installs when a Console view attaches (beside `controller.app`). The presenter shows
nothing while no Console view is attached to the runtime, so a review that settles
with no view (the launch wake attaches only the app, or Console was closed) keeps its
originals pending. Only a fresh view claim (a new runtime attach generation, or a
controller that has not seen this claim) runs `ConsoleCodeReviewCoordinator.present_pending`
off the loop, one pass at a time; ordinary ensure calls on the same view schedule
nothing. The pass skips fenced (Stop/closed) sessions and raises the verification offer
only after it presents a previously pending original, so an answered offer is never
raised again. A failed pass is logged by exception type only.
The transcript carries a body-free session reference. Reusing the reader for R2
updates both selector and body and fences older reads with a generation counter.
`ConsoleFleetWakeCoordinator.on_review_boundary` preserves ordinary-spawn counts,
finite chain allowance, manual priority and stop/close fencing.

Only an exact accepted review boundary gets `_resolve_accepted_review_capacity`
before capacity/schema freeze. It rechecks the accepted attempt, retained grant,
source, session, destination and priority before and after bounded metadata work;
no source lease or publication lock spans the metadata await. The gateway projects
that immutable resolution into the existing context cache target, without reading
mutable selection settings. Unverified capacity refuses the accepted continuation
without replay or refund. The normal first-request planner must still disclose or
permit discovery of review reading/guidance and retain `spawn_subagent`; otherwise
it pauses with `review_stage_tools_unreachable`. Ordinary sends and other wake
kinds retain nonblocking metadata and the conservative bootstrap fallback.

Completed engine history is serialized by `console_prepared_request` from the
admitted saved owner/checkpoint through `project_provider_continuation`: recorded
assistant/tool rows followed by the exact saved final assistant once. Runtime's
already-expanded canonical rows normalize back to that single semantic owner
before counting and whole-unit eviction. Persisted review-tool records retain
`CodeReviewToolProvider`'s body-free projection; replay never restores ephemeral
raw arguments. Selected historical groups coexist with the live group only in
the primary; child histories do not inherit the primary sidecar.

Captured eligible MANUAL/QUEUED turns derive replay provenance from the exact
saved revision plus continuation attachment. `ConsoleTraceService` recomputes
canonical rows and checks terminal response lineage, frozen policy and bounded
old span before the existing compound replacement. Old trace checkpoint rows
compare through the frozen credential/PII disclosure projection; provider rows
remain canonical. A first new host-custodied native batch uses existing per-domain
bootstrap append with exact retained message/attachment prefixes and current
origin proof. Existing native advance/retry, final binding, CAS and rollback gates
remain in place. Automatic AGENT_WAKE still uses CaptureOff; its transcript SYSTEM
notice remains a payload USER row.

Native report pages materialize at most 8,192 input characters, sanitize off-loop
(control escapes expand by at most four),
and keep source evidence as explicit captured references. The source selector reads
each captured document independently in at most 8,192 bytes per page, preserving
UTF-8 boundaries. First report display never expands cited source ranges. Whole
accepted originals and raw streams remain retained and exportable beyond model
transport limits. Export contains full original records, dispositions and evidence
references; it does not duplicate source documents for every citation. Source pages
expire with the session bundle. Explicit export/release still uses the existing
native destination/discard UI; model flags confer no permission.

Source guards can synchronously enter Textual. Service/source readers and native
stage/boundary callbacks pin a lease, call guards outside publication locks, then
revalidate live context, binding/revision/epoch and exact attempt/receipt as
applicable under the publication lock. Manual stage publication acquires boundary
admission before publication, after source callbacks. Guard exit also runs after
publication unlock. Stop can therefore acquire publication while a reader awaits
its UI guard; the reader subsequently fails closed on revoked authority.
