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

Console skill discovery (`UI/Console_Modules/skill.py`) prepares cold trust with its existing App worker, then recaptures the current stock source before reading context. The original decorated `SkillsScopeService.get_context(mode="local")` runs through `run_preparation_read`, including local-content admission, policy enforcement and catalog/trust checks. Runtime preparation-read tracking retains the physical call through cancellation and shutdown; source and view attachment checks gate publication afterward. Custom context, catalog, policy, disabled-builtins and trust-status callbacks retain the direct path. This changes scheduling only: no catalog cache, permission verdict reuse, or display proof is used as execution authority (ADR-126/230/231; TASK-35590).

Cold context and cost estimates share tokenizer preparation with `ConsoleReadinessConfigProjection` (TASK-35591). The selected stock model/provider's original custom-tokenizer manager admission and encoding load run as finite, runtime-observed preparation reads, outside the config operation. Display estimates remain unavailable until their own request completes; model, manager, mapping, source and view changes invalidate readiness. Recorded provider usage still supplies Current spend while preparation is unavailable, provided every contributing history row has usage; mixed estimated history stays unavailable rather than reporting a partial total. Fingerprint, cache-break and Next-send estimates wait for readiness, and readiness participates in the settled display-cache key. The original token arithmetic and live action counters are unchanged. Custom callbacks and heuristic custom-tokenizer directory selection retain their direct path; this optimization covers cold stock backends and exact custom mappings, not long-text counting CPU or arbitrary extension callbacks. Backend loading errors remain unavailable and retry on the existing bounded display cadence.

Console attention event callbacks use `ConsoleRuntime.request_console_attention_refresh` (TASK-35592). Stock file-backed unseen-mark queries are coalesced through finite runtime preparation reads with operation-owned connections. Existing read revisions invalidate results after newer refreshes or exact receipt acknowledgements; source, store, runtime and disposal checks gate publication. Visibility, receipt-outcome classification, notifications and projection still run on the caller loop. Explicit `recompute_console_attention` and custom readers retain their synchronous contract; no durable mark or permission cache is added.

Ordinary resume prepares the original active-workspace registry read after successful FULL sync and before restoring runtime answerability (TASK-35593). `prepare_console_workspace_reconciliation` supplies a single-use result to the original synchronous reconciler in that resume stack. Native admission, current registry generation, source/session/view checks and physical retirement remain intact; workspace repair and subsequent resume actions retain their order. Explicit saved-chat resume and custom callbacks keep their original routes. The prepared result is discarded at the end of the stack and is not a reusable display or authority cache.

FULL synchronization awaits the existing stock `ConsoleReadinessConfigProjection.warm()` before synchronous rail/control presentation (TASK-35594). This avoids entering native config admission on the UI thread when the display projection expires. The await is outside the workspace build scope and config operation; source/session ownership is checked by the existing projection, and the screen rechecks its attach visit and maintenance state before rendering. Refusal retains whole-FULL replay without declaring attach complete. Direct/custom presentation keeps its original fallback, live actions keep their checks, and cancellation retains the finite native read until physical retirement. No display lifetime is extended and general narrow polling remains Proposed (ADR-231).

Character presentation uses `UI/Console_Modules/character_context.py`. Stock active displays carry the original current-owner closure through the one-second run scope check into the same finite inner batch as idle display; direct/custom actions retain fresh no-argument dispatch. A stock display whose previous exact scope fingerprint names a different open conversation can skip the preliminary change-detection read: the existing stock reader selector must qualify first, and the unchanged refresh still validates fresh pair/groups/pair metadata and current ownership before publication. Missing/invalidated fingerprints, unchanged conversations, forced/direct and custom routes keep their original change detection. The existing two-second display memo, invalidation, source checks and physical callback retirement remain separate contracts (ADR126/230).

Finite preparation callbacks temporarily use at most1ms interpreter thread handoffs to reduce scheduling convoys across fresh native reads (TASK-34406.11, ADR230). The existing bookkeeping lock counts overlapping physical callbacks; last exit restores the prior value if still owned. Lower prior or later different external settings remain. This affects interpreter scheduling only: original source, permission, native ownership and cancellation checks are unchanged.

Console presentation avoids an unused leading inspector/rail build before the
control pass (TASK-35610). Controls supply their freshly built inspector to the
existing rail projection; hidden inspector updates and the fresh post-await FULL
projection remain. Conversation appearance display caching records the last
successfully queried IDs, including absent rows, under its existing short TTL
and service identity. New IDs, expiry and writes require another original read;
failed reads do not renew the cache. Deleted appearances are removed on successful
refresh. These are disposable icon/color values, never storage or action authority.

An inspector build shares provider/config derivation through the existing synchronous
memo, discarded when that build ends (TASK-35612). A subsequent build, including
FULL's post-await projection, starts fresh. Checked-display coordinator contention
requests the existing repaint retry without entering its body or a nested metadata
fallback. A qualified expired display also defers to the existing finite checked
refresh, even when that direct call started the refresh (TASK-35646). The original
coalesced retry renders only after a fresh check; refused worker submission stays
retryable. Unknown/custom and unscoped callers retain their native route. If such
an entry later refuses display, retry waits for physical retirement and cleanup
errors still propagate. No source, pause, ownership, lifetime or action fence is
relaxed.

Independent provider-label and readiness builds also open that existing synchronous
scope (TASK-35613). Each loads current config once, reuses an enclosing scope when
present, and drops the memo on return or failure. The next independent call reads
again; config path posture checks and native action admission remain unchanged.

The transcript memory banner shares the controls' existing disposable context
inputs for qualified stock readers (TASK-35679). Cold preparation hides only
the banner while message painting continues. Effective-memory publication joins
the transcript tick gate, so a finite read can repaint an otherwise unchanged
transcript. The memo key retains reader binding/code/default identities, refusing
custom results after a stock reader is restored. Custom callbacks keep their
original synchronous post-gate path and call cadence. Branch-memory validation,
finite native retirement, freshness deadlines and live action reads are unchanged.

Attach/resume reconciliation launches from a synchronous after-refresh callback
into the finite `console-attach-reconciliation` view worker (TASK-35614).
Textual awaits async after-refresh callbacks inside the screen message pump;
holding that pump through native preparation prevented even draft typing.
The launcher deduplicates without cancelling an older visit's physical read.
Existing visit fences, bounded retries, runtime answerability and original
native-read retirement remain in the reconciler. The group participates in the
same exact-worker capture and view/App shutdown drain as other finite workers.
Typing remains session-owned while the reconciliation awaits; Send retains its
existing Preparing, fresh configuration and admission checks.
The resume-triggered hook indicator refresh is scheduled after reconciliation,
avoiding a duplicate read/repaint during the FULL await. Session-change and Send
refreshes remain independent. A failed attach can retain the previous indicator
count until reconciliation or explicit Hooks/Send refresh; permission checks still
read fresh state on the action path.

Explicit return acknowledgement (TASK-35615) captures its original current
unread token before deciding whether it needs an acknowledgement-only FULL pass.
An absent visit, refused read or absent token needs no acknowledgement; ordinary
attachment reconciliation still owns FULL. A real unread token retains the
existing FULL, after-paint scheduling, current-owner checks and compare-and-clear.
This avoids queuing a second FULL behind attachment for unsaved/already-read chats.

CSS application retains upstream Textual style resolution. A bounded 1024-entry
candidate LRU uses live subject/ancestor names and the parsed rules-map identity;
it retains no widgets or resolved styles. Dynamic pseudo-classes, sibling order,
theme changes and virtual component styles still pass through upstream matching.
The static filter (TASK-35611) requires all known type/class/id names in each
selector alternative, separating its final subject compound from ancestor names.
Types follow Textual's CSS inheritance, including its opt-out. A rule is rejected
only if every comma alternative is impossible. Presence alone never establishes
a match: upstream still checks compound co-location and child/descendant order.
At the stylesheet matcher seam, a false final pseudo-state is rejected using
Textual's live `Selector.check(subject)` before its ancestor scan (TASK-35613).
Candidates remain present for refresh-flag computation; successful prechecks still
delegate the full match. No pseudo-state result is cached, and DOM query matching
is unchanged. Upstream-comparison tests restore both application and matcher seams.
Composer Send/Stop class changes use Textual's native `update_classes` batch,
avoiding intermediate style refreshes while preserving unrelated classes and the
final disabled, tooltip, queue and setup states (TASK-35614).

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
6. Approved Quit cancels the exact cache/flat browser workers before cleanup yields; the existing manager still drains their finite native reads. New read admission checks the same App's permanent shutdown flag as well as runtime disposal, because runtime detachment can precede the end of screen cleanup. Ordinary no-runtime/custom routes remain available before that terminal flag.


### Learned response rules (ADR232)

The app-owned `ConsoleRuntime.ensure_response_rules` lazily binds the shared rule repository, helper pool, queue and controller on first native generation or manager access. `set_chat_controller` installs the lazy factory even for non-screen assembly. `response_rules/generation.py` holds successful completion until required lifecycle settlement and assessment; app shutdown drains retained preparation, transport and finite SQLite callbacks before retiring their owners. `response_rules/chat_state.py` stores native source fences and body-free tool evidence through `ConsoleChatStore` facades. `console_tool_result_observation.py` composes the existing Watchlists receipt observer with definitive rule facts.

`response_rules/goal_input.py` validates a real, accepted coordinator capability and the exact runtime custody handoff/parent version; ordinary SYSTEM content is not rule authority. A settled Goal's assessment is retained, but automatic repair is explicitly refused because ADR227 has no second-run continuation authority. Native/manual and genuine external-hook feedback otherwise share `ConsolePromptQueueCoordinator.admit_machine_followup` and one atomic receipt/checkpoint contribution.

The same scoped manager is used by `/rules`, Chat settings and canonical F4 Settings. It loads before or after a Chat exists; source/Workspace/profile changes revoke current tests, and Save uses the exact tested revision and binding CAS. Source CSS is `_console_response_rules.tcss`, included by the existing CSS builder. Successful Save refreshes the selected row, details and tested definition from the exact committed revision, which remains the input to subsequent promotion or Disable/Enable.

Rule phase events refresh the visible run chip directly from current in-memory state. They keep the control bar's nonblocking configuration deferral; rebuilding the hidden mode bar would re-enter configuration during a pause. Stop uses the already-bound controller for the viewed Chat and, for drafting/testing/checking, relies on that same rule event to repaint. It does not construct a controller or refresh provider selection to cancel admitted work.

Disable, Exclude and Delete first revoke affected assessment authority under a short current-user-scope fence, even if later persistence fails. That fence is released before writer admission. The mutation then rechecks the user scope and binding CAS, and invalidates again to fence work started during the wait. Rule writes acquire SQLite writer admission before the store/currentness mutation locks, recheck source and binding CAS after admission, and retain the short fences through commit or rollback. Temporary publication follows successful settlement; transcript adoption still uses its caller-owned transaction and completion observer. Waiting for another SQLite writer therefore cannot hold the synchronous Stop/source-revocation locks.

Recovery-card synchronization runs before the transcript message-token gate: pre-send holds and continuation changes must remain visible even when message projection is unchanged. The expensive message pipeline retains its existing gate.
