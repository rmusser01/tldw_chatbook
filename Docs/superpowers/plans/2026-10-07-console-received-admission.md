# Console received admission foundation implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development with the three authorized lanes and root as the sole integration/native-run owner. Steps use checkbox syntax.

**Goal:** Reserve one session admission synchronously when an existing complete request enters runtime custody, then promote that exact reservation into its full preparation without a release/reacquire gap.

**Architecture:** Extend the existing store admission slot and runtime custody record. A distinct opaque received claim owns the slot before a complete ConsoleTurnPreparation exists; the preparation ID index remains complete-only. Existing origin coordinators, durable effects and recovery stay authoritative.

**Tech Stack:** Python >=3.12, asyncio, existing threading.RLock, Textual 8.x and SQLite.

**Spec:** Docs/superpowers/specs/2026-10-06-console-send-preparation-architecture-design.md.

ADR required: yes, existing ADR-225 applies.
ADR path: backlog/decisions/225-console-send-preparation-and-io-ownership.md.
Reason: implements its shared admission/promotion contract through existing runtime/controller/store boundaries.

## Scope and constraints

- Use the isolated console-send-preparation worktree. Primary checkout is untouched. No push or merge.
- Existing complete ConsoleTurnCustodyRequest is the only runtime input in this slice; do not add an unconfigured intent DTO or change the UI intake yet. This independently fixes duplicate runtime acceptance and establishes the permanent admission API.
- Later UI intake still requires synchronous domain draft/staging revisions, screen-free selected-input capture and queue-intake handoff before enabling early receipt. No actual 100 ms feedback or subsecond adapter claim follows from this foundation.
- Saved acceptance failure keeps the draft. Preserve existing temporary chats, complete-request attachment-prefix transfer, origin authorization, WAL/NORMAL, required consent/trace/checkpoint/effect order and awaited history.
- No new scheduler, admission map, generic phase graph, native lease across awaits or reusable permission authority. Store admission and runtime lifetime ownership remain distinct.
- Targeted tests only. Root runs every native check sequentially in a coordinated window, with original bounds and physical retirement evidence. Other lanes do source work only.

## File ownership

| Lane | Owned files |
| --- | --- |
| Shared preparation | Chat/console_received_turn.py (new opaque claim and same-store admission methods); Chat/console_chat_store.py; Tests/Chat/test_console_received_admission.py (new) |
| Controller/provider integration | Chat/console_runtime.py; Chat/console_chat_controller.py; Chat/console_prompt_queue_coordinator.py; Tests/Chat/test_console_received_custody.py (new) |
| Baseline verification | Existing-test selection and read-only compatibility review; no product edits or parallel native execution |
| Root integration | This plan, Backlog task and report; final merged review, targeted checks and commit |

Paths above are under tldw_chatbook/ except Tests/; no lane edits another lane's files. Negotiate interface changes with root first.

## Shared interface

Store methods (same _preparation_lock and existing per-session slot):

```python
claim_received_turn(session_id: str, request_id: str, *, origin: ConsoleSubmissionOrigin = ConsoleSubmissionOrigin.MANUAL) -> ConsoleReceivedTurnClaim | None
received_turn_for_session(session_id: str | None) -> ConsoleReceivedTurnClaim | None
received_turn_is_current(claim: ConsoleReceivedTurnClaim) -> bool
promote_received_turn(claim: ConsoleReceivedTurnClaim, preparation: ConsoleTurnPreparation) -> ConsoleTurnPreparation | None
seal_received_turn(claim: ConsoleReceivedTurnClaim) -> bool
release_received_turn(claim: ConsoleReceivedTurnClaim) -> bool
```

The claim is opaque, exact-identity, body-free and session/request-bound. Capture the live session witness and a request generation under the existing lock; a seal is irreversible. Any private state belongs on that slot occupant, not a parallel registry. Keep preparation_by_id/preparation_for_session preparation-only. Direct begin_preparation refuses a received occupant before any old-ID idempotency shortcut. Every CAS/remove/cancel/purge path must discriminate the occupant type and preserve successor identity. Promotion is one lock-held exact replacement and ID-index update, checks the original live session and seal, and cannot mutate another session's preparation. Releasing a promoted claim does nothing to its preparation. Do not call arbitrary callbacks/native readers under the lock.

Private task handoff preserves public submit_draft and injected callback signatures. Define a narrowly scoped task-local claim binding in console_received_turn.py (stdlib ContextVar/context manager), with exact current-task identity; inherited bindings in a child task must not authorize promotion. Runtime binds its retained claim only around the live controller submit call. Controller consults that token in its existing _begin_submit_preparation and promotes or uses ordinary begin_preparation. This is admission identity only; existing queue/wake/AgentChatStart authorization still runs unchanged. No new public callback argument is passed.

Runtime accept_turn reserves before attachment transfer, registration or task scheduling. Duplicate refusal therefore moves no inputs and adds no archive count. Extend the existing custody record with the claim reference; capture the issuing store so replacement cannot release another store. Use one private lazy stdlib Task so configurable eager-start-then-raising factories cannot lose the lifetime handle. Scheduling/transfer failure unwinds exact inputs and claim. Normal terminal cleanup releases the original claim only after its issued work has retired; promotion makes that release an exact no-op. A stale or sealed claim refuses before submit/promotion.

Activity/Stop/Close/shutdown use received_turn_for_session, not preparation-only getters, and publish received work as preparing/slot-occupying. Stop seals the exact claim; its running task follows existing cancellation/drain semantics. Close/dispose fences before cancellation snapshots and seals received claims before resources close. Existing runtime custody/archive counts include this work. Retry of a paused preparation retains its original preparation/CAS; competing direct, queue, wake and automatic calls cannot bypass a received occupant. Origin checks remain before their actual effects and forged authorizations keep their errors. Do not add an automatic retry.

## Review focus

1. Two synchronous accept_turn calls before either task runs: one owner; loser leaves attachments and archive count unchanged.
2. Received claim vs direct begin, and promotion vs seal: one lock winner; no unowned interval, no post-Stop dispatch.
3. Old-ID idempotency and late cleanup: an old claim/preparation cannot replace or release a successor; controller/store replacement cannot redirect cleanup.
4. Existing queue/retry/automatic paths: receipt is busy, exact runtime submit recognizes its own claim, and valid/forged origin behavior is unchanged.
5. Scheduling failure and cancelled native capture: retain physical work and precise input recovery; no receipt survives terminal cleanup, no live native producer loses ownership.

## Execution

- [x] Baseline lane selects the existing controls below; root runs them sequentially before changed product source.
- [x] Shared lane writes receipt-vs-preparation, exact promotion/seal/release and stale-owner race tests; integration lane writes actual duplicate accept_turn and lifecycle tests. Root observes meaningful RED on the original runtime API before product changes.
- [x] Shared lane implements the slot/API and type-aware existing store operations. Integration lane implements the exact private handoff, runtime custody and activity/lifecycle/origin consumers using only the agreed API. Preserve supported signatures.
- [x] After both lanes are ready, root runs the new tests and affected original controls together, investigates failures at their actual boundary, and checks source/current retirement evidence.
- [x] Review the integrated source, run scoped lint/format/diff checks without widening baseline ratchets, update task/report and commit only qualified scope. Keep wider responsiveness and UI-intake work explicit.

Baseline selection:

- Tests/Chat/test_console_automatic_library_preparation.py::{test_store_preparation_cas_is_exact_and_survives_controller_replacement,test_store_racing_actions_have_one_winner,test_close_cancels_preparation_through_same_store_path}
- Tests/Chat/test_console_runtime_lifetime.py::{test_custody_registers_before_the_runtime_task_starts,test_custody_task_creation_failure_releases_registration,test_archive_reservation_and_recovery_follow_actual_custody}
- Tests/Chat/test_console_turn_execution_context.py::{test_runtime_custody_transfers_exact_attachments_and_frozen_inputs,test_recovery_restores_exact_objects_before_later_attachment_suffix}
- Tests/Chat/test_console_prompt_queue_coordinator.py::{test_queued_origin_requires_coordinator_authority,test_queued_drain_enters_runtime_custody_before_controller_with_frozen_config,test_failed_retry_adopts_authorized_epoch_then_drains_next_prompt}
- Tests/Chat/test_console_fleet_wake.py::test_agent_wake_origin_is_unreachable_without_the_coordinator_token

New focused modules reuse the existing real store/preparation/runtime fixtures and bounded barriers/held native probes. No entire-module census, full suite or performance inference from synthetic waits. Runtime custody record tests may add only the opaque claim/store lifetime handle, with no duplicated authoritative preparation state.

## Self-review

This intentionally scopes the accepted architecture to the first exercised admission boundary. It adds an immediate real consumer, not an unused received-intent framework. UI capture revisions and queue intake are still required before early feedback can be enabled; their absence is not claimed as completion. Existing original source/permission gates and durable postcommit order remain downstream of admission identity. Native tests and timing never overlap.

Baseline on unchanged 661da42: 13 targeted PASS in two sequential batches. Original global task-creation failure injection emits unawaited Canvas coroutine warnings; narrowly retarget that assertion to the new private custody-task constructor while preserving its failure/cleanup behavior. Integration lane may edit those exact assertions in Tests/Chat/test_console_runtime_lifetime.py. Evidence: .superpowers/sdd/2026-10-07-console-received-admission/baseline-audit.json.

## Interface review refinements

- Private helpers are bind_received_turn_claim(store, claim) and received_turn_claim_for(store, session_id). The exact-task binding must preserve a stale/sealed claim's identity so it cannot silently fall back to ordinary admission. Promotion always checks the authoritative store occupant. A child task cannot use the inherited parent binding to promote.
- Claim origin is projection data, never authority. Every received claim occupies its slot; only manual/queued claims promise queue availability after acceptance. Wake keeps the existing chainless preparing=False distinction.
- Runtime captures its own original store rather than requiring controller doubles/custom callbacks to expose .store. Archive refusal still precedes prompt-chain/submit access.
- The private runtime _create_custody_task(coroutine) constructs a lazy stdlib Task. Retarget only the original scheduling-failure injection to this seam; preserve assertion strength and original cleanup behavior.

- Same-task binding lookup with a changed store or session raises RuntimeError rather than returning unbound and falling back to ordinary admission. A genuinely absent binding or inherited foreign-Task binding returns None; a sealed/released same-source binding retains its original claim and refuses authoritative promotion.
- Some legitimate attachment-only/Capture-Off and machine paths never build a ConsoleTurnPreparation. Keep their received claim through the initial submit's physical completion, then release it in that submit's finally before the outer run_prompt_chain drains the next queued turn. Outer custody completion remains exact fallback cleanup. Do not release at durable acceptance or provider entry. Add a runtime-first, no-full-preparation queue regression.
- Preserve queued/wake/AgentChatStart authorization errors before generic busy refusal. Place foreign-receipt refusal before the queue-authorized early return in _active_run_rejection. Capacity checks exclude only their exact valid bound claim, never another request.
- Old-ID capture publication methods must verify the exact current slot before writing it. Direct session Close must discriminate claim from full preparation.
- Integration lane also owns only the lifetime-field assertion in Tests/UI/test_console_runtime_ownership.py; add opaque original-store/claim handles, no duplicate stage/status authority.

Independent source-only plan review is complete with these refinements. The approved architecture and execution authorization continue; no new user decision is required for this implementation slice.

Original 661da42 meaningful RED: second real synchronous accept_turn did not raise and admitted the losing request before either task stepped. Shared contract RED confirms missing API. Both runs retired contained Jobs/profile normally; the runtime RED fixture also left a pending Canvas watcher, now explicitly assigned to integration to fix via actual runtime disposal before any final cleanup claim. Product implementation released only after these results.

Integration review adds one read-only predicate, received_turn_is_current(claim), sharing promotion's exact slot/session-witness/seal validation under the existing lock. Nonpromoting paths and post-await archive checks can validate the original admission without reading private claim fields. No I/O or callback is performed by this check. Coordinator activity receives an optional named read-only received_turn_for_session accessor; no duplicated projection state.


### Native-lifetime review qualification

Source review found that the existing pre-promotion hook-admission read, reference expansion and Library-policy repository calls await bare `asyncio.to_thread` calls. Cancelling the submit Task does not prove those original native calls retired. The received slot is admission identity, not a native retirement receipt: nonpromoting Capture-Off/machine routes can retain it through arbitrary provider/custom awaitables. Therefore this slice must not add an unbounded drain of whole received-submit Tasks or claim that it solves these pre-existing native lifetime gaps. Preserve bounded provider shutdown and the exact existing ordinary-save/hook-review retirement drains. The received claim is sealed before cancellation and stale promotion is refused. AC #3's native-retirement qualification remains open until those specific finite producers have physical completion ownership; early UI capture must not move additional I/O behind admission without that ownership. This is an explicit limitation, not a passing native-cleanup claim.


### Final source-review refinements

- Nonpromoting attachment-only/Capture-Off routes revalidate the exact bound receipt after their last pre-acceptance await and before persistence or input consumption. A sealed or source-displaced receipt cannot reach acceptance merely because no full preparation was constructed. Exercise the real gateway-resolution boundary with a held result and a changed receipt/source.
- Existing recovery entries may retain an optional opaque source claim as provenance. Add `received_turn_matches_session(claim)` to the same store mixin: original store and live session witness/incarnation/binding/ephemeral must match, while seal/slot occupancy is irrelevant after retirement. Reuse its locked source check for current admission. Filter/refuse stale recovery listing/restoration, including replacement after the recovery was already recorded. This carries no new state map and does not authorize admission; legacy recovery entries retain existing behavior.

Root also owns the narrow integration test `Tests/Chat/test_console_received_saved_turn.py`: enter the existing real SQLite checkpoint-failure fixture through actual runtime custody and verify failed save retains the original draft/recovery, dispatches nothing and leaves no committed rows. This qualifies the user-selected saved-turn failure policy through the new admission boundary.

Final integrated behavior: 69 targeted PASS after4 invalid new-fixture corrections; no product changes during execution. Exact-source recovery, nonpromoting acceptance and late callback controls pass. Static introduces no diagnostics; existing module-size/lint/format and one legacy Canvas fixture warning remain. See Docs/Development/2026-10-07-console-received-admission-verification.md. Native-read retirement and actual responsiveness remain unqualified, so task status stays In Progress.
