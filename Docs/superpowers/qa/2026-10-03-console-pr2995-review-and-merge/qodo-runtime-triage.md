# Qodo runtime triage: findings 1, 5, and 8

Inspected HEAD `7a15e3e9b4998dba8d8726bbdb722ff60a3df83d`. This is a read-only source/control review, not a test receipt. No application source, index, or Git state was changed; no tests were run. Line references below describe that immutable HEAD.

## Decisions

- **1: Valid performance defect; the suggested bare worker conversion is unsafe.** The current synchronous critical section enforces a stated ownership contract. There is no existing handshake that makes the final loop-owned checks atomic with an off-thread durable acceptance. A complete off-thread fix requires an explicit admission/withdrawal protocol, not just shielding a task.
- **5: Valid settlement defect; reject the suggested accepted-generation refund.** Settle accepted/unreceipted attempts to review-required in-process, retain their committed charge, and pause the canonical shared root. Automatic work from that allowance must remain blocked. Explicit manual work may establish a new allowance.
- **8: Valid observability defect.** Give unexpected preaccept failures a distinct reason and sanitized type-only logging. Do not use `logger.exception`: it includes exception text and potentially diagnosed locals.

ADR required: no for the bounded settlement/diagnostic repairs; existing ADR-211 applies. ADR path: `backlog/decisions/211-console-chat-destinations-and-bounded-starts.md`. Reason: these repairs restore its existing accounting and truthful-outcome rules. A redesigned worker admission/withdrawal boundary for finding 1 requires an explicit architecture decision before implementation; do not implicitly move the cutoff to worker enqueue or an in-memory flag.

## Binding contract

ADR-211 lines 43–50 require all descendants to share the original allowance and require descendant uncertainty to block siblings. Lines 64–74 say Manual Send withdraws still-prepared work, accepted targets become independent of their source, the **AgentRunsDB acceptance transition is the source-ownership cutoff**, only proven preaccept refusal refunds an uncommitted reservation, and accepted/uncertain work stays charged. Restart recovery must not resend automatically.

## 1. Synchronous acceptance blocks the event loop

### Evidence

`ConsoleChatStartCoordinator.accept` in `tldw_chatbook/Chat/console_chat_start.py:445–480` drains the draft writer, rechecks runtime and destination, then checks exact live authorization, source, target, and automatic-primary claim. Its comment explicitly forbids an await between those checks and ledger acceptance.

`AutomaticWorkLedger.accept_chat_start` in `tldw_chatbook/DB/automatic_work.py:824–855` opens a connection, enters an admission transaction, checks durable runtime ownership and budget, commits the reservation, marks the attempt accepted, and starts the original automatic clock. It does **not** recheck source stop state, target draft/settings/context epoch, store/bridge identity, or source incarnation. `transaction` at lines 61–82 uses BEGIN IMMEDIATE and synchronous=FULL. `AgentRunsDB._get_connection` at lines 358–384 sets busy_timeout=5000. This can stall Console for lock waits and durable I/O.

The relevant invalidators are real synchronous owners:

- `withdraw_for_manual` and `withdraw_prepared` in `console_chat_start.py:83–91,498–514`; withdrawal sets `withdrawn`, then cancels the owned task while `accepted` is false.
- `submit_draft` in `console_chat_controller.py:10235–10244`; Manual Send drains withdrawal before ordinary admission.
- `_signal_stop` at controller lines 15292–15309; it withdraws before setting the source/target cancel event.
- `_chat_creation_source_live` at controller lines 19118–19175; the source depends on session lifetime, revoked run IDs, active assistant identity, bridge primary ownership, and cancel events as well as persisted run state.
- `set_session_draft` in `console_chat_store.py:9754–9777`; it mutates the draft and revision synchronously, then invokes the withdrawal callback.
- `dispose` at coordinator lines 516–524 and controller begin-shutdown at lines 21060–21065; they invalidate admission and cancel owned work.
- `_target_unchanged` at coordinator lines 129–147 includes settings, context epoch, pending attachments, one-shot prefill, staged evidence, and queue ownership. The ledger does not own these values.

### Why simple offloading is not safe

With `await run_owned_db_call(...accept_chat_start...)`, a source Stop can run while SQLite is blocked before BEGIN IMMEDIATE. It marks the item withdrawn and cancels its coroutine. The physical worker then resumes and accepts the still-prepared row because no durable source revocation participates in that transaction. Setting `item.accepted` before enqueue instead moves the cutoff earlier than ADR-211. Checking again after await prevents some dispatch but does not undo the unauthorized acceptance/charge. Reading mutable store objects from the worker also does not make checking and committing atomic.

A thread Event checked just before the writes has a check-to-commit race. An asyncio lock around `accept` alone does not cover the synchronous draft/Stop/lifetime mutators. A threading lock held through SQLite commit blocks those mutators on the event loop, recreating the performance defect. Queueing a DB abort on withdrawal gives the earlier accepted DB operation priority even if source Stop happened while acceptance was waiting, unless the mutation semantics are deliberately changed. These are not equivalent to the current final-check/no-await section.

### Safe design requirements for a full worker implementation

A real worker protocol must give **all** source/target/lifetime invalidators and the ledger acceptance one ordered authority boundary, while keeping mutable UI/store access on its owner loop. Either persist revocable authority and serialize revocation with acceptance, including the corresponding source/target mutation publication, or deliberately change the cutoff to an explicit in-memory admission claim. The latter contradicts the existing ADR and is not a routine refactor. A durable protocol likewise needs a specified ordering policy for Stop/edit versus a pending SQLite transaction, plus participation from store/context/runtime invalidators; no such protocol exists in the inspected path.

Regardless of that architecture choice, the worker must be created and retained on the authorization before its first await, shielded, and drained despite repeated cancellation. Manual withdrawal must await its physical result before admission/slot release. Record actual accepted/rolled-back/unknown status after drain; cancellation of an await is never evidence of rollback. Keep the exact captured database, ledger, runtime owner, store, bridge, target incarnation, attempt identity, and capacity token throughout. A known acceptance must remain charged even if its coroutine was cancelled; an uncertain result must enter review, and source Stop after the actual cutoff must not retroactively withdraw the accepted target. Destination/lifetime/target invalidation must still prevent stale conversation publication or provider dispatch.

### Bounded alternative under the existing contract

Keep the current synchronous final cutoff while landing findings 5 and 8. Do not represent this as resolving finding 1. A narrowly scoped, fail-fast SQLite writer acquisition for this one final admission can reduce the 5-second lock wait without moving the cutoff; it must preserve FULL durability, retire its owned connection, restore connection settings, and report/refund only a proven preaccept refusal. This is only a contention mitigation: connection initialization, reads, and fsync may still block, and the getter currently resets busy_timeout. A global timeout change or a timeout placed before an actual connection opener that resets it is not a valid implementation. Full nonblocking acceptance remains an explicit architecture task.

### Concrete connection-scoped fail-fast alternative: supported

Further inspection verifies that the current native API supports the parent review's narrow proposal. `AgentRunsDB.connection` (lines 437–447) yields `_held_connection`; `_held_connection` (403–435) reuses the registered thread-local handle. `_get_connection` (358–399) applies busy_timeout=5000 only when opening/reviving a handle, not on each nested connection scope. Therefore this sequence is sound for the existing native owner:

1. Retain `operation_owned_connection(item.database)` as the outer lifetime scope.
2. Enter `item.database.connection() as conn` synchronously after the final authority checks; capture `PRAGMA busy_timeout`, set `PRAGMA busy_timeout=0`.
3. Invoke the existing synchronous `ledger.accept_chat_start` without any await. Its initial attempt read and its `_admission_transaction`/`transaction` connection scopes reuse this exact held connection. `connection()` does not itself BEGIN a SQL transaction, so this outer handle scope does not cause the ledger's nested-transaction refusal.
4. Restore the captured bounded integer timeout in finally, while still inside the handle/lifetime scope. Retire only a newly opened owned connection; preserve a borrowed caller handle and its original timeout. Keep synchronous=FULL as implemented by the ledger.

This changes writer contention into an immediate refusal instead of a 5-second wait and leaves source/target/manual/lifetime serialization unchanged. The existing async abort then releases only a still-prepared reserved generation. Use a bounded static contention reason if desired, based on the SQLite error code rather than exception text, but only report not_started once ledger settlement proves preacceptance. A lock-like error during commit/cleanup is not by itself proof that acceptance did not commit; abort returning false must produce review reconciliation, never a refund.

Limitations: first handle opening still sets up WAL before the override, and FULL durability includes fsync. This does not establish a hard real-time I/O bound or fully off-thread admission. A focused real file-backed writer-lock test should warm the normal handle/setup path, hold BEGIN IMMEDIATE on another connection, invoke the actual coordinator acceptance, and prove a loop ticker continues without the 5-second stall. Separately measure ordinary uncontented cutoff latency. Preserve source Stop before cutoff, accepted source independence, exact draft revision, borrowed-timeout restoration, and operation-owned handle retirement controls. Contention keeps the saved draft/reason; it must not enqueue an automatic retry.

## 5. Accepted/unreceipted attempts never settle in the current process

### Evidence and correction to Qodo

Controller lines 12761–12768 accept in AgentRunsDB before the separate conversation commit at 12773–12849. The controller returns on a failed commit, so receipt confirmation is absent. Coordinator `_run` at lines 364–408 reports review-required but only completes `receipted` work or aborts `not accepted` work. Accepted/unreceipted rows remain accepted. `_target_active` at ledger lines 642–653 counts them indefinitely. `recover` at lines 1082–1126 only rewrites foreign-owner unfinished attempts, so current-owner recovery is not a solution.

The missing transition is real. The retained generation charge is correct. Qodo's suggestion to refund `ContinuationAdmissionRefused` is incompatible with ADR-211: that exception proves the **second database's** transaction rolled back, after the first database already crossed the ownership cutoff. Its proposed test that immediately admits another automatic start under the same allowance is also wrong for uncertain work; siblings must be blocked by shared review state.

### Concrete bounded repair

1. Add an owner-fenced, idempotent ledger transition such as `mark_chat_start_review_required(attempt_id, owner_id)` for accepted/unreceipted work. Inside one existing FULL transaction, validate the exact attempt owner and current runtime owner; transition accepted to review_required; resolve `_allowance_chain` and set the canonical root status to review_required with pause_reason=interrupted_work. Preserve the committed generation reservation, original counters, and original deadline. Existing schema already allows review_required, so this does not require a migration. Refuse to rewrite aborted/completed/replacement-owner rows. An already-review-required same-owner attempt should be a successful idempotent terminal result.
2. Call it from coordinator finally for accepted/unreceipted work after physical DB/provider work drains. Treat false/exception settlement as settlement_unconfirmed; do not publish a false success. Use the owned database worker and existing shield/drain pattern. A rejected stale-owner callback must not pause a healthy replacement root.
3. Preserve `abort_chat_start` for proven prepared/uncommitted work only. The ledger's transaction proves that classification. If acceptance can raise after a committed write (for example during transaction cleanup), do not trust the Python boolean alone: inspect/reconcile the exact persisted attempt after the worker drains, and route accepted to review instead of claiming a refund. If no durable result can be established, report review-required and retain charge/authority restrictions. Do not call global recover from this per-attempt path.
4. Do not automatically replay the opening text. Removing the accepted row from `_target_active` clears the stale execution slot; the canonical root's review state continues blocking automatic admissions. Explicit manual recovery uses its existing independent allowance path.

### Physical conversation-commit ownership is part of this repair

`_run_durable_db_call` at controller lines 12078–12100 directly awaits `asyncio.to_thread`. Its own documentation says cancellation leaves the transaction running. Coordinator currently retains only `provider_worker` (lines 160–168 and 374–378), not the second database's commit worker. Therefore a target Stop or shutdown during conversation commit can enter finally and release the automatic-primary claim while that transaction is still physically writing.

For AGENT_CHAT_START only, retain a separate exact conversation-commit task on its authorization, shield it, and drain it on cancellation before settlement, outcome resolution, or capacity release. Keep the captured persistence database/store stable rather than looking up a replacement store inside the worker. Cancellation must stop onward provider dispatch even if the drained commit succeeded. A late receipt may establish that a conversation was saved; it must not resume execution or replenish the allowance. With no confirmed live receipt, conservative review settlement is correct. Keep provider draining separate so one task reference cannot overwrite another physical owner. This bounded wrapper avoids changing unrelated manual-send cancellation semantics.

If review settlement itself cannot reach the DB, metadata alone does not enforce shared-root denial: it is a remaining uncertainty, not a completed durable repair. Preserve an appropriate runtime fail-closed block until persisted reconciliation succeeds or owner recovery takes over. Do not claim the root is paused merely because the launch status says review_required.

## 8. Unexpected errors are misreported and lost

At coordinator lines 347 and 368–372, an unexpected exception after successful preparation but before acceptance retains not_started/preflight_refused. This differs from an ordinary expected preflight refusal and loses the exception entirely.

Repair the exception classification explicitly: expected `AutomaticWorkRefused` retains its bounded known refusal; `asyncio.CancelledError` preserves withdrawal/interruption semantics without noisy error logging; unexpected `Exception` logs only its class and a fixed bounded stage, and sets not_started/start_failed when a prepared preaccept state is then proven and refunded. Missing/uncertain preparation, acceptance, receipt, or settlement stays review_required. Fatal BaseException subclasses must still receive ownership cleanup rather than being mislabeled as normal refusal. No exception string, prompt, settings, request repr, API keys, or traceback locals should enter the log.

Use the nearby controller's pattern at lines 12826–12831: `logger.warning("... (exception_type={})", type(exc).__name__)`. Qodo's `logger.exception` recommendation is not type-only: standard traceback formatting includes the original exception message, and Loguru diagnose can expose locals. If stack locations are required, format sanitized filename/function/line locations separately without exception text or locals; a traceback is not necessary for the bounded repair.

## Existing focused controls and required deltas

Inspected, not executed:

- `Tests/Chat/test_console_chat_start.py::test_before_cutoff_withdrawal_preserves_latest_draft_and_refunds` covers source Stop/edit/clear during preflight.
- `test_ledger_cutoff_retains_charge_and_requires_conversation_receipt` covers post-ledger source Stop, target Stop, and conversation write failure. Extend it to assert terminal attempt review state, original generation charge, shared root review, sibling refusal, and no replay.
- `test_initial_preparation_keeps_owner_until_uncertain_refund` and `test_unconfirmed_preaccept_settlement_requires_review` cover preparation-worker drain and refund uncertainty, not acceptance-worker linearization.
- `test_stop_keeps_native_claim_until_actual_bridge_worker_exits` and the generic-adapter counterpart cover provider workers. Add a corresponding held conversation-commit worker case, repeated cancellation, and exact claim retention until physical exit.
- `test_manual_send_withdraws_prepared_start_before_busy_gate`, `test_prepared_native_start_keeps_exact_runtime_and_destination`, and `test_runtime_update_during_readiness_rechecks_native_acceptance` protect existing preaccept ownership; none proves an await inside the final acceptance critical section is safe.
- `Tests/DB/test_automatic_chat_starts.py::test_failed_native_acceptance_keeps_prepared_charge_and_unstarted_clock`, `test_acceptance_and_abort_race_has_one_winner`, and `test_native_abort_never_refunds_a_separately_committed_reservation` protect the reservation distinction.
- `test_recovery_marks_foreign_starts_and_fences_old_owner`, `test_descendant_unknown_usage_blocks_siblings`, and `test_stale_owner_cannot_pause_a_completed_native_replacement` provide the ledger patterns for the new review transition. Add same-owner idempotency, canonical-root pause, unchanged committed charge/deadline, stale-owner denial, and no completed-row rewrite.
- Add a postprepare/preaccept injected unexpected exception with secret-like exception text; assert start_failed after proven refund, type-only log, no provider call, preserved draft, and unchanged normal cancellation classification.

A future worker-acceptance implementation needs controlled interleavings while SQLite is held: source Stop/close, manual withdrawal, draft edit/clear, target incarnation/store/runtime replacement, and shutdown; then the reverse order proving source Stop after ledger acceptance does not retract the target. A responsiveness assertion alone would miss the authority regression.
