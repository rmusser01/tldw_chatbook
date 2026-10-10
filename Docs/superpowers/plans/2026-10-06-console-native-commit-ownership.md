# Retained Console commit outcomes implementation plan

Status: implemented and reviewed; scoped verification complete with existing host/performance limits recorded below.
Planning task: [TASK-34563.5](../../../backlog/tasks/task-34563.5%20-%20Plan-retained-Console-commit-outcomes.md).
Implementation task: [TASK-34563.8](../../../backlog/tasks/task-34563.8%20-%20Retain-ordinary-Console-save-ownership-through-cancellation.md).

## Goal and boundary

Retain each issued ordinary Console save and its actual outcome through Stop, Close and shutdown. This is the accepted native-ownership prerequisite for immediate Preparing feedback; it does not enable early receipt or claim to solve the measured 9–10 second Send delay alone. Keep the existing failed-save policy: refuse dispatch and preserve the draft. Keep temporary chats, WAL/NORMAL, checkpoints, consent, trace facts and awaited history unchanged.

ADR required: no new ADR; existing ADR-225 applies.
ADR path: backlog/decisions/225-console-send-preparation-and-io-ownership.md.
Reason: direct implementation of the accepted native-outcome lifetime boundary using existing runtime, preparation, recovery and repository owners. No schema, dependency, new admission ledger, general job manager or permission cache.
Spec: [accepted Send preparation architecture](../specs/2026-10-06-console-send-preparation-architecture-design.md).

The finite tool/run-log/activation slices are committed through `8fff47b36c`. Their actual adapter timings remain above one second. The original save itself is a small part of that delay; this change makes later early receipt safe, rather than moving durability after dispatch.

## Exact shared contracts

`Chat/console_native_commit.py` exposes a frozen/slotted `ConsoleNativeCommitCompletion` with `commit: ConsoleDurableTurnCommit | None`, `error: BaseException | None`, and `caller_cancelled: bool`, plus:

```python
async def commit_durable_turn_owned(
    store: ConsoleChatStore,
    acceptance: ConsoleDurableTurnAcceptance,
) -> ConsoleNativeCommitCompletion:
    ...
```

Capture the exact store, database and synchronous bound callback before issuance. Preserve the current memory-backed inline path and custom callback invocation/error behavior. For offloaded work, copy the context and use the existing loop executor with a **private Future**, not a separately created `to_thread` Task. Shield/drain that exact Future through repeated caller cancellation; do not replace, cancel or expose it. A global all-Task cancellation must not turn wrapper cancellation into false native completion. Preserve the existing `operation_owned_connection` and exact CharactersRAGDB `_core_operation` boundary. Return only after the original callback and its owned cleanup have completed or reported their actual error. Submission failure produces no abandoned coroutine. A small submission seal must also deny native entry if the executor queues the wrapper but thread-start failure raises before returning its Future; the late wrapper must not touch the database. No executor replacement or new pool.

The result is a finite outcome, not rollback proof or admission authority. If the callback raises after durable success, consult the captured original store's commit/fingerprint/recovery state. Absence of a returned result does not prove rollback. Unknown or failed reconciliation retains the existing accepted/unresolved recovery; no automatic second user turn or provider replay.

The store adds one narrow entry point:

```python
def settle_accepted_durable_turn(
    preparation_id: str,
    *,
    fingerprint: ConsoleDurableAcceptanceFingerprint,
    terminal_state: str,  # only stopped or failed
    content: str,
    metadata_json: str | None = None,
) -> bool:
    ...
```

Under the existing generation-owner and preparation locking rules, validate the exact retained commit/fingerprint, session and message identities, assistant baseline, current checkpoint state/revision **ACCEPTED**, and live runtime ownership. Claim only the matching recovery record atomically and reuse `_settle_dispatch_recovery` plus the existing repository assistant/checkpoint transaction. Preserve generation fencing and restore/retain recovery on refusal or failed settlement. Do not manufacture DISPATCH_STARTED, relax other settlement claims or change schema. Tests must independently observe ACCEPTED before settlement.

## Controller lifetime and publication

Register one private ordinary-commit lifetime owner per preparation synchronously **before executor issuance or any await**. Bind it to the original store/database, acceptance, session and outer submit task. This is custody of one issued operation, not another authoritative admission state. The queued-before-worker-start interval is included; `_durable_commit_in_flight` alone cannot cover it because that reservation is installed inside the worker. The store indexes the same opaque owner token through narrow private retain/release hooks under its existing preparation lock, so direct cleanup can refuse while that owner is live. It does not copy lifecycle state, tasks or outcomes. Release requires exact token identity. Source review showed why this index is necessary: `discard_uncommitted_durable_preparation` leaves the preparation live, and a persisted-session commit can restage its removed durable IDs. Guard direct discard/retire and session close before destructive mutation; preserve ordinary retry behavior when no native owner is retained.

Central abandonment/Stop and `begin_session_close` must preserve the exact owned preparation, commit reservation and postcommit continuation. In particular, Close currently abandons READY/PREPARING entries before its grace period: guard that point as well as final deletion. Keep the existing fences and stop signals effective immediately. Do not allow another Send to reuse the slot while the outcome is unknown.

On normal success, preserve the existing required postcommit effects and their order. Release the ordinary native owner after required preparation publication and before provider execution; it must not keep an unrelated long provider response inside the native-save lifetime. On cancelled late success, complete required owner/projection/queue/history bookkeeping, preserve composer text on the cancelled path, then settle stopped for explicit Stop/Close/shutdown or failed for reasonless cancellation. Both return accepted=True and provider_started=False. The early-cancel path must stop before checkpoint dispatch transition, trace launch or provider entry. Settle immediately after required publication and before the existing `continue_to_provider=False` branch would mark runtime recovery inactive. Do not create a provider-generation token for this cancellation.

Preserve exact attachment IDs and prefill revisions. Skip cancelled-path composer clearing and callbacks that would clear it; identical retyping must survive, as must different newer text. Full composer revision plumbing is part of the later early-receipt work. Preserve custom public signatures and AgentChatStart behavior.

Capture and validate the exact persistence/database binding as well as the store: its bound stock commit method currently performs a late persistence lookup. Preserve custom callback argument shape. Use the captured original store for reconciliation. If controller/store/database ownership changes while the worker runs, finish against the original owners, retain/reconcile that original acceptance, and refuse publication, clearing or dispatch into the replacement. Never temporarily swap a controller's store. Release the lifetime owner only once its actual outcome and required settlement or recovery ownership have been recorded.

## Close and dispose

Keep the existing Close/dispose awaiter and original general grace periods (2 seconds / 3 seconds). After the bounded drain of all surrounding work, shield/drain **only the exact issued ordinary native owners and their settlement** before releasing their custody, finalizing session deletion, ending the app runtime or closing their storage. The event loop remains responsive while physical completion is pending. Cancellation of the Close/dispose waiter, including repeated cancellation, does not release those owners. Final app exit keeps the owning event loop alive through this exact drain.

This deliberately distinguishes a responsive UI from claiming completed physical shutdown. Preserve existing bounded behavior for unrelated fleet/gateway/controller doubles with no issued native save. Reuse the existing custody, ticket/fence and shutdown lifecycle. No detached close waiter, additional closing-state API or generic native registry. The store retention index and controller task reference point to the same finite owner; they are not separate admission decisions. A repeated Close must not acquire another ticket or cause earlier cleanup.

## Lanes and exclusive files

| Lane | Exclusive product files | Responsibility |
| --- | --- | --- |
| Shared preparation/store | `Chat/console_native_commit.py`, `Chat/console_chat_store.py` | Captured native outcome and exact ACCEPTED settlement API; defensive retention at directly affected store cleanup points. |
| Controller/provider integration | `Chat/console_chat_controller.py`, `Chat/console_runtime.py` | Consume the exact API, register/release lifetime ownership, preserve publication/drafts, and drain exact owners through lifecycle. |
| Baseline verification | Tests only | New native outcome and integration regression tests; narrowly update the two old tests whose early-cancellation assertion intentionally changes. No product edits or native execution. |
| Root | Plan/task/report/evidence and integration | Sole integration owner; run all native/baseline/timing checks sequentially after both implementations are ready. |

All product files are under `tldw_chatbook/`. Test ownership includes new `Tests/Chat/test_console_native_commit.py` and `Tests/Chat/test_console_native_commit_integration.py`, plus necessary targeted edits in `test_console_durable_commit_offload.py` and `test_console_first_send_atomicity.py`. Existing acceptance/shutdown/composer tests remain controls unless a concrete reviewed assertion change is needed. Request a handoff before crossing lane ownership. No commits by lanes; root commits only the integrated result.

## Verification sequence

1. Baseline is recorded on unchanged `8fff47b36c`: 11 selected acceptance/offload/first-send/shutdown/composer/AgentChatStart controls pass in two sequential contained runs. Both prove normal Job emptiness/identity release and pump retirement, current sources, zero recorder overflow/races, and private-profile removal. These establish current behavior; accepted-cancellation tests already enter dispatch and do not qualify the new ACCEPTED path.
2. Baseline lane writes meaningful real file-backed SQLite controls before product edits. Root records original RED that demonstrates detached cancellation/early ownership release, not only missing imports. Keep original timing budgets. The two old ordinary cancellation tests explicitly awaited cancellation before releasing the lock; change that assertion to pending retention, recording why.
3. New controls cover one exact callback through repeated Stop; original-owner/database replacement; memory affinity; custom errors and exception-after-commit; executor submission failure; queued issuance before worker entry; and global Task cancellation without losing physical completion. Confirm original worker connection retirement, not merely Task.done().
4. Independently observe accepted saved rows and an ACCEPTED checkpoint before explicit/reasonless cancellation settlement. Assert stopped/failed, provider_started=False, no dispatch transition or provider call, no duplicate user row, and retained recovery if settlement fails. Preserve newer and identical-retyped composer content, attachments and prefill revisions.
5. Hold real native work beyond unchanged Close/dispose grace. Assert exact reservation, preparation, store and runtime custody remain while pending; release the worker, then require actual settlement and physical retirement before deletion/end_app_runtime. Repeat cancellation while draining. Run old bounded cleanup controls unchanged.
6. Root reviews final shared API and source semantics after both lanes finish, then runs combined targeted tests, scoped lint/format and relevant existing controls. Rerun only changed/failing scopes. Native runs remain sequential. No full suite, push or merge.
7. Record actual test counts, original RED, source hashes, native retirement and limitations. No new clean Send timing is required solely for this prerequisite unless behavior under test changes Send preparation; it does not claim a speed gain. The subsequent early-receipt slice still requires atomic admission/promotion, screen-free capture and runtime-owned initial hook review before measuring actual 100 ms paint.

## Review resolution

The earlier draft could not proceed because it reused dispatch-only cancellation, allowed bounded Close to retire a live commit, and compared only draft text. Source review found an existing ACCEPTED repository settlement contract; this plan adds its narrowly owned store claim rather than a new transaction format. Root and the source lane selected exact native-owner draining within existing Close/dispose instead of a detached cleanup design. Cancelled composer preservation avoids the unsafe text-equality shortcut. Root additionally required pre-issuance ownership and a private executor Future to cover queued work and direct global Task cancellation. The integration lane completed this review and found no remaining contract blocker, with the submission seal, captured persistence binding and cancellation-settlement ordering explicitly added above. Product writes still follow root-recorded behavioral RED; original baseline evidence is retained.


## Integrated review refinements

The captured source is revalidated after actual native admission, and stock callbacks consume a narrow, pure source binding for exactly one callback lifetime. The binding never carries a permission or lease. The regression held the original real `_core_operation`, switched persistence, and observed the first implementation call the callback once and write two successor messages; `commit-owner-source-drift-red` preserves that failure and normal native retirement.

Accepted cancellation settlement is itself a finite SQLite operation. A named `settle_accepted_durable_turn_owned` helper returns `settled`, `error`, and `caller_cancelled`, accepts the controller owner's original persistence/database explicitly, and shares the private executor/drain implementation. It preserves the synchronous store API and custom callback arguments, uses original-source repository lookup, and keeps the same owner retained through actual terminal settlement or recovery. The memory path stays inline. A real lock regression checks loop responsiveness and retained native ownership during this terminal write.

Runtime shutdown waits for each exact owner's retirement notification, not the whole submit Task: required save publication can finish before unrelated hooks/provider work. A private Future on the existing owner signals that release; it is not another admission state. Existing task snapshots remain available for stop signals. A regression retains the submit Task after native release and requires the native drain to finish.


The real runtime-backed terminal-write regression reproduced a 2.078-second event-loop stall under a two-second SQLite write lock before the async settlement correction. Earlier attempts did not reach settlement: a bare-controller fallback was still in hook configuration admission (confirmed by a bounded failure-only stack snapshot), and extending that new test's setup guard did not resolve it. Those attempts are retained as setup failures, not settlement evidence. The final fixture uses the normal runtime permission-owner wiring, captures real configuration through the supported API, starts its heartbeat only after independent ACCEPTED observation, and acquires its competing lock at settlement entry after commit-worker connection cleanup. Its original five-second setup guard, two-second lock and 500 ms heartbeat limit remain. The existing 2/3-second production grace periods and 15-second native regression are unchanged. The runtime drain control passed with exact retirement notification; its test now disposes the runtime it created.


Final integration review requires three compatibility corrections before completion: cancellation during dispose's original grace must retain the exact native owner; an exception after actual terminal settlement must reconcile the completed original projection instead of leaving an ACCEPTED preparation with no recovery; and trace-provenance early returns must release a physically completed save owner before existing retirement, while preserving a cancelled draft. The first two have independent real-SQLite REDs in `commit-owner-review-red`.

Dispose retains its same caller Task. A small public wrapper drains captured native owners in `finally` around the existing body, while cancellation during the original bounded drain resumes that same remaining grace when a native owner is retained. Normal cleanup completes before cancellation is re-delivered. Earlier cancellation in unrelated cleanup still must wait for native retirement; this does not claim that unrelated skipped cleanup completed.

Supported source replacement means replacing the store or its persistence service, plus defensive refusal of a changed database before callback entry. The stock persistence service assigns its database only at construction; arbitrary mutation of `ChatPersistenceService.db` inside an already-issued custom/stock callback is outside that binding contract. No context-sensitive public database property or persistence-service redesign is introduced to support that otherwise-unused mutation.


## Final combined verification (in progress)

Both implementation lanes handed off frozen source before integrated runs. The last read-only review found the trace-provenance return must preserve the draft for a callback error after actual success as well as caller cancellation. `commit-owner-trace-error-red` independently observed ACCEPTED and then failed `should_clear_draft`; the exact predicate correction passes all three queued/cancelled/error trace routes. The final reviewer reports no remaining actionable source findings.

The current combined scope has 44 passing cases: 15 helper/store regressions, 16 new lifecycle/controller cases, and 13 existing save/shutdown/provider controls. Three additional old controls remain under investigation: the human/revoked hook continuation fixtures never entered their held callback within their original 30 seconds, and the existing whole-Send heartbeat measured 718 ms against its unchanged 500 ms limit. These are not suppressed or counted as passes; an unchanged-source comparison follows the other chat's exclusive native slot. The helper assistant-drift fixture initially attempted a raw UPDATE rejected by the existing semantic mutation guard; its corrected setup uses the authorized optimistic update API and independently verifies the actual changed content/version before testing refusal.

All five final batches retired their contained Job, released its identity, retired pipe/monitor tasks, retained current containment source, recorded zero identity overflow/races, and removed their private profiles. This remains scoped containment evidence, not general production-native qualification. Eight final source/test hashes are retained in `checks/commit-owner-final-v2-source.json`; aggregate receipts are in `checks/commit-owner-final-status.json`. The earlier lifecycle control batch predates only the final trace-error predicate correction; its runtime/helper/store sources were already final.

Ruff has no new diagnostics: 169 store and 61 controller findings pre-exist; helper/runtime and all four changed tests are clean. Six files fully pass formatting. Controller/runtime residual formatter changes have exactly the same 15/38 transformation hunks as committed HEAD, with no new ones. Do not rewrite those unrelated existing lines. Original 2/3-second shutdown grace, five-second new-test setup guards, 500 ms heartbeat threshold and the original 15-second native performance regression remain unchanged. No clean Send timing is claimed for this ownership prerequisite.


## Final result and baseline comparison

The final selected candidate scope has **45 passing cases**: 15 helper/store cases, 17 controller/runtime cases, and 13 existing controls. Three original controls remain failed, explicitly outside a clean-suite claim. On an isolated exact `8fff47b36cd28177bf03edbe5c5922acfaad6ab2` checkout with identical runner and original deadlines, the whole-Send heartbeat also fails (2.031 seconds against 500 ms; candidate 718 ms), and the original human continuation also never reaches the held machine callback within 30 seconds. No timing improvement is inferred from these single samples.

Source diagnosis established the continuation cause: the unchanged stock command executor returns `unsupported_platform` on Windows before launch. Both human/revoked tests require that command to emit a proposal, so neither tests native save refusal here. The added platform-independent control supplies one result through the real declared-result parser, while retaining the original Stop lifecycle, scheduler, authority callback, queue authorization, one-use gate and SQLite transaction. Revocation reaches actual `ContinuationAdmissionRefused`; the transaction rolls back, only the parent user remains, and the queue/preparation/recovery/native owner and worker connection retire. This qualifies host handling of a checked proposal, not command execution.

The final source manifest is `checks/commit-owner-final-v3-source.json`. Product bytes remained frozen throughout the final native comparison; only that one new test was appended and run afterward. The unchanged-baseline and final continuation runs also have normal contained Job/identity/pump retirement, zero overflow/races, current sources, and removed private profiles. Existing whole-Send responsiveness, original native performance acceptance, other-host coverage and pre-existing static-analysis findings remain open. The implementation can be reviewed/committed locally without claiming the overall latency goal or full repository Definition of Done.
