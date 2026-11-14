# Task 15: Qodo cleanup preflight

## Scope and result

Read-only diagnosis at `b1efc37ae8f481c1df01e2f253d5878ba68e35a4` in `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Both current findings **4179166076** and **4179166078** are valid. Finding4179166071 is excluded. Read the task14 inline comments and current summary, production coordinator and ledger contracts, ADR219, TASK34213, relevant existing tests, and the testing-evidence lessons. No tracked/source/test/index/HEAD mutation, full suite, passed-cohort replay, dependency installation, or global configuration change occurred. `git status --short` remained empty and HEAD remained fixed after the probe.

Private evidence: `task-15-qodo-private-probe.py`, `task-15-qodo-private-probe.json`, empty `task-15-qodo-private-probe.stderr`, and `task-15-private-data/`, all in this SDD. Each private probe was executed once with the shared Python and worktree PYTHONPATH, with bytecode disabled and private process configuration/data paths. A second private probe (`task-15-qodo-private-owned-probe.py` / `.json` / `.stderr`, `task-15-private-data-owned/`) explicitly seeds and records durable runtime owner `owner`; all five results exactly match the original cases. The original probe left that row absent under the existing permissive contract. Both receipts are retained. The probe calls production `_run`, real file-backed AgentRunsDB, real preparation/acceptance/abort/review transactions and admission. It substitutes the preparation boundary and outcome publication with small surrogates. Attached worker tasks are event-held asyncio tasks, not native provider/commit threads. This proves cleanup control flow and real ledger consequences, not a complete mounted/native journey. No repair was implemented or claimed verified.

## 4179166076: registration can bypass cleanup

`console_chat_start.py:424–436` invokes `_restrict_chat_start` before any protected cleanup region. The second invocation at532–539 is inside the settlement exception handler but has no inner protection. Either can escape `_run` before outcome resolution and capacity release. The first also skips physical commit/provider draining. `_restriction_database` calls `Path.resolve()` for file stores; failures from that filesystem observation can propagate. Current registration performs that observation while holding the restriction lock. The probe injects an OSError at the registration boundary; it does not claim to reproduce a real filesystem failure.

Private results:

| Case | Durable attempts/reservations | Outcome | Claim / active item | Attached workers |
| --- | --- | --- | --- | --- |
| First registration raises | 0 / 0 | unresolved; OSError escapes | retained / retained | both unfinished |
| Second registration raises after missing-row abort | 0 / 0 | unresolved; OSError escapes | retained / retained | none attached |

`start()` awaits the separate outcome future (`:318`), not the task itself. A failed coordinator task can therefore leave the caller waiting even after the task set's done callback discards that task. Capacity is held by the captured owner/token, and release exists only at591–597. Retaining capacity is required while physical work remains, but retaining it forever after cleanup escapes is the defect.

## 4179166078: absence is conflated with unresolved settlement

`_prepare` assigns `item.context` only after its owned preparation task returns. A non-refusal exception before any write, or after a successful commit but before the return is observed, leaves context unset. `_run` correctly treats preparation as potentially real work, installs a None-chain restriction and attempts abort. However, `_chat_start_attempt` raises `ValueError("unknown chat start attempt")` on actual absence; abort propagates it; the settlement handler installs the same wildcard again.

`_check_unconfirmed_starts` treats None as every allowance root for the current database/runtime owner. `_check_admission` applies it to the automatic work operations. Same-owner recovery deliberately does not clear that entry; different-owner recovery retires covered entries. This is a current-owner/database admission problem, not a process-wide block across unrelated databases.

Private trigger rollback case: the actual attempt INSERT raises SQLite IntegrityError after the target-chain and reservation INSERTs. The ledger rolls back the transaction: zero attempts, zero reservations, two original chains (source and unrelated). Despite zero charges, both source and unrelated `check_active` calls fail with `settlement_unconfirmed`; outcome is review_required/settlement_unconfirmed. Capacity and active registration do clear when registration itself works.

### Transaction evidence and ambiguity controls

`AutomaticWorkLedger.transaction` runs `PRAGMA synchronous=FULL`, `BEGIN IMMEDIATE`, commits after the body, rolls back on BaseException, then restores the previous synchronous policy. Preparation inserts its target chain, reservation and attempt in the same admission transaction. A successful later serialized absence observation, after the original physical operation is drained, can establish there is nothing to settle. A Python exception alone cannot: commit may already have succeeded, the policy-restoration/connection-retirement phase may fail, or a wrapper may raise after a successful return.

The same private probe preserves these current behaviors with context unset and accepted false in Python:

- Prepared commit then raise: row becomes aborted; reserved0, used0; both chains admitted. Existing outcome remains review_required/preparation_unconfirmed. This report does not propose a new launch-status vocabulary or unrelated outcome downgrade.
- Accepted commit then raise: abort returns false; review settlement finds accepted authority, retains used generation1 and the original allowance accounting, sets attempt/root review_required, and makes source admission fail with interrupted_work while unrelated admission succeeds.

The latter is an adversarial storage-boundary control, not a claim that ordinary `_prepare` accepts work itself. It establishes why local flags and exception names cannot authorize refunds.

## Smallest bounded repair to brief

1. Protect **both** restriction-registration calls with a small nonthrowing coordinator cleanup helper (type-only phase logging). Registration failure must not skip commit/provider draining, settlement, outcome fallback, active-item cleanup or the captured owner's exact-token release. Reattempt registration when durable settlement remains uncertain. Preserve an already installed restriction if a later registration fails. Keep both physical drains independently protected and reachable. After those drains, use nested try/finally cleanup so outcome/publication, abandonment or run-state cleanup exceptions cannot skip exact active-item removal and the captured owner/token release. Keep outcome fallback independently guarded so an abandonment error cannot strand its future. Do not put release in a finally that can run before physical drains finish; the release guarantee starts after both owned physical tasks have actually settled.
2. Preserve `abort_chat_start(attempt_id, owner_id) -> bool` and its existing strict unknown-row/owner checks. Add a narrowly named private ledger predicate for **confirmed missing attempt**: use the existing FULL/BEGIN IMMEDIATE transaction, require a present durable runtime-owner row equal to the captured owner, query the exact attempt ID, and return true only after the transaction context exits successfully. A missing runtime-owner row is uncertainty: the existing `_check_runtime_owner` helper alone is insufficient for this predicate because it accepts an absent owner row. Any row, including a foreign-owner row, returns false or raises; any read/begin/commit/restore failure propagates as uncertainty. No reservation or attempt mutation is necessary.
3. Use that predicate only for the unaccepted, context-unavailable preparation-failure cleanup after owned work has drained and ordinary settlement failed. A true result permits clearing only this attempt/owner's restriction and avoids re-adding it. For this positively confirmed absence branch, use the existing not_started/start_failed outcome (retaining the existing withdrawal-reason override). False/exception continues existing review-required/settlement-unconfirmed behavior. Do not infer absence from exception text, `OperationalError`, `context is None`, `item.accepted is False`, or a false abort result.
4. Preserve the current wildcard semantics for genuinely unknown roots. Qodo's optional source-root narrowing is unnecessary to repair proven absence and changes the scope of unconfirmed authority; omit it. Do not call recover under a new owner just to clear this entry.

A bounded nonthrowing registration helper preserves denial where registration or durable settlement succeeds. It cannot honestly guarantee cross-handle denial when **both** registry insertion and durable settlement are unavailable. Do not hide that limitation with catch-and-pass. If the implementation includes removal of the known Path.resolve failure surface, establish the same canonical database restriction identity before acquiring automatic authority and reuse it for cleanup; refuse admission if initial identity cannot be established. Keep canonical alias equivalence and private memory-store identity. Avoid an ad hoc lexical/object-key fallback: peers would use a different key. Such hardening needs its own narrowly scoped alias/identity control, not a broad storage-identity redesign.

### Outcome decision for confirmed absence

Recommend existing `not_started` / `start_failed` only after the new absence predicate returns true. Production `_run` cannot reach `submit_draft` until `_prepare` returns successfully and installs context; this branch requires no context, no local acceptance and the drained preparation worker. The same-owner transaction then positively proves no attempt/reservation authority survived. There is consequently no native acceptance/dispatch or second conversation receipt to reconcile for this failed preparation. ADR219 Decision7 calls for truthful launch outcomes and review for accepted/uncertain work; once absence is proven, this particular uncertainty has ended. The target's already durable conversation/draft remains intact.

Keeping review_required/preparation_unconfirmed would be conservatively compatible and minimize display changes, but would ask the user to review an uncertainty the ledger has just resolved. The more precise existing status is preferable and should be pinned by the new real-row test. Do not extend this status change to the existing prepared-commit-then-raise path, false/failed abort, failed absence observation, wrong/missing runtime owner, accepted-row evidence or publication failure. Those remain under current behavior; outcome publication can still truthfully replace any result with review_required/outcome_unconfirmed.

### Interaction and caller compatibility

Fixing only registration lets missing-row cleanup finish but leaves the wildcard denial. Fixing only missing-row settlement leaves the initial-registration escape and the second-registration escape for other settlement failures. Test them separately and in combination.

The only production caller of `abort_chat_start` is the coordinator (passed as a callable at505). Existing DB callers/tests rely on true meaning an actual prepared/reserved refund, false for repeated abort/accepted/completed/separately committed reservation, and owner/recovery fencing. Coordinator tests deliberately replace abort with false/raise to represent unresolved settlement. Changing unknown absence to truthy success on that public method needlessly conflates "nothing existed" with "refunded" and risks weakening those controls. A private transactional predicate keeps all existing abort seams operative and avoids string-matching a generic ValueError.

The initial preparation task is already shielded and drained on cancellation. Preserve it. Commit/provider tasks remain distinct physical owners; repeated coordinator cancellation must not free their slot. Keep the exact captured capacity owner/token and do not remove a replacement item's active entry or preparation state. No retries of the opening request.

## Exact RED proposals (new selectors)

Use the real `_native_start_rig` in `Tests/Chat/test_console_chat_start.py`; retain private profile/bootstrap conventions. These are proposed names, not already existing tests:

- `Tests/Chat/test_console_chat_start.py::test_restriction_registration_failure_still_drains_exact_native_owners`: parameterize first registration and settlement-handler registration failure. Hold actual commit or provider worker, cancel repeatedly, prove claim/outcome stay owned until exit, then outcome resolves and exact claim retires. Include an already installed restriction and a replacement-token control. Add outcome-publication, abandonment and run-state cleanup failures after physical exit to prove the final exact release remains reachable without altering successful cleanup semantics.
- `Tests/Chat/test_console_chat_start.py::test_failed_preparation_with_no_attempt_clears_only_its_restriction`: real trigger rollback and real transient writer contention released before reconciliation; prove zero attempt/reservation, saved draft intact, no provider dispatch, unrelated automatic admission succeeds, exact capacity retires. Include registration failure plus actual missing row.
- `Tests/Chat/test_console_chat_start.py::test_preparation_exception_reconciles_persisted_authority_conservatively`: prepared commit-then-raise, accepted commit-then-raise, and reconciliation DB failure. Accepted charge/root pause and uncertain wildcard survive; a failed observation must never count as absence.
- `Tests/DB/test_automatic_chat_starts.py::test_missing_start_confirmation_requires_successful_owned_transaction`: absent/present/foreign-owner/stale-runtime/missing-runtime-owner cases; injected transaction exit failure proves no positive absence result; assert public unknown-row abort still raises and existing rows/charges unchanged.

Avoid a synthetic exception-only missing-row test as the sole evidence: the real trigger proves the transaction discarded its earlier chain/reservation writes. New controls should inspect admission through a second handle and prove exact-attempt-only clearing when another entry exists.

## Existing covering selectors to retain in the eventual brief

No existing cohorts were rerun during diagnosis. Relevant exact selectors:

- `Tests/Chat/test_console_chat_start.py::test_unconfirmed_preaccept_settlement_requires_review`
- `Tests/Chat/test_console_chat_start.py::test_initial_preparation_keeps_owner_until_uncertain_refund`
- `Tests/Chat/test_console_chat_start.py::test_stop_keeps_native_claim_until_actual_bridge_worker_exits`
- `Tests/Chat/test_console_chat_start.py::test_stop_keeps_native_claim_until_generic_provider_adapter_exits`
- `Tests/Chat/test_console_chat_start.py::test_native_commit_keeps_exact_worker_and_capacity_until_repeated_cancel_drains`
- `Tests/Chat/test_console_chat_start.py::test_unexpected_start_failure_is_type_only_and_accepted_cleanup_is_review`
- `Tests/Chat/test_console_chat_start.py::test_failed_review_settlement_blocks_siblings_across_ledger_handles`
- `Tests/DB/test_automatic_chat_starts.py::test_preparation_identity_abort_and_completion_are_exact`
- `Tests/DB/test_automatic_chat_starts.py::test_failed_native_write_rolls_back_member_and_reserved_generation`
- `Tests/DB/test_automatic_chat_starts.py::test_acceptance_and_abort_race_has_one_winner`
- `Tests/DB/test_automatic_chat_starts.py::test_native_abort_never_refunds_a_separately_committed_reservation`
- `Tests/DB/test_automatic_chat_starts.py::test_same_owner_uncertain_start_pauses_root_without_refund_or_replay`
- `Tests/DB/test_automatic_chat_starts.py::test_review_settlement_never_rewrites_nonaccepted_work`
- `Tests/DB/test_automatic_chat_starts.py::test_stale_review_settlement_cannot_pause_healthy_replacement`
- `Tests/DB/test_automatic_chat_starts.py::test_transient_uncertainty_is_store_scoped_and_stale_owner_cannot_poison_replacement`
- `Tests/DB/test_automatic_chat_starts.py::test_late_recovery_cleanup_preserves_new_owner_uncertainty`

## ADR check

ADR required: no new ADR for this bounded repair.
ADR path: backlog/decisions/219-console-chat-destinations-and-bounded-starts.md.
Reason: Restore existing Decisions4/7 (uncertainty blocks siblings, proven preacceptance cleanup does not charge, accepted uncertainty stays charged) and physical ownership/cleanup guarantees. No schema, public abort contract, allowance, recovery-owner, retry or unknown-root policy change. The positive absence branch uses existing truthful not_started/start_failed vocabulary, without changing uncertainty outcomes. If the source task chooses to change any of those contracts, reassess ADR requirements before implementation.

Implementation sizing: these changes belong only to the existing start coordinator, ledger and exact tests. Do not move controller code or helpers to satisfy unrelated module-size rows, and do not raise any module/storage budgets.
