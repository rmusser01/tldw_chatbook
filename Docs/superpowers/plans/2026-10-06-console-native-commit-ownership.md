# Retained Console commit outcomes implementation plan

Status: Draft, not execution-ready. The material source-review findings below must be resolved before product changes.

Task: [TASK-34563.5](../../../backlog/tasks/task-34563.5%20-%20Plan-retained-Console-commit-outcomes.md).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Preserve the user-requested three parallel lanes with distinct file ownership and root as the sole integration owner. Run integrated checks after both implementation lanes are ready; keep native timing and held-database tests sequential.

**Goal:** Retain each issued ordinary Console save until its actual outcome is known, so Stop, Close and later early receipt cannot release an uncertain turn for duplicate submission.

**Architecture:** One finite Chat operation captures the exact store, database and commit callback, retains its issued native task through cancellation, and returns its actual outcome. The existing controller/store admission, fingerprints, checkpoints and postcommit machinery remain the sole domain authorities. This is the native-custody prerequisite of ADR-222; it does not enable early receipt yet or claim to remove the measured preparation delay.

**Tech stack:** Existing Python 3.12 asyncio/dataclasses, SQLite and native database ownership helpers. No dependency or schema change.

**Spec:** [Console Send preparation architecture](../specs/2026-10-06-console-send-preparation-architecture-design.md), especially sections 3 and 6.

ADR required: yes, existing decision applies.
ADR path: backlog/decisions/222-console-send-preparation-and-io-ownership.md.
Reason: implements the accepted explicit ordinary native-work retention prerequisite, preserving ADR-126 storage ownership and existing recovery semantics.

## Scope and evidence

Continue the isolated codex/console-send-preparation-plan candidate from f70cacae3b. The completed shared-tool slice and current attribution diagnostics remain intact. Ordinary clean commit-stage samples were 0.118–0.257 seconds; most elapsed delay is elsewhere. This change is required for safe immediate feedback, not justified as a save-speed optimization. Subsequent receipt enablement still requires atomic admission/promotion, screen-free capture and runtime-owned initial hook review.

Current ordinary `_accept_durable_turn` awaits `_run_durable_db_call`, whose cancelled `to_thread` await leaves its physical transaction running. AgentChatStart already retains `commit_worker`. Limit this change to the ordinary acceptance commit; leave general durable transaction calls and AgentChatStart behavior intact.

## Global constraints

- Overall targets remain actual rendered receipt/input feedback within 100 ms and ordinary application overhead to actual adapter under one second. This prerequisite alone cannot qualify either target.
- A failed saved-turn acceptance stops Send and keeps the draft. Existing temporary chats remain available. Preserve WAL/NORMAL, required checkpoints and consent/trace/context facts.
- Stop fences provider entry; cancellation does not imply rollback. Native work retains its exact captured owners through repeated cancellation until physical retirement and observed outcome.
- Preserve required postcommit ordering and awaited prompt history. No unsaved fallback, automatic uncertain replay, second admission ledger, native proof across await, or new recovery format.
- Preserve supported public signatures, custom callback behavior, memory-database affinity, source/revision fences, queued origins, attachment ownership and newer draft revisions.
- Use targeted checks with original deadlines. No full suite without the user's opt-in. No push, merge or change to another chat's worktree.

## Review focus

1. Stop twice while a real SQLite writer holds the commit: neither provider entry nor retry/admission release may occur before the exact worker finishes.
2. The physical transaction succeeds after cancellation: preserve the accepted durable identity and settle/recover it without a second user turn.
3. The worker fails after cancellation, including an exception after durable success: consume its exception and distinguish proven rollback from existing unresolved acceptance; absence of an in-memory result is not rollback proof.
4. The controller's store/database is replaced, or its chat closes during the held operation: the issued callback retires against its original owners and cannot publish into a replacement.
5. Shutdown is cancelled again while draining, or a caller supplies a custom/memory-backed persistence object: preserve exact lifetime, thread affinity and existing error/call shape.

## Files and lane ownership

| Lane | Exclusive files | Responsibility |
| --- | --- | --- |
| Shared preparation | `tldw_chatbook/Chat/console_native_commit.py`; `Tests/Chat/test_console_native_commit.py` | Finite captured-owner operation and outcome retention. |
| Controller integration | `tldw_chatbook/Chat/console_chat_controller.py`; `Tests/Chat/test_console_durable_commit_offload.py`; `Tests/Chat/test_console_durable_turn_acceptance.py`; `Tests/Chat/test_console_first_send_atomicity.py` | Consume outcome, preserve accepted cancellation/recovery, and pin actual user-facing behavior. |
| Baseline verification | New ownership verification report and existing qualification harness evidence only | Establish unchanged controls, review final diff, run qualification after root integration. No competing product edits. |

Root owns plan/task records, API decisions and integration. A lane must request a handoff before editing another lane's file. Native runs are serialized through root.

## Task 1: Retain the exact ordinary commit operation

**Interfaces produced in `console_native_commit.py`:**

- `ConsoleNativeCommitCompletion`, frozen/slotted, with `commit: ConsoleDurableTurnCommit | None`, `error: BaseException | None`, and `caller_cancelled: bool`. This is a short-lived outcome of one issued operation, not authority or a retry instruction. Exactly one of commit/error is populated.
- `async commit_durable_turn_owned(store: ConsoleChatStore, acceptance: ConsoleDurableTurnAcceptance) -> ConsoleNativeCommitCompletion`.

The operation captures the bound commit callback and database before scheduling. Preserve the current inline memory-database branch; use the existing owned native DB callback boundary for offloaded work, with captured owners rather than a late controller lookup. Keep the exact issued task referenced and shield/drain it through repeated caller cancellation. Consume its terminal result/error before returning. Do not start a replacement worker. If scheduling fails before issuance, close the unowned coroutine and preserve that failure.

- [ ] Write `test_repeated_cancel_retains_exact_native_commit_until_retirement`, holding a real file-backed SQLite transaction. Assert the caller stays pending after repeated cancellation, exactly one original callback executes, the original database owns it, and the final completion records actual success plus cancellation.
- [ ] Add meaningful failure controls: original callback rollback/error is retained without unhandled-task noise; a changed external store reference cannot redirect the captured callback; memory-backed work retains its existing thread; custom callbacks preserve invocation count and error identity; task creation failure leaks no coroutine/native work.
- [ ] Run the new targeted file on unchanged source and record the expected failures, including the ordinary unretained behavior rather than only a missing import.
- [ ] Implement the finite operation without eager product initialization or another worker manager. Run its tests and scoped Ruff/format checks.
- [ ] Commit the independently verified source operation after root review.

## Task 2: Integrate actual commit outcomes before cancellation settlement

**Consumes:** Task 1's exact operation/result API. **Produces:** unchanged public submit APIs; an optional private `cancel_before_dispatch: bool = False` keyword on `resume_durable_postcommit` if needed to route the observed cancellation through its existing accepted-cancellation handler.

- [ ] Extend the existing held-commit test in `test_console_durable_commit_offload.py` to assert that cancellation remains pending until the held transaction is released, then verify original checkpoint/message identity, no provider entry and no duplicate Send. The previous test deliberately expected its awaiter to finish before releasing the worker; document that this assertion changes because ADR-222 intentionally strengthens lifetime ownership, not to relax a timing budget.
- [ ] Add repeated Stop, Close and shutdown during the same real held commit. Assert no handle is retired early, no admission is reused while outcome is unknown, all native owners retire, and newer composer content survives.
- [ ] Add late-success, proven-rollback and exception-after-commit cases. Use `durable_turn_commit_for`, fingerprints and repository outcomes to preserve the existing accepted or unresolved recovery owner. Never infer rollback from cancellation or an empty in-memory result. Reuse the existing per-write rollback controls.
- [ ] Add captured-store/database replacement tests: original work drains; no successor store/session receives publication, clearing or cleanup from the old attempt.
- [ ] Replace only the ordinary commit branch in `_accept_durable_turn`. Normal success/error keeps the existing behavior. Observed cancellation after actual success must mark real acceptance and use existing postcommit publication/cancellation settlement before any checkpoint dispatch transition or adapter call. A reasonless cancellation retains the existing failed/recovery semantics; explicit Stop retains the stopped semantics. No automatic retry.
- [ ] Preserve the existing AgentChatStart branch, unrelated `_run_durable_db_call` users, save-refusal copy, awaited history and custom/memory-backed routes. Ensure the captured original owner survives through settlement; reject source replacement before publication.
- [ ] Run the changed-file controls plus existing accepted cancellation, first-send atomicity and runtime shutdown ownership cases, with unchanged deadlines. Inspect every failure and source/native retirement receipt. Commit after root review.

## Task 3: Integrated qualification and handoff

- [ ] Before implementation, baseline the exact targeted acceptance/offload/runtime ownership controls on current source using the established contained private-profile runner. Preserve any existing failure as a baseline finding.
- [ ] After both lanes finish, root verifies the agreed API, source diff and no competing ownership. Run the combined targeted scope once, then only rerun changed/failing scopes.
- [ ] Review the integrated cancellation paths against all five Review Focus cases, including actual worker completion and live original connection ownership rather than only task cancellation.
- [ ] Run source-current actual Enter/button/Stop pump controls while the native operation is held. Report the original responsiveness limit honestly; do not claim 100 ms paint from the old pump tests.
- [ ] If clean Send timings are rerun, use sequential baseline/candidate samples with the same original budgets, enabled features and final adapter seam. Retain raw timings and retirement receipts; this prerequisite is not expected to cure the multi-second preparation cost.
- [ ] Record focused test counts, source hashes, native retirement and remaining speed/receipt/host gaps. Keep the main performance work open. The next receipt plan may consume this verified prerequisite; it must also include the initial hook bridge and atomic promotion before enabling early feedback.

## Review findings that must be resolved before implementation

A source review found three material gaps in this draft. It is not yet an execution-ready prerequisite:

1. ACCEPTED is not DISPATCH_STARTED. The existing stream cancellation helper refuses ACCEPTED recovery, and the ordinary resume tail reports provider_started=True. Specify an owned terminal transition from ACCEPTED through existing recovery claim/generation and repository settlement machinery, without manufacturing dispatch. Test explicit and reasonless cancellation with provider_started=False.
2. The runtime's bounded Close/dispose wait can release custody and remove store reservations while the worker still runs. Add exact retained-native-owner handoff to runtime/store lifecycle ownership, or keep that lifecycle requirement explicitly unmet. Qualification must hold the worker beyond the existing close/dispose budget and preserve its real owner; a prompt release is insufficient.
3. Existing draft clearing compares text, so identical retyping can be cleared by late-success publication. Require an exact draft revision proof, or preserve the composer on the cancelled path where that proof is absent; include identical retyping as well as different newer text. Attachment IDs and prefill revisions remain exact.

Reconciliation must use the captured original store throughout. The operation/result interface is feasible, but the consuming settlement/lifecycle contract must be completed before product changes. This draft is retained so those gaps cannot disappear in a later implementation. The measured latency work continues independently.

## Review state

This plan preserves the execution method already chosen by the user. Product implementation has not begun. Complete source/spec review of this concrete plan before enabling its product changes; do not reopen the saving policy or execution-method decision.
