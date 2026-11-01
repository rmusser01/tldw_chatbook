### Task 4: Settle uncertain starts and bound admission contention

**Files:**
- Modify: tldw_chatbook/Chat/console_chat_start.py (owned acceptance/commit drain/cleanup/logging/status constants).
- Modify: tldw_chatbook/DB/automatic_work.py (same-owner accepted uncertainty settlement using existing review state).
- Modify: tldw_chatbook/Chat/console_chat_controller.py (native-start owned durable-commit task only).
- Modify: tldw_chatbook/Chat/message_metadata.py (shared named handoff launch status values, no new module).
- Test: Tests/Chat/test_console_chat_start.py; directly affected automatic-work ledger and controller/owned-DB lifecycle controls; startup ratchets after combined fixes.
- Read: qodo-runtime-triage.md; ADR-211; lessons-testing-evidence.md; existing owned DB call contracts.

**Interfaces:**
- Consumes: original finite allowance root, exact attempt/runtime owner, dual-store receipt, loop-owned source/target/lifetime checks, operation-owned database connection and physically running commit task.
- Produces: accepted-unreceipted attempt leaves active state and requires explicit review in-process; root pauses atomically and stays charged; slot held until all owned physical DB/provider work drains; bounded lock-contention refusal at unchanged ledger source cutoff.

ADR required: no
ADR path: backlog/decisions/211-console-chat-destinations-and-bounded-starts.md (existing).
Reason: Implement existing accepted/uncertain charging and review contracts; use existing review state and same-owner settlement, no schema or new retry policy. Document runtime contention mitigation and retained I/O limit as implementation notes.

- [ ] Verify Qodo1/5/7/8 against triage and real owners. Add RED real-ledger tests for accepted/unreceipted failure and canonical-root pause/charge/no replay, blocked physical commit cancellation retaining capacity until drain, misleading preaccept exception and hostile exception-text nonlogging, and actual SQLite contention event-loop behavior/timeout restoration.
- [ ] Preserve final no-await source/target checks through ledger acceptance. Use the held operation-owned acceptance connection with scoped busy_timeout=0 restored in finally to refuse writer contention rather than freezing the loop. Do not simply offload acceptance or weaken its cutoff. Assign accepted custody before any subsequent fallible restoration. Distinguish proven preaccept lock refusal from accepted/uncertain commit; no accepted refund.
- [ ] Add an owner-fenced atomic ledger settlement of accepted/unreceipted work to existing review_required state, pausing the canonical root with a fixed review reason and retaining the committed generation/deadline/counters. No automatic replay. Handle failed abort or uncertain accept outcomes conservatively if actual durable state already accepted; do not leave same-owner active accepted rows forever. If durable review settlement itself fails, retain a fail-closed runtime restriction on the affected shared allowance until verified reconciliation or owner recovery; launch metadata alone cannot block sibling work. Preserve healthy replacement-owner authority.
- [ ] Own, shield and drain the exact native-start conversation-commit worker, including cancellation, before settling uncertain work or releasing automatic-primary capacity. Preserve ordinary manual/queue durable commit behavior and all source/target/incarnation fencing.
- [ ] Replace shared handoff launch status raw runtime/metadata literals with named values/type vocabulary in already loaded message_metadata; no new eager module. Log unexpected failures with static phase and exception type only. Never log raw exception messages/tracebacks/prompts; do not use logger.exception. Distinguish unexpected preaccept start_failed from ordinary refusal and cancellation.
- [ ] Run complete affected start/ledger lifecycle owners and focused controller commit cancellation controls, actual real-file lock/loop progress control, with private profiles. Controller qualifies combined startup/import budgets after the sequential CI repair so the final integrated tree is measured once. Retain precise cancellation/restore/error receipts and latency limits. No full suite/new markers/suppression/installs.
- [ ] Fatal Ruff, formatter ratchet against recorded BASE and source whitespace; commit only owned source/tests; report exact SHA, RED/GREEN, warnings, connection-opening/fsync limits and self-review.
- [ ] Independent task-scoped spec/quality review; publish verified fixes, reply/resolve each initial Qodo thread and no-inline summary finding, then await current-head review/check completion.

## External review and publication

- [x] Independent task review of Task1, then whole-branch review of the rebased PR. Package diffs before dispatch; reviewers do not repeat completed test runs.
- [x] Push with force-with-lease pinned to the verified old remote head; update the PR description and mark ready to enable reviews.
- [ ] Retrieve all PR issue/review comments and Qodo suggestions. Verify each actionable finding; dispatch concrete follow-up fixes with reproductions and covering tests. Reply and resolve addressed threads with commit/test evidence.
- [ ] Wait for current-head checks and Qodo completion. If future waiting is required, create a quiet thread heartbeat that continues review fixes and merge under the user's authorization.
- [ ] Fetch latest dev again; update/requalify if it advanced. Merge using normal repository gates and exact reviewed head, without admin bypass.
- [ ] Confirm merged state/SHA, close current Backlog criteria, archive this plan's evidence and remove only its disposable SDD directory after completion.
