### Spec Compliance

- ❌ Issues found: the failed-settlement restriction can be cleared by a superseded recovery after a replacement owner has started uncertain work. This violates the required shared-allowance denial until verified reconciliation/recovery; see Important I1.
- ✅ The remaining Task4 implementation requirements are present: charged atomic root review, separate physical commit custody, unchanged no-await acceptance cutoff, scoped writer timeout, bounded diagnostics, shared metadata vocabulary, and public API documentation. References: `DB/automatic_work.py:936`, `Chat/console_chat_controller.py:12783`, `Chat/console_chat_start.py:432`, `Chat/console_chat_start.py:617`, `Chat/message_metadata.py:35` (all production paths under `tldw_chatbook/`).
- ⚠️ Final combined startup/import qualification, external current-head CI/Qodo, publication and merge are controller-owned pending work, not verified by this task review. Cold-open/WAL/fsync latency remains synchronous as explicitly scoped in the brief; the warm contention control does not prove a general I/O latency bound.

### Strengths

- `tldw_chatbook/DB/automatic_work.py:958–970` checks durable runtime and exact attempt ownership inside the FULL transaction, updates the attempt and canonical root together, and changes neither counters nor clock anchors. `Tests/DB/test_automatic_chat_starts.py:390` checks retained charges/deadlines, idempotence, sibling refusal and no replay.
- `tldw_chatbook/Chat/console_chat_controller.py:12783–12800` captures the exact native store/database and retains a shielded commit task before its first await. `Chat/console_chat_start.py:443–457` drains that owner separately from provider custody before settlement and capacity release. The real blocked-commit control at `Tests/Chat/test_console_chat_start.py:2728` exercises repeated cancellation, physical exit, retained capacity and connection retirement.
- `tldw_chatbook/Chat/console_chat_start.py:617–654` keeps live source/target checks and durable acceptance in one no-await region, changes only the held connection's busy timeout, and records accepted custody before restoration. The file-backed contention test at `Tests/Chat/test_console_chat_start.py:2933` exercises a real SQLite writer lock, borrowed timeout restoration, loop progress and a proven prepared refund.
- Failure logging uses fixed phases and exception types; `Tests/Chat/test_console_chat_start.py:2783` checks hostile exception text is absent and differentiates preaccept failure from accepted cleanup uncertainty. The public settlement API has Args/Returns/Raises at `tldw_chatbook/DB/automatic_work.py:944–955`.
- Fixture changes preserve the original test bodies and assertions and use the existing exact-node private-profile runner. The runner invokes the original function in its child and checks the exact JUnit case and outcome (`Tests/private_profile.py:51–60,164–179`); no guards, assertion bodies or skip markers were weakened in this patch.

### Issues

#### Critical (Must Fix)

- None found.

#### Important (Should Fix)

- **I1 — Fence recovery's transient cleanup against later owners.** `tldw_chatbook/DB/automatic_work.py:1221–1228`: the registry sweep happens after `self.transaction()` has committed and released SQLite ownership. It deletes every entry whose owner differs from this invocation's `current_owner_id`, including entries created by a newer owner after a subsequent recovery. Concrete interleaving: recovery A commits owner A and pauses before the sweep; recovery B commits owner B and returns; B accepts a chat start whose settlement is unconfirmed and installs its restriction; A resumes and deletes B's entry. The attempt remains accepted, the root remains active, and sibling admission is available again without reconciliation. Runtime recovery really runs in worker threads (`tldw_chatbook/Chat/console_fleet_wake.py:979`), so the DB transaction is the ordering boundary, not function return.
  - **Evidence:** [focused real-file probe](task-4-review-recovery-race.json), run with the existing Python 3.12 venv against production SHA256 `05c65a259f916744541e569b5f6f4812d0a3add79016a40bd23a69f16318617b`. Before A's late cleanup, B's `check_active` raised `settlement_unconfirmed`; afterward it returned `active`, while B's attempt was still `accepted` and its used generation was 1. The probe pauses only the old transaction context's return after its actual commit; it does not replace admission or recovery SQL.
  - **Fix:** retire only restrictions covered by this committed recovery, with ordering that cannot remove later-owner records. For example, capture the specific foreign-owner entries while this recovery still owns its durable transaction, then retire only that captured set after successful commit. Add a deterministic overlap control alongside the existing sequential replacement/current-owner tests. Do not add a fresh unfenced current-owner read followed by another global sweep.

#### Minor (Nice to Have)

- None requiring a change. The acknowledged owner-wide restriction when the root is unknown is conservative but consistent with fail-closed uncertainty; an independent established-root manual allowance remains usable. The synchronous cold-open/fsync limit is accurately reported and is not represented as full off-thread admission.

### Checks and Evidence

- Read the complete immutable 2,408-line `review-fc1627e99b..d3b2ade44b.diff` once, covering all ten changed paths. Reviewed brief, binding constraints, triage, ADR-211 and relevant testing-evidence guidance. No broad branch review, source/index/HEAD changes or duplicate suites.
- Named cross-file checks: physical-worker lifetime and native handle retirement (`DB/base_db.py:39–99`); held-connection timeout reuse (`DB/AgentRuns_DB.py:358–447`); shutdown draining (`Chat/console_chat_controller.py:21006–21008`); loop-owned source checks (`Chat/console_chat_controller.py:19150–19210`); exact native store commit contract (`Chat/console_chat_store.py:6006–6060`); private-profile body execution (`Tests/private_profile.py:51–179`); recovery worker call site (`Chat/console_fleet_wake.py:940–979`). Read unchanged/cut-off coordinator preparation, authority, drain and ledger transaction/admission functions only to evaluate these concrete custody/registry risks.
- Independently parsed final JUnit and receipts: successful complete owner batch **239** cases; final-byte DB/manual diagnostic batch has **29 DB + 30 manual passing cases** but overall exit 1 because the unisolated child owner failed; repaired final DB/child batch **50** cases; final latency controls **2** cases. The deduplicated union is **260** passing cases. The failed batch is not counted as a green invocation. Final production hashes are identical across these receipts; final test hashes match the frozen manifest in their relevant later receipts. Commit-verification metadata binds that manifest to `d3b2ade44b49afc96f78c592ac271e3231472b27` and parent `fc1627e99b8722a0a69812530d950410e8356ba0`.
- Historical RED/oracle/profile/FD diagnostic failures remain disclosed. Final successful logs report no warnings. Fatal Ruff and committed formatter-ratchet receipts report exit 0 on the final manifest. No test suite was rerun by this reviewer.
- Executed only the new unanswered recovery-overlap probe. Its first invocation stopped before DB creation because the probe had not created private config parent directories; corrected that probe setup, then the real-file interleaving reproduced I1. The successful evidence records this setup diagnostic. All probe threads were joined, the private DB/profile removed, and only this review and its evidence were written.

### Assessment

**Task quality: Needs fixes.**

The acceptance, durable review and physical commit repairs are well scoped and have strong real-owner coverage. The new registry's postcommit recovery sweep has an uncovered ownership race that can reopen automatic admission while accepted uncertainty remains unresolved; fix I1 before approving Task4.
