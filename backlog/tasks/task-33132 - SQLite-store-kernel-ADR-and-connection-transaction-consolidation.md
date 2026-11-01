---
id: TASK-33132
title: SQLite store kernel ADR and connection-transaction consolidation
status: In Progress
assignee:
- '@robert'
created_date: 2026-09-27 19:46
labels:
- refactor
- db
dependencies: []
priority: high
updated_date: 2026-10-31 21:11
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The DB layer carries 21 independent transaction() context managers and 9 verbatim copies of the held thread-local connection plus liveness-ping idiom, because base_db.py provides almost none of what a store actually needs. The copies know they are copies: RAG_Indexing_DB.py documents copying the Workspace_DB idiom, AgentRuns_DB.py says it follows the Workspace_DB pattern, and three classes carry close() aliases to feel like BaseDB without inheriting it. One kernel in base_db (held connection with liveness ping, transaction with defined nesting semantics, column-exists helpers) makes the base class real and deletes roughly 2k+ LOC of copied plumbing. ChaChaNotes' richer transaction semantics (depth tracking, quiescence) become the canonical implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An ADR records the kernel decision with ChaChaNotes transaction semantics as canonical.
- [ ] #2 The held connection plus liveness-ping idiom is implemented once and consumed by every store that currently copies it.
- [ ] #3 Transaction nesting semantics are documented and covered by tests.
- [ ] #4 No store retains a private copy of the consolidated machinery.
- [ ] #5 Targeted DB test suites for every touched store pass.
<!-- AC:END -->

## Renumbering provenance

Filed 2026-09-27 as task-33087. origin/dev minted its own task-33087 before this branch merged, so per the landed-keeps-id rule this task moved to task-33132; every inbound reference moved with it.

## Implementation Plan

Premise re-verified against origin/dev (2026-10-12): 25 `transaction()` definitions (growing: 18→21→25), 10 held-connection copies. Two design-relevant discoveries: `Backup_Recovery/participants.py` already coordinates stores via `_core_*` hooks/decorators (the kernel must keep it observable by construction), and `RAG_Indexing_DB` is the hardened, best-documented reference port of the Workspace pattern (task-15465/15466/19566/3012/32801.5 lineage) — it becomes the pilot and the comment donor.

1. ADR-228 (backlog/decisions/228-sqlite-store-kernel.md): kernel = `DB/store_kernel.py` mixin; canonical semantics from the hardened Workspace/RAG pattern (autocommit held connections, one-place PRAGMA configuration, BEGIN IMMEDIATE with BaseException rollback, fail-loud nesting, participants integration built in, per-store `:memory:` overrides). AC #1's original wording ("ChaChaNotes semantics canonical") is refined there: the fleet's de facto canonical is the fail-loud variant; ChaChaNotes' depth-tracked manager stays the documented exception.
2. Implement `DB/store_kernel.py` with tests covering nesting failure, BaseException rollback, liveness replacement, and participants observability.
3. Pilot-migrate RAG_Indexing_DB onto the kernel (comments move verbatim where they describe kernel behavior); run its test suite.
4. Migrate remaining held-connection copyists one store per PR (Workspace, AgentRuns, Prompts, Client_Media, Library_Collections, event-state, client-notifications, notes-device, receipts...), 25 transaction() definitions shrinking accordingly.
5. Stores with genuinely different contracts (Chunking_Lab durability, Client_Media blob layer) may consume the kernel without its policy — documented per store.



Migration 3 (2026-10-14): AgentRuns_DB onto the kernel (BaseDB provenance, busy_timeout-first policy, no transaction override — kernel immediate default matches). Same PR lands the tri-state _transaction_immediate_default decision from the workspace-PR review and simplifies Workspace to the class attribute. Direct battery 125/125.


Migration 4 (2026-10-16): Library_Collections_DB onto the kernel. Deltas preserved as overrides: the WAL-retry helper (_enable_wal, concurrent-opener tolerance), the guarded read_transaction (BEGIN DEFERRED snapshot + total_changes write-guard), and the busy-safe PRAGMA policy; the standard transaction()/close()/held-connection machinery deleted (kernel default BEGIN IMMEDIATE matches the store's cross-process MCP rationale, noted at the owner attribute). Battery: collections capture migration + offline store + capture extraction/app wiring + kernel + pragma suites 135/135.

Migration 6 (2026-10-19): GuardianDB onto the kernel — the standard BaseDB-provenance shape (db.base open, WAL via the retained concurrent-opener retry helper, synchronous NORMAL, autocommit, no transaction override). The kernel gains the _on_connection_opened hook (fires once per fresh connection incl. liveness replacement, after cache entry — pinned) as the seam Dreams_DB's cross-thread close_all registry builds on next. Surgery lesson recorded: the line-splice approach's count invariants cannot see decorators — review round 1 caught _expected_schema losing @classmethod/@lru_cache; restored verbatim. Battery: kernel + guardian owner/activation/db + adjacent lifecycle suites 65 green; hook pin added.

Migration 5 (2026-10-18): ClientNotificationsDB onto the kernel (connection/transaction/close + held-connection file branch; task-32801.5 BaseException rollback upgrade over its old except-Exception). Store deltas preserved as overrides: ensure-schema lock, dual-owner _open_connection, shared-connection :memory: policy (ADR-228 per-store clause). One migration bug caught by the battery: the preserved super()._get_connection() resolved to the KERNEL under the new MRO, recursing through _open_connection — fixed with the explicit BaseDB._get_connection(self) provenance pattern; a reminder that preserved code needs re-reading against the new MRO, not just moving. EventStateRepository classified as an ADR-228 exception (dict-keyed cross-thread close; shared memory branch). Battery: full Tests/Notifications/ suite 101/101, matching pristine-dev baseline exactly.

Review round 1 (yes-verdict): 174 green across Notifications + Subscriptions/test_client_notifications_db.py + DB suites (the store's connection-machinery batteries live outside Tests/Notifications/ — cite those per migration going forward). The recursion fix is doubly load-bearing: BaseDB._get_connection also preserves participants provenance (notifications.client → db.base mapping fails closed otherwise). Stale noqa dropped; the store liveness constant kept (documented redundancy).

### Migration 6 backlog repair plan (2026-10-31)

ADR required: no new ADR
ADR path: backlog/decisions/228-sqlite-store-kernel.md
Reason: mechanical integration of the existing Guardian kernel migration and reconciliation of its removed diagnostic; no new database policy.

1. Rebase the two reviewed migration commits onto current dev, preserving BaseDB provenance, WAL retry, schema decorators and participant retirement.
2. Reproduce the diagnostic inventory failure; review the removed Guardian close diagnostic against the inherited kernel close behavior, then regenerate the inventory without changing guard thresholds.
3. Run the targeted kernel, Guardian database/finite-worker, backup owner/activation and provenance checks; independently review the complete rebased diff before publishing.

4. Preserve Guardian raw connection admission rechecks before/after PRAGMA setup; independent review found the kernel migration removed them. Pin allocation/setup races through both raw getters with failing-then-passing tests.


### Migration 6 implementation notes (2026-10-31 backlog repair)

Rebased PR #3172 onto dev d999e7f4a9. Independent review found that the kernel migration lost Guardian raw-getter admission rechecks around native setup; restored those checks in Guardian configuration, preserving the inherited exception cleanup. Four allocation/setup race cases failed before the repair and pass after it; all four also pass on pristine dev. This implements the existing ADR-228 contract, with no new schema or admission policy.

Reviewed the sole removed diagnostic (`Guardian native close failed`): inherited kernel close retains the retryable cache on failure without emitting that per-store debug message. Regenerated the inventory (Guardian 2 to 1 calls; total TASK-494 7769 to 7768), with no sink changes, and verified the complete rebuild.

Validation: 51 kernel/Guardian/finite-worker/backup-owner tests pass; two relevant private-SQLite inventory contracts pass; scoped ruff F401/F821, changed-code formatting, and diff whitespace checks pass. Independent re-review found no remaining blockers. Broader local checks retain 24 failures also reproduced on pristine dev: seven backup activation credential-processing failures, sixteen existing owner/privacy tests (Windows path behavior and stale RAG connection seams), and one raw-connection census entry for app_service_wiring. These are recorded as baseline limitations, not reported as passing. Current-head hosted CI and Qodo review remain required before arming this PR.

Updated Guardian configuration and its race tests, diagnostic inventory, database architecture reference, and the testing-evidence lesson. The parent fleet-migration task remains In Progress.

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Current-base integration after PR #3162 merged (2026-10-31): rebased onto a0dbc1174c3dbd4d2732b0212786ec9e08bfc3b5. Independent review approved the precise overlap: install a newly opened connection in the thread cache, notify the opening hook inside that block, then retain the merged deadline check and participant-gated cleanup. Borrowed live connections skip the hook and retain their ownership. Both original regression functions and all archive lessons were preserved. ADR-228 applies; no new ADR or deadline policy.

Added two deterministic native integration cases (fresh and liveness revival) proving the hook sees the installed cache, its elapsed time counts toward the deadline, and an expired new handle closes and leaves no cached handle. Eleven selected hook/deadline/revival/configuration/live-probe and Guardian raw-admission cases passed in 4.91s, with no errors. Command, JUnit, logs and before/after source hashes are retained in pr3172-resync/test-runtime/kernel-integration-1793481021535174900. Formatting of the new block is AST-identical to the passing source; existing unrelated formatter differences remain. User waived hosted CI and size gates and selected independent review in place of Qodo. Final integrated-head review remains due before oldest-first merge.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Migration 7 (Dreams) PARKED 2026-10-19, not shipped: the migration is implemented on branch chore/task-33132-dreams-migration (worktree .worktrees/task-33132f-dreams) with the decorator-census validation catching a duplicated @staticmethod pre-write (the Guardian lesson working), but the full Tests/Dreams/ battery shows 16 REAL regressions (baseline pristine-dev: 927 passed clean) concentrated in test_track_reconcile (x10), test_dreams_db_track (x5), test_boot_dispatch_admission (x1). Confirmed symptom: test_failed_ownership_classification_retires_boot_owned_native_handle — a retired handle does NOT raise 'closed database' on use, i.e. the participants native-retirement path and the kernel/hook overrides interact incorrectly somewhere in the retirement flow. NOT root-caused; branch parked unshipped rather than guessed at. Next session: diff the old probe-replacement path (old code used _close_connection, removing from _opened; kernel uses plain conn.close()) and the participants retirement call path against the new hook timing, with the track-reconcile tests as the repro.

Migration 7 UN-PARKED (2026-10-31): root cause of the 16 regressions was a STACKING bug, not a kernel bug — the branch was cut from dev before Guardian (PR #3172, carrying the _on_connection_opened hook) merged, so Dreams' hook override never fired, _opened stayed empty, and close_all closed nothing (hence the retired handle not raising 'closed database'). After rebasing onto merged dev: full Tests/Dreams/ + kernel suite 943/943 green. The parked-branch repro tests are the reason this was caught — the 'red battery vs green baseline means stop' rule converted a would-be shipped broken migration into a one-rebase fix. Deltas preserved as overrides: check_same_thread=False open (close_all's cross-thread reach), schema-lineage validation first in configure, WAL retry, the _opened registry via the kernel hook (strong refs, close_all resets thread-local), and close/_close_connection/close_all verbatim. ADR-228 exception list UNCHANGED — Dreams migrates cleanly now that the hook seam exists.


Migration 7 review round 1: the admission-recheck loss Guardian also suffered was caught here too — _configure_connection now rechecks admission BEFORE setup (old-Dreams parity) AND after (the repaired fleet standard), with the Guardian race test ported to Tests/Dreams/test_admission_race.py (4/4: pause at allocation or setup x both getters, asserting RecoveryRequired plus a closed unadmitted handle). Recorded delta: the kernel's probe-replacement uses plain conn.close(), so a handle killed behind the store's back stays in _opened until close_all's double-close no-op removes it (benign — descriptors already released; the strong-ref rationale concerns open handles — but _opened grows monotonically between close_all calls under repeated external closes). Reviewer's design note adopted for migration 8+: propose the admission recheck as a kernel feature so stores cannot keep re-losing it — two stores have now lost it to migration.
