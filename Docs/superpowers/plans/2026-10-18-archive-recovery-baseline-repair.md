# Archive recovery baseline and resource repair implementation plan

> Use systematic debugging and focused TDD. Independent trace/governance investigations use dispatching-parallel-agents with fresh, bounded context; root owns descriptor investigation, integration, review and the Git index. Follow existing PR3162 source/authority contracts.

**Goal:** Resolve qualification failures and descriptor retention affecting PR3162, then update the existing draft PR with exact targeted evidence.
**Spec:** User request to investigate all ten baseline failures and descriptor-growth warnings; TASK35584 acceptance criteria; Docs/superpowers/specs/2026-09-17-agent-archive-recovery-design.md and the preceding rebase qualification report.
**Architecture:** Preserve existing local archive capability, confirmation, restoration/Open, native I/O custody and prospective trace profile semantics. Repair actual resource owners and stale fixture assumptions at their shared source. No warning-threshold, sanitizer, migration or governance bypass.
**Tech stack:** Existing Python3.12, Textual8, SQLite, pytest, Ruff and CSS builder. No dependency changes.

ADR required: no
ADR path: existing backlog/decisions/166-agent-assisted-archive-recovery.md; current SQLite/kernel/native-custody and design-token ADRs remain binding.
Reason: targeted bug/fixture/governance repairs within existing ownership contracts. A proposed new boundary requires a separate ruling before implementation.

## Global constraints

- Existing isolated worktree: /Users/macbook-dev/.codex/worktrees/agent-archive-recovery/tldw_chatbook, branch codex/agent-archive-recovery-pr. Starting verified/published head ccab42365855c5a37c17a2c6bed5c4bb6f2ee96f. Original checkout and original verified feature branch remain untouched.
- Root alone stages, commits and pushes. Helpers leave edits unstaged, never use stash/cleanup/ref changes, and stay within assigned files. No full suite or extra live/model requests; the authorized model budget remains exhausted at30/30.
- All tests use shared /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python, existing Tests/conftest isolation, fresh basetemp and zero real provider traffic. Diagnostic plugins import only stdlib/pytest before conftest. Do not globally inject bootstrap markers or change FD warning thresholds.
- Preserve exact source/run/target/scope/destination/lifetime/persona/kill/privacy checks, cancellation settlement and immutable historic QA/captures. Quiescence and physical closure remain distinct from transaction success and GC.

## Tasks and interfaces

- [x] **Trace fixture failures (helper):** own Tests/Chat/test_console_trace_service.py and test_console_trace_settlement.py test bodies; reserve shared db fixture cleanup for root. Reproduce exactly failed_rollback_releases_observers_and_reports_ambiguous_outcome[False/True], oversized_response_is_replaced_by_one_bounded_labeled_artifact, queued_settlement_drops_oversized_response_and_usage_values, cold_restart_recovers_open_calls_monotonically_and_idempotently. Read connected transaction/sanitizer/recovery owners and callers. Compare working fixtures, preserve real rollback/size/recovery outcomes, correct stale lifecycle/time/limit setup only after actual RED. Request root ruling for any production edit. Return exact changes, actual RED/GREEN, checks, source hashes and resource observations; do not run full long trace modules.
- [x] **Governance failures (helper):** own Tests/DB/test_migration_version_short_circuit_guard.py and exact Evals allowlist resource; current UI/CSS files named by component-governance failures. Read design-language/component-patterns and relevant architecture first. Reproduce the two named migration guard failures and component dimension/style/sheet-ceiling failures. Use existing semantically correct tokens/classes and consolidate repeated rules where appropriate; retain rendered values/behavior. Do not weaken tests, budgets, allowlists or existing token values. Trace exact current Evals migration/version justification and repair stale inventory only. Report proposed paths before edits beyond tests; verify exact guards, generated bundle and narrow affected UI tests. Return independent source/evidence hashes and baseline classification.
- [x] **Descriptor growth (root):** inspect prior storage/root/UI raw command/log evidence and native close/quiescence ownership. Instrument per-test post-teardown descriptor counts and open-file targets, with optional collection as a diagnostic comparison only. Reproduce representative storage, captured follow-up and mounted recovery paths; distinguish SQLite inode reuse, fixture retention, delayed GC and genuinely reachable native owners. Use concurrent held workers when testing thread-keyed maps. Record exact handles/owners and test the smallest fault before any production repair. Fix proven owner/fixture cleanup with deterministic real SQLite regression; never close borrowed active handles or replay admitted effects. A second repeated round must plateau, and final owning close must retire native resources.
- [x] **Integration and verification (root):** read all actual diffs; resolve shared fixture ownership sequentially. Run affected targeted cohorts with ordinary GC settings after fixes, retaining actual warnings rather than suppressing them. Validate source syntax, duplicate definitions, changed-code lint/format, CSS build/token checks and archive/capture controls touched by fixes. Re-run passed tests only for changed behavior or unresolved evidence.
- [x] **Independent review and publish (root):** fresh review of combined fixes and native retirement evidence. Update TASK35584 and new QA report with ten-failure accounting, descriptor attribution, test commands/counts/limits and existing ADR links. Push only the PR branch with an exact expected-head lease and confirm published source, dev ancestry, no conflicts, clean worktree and preserved draft state. No merge or auto-merge.

## Evidence

Raw evidence: /private/tmp/archive-recovery-pr3162-baseline-20261018/{trace,governance,resources}/. Durable final report: Docs/superpowers/qa/console/2026-10-18-archive-recovery-baseline-repair.md. Previous reports/artifacts remain historical evidence; new fixes append qualification rather than rewriting earlier results.

## Evidence-driven scope refinements

- Registry factory/media owners, raw migration inspection connections and replay-controller worker handles required test-owned cleanup. WorkspaceDB initialization failure also required production cleanup; existing ADR228/native ownership applies.
- Independent review reproduced a search WAL checkpoint exceeding the budget after SQL completed. The existing operation-owned guard now accepts an optional cleanup deadline; default close and live borrowed handles retain their behavior. Real maintenance retirement is recognized without masking the original operation failure.
- Mounted factory cleanup requires the existing dispose-and-detach helper. Additional notification controls must await actual rendered receipt acknowledgement; an unmounted factory app's scheduler must not stand in for the mounted harness message pump. No acknowledgement, notification or navigation policy may be bypassed.
- Raw evidence and actual intermediate failures remain at /private/tmp/archive-recovery-pr3162-baseline-20261018. Current qualification and remaining limits are in Docs/superpowers/qa/console/2026-10-18-archive-recovery-baseline-repair.md.
