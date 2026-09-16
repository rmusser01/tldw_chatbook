---
id: TASK-32700
title: Verify local Import execution and restart recovery
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-16 06:44'
updated_date: '2026-09-16 19:32'
labels:
  - ui
  - ingestion
  - design-system
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close the execution and restart evidence gap left by UI-only Import reviews, using the real local parse pool, writer and persisted history with disposable sources.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Real local text and Markdown imports persist the expected content and expose a usable completion/Open action at compact and wide sizes.
- [ ] #2 Repeated imports resolve existing media without duplicating content; retryable failure recovers through the mounted Retry control with lineage retained.
- [ ] #3 Clearing an unsubmitted source creates no job; local text jobs expose only supported actions; interrupted local work is honestly reconciled after restart.
- [ ] #4 Fresh app processes reopen persisted media and import history in both themes using isolated files/config/databases, with targeted checks and explicit evidence limits.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/014-library-ingest-service-authority-and-recovery.md; backlog/decisions/065-active-ingest-source-admission-and-override.md; backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md
Reason: qualify and repair existing lifecycle/presentation contracts without changing authority, schemas, providers or supported cancellation.

1. Trace the real app submission, parse pool, writer, duplicate resolution and restore paths; identify the shipped local-text action boundary.
2. Exercise disposable text/Markdown success, duplicate, staged-source clear, retryable local failure and interruption through a spawn-safe native harness with real services.
3. Reopen the same private profile in a new app process; assert exact media/history persistence, interruption recovery and retry lineage.
4. Add targeted regression coverage for any demonstrated defect, fix only within these criteria, and reconcile misleading documentation.
5. Inspect compact/wide dark/light captures in bounded passes, run relevant automated/static checks, independently review and record evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Partial checkpoint: corrected stale lifecycle and cancellation documentation in library_ingest_jobs.py; no runtime behavior changed. Added a spawn-safe, failure-specific native runner and evidence under Docs/superpowers/qa/2026-09-16-ingest-lifecycle/. Existing ADR-014/065/150/161 apply; no new ADR.

The real local parse pool cannot allocate a multiprocessing semaphore on this host (errno 28), reproduced by the real-spawn test inside and outside the sandbox. Disk has 75 GiB available; kernel semaphore maximum is 10,000. Resource ownership was not established. No unrelated processes or system settings were changed.

76 targeted tests passed; the real-spawn test remains blocked. Native run-003 verified staged Clear creates no job/media, real pool-start failure persists across a fresh app process, and mounted Retry creates a durable successor that remains honestly failed and retryable. Four compact/wide dark/light captures inspected; both app processes returned and exited 0. Ten private databases passed integrity checks, two failed attempts retain lineage, media/messages remain empty, and source/default-profile hashes are unchanged. The registry has 21 inherited Ruff diagnostics with no new findings; the native runner is lint-clean and both edited Python files pass formatting.

All AC remain unchecked and status remains In Progress. Pending after host resource recovery: real text/Markdown success and Open, duplicate no-extra-content behavior, successful Retry after permission failure, actual parse interruption and restart reconciliation followed by successful Retry. Passing controlled-pool tests and honest failure receipts do not qualify these native success journeys.

Independent review found a missing pre-import isolation guard and overstrong persistence wording. Added a fail-closed profile guard, verified six regression cases red/green and the recorded run-003 profile, and clarified best-effort ordinary persistence. Total targeted passes: 82 (76 application checks + 6 runner guard checks). Native journeys were unchanged after adding the guard; no native rerun claimed. Production AST is unchanged excluding docstrings.

Independent follow-up review approved both corrections with no remaining actionable findings. Checkpoint is ready to retain; native success criteria remain explicitly unqualified.

2026-09-16 19:31 UTC recovery check: one fresh isolated multiprocessing.get_context("spawn").Lock() probe passed outside the sandbox. The previously failing Tests/Local_Ingestion/test_ingest_parse_worker.py::test_run_parse_job_through_real_spawn_pool also passed outside the sandbox (1 passed in 0.70s), including real worker parsing. The semaphore blocker is cleared at this check; no host settings or unrelated processes were changed, and the recovery cause remains unknown. Evidence: Docs/superpowers/qa/2026-09-16-ingest-lifecycle/semaphore-recheck.json. Full native success/restart journeys remain pending; retain In Progress and all AC unchecked.
<!-- SECTION:NOTES:END -->
