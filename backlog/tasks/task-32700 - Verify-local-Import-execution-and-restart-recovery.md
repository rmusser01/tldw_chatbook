---
id: TASK-32700
title: Verify local Import execution and restart recovery
status: Done
assignee:
  - '@codex'
created_date: '2026-09-16 06:44'
updated_date: '2026-09-16 20:13'
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
- [x] #1 Real local text and Markdown imports persist the expected content and expose a usable completion/Open action at compact and wide sizes.
- [x] #2 Repeated imports resolve existing media without duplicating content; retryable failure recovers through the mounted Retry control with lineage retained.
- [x] #3 Clearing an unsubmitted source creates no job; local text jobs expose only supported actions; interrupted local work is honestly reconciled after restart.
- [x] #4 Fresh app processes reopen persisted media and import history in both themes using isolated files/config/databases, with targeted checks and explicit evidence limits.
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
Completed the real local Import lifecycle qualification after the host semaphore gate recovered. Added native_success_check.py and the resume/ evidence bundle under Docs/superpowers/qa/2026-09-16-ingest-lifecycle/. Four fresh native processes verified text/Markdown import and Open, duplicate resolution without extra content, real permission-failure Retry, Quit during real chunking, restart reconciliation and successful Retry, followed by persisted reopening in both themes/sizes. No application runtime code changed in this resumed work; the earlier checkpoint corrected lifecycle documentation only.

All four app processes returned normally and exited 0; independent checks confirmed app/worker absence. Final read-only checks found ten healthy private databases, four exact source-matching media and document versions, seven durable attempts with both retry chains, unchanged source/default-profile hashes, and zero conversations/messages. Sixteen final captures were inspected in one batch. The private log contains three caught ChatScreen _sidebar_state_save_timer startup errors outside the Import path; scope limits and these diagnostics are preserved in the evidence.

Verification: 83 targeted tests passed in 80.95s, including the real spawn-pool parser test. The new runner passes Ruff lint/format. Evidence JSON, local links, SVG XML/hashes and whitespace checks pass. Independent review has no remaining actionable harness findings. No full suite, external provider exercise, host resource cleanup, or unrelated process termination occurred. Historical semaphore failure and recovery receipts remain in the parent QA record.

ADR required: no. Existing ADR-014, ADR-065, ADR-150 and ADR-161 govern this verification; no authority, schema, provider, dependency or UX boundary changed. Evidence and reproduction: Docs/superpowers/qa/2026-09-16-ingest-lifecycle/resume/README.md.
<!-- SECTION:NOTES:END -->
