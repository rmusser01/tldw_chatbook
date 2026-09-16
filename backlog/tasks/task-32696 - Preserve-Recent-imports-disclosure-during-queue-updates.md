---
id: TASK-32696
title: Preserve Recent imports disclosure during queue updates
status: Done
assignee:
  - '@codex'
created_date: '2026-09-16 05:15'
updated_date: '2026-09-16 05:30'
labels:
  - library
  - ui
  - design-system
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reading Recent imports should survive background queue changes without collapsing the history, moving keyboard focus into the next draft, or losing the draft. Qualify grouped recovery and Clear finished through the existing UI seams.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Recent imports preserves its open or closed state and focused title through queue rebuilds; newer user focus wins.
- [x] #2 Grouped Show/Hide, Retry all, Dismiss all, and Clear finished remain keyboard reachable with painted labels at wide and compact sizes while preserving the next draft and recent ledger.
- [x] #3 Targeted tests and isolated native evidence document the repair, scope, and inherited checks.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/014-library-ingest-service-authority-and-recovery.md; backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md
Reason: repair existing disclosure and focus contracts; preserve queue ownership, execution seams, consent, and tokens.
1. Reproduce Recent disclosure/focus loss in the production CSS harness and pin open, closed, and newer-focus cases.
2. Preserve panel-owned Recent disclosure state and restore its title through the existing queue callback.
3. Exercise grouped recovery, Dismiss/Clear ledger, and unsaved draft continuity at wide/compact sizes.
4. Run targeted neighbors, static comparison, an isolated native journey, and bounded code review; record evidence and close the task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Recent imports now snapshots its mounted disclosure state before a queue rebuild and restores its ID-less title through the existing guarded focus callback. Newer attached focus wins; the next draft and ledger retain their owners. The production change is confined to library_ingest_canvas.py; the user guide documents continuity.

94 targeted tests pass, including eight new production-CSS journeys and grouped Retry/Dismiss/Clear checks. Final isolated native run-003 passes at 170x48 dark and 80x24 light; six captures inspected, ten private DBs healthy, zero media/messages/jobs, source/default-profile hashes unchanged, normal exit 0. Retry used an explicit registry-only UI seam, with no worker or real import. Zero new Ruff diagnostics; six inherited canvas diagnostics retained; changed ranges/new files formatted. Independent review found no actionable issues.

ADR required: no; existing ADR-014, ADR-150 and ADR-161 apply (paths in plan). No tokens or CSS changed. QA: Docs/superpowers/qa/2026-09-16-recent-imports/README.md. A native capture wait initially queried the wrong surface; the active-screen Toast lesson was added to lessons-live-verification.md. Full suite, live-resize ownership, actual provider recovery and dev integration remain outside this slice.
<!-- SECTION:NOTES:END -->
