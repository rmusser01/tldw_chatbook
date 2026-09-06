---
id: TASK-31812
title: Close fixture-owned rewind database and controller resources
status: Done
assignee:
  - '@codex'
created_date: '2026-09-06 05:02'
updated_date: '2026-09-06 05:34'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Eliminate the reproduced retained SQLite handles in rewind tests while preserving real summary, dispatch and restart behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Rewind fixture controllers finish supported shutdown before exact owned databases are quiesced and registries report zero connections.
- [x] #2 Complete rewind and parent-persistence selections pass without the reproduced 209-descriptor growth or retained fixture SQLite descriptors.
- [x] #3 Existing behavioral assertions and resource thresholds remain unchanged, with lint, negative controls and independent review.
- [x] #4 An individual cleanup failure does not skip later owned resources; all controller shutdown attempts precede database cleanup and collected errors remain visible.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A. Reason: test-only lifecycle correction reuses existing controller shutdown and database quiescence. 1. Preserve the reproduced 209-FD baseline and native per-test attribution. 2. Add one explicitly imported fixture in Tests/console_resource_fixtures.py for the two rewind files; retain canonical constructors, track controllers created during each test and only ChaChaNotes files under its tmp_path. Teardown shuts down all controllers before quiescing any exact owned DB, asserts zero registered connections, and releases references before the existing cleanup fixture. First prove the zero-connection assertion fails with shutdown alone. 3. Give the agent-only summary ordering case an explicitly owned real workspace registry and AgentRuns DB using the existing file-roots registry factory seam; close both after assertions without touching a foreign cached registry. 4. Run complete summary/rewind/parent files with native FD attribution, then a negative control bypassing quiescence; preserve all behavioral assertions and thresholds. 5. Lint, changed-region format, independent review, evidence and scoped commit. New shared test fixture avoids duplicating the same lifecycle cleanup in both files; no shared conftest or production edits.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented explicitly imported real-owner cleanup in Tests/console_resource_fixtures.py and the two rewind files, plus seven lifecycle fault controls. Controllers are attempted before exact tmp_path database quiescence; zero connections asserted before existing cleanup, no extra GC or foreign registry draining. Agent-only summary owns its real workspace registry and runs DB. Shutdown-only zero-registry RED produced2 teardown errors; ordinary-fault RED1pass4fail and cancellation RED5pass2fail all pass after BaseExceptionGroup containment. Independent review findings resolved and root-reviewed. Final affected rewind/recovery/boundary/control selection247passed147.85s, no native retained SQLite descriptors or FD growth warning; three dependency warnings remain. Full scoped lint/changed-region formatting/diff checks pass. Evidence and lesson recorded in checkpoint and 2026-09-06 rewind plan. ADR required:no; existing lifecycle interfaces unchanged.
<!-- SECTION:NOTES:END -->

Review refinement: add a RED-first failure-path regression for shutdown, quiescence and connection-count errors; attempt all independent owner cleanup in lifecycle order and report collected errors afterward without forcing active handles closed.

Cancellation refinement: cover a shutdown or database cleanup raising CancelledError, which derives from BaseException. Preserve cancellation in the final grouped error while still attempting later independent owners; repeat complete resource verification after this refinement.
