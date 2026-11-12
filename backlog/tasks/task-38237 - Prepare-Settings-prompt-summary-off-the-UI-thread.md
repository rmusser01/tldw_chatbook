---
id: TASK-38237
title: Prepare Settings prompt summary off the UI thread
status: Done
assignee:
  - '@codex'
created_date: '2026-10-19 03:44'
updated_date: '2026-10-19 04:09'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep Settings navigation responsive while its customized-prompt summary performs checked configuration reads.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Settings composition and refresh never perform the customized-prompt storage read on the UI thread.
- [x] #2 Count publication rejects stale config, edited, cancelled, and unmounted results; failures show unavailable rather than Defaults.
- [x] #3 Existing prompt storage admission remains unchanged and targeted tests plus native Windows profiling verify the change.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A (preserve ADR-126). Reason: bounded display read offloading using existing worker patterns; no storage boundary change. 1. Add deterministic blocked-read and stale-result tests. 2. Move checked count acquisition to an exclusive worker with config identity and request revision guards. 3. Refresh displays in place on completion and prompt edits. 4. Run focused tests/lint and repeat the retained native Windows profile.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Prepared the original checked customized-prompt read in an exclusive worker after first refresh. Rendering now uses an honest loading/unavailable snapshot; config identity, revision, cancellation and attachment reject stale publication, and edits refresh rows in place. Controlled Windows repeat-visit median improves 1141ms to 833ms (27%); first-visit improvement is not established. Actual unmocked worker read, held-read and stale-result controls pass, with original storage admission unchanged. Shared final regression set: 51 passes; final focused check: 22 passes including one additional native worker case. Eight older failures reproduce against unchanged HEAD. New-code lint/format passes; per-file existing lint debt unchanged. Independent review approved. ADR required: no; preserve ADR-126. Details and raw evidence: Docs/Development/evidence/windows-performance-2026-10-18/FIX_RESULTS.md. Architecture documentation updated.
<!-- SECTION:NOTES:END -->


## Renumbering provenance — 2026-10-31

Renumbered the incoming PR3166 Settings prompt-summary task from TASK-35587 to TASK-38237 because TASK-35587 already identifies the session-close pending-journeys task on current dev (latest recorded commit 3f40eaecb0). Preserve that landed identity, this task's original content/status, and the historical raw performance receipts. The root integration census checked 71 worktrees and 290304 task paths, found maximum38207, allocated the buffered38237, and found no history claim for it. Recheck current refs/worktrees before publication.
