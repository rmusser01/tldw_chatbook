---
id: TASK-32779
title: Recover Tool Profile exports from destination conflicts
status: Done
assignee:
  - '@codex'
created_date: '2026-09-18 09:24'
updated_date: '2026-09-18 09:44'
labels:
  - settings
  - ui
  - tool-packs
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users correct Tool Pack export destinations without losing the reviewed profile or receiving a misleading platform error, while preserving no-overwrite publication.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Existing destination bytes remain unchanged; the picker explains the conflict, retains the location and filename, and permits correction without repeating profile review.
- [x] #2 Invalid archive filenames and destinations changed before publication recover through a new filename; stale profile, cancellation, uncertain publication and unsupported-platform outcomes retain their existing authority boundaries.
- [x] #3 Targeted workflow/publication tests and native compact/wide dark/light export journeys verify recovery, successful archive contents and cancellation.
- [x] #4 The export review renders real ToolProfilePayload data, displays accurate policy counts, and reaches filename selection without crashing.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce existing and invalid destinations through the canonical Settings export worker, retaining real archive publication in tests.
2. Keep the reviewed export snapshot while reopening the existing save picker with local recovery copy and the selected location/name; continue to use captured-destination validation and the existing publisher.
3. Cover corrected filename, cancellation, collision before publication, stale policy, uncertain and unsupported outcomes; run publication and relevant UI regressions.
4. Verify native export review, conflict recovery and saved archive in dark/light compact/wide cells, review independently, and save to draft PR 2707.
ADR required: no
ADR path: backlog/decisions/107-portable-tool-use-packs.md
Reason: This repairs presentation and recovery under the existing review-first, no-overwrite publication contract; no storage or authority policy changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Settings export now retains the reviewed immutable snapshot while reopening the existing save picker for conflicting, invalid or changed destinations. Previous location/name and local recovery text are preserved; incumbent bytes and publisher authority remain unchanged. Real capture also exposed a review crash: ToolProfilePayload.tools replaces the incorrect .rules access, and the existing policy-count fixture now constructs the real type.

Validation: 86 distinct targeted cases pass (9 export worker/recovery, 30 existing Tool Profiles, 47 service/publication). All 137 pre-existing assertions are unchanged. Four native dark/light by 80x24/170x48 journeys verify real archive manifests, correction/cancel, unchanged existing files and permission-store bytes. Twelve SVGs were rendered and inspected. Final native run exited normally; all 11 private DBs are healthy, default-profile fingerprints unchanged, instance lock released and app PID absent. No full suite or provider requests.

Scoped Ruff adds no diagnostics (existing baselines 114/9/3); new tests/runner and changed-method/small-file formatting pass. Backlog, diagnostic and diff guards pass. Independent review found no introduced blocker and verified missing-parent correction. Evidence: Docs/superpowers/qa/2026-09-18-tool-profile-export/README.md. Remaining delayed-import and refresh-focus defects are recorded in Docs/superpowers/reports/2026-09-18-tool-profiles-review.md. Added the real-payload incident to testing lessons.

ADR required: no; existing backlog/decisions/107-portable-tool-use-packs.md applies. Capture checks initial policy authority; publication continues to write the reviewed snapshot without recapturing policy. No storage, permission, dependency or licence changes. Draft PR 2707 remains unmerged pending its separate approval.
<!-- SECTION:NOTES:END -->
