---
id: TASK-414
title: Skills trust - show content changes in Review changes before Approve
status: Done
assignee:
  - '@claude'
created_date: '2026-07-21 15:18'
updated_date: '2026-07-21 15:50'
labels:
  - skills
  - ux
  - trust
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
P1 from the 2026-07-21 Skills UX/NNG review (verified live). The trust flow's Review changes button renders only a comma-joined filename list - which the trust state line already shows - so Approve is effectively blind sign-off on unseen executable skill content. capture_review already returns current_files with full text; the UI just never renders it. Also approve-time snapshot_mismatch surfaces only as a generic warning toast. NNG heuristic 6 (recognition over recall) plus security-UX honesty.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After Review changes the user can see the content of changed files (at minimum a per-file preview; a before/after diff against the trusted baseline preferred) before Approve is enabled,Approve failure due to snapshot mismatch produces a specific actionable message instead of the generic trust-action warning,Review presentation handles multi-file and deleted-file cases without breaking layout,Covered by tests
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Content preview: new pure helper skill_trust_review_preview (library_skills_canvas.py, re-exported via Widgets/Library) renders one labelled block per changed file from capture_review's current_files - '(deleted - no longer on disk)' markers for missing files, 4000-char per-file cap with an explicit truncation note. Rendered in an always-present #library-skill-trust-review-content Static below the changed-files line (markup=False), patched in place by _render_library_skill_trust_panel under the same no-recompose contract. NOTE: an actual before/after diff needs baseline TEXT, but the trust store keeps only fingerprints - a service-level baseline-content store would be its own task; the AC's minimum (per-file preview) is met. Approve failures: _call_library_skill_trust_service gained failure_copy (exception message -> specific toast); approve passes snapshot_mismatch -> LIBRARY_SKILL_TRUST_MISMATCH_COPY, and on ANY trust_reviewed_snapshot failure the UI drops the captured review + re-renders + re-fetches status (the service pops the review on every raise path, so keeping it would leave Approve armed against a dead review id). 8 new tests (4 pure helper, canvas render, in-place patch, failure_copy override, stale-review discard) all watched fail first; skills canvas 53 passed, Tests/Skills 129 passed.
<!-- SECTION:NOTES:END -->
