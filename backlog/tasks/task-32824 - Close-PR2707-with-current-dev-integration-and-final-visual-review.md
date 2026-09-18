---
id: TASK-32824
title: Close PR2707 with current dev integration and final visual review
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-18 21:53'
updated_date: '2026-09-18 22:13'
labels:
  - ui
  - design-system
  - integration
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close the existing PR at its approved scope boundary, verify current dev integration and accumulated findings, and present exact visual/conflict evidence for merge approval while preserving remaining reviews for follow-up PRs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Unfinished inspector work is safely preserved outside PR2707 and remaining destination reviews are explicitly deferred without becoming closeout blockers.
- [ ] #2 Current dev is integrated with an exact conflict-resolution record; targeted affected checks and current-head CI are reviewed with any remaining failures identified.
- [x] #3 A concrete final visual and conflict review identifies the proposed merge result; PR remains unmerged until this review receives user approval.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A. Reason: integrate current dev and close the already-authorized PR without new feature or architectural decisions. 1. Preserve unfinished TASK32823 independently. 2. Inspect remote reviews/CI and merge current dev, retaining both testing-lesson entries at the sole text conflict and checking overlapping automatic merges. 3. Run targeted integration/preflight checks and representative native visual replays on the integrated source. 4. Update PR, closeout report, visual/conflict approval packet and explicit follow-up handoff; wait for user merge approval. Fresh 232-ref/33-worktree scan reserved32824; CLI initially minted32823 because the follow-up is on another ref, so filename/frontmatter were corrected before implementation. Targeted integration exposed two child-only composer test injections overwritten by parent refresh. Correct their setup/observation boundaries and verify the affected file; production behavior is unchanged. Inspect the recurring Windows GGUF focus/SelectOverlay CI failure and retain its exact qualification limits.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Preserved TASK32823 on separate pushed branch 135f226888. Integrated dev53b56384cc in863e56b5e9, retaining both text-conflict additions and reviewing four automatic overlaps.53 distinct affected tests have passing final results; corrected two parent-refresh test races without production edits. Seven preflight guards and scoped lint pass. Sixteen integrated-source native captures inspected with clean exit,11 healthy private databases, unchanged defaults and19 matching source hashes. Final visual/conflict packet is ready; current pushed-head CI and owner visual approval remain pending. ADR required:no (routine integration under existing ADR150/161/168/170).
<!-- SECTION:NOTES:END -->
