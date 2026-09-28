---
id: TASK-33125
title: 'Docs: record UI verification in task notes, not User Guide stamps'
status: Done
assignee: []
created_date: '2026-09-27 23:46'
updated_date: '2026-09-28 02:09'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Spec 2026-09-27-ci-conflicts-and-waste-design.md part B: parallel PRs appending Verified-against paragraphs to User Guide pages caused most User Guide sync conflicts in the 2026-09-27 CI spec's replay of real sync merges.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 CLAUDE.md UI-changes rule records verification in task Implementation Notes, not User Guide pages
- [x] #2 lessons-live-verification.md stamping lesson carries a dated pointer to the new rule
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`CLAUDE.md`'s UI-changes rule now records verification in the task's Implementation Notes -- branch and date -- instead of a "Verified against" paragraph on the User Guide page. `backlog/docs/lessons-live-verification.md` carries a dated 2026-09-27 pointer (directly under the heading of its "A 'Verified against' stamp verifies what it names" entry) noting where the claim is now written. Merged as #2867.
<!-- SECTION:NOTES:END -->
