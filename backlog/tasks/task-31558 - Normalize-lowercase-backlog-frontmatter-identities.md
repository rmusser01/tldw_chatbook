---
id: TASK-31558
title: Normalize lowercase backlog frontmatter identities
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 01:35'
updated_date: '2026-09-05 01:37'
labels:
  - backlog
  - tests
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the repository-wide backlog identity invariant by correcting the recent task files whose frontmatter uses a lowercase `task-` prefix.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every lowercase `task-*` frontmatter ID is normalized to the canonical uppercase `TASK-*` form without changing its numeric identity.
- [x] #2 The repository-wide backlog task identity contract passes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the identity-contract failure and inventory every lowercase frontmatter ID, confirming each filename and body refer to the same numeric task.
2. Normalize only the frontmatter prefix case across the bounded invalid set.
3. Run the repository-wide backlog identity contract and diff checks.

ADR required: no
ADR path: N/A
Reason: this is metadata hygiene with no application or architecture change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Normalized the frontmatter prefix in the complete 13-file lowercase inventory (TASK-22060 through the later TASK-25890-era files) without changing numeric identities, filenames, task bodies, or statuses.
- Evidence: the repository-wide uniqueness/validity contract passes after advancing through the scanner's first-failure behavior; no lowercase `id: task-` entries remain.
- ADR required: no; metadata-only hygiene.
<!-- SECTION:NOTES:END -->
