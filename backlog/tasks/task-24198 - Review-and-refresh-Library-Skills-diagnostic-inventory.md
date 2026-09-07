---
id: TASK-24198
title: Review and refresh Library Skills diagnostic inventory
status: Done
assignee:
  - '@codex'
created_date: '2026-08-29 12:37'
updated_date: '2026-08-29 12:40'
labels:
  - security
  - library
  - tests
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the production-diagnostic security ratchet after the Library Skills browse controller added one constant, non-sensitive warning diagnostic.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Library Skills warning is reviewed for sensitive interpolation and the canonical production diagnostic inventory matches the shipped diagnostic owners and call counts
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A. Reason: this is a review and synchronization of an existing generated security inventory, with no diagnostic policy or sink-topology change. Inspect the exact added statement against the inventory baseline, confirm it logs only constant text and the exception class name, regenerate the canonical inventory, run the exact ratchet test plus script check and static diff validation, then record evidence and close.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reviewed the only inventory delta introduced by the Library Skills browse controller. The warning logs fixed text plus type(exc).__name__ only; it does not interpolate exception text, user content, secrets, paths, or URLs, and the persistent-sink topology is unchanged. Regenerated the canonical inventory, adding one TASK-494 owner row and changing only owner_files 540->541 and task_494_calls 7399->7400. ADR required: no; ADR path: N/A; no diagnostic policy or sink boundary changed. Verification: exact architecture ratchet 1 passed in 32.09s; standalone inventory checker exited 0; JSON validation passed; git diff --check passed.
<!-- SECTION:NOTES:END -->
