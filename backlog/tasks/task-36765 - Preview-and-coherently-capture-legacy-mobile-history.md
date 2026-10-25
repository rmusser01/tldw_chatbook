---
id: TASK-36765
title: Preview and coherently capture legacy mobile history
status: To Do
created_date: 2026-10-25 17:39
dependencies:
- TASK-36729
labels:
- console
- web-server
- mobile
- adr-244
priority: high
documentation:
- Docs/superpowers/plans/2026-10-25-shared-console-mobile-supervision-implementation.md
- Docs/superpowers/specs/2026-10-25-shared-console-mobile-supervision-design.md
- backlog/decisions/244-shared-console-ownership-and-mobile-supervision.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make legacy mobile data and its destination reviewable and safely capturable before any shared-storage import changes user history.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Read-only preview identifies the physical legacy source, admitted canonical destination candidates, conflicting identities and ambiguous cross-configuration attribution without selecting a destination silently.
- [ ] #2 The three legacy stores and required referenced files participate in normal coherent recovery capture, WAL consistency and destination rollback coverage without constructor reconciliation.
- [ ] #3 Unsupported schemas, unsafe paths or aliases, unknown populated data and missing dependencies block the operation with actionable issues and original bytes preserved.
- [ ] #4 An absent or empty legacy source is reported without creating legacy databases, and the existing recovery CLI exposes the preview without mutating source or destination data.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
