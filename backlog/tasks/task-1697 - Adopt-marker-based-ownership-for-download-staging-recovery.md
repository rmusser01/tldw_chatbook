---
id: TASK-1697
title: Adopt marker-based ownership for download-staging recovery
status: To Do
assignee: []
created_date: '2026-08-01 07:02'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reconciliation item 4: replace shape-sniffing of the fetch-state sidecar in reconcile()'s GC with the parallel branch's marker-based ownership proof (schema, operation kind, artifact reference, descriptor fingerprint), so recovery refuses and reports when containment or ownership cannot be proven rather than guessing. Do after the finalization-seam port, which makes marked stages natural.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Staging entries are classified by an owned marker, not by sidecar shape
- [ ] #2 Recovery refuses to delete when ownership or containment cannot be proven
<!-- AC:END -->
