---
id: TASK-32872
title: Complete Library artifact browsing and navigation cutover
status: To Do
assignee: []
created_date: '2026-09-19 22:05'
labels:
  - library
  - artifacts
dependencies:
  - TASK-32871
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Combine registered Chatbooks and all report copies into Library All artifacts, then make Library the permanent browse destination after capability parity. Preserve Ctrl+6, legacy artifacts routes, exact saved-item handoffs, and the existing ZIP manager route. Governed by ADR-172 and stage 3 of Docs/superpowers/plans/2026-09-19-library-artifacts.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All artifacts reaches the complete admitted inventory in bounded pages with deterministic ordering, truthful totals, direct target location, and independent recovery for healthy views.
- [ ] #2 Ctrl+6, configured artifacts defaults, command palette routes, and exact Chatbook handoffs resolve into Library without changing other shortcuts or manager identity.
- [ ] #3 Artifact-only profiles, local scope, sharing lifetime, generation fences, and restoration remain correct across navigation and source failures.
- [ ] #4 Capability parity passes targeted tests and live production-CSS verification before removal of the permanent Artifacts destination.
<!-- AC:END -->
