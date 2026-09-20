---
id: TASK-32872
title: Complete Library artifact browsing and navigation cutover
status: Done
assignee:
  - '@codex'
created_date: '2026-09-19 22:05'
updated_date: '2026-09-20 07:18'
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
- [x] #1 All artifacts reaches the complete admitted inventory in bounded pages with deterministic ordering, truthful totals, direct target location, and independent recovery for healthy views.
- [x] #2 Ctrl+6, configured artifacts defaults, command palette routes, and exact Chatbook handoffs resolve into Library without changing other shortcuts or manager identity.
- [x] #3 Artifact-only profiles, local scope, sharing lifetime, generation fences, and restoration remain correct across navigation and source failures.
- [x] #4 Capability parity passes targeted tests and live production-CSS verification before removal of the permanent Artifacts destination.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/172-library-artifacts-browse-and-navigation.md. Reason: Completes accepted Library navigation structure without changing artifact ownership. 1. Expose composite All artifacts using the bounded catalog. 2. Preserve exact target claims, explicit mode precedence and Ctrl+6 while resolving legacy Artifacts entry to Library and retaining the actual Chatbook manager route. 3. Remove the top-level Artifacts destination after parity verification. 4. Verify targeted routing/handoff, mixed catalog, production CSS and native behavior; update the user guide. Prepare cutover while stage2 finishes, apply after parity is checked.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented mixed All artifacts and completed the parity-gated navigation cutover. Exact targets locate beyond page one; failures retry without consuming claims, missing targets clear stale detail, and explicit Library routes retain precedence. Ctrl+6, palette, legacy aliases and configured defaults reach Library without renumbering other shortcuts; Chatbooks remains the actual manager. 21 navigation and 9 cutover checks pass. Final native verification covers four geometries, artifact-only onboarding, sharing across Notes, manager entry and keyboard routes. ADR: backlog/decisions/172-library-artifacts-browse-and-navigation.md. Evidence and explicit baseline limits: Docs/superpowers/plans/2026-09-19-library-artifacts-verification.md. User guide updated. No full suite requested; changed/new focused tests pass, added-line Ruff diagnostics are zero, generated CSS checks pass, and native private-profile TldwCli was verified. Existing recovery-initialization and repository-wide screen-size/workflows-style failures remain documented; their limits were not raised.
<!-- SECTION:NOTES:END -->
