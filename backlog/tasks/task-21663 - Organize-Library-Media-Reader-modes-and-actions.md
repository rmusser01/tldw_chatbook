---
id: TASK-21663
title: Organize Library Media Reader modes and actions
status: Done
assignee: []
created_date: '2026-08-24 14:48'
updated_date: '2026-08-24 15:16'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the permanent Library Media Reader content-first while keeping mode, provenance, supported actions, and one-off server detail capabilities truthful.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reader defaults to Read and preserves the selected mode across local item changes.
- [x] #2 Exactly one of Read, Analysis, Highlights, or Info is composed and each mode has truthful empty states.
- [x] #3 The Reader toolbar and inline More region keep every supported action reachable at wide and 80x24 sizes.
- [x] #4 Info names backend, canonical ID, source, stored representation, and the exact Console handoff representation.
- [x] #5 One-off server detail is explicit, read-only, and never claims selection in the local Items list.
- [x] #6 Focused tests, required inverses, static checks, and self-review pass without scoped regressions.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add RED state and compositor tests for mode persistence, one-body composition, empty states, provenance, toolbar reachability, Console payload truth, and external-detail capabilities.\n2. Extend viewer display facts and recompose one active Reader mode with the smallest inline More region while preserving existing content, analysis, highlights, and metadata actions.\n3. Replace the remote-detail boolean with backend-qualified session transitions and keep the local Items snapshot unclaimed by external detail.\n4. Run the focused Task 4 suite, required inverses, static checks, and scoped self-review.\n\nADR required: yes\nADR path: backlog/decisions/084-library-media-reader-ia.md\nReason: This task implements ADR-084's long-lived Reader mode, provenance, capability, and external-detail boundaries; no new architectural choice is introduced.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Reworked the permanent Reader into persisted Read, Analysis, Highlights, and Info modes with one active body, truthful empty states, a compact primary toolbar, and inline More actions.
- Replaced the remote-detail boolean with backend-qualified reader-session state. External server detail is read-only, does not claim a local row, and normalizes digit-only server IDs at the service boundary because the server client requires numeric IDs.
- Added regressions for external detail capabilities, stale external A→B responses, exact stored Markdown text represented in Info and sent to Console, and Edit metadata routing from Read into the editable Info form. The external request fence now requires both its dispatch generation and canonical loaded ID at every commit boundary. Focused suite: 102 passed (2 environment warnings). Both required inverses failed as expected (external Read Later exposed; four mode bodies mounted) and passed after restoration. Ruff, compileall, and `git diff --check` passed.
- ADR-084 governs the Reader IA, provenance, capability, and external-detail boundary; no new ADR was required.
<!-- SECTION:NOTES:END -->
