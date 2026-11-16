---
id: TASK-41216
title: Account for plugin restore adoption in the modal layout inventory
status: Done
assignee:
  - '@codex'
created_date: '2026-11-15 22:36'
updated_date: '2026-11-16 00:05'
labels:
  - ci
  - plugins
  - tests
dependencies: []
references:
  - >-
    https://github.com/rmusser01/tldw_chatbook/actions/runs/43750308318/job/132962026467
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The current modal layout inventory rejects PluginRestoreAdoptionModal even though it uses the shared plugin review shell. Record its existing viewport treatment alongside the other plugin modals so the inventory remains complete and stale exceptions remain detectable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every ModalScreen class including PluginRestoreAdoptionModal is classified and the explicit exception names its actual shared-shell behavior.
- [x] #2 Existing wide-tier layout and stale-exception checks pass without changing modal styles or tokens.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: register an existing shared-shell modal under its established viewport exception; no layout or design-token changes.
1. Confirm the adoption modal uses the existing plugin-review-shell and shares the documented viewport treatment.
2. Add the modal class to the existing explicit inventory exception with its precise rationale.
3. Run modal inventory and relevant layout/exception contracts, independently review and record evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added PluginRestoreAdoptionModal to the explicit wide-tier inventory exception alongside the other four plugin modals sharing plugin-review-shell. The existing shell uses 90 percent viewport width without a cap; the generic 85 percent tier would shrink it. No widget, stylesheet or token changes.

Validation: the full modal-wide-tier module passes, including class coverage, stale exception checks and geometry/live-resize contracts. Scoped Ruff and independent review pass. Only Tests/UI/modal_wide_tier_registry.py changes.

ADR required: no; this records an existing layout treatment.
<!-- SECTION:NOTES:END -->
