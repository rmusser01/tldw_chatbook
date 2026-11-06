---
id: TASK-34994
title: Delete or revive the unwired server panels (OutputsPanel SharingPanel)
status: Done
created_date: 2026-10-13 02:04
labels:
- simplification
- dead-code
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deletion-wave review finding: OutputsPanel and SharingPanel have zero production importers or instantiations (only the CSS census references OutputsPanel). Either wire them into a reachable surface (they are the only UI for server-mode Outputs/Sharing) or delete them with their CSS. Coordinate with the consumerless-Interop triage (task-34900) - if those domains are pruned the panels go with them.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Per-panel disposition decided (wire or delete) with rationale,If deleted: panel modules plus their CSS rules and census entries removed and bundle rebuilt,If wired: a reachable route mounts the panel and a mount test exists
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

Deleted by TASK-38605 (the 2026-11-02 audit's dead-code purge, PR #3361). `UI/Outputs_Panel.py` and `UI/Sharing_Panel.py` had no production route, so both were deleted along with:
- their seam tests;
- their class-level CSS allowlist rows;
- their markup-interpolation and await-DOM census rows.

The CSS bundle reproduces (preflight).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
