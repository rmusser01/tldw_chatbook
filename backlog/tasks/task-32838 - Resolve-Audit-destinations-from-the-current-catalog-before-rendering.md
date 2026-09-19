---
id: TASK-32838
title: Resolve Audit destinations from the current catalog before rendering
status: Done
assignee:
  - '@codex'
created_date: '2026-09-19 05:54'
updated_date: '2026-09-19 06:05'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep Audit tool and permission drilldowns consistent with catalog replacements that finish while navigation is pending.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both Audit destinations render the current same-ID tool definition after a pending catalog refresh, including updated description and availability.
- [x] #2 A vanished catalog target clears detail and warns; profile changes continue to reject stale navigation.
- [x] #3 Navigation rendering and the existing catalog publication lock preserve coherent row and permission detail, verified by targeted races and private native journeys.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce same-ID catalog replacement/removal while each Audit destination waits, using real catalog/table publication and the existing profile context.
2. Re-resolve identity under the existing catalog sync lock before selecting/rendering so pending publication cannot mix an old definition with current rows and permissions. Preserve profile rejection and missing-target warnings.
3. Verify focused new/adjacent tests, guards, static analysis, independent review and private native dark/light journeys. Save an independent draft PR against dev.
ADR required: no
ADR path: backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md
Reason: routine repair within existing workbench synchronization and inspector contracts; no new interface, authority or persistence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Both Audit drilldowns now resolve the current identity under the existing catalog publication lock and retain that lock through row selection and inspector rendering. Pending refreshes cannot combine a captured definition with new rows or permissions; removed identities clear detail and warn, and profile-change checks remain. Added 12 regressions covering same-ID description/schema replacement, disconnection, removal, incomplete publication and real-app action routing. All 62 targeted cases, seven preflight guards, static-diagnostic baseline checks and changed-range formatting pass. Independent read-only review found no blocker. Eight inspected native dark/light captures at 120x40 and 170x48 use controlled collector replacement in a real private app; clean exit, healthy databases and unchanged default profile, with no tool execution or permission mutation. Evidence: Docs/superpowers/qa/2026-09-18-mcp-audit-catalog-freshness/README.md. ADR-150/161 apply; no new ADR needed. Updated both review ledgers. PR2724 row/control ownership and ongoing refresh of already-open inspectors remain separate; next review is Permissions restored roots. Current-head CI and final visual approval are required before merge.
<!-- SECTION:NOTES:END -->
