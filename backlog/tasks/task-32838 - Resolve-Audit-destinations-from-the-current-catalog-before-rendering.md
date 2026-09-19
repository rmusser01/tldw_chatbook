---
id: TASK-32838
title: Resolve Audit destinations from the current catalog before rendering
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-19 05:54'
updated_date: '2026-09-19 20:07'
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
1. Rebase the existing PR2726 onto merged dev cccf0acdad, retaining both report histories and all PR2724 row-selection, profile and control-ownership guards.
2. Resolve both destination identities under the existing publication lock before selecting and rendering. Preserve post-selection profile validation and vanished-row warnings; verify old and new regressions against the combined implementation.
3. Refresh the native launcher to shared validation and supported terminal warm-up; verify private dark/light compact/wide journeys, focused tests, guards and independent review. Update the existing draft with exact conflict choices and fresh visuals for its own final owner approval.
ADR required: no
ADR path: backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md
Reason: routine integration of existing catalog-publication and inspector contracts without new authority, persistence or service boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Both Audit drilldowns now resolve the current identity under the existing catalog publication lock and retain that lock through row selection and inspector rendering. Pending refreshes cannot combine a captured definition with new rows or permissions; removed identities clear detail and warn, and profile-change checks remain. Added 12 regressions covering same-ID description/schema replacement, disconnection, removal, incomplete publication and real-app action routing. All 62 targeted cases, seven preflight guards, static-diagnostic baseline checks and changed-range formatting pass. Independent read-only review found no blocker. Eight inspected native dark/light captures at 120x40 and 170x48 use controlled collector replacement in a real private app; clean exit, healthy databases and unchanged default profile, with no tool execution or permission mutation. Evidence: Docs/superpowers/qa/2026-09-18-mcp-audit-catalog-freshness/README.md. ADR-150/161 apply; no new ADR needed. Updated both review ledgers. PR2724 row/control ownership and ongoing refresh of already-open inspectors remain separate; next review is Permissions restored roots. Current-head CI and final visual approval are required before merge.
<!-- SECTION:NOTES:END -->
