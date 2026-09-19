---
id: TASK-32794
title: Preserve MCP Tools filters and keyboard focus during refresh
status: Done
assignee:
  - '@codex'
created_date: '2026-09-18 15:39'
updated_date: '2026-09-18 16:04'
labels: []
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep MCP Tools usable while its catalog refreshes: the current implementation replaces an open filter, accepts obsolete choices, and loses keyboard focus when rows disappear.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Background catalog refresh preserves a focused or open server filter and its valid highlighted choice, including changed option labels and ordering.
- [x] #2 Delayed or detached filter events cannot revive a removed server or replace the currently displayed choice; valid pending choices and text drafts survive refresh.
- [x] #3 Catalog and diagnostic-action visibility changes retain reachable keyboard focus without stealing focus from other controls, and row identity remains stable for retained tools.
- [x] #4 Targeted production-CSS regressions and native compact/wide dark/light evidence verify the repair; review and documentation identify unqualified execution behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR. ADR path: backlog/decisions/150-design-token-system-and-design-language.md and backlog/decisions/161-component-pattern-library.md (existing). Reason: routine continuity repair within the existing MCP catalog, without changing persistence, execution, or permission boundaries.
1. Record failing mounted regressions for open/focused server controls, pending and obsolete choices, empty-state focus, and retained text/row identity using production CSS.
2. Retain the Select and update only changed options, reconcile live valid choices and highlighted identity, reject obsolete events, and transfer focus before hiding its owner.
3. Run targeted adjacent tests and static/design checks; independently review the implementation.
4. Verify the real private-profile app in dark/light compact/wide terminal journeys, record cleanup and visuals, update the MCP review ledger and PR evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Retained the server filter and its open/highlighted state through catalog refresh, committing pointer and keyboard choices before queued option indices can be reused. Live text/server reconciliation and guarded events keep explicit tool drills authoritative. Hidden table/recovery controls transfer focus to the text filter; the compact recovery action scrolls using measured canvas coordinates. Existing ADR-150/161 apply; no new authority, persistence or execution boundary.
Verification: 127 distinct targeted cases (66 final Tools/full-app compact, 30 adjacent Workbench, 31 governance), clean touched-file Ruff/format, diagnostic inventory and task-ID checks, independent review, and native dark/light 80x24 and 170x48 journeys. Run001 exposed the recovery-action clipping and is retained; final run003 has twenty captures, clean exit, healthy private databases and unchanged default fingerprints. Evidence: Docs/superpowers/qa/2026-09-18-mcp-tools-refresh/README.md. Updated MCP review ledger and incident-based testing lesson.
The previous saved head b46319eb17 has a separate unresolved PR Fast Lane startup/teardown failure; this task does not claim to repair it. Inspector/execution and broader MCP review remain open. PR2707 remains draft and unmerged.
<!-- SECTION:NOTES:END -->
