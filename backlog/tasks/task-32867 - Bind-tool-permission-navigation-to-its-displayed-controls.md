---
id: TASK-32867
title: Bind tool permission navigation to its displayed controls
status: Done
assignee:
  - '@codex'
created_date: '2026-09-19 17:04'
updated_date: '2026-09-19 18:46'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep Change in Permissions navigation in the Tools inspector and Test Tool panel tied to the tool and profile displayed when its control was mounted. Audit navigation is already tracked separately in TASK-32837.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Queued presses from replaced or cleared Tools and Test Tool views cannot navigate to a different tool or profile, including while teardown is pending.
- [x] #2 Current visible controls retain their rendered tool and profile identity, remain retryable, and closing Test Tool does not invalidate the separate tool-details control.
- [x] #3 Targeted regression tests and private native dark/light journeys verify both navigation controls without connecting to an MCP server or executing a tool.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce delayed public Button.Pressed events after selection/profile replacement, clear, and pending teardown for both permission navigation controls.
2. Bind each navigation control to its rendered tool/profile tuple and invalidate it before its owning view starts pruning. Keep navigation retryable and reject hidden or disabled controls.
3. Run targeted regressions and independent review; verify both real keyboard navigation journeys in a private profile at compact/wide dark/light sizes. Save a bounded draft PR against dev for final visual approval.
ADR required: no
ADR path: backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md
Reason: routine repair of existing inspector action ownership; no new storage, authority, service contract or UI structure.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented captured button ownership for the two Tools/Test Tool permission jumps. Selection refresh, permission rerender and Test Tool close retire the affected controls before asynchronous removal; handlers reject detached, hidden, disabled and inactive-screen controls while leaving live navigation retryable. Audit navigation remains in separate TASK-32837 / PR2724.

Verification: 75 targeted cases pass (23 new navigation cases, 37 write/revoke cases, 15 existing routing/panel cases); seven preflight guards pass. Independent review findings were reproduced and fixed. New files pass Ruff; existing inspector diagnostics unchanged at 13, edited lines formatted. Twelve inspected native dark/light captures at 120x40/170x48 verify eight real keyboard routes with unchanged permissions, no connection/execution and clean private-profile shutdown. No full suite ran.

Evidence: Docs/superpowers/qa/2026-09-19-mcp-permission-navigation/README.md and GALLERY.md. Existing ADR-150/ADR-161 apply; no new ADR. Implementation is complete; draft PR current-head CI/review and owner visual approval remain merge gates.

Owner approved the visual gallery. Rebased conflict-free onto dev030fd9935d; all 75 targeted cases pass again. Addressed both Qodo findings in QA only: separated the validated bootstrap from grouped journey imports and documented SystemExit. 35 argument/path cases pass; fresh eight-route native replay has twelve captures identical to the approved gallery. Evidence: Docs/superpowers/qa/2026-09-19-mcp-permission-navigation/qodo-followup/README.md. Current-head CI/review remains required before merge.
<!-- SECTION:NOTES:END -->
