---
id: TASK-32787
title: Preserve Tool Profile writes across Settings recreation
status: Done
assignee:
  - '@codex'
created_date: '2026-09-18 12:03'
updated_date: '2026-09-18 12:38'
labels:
  - settings
  - tool-profiles
  - lifecycle
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep admitted Tool Profile writes observable and current across Settings destruction and recreation and settle them before application resources close.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Recreated Settings shows pending writes and receives actual terminal outcomes and refreshed profile facts without stealing newer focus.
- [x] #2 Repeated actions cannot admit duplicate same-operation writes across screen visits; preparation and exact review authority remain unchanged.
- [x] #3 Observer cancellation returns promptly while the admitted operation retains ownership; export cancellation remains subject to its publication boundary.
- [x] #4 Shutdown closes admission and drains owned work before dependent resources close without adding a separate timeout or bypassing the existing exit watchdog.
- [x] #5 Targeted lifecycle and real-service native evidence qualify the change and document ownership in an ADR.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implement Docs/superpowers/plans/2026-09-18-tool-profile-write-lifetime.md: reproduce recreation and lifecycle gaps; add lazy app-owned admitted-write coordination; retain screen-owned preparation and prompt observer cancellation; project current pending/terminal facts in replacement Settings; integrate shutdown drain before resource closure; run targeted coordinator/UI/service/shutdown checks and independent review; qualify native recreation/shutdown with exact-source lifecycle evidence; update ledgers and save draft PR2707. ADR required: yes. ADR path: backlog/decisions/167-tool-profile-write-lifetime.md. Reason: Move admitted operation/outcome ownership from disposable Settings screens to the existing application lifecycle while preserving ADR-107 service authority and ADR-150 presentation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
App-owned admitted Tool Profile writes now survive Settings recreation, retain bounded pending/terminal receipts and refresh actual facts without taking newer focus. Observer cancellation remains prompt; exact service review authority and export publication boundaries are preserved. Normal shutdown closes admission and drains before resources, with the existing process-owned watchdog armed first. ADR-167 records the lifecycle boundary; ADR-107/150 remain applicable. 229 distinct targeted cases pass, including coordinator ordering, malformed-result/cancellation recovery, mounted recreation and existing real publication/removal boundaries. Independent review corrections are verified. Four native recreation cells plus a fifth pending-write shutdown journey pass on final pinned source; all 17 captures inspected, 11 private databases clean and default fingerprints unchanged. Earlier fixture/isolation failures are retained and explained in Docs/superpowers/qa/2026-09-18-tool-profile-write-lifetime/README.md. Scoped lint/format, governance, Backlog and diagnostic inventory guards pass. Changed operations coordinator, app lifecycle wiring, Settings projection, targeted tests, QA and ledgers. Broader component/MCP review and draft PR2707 merge approval remain open.
<!-- SECTION:NOTES:END -->
