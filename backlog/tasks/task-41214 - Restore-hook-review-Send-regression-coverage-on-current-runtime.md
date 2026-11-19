---
id: TASK-41214
title: Restore hook-review Send regression coverage on current runtime
status: Done
assignee:
  - '@codex'
created_date: '2026-11-15 22:32'
updated_date: '2026-11-16 04:43'
labels:
  - ci
  - console
  - hooks
dependencies: []
references:
  - >-
    https://github.com/rmusser01/tldw_chatbook/actions/runs/43750308318/job/132962025958
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
CI run 43750308318 fails eight hook-review Send regressions on current dev. Declined reviews lack the expected visible cancellation and approved workbench or Enter sends are not observed dispatching. Resolve the verified cause so these user outcomes are exercised reliably on the current runtime.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Declining hook review through each Send route settles, keeps the draft, shows cancellation copy and leaves both message pumps responsive.
- [x] #2 Allowing review resumes each captured Send exactly once on all routes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: keep historical pump-freeze tests on the existing supported compatibility route; no runtime ownership or approval-policy changes.
1. Trace received-intent routing and confirm that dispatch alone is not a tracked compatibility adapter; retain CI run 43750308318 as red evidence.
2. In the shared dispatch-recording test helper, wrap the existing tracked async launcher with a forwarding callback to select compatibility routing explicitly, following the send/resend guard fixture precedent.
3. Await the enabled review button actually owning focus before sending driver-style Enter: Textual schedules focus through app.call_later. Keep the existing five-second answer bound and all outcome assertions.
4. Run the route matrix and both initial-review consumers plus resident hook-review contracts; independently review and record evidence, including any Windows teardown limitation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Kept latest dev's captured custom-review adapter seam and its busy-controller assertion, removing this branch's duplicate launcher override during rebase. This PR makes the answer helper async and waits for the enabled target to own focus before driver-style Enter, because Widget.focus schedules app.call_later. All five callers await it; Escape, draft retention, cancellation copy, responsive-pump and exactly-once assertions remain intact. No production runtime or timeout changes.

Validation on rebased dev 713fa22f92: all approved and declined routes plus both resident remount/Stop consumers pass locally (11 cases). The earlier hosted PR attempt also passed the full Console P0 gate. CI-pinned Ruff and independent rebase review pass. Earlier Windows timing/retirement failures remain in the receipts; the final rebased targeted run is clean.

Only Tests/UI/test_console_hook_review_send_freeze.py changes. ADR required: no; existing compatibility and resident-review boundaries are preserved. Adopting dev's seam replaces the duplicate fixture-selection step in the original plan.
<!-- SECTION:NOTES:END -->
