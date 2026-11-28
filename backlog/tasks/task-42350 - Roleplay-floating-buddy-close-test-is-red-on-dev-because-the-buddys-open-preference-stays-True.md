---
id: TASK-42350
title: Roleplay floating buddy close test is red on dev because the buddy's open preference
  stays True
status: In Progress
assignee: []
created_date: 2026-11-22 16:37
updated_date: 2026-11-24 17:21
labels:
- follow-up
- tests
- roleplay
priority: medium
references:
- Tests/UI/test_personas_workbench.py
- tldw_chatbook/Widgets/Persona_Widgets/persona_buddy_widget.py
- tldw_chatbook/UI/Screens/personas_screen.py
- scripts/ui_pr_gate_census.txt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the TASK-33910.49 port (round 20, fu/r20-roleplay), which reported it red on dev 8a55555798. Re-run at 59cc99e61b through the scratch wrapper: red. TASK-41800 had already noted it red on dev 5266c33936 and on no lane, and filed nothing for it.

**Why it matters.** `test_floating_buddy_close_refreshes_active_personas_inspector` (`test_personas_workbench.py:15502`) clicks the floating buddy's close button (`#persona-buddy-close`) and expects the buddy to close and the Personas Inspector to offer **Show** again. It fails at the first check after the click, `assert controller.current_preferences().open is False`, with `open=True`. Either closing the buddy does not stick, which a user would see, or the test drives the close in a way the widget now ignores. Until it is fixed, nothing checks that the buddy's close and the Inspector's Show/Close buttons agree, and the node is on no PR lane.

**Evidence.** The widget closes only when its view is current (`persona_buddy_widget.py:1170-1176`, `if not self._is_current_view(): return`) and publishes the change through `_apply_and_schedule_preferences` (`:1181`). Not traced further.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The cause is named: a product bug in the close, or a test that no longer drives the close the way a user does
- [x] #2 Clicking the floating buddy's close button closes the buddy, persists `open` as False, and leaves the Personas Inspector's Show enabled and Close disabled, and the test checks all of it
- [x] #3 The test passes through the scratch wrapper on a branch carrying the fix
- [x] #4 The node runs on a PR lane
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace mouse-down and mouse-up against the compositor hit regions.
2. Run Buddy workbench tests with the production width utilities and click each resting close cell.
3. Enroll the assertions in the UI PR lane.
ADR required: no
ADR path: backlog/decisions/150-design-token-system-and-design-language.md
Reason: Verify the existing design-token and Buddy preference contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Mouse-down hit the 3-cell close button at x=96. Focus updated the label to Close and realigned the control to x=92, but the unstyled harness omitted w-7 so mouse-up hit the underlying frame. The styled harness loads the production width utilities and retains all three original click cells. Clicks in left, center and right each persisted open=False, removed the overlay, and refreshed the active Inspector. Added the parametrized test to the UI PR census.
Verified in an isolated checkout using the test bootstrap scratch profile.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
