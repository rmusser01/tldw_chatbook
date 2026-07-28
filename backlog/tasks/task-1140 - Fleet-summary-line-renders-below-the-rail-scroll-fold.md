---
id: TASK-1140
title: Fleet summary line renders below the rail scroll fold
status: Done
assignee:
  - '@claude'
created_date: '2026-07-27 18:05'
updated_date: '2026-07-28 00:51'
labels:
  - console
  - fleet-ux
  - uat
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UAT (Docs/superpowers/qa/parallel-agents-uat-2026-07-27, F1): #console-agent-fleet-summary sits at the bottom of the Agent rail section below the viewed session's status/step bullets, so after any agent run it is off-screen unless the user wheel-scrolls deep into the rail — defeating the spec's "at a glance" intent. Headless proof: region y=48 in a 44-row viewport with an all-True display chain, so the existing render-path test passes while nothing is visible. Move the line to the top of the Agent section (or pin it outside the scrollable flow) and strengthen the test to assert viewport intersection, not just the display chain.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 With a busy fleet and any amount of rail content, the fleet line is visible without scrolling.
- [x] #2 A test asserts viewport intersection (region within the visible viewport), failing against the current placement.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read task-1140 (2 ACs) and F1 in Docs/superpowers/qa/parallel-agents-uat-2026-07-27/report.md.
2. Locate the Agent rail section's compose block in tldw_chatbook/UI/Screens/chat_screen.py (agent_body Vertical) and confirm the fleet Static's placement (last child) plus the update/auto-open/sticky-collapse machinery's dependency on it (query_one by id + `_console_agent_fleet_summary_line()` return value only -- not DOM position).
3. Reorder the compose yields so #console-agent-fleet-summary is the FIRST child of agent_body (directly under the "Agent" section header), before status/steps/subagents/back-button.
4. Extend Tests/UI/test_console_parallel_runs.py with a new AC#2 test: a done viewed session with several long step bullets (fake agent bridge) + a parked background session, Session/Model rail sections collapsed to isolate the Agent-section-specific regression, asserting the compositor (`App.get_widget_at`) actually paints the fleet Static at its own reported region -- not just that its display chain is all-True.
5. Revert-check: temporarily stash the chat_screen.py change and confirm the new test fails against the pre-fix placement (evidence for Implementation Notes), then restore the fix.
6. Run the required gate (Tests/UI/test_console_parallel_runs.py + Tests/UI/test_console_rail_sections.py) and confirm no regressions.
7. Update the task file (ACs, Done, Implementation Notes) and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approach: reordered the Agent rail section's compose block in ChatScreen so #console-agent-fleet-summary is the FIRST child yielded into agent_body, directly under the "Agent" DestinationRailSectionHeader, above the status/steps/subagents Statics and the Back button. Pure reorder of existing yield statements -- no new widgets, no CSS changes. _sync_console_agent_section, _apply_fleet_agent_section_auto_open, and the task-915 sticky-collapse flag all resolve widgets via query_one(id) and key off _console_agent_fleet_summary_line()'s return value, never DOM position, so none of that machinery needed touching.

Testing: extended Tests/UI/test_console_parallel_runs.py with test_fleet_summary_line_intersects_the_visible_viewport (AC#2), sitting alongside (not replacing) the existing display-chain-only test_fleet_summary_line_is_reachable_on_the_live_rendered_surface. Setup: a done viewed session with 5 long wrapped step bullets (fake ConsoleAgentBridge) + a parked background session (console._park_console_approval), Session/Model rail sections collapsed via _set_console_rail_preference to isolate the Agent-section-specific regression F1 describes (Session alone renders ~22 rows of content against a ~10-row rail viewport in this harness, which is an unrelated, pre-existing rail-budget characteristic -- collapsing it keeps the test scoped to task-1140's actual claim).

Discovered mid-implementation that Widget.region is reported in an unclipped coordinate space -- a widget scrolled out of its VerticalScroll ancestor still reports a region, just one the container never paints -- so the task's suggested `region.y < app.size.height` check is too weak (it only catches "past the bottom of the whole 44-row screen", not "past this scrollable ancestor's own ~10-row fold", which is the actual live-UAT failure mode). Used the compositor's own hit test (App.get_widget_at(x, y)) instead: it asks what widget is ACTUALLY painted at the fleet Static's own region, matching the real terminal-rendering bar. Revert-check (task requirement): stashed only the chat_screen.py reorder and reran the new test -- it fails, with the compositor painting a DIFFERENT widget (console-setup-modal-title) at the fleet line's pre-fix region (y=35) instead of the fleet Static itself. Restored the fix afterward; all 61 tests across Tests/UI/test_console_parallel_runs.py + Tests/UI/test_console_rail_sections.py pass, plus Tests/UI/test_console_agent_rail.py (21) as an extra sanity check (not part of the required gate).

Files modified: tldw_chatbook/UI/Screens/chat_screen.py (compose reorder only), Tests/UI/test_console_parallel_runs.py (new import + _TallStepsFleetBridge fixture + new test).
<!-- SECTION:NOTES:END -->
