---
id: TASK-1140
title: Fleet summary line renders below the rail scroll fold
status: Done
assignee:
  - '@claude'
created_date: '2026-07-27 18:05'
updated_date: '2026-07-28 01:03'
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
Round 1 (reorder-only, commit f82275a22): moved #console-agent-fleet-summary to be the FIRST child of the Agent section's compose body, above status/steps/subagents, but still inside the shared `#console-left-rail-body` VerticalScroll. My own test only exercised that placement with Session/Model rail sections COLLAPSED -- an arranged state chosen (wrongly) to "isolate" the Agent section's own content as the sole variable, which hid the real gap: Session and Model are BOTH open by PERSISTED DEFAULT (ConsoleRailPreferences.session_open/model_open=True), the layout every real session actually starts in. Reviewer reproduced with the same harness, changing only that one thing (left Session/Model at their defaults) and found the fleet line's region still landed at y=45 on a 44-row screen (y=67 with the round-1 fix reverted) -- entirely unpainted (get_widget_at raised NoWidget). Root cause: Session's own content alone already exceeds the rail body's ~10-row visible budget in a 44-row terminal, so ANY position inside that shared scrollable flow -- including the top of the Agent section -- can land below the fold regardless of the Agent section's own step-content length. My round-1 framing ("isolate the pre-existing characteristic") mischaracterized this as an unrelated rail-budget quirk; it was actually AC#1's literal claim ("any amount of rail content") failing in the default case.

Round 2 (this fix, required by review): pinned #console-agent-fleet-summary OUTSIDE the scrollable rail flow entirely -- moved (not duplicated; two widgets sharing one id is invalid) from inside the Agent section's compose body to a plain, non-scrolling sibling of `left_rail_header`, yielded just before `with VerticalScroll(id="console-left-rail-body")` begins. `#console-left-rail-body` is CSS `height: 1fr`, so it only ever claims whatever vertical space remains after its non-scrolling siblings (the header, now also this line) are laid out -- the fleet line is therefore painted unconditionally: independent of rail scroll position, which sections are open or collapsed, and how much step content the Agent section carries. Same id, verbatim copy, markup=False, and display:none-at-zero-counts as before (Textual excludes a display:none widget from layout entirely, so the pinned slot occupies zero rows when quiet, not a blank line). `_sync_console_agent_section` already resolved it via query_one(id) and toggled its display independently of the Agent section's own open state, so no change was needed there; `_apply_fleet_agent_section_auto_open`'s docstring was corrected to stop claiming the fleet line lives in the Agent section body (it still force-opens that section for its own status/steps/subagents detail, just no longer for fleet-line visibility).

Testing: replaced the single AC#2 test with three, sharing a `_setup_tall_steps_and_parked_fleet` helper (done viewed session with 5 long wrapped step bullets via a fake ConsoleAgentBridge + a parked background session): (1) `..._default_sections` -- Session/Model left at their real persisted defaults (open), the reviewer's own repro promoted to a permanent regression guard; (2) `..._collapsed_sections` -- same repro with Session/Model explicitly collapsed, kept alongside (not replacing) the default case so both arrangements are covered; (3) `test_fleet_summary_line_occupies_no_row_when_fleet_is_quiet` -- hidden-at-zero, asserted against the rendered region (width==0, height==0) and display, both at the quiet baseline and after a busy-then-quiet-again cycle. All three use a shared `_assert_painted_at_own_region` compositor hit-test (App.get_widget_at) rather than a raw region.y bound, since Widget.region is reported in an unclipped coordinate space -- a widget below a scrollable ancestor's fold still has a region, just one that ancestor never paints, so `region.y < app.size.height` alone can't distinguish "below this ancestor's fold" from "within it".

Revert-check (required evidence, this round): stashed only the chat_screen.py round-2 change (keeping the round-2 test file) against the round-1 commit (f82275a22) and reran the two viewport-intersection tests: `..._default_sections` FAILS (`nothing is painted at Static(id='console-agent-fleet-summary'...)`, matching the reviewer's finding), `..._collapsed_sections` still PASSES (matching round-1's own -- narrower -- claim). Restored the round-2 fix; all three new tests plus the full required gate pass.

Gate: Tests/UI/test_console_parallel_runs.py + Tests/UI/test_console_rail_sections.py, one blocking foreground pytest call, interpreter /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python (import path verified -> /private/tmp/tldw-uatfix) -- 63 passed. Tests/UI/test_console_agent_rail.py (21 passed) run again as an extra sanity check, not part of the required gate.

Files modified: tldw_chatbook/UI/Screens/chat_screen.py (fleet Static moved from the Agent section body to a pinned sibling of the rail header; docstring correction on _apply_fleet_agent_section_auto_open), Tests/UI/test_console_parallel_runs.py (shared setup/assertion helpers + three tests replacing the single round-1 test).
<!-- SECTION:NOTES:END -->
