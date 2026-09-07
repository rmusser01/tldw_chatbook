---
id: TASK-915
title: Sticky manual collapse of Agent rail section while fleet busy
status: Done
assignee:
  - '@claude'
created_date: '2026-07-27 03:55'
updated_date: '2026-07-27 20:11'
labels:
  - console
  - fleet-ux
  - polish
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Agent rail section auto-opens whenever the fleet summary has content (parallel-agents train). Manually collapsing it while the fleet is busy holds only until the next agent-section payload change, which re-forces it open; the persisted preference is honored again once the fleet quiets and is never corrupted. Add a transient "user dismissed during this busy window" flag so the collapse sticks until the fleet quiets or a new run starts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Collapsing the Agent section during a busy fleet sticks across payload changes within that busy window.
- [x] #2 Auto-open still triggers for a newly busy fleet after quiet; persisted preference still never overwritten by the transient force.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a transient, never-persisted `_agent_section_user_dismissed_while_busy` flag on ChatScreen (init to False alongside `_console_agent_section_last`).
2. `_toggle_console_rail_section`: when toggling the "agent" section, set the flag True on manual close while `_console_agent_fleet_summary_line()` is non-empty; clear it on manual reopen.
3. `_apply_fleet_agent_section_auto_open`: compute the fleet line once; if empty, clear the flag and return unchanged (fleet quiet -> next busy window auto-opens again); if non-empty and the flag is set, skip the force (persisted preference still never written).
4. TDD: add regression tests to Tests/UI/test_console_parallel_runs.py covering (a) collapse sticks across a same-window payload change, (b) auto-open fires again for a new busy window after quiet, (c) persisted rail preference never force-written to agent_open=True, (d) manual reopen clears the dismissal.
5. Run Tests/UI/test_console_parallel_runs.py + Tests/UI/test_console_rail_sections.py together; verify tests fail (AttributeError) with the flag/logic reverted, confirming they exercise the fix.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a transient, never-persisted `_agent_section_user_dismissed_while_busy` bool on ChatScreen (init alongside `_console_agent_section_last`). `_toggle_console_rail_section` sets it when the user manually collapses the "agent" section while `_console_agent_fleet_summary_line()` is non-empty, and clears it on manual reopen. `_apply_fleet_agent_section_auto_open` now computes the fleet line once: empty -> clear the flag and return unchanged (releases the next busy window to auto-open); non-empty + flag set -> skip the force (returns the rail state as-is instead of `replace(..., agent_open=True)`). The persisted preference path (`_set_console_rail_preference`) is untouched by any of this -- only the transient flag and the rendered dataclass change, matching the existing `_apply_pending_launch_inspector_auto_open` pattern this mirrors.

Added 3 regression tests to Tests/UI/test_console_parallel_runs.py: manual collapse sticks across a same-busy-window payload change (fleet count text itself flips from "1 other..." to "2 other..."), auto-open fires again for a genuinely NEW busy window after the fleet quiets, and manual reopen clears the dismissal. All three assert the persisted rail-state config never has `agent_open: True` written by the force (only by the user's own explicit reopen toggle, which is expected/out of scope for AC2). Confirmed each new test fails with `AttributeError` when the fix is reverted (git stash), so they exercise the change rather than passing vacuously.

Files touched: tldw_chatbook/UI/Screens/chat_screen.py (`__init__`, `_apply_fleet_agent_section_auto_open`, `_toggle_console_rail_section`), Tests/UI/test_console_parallel_runs.py (3 new tests + 2 small helpers).

Tests: Tests/UI/test_console_parallel_runs.py + Tests/UI/test_console_rail_sections.py, 60 passed, 0 failed.
<!-- SECTION:NOTES:END -->
