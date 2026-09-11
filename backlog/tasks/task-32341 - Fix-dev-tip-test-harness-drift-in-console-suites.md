---
id: TASK-32341
title: >-
  Fix dev-tip test harness drift in console suites
status: To Do
assignee: []
created_date: '2026-09-11 10:00'
labels:
  - console
  - testing
  - tech-debt
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verified during the TASK-32320..32340 arc (2026-09-10/11): a set of
console test failures exist on origin/dev tip a0b8f96416 itself, before
any of this arc's changes (confirmed repeatedly by stashing arc edits
and re-running on the clean tree — identical failures). They are harness
drift, not product regressions:

- `Tests/UI/test_console_workbench_contract.py` — 14 failures incl.
  test_console_left_rail_keeps_session_and_moves_staged_context_out
  (mount order: test expects Sources tray at body index 2; dev mounts the
  Agents fleet section there), the four
  test_console_empty_recovery_action_copy_matches_setup_blocker
  variants, blocked/ready inspector groups, composer actions, header
  inline-subtitle trio, counter chips, active-stream sync.
- `Tests/UI/test_console_agent_controller.py` — test_agent_bridge_is_*
  (harness SimpleNamespace lacks get_message_exchanges /
  capture_console_settings_origin).
- `Tests/UI/test_console_rail_color_grammar.py` — two failures passing
  `agent_fleet_section_state` kwarg dev's ConsoleLeftRail never accepted.
- `Tests/UI/test_console_fleet_survivor_tick.py` — survivor elapsed +
  unseen-mark tests.
- `Tests/UI/test_console_session_settings.py` (UI) — roleplay writer
  transition tests reference `chat_screen._release_console_roleplay_
  transition_after_writer` (missing on dev), name-refresh notify, and
  test_console_inspector_hosts_staged_context_above_source_readiness
  (same mount-order drift).
- `Tests/UI/test_console_narrow_layout.py` — retry speech button routes.
- `Tests/UI/test_console_inspector_navigation.py` — staged owner sync
  cue, responsive hide/reveal handoff.
- `Tests/UI/test_console_native_chat_flow.py` — duplicate send/stop
  control.

Each was individually confirmed on the clean tree during the arc; the
failure sets were byte-identical with and without the arc's changes.

Filed so the board records that these pre-date the rail-UX arc and are
not its regressions; whoever picks this up decides per-test whether the
test or the code moved (the mount-order ones look like the tests
predating the rail redesign's section order).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each listed failing test is triaged: test updated to dev's current contracts, or a defect is filed for the code
- [ ] #2 The suites listed pass on a clean origin/dev checkout
- [ ] #3 No product behavior changes are masked by the fixes (each triage notes its direction)
<!-- AC:END -->
