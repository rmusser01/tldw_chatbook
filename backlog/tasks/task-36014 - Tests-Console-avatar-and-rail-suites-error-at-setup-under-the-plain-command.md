---
id: TASK-36014
title: 'Tests: Console avatar and rail suites error at setup under the plain command'
status: To Do
assignee: []
created_date: '2026-10-21 07:47'
labels:
  - tests
  - console
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while checking TASK-33620.15.7 (2026-10-21), identical on dev 906497d9b4 and on that branch. Under the plain command, these suites never reach their assertions:

- With `RecoveryRequired: raw_source_selection_changed` at setup: `test_console_character_avatar_copy_budget.py`, `test_character_expression_avatar.py`, `test_console_character_context.py` (16 of 61), `test_console_context_rail_content.py`, `test_console_reaction_picker.py`, `test_console_send_authority_summary.py`, `test_console_run_inspector.py`, `test_console_voice_chip.py`, `test_console_rail_sections.py`, `test_console_rail_section_layout.py` and `test_console_left_rail.py` (13 of 14).
- With "fixture 'owned_console_apps' not found": `test_console_avatar_geometry_offloop.py` (6 of 8). It imports the avatar fixtures without the ownership fixture they request, and importing it too then fails with "Cannot register a database for an uncaptured runtime", because the fixture patches only the requesting module's `_build_test_app`.

With every case marked `bootstrap_profile` through a scratch plugin, all of the first group pass on both trees except `test_console_character_avatar_copy_budget.py` (1 failure), `test_console_left_rail.py` (5 failures, for example "Section 'details' did not toggle") and `test_console_voice_chip.py` (TASK-36013), each the same set on both trees. So a change to the Console avatar, the Character section, the send-authority summary or the rails gets no signal from these suites unless its author knows the workaround.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each listed suite reaches its assertions under the plain command (marked bootstrap_profile, or its fixtures fixed)
- [ ] #2 test_console_avatar_geometry_offloop.py runs its cases with the shared avatar fixtures and their ownership fixture
- [ ] #3 No suite's pass count drops compared with the plugin-marked run on the same base
<!-- AC:END -->
