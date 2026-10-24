---
id: TASK-36478
title: 'Console internals decomposition tests fail in some xdist orders'
status: To Do
assignee: []
created_date: '2026-10-23 11:02'
labels:
  - tests
  - console
dependencies: []
references:
  - Tests/UI/test_console_internals_decomposition.py
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Exists on dev today (measured on dev 7700c63a68 during Roleplay frame B2a's gates, TASK-33910.3; B2a changes no Console code and does not fix it): test_console_empty_regions_do_not_stack_nested_terminal_frames failed in a single-process replay of one xdist worker's 180-test sequence (9 component_pattern_governance, 84 console_character_avatar, 87 console_internals_decomposition tests) on BOTH arms, twice (base 21 and 20 failed, head 23 and 22 failed), while passing alone. In the same file, test_console_composer_cursor_blink_toggles, test_console_composer_hides_cursor_when_blurred, test_console_composer_shows_cursor_when_focused, test_console_rag_query_validation_blocks_unsafe_markup and test_console_empty_transcript_promotes_setup_card_over_banner failed under xdist and passed every alone run on both arms, and three single-process runs of the whole file failed different sets on both arms (11-61 failures). The evidence (traces, replay id lists, logs) is in the B2a evidence directory, gates/ and draft/t10/.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The order dependence is found: the earlier test that leaves the state these tests read is named, with a minimal reproduction
- [ ] #2 The file passes in any order: three single-process runs of it and of the recorded worker sequence pass on dev
<!-- AC:END -->
