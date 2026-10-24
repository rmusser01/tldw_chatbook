---
id: TASK-36478
title: Console internals decomposition tests fail in some xdist orders
status: In Progress
assignee:
  - '@claude'
created_date: '2026-10-23 11:02'
updated_date: '2026-10-24 15:03'
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce on fu-base (dev 0225e77310) with the b1_bootstrap_all evidence plugin: single-process file runs, the recorded B2a worker sequence (f5-replay.ids), and a per-test config-cache trace (cfgleak_trace plugin) to see what state crosses test boundaries.
2. Bisect the recorded sequence down to a minimal (two-test) reproduction for test_console_empty_regions_do_not_stack_nested_terminal_frames and name the leaking test; find the within-file leakers the same way.
3. Write a failing regression test first: an outer test that runs a child pytest over two bootstrap-profile probes (a writer that persists config and mutates the cached settings dict, then a reader that asserts the profile config is pristine). Show it RED on the current conftest for the right reason.
4. Fix at the source of the shared state: Tests/conftest.py isolate_test_environment hands every bootstrap-profile test the same config.toml; snapshot it before each such test and restore it (and drop config.py's settings/config caches) after teardown, so no bootstrap-profile test sees another's config writes or in-place cache mutations.
5. Verify: regression test GREEN plus named mutations; the minimal repro RED on base and GREEN on head; three single-process runs of the file and of the recorded worker sequence on each arm; paired runs of the affected files under the flake protocol; ruff check/format; preflight.
6. Record tests that fail ALONE on dev (not order-dependent) separately with their reasons; file a follow-up task for them rather than editing their assertions blind.
7. Implementation Notes, ACs from evidence, report, PR body.
<!-- SECTION:PLAN:END -->
