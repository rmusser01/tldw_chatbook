---
id: TASK-911
title: 'Test the cap-refusal and-K-more title suffix'
status: Done
assignee: []
created_date: '2026-07-27 03:55'
labels: [console, tests]
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
send_refusal_copy's cap message truncates busy-session titles to 3 plus an "and K more" suffix. The suffix branch (more than 3 busy sessions) has no test; it is user-facing spec copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A unit test drives 4+ busy sessions and asserts the exact "and K more" refusal copy.
<!-- AC:END -->

## Implementation Notes

Added `test_cap_refusal_truncates_and_k_more_suffix()` to `Tests/Chat/test_console_run_state_per_session.py`. The test:
- Creates 5 sessions using `controller.new_session()` (avoiding deduplication)
- Marks 4 sessions as STREAMING (busy state)
- Calls `send_refusal_copy()` on a fresh session (which should be refused)
- Asserts the refusal message contains exactly 4 busy agents, the first 3 titles (Alpha, Bravo, Charlie), and the literal " and 1 more" suffix
- Validates against the `CONSOLE_CAP_REFUSAL_TITLE_LIMIT` constant (imported, not hardcoded)

Verified test catches regressions: temporarily lowering the title limit constant from 3 to 2 causes the test to fail as expected (suffix changes to " and 2 more", first 3 titles are no longer present).

All 18 tests in the file pass.
