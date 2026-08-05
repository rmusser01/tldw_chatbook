---
id: TASK-687
title: Library shell reads CLI config when app_config already answered
status: Done
assignee: []
created_date: '2026-07-26 05:36'
updated_date: '2026-07-26 18:06'
labels:
  - library
  - bug
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two Library shell tests assert that get_cli_setting is not consulted once app_config already carries the search-history and rail-preference values, and both fail: something reads the CLI config anyway, so a value set in app_config can be overridden by a stale one on disk. Pre-existing on dev, found while regression-testing the 684.2 registry work; both fail identically at 05ebe2ab7.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Search history and rail preferences come from app_config when it has them
- [x] #2 get_cli_setting is consulted only as a fallback
- [x] #3 Both existing precedence tests pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Not a product bug either. Precedence works; both tests were over-broad.

Probed by recording every get_cli_setting call during a shell load instead of raising on it. The result is symmetric: in the search-history test the history comes back as ('from-app-config',) and library.search is NEVER read; in the rail test details_open is True and library.rail_state is NEVER read. In each case what tripped the blanket raise was the OTHER setting -- the one that test did not seed, so its CLI fallback is correct -- plus twenty library.ingest_options.* reads from the ingest canvas loading its persisted per-type options.

Then a second, worse problem surfaced. Narrowing each trap to raise only on its own key looked right and was VACUOUS: the rail fallback sits inside an except-Exception block, which swallows an AssertionError raised from the patch, so the test could not fail at all. Mutation testing caught it -- breaking precedence outright still passed. That also explains the original symptom: the tests failed with 'Library shell never loaded' rather than the assertion message, because the raise that actually escaped came from a call site without a try/except and killed the load worker.

Both tests now RECORD the calls and assert afterwards, outside the screen's exception handling. Mutation-checked: removing the app_config branch from _library_rail_preferences fails the test with 'rail preferences fell back to the CLI config despite app_config'.

Files: Tests/UI/test_library_shell.py.
<!-- SECTION:NOTES:END -->
