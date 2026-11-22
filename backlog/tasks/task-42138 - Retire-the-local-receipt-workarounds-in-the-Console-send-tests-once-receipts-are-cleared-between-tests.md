---
id: TASK-42138
title: "Retire the local receipt workarounds in the Console send tests once receipts are cleared between tests"
status: Done
assignee:
- '@claude'
created_date: 2026-11-20 20:06
updated_date: 2026-11-21 19:12
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/UI/test_console_send_without_mcp_store.py
- Tests/UI/test_console_runtime_ownership.py
- Tests/conftest.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the consoletests stream (TASK-42018, round 18). TASK-42018 (round-18 consoletests branch, d10411ca0b, not yet on dev) clears unseen Console receipt marks between bootstrap-profile tests in `Tests/conftest.py`. Before it, TASK-41900 and TASK-41904 added two local workarounds to `Tests/UI/test_console_send_without_mcp_store.py`:
- `_acknowledge_leaked_receipts` (line 105) marks every unseen receipt seen before a test mounts its app;
- `_wait_for_own_receipts_seen` (line 123) waits until the test's own receipts are seen, so a later test's app does not announce them.

`Tests/UI/test_console_runtime_ownership.py` imports both (lines 452-453; calls at 479 and 533). Once the conftest clear is on dev, the first is redundant and the second may be.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 With TASK-42018's clear on dev, each helper is either removed with its call sites, or kept with a comment naming what it still guards that the clear does not
- [x] #2 For each removed helper, the tests that called it pass in one process right after a test that leaves an unseen receipt (`-p no:xdist`), and the notes record that run
- [x] #3 Both files' tests pass
<!-- AC:END -->

## Implementation Plan

1. Remove `_acknowledge_leaked_receipts` and `_wait_for_own_receipts_seen` and their call sites in both files, unless a run shows one still guards something the conftest clear does not.
2. Evidence: in one process (`-p no:xdist`), a temporary test that leaves an unseen receipt runs first, then every former caller; and a negative control with the conftest clear disabled shows the planted receipt would reach them.
3. Run both files at -n 4 several times.

## Implementation Notes

**Removed both helpers and every call site.** TASK-42018's clear (`_clear_bootstrap_console_receipts`, a91a9645c4, on dev) runs in `Tests/conftest.py`'s teardown of every bootstrap-profile test and deletes the unseen-receipt marks and their outcome companions from the shared ChaChaNotes database. That is what both helpers guarded:
- `_acknowledge_leaked_receipts` (before mount: mark earlier tests' receipts seen) is redundant, because the earlier test's teardown already cleared them.
- `_wait_for_own_receipts_seen` (before leaving: wait until this test's own receipts are seen, so the next test's app does not announce them) is redundant too, because this test's own teardown clears whatever it leaves. It guarded nothing about the test's own outcome: both callers assert their reply and notices before it ran.

Call sites removed: `test_ordinary_send_reaches_the_provider_and_saves_the_reply` (6 cases) and `test_unbound_store_with_a_replaced_mcp_reader_is_still_refused` in `test_console_send_without_mcp_store.py`; `test_a_send_pressed_during_the_first_send_save_waits_and_is_sent` in `test_console_runtime_ownership.py` (both helpers and their import).

**Evidence (AC#2).** A temporary, uncommitted bootstrap-profile test planted an unseen receipt (probe_g's shape: `set_console_terminal_with_cursor(..., "complete")` for a made-up chat) in the shared database. In one process each (`-p no:xdist`, load 45-50), with it first:
- plant + `test_unbound_store_with_a_replaced_mcp_reader_is_still_refused`: 2 passed (25.9 s);
- plant + `test_a_send_pressed_during_the_first_send_save_waits_and_is_sent`: 2 passed (40.2 s);
- plant + `test_ordinary_send_reaches_the_provider_and_saves_the_reply` (all 6 cases): 7 passed (123.6 s).
Negative control: with the conftest call to `_clear_bootstrap_console_receipts()` replaced by `pass`, plant + the unbound-store test failed: `['A Console turn completed while hidden. Return to Console to review.'] != ['Console preparation source unavailable; Send again.']`. Restored; the planting file was deleted.

**AC#3.** Both files at `-n 4`: 130 passed, 1 xfailed in 299 s, 303 s and 307 s (three runs, load 45-61). Both files are laned whole (the send file in the UI census, the ownership file in the admission-sensitive PR Fast Lane).

Files: `Tests/UI/test_console_send_without_mcp_store.py`, `Tests/UI/test_console_runtime_ownership.py`.
