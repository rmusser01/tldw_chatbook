---
id: TASK-31982
title: 'Library media: a faulting read cascades and its Retry offers no recovery path'
status: Done
assignee: []
created_date: '2026-09-07 22:48'
updated_date: '2026-09-08 00:12'
labels:
  - library
  - media
  - robustness
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #6 P1, adjudicated from Assessment A's undo P0. The undo handler itself is correct (it counts rows actually restored and increments the rail by exactly that, `library_screen.py:24537-24552`; the N-of-M receipt counts genuine restore exceptions), and Assessment B got a clean undo on the same code, so the catastrophic case required a real media-DB fault. The deterministic residue: when the media read faults, the failure disables independent controls that did not need that connection (Export, Select, Review these, the facet counts, the whole Trash view); the fault-state Retry repeats one sentence with no attempt counter and no next step; and a restore that returns a non-Mapping is counted as neither success nor failure, so the receipt count can drift. Related: task-31972 (selection reconcile id-shape), task-31942 (analysis-save commit).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A media read failure is scoped to the failing surface; Export, Select, Review, facet counts and Trash remain usable when their own reads succeed
- [x] #2 The fault-state Retry either reconnects or, when it cannot, states the recovery action (e.g. Reopen Chatbook to reconnect to the media database) instead of repeating the same sentence
- [x] #3 The bulk-undo receipt count reflects rows the write layer actually committed, including a restore that returns an unexpected shape
- [x] #4 A test reproduces a faulting media read and asserts the independent controls stay live
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Part A (AC#3): in `_undo_library_media_bulk_delete`, count a restore that returns a non-Mapping (no exception) as a failure, not as neither. TDD with a fake `restore_media_item` returning a non-Mapping for one id.
2. Part B (AC#2): make the Media fault callout name a recovery step when the same reason recurs on a consecutive Retry, instead of repeating one sentence. TDD with two consecutive failed retries.
3. Part C (AC#1/#4): investigate whether the facet read and the row read are independent, then RULE; pin the ruling with a test.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Three-part fault-scope residue from critique #6 P1. The undo handler was already correct; this is the deterministic remainder.

- **Part A (AC#3)** — `_undo_library_media_bulk_delete` (`UI/Screens/library_screen.py`): a restore that returned a non-Mapping was counted as neither success nor failure, so the `N of M` receipt undercounted and the id silently dropped from the retry set. Added an `else` branch that treats it as a failure (`"restore returned no record"`), keeping the id retryable and the total honest.
- **Part B (AC#2)** — `UI/Library_Modules/library_media_browse_controller.py`: added `_page_fault_reason`/`_facet_fault_reason` trackers (kept across Retries, since `begin()` clears `page_failure` each request) and a `repeated` flag on `_raised_failure`. When the same reason recurs consecutively, the callout appends `· reopen Chatbook to reconnect to the media database` (new `_REOPEN_RECOVERY`) instead of repeating; no raw exception text (PR G privacy rule). Applied to both the page and facet fences.
- **Part C (AC#1/#4) — RULING (branch 1):** the facet read (`list_library_media_types`, `_FACET_WORKER_GROUP`) and the row read (`search_media`, `_PAGE_WORKER_GROUP`) are independent queries on separate workers with independent failure state (`page_failure`/`facet_failure`, each cleared by its own success). The whole-list gate is **already** scoped (task-31635 item 6 / task-31960): `_gate_failed_action` fires only on `failure AND list_unselectable` (= failure with no retained rows), Select gates on `rendered_count == 0`, Trash never gates. So a facet-only failure over a healthy page leaves Export/Review/Select/Trash live. AC#1 was therefore already satisfied by shipped code; the residue was the missing pin, added as `test_facet_only_failure_leaves_the_row_supported_actions_live`. **AC#1 text not amended** (branch 2 was not taken).

Tests (TDD, red→green): `Tests/UI/test_library_media_side_by_side.py::test_bulk_undo_counts_a_non_mapping_restore_as_a_still_failed_id` (Part A, + `NonMappingRestoreScopeService`); `Tests/UI/test_library_media_render_fixes.py::{test_repeated_media_load_failure_names_the_reopen_recovery, test_facet_only_failure_leaves_the_row_supported_actions_live}` (Parts B, C). Regression: the three affected files ran whole → 216 passed. Inventory unchanged; preflight passes.

Modified files: `tldw_chatbook/UI/Screens/library_screen.py`, `tldw_chatbook/UI/Library_Modules/library_media_browse_controller.py`, `Tests/UI/test_library_media_side_by_side.py`, `Tests/UI/test_library_media_render_fixes.py`, `Docs/User_Guide/library/media-and-conversations.md`.

## Renumbering provenance

Filed as TASK-31978 during critique #6's fix wave; renumbered to TASK-31982 because a concurrent session landed its own TASK-31978 on dev first (2026-08-21 owner rule, TASK-19601: older arrival keeps the id). No other task references this one.
<!-- SECTION:NOTES:END -->
