---
id: TASK-32103
title: >-
  Library Collections rail count: duplicate page-1 read per snapshot, re-entry
  scope window, serial deadline, stale authority total
status: Done
assignee:
  - '@claude'
created_date: '2026-09-08 22:42'
updated_date: '2026-09-10 15:48'
labels:
  - library
  - collections
  - critique-8
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the task-32057 reviews (PR #2525): `get_library_user_content_evidence` already reads `list_page(page=1).total` in the same snapshot pass as the new count prefetch (two HTTP round trips per snapshot in server mode, only one behind the deadline); re-entering a previously scoped Collections canvas (scope persists, page reset by unmount) paints the unfiltered prefetch for one load window although the guide says it no longer flashes; the count read is awaited serially before the gather (worst case 5 s + 5 s) and a count timeout degrades silently; after `deactivate()` without re-adopt the controller-state path still paints the previous authority's `exact_total`. Rider from the critique-8 fix wave reviews (plan Docs/superpowers/plans/2026-09-08-library-crit8-wave.md; wave PRs #2519 #2523 #2524 #2525 #2528 #2531). The serial-deadline half was fixed in PR #2525's second review round: the count read is now started with the snapshot gather instead of awaited ahead of it, so the two deadlines overlap. The duplicate page-1 read, the re-entry scope window, the silent count timeout and the stale-authority `exact_total` remain open here.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One page-1 read per snapshot pass feeds both the evidence owner and the rail count
- [x] #2 The rail never paints an unfiltered total while a scoped page is pending, on every path
- [x] #3 A count timeout is visible (deadline sentence or Retry), and the previous authority's total is never painted
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Coalesce the duplicate page-1 read inside `CollectionsCaptureScopeService.list_page`: an unfiltered page-1 request for the active authority shares one in-flight task (shielded, so one caller's deadline cannot cancel the other's read), and `get_library_user_content_evidence` routes through `list_page` instead of its own `_invoke`.
2. Move the rail's count decision into one `_library_collections_rail_count()` helper: the unfiltered prefetch answers ONLY while the canvas scope is unfiltered (so re-entry into a scoped canvas, whose `page` is reset by `unmount()` while `active_scope` persists, paints no number), and the controller-state path returns None when the page belongs to an authority that is no longer active.
3. A failed/timed-out count read is visible: a `collections_count_unavailable` flag paints " (—)" on the row instead of nothing, and the rail Details block gains the deadline sentence.
4. Fix the `collections.md` stamp that overstates 'no longer flashes'.
TDD in `Tests/UI/test_library_crit8_collections_row.py` (this file is the 32057 pin set; no other session owns Collections) plus a service-level dedup test.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Four fixes, one of them a shared read.

- **One page-1 read per pass** (`Library/collections_capture_service.py`).
  New `CollectionsCaptureScopeService.read_unfiltered_first_page()`: the
  rail's count prefetch and `get_library_user_content_evidence` now share
  one in-flight task instead of issuing the same unfiltered page-1 request
  twice (two HTTP round trips in server mode for one number). Only the
  IN-FLIGHT read is shared -- a settled result is never replayed, so it
  cannot go stale -- and awaiters `asyncio.shield` it so the count read's
  5 s deadline cannot cancel the evidence read. Deliberately NOT folded
  into `list_page`: the canvas's own page reads keep their unshared path
  and their `page_snapshot` side effect (an early draft that coalesced
  inside `list_page` also put an extra loop hop in front of the canvas's
  first page load).
- **One rail-count decision** (`_library_collections_rail_count()` in
  `library_screen.py`, replacing the inline branch in
  `_build_library_shell_input`). The unfiltered prefetch answers ONLY while
  the canvas scope is unfiltered, which closes the re-entry window:
  `unmount()` resets `page` while the screen-owned `active_scope`
  persists, so returning to a scoped list landed back on the
  never-loaded branch and painted the whole-library total. And a retained
  page whose authority is gone (`deactivate()` without re-adopt) is no
  longer painted at all -- it was the previous authority's number under the
  new one's name.
- **A failed/timed-out count is visible**: `_library_collections_count_unavailable`
  (set only by the FAILURE branch, never by "no authority") paints
  " (—)" on the row via the new `LibraryShellInput.collections_count_unavailable`,
  and the rail's Details block gains the deadline sentence. A silently
  missing number was indistinguishable from Search / RAG, whose count is
  off by design.
- **Guide**: `collections.md`'s task-32057 stamp claimed the mid-load flash
  was gone; it was not, on the re-entry path. The old stamp now says so and
  the new one records what actually closed it.

Trade-off: the shared read is scoped to the unfiltered page-1 question
only. A general `list_page` cache would have needed invalidation; this
needs none because nothing survives the read.

Tests: `Tests/UI/test_library_crit8_collections_row.py` (4 new pins:
single read, re-entry, departed authority, visible timeout). Note
`test_rail_count_never_falls_back_to_the_unfiltered_total_after_a_visit`
is flaky under this host's load -- it failed 5/5 on an UNPATCHED origin/dev
checkout during this task and passes in a quieter run, same class as
task-32105.

Files: `tldw_chatbook/Library/collections_capture_service.py`,
`tldw_chatbook/Library/library_shell_state.py`,
`tldw_chatbook/UI/Screens/library_screen.py`,
`Tests/UI/test_library_crit8_collections_row.py`,
`Docs/User_Guide/library/collections.md`.
<!-- SECTION:NOTES:END -->
