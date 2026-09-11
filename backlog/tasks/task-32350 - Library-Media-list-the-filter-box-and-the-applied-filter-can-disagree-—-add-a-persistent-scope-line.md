---
id: TASK-32350
title: >-
  Library Media list: the filter box and the applied filter can disagree — add a
  persistent scope line
status: Done
assignee: []
created_date: '2026-09-11 06:15'
updated_date: '2026-09-11 08:00'
labels:
  - library
  - media
  - ux
  - critique-10
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The filter box keeps a draft while the list shows the last applied query and the count reads 'Media (9)' with nothing saying which filter is live (A caps 41/47; the draft-vs-applied model is deliberate, the missing scope line is not). Evidence: critique #10 snapshot (.impeccable/critique/2026-09-11T06-13-11Z__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 1f3184655b, captures under the session scratchpad crit10/A and crit10/B).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A scope line under the Media header states the applied filter, type and sort and the count as N of M, with a Clear action
- [x] #2 Clearing the applied filter also clears the box
- [x] #3 Pinned
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing UI tests for #library-media-scope-line and #library-media-scope-clear
2. scope_line/scope_clearable on LibraryMediaCanvasState, built inside build_library_media_browse_state from the APPLIED scope
3. unfiltered_total passed from library_media_controller
4. scope row in library_media_canvas compose + Clear routed to the existing clear path, which now also blanks the Input
5. TCSS rule + bundle rebuild
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The scope line is built inside build_library_media_browse_state from result.scope and result.total -- the APPLIED scope -- so the canvas has no way to render the filter Input's unsubmitted draft by accident. LibraryMediaCanvasState gained scope_line/scope_clearable (both defaulted, so build_library_media_state's callers are unaffected) and the builder gained a keyword-only unfiltered_total for the 'N of M' half.

The sort segment reuses library_choice_label with the MEDIA_SORT_CHOICES display label -- the identical call the '#library-media-sort' Button label makes -- so the line and the control cannot name the same sort two ways.

unfiltered_total comes from a new _library_media_unfiltered_total property on LibraryMediaController that reads the screen's _library_loaded / _library_lookup_error / _local_source_counts. DEVIATION worth knowing: the controller is a separate object wired with injected accessors, not a mixin, so the plan's literal 'self._local_source_counts' raised AttributeError at runtime (caught by the new UI test). Adding a proper accessor would mean editing the LibraryMediaController(...) construction block in library_screen.py, which is outside this branch's ownership, so the property reads self._screen directly and says why.

AC#2: handle_library_media_filter_clear now carries both '#library-media-filter-clear' and '#library-media-scope-clear' and blanks the Input before requesting. The scope line's Clear additionally drops the TYPE facet (_request_library_media_filter grew a clear_type keyword) -- without it the Clear rendered by a type-only scope was a dead button, since the query was already ''. The toolbar's 'Clear filter' still clears only the filter its label names, so neither control's copy lies.

LIVE-FOUND DEFECT, fixed: with the plan's 'width: auto' the scope Static pushed its own Clear Button off the Items pane edge the moment the Reader narrowed the list (235x52). The .library-media-scope-line rule takes 1fr with text-overflow: ellipsis instead; the squeeze lands on the Static and the compact Button keeps its natural width, so task-4023's zero-width-Button hazard is not in play.

Live-verified 235x52: 'Media · 11 of 11 · all types · sort: Newest' with no Clear, then 'Media · 2 of 11 · filter “notes” …   Clear' with the list showing 2 rows, then Clear -> box empty, list back to 11, Clear gone. Also checked at 100x30.

Files: tldw_chatbook/Library/library_media_state.py; tldw_chatbook/UI/Library_Modules/library_media_controller.py; tldw_chatbook/Widgets/Library/library_media_canvas.py; tldw_chatbook/css/components/_agentic_terminal.tcss (+ regenerated bundle); Tests/UI/test_library_crit10_media_rows.py; Docs/User_Guide/library/media-and-conversations.md.
## Fix round 1

Two review findings against the scope line:

- **`N of M` now respects `_local_source_total_known`** (finding 2). The
  screen keeps that flag because the snapshot's count is sometimes a lower
  bound, and every other consumer renders `5+`. The line drops its "of M"
  half rather than state a flat total the rail itself refuses to state.
  Pinned by `test_the_scope_line_says_nothing_it_cannot_stand_behind`.
- **The live-found off-screen Clear is pinned** (finding 3) by its painted
  region at 235x52 and 100x30 -- `press()` succeeds just as happily on a
  Button painted past the pane edge, which is the exact state the `1fr` +
  ellipsis CSS fix was made for. Verified the pin fails against the reverted
  `width: auto`.

Qodo then found a third: the no-op guard in `_request_library_media_filter`
compared only `requested_scope`, so after a FAILED browse the still-visible
Clear (derived from the applied result) could never retry. It now suppresses
only when the applied scope is at the target too, pinned by
`test_a_clear_whose_request_failed_can_be_pressed_again`.
<!-- SECTION:NOTES:END -->
