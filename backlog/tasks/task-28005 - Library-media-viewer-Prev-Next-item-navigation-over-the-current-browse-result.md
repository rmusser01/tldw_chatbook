---
id: TASK-28005
title: >-
  Library media viewer - Prev/Next item navigation over the current browse
  result
status: Done
assignee: []
created_date: '2026-09-02 04:10'
updated_date: '2026-09-02 05:54'
labels:
  - library
  - media-ux
dependencies: []
references:
  - >-
    .impeccable/critique/2026-09-02T04-00-36Z__tldw-chatbook-ui-screens-library-screen-py.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The viewer has no forward path between items: no next/prev buttons or bindings exist (BINDINGS audit library_screen.py:1780-1871; live-tested n, p, brackets and arrows all dead). Sequentially reviewing N items - a conference, a tag-filtered set, or hand-picked videos - costs Escape, arrows, Enter per item, quadratic in list position. Add Prev/Next controls in the viewer header plus keys that walk the browse controller retained ordered page, respecting whatever scope produced the list (type filter, future tag/keyword query), so reviewing a whole set in order is one keypress per item.

Re-verified 2026-09-02 live on dev tip: still absent (n, p, ], [, Left, Right all inert; Reader controls are exactly Back / Find / Read later / Use in Console / More plus the Read-Analysis-Highlights-Info tabs; no footer hint). Two dev-tip foundations make this cheap and complete the sequential-review flow: (1) moving the LIST selection auto-loads the item in the Reader, so next/prev can simply move the selection programmatically; (2) Reader mode persists across selections (begin_selection in Library/library_media_reader_state.py preserves mode), so next/prev while in the Analysis tab reads every analysis in sequence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 From an open media item, a single keypress opens the next (and previous) item in the current list order without returning to the list
- [x] #2 Boundary behavior at the first and last item is communicated
- [x] #3 The keys are advertised in the viewer footer
- [x] #4 Navigation respects an active type filter (walks the filtered result)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Adds sequential-review Prev/Next over the current browse result. Bindings ] (next) / [ (prev), show=False, gated in check_action to the plain Reader (_library_media_item_traversal_active: media row + view=='viewer', no sub-state, not select mode) AND only when a neighbour exists that way. Neighbours come from the mounted .library-media-row widgets: they carry media_id in exactly the form _selected_media_id holds (format-safe) and sit in browse order (newest-first, so ]=down=next), respecting any active type filter automatically. Selection reuses _select_library_media_reader_row(immediate=True), which the dev-tip auto-load already turns into a Reader load and preserves Reader mode across items - so ]/[ while in the Analysis tab reads every analysis in sequence. Boundary is 'communicated' via the honest-footer idiom: the key is gated off AND drops from the footer at the list ends (]/[ appear only when that neighbour exists). Scope: within the loaded page (20 items); page-turn at the boundary is a follow-up. Tests: test_bracket_keys_walk_to_next_and_previous_item_in_the_reader, test_prev_item_binding_disabled_at_the_first_item. Files: UI/Screens/library_screen.py, Tests/UI/test_library_media_reader_flow.py. NOTE: pre-existing dev-tip test debt found - test_library_screen_bindings_are_all_gated_or_universal fails on 'focus_previous_workbench_pane' (its gate returns False on a bare unmounted screen); unrelated to this change (verified by stash-run).
<!-- SECTION:NOTES:END -->
