---
id: TASK-31271
title: Media footer promises the wrong key at four seams
status: Done
assignee: []
created_date: '2026-09-04 13:54'
updated_date: '2026-09-04 19:58'
labels:
  - library
  - media-ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 P1: (a) after Escape closes Find the footer still says `esc close find` because `_library_media_escape_label` reads the DOM before the recompose lands (A cap 08/23, B cap_21); (b) right after `s` the footer promises `space toggle selection` while Space is a no-op unless a row is focused and, with focus on the pane grip, collapses the Library pane (A cap 31→32, B cap_69) — `_toggle_library_media_select_mode` never moves focus to a row; (c) on the last item of a set the chip still reads `] next in set` although that ] is the completion gesture (B cap_50); (d) l/c/t are real Reader keys with show=False and `t` arms delete unadvertised (B cap_97). The footer is the surface's main trust instrument.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After Escape closes Find, the footer no longer shows `esc close find`
- [x] #2 Pressing s focuses the first media row so the Space chip is true immediately; Space never falls through to the pane grip
- [x] #3 On the last item of a review set the ] chip names the completion gesture (e.g. `] finish review`) instead of `next in set`
- [x] #4 l, c and t are advertised in the Reader footer set (t at minimum, since it arms delete)
- [x] #5 Each seam has a pinning test
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED per seam (painted footer for the stale esc chip, s→row focus + Space, grip fall-through, last-item chip, l/c/t)
2. GREEN: escape label + action read one _library_media_find_state() seam; select-mode entry focuses a row via the sync then-hook; Space claimed (priority binding) only on media rows and media pane grips; _review_footer_entries(progress, at_last); l/c/t chips gated by their bindings; footer re-registered at the detail-settle seam
3. Live tmux 235x52
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Four seams, one truth source each. (a) The Escape chip and Escape action now derive the Find state from one seam (_library_media_find_state: find_open/content_query plus the mounted bar), so neither lags the DOM by a refresh. (b) s enters select mode and focuses the loaded/first row via the canvas sync's then-hook; Space is a priority binding whose check_action is True only in select mode with focus on a media row or a media pane grip (library-media-pane-grip + the LIBRARY_ROW_BROWSE_MEDIA guard — a first cut matched the shared adaptive-grip class and would have swallowed Space on five other canvases), so the pane never collapses and buttons (Enter-bound) are untouched. (c) _review_footer_entries(progress, at_last=True) names the last item's ] as 'finish review'. (d) l/c/t are advertised in the Reader footer (read later / use in Console / trash), each gated by its own binding; live verification found the chips only appeared after an unrelated F6 because the footer registered at open time was never re-derived when the detail landed — fixed at the settle seam and pinned with a painted-footer assertion. Deferred: _active_review_set_progress was later deleted in the final wave; 'space toggle selection' is not literally true when focus sits on a select-mode button (Space no-ops there, as before).
<!-- SECTION:NOTES:END -->
