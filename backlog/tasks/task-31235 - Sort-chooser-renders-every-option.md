---
id: TASK-31235
title: Sort chooser renders every option
status: Done
assignee: []
created_date: '2026-09-04 01:50'
updated_date: '2026-09-04 04:21'
labels:
  - library
  - media-ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 P1: the sort chooser renders "✓ Newest  Oldest  Title A-" in the ~38-col items pane — "Title A-Z" is truncated and "Title Z-A" is entirely invisible with no overflow cue, yet a keyboard user can still arrow to and select the unrendered option. Recognition-over-recall failure: an option you can't see doesn't exist. Verified: the sort chooser is a horizontal compose_library_choice_strip (library_media_canvas.py:690-701) while the sibling type chooser is a vertical OptionList (line 681) that fits fine.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All four MEDIA_SORT_CHOICES options are fully visible when the sort chooser is open at the shell's narrow pane width
- [x] #2 The active option keeps its ✓ marker and keyboard selection works over the visible list
- [x] #3 A test pins that every sort option's label renders (painted-text or equivalent, not just presence in the DOM)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: painted-text test that every MEDIA_SORT_CHOICES label renders at the shell's narrow pane width
2. GREEN: replace the horizontal choice strip with the type chooser's vertical OptionList twin; move the handler to OptionList.OptionSelected
3. Suppress the app-tier *:focus outline for the new id (31221 family) and pin it
4. Live tmux verify at 200x52
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Sort chooser is now the type chooser's vertical OptionList twin (library_media_canvas.py): all four MEDIA_SORT_CHOICES render with ✓ on the active option, pre-highlighted. Handler moved from the strip's Button.Pressed to OptionList.OptionSelected with identical validation (allow-list derived from MEDIA_SORT_CHOICES) and close-on-same-value semantics; open-focus rides _sync_library_canvas's then hook because a bare call_after_refresh has no ordering against the canvas-scoped recompose (test caught it). The app-global *:focus outline overwrote 'Newest' and 'Title Z-A' at option-count height — suppressed at app tier for the new id and pinned with a painted-text test (region assertions are blind to paint-over). Live-verified at 200x52: four options vertical, picking 'Title A-Z' re-sorts and closes. Shipped in PR #2367 with task-31237.
<!-- SECTION:NOTES:END -->
