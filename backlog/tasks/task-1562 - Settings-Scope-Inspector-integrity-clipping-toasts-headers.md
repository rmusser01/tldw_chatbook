---
id: TASK-1562
title: 'Settings: Scope Inspector integrity — clipping, toast occlusion, header variants'
status: Done
assignee: []
created_date: '2026-07-31 02:00'
labels: [settings, ux, P1]
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique finding (P1): the pane that carries recovery/boundary copy loses
text silently. Evidence: rail bottom clips mid-sentence with no scroll
indicator ("Saves apply to your local", "Boundary Library owns: indexing,
query,", "...Nothing is sent to"); on Console Behavior the rail TOP shows a
floating clipped fragment ("sends"); toasts stack over the inspector's
"Owns:" list (Image Gen); Providers & Models and Storage drop the "Scope
Inspector" header + selected-category block entirely while all other
categories show them; raw config keys wrap mid-token ("api_k ey").
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Inspector content never clips silently: scrollable with a visible indicator, or copy budgeted to the column.
- [x] #2 Toasts do not occlude the inspector (docked to the footer strip or offset).
- [x] #3 All categories share the same inspector header structure.
- [x] #4 Config keys/paths wrap at separators, not mid-token.
<!-- AC:END -->

## Implementation Plan

1. Fixed header + scrollable body split with a visible scrollbar.
2. Owns: keys one per line (no mid-token wraps).
3. Offset toasts off the inspector column.

## Implementation Notes

- `#settings-impact-pane` is now a Vertical wrapper (explicit 100% height -- under the real CSS bundle the pane class sizes a scroll container, and the 1fr body collapsed in the styled harness; the plain harness could not reproduce it) containing the pinned header and `#settings-impact-pane-body` (VerticalScroll, scrollbar_size 1). Bottom clipping now shows a scrollbar; the floating top fragment ("sends") is gone.
- The "header variants" finding (Providers/Storage 'missing' the Scope Inspector header) was a SCROLL STATE, not composition -- the field-guide auto-scroll had pushed the header off-view; pinning fixes it structurally. Two tests updated to the new structure; one grew its harness height (the pinned header consumes rows the old single-scroll pane did not).
- Owns: config keys render one per line.
- Toasts: `ToastRack` align/margin overrides were empirically inert (its built-in right-bottom anchoring resisted app-tier CSS); a `Toast { margin-right: 46 }` rule reliably offsets the toast left of the inspector column -- verified by screenshot.
