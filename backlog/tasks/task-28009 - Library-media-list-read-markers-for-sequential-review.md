---
id: TASK-28009
title: Library media list - read markers for sequential review
status: Done
assignee:
  - '@claude'
created_date: '2026-09-02 04:10'
updated_date: '2026-09-06 23:45'
labels:
  - library
  - media-ux
dependencies: []
references:
  - >-
    .impeccable/critique/2026-09-02T04-00-36Z__tldw-chatbook-ui-screens-library-screen-py.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Nothing marks an item as opened or reviewed; a user reviewing a whole set keeps the which-ones-are-left ledger in their head across pager pages. Persist a lightweight viewed state per item and render it as a row glyph, so a sequential pass over a conference, a tag-filtered set, or a hand-picked collection has visible progress. The reading-scope service (read-it-later flag) is the persistence precedent.

Dev-tip foundations noted 2026-09-02: per-item reading POSITION is already persisted (library_media_reading_progress worker drain, library_screen.py ~42510), and the loaded row carries a "Loaded in Reader" tag - a read/reviewed marker can build on both precedents.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Marking an item done in the active review set (`m`, or advancing with `]`) marks it, and the list shows the mark after returning (AC amended at close to the approved task-31278 Option A v1 scope: `reviewed` is sourced from the active review set's done marks, not from opening an item)
- [x] #2 Marks survive app restart
- [x] #3 Unreviewed items are distinguishable at a glance across pages
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reuse task-28008's seven-key contract (`reviewed`), decorated by the screen from the active review set: True = in set and done, False = in set not done, None = outside a set / no set.
2. Row grammar: one-cell slot `✓` / `·` / space before the title; select mode's ☑/☐ replaces the slot; re-decorate on every done mark (`m`, `]`), pinned as painted text on the mounted screen.
3. Marks persist with the review set (collections DB) → survive restart; decoration is per page from the set → distinguishable across pages.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`reviewed` is decorated at one seam (`_decorate_library_media_reviewed`) from `get_active_review_set()` (84 µs per canvas build; no cached set exists), fails open on storage error; the walker/banner's sync seam re-decorates and the viewer-scoped sync patches the rows in place, so `m` and the final `]` repaint the slot (painted pin RED with the trigger disabled → GREEN). The current row's `▸` is demoted to its state while a set is active; it keeps the selected styling (bold/underline/background — CSS read). Marks are the review set's done marks, persisted in the collections DB (AC#2). Trade-off: opening alone does not mark (v1 scope per task-31278; a read-marker independent of review sets is a later phase). User Guide paragraph + stamp.
Riders (to be filed in the wave-5 close-out docs PR): decoration cost is O(active-set size) via get_active_review_set (cache the done-map on the screen if it ever matters); the review-set enumeration loop runs the keyword probe and discards it; the preview pane was not extended with the marker (v1 omission).
<!-- SECTION:NOTES:END -->
