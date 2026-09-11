---
id: TASK-32347
title: >-
  Library Media rows: the age suffix '· 10m' reads as a duration on audio and
  video
status: Done
assignee: []
created_date: '2026-09-11 06:14'
updated_date: '2026-09-11 08:00'
labels:
  - library
  - media
  - ux
  - critique-10
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every media row ends '· 10m' (audio · 10m, video · 10m, pdf · 10m). The value is the item's age (the same row read 11m and 14m later) but on audio/video rows it reads as length (A caps 36/39/47). The secondary-line format is pinned by test_media_secondary_fallback_when_no_type_no_age, so this is a pinned design decision the critique disagrees with. Evidence: critique #10 snapshot (.impeccable/critique/2026-09-11T06-13-11Z__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 1f3184655b, captures under the session scratchpad crit10/A and crit10/B).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Age is labelled ('added 10m ago') or replaced by type-aware metadata (duration for audio/video, pages for pdf/ebook, words for article/document) with the date on Info
- [x] #2 The pin is updated to the new format, not loosened
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. New test for media_added_age_copy in Tests/UI/test_library_crit10_media_rows.py
2. Add media_added_age_copy beside media_trash_age_copy in library_media_state.py
3. Swap the two browse call sites (browse state + legacy build_library_media_state); leave the Trash row alone
4. Update the four pinned secondary strings to the labelled form
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added media_added_age_copy() beside media_trash_age_copy in library_media_state.py and swapped the two BROWSE call sites (build_library_media_browse_state and the legacy build_library_media_state) to it; the Trash row keeps its own 'trashed <age>' label untouched. format_console_relative_age returns the bare word 'now' under a minute, which no 'N ago' phrasing survives, so that case reads 'added just now'.

AC#2 (the pin updated, not loosened): the four strings that actually constrain the format -- test_library_media_state.py:305-306 and :529/:534 -- now read 'pdf · added 3m ago' / 'video · added 2h ago' / 'video · added 3m ago' / 'pdf · added 2h ago', with task-32347 named in both docstrings. test_media_secondary_fallback_when_no_type_no_age, which the task names, turned out to pin only the NO-TYPE fallback ('media') and needed no change. test_library_media_trash_state.py:490-491 is byte-identical.

Three painted-row pins in Tests/UI/test_library_media_render_fixes.py were re-measured against the real output rather than softened: the labelled age costs ~10 cells on every secondary line. KNOWN COST, recorded in those pins: at the Items pane's narrow (100x30) width a keyword row now clips its term where it used to paint it whole, and at the 36-cell floor a keyword row no longer reaches its 'keyword:' label at all ('article · added 2m ago · …'). Dropping ' ago' would buy 4 of those cells back; the AC's own example and the plan both specify the 'added N ago' wording, so that is a product call left open rather than taken here.

Live-verified at 235x52 and 100x30 on the seeded profile: rows read 'audio · added 1h ago', 'pdf · added 1h ago', 'video · added 1h ago'.

Files: tldw_chatbook/Library/library_media_state.py; Tests/Library/test_library_media_state.py; Tests/UI/test_library_media_render_fixes.py; Tests/UI/test_library_crit10_media_rows.py (new); Docs/User_Guide/library/media-and-conversations.md.
## Fix round 1 + re-review round 1

Two copy rulings landed after the notes above were written, so read those as
the original round only:

1. **`added 3m ago` -> `added 3m`** (review round 1): the LABEL removes the
   duration reading, and the Trash list's own grammar (`trashed 3m`) is the
   house style. The four cells went straight to the narrow Items pane -- a
   keyword row paints its term again, and the 36-cell floor reaches into the
   `keyword:` label instead of ending before it. So the "product call left
   open" in the notes above was TAKEN.
2. **`added 3m` -> `updated 3m`** (re-review finding A): the value is the
   record's `last_modified`, not its ingest time -- the browse contract maps
   `updated_at` to `last_modified` and never projects `ingestion_date`, and
   saving an analysis writes it. "added" would have swapped one ambiguity for
   another, so the row now uses the same word the preview pane already uses
   for the same field ("Updated:"). The helper is `media_updated_age_copy`.

Current pins: `pdf · updated 3m`, `video · updated 2h`, `video · updated 3m`,
`pdf · updated 2h`; the helper returns `updated 10m` / `updated just now` /
`""`. Every painted pin in `test_library_media_render_fixes.py` was
re-measured at each step rather than hand-edited.
<!-- SECTION:NOTES:END -->
