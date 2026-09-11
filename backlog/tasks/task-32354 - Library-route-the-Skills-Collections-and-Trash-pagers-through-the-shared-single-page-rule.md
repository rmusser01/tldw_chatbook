---
id: TASK-32354
title: >-
  Library: route the Skills, Collections and Trash pagers through the shared
  single-page rule
status: Done
assignee: []
created_date: '2026-09-11 06:17'
updated_date: '2026-09-11 07:42'
labels:
  - library
  - ux
  - critique-10
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
At 100x30 the Skills canvas spends four lines on '1-2 of 2 / Page 1 of 1 / Already on the first page. / ○ Previous ○ Next' for two items; at 60x24 that is 4 of 18 rows and pushes the second skill off-screen (A caps 60/61). PROVEN: library_pager_layout (task-28016/32104) is imported by the media, conversations and prompts canvases only; library_skills_canvas.py composes its own _compose_pager. Pinned by test_skills_canvas_renders_exact_pager_and_source_wide_trust_count — a design decision to reverse deliberately. Evidence: critique #10 snapshot (.impeccable/critique/2026-09-11T06-13-11Z__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 1f3184655b, captures under the session scratchpad crit10/A and crit10/B).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Skills, Collections and Trash render no pager chrome when everything fits on one page, like Media/Conversations/Prompts
- [x] #2 The existing pin is updated to the shared rule
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Split the Skills pager pin into a suppressed-single-page test and a full-two-page test (deliberate reversal).
2. Route library_skills_canvas._compose_pager through library_pager_layout.
3. Add simple_library_pager_display to library_pager_state; route the Collections pager through it.
4. Route the Trash pager through library_pager_layout (height 1 when suppressed).
5. Live-verify at 235x52 / 100x30 / 60x24; docs + stamps.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Skills, Collections and the media Trash now render their pagers through `library_pager_layout`, so the single-page rule (task-28016/31237/32104) finally holds for every Library list.

**Skills** (`library_skills_canvas.py::_compose_pager`) hand-rolled its boundary reasons and yielded the range, the page copy and both controls unconditionally. It now renders what the layout returns: the range Static carries the joined `status_parts`, so the separate `#library-skills-page` Static is gone and the page copy rides the range line on two pages. Live at 60x24 this gives back 3 of 18 usable rows and the second of two skills keeps its description line (compare assessor A's cap 61 with `crit10/wave/pagers/caps/09-skills-60x24.txt`).

**Collections** could not use `build_library_pager_display` — it raises when `row_count` disagrees with `applied_page`/`total`, and this canvas's service can return a short page — so `simple_library_pager_display` was added to `library_pager_state.py` for a source that pages itself. It carries the same copy and the same boundary reasons, so such a source still goes through the one shared rule. `CAPTURE_PAGE_SIZE` (already in `collections_capture_models`) replaced the three hand-written `20`s.

**Trash** keeps its fixed-height pager block, at 1 row instead of 2 when the controls are suppressed.

## The pin this reverses (AC#2)

`test_skills_canvas_renders_exact_pager_and_source_wide_trust_count` pinned the four-line Skills pager verbatim. It is split, not deleted or loosened, into `test_skills_canvas_suppresses_single_page_chrome_and_keeps_the_trust_count` (the suppressed shape, keeping the source-wide `blocked_total` assertions) and `test_skills_canvas_renders_the_full_pager_when_a_second_page_exists` (every part back at 21 skills, one more than `DEFAULT_SKILL_BROWSE_PAGE_SIZE`). Four settle loops that polled `#library-skills-page` for "Page N of 3" now poll `#library-skills-range`, which carries that copy.

## Two deliberate narrowings

1. **Collections suppresses only when `bounded`** (paging enabled AND an exact total). Neither direction moving because paging is PAUSED — a stale page withholds totals — is not "this is the only page", and `test_items_keep_capture_controls_rows_and_stale_recovery_reachable` pins the paused controls. They stay, disabled, with their old reason. Pinned from this side too by `test_collections_keeps_paused_controls_when_the_page_is_stale`.
2. **Trash keeps `#library-media-trash-page` (empty) whenever the controls render**, because the initial-error posture pins an empty page Static in `test_media_trash_geometry_four_sizes_paints_all_fixed_controls`. No new boundary-reason row was added to Trash either: that would ADD a line at 60x24, the opposite of the finding, and no AC asks for it.

## Fix round 1 (task review, 2026-09-11)

- The pin split had lost the assertion that the Skills header counts the
  SOURCE (`state.pager.title_count`) rather than the mounted rows: the
  single-page test's `total` and row count are both 2, so a regression to
  `len(rows)` would have passed. The two-page test (21 total, 2 mounted
  rows) now asserts `"Skills (21)"`, which only `title_count` can produce.
- `Docs/User_Guide/library/media-and-conversations.md` — the Trash's own
  page — now documents the single-page pager too, not just `library.md`.
- `Docs/User_Guide/library/skills.md`'s trust-header row had the Python
  string's `\"` escapes copied into the Markdown table, where they render as
  visible backslashes; it now uses curly quotes. The product string is
  unchanged.

## Files

- `tldw_chatbook/Library/library_pager_state.py` — `simple_library_pager_display`.
- `tldw_chatbook/Widgets/Library/library_skills_canvas.py`, `library_collections_capture_reader.py`, `library_media_trash_canvas.py`.
- `Tests/Library/test_library_pager_state.py` (+3), `Tests/UI/test_library_skills_canvas.py` (pin split), `Tests/UI/test_library_crit10_pagers.py` (new, 8 tests).
- `Docs/User_Guide/library.md`, `library/collections.md`, `library/skills.md`.
<!-- SECTION:NOTES:END -->
