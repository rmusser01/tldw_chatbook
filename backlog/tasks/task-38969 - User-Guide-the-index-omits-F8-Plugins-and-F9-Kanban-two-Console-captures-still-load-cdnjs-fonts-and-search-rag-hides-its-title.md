---
id: TASK-38969
title: "User Guide: the index omits F8 Plugins and F9 Kanban, two Console captures still load cdnjs fonts, and search-rag hides its title"
status: Done
assignee:
- '@claude'
created_date: 2026-11-06 01:21
updated_date: 2026-11-10 18:28
labels:
- follow-up
- docs
- screenshots
priority: low
references:
- Docs/User_Guide/index.md
- Docs/User_Guide/kanban.md
- tldw_chatbook/UI/Navigation/shell_destinations.py
- Docs/User_Guide/images/console/approval-card.svg
- Docs/User_Guide/images/console/context-modal.svg
- scripts/regen_approval_card_svg.py
- Docs/User_Guide/images/library/search-rag.svg
- Docs/User_Guide/_template.md
dependencies:
- TASK-36521
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down. Checked on dev 29592ecc9b and on the TASK-36521 branch, 2026-11-05. That branch recaptures the nav-bar screenshots and merges first, so the paths and lines below are its post-merge state. TASK-36521's notes raise all three items.

**The index stops at F7.**
- The shipped nav bar maps `"plugins": "f8"` and `"kanban": "f9"` in `SHELL_DESTINATION_SHORTCUTS` (`tldw_chatbook/UI/Navigation/shell_destinations.py:210-227`). `app.py:963-973` binds every destination. The recaptured `images/library/search-rag.svg` shows `… F7 Meetings F8 Plugins F9 Kanban` on its second row.
- `Docs/User_Guide/index.md` still stops at F7 in five places:
  - the nav table (`:34-49`);
  - the walk paragraph, which lists "F2, F3, F4, F5, F7" (`:51-56`);
  - the hotkey note, "F2–F5 / F7 for the last five" (`:79-80`);
  - the no-borrow paragraph, "F2–F5 / F7 tail" (`:88-89`);
  - the keyboard table, "F2 / F3 / F4 / F5 / F7" (`:101`).
- `kanban.md` exists (TASK-18934), but the index never links it. There is no Plugins page under `Docs/User_Guide`. No Docs test compares the index with the shortcut map.

**Two captures still load cdnjs fonts.**
- Of the 20 User Guide SVGs, `images/console/approval-card.svg` and `images/console/context-modal.svg` are the only two that still carry cdnjs font URLs, four lines each.
- `_template.md:119-129` says docs images must have them stripped. `scripts/regen_approval_card_svg.py:75` calls `save_screenshot` without stripping, so the next regeneration puts the URLs back.
- The 2,058 other Docs SVGs with cdnjs URLs are frozen QA, review and spec evidence and are out of scope.

**search-rag is scrolled.**
- In `images/library/search-rag.svg`, the main pane's first row (row 7) is the mode toggle ("mode: ✓ Search ⇄ RAG Answer"), with a scrollbar mark at its right end.
- The panel's first child is the "Search / RAG" title Static (`tldw_chatbook/Widgets/Library/library_search_rag_panel.py:303-308`). The capture is scrolled past it, so the title is out of view.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The index's nav table lists F8 Plugins and F9 Kanban with working links, and every F-key list on the page (walk paragraph, hotkey note, no-borrow paragraph, keyboard table) matches SHELL_DESTINATION_SHORTCUTS; a Docs test compares the nav table with the map and is red on dev
- [x] #2 No SVG under Docs/User_Guide contains a cdnjs.cloudflare.com URL and scripts/regen_approval_card_svg.py writes a stripped SVG; a Docs test is red on dev and names the two files
- [x] #3 search-rag.svg shows the "Search / RAG" panel title and at least one result in a single 235x52 capture
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a Docs test that parses index.md's nav table and every F-key list and compares them with SHELL_DESTINATION_SHORTCUTS (red on dev: no F8/F9).
2. Add F8 Plugins (new stub plugins.md) and F9 Kanban (kanban.md) to index.md: nav table, walk paragraph, hotkey note, no-borrow paragraph, keyboard table.
3. Add a Docs test that no SVG under Docs/User_Guide carries a cdnjs URL (red on dev, names approval-card.svg and context-modal.svg); strip both files; make scripts/regen_approval_card_svg.py write a stripped SVG through one shared strip helper, pinned in Tests/Scripts.
4. Recapture search-rag.svg at 235x52 with the checked-in driver from TASK-38970 (scrolled to the panel title, one result visible).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added F8 Plugins and F9 Kanban to the index, stripped the last two cdnjs captures through one shared helper, and recaptured search-rag with the checked-in driver from TASK-38970.

- **AC#1** `index.md`: nav-table rows `F8 | [Plugins](plugins.md) 🚧` and `F9 | [Kanban](kanban.md)`. There was no Plugins page, so `Docs/User_Guide/plugins.md` is a new stub (orientation, the 1/2/3 views, `/`, Escape, the action rows, all read from `plugins_screen.py`). The walk paragraph, the hotkey note (`F2–F5 / F7–F9 for the last seven`), the no-borrow paragraph and the keyboard table now cover F2–F5 and F7–F9. New `Tests/Docs/test_user_guide_contract.py`: `test_index_nav_table_matches_the_shell_shortcut_map` compares the table with `SHELL_DESTINATION_SHORTCUTS` plus the Ctrl+6/Ctrl+9 compatibility keys, checks each row names the destination label and links an existing page. Red on dev: `missing ['f8', 'f9']`. `test_every_f_key_list_in_the_guide_names_the_whole_f_row` (index-only in the first version) expands every F-key run on the page and compares it with the map's F-keys. Red on dev: `'F2, F3, F4, F5, F7' should cover [... 'F8', 'F9']`.
- **AC#2** Both SVGs lose only their four cdnjs `url(...)` lines (the `local(...)` source stays). `strip_remote_fonts`/`write_guide_svg` live in `scripts/capture_user_guide.py`; `scripts/regen_approval_card_svg.py` now writes through `write_guide_svg(out, app.export_screenshot())`. `test_user_guide_svgs_load_no_remote_fonts` was red on dev naming `images/console/approval-card.svg` and `images/console/context-modal.svg`. `Tests/Scripts/test_regen_approval_card_svg.py::test_written_card_loads_no_remote_fonts` (real write into tmp_path) was red with dev's script swapped back in (`'cdnjs.cloudflare.com' is contained here`), green after; the file's three other tests now patch `write_guide_svg` instead of `App.save_screenshot` (6 passed).
- **AC#3** `scripts/capture_user_guide.py search-rag` (real `TldwCli`, 235x52): it makes a note through Library ▸ Create ▸ New note, types a query on Search / RAG and presses Enter, then keeps `#library-rag-panel-title` in view. The capture paints "Search / RAG" on row 7 and the card "1. Release review" with its snippet; 2885 x 1318.8 viewBox like the other 235x52 captures. The result is now a note, not a conversation; no page prose described the old result.
- Files: `Docs/User_Guide/index.md`, `Docs/User_Guide/plugins.md` (new), `Docs/User_Guide/images/console/{approval-card,context-modal}.svg`, `Docs/User_Guide/images/library/search-rag.svg`, `scripts/regen_approval_card_svg.py`, `Tests/Docs/test_user_guide_contract.py` (new), `Tests/Scripts/test_regen_approval_card_svg.py`.
- **Review round 1 (r13h-rev1).**
  - M-3: `settings.md` still listed "F2, F3, F4, F5, then F7" and called Settings "the thirteenth of fifteen"; it is the eleventh of the fifteen nav bar destinations (`SHELL_DESTINATION_ORDER`), and the paragraph now names F7–F9 and the Ctrl+6/Ctrl+9 compatibility keys. `home.md` read "F2–F5/F7" and now reads "F2–F5/F7–F9". The F-key test now scans every User Guide page. It was red on those two pages first: `home.md: 'F2–F5/F7'`, `settings.md: 'F2, F3, F4, F5'` and settings' quoted list.
  - M-6: the test now also compares quoted nav-label runs ("F2 Lab", "F3 Logs", …) with the map and reads ASCII-hyphen ranges. "Shift+F6" never starts a run, and runs of fewer than three keys ("F1, F6", finding ids like "F1/F2") are skipped. Red on a temporary edit that dropped "F9 Kanban" from the index's quoted list and changed home.md to "F2-F5/F7": both were named. Reverted, 5 passed.
  - Reviewer recommendation: `plugins.md`'s keys section is now the template's "Keyboard & commands" table; the action rows moved to "What this screen is for".
- **Review round 2 (r13h-rev2).**
  - "then" is now a list separator in both the run regex and the range split. Same temporary edits, run first with the round-1 test and then with the new one: home.md set to "F2, F3, then F7" and settings.md to the correct "F2, F3, F4, F5, then F7, F8, F9". The round-1 test let home.md through (a 2-key run plus a lone F7) and wrongly flagged settings.md (`'F2, F3, F4, F5'`, `'F7, F8, F9'`). The new test names only `home.md: 'F2, F3, then F7'`. Reverted.
  - Two single-key F9 references meant Settings, which is F4 (F9 is Kanban). `settings.md` now reads "F4 → Console Behavior → Goal runs": the category label is `Console Behavior` (`settings_screen.py`) and its card composes `Goal runs` (`settings_goal_policy.py`). `console/context-and-rag.md` now reads "Press F4 → Workspaces" (the `Workspaces` category).
<!-- SECTION:NOTES:END -->
