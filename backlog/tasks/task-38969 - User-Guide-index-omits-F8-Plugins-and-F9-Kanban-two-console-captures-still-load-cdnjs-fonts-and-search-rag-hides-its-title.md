---
id: TASK-38969
title: "User Guide index omits F8 Plugins and F9 Kanban two console captures still load cdnjs fonts and search-rag hides its title"
status: To Do
created_date: 2026-11-06 01:21
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
dependencies: []
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
- [ ] #1 The index's nav table lists F8 Plugins and F9 Kanban with working links, and every F-key list on the page (walk paragraph, hotkey note, no-borrow paragraph, keyboard table) matches SHELL_DESTINATION_SHORTCUTS; a Docs test compares the nav table with the map and is red on dev
- [ ] #2 No SVG under Docs/User_Guide contains a cdnjs.cloudflare.com URL and scripts/regen_approval_card_svg.py writes a stripped SVG; a Docs test is red on dev and names the two files
- [ ] #3 search-rag.svg shows the "Search / RAG" panel title and at least one result in a single 235x52 capture
<!-- AC:END -->
