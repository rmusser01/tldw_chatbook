---
id: TASK-19046
title: >-
  Wire-or-retire CollectionsTagWindow — unmounted since Aug 2025, its whole
  keyword-event loop is dead
status: To Do
assignee: []
created_date: '2026-08-20 08:40'
labels:
  - ui
  - dead-code
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Review-confirmed during the wave-2 close-out and re-verified at dev
`1bf7f234e`: `Widgets/collections_tag_window.py::CollectionsTagWindow` (:128,
the media-DB keyword/tag manager) is constructed nowhere in production — a
definitive whole-tree grep finds only the class definition, lazy imports
inside `Event_Handlers/collections_tag_events.py` handlers, and tests. Its
mount was lost at commit `de367762a` (2025-08-02).

The entire loop behind it is therefore unreachable: the
KeywordRename/Merge/Delete events are posted only from inside the widget
itself (:523/:529/:542), so `app.py`'s dispatch (:11953-11960) and every
`collections_tag_events` handler can never fire — and those handlers
`query_one(CollectionsTagWindow)`, which would raise on the unmounted widget
even if an event somehow arrived. TASK-15471's collections keyword-delete
threading repair (Done) landed on this dead path — corpse-groundwork, exactly
the shape this queue item predicted.

Tests keeping the corpse green: `Tests/UI/test_bulk_selection_tooltips.py`,
`Tests/UI/test_tag_action_recovery_tooltips.py`, the CollectionsTagWindow
slice of `Tests/Widgets/test_reactive_default_aliasing.py`, and
`Tests/Event_Handlers/test_collections_tag_events.py`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 CollectionsTagWindow is either mounted and reachable from a live screen (with the event loop verified end-to-end) or retired — widget, events, handlers, the app.py dispatch block, and its tests handled together with provenance; per the owner ruling, prefer the durable option over speculative resurrection
- [ ] #2 No unreachable keyword-event dispatch remains in app.py
- [ ] #3 Targeted suites green; if retired, whole-tree grep for the removed names returns nothing
<!-- AC:END -->
