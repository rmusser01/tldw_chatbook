---
id: TASK-41724
title: "The timer-path static update inventory is red on dev for project_index_binding_card.py"
status: To Do
assignee: []
created_date: 2026-11-18 06:57
updated_date: 2026-11-18 06:57
labels:
- follow-up
- tests
- architecture
priority: high
references:
- Tests/Architecture/test_timer_path_static_update_inventory.py
- tldw_chatbook/Widgets/Settings_Widgets/project_index_binding_card.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by several round-16 implementers and re-run by the coordinator (2026-11-18): `test_no_timer_path_update_defaults_to_layout_true` and `test_clock_root_set_is_pinned` fail on a `set_interval` root (`_poll`) in `project_index_binding_card.py`, at dev 45b1949ca3 and at dev 97f8c58c0c. No round-16 branch touches that file. The Architecture guard step runs these on every PR.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The `_poll` timer root either updates without `layout=True` or is pinned in the inventory with its reason, and both nodes pass
- [ ] #2 The commit that made them red is named in the notes
<!-- AC:END -->
