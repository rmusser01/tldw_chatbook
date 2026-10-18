---
id: TASK-35309
title: 'Bring four Library controllers back within their size ratchet rows'
status: To Do
assignee: []
created_date: '2026-10-17 09:11'
labels:
- follow-up
- library
- architecture
dependencies: []
references:
- Tests/Architecture/test_library_modules_size_ratchet.py
- backlog/docs/library-decomposition-recipe.md
- tldw_chatbook/UI/Library_Modules/library_artifacts_controller.py
- tldw_chatbook/UI/Library_Modules/library_export_controller.py
- tldw_chatbook/UI/Library_Modules/library_media_controller.py
- tldw_chatbook/UI/Library_Modules/library_notes_sync_controller.py
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Seen while checking PR #3144 (Phase 8); red on dev too. Run serially at origin/dev 111f8e48fc, Tests/Architecture/test_library_modules_size_ratchet.py fails 4 of its rows (`test_controller_does_not_grow_past_its_budget`, 47 other tests pass):
- library_artifacts_controller.py: 904 lines against 806 (+98); last grown by b651a99389 (2026-10-12, Dreams stories in Library).
- library_export_controller.py: 1,484 against 1,453 (+31); last changed by 43849abe13 (2026-10-11, TASK-21243).
- library_media_controller.py: 4,812 against 4,769 (+43); last changed by 6571c47f86 (2026-10-15) and 659e60f590 (2026-10-13, tasks 34935-34937).
- library_notes_sync_controller.py: 2,394 against 2,375 (+19); last changed by e917ce817d (2026-10-11).

The ratchet's own message says a sanctioned move re-pins its row in the same commit with a dated comment, and anything else is the creep it exists to catch. No PR lane runs Tests/Architecture, which is how four rows went red.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each of the four rows passes at dev: the growth is traced to its commits, and each file either moves the growth out (recipe backlog/docs/library-decomposition-recipe.md section 1) or is re-pinned with a dated comment naming the sanctioned change
- [ ] #2 Tests/Architecture/test_library_modules_size_ratchet.py passes in a serial run
- [ ] #3 No other row in that file rises
<!-- AC:END -->
