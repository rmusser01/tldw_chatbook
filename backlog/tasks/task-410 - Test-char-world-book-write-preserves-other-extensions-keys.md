---
id: TASK-410
title: 'Test: char world-book write preserves other extensions keys'
status: To Do
assignee: []
created_date: '2026-07-21 03:42'
labels:
  - roleplay
  - lore
  - test-coverage
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
P2f (#728) char world-book attach/detach writes extensions['character_world_books'] via read-modify-write of the whole extensions dict. The opus final review flagged that no test asserts the load-bearing invariant that this preserves OTHER extensions keys (native 'character_book', 'chat_dictionaries'). Implementation was verified correct by review but is untested; a regression here would silently drop a character's native lore or embedded dictionaries on a world-book attach.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A test attaches then detaches a world book on a character whose extensions already contain both 'character_book' and 'chat_dictionaries', and asserts BOTH keys (and their values) survive unchanged through attach and through detach,WorldBookManager.get_world_books_for_character reflects the attach/detach,test lives in Tests/Character_Chat/test_world_book_manager.py and passes
<!-- AC:END -->
