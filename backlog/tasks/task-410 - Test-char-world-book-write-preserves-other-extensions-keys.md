---
id: TASK-410
title: 'Test: char world-book write preserves other extensions keys'
status: Done
assignee:
  - '@zcode'
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
- [x] #1 A test attaches then detaches a world book on a character whose extensions already contain both 'character_book' and 'chat_dictionaries', and asserts BOTH keys (and their values) survive unchanged through attach and through detach,WorldBookManager.get_world_books_for_character reflects the attach/detach,test lives in Tests/Character_Chat/test_world_book_manager.py and passes
<!-- AC:END -->

## Implementation Plan

1. Locate the character attach/detach write path on current dev (WorldBookManager.attach_world_book_to_character / detach_world_book_from_character -> _write_character_world_books).
2. Add a test next to the existing attach round-trip test in Tests/Character_Chat/test_world_book_manager.py: create a character whose extensions carry populated character_book and chat_dictionaries, attach then detach a world book, assert both sibling keys and values survive both writes and get_world_books_for_character reflects the attach/detach.

ADR required: no
ADR path: N/A
Reason: Test-only coverage addition; no production change.

## Implementation Notes

Added ``test_attach_detach_preserves_other_extension_keys`` to Tests/Character_Chat/test_world_book_manager.py, beside the existing round-trip test whose fixtures it reuses. The character is created with ``extensions`` containing a populated ``character_book`` (native lore) and ``chat_dictionaries`` (two slugs); the test attaches a world book, reads back the persisted extensions (JSON-parsed defensively, mirroring _normalize_extensions), asserts both sibling keys and values are unchanged and get_world_books_for_character shows the book; then detaches and asserts the same two invariants plus the empty attachment list. A regression in _write_character_world_books' read-modify-write that dropped sibling keys fails the equality assertions on either phase.

The implementation was already correct by review (as the task recorded); this pins it. File suite: 33 passed (32 before + this test). Ruff clean after format.

Modified: Tests/Character_Chat/test_world_book_manager.py.
