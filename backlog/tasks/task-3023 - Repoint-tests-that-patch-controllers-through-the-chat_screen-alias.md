---
id: TASK-3023
title: Repoint tests that patch controllers through the chat_screen alias
status: To Do
assignee: []
created_date: '2026-08-07 14:58'
labels:
  - tech-debt
  - tests
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave 4 moved every Console controller construction into UI/Console_Modules/wiring.py, leaving five controller imports in chat_screen.py that no code references. They cannot be deleted: 18 test sites across 5 files patch them through the screen module's namespace (chat_screen_module.ConsoleDictationController and friends), which no import-grep can see because the alias hides it. Deleting them turns 28 tests red -- tripped once during the extraction. The imports now carry noqa markers and a block comment so a linter does not harvest them, but the real fix is repointing the patch sites at the modules that own the classes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every test that patches a Console controller patches it on the module that defines it, not through chat_screen
- [ ] #2 The five re-export imports and their noqa markers are removed from chat_screen.py
- [ ] #3 pyflakes on chat_screen.py returns to its pre-wave-4 count
<!-- AC:END -->
