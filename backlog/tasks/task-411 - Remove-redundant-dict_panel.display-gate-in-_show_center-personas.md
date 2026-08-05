---
id: TASK-411
title: Remove redundant dict_panel.display gate in _show_center (personas)
status: To Do
assignee: []
created_date: '2026-07-21 03:42'
labels:
  - roleplay
  - tech-debt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
P2f (#728) moved PersonasCharacterDictionariesWidget into the #personas-character-attachments wrapper, whose display is now gated by _show_center on the character card/editor views. _show_center STILL sets dict_panel.display independently by the identical condition — redundant/dead-ish now that the wrapper controls both panels. Harmless (always consistent) but should be removed so the wrapper is the single source of truth for character-attachment panel visibility.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 _show_center no longer sets PersonasCharacterDictionariesWidget.display separately; the character-attachment panels' visibility is driven solely by the #personas-character-attachments wrapper gate,character-dictionary and world-book screen tests stay green (no visibility regression in card/editor/transcript views)
<!-- AC:END -->
