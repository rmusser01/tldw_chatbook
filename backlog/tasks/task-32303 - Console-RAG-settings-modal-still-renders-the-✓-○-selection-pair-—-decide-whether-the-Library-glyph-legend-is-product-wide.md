---
id: TASK-32303
title: >-
  Console RAG settings modal still renders the ✓/○ selection pair — decide
  whether the Library glyph legend is product-wide
status: To Do
assignee: []
created_date: '2026-09-11 00:54'
labels:
  - console
  - ux
  - critique-9
  - decision-needed
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Library glyph legend (task-32235) moved selection to ☑/☐ and kept ○ for blocked/disabled. `Widgets/Console/console_rag_settings_modal.py` still mirrors the old ✓/○ pair, so the same state reads differently on the Console screen. Product decision: adopt the Library legend app-wide, or record that Console keeps its own.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A decision is recorded (product-wide legend or Console-specific), and if product-wide the modal uses the shared constants
<!-- AC:END -->
