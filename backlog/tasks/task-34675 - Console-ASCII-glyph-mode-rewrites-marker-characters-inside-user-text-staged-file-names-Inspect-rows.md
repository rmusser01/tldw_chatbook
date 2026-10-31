---
id: TASK-34675
title: >-
  Console: ASCII-glyph mode rewrites marker characters inside user text (staged
  file names, Inspect rows)
status: To Do
assignee: []
created_date: '2026-10-09 17:24'
labels:
  - console
  - bug
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On dev today (not B1's), in ASCII-glyph mode resolve_glyph_text maps every character of Console user text through ASCII_GLYPH_FALLBACKS, so user text containing any of ● ◆ ⚠ ✗ ✓ ◈ ◌ ◉ ◐ ✕ ▸ ◂ ▾ ▢ ▦ ✎ ▌ 📎 is rewritten. Call sites: Widgets/Console/console_composer_bar.py (the staged-attachment indicator, resolve_glyph_text(f"📎 {normalized}")) and Widgets/Console/console_inspector_section.py (resolve_glyph_text(row.primary_text), three call sites). B1's own frame glyphs were already kept out of that table (TASK-33910.2, Task 12b); this task covers dev's markers.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 User text keeps its characters in ASCII-glyph mode while app-authored markers still resolve to their ASCII substitutes
- [ ] #2 A regression test per surface (staged-attachment indicator, Inspect rows) fails on the current code
<!-- AC:END -->
