---
id: TASK-1993
title: Front-matter-aware markdown parsing for notes/library previews
status: To Do
assignee: []
created_date: '2026-08-02 22:30'
labels:
  - notes
  - library
  - markdown
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Notes synced from files commonly carry YAML front matter (`---` blocks). Textual's `Markdown` widget with its default gfm-like parser renders the front-matter delimiters as a thematic break plus stray text — noise at the top of every preview.

Frogmouth passes `parser_factory=lambda: MarkdownIt("gfm-like").use(front_matter.front_matter_plugin)` (from `mdit-py-plugins`) so front matter is consumed, not rendered. Textual's default is already gfm-like, so the plugin is the only delta. `mdit-py-plugins` should be an optional extra following the `optional_deps.py` pattern; when absent, previews render exactly as today.

Primary surface: the Library note preview (`Widgets/Library/library_notes_canvas.py`). Apply the same factory to other markdown preview surfaces only where front matter can plausibly appear.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A note whose content starts with YAML front matter previews without rendering the front-matter block
- [ ] #2 With mdit-py-plugins not installed, previews render exactly as today (graceful degradation, no import error)
- [ ] #3 The dependency is declared as an optional extra and checked via the optional-deps pattern
<!-- AC:END -->
