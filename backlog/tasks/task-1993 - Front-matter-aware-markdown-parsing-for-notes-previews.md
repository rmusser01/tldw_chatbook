---
id: TASK-1993
title: Front-matter-aware markdown parsing for notes/library previews
status: Done
assignee:
  - '@claude'
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
- [x] #1 A note whose content starts with YAML front matter previews without rendering the front-matter block
- [x] #2 With mdit-py-plugins not installed, previews render exactly as today (graceful degradation, no import error)
- [x] #3 The dependency is declared as an optional extra and checked via the optional-deps pattern
<!-- AC:END -->

## Implementation Plan (the how)

1. Shared `Utils/markdown_parsing.front_matter_parser_factory()` — returns a gfm-like MarkdownIt factory with the front-matter plugin, or None (Textual default) when mdit-py-plugins is absent, gated via `optional_deps.check_dependency`.
2. Wire into the Library note preview and the HF README display (live 1992 capture showed a real README rendering its YAML front matter as a garbled list — evidence the surface plausibly carries front matter).
3. `frontmatter` optional extra in pyproject; tests for parse behavior, degradation, and the wired surface.

## Implementation Notes

`Utils/markdown_parsing.py` (new), `Widgets/Library/library_notes_canvas.py`, `Widgets/HuggingFace/model_card_viewer.py`, `pyproject.toml` (`frontmatter` extra). Factory preserves gfm-like table support (asserted). Tests: `Tests/UI/test_front_matter_previews_1993.py` — front_matter token consumed, degradation to None without the dep, mounted README strips the block while the document renders. Consuming suites (HF 1991/1992, library multiselect notes) green.
