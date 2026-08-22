---
id: TASK-20940
title: 'FB2 extractor duplicates section titles and nested-section content'
status: To Do
assignee: []
created_date: '2026-08-22'
updated_date: '2026-08-22'
labels:
  - ingestion
  - ebook
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`process_fb2` (`tldw_chatbook/Local_Ingestion/Book_Ingestion_Lib.py:2333`) walks `body.findall(".//section")` and, per section, reads the title and paragraphs with descendant axes (`.//title`, `.//p`). Two duplication defects follow: FB2 `<title>` elements contain their own `<p>` children, so every section title is emitted twice (the `# title` heading line plus the title's own `<p>` caught by `.//p`), and nested sections' paragraphs are re-emitted once per ancestor section. Surfaced by the chunking-agent-tools T6 story test (a real-FB2 fixture dry run doubled the chapter units and misaligned the chapter↔node correspondence, forcing the story onto the EPUB route); pre-existing and out of that sub-project's scope — filed per its final review.

Discovery record: `.superpowers/sdd/2026-08-22-chunking-agent-tools/task-6-report.md` (Deviations #1, Concerns #2).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] Each section title appears exactly once in the extracted content (the heading line; the title's own `<p>` is not re-emitted as body text)
- [ ] Nested-section content is not duplicated per ancestor section (direct-children axes, or equivalent de-duplication, for title and paragraph extraction)
- [ ] A regression fixture from the T6 discovery: a real-FB2 ebook with titled and nested sections, asserted through `process_fb2` (title-once, no ancestor duplication) and kept as the pin for the fix
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
(To be added when the task is picked up.)
<!-- SECTION:PLAN:END -->
