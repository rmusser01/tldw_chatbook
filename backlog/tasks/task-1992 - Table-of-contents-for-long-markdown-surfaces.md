---
id: TASK-1992
title: Table of contents for long markdown surfaces (MarkdownTableOfContents)
status: To Do
assignee: []
created_date: '2026-08-02 22:30'
labels:
  - ux
  - markdown
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
No markdown surface in the app offers a table of contents; long documents (HF model READMEs, media content/analysis panes, Library note previews) are scroll-only. Textual ships the pieces unused: `MarkdownTableOfContents` fed by `Markdown.TableOfContentsUpdated` events, plus `scroll_to_block()` to jump to a heading. Frogmouth's wiring (`widgets/navigation_panes/table_of_contents.py` + `Viewer.scroll_to_block`) is a direct reference implementation, including the decoupled-ToC construction workaround (throwaway `Markdown()` argument).

Start with the HF README pane (longest documents, clearest win). Team convention requires a live render-capture before building on complex widgets — `MarkdownTableOfContents` wraps a `Tree`, which is capture-verified elsewhere, but the gate still applies.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The HF README pane offers a heading-tree table of contents for the rendered document
- [ ] #2 Selecting a TOC entry scrolls the document to that heading
- [ ] #3 The TOC stays out of the way in tight layouts (collapsed or hidden by default at compact heights; the 32-row contract keeps working)
- [ ] #4 Live tmux capture recorded showing the TOC populated from a real README and a successful jump
<!-- AC:END -->
