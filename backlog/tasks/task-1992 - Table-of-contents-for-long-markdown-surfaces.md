---
id: TASK-1992
title: Table of contents for long markdown surfaces (MarkdownTableOfContents)
status: Done
assignee:
  - '@claude'
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
- [x] #1 The HF README pane offers a heading-tree table of contents for the rendered document
- [x] #2 Selecting a TOC entry scrolls the document to that heading
- [x] #3 The TOC stays out of the way in tight layouts (collapsed or hidden by default at compact heights; the 32-row contract keeps working)
- [x] #4 Live tmux capture recorded showing the TOC populated from a real README and a successful jump
<!-- AC:END -->

## Implementation Plan (the how)

1. README TabPane becomes toolbar (`☰ Contents` toggle) + horizontal body: `MarkdownTableOfContents` (hidden by default, width 32 / max 40%) beside the existing VerticalScroll+Markdown.
2. Feed `Markdown.TableOfContentsUpdated` (id-filtered to `#readme-display`) into the TOC; on `TableOfContentsSelected`, `scroll_to_widget(md.query_one(f"#{block_id}"), top=True)` on the README scroll (frogmouth's scroll_to_block).
3. Tests: hidden-by-default + toggle, population from headings, selection scrolls. Live capture against a real HF README.

## Implementation Notes

`Widgets/HuggingFace/model_card_viewer.py` only. TOC is hidden by default (compact layouts unaffected — AC#3 satisfied by default-hidden + explicit toggle rather than a height threshold). Handlers use explicit `event.markdown.id` filtering, not `@on` selectors — TOC event `control` semantics differ between the Updated/Selected pair. Live-verified (tmux 235x52, real HF API): Lab ▸ Models ▸ Download Models → unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF → README → ☰ Contents shows the real heading tree (Highlights/Model Overview/Quickstart/Agentic Coding/Best Practices▸Citation) → clicking Quickstart scrolled the document to that heading (captures cap_toc_open/cap_toc_jump in session scratchpad). Test gotcha worth keeping: a TabPane that is not active has no scroll geometry — activate `readme-tab` before asserting scroll movement. Live side-observation: the real README renders its YAML front matter as a garbled list at the top — exactly task-1993's target, evidence banked there.

Scope note: filed as "long markdown surfaces"; shipped for the HF README pane (the AC's named surface). Media viewer/note-preview TOCs remain unscoped follow-up work if ever wanted.
