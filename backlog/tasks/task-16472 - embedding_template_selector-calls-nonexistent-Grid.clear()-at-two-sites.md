---
id: TASK-16472
title: embedding_template_selector calls nonexistent Grid.clear() at two sites
status: To Do
assignee: []
created_date: '2026-08-14'
labels:
  - bug
  - tech-debt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`tldw_chatbook/Widgets/embedding_template_selector.py:144` and `:163` call `grid.clear()` on a `Grid` — the same bug class TASK-15992 fixed in the selection dialogs (`hasattr(Grid, 'clear')` is False; `remove_children()` is the idiom), so exercising either path would raise AttributeError. Found by the TASK-15992 review's AST sweep of the whole package (assignments from `query_one(..., <container type>)` followed by `var.clear()`); these were the only two remaining hits (scratchpad `review15992.md`, section S1).

Reachability finding, recorded per the review: nothing imports `EmbeddingTemplateSelector` outside its own module, so the code is currently unreachable — the mechanical fix is one line per site, but the right disposition may be retirement per the repo's dead-code ruling rather than fixing an orphan.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No `.clear()` calls on Textual containers remain repo-wide; if cheap, extend the review's AST sweep into a small guard test so the bug class cannot return
- [ ] #2 Fix-vs-retire is decided with reachability evidence recorded (who imports/mounts the widget, or proof nobody does)
<!-- AC:END -->
