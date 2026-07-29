---
id: TASK-1348
title: Untrusted watchlist text reaches two sinks that do parse markup
status: To Do
assignee: []
created_date: '2026-07-29 05:30'
labels:
  - watchlists
  - security
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase D established the correct rule for the reader: escape where the parser actually is, not
everywhere. `Text.append()` does not parse markup, so escaping there bought nothing and put a
visible backslash in front of every `[docs](url)` and `[sic]` in real feed prose — that escaping
was removed. Two sinks that **do** parse remain.

**`DataTable` cells (pre-existing).** `items_pane.py:78-84` passes `str` cells built from item
titles, source names and URLs straight to `DataTable`, which markup-parses them. A feed title
containing `[bold red]` is interpreted rather than displayed. Phase D did not make this worse — its
`update_cell` writes only app-controlled status strings — but it did not fix it.

**`rich.markdown.Markdown` (latent).** `content_pane.py:90` routes a markdown body through
`Markdown`, which renders `[text](url)` as a real OSC-8 hyperlink from remote feed content.
Unreachable today because nothing writes `content_format` (see TASK-1343), so this becomes live the
moment that task lands.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Item-derived text reaching DataTable cells is escaped at that boundary, with a test using a markup-shaped feed title that fails without it
- [ ] #2 A decision is recorded on whether remote markdown bodies may produce real hyperlinks, and the Markdown branch matches it
- [ ] #3 The rule is stated once where a future contributor will find it: escape at sinks that parse, not at sinks that do not
<!-- AC:END -->
