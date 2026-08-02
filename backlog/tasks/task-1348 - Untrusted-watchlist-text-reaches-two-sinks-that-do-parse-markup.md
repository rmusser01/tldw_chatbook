---
id: TASK-1348
title: Untrusted watchlist text reaches two sinks that do parse markup
status: In Progress
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
- [x] #1 Item-derived text reaching DataTable cells is escaped at that boundary, with a test using a markup-shaped feed title that fails without it
- [x] #2 A decision is recorded on whether remote markdown bodies may produce real hyperlinks, and the Markdown branch matches it
- [x] #3 The rule is stated once where a future contributor will find it: escape at sinks that parse, not at sinks that do not
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
AC#1 (the one real remaining gap): `items_pane.py`'s DataTable `add_row` now wraps every
item-derived string cell in `rich.markup.escape` -- a feed title `[bold red]BREAKING[/]` or a
source name `[link=...]` is displayed as literal text instead of being markup-parsed. status /
created_at / the queued glyph are app-controlled but escaped too, so every non-constant cell is
uniformly safe and no future editor has to re-audit which column carries remote text. Pinned by
`test_markup_shaped_item_text_is_escaped_at_the_datatable_boundary` (fails if the boundary escape
is removed -- mutation-verified).

AC#2 + AC#3 were ALREADY satisfied on dev before this task was picked up: `content_pane.py`'s
`_MARKDOWN_HYPERLINKS = False` (line 63, used at the `Markdown(...)` call ~line 133) is the
recorded decision that remote markdown bodies must NOT emit real OSC-8 hyperlinks (phishing-anchor
reasoning documented in full there), and `render_article`'s docstring states the governing rule
once for a future contributor: escape/​defend at the sink that actually parses, not at sinks that
do not. This task's items_pane comment cross-references that rule rather than restating it.

Files: `tldw_chatbook/UI/Watchlists_Modules/items_pane.py`,
`Tests/Watchlists/test_watchlists_items_pane.py`.
<!-- SECTION:NOTES:END -->
