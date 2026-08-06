---
id: TASK-2307
title: Reader renders feed HTML as readable text
status: In Progress
assignee: []
created_date: '2026-08-04'
labels:
  - watchlists
  - ux
  - uat-2026-08-04
dependencies: []
priority: high
---

## Description (the why)

UAT: the item reader displays raw HTML markup — "<p>Article URL: <a
href=...>" shown literally. It is correctly escaped and inert (keep that
property), but unreadable as content. The reader region is also ~9 rows at
52-row terminals with no advertised way to expand it (z/Z region controls
exist but nothing on screen says so).

UAT findings F26 (high), F27.

## Acceptance Criteria (the what)

- [ ] Feed HTML content renders as readable text (tags stripped or
      converted; links presented legibly), while remote-derived text remains
      inert against markup/injection — the escaping-terminal rule holds at
      the NEW final render step.
- [ ] The reader advertises how to give itself more room (or provides a
      visible expand affordance).
- [ ] Regression tests cover HTML-to-text rendering and the injection-
      inertness of rendered remote content.

## Implementation Plan (the how)

1. Grep for an existing html-to-text helper first. (Done: the repo has
   `ContentExtractor.extract_text_from_html` in `monitoring_engine.py`, but it
   collapses the whole document to ONE line and drops every `href` -- unusable
   as a reader. `html2text` is an optional `[ebook]` extra, not a hard dep.)
2. New module `Subscriptions/html_text.py`, stdlib `html.parser` only: block
   tags become newlines, `<a href>` becomes `label (url)` as plain text,
   `<li>` becomes a bullet, `script`/`style`/`head` content is dropped, entity
   refs are unescaped once. Plus `strip_control_characters` (C0 except
   `\n`/`\t`, and C1) so a body carrying raw ESC cannot reach the terminal.
3. `content_pane.render_article`/`render_change` run every remote-derived
   field through that pair as the LAST step before `Text.append`. The result
   is a plain `str` appended to a `rich.text.Text`, which never markup-parses
   -- the existing inertness property is preserved by construction, not by a
   sanitizer.
4. AC#2: a visible expand affordance. `ContentPane` grows an `Expand`/
   `Restore` button beside `Mark unread` (same row, zero extra rows) that
   posts `ExpandReaderRequested`; the screen solos/unsolos `Region.CONTENT`
   through the same `_apply_layout` path `Z` uses, and seeds the button's
   label from the live layout so it never lies.
5. Tests: a unit file for the converter (including hostile payloads --
   `[bold]`, `<script>`, ESC/OSC-8, malformed nesting) plus content-pane
   tests proving the rendered output is inert when actually rendered through
   a Rich console.
