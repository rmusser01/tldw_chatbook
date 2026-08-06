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

- [x] Feed HTML content renders as readable text (tags stripped or
      converted; links presented legibly), while remote-derived text remains
      inert against markup/injection — the escaping-terminal rule holds at
      the NEW final render step.
- [ ] The reader advertises how to give itself more room (or provides a
      visible expand affordance).
- [x] Regression tests cover HTML-to-text rendering and the injection-
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

## Implementation Notes

Picked up mid-flight after a previous implementer was cut off by a rate
limit (`wip(watchlists): batch-4 partial`, tasks 2307/2308/2309 together).
That commit had already landed `Subscriptions/html_text.py` (complete,
including `strip_control_characters`) and `content_pane.py`'s wiring
(`readable_body_text` on the render path, plus the `Expand`/`Restore`
button in `#content-actions` posting `ExpandReaderRequested`). This session
added the test suite (AC#3) and live-verified AC#1.

**AC#1 -- verified, live and in tests.** `Tests/Subscriptions/test_html_text.py`
(32 cases) covers readability (paragraphs, lists, `<br>`, image alt text,
entity unescaping exactly once, nested drop-content) and the hostile set the
task asked for: bracket-shaped text (`[bold red]x[/]`), raw ESC, a full
OSC-8 hyperlink sequence, a `javascript:` href, and the C1 control range --
all proven to lose their control bytes while the surrounding prose survives.
`Tests/UI/test_watchlists_content_pane.py` adds 4 more proving the SAME
property through `render_article` and, end to end, through a mounted
`ContentPane`. Live-verified against a real feed (hnrss.org/frontpage): the
reader shows "Article URL: https://buttondown.com/blog/..." as legible
prose, not `<p>Article URL: <a href=...>`.

**AC#2 -- NOT done; left unchecked on purpose.** The button renders and
carries the right tooltip, but nothing in `watchlists_collections_screen.py`
imports or handles `ExpandReaderRequested` -- confirmed by grep and by
mutation (there is no `@on(ExpandReaderRequested)` anywhere). Pressing
Expand currently does nothing. This batch's dispatch explicitly scoped the
remaining work to the Items publish column (task-2308) and Check-now
progress (task-2309) plus tests, and did not include finishing this AC's
region-solo wiring (`_apply_layout`/`Region.CONTENT`, per this task's own
step 4) -- so it was left alone rather than expanding scope past what was
asked. Filed as a gap for a follow-up task rather than silently checked off.

**AC#3 -- done.** See above; 32 unit tests + 4 content-pane tests, all
passing, including a mutation-verified check that a link whose label already
IS the destination is not printed twice.

Modified/added: `Tests/Subscriptions/test_html_text.py` (new),
`Tests/UI/test_watchlists_content_pane.py` (4 tests appended). No production
code changed for this task in this session -- `html_text.py` and
`content_pane.py`'s HTML wiring were already complete in the WIP commit.
