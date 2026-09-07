---
id: TASK-2307
title: Reader renders feed HTML as readable text
status: Done
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
- [x] The reader advertises how to give itself more room (or provides a
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

**AC#2 -- done, closed in the Qodo re-review round (Q4).** Was left
unchecked through two review rounds (the button rendered and posted
`ExpandReaderRequested`, but nothing handled it) because it was out of
those rounds' explicit scope. Qodo flagged the dead button as worse than no
button at all and asked for a decision: wire it to the existing mechanism
or remove it. Read task-1344's region-solo machinery
(`action_solo_region`/`RegionLayout.solo`, bound to `Z`) first, per the
review's instruction, and it was a clean fit -- `handle_expand_reader_
requested` now calls the exact same `self._apply_layout(self.region_
layout.solo(Region.CONTENT))` `action_solo_region` calls for a `Z`
keypress, through `_refuse_region_gesture_off_read_tab` for the same
defense-in-depth every other region-layout entry point goes through. No
second maximize mechanism was built. `_build_content_pane` also now seeds
`pane.expanded` from `self.region_layout.solo_region == Region.CONTENT`,
which the earlier rounds had not done either -- without it the button's
label would silently go stale across the very rebuild pressing it causes.
Live-verified: pressing Expand collapses Feeds and Items to one-line
headers and gives Content the whole centre stack (visible: the reader
showed more of the item, e.g. a "# Comments: 1" line that had been
scrolled off before); the button relabels to "Restore"; a second press
restores the three-pane view exactly.

**AC#3 -- done.** See above; 32 unit tests + 4 content-pane tests, all
passing, including a mutation-verified check that a link whose label already
IS the destination is not printed twice.

Modified/added: `Tests/Subscriptions/test_html_text.py` (new, later
extended), `Tests/UI/test_watchlists_content_pane.py` (tests appended
across rounds). No production code changed for AC#1/#3 in the original
session -- `html_text.py` and `content_pane.py`'s HTML wiring were already
complete in the WIP commit.

**Qodo Q5 (later round, same module):** `_HTML_SHAPED` (the `looks_like_
html` classifier) accepted a plain-text angle-bracket autolink
(`<https://x>`, `<mailto:a@b>` -- RFC 2822, standard in mailing-list/
plain-text feed bodies) as tag-shaped, so `html.parser` read it as a
namespace-prefixed start tag (`:` is a legal tag-name character) and
silently dropped the whole URL -- real content loss, the exact failure
`looks_like_html`'s own conservatism exists to avoid. Fixed in two parts:
`_HTML_SHAPED`'s tag alternative now requires a plausible tag name
(letters/digits only) immediately followed by a real delimiter
(whitespace/`/`/`>`), which excludes a bare autolink from classification
entirely; and a new `_AUTOLINK`-based protect/restore step inside
`html_to_display_text` covers the case tightening the classifier alone
cannot -- a body that legitimately mixes an autolink WITH real HTML (the
real tag correctly keeps the whole body routed through the parser, which
still cannot tell an autolink from a tag on its own). Verified against
every case Qodo listed, `Tests/Subscriptions/test_html_text.py` grew 8
tests, and both mechanisms are independently mutation-verified.

**Qodo Q4 (docstrings, Q1-Q3):** unrelated docstring completeness findings
on `watchlists_collections_screen.handle_check_now_requested`,
`sources_pane.source_last_scraped_text`, and `sources_pane.watch_busy_
source_ids` -- `Args:`/`Returns:` sections added, no behavioural change.
