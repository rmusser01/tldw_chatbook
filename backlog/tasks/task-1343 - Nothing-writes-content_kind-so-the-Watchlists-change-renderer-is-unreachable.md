---
id: TASK-1343
title: 'Nothing writes content_kind, so the Watchlists change renderer is unreachable'
status: Done
assignee:
  - '@claude'
created_date: '2026-07-29 05:30'
updated_date: '2026-07-29 23:04'
labels:
  - watchlists
  - dead-code
  - observability
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase D built two renderers for the Watchlists reader and dispatches between them on
`content_kind` (`content_pane.py` `render_for`). **No code anywhere in the repo writes
`content_kind`**, so the dispatch always falls through to the article renderer and
`render_change` is unreachable in production.

`monitoring_engine.py:754-763` (the site-change path) emits `change_percentage` and `change_type`
but never `content_kind` or `diff_summary`. The RSS path emits neither `content_kind` nor
`content_format`.

Consequences, all of correct code that cannot currently fire: site changes render as articles and
lose the percent-changed / change-type headline; the markdown branch (`content_pane.py:85-90`)
can never execute, so a markdown body would render as raw source if one ever arrived; and the
`diff_summary` line lives inside a renderer that is never dispatched.

`item_persist.persist_subscription_item` accepts and validates the field — the pairings
`("article","text")`, `("article","markdown")`, `("change","diff")` are enforced at the write
boundary — so the persistence half is ready. Only the producer is missing.

This is the fifth instance in this codebase of the same shape: built, wired, carrying nothing, and
reading as live to a grep.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The site-change detection path writes content_kind="change" and content_format="diff" when it persists an item, alongside the change_percentage and change_type it already writes
- [x] #2 The feed path writes content_kind="article" with a content_format matching what it actually captured
- [x] #3 A test asserts a real site check produces an item that render_for dispatches to render_change, failing if content_kind is absent
- [x] #4 diff_summary is populated by the change path, or removed from the renderer if nothing will produce it
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Establish where each source type's item dict is built and confirm it reaches persist_subscription_item unchanged.
2. Give the site-change path a real diff: re-segment both snapshots (extracted page text is ONE line), build a bounded unified-diff body, and emit content_kind/content_format/diff_summary/change_type.
3. Emit content_kind=article + content_format=text on the RSS, Atom, JSON Feed and API paths.
4. Name the vocabulary once, in item_persist, so no producer can typo a pairing.
5. Correct content_pane's stale change_percentage producer comment.
6. Tests driving the real producers end to end, plus mutation checks on every assignment.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Gave the reader's `content_kind` dispatch its missing producer, on every live path.

**The change path now emits a real diff, not the whole page.** `URLMonitor.check_url` stored `current_content["text"]` — the entire new page — as the item body, so a reader could see what a page says now but never what changed, while the same full text was ALSO already in `url_snapshots`. It now stores a bounded unified-diff body (`content_kind="change"`, `content_format="diff"`), plus `diff_summary` ("N line(s) added, M removed", counted over the whole diff so it stays true when the body is capped) and a derived `change_type`.

**Segmentation was the load-bearing detail.** `ContentExtractor.extract_text_from_html` joins every chunk of a page with a single space, so extracted page text is ONE line with no newlines. A line-based diff of two snapshots is therefore always exactly `-<entire old page>` / `+<entire new page>`: the full text twice. Both sides are re-segmented before diffing — real line breaks when present, otherwise sentence boundaries (which stay aligned under a local edit; fixed-width chunking does not) — and over-long segments are word-wrapped at 110 chars so no diff line is wider than the pane.

**Bounds:** 400 lines / 20,000 chars, whichever is hit first, with the truncation stated IN the body so a partial change never reads as complete. Losing the tail loses nothing recoverable — the full page is still in `url_snapshots`. Two notices (truncation, and "hash changed but text is identical after normalization") deliberately start with `[` so `render_change` does not colour them as changes; an empty body would otherwise make the renderer claim "no body captured", which would be false.

**`change_type`** was the hardcoded literal `"content"`. Now `new`/`removed`/`content`, the only three distinctions two snapshots support. `baseline_manager.ChangeReport`'s `structural`/`semantic` need DOM and embedding analysis `check_url` does not do; `baseline_manager` was not touched (TASK-1360).

**Feed and API paths** emit `article`/`text`. `text` is the honest answer: RSS `description`, `atom:content`, JSON Feed `content_html` and an API's mapped JSON field all arrive as the publisher's plain text or HTML and nothing converts them; claiming `markdown` would hand publisher HTML to a CommonMark parser.

**Two adjacent corrections.** (1) `change_percentage` is now scaled to 0-100 at the point it is handed to the reader, matching the column name, `render_change`'s `f"{float(pct):.0f}% changed"` and every renderer fixture; the threshold comparison still uses the 0-1 ratio. Unscaled, making the renderer reachable would have printed "0% changed" for a real 35% change. (2) `content_pane.py`'s comment claiming `baseline_manager.py` writes `change_percentage` now names only the real producer.

The kind/format vocabulary is named once in `item_persist.py` (where `_VALID_PAIRINGS` already lived) and imported by both producers, because an invalid pairing raises inside a scheduled fetch and `execute_run` converts that into a failed run that drops every item it collected.

Modified: `Subscriptions/monitoring_engine.py`, `Subscriptions/local_watchlists_service.py`, `Subscriptions/item_persist.py`, `UI/Watchlists_Modules/content_pane.py`. Added: `Tests/Subscriptions/test_watchlist_content_kind_producer.py` (12 tests, all driving the real producers through real persistence and the real renderer; 9 mutation checks confirmed each assignment is load-bearing).
<!-- SECTION:NOTES:END -->
