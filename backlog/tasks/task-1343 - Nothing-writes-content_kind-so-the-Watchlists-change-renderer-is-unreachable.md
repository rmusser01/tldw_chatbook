---
id: TASK-1343
title: Nothing writes content_kind, so the Watchlists change renderer is unreachable
status: To Do
assignee: []
created_date: '2026-07-29 05:30'
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
- [ ] #1 The site-change detection path writes content_kind="change" and content_format="diff" when it persists an item, alongside the change_percentage and change_type it already writes
- [ ] #2 The feed path writes content_kind="article" with a content_format matching what it actually captured
- [ ] #3 A test asserts a real site check produces an item that render_for dispatches to render_change, failing if content_kind is absent
- [ ] #4 diff_summary is populated by the change path, or removed from the renderer if nothing will produce it
<!-- AC:END -->
