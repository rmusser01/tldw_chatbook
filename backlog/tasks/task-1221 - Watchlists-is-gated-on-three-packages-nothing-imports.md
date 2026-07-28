---
id: TASK-1221
title: Watchlists is gated on three packages nothing imports
status: To Do
assignee: []
created_date: '2026-07-28 00:35'
labels:
  - watchlists
  - dependencies
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The `subscriptions` feature gate in `Utils/optional_deps.py` requires
`("markdown", "schedule", "feedparser", "beautifulsoup4", "cryptography")`. Three of those five are
imported nowhere in `tldw_chatbook/`:

- `markdown` — was used only by the briefing/export/distribution modules retired in TASK-1211
- `schedule` — no import site; belonged to the legacy `SubscriptionScheduler`
- `feedparser` — no import site. `FeedMonitor` parses feeds with `defusedxml`/`ElementTree` plus
  `bs4`, not feedparser

Only `beautifulsoup4` (19 files) and `cryptography` (`security.py`) are real requirements.

The consequence is user-facing: a user who has bs4 and cryptography but not, say, `feedparser` sees
Watchlists reported unavailable and is told to install a dependency the feature never uses.
`Utils/widget_helpers.py:302` lists the same three in the "missing dependencies" alert, so the
message names packages that would not help.

`markdown` became unused as a direct result of TASK-1211; the other two predate it.

The fix spans three files that must move together — `optional_deps.py`'s feature tuple,
`widget_helpers.py`'s alert list, and the `subscriptions` extra in `pyproject.toml`. Removing the
packages from the extra without narrowing the gate would leave fresh installs failing it, which is
why this was not folded into the retirement PR.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The subscriptions feature gate requires only packages that are actually imported
- [ ] #2 The missing-dependency alert names only packages whose absence genuinely disables the feature
- [ ] #3 The subscriptions extra in pyproject.toml matches the narrowed gate, and a fresh install of that extra satisfies it
- [ ] #4 A test asserts every package named in the subscriptions gate has at least one real import site in tldw_chatbook/, resolving the receiver rather than matching the bare name
- [ ] #5 Watchlists remains available in an environment with bs4 and cryptography but without markdown, schedule and feedparser
<!-- AC:END -->
