---
id: TASK-1221
title: Watchlists is gated on three packages nothing imports
status: In Progress
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
- [x] #1 The subscriptions feature gate requires only packages that are actually imported
- [x] #2 The missing-dependency alert names only packages whose absence genuinely disables the feature
- [x] #3 The subscriptions extra in pyproject.toml matches the narrowed gate, and a fresh install of that extra satisfies it
- [x] #4 A test asserts every package named in the subscriptions gate has at least one real import site in tldw_chatbook/, resolving the receiver rather than matching the bare name
- [x] #5 Watchlists remains available in an environment with bs4 and cryptography but without markdown, schedule and feedparser
<!-- AC:END -->

## Implementation Notes

The gate is now `("beautifulsoup4",)` -- down from five packages, three of which nothing
imported.

**The fix went one package further than the task described, because the test caught me.** I first
narrowed the gate to `("beautifulsoup4", "cryptography")`, reasoning that cryptography's absence
means credentials are stored in plaintext. The AC#5 test then failed: cryptography is not installed
in this project's own dev environment, and Watchlists runs there perfectly well. Gating on it would
have marked a working install unavailable -- the same defect, one package over.

The distinction that resolves it: `security.py` imports cryptography in a `try/except` and warns at
the point of use. That is a **security degradation to surface where it happens**, not an
availability gate. Conflating "this feature is worse without X" with "this feature cannot run
without X" is what produced the original bug. Only `beautifulsoup4` survives the test, because
`monitoring_engine.py` does a module-level `from bs4 import BeautifulSoup` with no fallback.
`defusedxml` falls back to stdlib `xml.etree`, so it is out too.

**The alert was worse than the task recorded.** `alert_subscriptions_not_available` hardcoded
`["markdown", "schedule", "feedparser", "defusedxml"]` -- three unused packages plus one optional
one -- and omitted `beautifulsoup4` entirely, the only package whose absence actually breaks
Watchlists. It now reads the gate's own list through `_module_present`, the same resolver the gate
uses, so the two cannot drift apart.

That shared resolver matters: the gate keys on PyPI names (`beautifulsoup4`), which match neither
the import name (`bs4`) nor the lazily-populated `DEPENDENCIES_AVAILABLE` keys. A plain
`DEPENDENCIES_AVAILABLE.get(dep, False)` reports installed packages as missing -- I wrote that first
and caught it before committing.

The extra in `pyproject.toml` keeps cryptography and defusedxml (installing them is still the right
default; the fallbacks are plaintext credentials and a weaker XML parser) with a note explaining why
they are shipped but not gated.

**Test method.** The guard resolves imports with an AST walk over `tldw_chatbook/`, not a text
search -- a bare grep for `markdown` matches the comment "Extract cells from markdown table row" in
`Utils/file_extraction.py`, which is how this survived. It carries its own sanity check
(`test_the_ast_walk_actually_finds_imports`) so a broken walk cannot pass everything vacuously.

Modified: `tldw_chatbook/Utils/optional_deps.py`, `tldw_chatbook/Utils/widget_helpers.py`,
`pyproject.toml`. Added: `Tests/Utils/test_subscriptions_dependency_gate.py`.
