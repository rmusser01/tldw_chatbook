---
id: TASK-21230
title: >-
  Watchlists url_list and sitemap sources hide the permanent bs4 install hint
status: To Do
assignee: []
created_date: '2026-08-23'
labels:
  - bug
  - watchlists
  - subscriptions
  - error-handling
dependencies: []
priority: medium
---

## Description

Source: close-out of the 2026-08-22 holistic performance review burn-down (queue TASK-21100
– TASK-21134). Found by the TASK-21104 adversarial review and its implementation pass.
Base evidence doc: `Docs/Design/2026-08-22-holistic-perf-review.md`.

TASK-21104 made `beautifulsoup4` a guarded import so a base install can still boot, and the
guard reports a missing bs4 with an install hint. In
`Subscriptions/local_watchlists_service.py` the task-1394 per-URL privacy suppression used by
the `url_list` and `sitemap` loops (`_check_url_isolated`) reduces ANY exception — including a
**permanent** `ImportError` — to a type-only DEBUG line plus a `DISPOSITION_ERROR` carrying
`reason=None`. A user whose watchlist uses a `url_list` or `sitemap` source therefore sees an
unexplained error with no reason text and never learns to install the extra. The single-`url`
arm, which does not go through the suppression, surfaces the hint verbatim. The hint string is
a module constant containing no user data, so surfacing it does not conflict with the privacy
rule the suppression exists for.

Second, pre-existing hygiene in the same package, found during that pass and re-confirmed on
dev `b2b1e2e0d`: `Subscriptions/monitoring_engine.py` imports `from loguru import logger`
twice (lines 27 and 37).

## Acceptance Criteria

- [ ] A `url_list` or `sitemap` watchlist source that fails because `beautifulsoup4` is not installed surfaces the same install hint a single-`url` source surfaces
- [ ] The `DISPOSITION_ERROR` produced for that case carries a non-null reason
- [ ] Transient and network-shaped per-URL failures still carry no URL, no exception message and no user data — the task-1394 suppression is unchanged for them
- [ ] `Subscriptions/monitoring_engine.py` imports `logger` exactly once
- [ ] A test fails if a permanent ImportError is re-suppressed to `reason=None` on the isolated arm
