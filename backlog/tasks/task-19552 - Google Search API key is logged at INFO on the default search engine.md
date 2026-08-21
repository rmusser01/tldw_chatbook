---
id: TASK-19552
title: >-
  Google Search API key is logged at INFO on the default search engine
status: To Do
assignee: []
created_date: '2026-08-21 20:02'
labels:
  - security
  - websearch
  - logging
priority: high
dependencies: []
---

## Description

Source: 2026-08-21 holistic review, Lane 2 (security & privacy) — its **Tier 1
#2**. CONFIRMED by the lane, **independently re-confirmed by the review
controller**, and re-verified at this branch base.

`tldw_chatbook/Web_Scraping/WebSearch_APIs.py`:

```
3402:        params["key"] = google_search_api_key
3428:        logger.info(f"Prepared parameters for Google Search: {params}")
```

The whole `params` dict — including the API key just written into it — is
interpolated into an INFO-level log line. This is not a DEBUG-only path:

```
296:        "search_engine": search_params.get("engine", "google"),
```

**Google is the default engine**, so this fires on the ordinary `web_search`
tool path and on deep search, at the default log level, for any user who has
configured a Google CSE key.

Two things make this worse than a single leaky line:

- It is a **recurrence of a class a previous remediation wave already
  repaired**. The fix pattern exists in the repo; this call site never adopted
  it. (Bing on the same surface is clean — it passes the credential as a
  header and never formats it.)
- It compounds with TASK-19555: the in-app log buffer has no sanitizing filter
  and the Logs screen offers "Copy all" to the system clipboard, so this key
  does not merely sit in a file — it is on the share path the UI actively
  invites users to use.

## Acceptance Criteria

- [ ] No search-provider credential reaches any log record, at any level, on
      any engine — the Google path in particular never formats a dict that
      contains `key`
- [ ] The fix redacts at the point of formatting rather than relying on the
      caller's log level, so raising the log level for troubleshooting cannot
      re-expose it
- [ ] A regression test asserts that a configured Google CSE key does not
      appear in emitted log records for a search call, and that the test would
      fail if the redaction were removed (mutation-checked, not merely green)
- [ ] The other search-provider paths in `WebSearch_APIs.py` are swept for the
      same shape and any further instances are fixed in the same change
- [ ] Owner is advised whether the affected key needs rotating, given the key
      may already be in existing local log files
