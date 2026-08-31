---
id: TASK-26014
title: 'Anthropic prompt cache: 1-hour TTL tier'
status: To Do
assignee: []
created_date: '2026-08-31 15:45'
labels:
  - console
  - context
  - performance
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every cache breakpoint uses the 5-minute default, so a user returning to a conversation after a coffee break re-pays the full prefix. Verified on origin/dev: all three cache_control sites emit a bare {"type": "ephemeral"} marker - LLM_Calls/LLM_API_Calls.py:1466, :1511, :1544 - and a grep for "ttl" or 1h in that file returns zero. Anthropic supports a 1-hour tier. Chatbook already has the surrounding safety: a capability gate, a [caching] kill switch, and a degrade-retry that strips breakpoints on a 400 (:1597-1615), so the risk of adding a tier is bounded.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Cache breakpoints can emit a 1-hour TTL where the provider and model support it
- [ ] #2 The tier is configurable, and the default is stated explicitly with its cost reasoning in the task notes
- [ ] #3 A model or route that does not support the longer tier silently falls back to the 5-minute marker rather than erroring
- [ ] #4 The existing degrade-retry still strips all cache_control on a 400 mentioning it, unchanged
- [ ] #5 Cache read and creation token accounting continues to report correctly with the longer tier - verified against Chat/provider_usage.py:235,285-286
- [ ] #6 The [caching] kill switch disables the longer tier along with everything else
<!-- AC:END -->
