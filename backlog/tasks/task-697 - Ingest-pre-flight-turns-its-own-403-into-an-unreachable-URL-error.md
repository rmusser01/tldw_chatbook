---
id: TASK-697
title: Ingest pre-flight turns its own 403 into an unreachable-URL error
status: To Do
assignee: []
created_date: '2026-07-26 14:12'
updated_date: '2026-07-26 14:31'
labels:
  - library
  - ingest
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The ingest pre-flight probes a URL with a HEAD request and turns any failure into an 'URL unreachable' error that blocks the source. A probe that cannot verify a URL should not be able to veto it: some sites refuse HEAD outright, and some refuse unrecognised clients regardless of headers. Confirmed on a Wikipedia article that answers 403 to our client even with a browser User-Agent, while a tldw server clipped the same URL successfully -- its browser-based fetch stack succeeds where ours cannot. So the local pre-flight currently blocks content that the configured backend could actually retrieve. A failure to fetch belongs to the ingest attempt, where it becomes a failed job with a real reason, not to a gate that refuses to start.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A URL that answers 403 to the probe is not reported as unreachable,A page the server can fetch is not blocked by the local pre-flight,A genuinely unreachable URL is still reported as such
- [ ] #2 A URL the probe cannot verify is not reported as unreachable,A page the configured backend can fetch is not blocked by the local pre-flight,A genuinely unresolvable URL is still surfaced to the user,A fetch that fails during ingest is reported as a failed job rather than silently succeeding
<!-- AC:END -->
