---
id: TASK-697
title: Ingest pre-flight turns its own 403 into an unreachable-URL error
status: Done
assignee: []
created_date: '2026-07-26 14:12'
updated_date: '2026-07-26 23:45'
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed in PR #923 as a prerequisite for bringing web clipping into the canvas.

The pre-flight's HEAD probe turned any failure into 'URL unreachable' and dropped the source entirely -- not merely warned: the URL never reached type_groups, so total_files stayed 0. But any HTTP status proves the host resolved and answered, and sites routinely refuse HEAD (405) or unrecognised clients (403) while serving the page fine to whoever actually fetches it. Confirmed on a Wikipedia article that answers 403 to our client even with a browser User-Agent -- so the User-Agent is not the cause -- and that a tldw server clipped at 200, because its browser-based stack succeeds where ours is refused. The local pre-flight was blocking content the configured backend could retrieve.

A probe may now report doubt but not refuse: an uninterpretable status becomes a 'Could not check the link' warning. 404/410 still refuse, since there the host is stating the resource is not there.

Verified live: the Wikipedia article now reaches the canvas as the web group with that warning, while a real 404 is still refused.

The status was never updated when it shipped; the code has been on dev since #923.
<!-- SECTION:NOTES:END -->
