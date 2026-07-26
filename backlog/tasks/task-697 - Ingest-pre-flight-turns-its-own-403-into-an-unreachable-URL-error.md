---
id: TASK-697
title: Ingest pre-flight turns its own 403 into an unreachable-URL error
status: To Do
assignee: []
created_date: '2026-07-26 14:12'
labels:
  - library
  - ingest
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The ingest pre-flight probes a URL before accepting it and reports any failure as 'URL unreachable'. Sites that reject unadorned requests answer 403, so a perfectly ingestible page is blocked with a message stating it cannot be reached. Confirmed with a Wikipedia article that a tldw server clipped successfully moments later. A probe that cannot verify a URL should not be able to veto it outright.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A URL that answers 403 to the probe is not reported as unreachable,A page the server can fetch is not blocked by the local pre-flight,A genuinely unreachable URL is still reported as such
<!-- AC:END -->
