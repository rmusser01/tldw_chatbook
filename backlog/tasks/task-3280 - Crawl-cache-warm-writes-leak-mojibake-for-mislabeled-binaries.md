---
id: TASK-3280
title: Crawl cache warm-writes leak mojibake for mislabeled binaries
status: To Do
assignee: []
created_date: '2026-08-08 14:15'
labels:
  - web-tools
  - tech-debt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
web_crawl's page fetches warm the shared fetch cache under the same (url, FETCH_MAX_BYTES) key web_fetch reads. Crawl deliberately does not take task-1359's binary metadata path (recorded non-goal: crawl is for page text), so a binary served with a wrong/absent content-type during a crawl gets UTF-8-replace-decoded to mojibake and CACHED — and a subsequent web_fetch of that URL returns the cached garbage instead of the binary metadata ([image]/[archive]/[audio]) it would produce on a cold fetch. Pre-existing behavior class (web_fetch itself returned mojibake for these before 1359), but the shared cache now leaks the crawl-side non-goal into web_fetch's NEW behavior, making the same URL return different result shapes depending on which tool touched it first. Found by the task-1359 final review (probe: crawl page linking to an unlabeled ZIP → later web_fetch returns mojibake for the cache TTL).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A web_fetch after a crawl touched the same mislabeled-binary URL returns the binary metadata shape, not cached mojibake (cache entries for binary-kind bodies are either skipped at crawl warm-write or made kind-aware)
- [ ] #2 A regression test covers the crawl-then-fetch sequence for at least one sniffable binary kind
<!-- AC:END -->
