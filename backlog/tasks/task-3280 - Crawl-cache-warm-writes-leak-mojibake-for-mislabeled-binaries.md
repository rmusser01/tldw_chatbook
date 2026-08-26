---
id: TASK-3280
title: Crawl cache warm-writes leak mojibake for mislabeled binaries
status: Done
assignee: []
created_date: '2026-08-08 14:15'
updated_date: '2026-08-08 21:27'
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
- [x] #1 A web_fetch after a crawl touched the same mislabeled-binary URL returns the binary metadata shape, not cached mojibake (cache entries for binary-kind bodies are either skipped at crawl warm-write or made kind-aware)
- [x] #2 A regression test covers the crawl-then-fetch sequence for at least one sniffable binary kind
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Closed in the same PR that activated it (#1442, task-1359), after Qodo flagged the identical finding on the diff. Fix shape: option 1 — crawl's cache warm-write now SKIPS sniffed binary kinds (`kind not in ("image", "zip", "audio")` at the `_cache_put` site in web_crawl's page loop), so web_fetch's binary-metadata path always runs cold for those URLs; PDFs keep warm-writes (crawl marks them without decoding, and web_fetch's PDF path predates this). Regression test `test_crawl_warm_cache_does_not_mask_binary_metadata_for_later_fetch` drives the real crawl-then-fetch sequence with a mislabeled sniffable PNG and asserts web_fetch takes the binary path (raises [image-error] for the magic-only body) instead of returning cached mojibake text; mutation-verified (reverting the skip turns it red).
<!-- SECTION:NOTES:END -->
