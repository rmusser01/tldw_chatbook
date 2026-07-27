---
id: TASK-842
title: Semantic chunking depends on an undeclared NLTK download
status: Done
assignee:
  - '@claude'
created_date: '2026-07-27 02:02'
updated_date: '2026-07-27 02:31'
labels:
  - chunking
  - packaging
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The semantic chunking method needs NLTK tokeniser data that is fetched at runtime rather than declared as a dependency. Where that data is absent the method raises a lookup error naming an NLTK resource, which reads as an internal fault rather than a missing optional asset. Observed while exercising every chunking method: the same call succeeded in one environment and failed in another purely because of the cached download.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Semantic chunking either has its data available or explains what to install
- [x] #2 The failure is not presented as an internal error
- [x] #3 Other chunking methods are unaffected when the data is absent
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Semantic chunking now degrades instead of raising when its corpus is missing.

_ensure_nltk handled nltk being ABSENT -- falling back to a built-in regex sentence splitter -- but not nltk being PRESENT WITHOUT ITS DATA. In that case it bound the real tokeniser, which then raised LookupError deep inside a chunking call. That is why the identical call succeeded in one environment and failed in another: the punkt corpus is a runtime download, not part of the package, so the outcome depended on what happened to be cached.

The loader now probes the tokeniser once at bind time and keeps the existing regex fallback when the corpus is absent, logging the exact remedy: python -m nltk.downloader punkt punkt_tab. Simpler sentence splitting is a better outcome than a failed ingest, and this reuses the fallback that already existed for the nltk-absent case rather than inventing a second path.

Verified the way that settles it, by pointing NLTK_DATA at an empty directory so punkt is definitively unavailable: semantic chunking returned four usable chunks instead of raising. Regression test does the same via monkeypatch and a module reload.

Also corrected a slip in my own first draft: it used logging.warning while this module uses loguru, which would have sent the message somewhere the app's log filters do not look.

Tests/RAG + Tests/Chunking + Tests/Local_Ingestion: 810 passed, 8 skipped. The semantic case that previously had to be skipped for missing data now passes outright.

Files: Chunking/Chunk_Lib.py, Tests/RAG/test_chunking_service.py.
<!-- SECTION:NOTES:END -->
