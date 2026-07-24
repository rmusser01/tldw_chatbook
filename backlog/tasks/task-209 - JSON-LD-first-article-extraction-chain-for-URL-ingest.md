---
id: TASK-209
title: JSON-LD-first article extraction chain for URL ingest
status: To Do
assignee: []
created_date: '2026-07-12 18:56'
labels:
  - follow-up
  - ingest
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Improve URL/article extraction (task 162) with a JSON-LD-first strategy: parse <script type=ld+json> (schema.org Article) for title/author/date/articleBody, falling back to trafilatura when absent/empty. Often more reliable than trafilatura heuristics on structured news/blog sites. Borrowed from tldw_server2's extract_jsonld_entities (Article_Extractor_Lib). Discovered mining the server media pipeline during task 162.
<!-- SECTION:DESCRIPTION:END -->
