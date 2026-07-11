---
id: TASK-163
title: Slim remaining eager heavy imports at app startup
status: To Do
assignee: []
created_date: '2026-07-11 22:02'
labels:
  - follow-up
  - performance
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
F3 dropped the ingest chain from ~5.5s to ~0.9s, but app boot still eagerly loads nltk/torch/transformers via Web_Scraping/Article_Extractor_Lib (eager Summarization_General_Lib import) and Utils/optional_deps (eager check_dependency probe). Defer these to shed the remaining startup weight (~1s+).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 import tldw_chatbook.app no longer loads nltk/torch/transformers at boot
- [ ] #2 A subprocess regression test pins the absence
- [ ] #3 No feature regression from the deferral
<!-- AC:END -->
