---
id: TASK-842
title: Semantic chunking depends on an undeclared NLTK download
status: To Do
assignee: []
created_date: '2026-07-27 02:02'
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
- [ ] #1 Semantic chunking either has its data available or explains what to install
- [ ] #2 The failure is not presented as an internal error
- [ ] #3 Other chunking methods are unaffected when the data is absent
<!-- AC:END -->
