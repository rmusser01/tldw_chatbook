---
id: TASK-31684
title: Wait for the mounted Audio.cpp curated handoff presentation
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 18:24'
updated_date: '2026-09-05 18:34'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the slow-load handoff test precondition: active_view selection precedes lazy CuratedView mounting and its loading request.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The slow-load test waits for the actual mounted CuratedView and the first request before asserting exactly-once loading and retry behavior.
- [x] #2 The complete Audio.cpp handoff suite passes with unchanged timeout bounds and runtime behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce active_view becoming curated before the lazy child exists or its loading callback runs. 2. Wait for a mounted CuratedView and its first loading attempt using the existing bounded helper, then keep exact one-attempt/filter/loading/error/retry checks. 3. Run complete Audio.cpp handoff tests and static checks. ADR required: no. ADR path: N/A. Reason: test-only first-use readiness; no loading/runtime policy change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Required the current CuratedView to be mounted and its first request to exist before the slow-loading presentation assertions. Baseline reproduced missing child and prior broad run saw no loading attempt; active_view is only selection intent. Exactly-once request, audio filter, loading/error copy and retry checks remain unchanged. Complete122case file passed289.56s (/private/tmp/tldw-review-audio-cpp-handoff-final-20260905.xml). Ruff, changed-range format, diff whitespace and self-review passed; no ADR/runtime change.
<!-- SECTION:NOTES:END -->
