---
id: TASK-1261
title: Restore the lost warning when NLTK tokenizer data is unusable
status: To Do
assignee: []
created_date: '2026-07-28 18:51'
labels:
  - bug
  - logging
  - chunking
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ensure_nltk_data() degrades silently when NLTK is installed but its sentence-tokenizer data cannot tokenize. The readiness flag is set correctly, but nothing is logged, so a user whose chunking has silently fallen back to the non-NLTK path gets no signal anywhere. Tests/Utils/test_startup_polish_regressions.py::test_nltk_download_false_is_not_logged_as_success has been failing on dev because of this; the test is correct and the code lost its warning during a refactor.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `ensure_nltk_data()` emits a WARNING or ERROR naming the missing punkt/punkt_tab corpus when the tokenizer probe fails
- [ ] #2 No "downloaded successfully" message is logged on the failure path
- [ ] #3 `Tests/Utils/test_startup_polish_regressions.py::test_nltk_download_false_is_not_logged_as_success` passes
- [ ] #4 The orphaned, over-indented comment block left at `Chunk_Lib.py:267-268` by the refactor is removed
<!-- AC:END -->
