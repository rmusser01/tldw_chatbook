---
id: TASK-638
title: Sweep remaining stale dependency flag reads search_rag_window
status: To Do
assignee: []
created_date: '2026-07-25 18:00'
labels:
  - followup
  - uat
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-657 fixed the lazy embeddings_rag gate at EmbeddingFactory, but two raw DEPENDENCIES_AVAILABLE reads in search_rag_window.py retain the stale-flag anti-pattern (cosmetic banner/guard shows deps-missing to users who have them). Also: the 657 test module's skipif over-covers test_manually_forced_unavailable_is_still_honored, which needs no extras - losing CI coverage on no-extras environments.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 search_rag_window.py dependency reads route through the lazy ensure path
- [ ] #2 The forced-unavailable invariant test runs on no-extras environments
- [ ] #3 Existing optional_deps tests stay green
<!-- AC:END -->
