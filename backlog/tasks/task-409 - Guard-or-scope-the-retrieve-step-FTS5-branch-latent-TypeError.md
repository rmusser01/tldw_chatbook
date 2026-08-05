---
id: TASK-409
title: Guard or scope the retrieve-step FTS5 branch latent TypeError
status: To Do
assignee: []
created_date: '2026-07-21 09:48'
labels:
  - rag
  - bug
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Pre-existing latent bug confirmed dead-but-armed by two reviews (task-256's PR 681 noted it; the RAG-scope whole-branch review re-verified): pipeline_builder_simple's retrieve-step FTS5 branch passes a 4th positional keyword_filter arg that search_notes_fts5/search_conversations_fts5 signatures never accepted — any custom TOML pipeline using a bare retrieve step with those functions crashes with TypeError. Currently unreachable via builtins (they expand to parallel steps). Either fix the call to match signatures or fail loudly at pipeline load with a clear validation error.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A custom TOML pipeline with a bare retrieve step over notes/conversations FTS either works or fails at load with a clear message (no runtime TypeError)
- [ ] #2 Regression test covers the branch
<!-- AC:END -->
