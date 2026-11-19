---
id: TASK-409
title: Guard or scope the retrieve-step FTS5 branch latent TypeError
status: Done
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
- [x] #1 A custom TOML pipeline with a bare retrieve step over notes/conversations FTS either works or fails at load with a clear message (no runtime TypeError)
- [x] #2 Regression test covers the branch
<!-- AC:END -->


## Closeout audit 2026-10-11

Board-hygiene flip (cohort A, Jun-Jul 2026 status sweep). **Verdict: DONE** — evidence verified against origin/dev (a3c47c6c72).

The retrieve-step FTS5 branch now guards the argument shape: 'Only search_media_fts5 accepts keyword_filter' — search_conversations_fts5/search_notes_fts5 are called WITHOUT the 4th positional (pipeline_builder_simple.py:232-250); landed in the RAG-scope review wave (3b98e27557).
