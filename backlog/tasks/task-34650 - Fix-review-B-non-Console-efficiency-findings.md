---
id: TASK-34650
title: Fix review-B non-Console efficiency findings
status: In Progress
assignee:
  - '@Robert'
created_date: '2026-10-07 02:57'
updated_date: '2026-10-07 02:59'
labels:
  - performance
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the 24 efficiency defects unique to the 2026-10-06 review-B performance sweep (complement of the sibling remediation): unbounded RAG content assembly, per-sample model reloads in Evals, O(BxF) notes-sync hashing, serial ingestion/research LLM pipelines, missing conversations browse index, legacy Prompts_DB N+1 searches, per-keystroke UI rebuilds, video-store walk multiplication, fork/import batching. Spec: Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation-review-b.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 RAG conversation content assembly bounded with order preserved,B4 semantic model constructed once per process,Notes sync identity fallback O(B+F) digests per pass,Ingestion chunk analysis bounded concurrency order-preserving,Research gate overlaps scrape+summarize under semaphore,conversations(last_modified DESC,id DESC) index serves browse pages,Prompts_DB searches use FTS subquery+batch keywords+LIMIT,In-memory vector store uses dict index+matrix cosine+OrderedDict LRU,UI legacy widgets debounced and render-capped,Video store single snapshot per save and RecoveredMedia reuse per call,Fork commit and history import statement counts bounded,All targeted tests green with counted evidence
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation-review-b.md waves A-F (tasks B1-B27) via per-wave implementation agents; commits per wave; PR against dev.
<!-- SECTION:PLAN:END -->
