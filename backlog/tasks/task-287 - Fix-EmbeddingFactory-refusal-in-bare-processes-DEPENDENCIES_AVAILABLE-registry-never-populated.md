---
id: TASK-287
title: >-
  Fix EmbeddingFactory refusal in bare processes - DEPENDENCIES_AVAILABLE
  registry never populated
status: Done
assignee: []
created_date: '2026-07-17 14:56'
labels:
  - rag
  - embeddings
  - bug
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The lazy optional-deps registry (Utils/optional_deps.py DEPENDENCIES_AVAILABLE) starts all-False and nothing populates it in a bare process, so Embeddings/Embeddings_Lib.EmbeddingFactory raises a false 'dependencies not installed' ImportError even when the embeddings_rag extra is fully installed. Discovered during task-246 (PR #656): the config-layer probe now populates the registry as a side effect on the auto path, which masks the bug there, but any path that reaches EmbeddingFactory without first running vector-store type resolution (e.g. explicit type=memory config, direct service construction in scripts/tests) still hits the false refusal. The cheap probe embeddings_rag_deps_installed() (find_spec) exists since #656 — EmbeddingFactory's gate should use it or trigger real registry population instead of trusting a registry nothing fills. Pre-existing on dev before the RAG program; verified identical on unmodified origin/dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 EmbeddingFactory constructs successfully in a bare process with embeddings deps installed regardless of which code path reaches it first
- [x] #2 Without the extra installed the ImportError message still points at pip install tldw_chatbook[embeddings_rag]
- [x] #3 A regression test constructs the factory in a subprocess (or with a reset registry) for both cases
<!-- AC:END -->


## Closeout audit 2026-10-11

Board-hygiene flip (cohort A, Jun-Jul 2026 status sweep). **Verdict: DONE** — evidence verified against origin/dev (a3c47c6c72).

The availability gate was moved off the pristine-False registry: Embeddings_Lib._embeddings_rag_available() delegates to optional_deps.lazy_embeddings_rag_available() ('instead of reading DEPENDENCIES_AVAILABLE["embeddings_rag"] directly and getting stuck on a stale pristine-False default', Embeddings_Lib.py:130-163; lifted to a shared seam in task-638, commit d639ec6f4d).
