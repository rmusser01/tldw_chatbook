---
id: TASK-26023
title: Model metadata and pricing from an upstream catalog
status: To Do
assignee: []
created_date: '2026-08-31 15:45'
labels:
  - providers
  - ops
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Model capabilities and prices are hand-maintained and go stale silently. Verified on origin/dev: model_capabilities.py:27-45 infers vision support and context window from a regex pattern table, and LLM_Calls/pricing_catalog.py:41-48 carries a hand-seeded price table with a _SEED_AS_OF staleness stamp that a human must re-verify. Every new model release needs a code change. Hermes pulls models.dev, which covers 4000-plus models with ETag conditional GET, disk cache and stale-while-revalidate. Chatbook already has the expensive half: memory cache, disk cache, merge and persistence modules under LLM_Provider_Catalog/.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Model context window, capability flags and per-token pricing can be sourced from an upstream catalog
- [ ] #2 Upstream data is a lower-priority merge layer beneath existing hand-maintained entries, so a deliberate local override always wins
- [ ] #3 The catalog is fetched with conditional requests and cached to disk; a cold start with no network uses the cache and then the hand-seeded values
- [ ] #4 No network call occurs on a hot path - fetching is background or explicit, never blocking a send
- [ ] #5 The origin of a displayed price or window is inspectable, so a wrong number can be traced to its source
- [ ] #6 The existing honest cost-unavailable behavior is preserved: an unknown price still refuses to fabricate a figure
- [ ] #7 Fully offline operation is unchanged
<!-- AC:END -->
