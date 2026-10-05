---
id: TASK-34410
title: Vision support is never inferred for Vercel models
status: To Do
assignee: []
created_date: '2026-10-04 23:43'
labels:
  - providers
  - discovery
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Discovery infers a model's vision capability from its listing metadata (model_discovery_merge._metadata_has_positive_capability), reading modalities or input_modalities as a plain list. All 407 models in Vercel AI Gateway's listing (2026-10-04) give modalities as a mapping, {"input": ["text", "image", ...], "output": [...]}, so no Vercel model is ever marked as inferred-vision, including ones that list image input. Found while checking TASK-34363's metadata fallback live.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A model whose listing gives modalities as a mapping with an input list that includes image is inferred as vision-capable
- [ ] #2 The existing list-shaped modalities and input_modalities hints behave as before
<!-- AC:END -->


## Renumbering provenance

The older Console pause investigation first added TASK-34402 in `f6f9655a22d7b8bba6e91e10c2ea6f54d3e0b48d` (2026-10-04 19:21:59 UTC). This younger vision task first appeared in `d3a625ea8e573cebc87b4a6e78f12405759153e8` (2026-10-04 23:46:50 UTC), so it moves to TASK-34410 during the PR3019 reconciliation merge. The task status, creation date, description and acceptance criteria are unchanged. No inbound reference to the vision task existed; Console references keep their original ID. The immediate all-ref/history and 37 active-worktree filename sweep found a maximum ID of 34409. Backlog CLI1.53.0 has no ID-renumber operation, so this administrative move uses the exact verified task path.
