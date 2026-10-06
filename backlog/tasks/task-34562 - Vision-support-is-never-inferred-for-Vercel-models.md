---
id: TASK-34562
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

### 2026-10-06 landed-task collision

This unlanded Vision task moves from TASK-34410 to TASK-34562. Landed dev `76d5d157aa6a628584ead573976c9adee72f5412` owns TASK-34410 for skipping empty frozen-workspace registry admission (first add `58b6611bb54af8854afb73b26d3489634798389f`), so landed-keeps-id applies. This file arrived at TASK-34410 through PR3019 reconciliation `34cf8b4fed291378f098afd3542b7fe94b6bbee1`; the Vision original add `d3a625ea8e573cebc87b4a6e78f12405759153e8` and historical TASK-34402 to TASK-34410 provenance above remain factual. No current-tree title twin or live external inbound Vision reference exists. The read-only sweep of 318 refs and 40 worktrees found maximum 34560. TASK-34562 remains provisional until installation-time recheck. Status, creation date, description and acceptance criteria are unchanged; the landed empty-authority task, QA records and incident lesson keep TASK-34410.
