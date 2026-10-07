---
id: TASK-34402
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
