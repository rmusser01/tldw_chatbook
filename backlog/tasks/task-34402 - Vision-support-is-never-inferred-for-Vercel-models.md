---
id: TASK-34402
title: Vision support is never inferred for Vercel models
status: Done
assignee:
- '@claude'
created_date: 2026-10-04 23:43
labels:
- providers
- discovery
dependencies: []
priority: low
updated_date: 2026-10-11 15:50
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Discovery infers a model's vision capability from its listing metadata (model_discovery_merge._metadata_has_positive_capability), reading modalities or input_modalities as a plain list. All 407 models in Vercel AI Gateway's listing (2026-10-04) give modalities as a mapping, {"input": ["text", "image", ...], "output": [...]}, so no Vercel model is ever marked as inferred-vision, including ones that list image input. Found while checking TASK-34363's metadata fallback live.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A model whose listing gives modalities as a mapping with an input list that includes image is inferred as vision-capable
- [x] #2 The existing list-shaped modalities and input_modalities hints behave as before
- [x] #3 A model whose listing nests input_modalities under architecture (OpenRouter's shape) with image in it is inferred as vision-capable
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace the hint from the listing through the bounded metadata to the inference function
2. Pin the mapping, nested and flat-list shapes with failing tests, one through the real listing normaliser
3. Read every shape in the one inference function; count only the input side of a mapping
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Discovery's one inference function (model_discovery_merge._metadata_has_positive_capability) now reads every spelling of the image-input hint: the flat modalities / input_modalities lists as before, Vercel's modalities mapping (its input list only), and OpenRouter's architecture.input_modalities.

- Traced first: the listing normaliser keeps nested mappings in the bounded metadata, so the hint already reached the function and was only ignored there. One test goes through the real normaliser to prove the frozen metadata types are read.
- Only the input side of a mapping counts. A model that outputs images but takes text stays unknown.
- AC 3 was added during the work: the repo's own OpenRouter fixture (the live API's field set) nests the hint under architecture, and no product code read that key, so OpenRouter models missing from the capability catalog had the same gap.
- Behaviour change at one edge: the hints are now any-of. Before, a flat modalities list without image stopped the function before it looked at input_modalities.
- Effect for the user: such models lose the 'capability unknown' label and warning in model pickers.

Verified: Tests/LLM_Provider_Catalog 402 passed (3 of the 4 new tests failed before the change; the flat-list test pins the old behaviour).

Files: tldw_chatbook/LLM_Provider_Catalog/model_discovery_merge.py, Tests/LLM_Provider_Catalog/test_model_discovery_cache_merge_persistence.py
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
