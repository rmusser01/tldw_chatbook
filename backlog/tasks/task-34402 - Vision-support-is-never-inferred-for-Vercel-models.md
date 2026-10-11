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
updated_date: 2026-10-11 17:18
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Discovery infers a model's vision capability from its listing metadata (model_discovery_merge._metadata_has_positive_capability), reading modalities or input_modalities as a plain list. All 407 models in Vercel AI Gateway's listing (2026-10-04) give modalities as a mapping, {"input": ["text", "image", ...], "output": [...]}, so no Vercel model is ever marked as inferred-vision, including ones that list image input. Found while checking TASK-34363's metadata fallback live.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A model whose listing gives modalities as a mapping with an input list that includes image is inferred as vision-capable
- [x] #2 The flat list-shaped modalities and input_modalities hints still infer vision, and any one hint that lists image input is enough
- [x] #3 A model whose listing nests input_modalities under architecture (OpenRouter's shape) with image in it is inferred as vision-capable
- [x] #4 The Settings discovery list shows a freshly listed model with an image-input hint as capabilities inferred
- [x] #5 A model whose listing carries a true supports_image_input flag (Fireworks), a true vision flag under capabilities (Mistral), or capabilities.image_input.supported true (Anthropic) is inferred as vision-capable
- [x] #6 When a listing's metadata has to shrink to fit its bounds, the field carrying the vision hint is dropped last
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace the hint from the listing through the bounded metadata to the inference function
2. Pin the mapping, nested and flat-list shapes with failing tests, one through the real listing normaliser
3. Read every shape in the one inference function; count only the input side of a mapping
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Discovery's inference function (model_discovery_merge._metadata_has_positive_capability) reads the listing shapes it used to miss, and the Settings discovery list now applies the inference too.

Shapes read, with the evidence for each:
- Vercel's modalities mapping, input side only. Live listing 2026-10-11: 415 of 415 models use it.
- OpenRouter's architecture.input_modalities. Live listing 2026-10-11: 458 of 458 models use it.
- Fireworks' supports_image_input flag. In the repo's capture (Tests/fixtures/cloud_live/fireworks.json): 7 of 20 sampled models.
- A capabilities block: a vision flag (Mistral's API reference, read 2026-10-11) or image_input.supported (Anthropic's Models API, from the Claude API reference). Neither is captured in the repo or checked against a live listing.
- The flat modalities and input_modalities lists, as before.

Measured through the real listing normaliser, before and after: Vercel 0 -> 193 models inferred, OpenRouter 0 -> 296. Both equal the number of models whose raw listing names image input. The two public listings were fetched without a key; nothing was saved to the repo.

Behaviour notes:
- The hints are any-of. Before, a flat modalities list without an image entry ended the reading, so a listing with modalities [text] and input_modalities [text, image] was unknown. It is inferred now (pinned by a test).
- Wherever a listing separates input from output, an image on the output side never counts. A flat modalities list does not say which side it is and is read as input, as it always was.
- A flag counts only when it is exactly true; a string or a number does not.
- A listing that declares text-only input stays unknown. An explicit negative has never made a model known or inferred (test_vision_false_does_not_make_capabilities_known pins that), and this task does not change it.
- When a model's metadata has to shrink to fit the normaliser's bounds, the fields the inference reads are now dropped last. No live record needed this (0 of 873); a synthetic one with 200 small fields lost its capabilities block before.
- Production wires no capability resolver, so before this change every discovered model from these providers was labelled capability unknown.

What the user sees: in the session that fetched a listing, models with image input lose the capability unknown label and warning in the model pickers, and the Settings discovery list says capabilities inferred. Limit: the on-disk catalog keeps ids and fetch times only, so after a restart inside the cache's freshness window (24 hours by default) the label is back until the next fetch. That was already true of the flat-list hints; TASK-34793 tracks it.

Tests: Tests/LLM_Provider_Catalog and Tests/UI/test_settings_provider_test_draft.py, 528 passed. One other test in that Settings file (test_t_hotkey_does_not_run_test_while_input_focused) fails on this machine on clean dev as well. None of the three changed test files is in the PR gate, so the required CI check does not run the new tests; the evidence for them is the local run and two independent review rounds with mutation checks.

Files: tldw_chatbook/LLM_Provider_Catalog/model_discovery_merge.py, tldw_chatbook/LLM_Provider_Catalog/openai_compatible_model_discovery.py, tldw_chatbook/UI/Screens/settings_screen.py, Tests/LLM_Provider_Catalog/test_model_discovery_cache_merge_persistence.py, Tests/LLM_Provider_Catalog/test_openai_compatible_model_discovery.py, Tests/UI/test_settings_provider_test_draft.py
<!-- SECTION:IMPLEMENTATION_NOTES:END -->