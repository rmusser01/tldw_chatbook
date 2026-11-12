---
id: TASK-34361
title: Model discovery rejects Together's bare-array /models listing
status: Done
assignee:
  - '@claude'
created_date: '2026-10-04 17:52'
updated_date: '2026-10-04 18:01'
labels:
  - providers
  - discovery
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Together's GET /v1/models answers 200 with a bare JSON array of model objects (264 on 2026-10-04) instead of the OpenAI {"data": [...]} envelope. The app's OpenAI-compatible discovery accepts only the envelope, so every Together discovery fails with "The models endpoint did not return a valid OpenAI-compatible response." Together ships with no seed models and relies on discovery (auto_refresh), so a Together user gets an empty model list and must type a model id by hand. Found by the TASK-33640 keyed capture; the no-key probe could not see it because the listing needs a key.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Discovery against Together's real listing returns its models instead of an invalid_response error
- [x] #2 Listings in the {"data": [...]} envelope behave exactly as before, including pagination and the model-count cap
- [x] #3 A bare array that is not a list of model objects is still rejected as invalid_response
- [x] #4 A captured Together listing is replayed offline as a regression fixture
- [x] #5 A model whose metadata holds one oversized text field is still listed, with only that field dropped; every other metadata bound still rejects
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Together model discovery now lists all 264 models (live, 2026-10-04); on dev it returned invalid_response and an empty list. Two causes, both in LLM_Provider_Catalog/openai_compatible_model_discovery.py:

1. Together answers /v1/models with a bare JSON array. Right after the body parses, a bare array is wrapped as {"data": [...]}, so the count cap, pagination (no has_more, so one page) and per-item checks run unchanged. An array of non-objects still fails as invalid_response.
2. 30 of Together's 264 model objects carry config.chat_template up to 16,317 characters, over MODEL_METADATA_MAX_VALUE_CHARS (4096); one oversized field rejected the whole listing. An oversized text value inside a metadata mapping is now dropped, like a credential-looking key, and the model is kept. Every other bound still rejects (depth, item count, serialized size, an oversized string inside a list). This is the same mis-calibration as the OpenAI-128 and OpenRouter-456 count caps recorded in that file. I rewrote test_normalize_models_rejects_unbounded_metadata's {"large": "x"*5000} case on purpose; its list variant keeps the rejection pinned.

The capture script (Tests/fixtures/cloud_live/capture.py) reads both listing shapes through one _listed helper and records models_response.envelope. Tests/LLM_Calls/test_live_capture_replay.py serves each capture's sample back in its recorded shape through the real discover_openai_compatible_models. Mutation-checked: removing either fix fails the new tests. The no-key probe could not see either problem because Together's listing needs a key (401).
<!-- SECTION:NOTES:END -->
