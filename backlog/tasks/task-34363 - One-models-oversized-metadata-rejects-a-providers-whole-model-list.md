---
id: TASK-34363
title: One model's oversized metadata rejects a provider's whole model list
status: Done
assignee:
  - '@claude'
created_date: '2026-10-04 18:27'
updated_date: '2026-10-05 00:44'
labels:
  - providers
  - discovery
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
OpenAI-compatible discovery bounds each model's metadata (depth, item count, serialized size, value types) and, when any single model exceeds a bound, rejects the entire listing. Live on 2026-10-04, Vercel AI Gateway's public /models lists 0 of its 407 models on dev for every user: seven models (e.g. openai/gpt-5.6-sol) carry tiered pricing that brings their metadata to 270 items against a 256-item bound, and the user sees "The models endpoint did not return a valid OpenAI-compatible response." TASK-34361 fixed only the oversized-text-field case Together hit. The metadata is used only for an inferred-vision hint, so one model's details should never cost a user the whole list. The no-key probe test passed for Vercel because its fixture keeps model ids and drops metadata.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Vercel AI Gateway discovery lists all of its models against the real API
- [x] #2 A model whose metadata exceeds any bound is still listed, and none of its unbounded metadata is kept
- [x] #3 Invalid model ids still reject the listing, and the model-count and response-size bounds are unchanged
- [x] #4 Models with metadata inside every bound keep it exactly as before
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
normalize_models_response (LLM_Provider_Catalog/openai_compatible_model_discovery.py) now catches a model's metadata ValueError and keeps that model with empty metadata, instead of rejecting the whole listing. Every metadata bound still applies: none of an oversized model's details is stored. Model-id validation, the model-count cap and the response-size bound still reject the listing as before. Metadata feeds only the inferred-vision hint (model_discovery_merge._metadata_has_positive_capability), so an affected model shows capability "unknown". TASK-34361's field-level drop of an oversized text value stays, so Together's 30 templated models keep their other details.

Live on 2026-10-04 (no key): Vercel AI Gateway went from invalid_response with 0 models (dev) to 407 models, 7 of them kept without details. The other 9 discoverable public listings are unchanged (deepinfra 183, kilo 401, nous 425, novita 121, nvidia 81, ollama_cloud 17, sambanova 6, venice 128, zenmux 201). Command Code and OpenCode Zen report unsupported by design (discovery_route=None).

Tests: test_normalize_models_rejects_unbounded_metadata became test_normalize_models_keeps_a_model_but_none_of_its_unbounded_metadata. It covers depth, item count, an oversized string in a list, a non-finite float and Vercel-shaped tiered pricing, and checks that a sibling model keeps its metadata. Mutation-checked: removing the fallback fails all 5 cases. The no-key evidence test passed for Vercel because its fixture keeps only ids; this is recorded in lessons-live-verification.md.

Review follow-up (owner rule: fix every finding, minor included): the fallback no longer drops a model's whole metadata. _bounded_model_metadata drops a top-level field that breaks a bound on its own, then the largest field until the rest fit, so nothing unbounded is kept and the model's other details stay. Live 2026-10-04 (no key): Vercel lists 407 models, none with empty metadata, 7 without pricing; openai/gpt-5.6-sol keeps its other 20 fields. Mutation-checked (returning {} fails 8 cases). The inferred-vision hint is False for every Vercel model for an unrelated reason, filed as its own task: Vercel gives modalities as a mapping, which the hint does not read.

Qodo round (2026-10-05, after its credits returned): the partial drop was quadratic. A model with ~100k tiny top-level fields kept discovery busy (20,000 fields measured at 12 s). _bounded_model_metadata now returns no metadata when a model has more top-level fields than MODEL_METADATA_MAX_ITEMS (it cannot fit), measures each surviving field once, and drops largest-first in a single sorted pass. 100,000 fields now take 0.05 s. Tests: a 100k-field model finishes under 2 s with empty metadata; 200 small fields drop largest-first. Mutation-checked.

Governance clarification for the reconciliation merge: existing ADR-002 and ADR-020 now state the bounded per-field metadata policy, unchanged strict ID/list/response bounds, credential scrub and capability-hint-only use. No new ADR, provider authority or persistence mode is introduced. The separate Vercel mapping-shaped vision-hint task remains open; preserving its metadata does not implement that task.
<!-- SECTION:NOTES:END -->
