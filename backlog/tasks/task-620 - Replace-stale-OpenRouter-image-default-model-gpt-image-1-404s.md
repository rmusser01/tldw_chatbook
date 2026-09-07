---
id: TASK-620
title: Replace stale OpenRouter image default model (gpt-image-1 404s)
status: Done
assignee: []
created_date: '2026-07-25 10:15'
updated_date: '2026-07-25 16:52'
labels:
  - image-generation
  - bug
  - uat
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live UAT 2026-07-25 (real OpenRouter key): a fresh install configured with `default_backend = "openrouter"` fails every generation with `404 Not Found` on `/api/v1/chat/completions`. Root cause: `DEFAULT_OPENROUTER_IMAGE_MODEL = "openai/gpt-image-1"` (`Image_Generation/config.py:41`, ported verbatim from tldw_server) no longer exists on OpenRouter — the live image-output catalog is `google/gemini-2.5-flash-image`, `google/gemini-3*-image*`, `openai/gpt-5-image(-mini)`, etc. The 404 surfaces cleanly in-chat (good), but the out-of-the-box default is broken and the error gives the user no hint the model id is the problem.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The shipped default model is one that exists on OpenRouter today (e.g. `google/gemini-2.5-flash-image` or `openai/gpt-5-image-mini` — pick with cost in mind and document the choice).
- [x] #2 A 404 from OpenRouter's chat/completions on an image request produces an error message that names the model id and suggests checking `[image_generation.openrouter] default_model` (users cannot diagnose "404 on the endpoint URL").
- [x] #3 The shipped default-config example and any docs naming gpt-image-1 are updated.
- [x] #4 Existing Image_Generation suites stay green; a test pins the improved 404 messaging.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Image_Generation/config.py: change DEFAULT_OPENROUTER_IMAGE_MODEL to "google/gemini-2.5-flash-image"; add a code comment recording why (previous gpt-image-1 default 404s on OpenRouter as of 2026-07-25 live UAT) and the date/rationale for the replacement.
2. tldw_chatbook/config.py: update the shipped [image_generation.openrouter] default_model example from "openai/gpt-image-1" to the new default; grep for other gpt-image-1 references in tldw_chatbook/ and Docs/ (excluding backlog tasks and Docs/superpowers historical specs/plans) and update any.
3. adapters/openrouter_image_adapter.py: catch httpx.HTTPStatusError around fetch_json in generate(); on status_code==404, raise ImageGenerationError naming the attempted model id and pointing at [image_generation.openrouter] default_model; preserve existing generic-failure wrapping for all other exceptions/statuses.
4. TDD: add a test pinning that the new default model constant flows into the OpenRouter payload when nothing overrides it; add a test that fakes fetch_json raising the real httpx.HTTPStatusError(404) shape and asserts the adapter's ImageGenerationError message names the model + config path.
5. Run Tests/Image_Generation/, ruff check touched files, import smoke test.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced the dead OpenRouter image default and made 404s self-diagnosing.

- Image_Generation/config.py:41-46: DEFAULT_OPENROUTER_IMAGE_MODEL -> "google/gemini-2.5-flash-image" (verified live on OpenRouter's catalog 2026-07-25; picked over openai/gpt-5-image-mini on cost). Comment records the old default's death and instructs re-verifying against the live catalog before changing it again.
- tldw_chatbook/config.py:2812-2816: shipped [image_generation.openrouter] example default_model updated to match, with an inline note on why.
- adapters/openrouter_image_adapter.py: generate() now catches httpx.HTTPStatusError specifically; on status_code==404 it raises ImageGenerationError naming the exact attempted model id (payload["model"]) and pointing at "[image_generation.openrouter] default_model". Any other status/exception keeps the prior generic "OpenRouter request failed: {exc}" wrapping -- confirmed httpx.HTTPStatusError is what resp.raise_for_status() in http_client.fetch_json actually raises (it propagates uncaught out of fetch_json).
- No other product/doc references to gpt-image-1 found outside Docs/superpowers (historical, excluded per instructions) and backlog task files.

Tests added (Tests/Image_Generation/test_openrouter_adapter.py): test_openrouter_payload_uses_new_default_model_when_unconfigured (pins the new default flowing into the payload when request.model/env/config are all unset), test_openrouter_404_names_model_and_config_path (fakes fetch_json raising the real httpx.HTTPStatusError(404) shape and asserts the enriched message), test_openrouter_non_404_status_keeps_generic_message (500 stays on the generic wrapper, guarding against over-broad matching).

Verification: python -m pytest Tests/Image_Generation/ -q -> 98 passed, 6 skipped. ruff check on touched files clean (2 pre-existing unrelated F841s in tldw_chatbook/config.py at lines 757-758, confirmed present before this change via git stash). python -c "import tldw_chatbook.app" succeeds.
<!-- SECTION:NOTES:END -->
