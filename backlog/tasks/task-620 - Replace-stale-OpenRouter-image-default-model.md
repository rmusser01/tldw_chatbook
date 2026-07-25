---
id: TASK-620
title: >-
  Replace stale OpenRouter image default model (gpt-image-1 404s)
status: In Progress
assignee: []
created_date: '2026-07-25 10:15'
updated_date: '2026-07-25 10:45'
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
- [ ] The shipped default model is one that exists on OpenRouter today (e.g. `google/gemini-2.5-flash-image` or `openai/gpt-5-image-mini` — pick with cost in mind and document the choice).
- [ ] A 404 from OpenRouter's chat/completions on an image request produces an error message that names the model id and suggests checking `[image_generation.openrouter] default_model` (users cannot diagnose "404 on the endpoint URL").
- [ ] The shipped default-config example and any docs naming gpt-image-1 are updated.
- [ ] Existing Image_Generation suites stay green; a test pins the improved 404 messaging.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:NOTES:END -->
