---
id: TASK-686
title: >-
  fal/Gemini image backends follow-ups
status: To Do
assignee: []
created_date: '2026-07-26 03:30'
updated_date: '2026-07-26 03:30'
labels:
  - image-generation
  - followup
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Non-blocking follow-ups from the fal.ai + Gemini image-backend program's final whole-branch review and live UAT (spec `Docs/superpowers/specs/2026-07-26-imagegen-fal-gemini-fireworks-design.md`; Fireworks was dropped mid-program — vendor deprecated image generation 2026-06-10). None are shipped defects.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] Reference-image size cap validates `len(content)` rather than trusting the caller-supplied `bytes_len` field, and empty `content=b""` is refused at the choke point (harden BEFORE the Console attach-UX program opens a real production constructor; today both adapters fail loudly anyway).
- [ ] fal `_app_id` handles fal's `APP_NAMESPACES` endpoint grammar (`workflows/...`, `comfy/...` prefixes shift the owner/app segments by one) — today a namespaced model would misdiagnose as "fal queue URL shape changed" (loud, but wrong diagnosis).
- [ ] Pre-existing (out of this program's diff): the chat-side Google path (`LLM_Calls/LLM_API_Calls.py` ~:2906) sends `x-goog-api-key` through a redirect-following `requests.Session` — outside the guarded-helper credential-strip net that now protects the image path. Route it through guarded transport or strip on redirect.
- [ ] Per-backend keyring-source loader tests for `fal`/`gemini` (generic keyring path is covered; sibling backends share the same gap).
- [ ] `fetch_bytes_via_post`'s docstring cites Fireworks (dropped) as its example consumer — reword to a generic bytes-returning-POST description. Note the helper currently has NO production consumer (kept: tested, guarded, and the class of API it serves recurs).
- [ ] Gemini 429 (quota) surfaces the generic httpx text — add a friendlier enriched message ("rate limited / image quota exhausted — free-tier caps apply") alongside the 404 enrichment pattern (live-UAT observation).
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:NOTES:END -->
