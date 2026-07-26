---
id: TASK-686
title: fal/Gemini image backends follow-ups
status: In Progress
assignee: []
created_date: '2026-07-26 03:30'
updated_date: '2026-07-26 05:54'
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
- [x] #1 Reference-image size cap validates `len(content)` rather than trusting the caller-supplied `bytes_len` field, and empty `content=b""` is refused at the choke point (harden BEFORE the Console attach-UX program opens a real production constructor; today both adapters fail loudly anyway).
- [x] #2 fal `_app_id` handles fal's `APP_NAMESPACES` endpoint grammar (`workflows/...`, `comfy/...` prefixes shift the owner/app segments by one) — today a namespaced model would misdiagnose as "fal queue URL shape changed" (loud, but wrong diagnosis).
- [ ] #3 Pre-existing (out of this program's diff): the chat-side Google path (`LLM_Calls/LLM_API_Calls.py` ~:2906) sends `x-goog-api-key` through a redirect-following `requests.Session` — outside the guarded-helper credential-strip net that now protects the image path. Route it through guarded transport or strip on redirect.
- [x] #4 Per-backend keyring-source loader tests for `fal`/`gemini` (generic keyring path is covered; sibling backends share the same gap).
- [x] #5 `fetch_bytes_via_post`'s docstring cites Fireworks (dropped) as its example consumer — reword to a generic bytes-returning-POST description. Note the helper currently has NO production consumer (kept: tested, guarded, and the class of API it serves recurs).
- [x] #6 Gemini 429 (quota) and fal 403 (locked/exhausted-balance) surface generic httpx text — add friendlier enriched messages ("rate limited / image quota exhausted — free-tier caps apply"; "fal account locked or out of balance — top up at fal.ai") alongside the 404 enrichment pattern (live-UAT observations; both raw bodies carry actionable detail our sanitization rightly drops — the enrichment can name the CATEGORY without echoing bodies).
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Engine cluster (one unit): choke-point hardening (len(content) + empty-content refusal), fal APP_NAMESPACES app-id grammar, per-backend keyring tests, fetch_bytes_via_post docstring, gemini-429/fal-403 enriched messages.
2. Chat-side x-goog-api-key redirect gap (LLM_API_Calls.py Google path) as its own unit — guarded transport or redirect-strip.
Each unit: TDD, per-unit review; PR + merge per the standing procedure.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
### Unit 1 (engine cluster) -- 2026-07-25

Implemented the 5 engine-side follow-ups (AC #1, #2, #4, #5, #6). AC #3
(chat-side `x-goog-api-key` redirect gap in `LLM_Calls/LLM_API_Calls.py`) is
unit 2, tracked separately -- status stays In Progress until that lands.

- **Choke-point hardening** (`request_validation.py::_validate_reference_image`):
  the size cap now validates `len(content)`, never the caller-supplied
  `bytes_len` field (a mismatched constructor could previously bypass the
  cap by lying about `bytes_len` while `content` was actually oversized).
  `content is None` and `content == b""` are now refused identically
  ("reference image has no content bytes"); since both states derive from
  the same `content` field they're mutually exclusive with the oversize
  check (`elif`, not independent `if`s like the backend/mime checks).
  Rewrote the two existing size-boundary tests to use real content bytes
  instead of a bare `bytes_len` claim, split the old combined
  "multiple problems" test into a no-content variant and an oversize
  variant (they can no longer both fire at once), and added a dedicated
  red test for the lying-`bytes_len` bypass plus an empty-bytes test.

- **fal `APP_NAMESPACES` grammar** (`fal_image_adapter.py::_app_id`):
  re-verified live against fal_client's own SDK source
  (https://github.com/fal-ai/fal/blob/main/projects/fal_client/src/fal_client/client.py,
  `APP_NAMESPACES = ["workflows", "comfy"]` at line 718,
  `AppId.from_endpoint_id` at lines 743-761, queue-URL builder at lines
  1439-1440) -- confirms the Task-6 report's finding. A namespaced path
  (`workflows/...`, `comfy/...`) keeps its first THREE segments
  (`namespace/owner/alias`) instead of two; anything after that is dropped
  the same way a variant segment is dropped for plain paths. Implemented
  exactly this in `_app_id` (namespace check first, 3-segment floor +
  slice when namespaced, unchanged 2-segment behavior otherwise). Added
  unit tests for both namespace prefixes, the 4-segment-drops-to-3 case,
  the too-few-segments error, an explicit "plain path unchanged" pin, an
  end-to-end poll-URL test through `generate()`, and a namespaced variant
  of the vendor `status_url` cross-check test (the cross-check property
  itself needed no code change -- it just needed proof it still holds for
  the new shape).

- **Per-backend keyring tests** (`test_config_loader.py`): added
  `test_fal_key_sources_keyring` / `test_gemini_key_sources_keyring`
  following the existing `test_key_sources_keyring` (novita) pattern --
  pure coverage, no production code change (the shared
  `_resolve_secret`/`_keyring_get` machinery already handled both
  backends correctly).

- **`fetch_bytes_via_post` docstring** (`http_client.py`): reworded the
  Fireworks-specific example-consumer mention to a generic
  bytes-returning-POST description, and noted explicitly that no adapter
  in this codebase currently calls the helper (kept: guarded, tested, and
  the class of API it serves recurs across vendors).

- **Enriched vendor-wall messages**: added a 429 `elif` branch in
  `gemini_image_adapter.py` (before the generic wrapper, same shape as the
  existing 400/404 enrichment) naming the model id and
  "rate limited or image quota exhausted (free-tier caps apply)"; added a
  403 `elif` branch in `fal_image_adapter.py::_submit` (before the generic
  wrapper, same shape as the existing 404 enrichment) with
  "fal account locked or out of balance -- top up at
  fal.ai/dashboard/billing". Both are CATEGORY-level only -- never echo
  the response body; red tests assert the enrichment text is present and
  an injected body-detail marker is absent, and the existing 500/other
  generic-message tests stay green unmodified.

Verification: `Tests/Image_Generation/` full suite (251 passed, 9 skipped),
`ruff check` on all touched files (clean), `python -c "import
tldw_chatbook.app"` (imports cleanly). Files touched: `request_validation.py`,
`adapters/fal_image_adapter.py`, `adapters/gemini_image_adapter.py`,
`http_client.py`, `Tests/Image_Generation/test_request_validation.py`,
`test_fal_adapter.py`, `test_gemini_adapter.py`, `test_config_loader.py`.
<!-- SECTION:NOTES:END -->

<!-- SECTION:NOTES:END -->

<!-- SECTION:NOTES:END -->
