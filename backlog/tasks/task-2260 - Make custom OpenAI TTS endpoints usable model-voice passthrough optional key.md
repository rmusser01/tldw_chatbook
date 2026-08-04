---
id: TASK-2260
title: Make custom OpenAI TTS endpoints usable (model/voice passthrough, optional key)
status: Done
updated_date: '2026-08-04 12:00'
assignee:
  - '@claude'
created_date: '2026-08-04 12:00'
labels:
  - tts
  - settings
  - chrome-honesty
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A user could not point chatbook at a local OpenAI-compatible TTS server (pocket-tts). At
investigation time the Base URL setting was dead end-to-end; by the time this task was filed,
origin/dev had already gained the persistence + backend plumbing for `OPENAI_BASE_URL` /
`OPENAI_ORG_ID` (Speech settings model → `_TTS_SETTING_BINDINGS` → `OpenAITTSBackend`, with
validation and tests). What still breaks OpenAI-compatible servers is the request itself:
`OpenAITTSBackend.generate_speech_stream` silently coerces any model outside
`tts-1`/`tts-1-hd` to `tts-1` and any voice outside OpenAI's six to `alloy`, so a custom
server's model/voice names never reach it, and `_validate_api_key` refuses to run without an
API key even though local servers are typically keyless. Scope: when a custom base URL is
configured, pass model and voice through unmodified and make the API key optional (still sent
as a Bearer header when configured). Behavior against the default OpenAI endpoint must not
change.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 With a custom base URL, the model value passes through to the server unmodified (no coercion to tts-1)
- [x] #2 With a custom base URL, the voice value passes through to the server unmodified (no coercion to alloy)
- [x] #3 With a custom base URL and no API key configured, generation proceeds and the request carries no Authorization header; a configured key is still sent as a Bearer header
- [x] #4 Against the default OpenAI endpoint, existing behavior is unchanged: model/voice coercion and the API-key requirement remain
- [x] #5 Regression tests cover all of the above and fail if any of the behaviors regress
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. TDD: add `Tests/TTS/test_openai_compatible_endpoint.py` using the MockTransport injection
   pattern from `test_legacy_backend_registry.py`; watch the passthrough/keyless tests fail
   against current coercion behavior.
2. In `TTS/backends/openai.py`: derive `is_custom_endpoint` from the validated base URL;
   when custom, skip model/voice coercion, skip `_validate_api_key`, and omit the
   Authorization header when no key is configured.
3. Keep default-endpoint behavior byte-identical; pin it with characterization tests.
4. Run the TTS test files + a `--collect-only` sweep (targeted-tests ruling; no full-suite runs).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`OpenAITTSBackend` now derives `is_custom_endpoint` from the already-validated base URL
(`base_url != _DEFAULT_OPENAI_TTS_URL`). When custom: model and voice pass through to the
server unmodified, `_validate_api_key` is skipped, and the Authorization header is only sent
when a key is actually configured (previously the header was always built, and would have
carried a literal `Bearer None`). Default-endpoint behavior is byte-identical and pinned by
characterization tests. The no-key startup warnings are also silenced for custom endpoints,
since "Requests will fail" is false there.

TDD: `Tests/TTS/test_openai_compatible_endpoint.py` — passthrough and keyless tests watched
red against the old coercion before the fix; default-endpoint pins were mutation-tested
(forcing `is_custom_endpoint = True` makes both fail). Includes one end-to-end test against a
real local HTTP server shaped like pocket-tts (keyless, custom model/voice, real socket).

Scope note: at investigation time the Base URL setting was dead end-to-end on the reviewed
branch; origin/dev had meanwhile gained the persistence/backend plumbing for
`OPENAI_BASE_URL`/`OPENAI_ORG_ID` with its own tests, so this task narrowed to the request
behaviors that still broke OpenAI-compatible servers. Discoverability docs split out to
task-2261.

Files: `tldw_chatbook/TTS/backends/openai.py`, `Tests/TTS/test_openai_compatible_endpoint.py`.
Verified: 2,195 tests in `Tests/TTS/` + speech settings model tests pass; ruff check/format
clean; repo-wide `--collect-only` sweep clean (29,821 collected).
<!-- SECTION:NOTES:END -->
