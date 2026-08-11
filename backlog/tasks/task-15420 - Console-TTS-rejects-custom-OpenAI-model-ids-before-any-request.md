---
id: TASK-15420
title: >-
  Console TTS rejects custom OpenAI model ids before any request, breaking
  external OpenAI-compatible servers
status: Done
assignee: []
created_date: '2026-08-11 12:00'
labels:
  - tts
  - speech
  - bug
  - uat
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live UAT on `origin/dev` `82b595049` (2026-08-11), reproducing a user report that
TTS settings "don't work" for an externally hosted OpenAI-compatible server.

**Every Console/default speak with a custom OpenAI model id fails before any HTTP
request is made.** The admission layer builds
`internal_model_id = f"openai_official_{request.model}"`
(`TTS/request_admission.py` `_legacy_request`) and then
`resolve_legacy_route()` looks it up in the closed `LEGACY_ROUTES` dict
(`TTS/legacy_bridge.py`, `OPENAI_INTERNAL_IDS` = tts-1 / tts-1-hd / tts1 /
tts1hd only). Any other model id raises
`UnknownLegacyModelError("The selected TTS model is not available")`.

This is the TASK-2260 defect class reintroduced one layer up: the
`OpenAITTSBackend` passthrough for custom endpoints (PR #1332) is intact but
**unreachable** — the request dies in admission before the backend is even
constructed. The user guide (`Docs/User_Guide/openai-compatible-tts.md`)
explicitly instructs users to set Model policy = Exact with *their server's*
model name, so following the docs guarantees the failure.

Evidence (mock OpenAI-compatible server recording all requests; clean profile via
`TLDW_CONFIG_PATH`):

- Settings ▸ Speech & TTS saved Base URL / exact model `mock-model` / exact voice
  `mock-voice` correctly to `app_tts` (config verified on disk).
- Console 🔊 on an assistant message → zero requests at the server; in-app log:
  `TTS_Events.tts_events ERROR TTS generation failed (outcome_code=generation_failed)`.
- Headless `service.synthesize_default(...)` with the same config → traceback
  ending `legacy_bridge.UnknownLegacyModelError` from
  `request_admission.py:600 resolve_legacy_route`.
- Counterfactual: model set to `tts-1`, everything else identical → request
  reached the custom Base URL with `voice: "mock-voice"` passed through and no
  `Authorization` header when keyless. **Custom voice, custom base URL, and
  keyless operation all work; the model-id route allowlist is the only blocker.**

Note for the fix: the route only needs the *provider*, which the effective
selection already knows (`selection.provider_id == "openai"`); deriving routing
from the model string re-imposes an official-catalog constraint that custom
endpoints must not have.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: admission-level test (`test_tts_request_admission.py`) — global default
   provider openai + exact custom model/voice → `synthesize_default` must reach
   the adapter with both passed through; unit test
   (`test_legacy_bridge.py`) — `resolve_legacy_route("openai_official_<custom>")`
   routes to provider openai. Watch both fail on `UnknownLegacyModelError`.
2. GREEN: in `legacy_bridge.resolve_legacy_route`, when the id is absent from
   `LEGACY_ROUTES` but carries the `openai_official_` prefix with a non-empty
   model suffix, route to provider "openai" (the provider is fully determined
   by the prefix; the model string itself is the server's business). A bare
   `openai_official_` still rejects. No route-table changes.
3. Deliberately flip the pinned rejection of `openai_official_new-model` in
   `test_resolver_rejects_unlisted_internal_ids` (same precedent as
   task-2260's org-header flip).
4. Verify downstream: `BackendRegistry.get` wildcard (`openai_official_*`)
   accepts arbitrary suffixes; official-endpoint guardrails live in the
   backend and are untouched.
5. Live re-verification of the original repro: headless
   `synthesize_default` + full TUI Console 🔊 against a mock
   OpenAI-compatible server recording requests.
<!-- SECTION:PLAN:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] With a custom OpenAI Base URL and an exact model name outside OpenAI's official list, Console 🔊 speech reaches the configured server with the model and voice passed through unmodified
- [x] The official-endpoint guardrails (model/voice fallback against api.openai.com) are unchanged
- [x] A regression test covers the admission path (not just the backend) with a non-official model id
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
TDD: both new tests were watched failing on the real defect
(`UnknownLegacyModelError` at `legacy_bridge.py:257`) before the change.

**Fix** (`tldw_chatbook/TTS/legacy_bridge.py`, `resolve_legacy_route`): when an
internal model id is absent from `LEGACY_ROUTES` but carries the
`openai_official_` prefix with a non-empty model suffix, route it to provider
"openai" by prefix. The provider is fully determined by the prefix; the model
string is the server's business (custom OpenAI-compatible endpoints define
their own names — TASK-2260). A bare `openai_official_` still rejects; the
route table, all other providers, and the backend's official-endpoint
guardrails (model/voice fallback against api.openai.com, decided by
`_is_official_openai_endpoint` in `TTS/backends/openai.py`) are untouched.
`BackendRegistry.get`'s existing `openai_official_*` wildcard already accepted
arbitrary suffixes downstream.

**Tests**: new admission-level test
(`test_tts_request_admission.py::test_openai_exact_custom_model_default_is_admitted_with_passthrough`)
drives `synthesize_default` with global exact custom model/voice and asserts
both reach the adapter unmodified; new unit test
(`test_legacy_bridge.py::test_resolver_routes_custom_openai_models_by_prefix`).
Two deliberate expectation flips: `openai_official_new-model` removed from
`test_resolver_rejects_unlisted_internal_ids` (bare-prefix case kept), and
`test_briefing_audio_synthesis.py`'s realistic UnknownLegacyModelError specimen
became unreachable through `synthesize_turn` entirely (unknown providers are
rejected earlier; all admitted selections now map to routable ids), so that
test now injects the error at the established fake seam to keep pinning the
wrapping contract, with the reachability change documented in its docstring.

**Verified live** against a mock OpenAI-compatible server recording requests:
the exact pre-fix repro (Console 🔊, global exact model `mock-model`, voice
`mock-voice`, custom Base URL) now produces "Speaking message." and the server
receives `{"model": "mock-model", "voice": "mock-voice", ...}`; before the fix
the same click failed pre-request with outcome_code=generation_failed. Test
sweep: branch-relevant files 218 passed; full Tests/TTS + dependents 2901
passed with 4 failures + 1 error reproduced byte-identically on pristine
origin/dev (pre-existing, unrelated); repo-wide `--collect-only` clean (37,685
collected). Docs: user-guide verification trailer updated with the re-verify
stamp and the affected-window note.
<!-- SECTION:NOTES:END -->
