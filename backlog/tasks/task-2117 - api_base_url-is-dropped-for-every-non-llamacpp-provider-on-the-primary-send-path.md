---
id: TASK-2117
title: >-
  api_base_url is dropped for every non-llamacpp provider on the primary send path
status: Done
assignee: [rmusser01]
created_date: '2026-08-03 15:10'
labels:
  - llm-calls
  - config
  - console
priority: high
dependencies:
  - TASK-2114
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-2114 fixed the Anthropic case. Investigating it revealed the cause is not
Anthropic-specific: `ConsoleProviderGateway._chat_api_kwargs` — the shared kwargs
builder for the primary Console send path — structurally drops `api_base_url` for
**every** provider except llama.cpp.

Confirmed affected: openai, cohere, deepseek, google, groq, huggingface,
mistral/mistralai, openrouter, moonshot, zai. (`llama_cpp` / `local_llamacpp` are
exempt — they take a separate direct-base_url code path.)

Severity splits in two:

- **Unmasked, live defects — google, huggingface, moonshot, mistral/mistralai.** For
  these the config key is disconnected from Console's canonical
  `[api_settings.<provider>]` section and nothing else stops the send, so a configured
  base URL is silently ignored while requests go to the default endpoint. Same failure
  shape as TASK-2114: no error, no warning, indistinguishable from a working proxy.
- **Currently masked — openai, cohere, deepseek, groq, openrouter, zai.** Console's
  "unsaved endpoint" send-gate happens to block these before the drop matters. That is
  a coincidence of an unrelated guard, not a fix; if the gate is ever relaxed or
  bypassed these become live defects too.

Fixing this centrally in `_chat_api_kwargs` is preferable to ten per-provider patches,
but each provider adapter's parameter name for the base URL must be verified rather
than assumed — the adapters are not uniform (see `PROVIDER_PARAM_MAP`).
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

1. Verify the premise at base `cddc89d3e7` (done — see notes below; narrowed to
   google + huggingface live, plus session-pinned-endpoint drops on all seven).
2. Red-first: add tests in `Tests/Chat/test_console_provider_gateway.py` driving
   `resolve_for_send` + `stream_chat` with a capturing `chat_api_call_fn`:
   configured `[api_settings.<provider>].api_base_url` reaches the adapter kwargs
   for google/huggingface (unmasked) and the unconfigured default stays unpinned.
3. Central fix in `_chat_api_kwargs_from_prepared` and `_chat_api_kwargs`
   (`tldw_chatbook/Chat/console_provider_gateway.py`): for the seven remaining
   cloud keys (openai, cohere, deepseek, google, groq, huggingface, openrouter),
   pin `api_base_url = _adapter_api_base_url(resolution)` only when it differs
   from `builtin_provider_endpoint(execution_key)` — the auxiliary path already
   pins unconditionally, and the dispatch layer
   (`project_chat_handler_kwargs`) already force-forwards the kwarg, so the
   gateway drop is the only gap.
4. Verify each adapter's base-URL parameter against its signature (all seven
   accept `api_base_url`; documented in notes).
5. Targeted tests: `Tests/Chat/test_console_provider_gateway.py` plus
   `Tests/Chat/test_console_prepared_request.py` A/B against base if red.

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 A configured `[api_settings.<provider>].api_base_url` reaches the primary Console send path for the four unmasked providers (google, huggingface, moonshot, mistral/mistralai)
- [x] #2 The remaining affected providers (openai, cohere, deepseek, groq, openrouter, zai) either carry the fix too, or their masking gate is documented as the deliberate reason they are excluded
- [x] #3 With no `api_base_url` configured, every provider's request URL is unchanged from today (no regression for the default case)
- [x] #4 Each adapter's actual base-URL parameter name is verified against its signature/PROVIDER_PARAM_MAP rather than assumed uniform
- [x] #5 Tests cover at least one unmasked provider end-to-end (configured base URL reaches the posted request) plus the unconfigured default
- [x] #6 llama_cpp / local_llamacpp behavior is confirmed unchanged by the fix
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Premise verdict at base `cddc89d3e7` (2026-10-06): partially stale, still
live — narrowed from ten providers to seven, with two live unmasked defects.**

Verified per provider by reading each adapter's config source and the gateway
builders:

- anthropic: already pinned (TASK-2114, pre-filing).
- mistral/mistralai, moonshot, zai: gateway pinning landed post-filing
  (`_chat_api_kwargs_from_prepared` branches) — fixed by drift.
- cohere, deepseek, groq, openrouter: their handlers were refactored onto the
  runtime config snapshot / split modules and now self-serve
  `[api_settings.<provider>].api_base_url` directly — the CONFIG case works;
  only session-selected endpoints were still dropped.
- google: self-serves `[api_settings.google]` via
  `get_runtime_config_snapshot()` (`LLM_API_Calls.py:3391`); the config case
  works; session-selected endpoints were still dropped. (The filing's
  "reads legacy `[google]` only" no longer holds.)
- **huggingface: LIVE unmasked defect** — `chat_with_huggingface` reads only
  legacy sections (`huggingface_api` / `[API].huggingface`,
  `LLM_API_Calls.py:4147`), so a configured
  `[api_settings.huggingface].api_base_url` was silently ignored.
- Cross-cutting live shape for all seven: a session-selected/pinned endpoint
  was dropped from the adapter kwargs by `_chat_api_kwargs` /
  `_chat_api_kwargs_from_prepared` (the TASK-2114 failure shape), and the
  trace verifier's `reconstruct_provider_gateway_kwargs` mirrored the drop.

**Fix (AC#1/AC#2):** central conditional pin in both primary kwargs builders
plus the verifier's reconstruction:

- `tldw_chatbook/Chat/console_provider_endpoints.py` — new shared
  `CONDITIONAL_BASE_URL_EXECUTION_KEYS` (openai, cohere, deepseek, google,
  groq, huggingface, openrouter) and `meaningful_adapter_base_url()`:
  the pin is forwarded only when the resolved endpoint differs from the
  provider's shipped builtin default.
- `tldw_chatbook/Chat/console_provider_gateway.py` —
  `_meaningful_adapter_base_url()` wrapper; new `elif` branch in
  `_chat_api_kwargs_from_prepared` (after the evaluator structured-output
  branch, preserving its unconditional pin) and in `_chat_api_kwargs`.
- `tldw_chatbook/Chat/console_trace_final_values.py` — the independent
  reconstruction mirrors the same rule (imports the shared helper so the two
  cannot drift; Capture-On alignment stays green).

**Why conditional (AC#3):** `resolution.base_url` always carries something —
the builtin cloud default when nothing is configured. Unconditional pinning
would shadow each adapter's own fallbacks and legacy-section endpoints
(`[google]`, `huggingface_api`) that today win when the canonical table is
empty. With no `api_base_url` configured and no session selection, the kwarg
stays absent and every adapter's request URL is byte-identical to before
(pinned by `test_console_send_leaves_base_url_unpinned_in_default_case`).
Note: the auxiliary path (`_auxiliary_chat_api_kwargs`) already pinned
unconditionally for every provider; the primary path was the outlier.

**AC#4 verification:** all seven adapters accept `api_base_url` in their
signatures (openai/cohere/google/huggingface in `LLM_API_Calls.py`; deepseek/
groq/mistral/openrouter in their split modules), and
`project_chat_handler_kwargs` (`Chat_Functions.py:1617-1618`) force-forwards
the kwarg to the handler whenever it is non-None, so no PROVIDER_PARAM_MAP
entries are required.

**AC#6:** llama keys are outside the new set; the local branch is untouched
(`test_console_send_base_url_pin_leaves_llamacpp_paths_untouched` pins the
prepared-path pin and the plain path's historical no-pin).

**Tests (all in `Tests/Chat/test_console_provider_gateway.py`):**
`test_console_send_pins_configured_api_settings_base_url` (google+huggingface
end-to-end through `resolve_for_send` → `stream_chat` with a capturing
adapter), `test_console_send_pins_configured_base_url_for_every_affected_provider`
(all seven), `test_console_send_pins_session_selected_endpoint_for_affected_provider`
(session-pinned endpoint honored instead of dropped),
`test_console_send_leaves_base_url_unpinned_in_default_case` (all seven),
`test_chat_api_kwargs_pins_only_meaningful_provider_base_urls` (rewritten from
`..._omits_api_base_url_for_unpinned_provider`, which pinned the pre-fix
behavior), and the llama_cpp pin above.

**Commands and results** (worktree `.venv`, Python 3.12.13):

- `pytest Tests/Chat/test_console_provider_gateway.py` — base: 449 passed;
  after fix: 467 passed (18 new; the one stale expectation test rewritten).
- Red-first: the 11 pin assertions failed at base (KeyError 'api_base_url'),
  the 7 default-case assertions passed before and after.
- Regression A/B (checkout-swap, never stash): identical failure sets on base
  vs branch for `test_console_trace_final_values.py +
  test_console_provider_endpoints.py + test_console_trace_call_lifecycle.py +
  test_console_agent_bridge.py` (155 failed both sides — pre-existing
  `Backup_Recovery.bootstrap.RecoveryRequired` developer-machine reds), for
  `test_openai_streaming_usage.py + test_anthropic_streaming_usage.py` (9
  failed both sides, same env reds), and for
  `test_console_session_settings.py + test_console_chat_controller.py +
  test_databricks_console_surfaces.py + test_qwencloud_provider_contract.py +
  test_console_trace_service.py` (215 failed both sides, same env reds). Zero
  new failures introduced.

**TASK-2111 overlap check:** TASK-2111 (config exit-write clobbering) is
config-serialization territory and still `To Do`; this task touches only
read-side request building — no overlap. (Note: the wave brief said fleet PR
#2873 covers TASK-2111; PR #2873 is actually a TTS PR for TASK-32931 —
reported to the owner, no conflict either way.)

ADR required: no — routine cross-provider bug fix inside the existing gateway
kwargs-builder pattern (ADR-146/ADR-179 pinning precedent); no new schema,
sync, contract, or boundary decision.

Modified files: `tldw_chatbook/Chat/console_provider_endpoints.py`,
`tldw_chatbook/Chat/console_provider_gateway.py`,
`tldw_chatbook/Chat/console_trace_final_values.py`,
`Tests/Chat/test_console_provider_gateway.py`.
<!-- SECTION:NOTES:END -->
