---
id: TASK-2122
title: >-
  Nine of eleven providers report no usage on the streaming path
status: Done
assignee: [rmusser01]
created_date: '2026-08-03 19:20'
labels:
  - cost-ticker
  - llm-calls
  - correctness
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The cost ticker's core promise is "real usage from the API, never an estimate." On the
streaming path that holds for **OpenAI and Anthropic only**. An audit of every
`stream_generator` in `LLM_Calls/LLM_API_Calls.py`, bounded by indentation so the
non-streaming path in the same function is not miscounted, found that **nine of eleven
providers emit no usage chunk at all when streaming**.

| Provider | Streaming generator | Usage emitted |
|---|---|---|
| openai | 758-829 | yes (`stream_options`) |
| anthropic | 1548-1741 | yes (PR1 work) |
| cohere | 2363-2555 | **no** |
| deepseek | 2950-2966 | **no** |
| google | 3417-3541 | **no** |
| groq | 3969-3992 | **no** |
| huggingface | 4373-4416 | **no** |
| mistral | 4740-4750 | **no** |
| openrouter | 4993-5003 | **no** |
| moonshot | 5363-5399 | **no** |
| zai | 5691-5711 | **no** |

Every one of these providers **does** handle usage on its non-streaming path, so the
gap is invisible in any non-streaming test and looks like working code on inspection.

There are two distinct root causes, needing two different fixes:

**1. OpenAI-compatible passthroughs** (deepseek, groq, huggingface, mistral,
openrouter, moonshot, zai). These generators are pure SSE relays — `for line in
response.iter_lines(): yield line + "\n\n"` — so they would forward a usage chunk
faithfully if one ever arrived. It never does: `stream_options: {"include_usage": True}`
is set at **exactly one site**, line 649, inside `chat_with_openai`. These providers are
simply never asked for usage. The fix is to request it, reusing the 400-degrade retry
already proven on the OpenAI path (line 768) since not every compatible endpoint accepts
the field.

**2. Native-protocol translators** (cohere, google). These parse provider-shaped SSE and
synthesize OpenAI-shaped chunks, so no payload flag can help — the translator has to read
the usage and emit a chunk. Both drop data the provider is already sending:
- Cohere's `message-end` branch (line 2485) reads `delta.finish_reason` and ignores
  `usage`, even though the generator's own comment above it documents `message-end` as
  carrying `(delta.finish_reason, usage)`. Cohere's non-streaming path already knows the
  shape: `usage.billed_units.{input_tokens,output_tokens}` (line 2631).
- Google's generator never reads `usageMetadata`, which Gemini sends on the final
  streaming chunk. Its non-streaming path maps it at line 3637.

**How exposed is this today?** Measured, not assumed — a probe over the shipped config
template resolving `build_default_console_session_settings` per provider:

- `[chat_defaults]` ships **no** `streaming` key, so the `True` fallback in
  `console_session_settings.py:436` does not apply; resolution falls through
  `default_sources = (model_profile, saved_defaults, chat_defaults, provider_settings)`
  to the per-provider template value, which is `streaming = false`.
- So for anthropic/openai/google/cohere/groq/deepseek/openrouter/moonshot the Console is
  **non-streaming out of the box** and usage capture works today.
- **Mistral is the exception and is broken by default**: its `[api_settings.mistral]`
  block is the only one with no `streaming` key, so it falls through to the `True`
  default. Mistral streams out of the box and therefore records no usage at all. That
  template inconsistency is worth fixing on its own.
- For the other eight, enabling streaming is a **one-keystroke, first-class action** —
  the Alt+M quick popover and the Settings screen's `chat_defaults.streaming` field, and
  a `chat_defaults` value outranks the per-provider template. The moment a user turns
  streaming on, usage capture silently stops: cost falls back to estimation and the cache
  chip loses the ground truth it derives warm/cold from — the two things the ticker
  exists to provide. Nothing warns them.

Missed because PR1's verification targeted Anthropic and PR3's live verification ran
against Anthropic only.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

1. Verify the premise at base `cddc89d3e7` (done — see notes below; narrowed to
   huggingface as the only remaining live provider; cohere/google fixed by
   TASK-32805.2/PR #2738, the OpenAI-compatible split modules by their
   migrations onto the hosted engine).
2. Red-first: new `Tests/Chat/test_huggingface_streaming_usage.py` driving a
   recorded HF-router SSE stream (content deltas + trailing usage frame)
   through `chat_with_huggingface`, asserting the payload requests
   `stream_options.include_usage`, a usage chunk reaches the consumer as an
   OpenAI-shaped SSE line the gateway parses, and the endpoint that 400s on
   `stream_options` still streams (degrade retry).
3. Fix `chat_with_huggingface` streaming (`tldw_chatbook/LLM_Calls/
   LLM_API_Calls.py`): request `include_usage`, replicate the OpenAI
   400-degrade retry, and relay OpenAI-shaped SSE data lines (instead of bare
   text strings) so a usage frame can reach the gateway.
4. Guard test (AC#5): enumerate the streaming send paths and fail if a
   provider relays no usage.
5. Targeted tests: the new file plus the streaming-usage suites the repo
   already carries (openai/anthropic/cohere/google/kimi-zai/moonshot/
   groq-openrouter characterization); A/B any red against base.

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 `stream_options: {"include_usage": True}` is requested for the OpenAI-compatible providers, with the existing 400-degrade retry so endpoints that reject the field still stream
- [x] #2 Cohere's streaming `message-end` handler reads `usage.billed_units` and emits a usage chunk matching the non-streaming path's bucket mapping
- [x] #3 Google's streaming generator reads `usageMetadata` from the final chunk and emits a usage chunk matching its non-streaming mapping
- [x] #4 A test per provider drives a recorded SSE stream through the generator and asserts a usage chunk reaches the gateway with correct disjoint buckets
- [x] #5 A single guard test enumerates the streaming generators and fails if one emits no usage, so a newly added provider cannot silently regress
- [ ] #6 Verified live against at least one OpenAI-compatible provider and one native translator that a streamed Console turn persists real `usage_json` — **not verifiable in this worktree**: no live provider credentials are available in this environment and the wave's hard limits forbid work outside it. The mechanism is covered by the recorded-stream guard (all eleven providers) and the gateway-signals test; live verification needs an owner with HF-router + Cohere/Gemini credentials (stream one Console turn each, check `usage_json` in the persisted exchange).
- [x] #7 `[api_settings.mistral]` gets an explicit `streaming` key so it stops differing from every other provider block by omission — **satisfied by drift**: the shipped template now keys the block `[api_settings.mistralai]` WITH `streaming = false` (`config.py` CONFIG_TOML_CONTENT), the `[api_settings.mistral]` table no longer exists in the template at all, and the handler resolves its endpoint under the `mistralai` key (`LLM_Calls/mistral.py:15,252`). Measured: `build_default_console_session_settings(DEFAULT_CONFIG_FROM_TOML, "MistralAI").streaming` → `False` (was the filing's streaming-by-default hole). Adding a new `[api_settings.mistral]` block now would reintroduce the dual-table ambiguity `config.py:1781-1793` explicitly warns about.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Premise verdict at base `cddc89d3e7` (2026-10-06): 10 of the 9-no-usage
providers were already fixed by drift; huggingface was the only live one.**

| Provider | Status at base | Fixed by |
|---|---|---|
| openai | usage (include_usage + 400-degrade) | pre-existing |
| anthropic | usage | PR1 (pre-filing) |
| cohere | usage (`message-end` → `_cohere_usage_to_openai`) | TASK-32805.2 / PR #2738 |
| google | usage (`usageMetadata` → `_gemini_usage_to_openai`) | TASK-32805.2 / PR #2738 |
| deepseek, groq, mistral, openrouter, moonshot | usage (split modules set `include_usage`; the hosted engine's `HostedChatStream` forwards AND requires the trailing usage frame — silence is a protocol error, not an estimate) | ADR-062 engine migration |
| zai | usage (strict adapter; GLM sends usage natively; `HostedChatStream` terminal turn carries it) | strict zai adapter |
| **huggingface** | **no usage — live defect**: payload never asked for `include_usage`, and `stream_generator_huggingface` yielded bare text strings, so even a usage frame the provider sent could not reach the gateway (usage is parsed only out of OpenAI-shaped `data:` lines) | **this task** |

**Fix (AC#1):** `chat_with_huggingface` (`tldw_chatbook/LLM_Calls/
LLM_API_Calls.py`) streaming path:

1. `stream_options: {"include_usage": True}` added to streaming payloads.
2. OpenAI-path 400-degrade replicated: a 400 naming `stream_options` is
   retried once without the field, so strict OpenAI-compatible servers keep
   streaming.
3. The generator now relays OpenAI-shaped SSE `data:` lines instead of bare
   text (the HF router speaks chat.completions, so chunks pass through
   verbatim including the trailing usage-only frame), with exactly one
   guarded `[DONE]` sentinel emitted after the `finally` (the TASK-32805.1
   Stop/close ruling), and JSON error chunks on transport failure.

**AC#2/#3:** verified already implemented by TASK-32805.2 (PR #2738, merged
2026-09-21) and pinned by its tests; no code change needed. The new guard
covers both going forward.

**AC#4/#5:** new `Tests/Chat/test_huggingface_streaming_usage.py`:

- Six HF tests: payload flag (streaming yes / non-streaming no), trailing
  usage frame forwarded verbatim, gateway `_content_from_sse_data` +
  `ConsoleProviderStreamSignals` records the disjoint buckets
  (`prompt_tokens` 11 / `completion_tokens` 5), 400-degrade retry keeps
  streaming without usage, single-sentinel termination with and without a
  provider `[DONE]`.
- `test_every_first_party_streaming_path_emits_usage` (the AC#5 guard):
  enumerates all eleven providers through `chat_api_call(..., streaming=True)`
  with each family's real transport seam mocked (`requests.Session.post` for
  openai/anthropic/cohere/google/huggingface; `hosted_chat.owned_json_post`
  for deepseek/groq/mistral/openrouter/moonshot; `zai.owned_json_post` for
  zai) and asserts every output carries a usage block with both buckets
  (OpenAI or provider-native naming, matching the gateway recorder's
  semantics) — plus that every OpenAI-semantics provider still ASKS for
  usage. A new provider joining the streaming surface silently dropping
  usage fails this test.
- The file's `hermetic_hf_config` fixture pins `load_settings` /
  `get_runtime_config_snapshot` / `create_default_session` / `requests_verify`
  so the tests run on developer machines with pending Backup-Recovery state
  (the pre-existing env-red class; see the A/B evidence below).

**Commands and results** (worktree `.venv`, Python 3.12.13):

- Red-first: `pytest Tests/Chat/test_huggingface_streaming_usage.py` at base —
  5 HF behavior tests + the guard's `huggingface` param failed (no
  `stream_options`; bare-string output; no usage), 11 other guard params
  passed. After the fix: **17 passed**.
- `pytest Tests/Chat/test_huggingface_streaming_usage.py
  Tests/Chat/test_console_provider_gateway.py` → **484 passed**.
- Regression A/B (cp-aside checkout swap, never stash) over
  `test_chat_functions.py + test_sensitive_llm_logging.py +
  test_encrypted_api_key_never_sent.py + test_provider_rate_limits.py +
  test_google_native_tools.py + test_cohere_native_tools.py`: identical
  failure sets base vs fixed (78 failed both sides — the pre-existing
  `Backup_Recovery.bootstrap.RecoveryRequired` developer-machine reds; the
  same class keeps `test_openai_streaming_usage.py` /
  `test_anthropic_streaming_usage.py` red locally, 9 both sides at base and
  branch). Zero new failures.

**AC#6 gap:** see the AC annotation — needs live credentials; not fakeable
here, left unticked rather than silently re-ticked (lessons-testing-evidence).

ADR required: no — bug fix inside the established streaming-usage pattern
(OpenAI degrade precedent, TASK-32805.2 translator precedent, ADR-062 engine
contracts); no new interface, schema, or policy decision.

Modified files: `tldw_chatbook/LLM_Calls/LLM_API_Calls.py` (huggingface
handler only), `Tests/Chat/test_huggingface_streaming_usage.py` (new).
<!-- SECTION:NOTES:END -->
