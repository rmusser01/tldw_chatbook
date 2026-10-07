# LLM provider layer

This document describes how chat requests reach providers: the `chat_api_call` dispatch seam (there is **no** `chat_with_provider()` — see gotchas), the per-provider handlers and their payload surgery, streaming shapes, the error taxonomy, usage accounting, model capabilities, the model-catalog auto-refresh, and local backends.

## Authoritative files

| File | Role |
| --- | --- |
| `Chat/Chat_Functions.py` | THE dispatcher: `API_CALL_HANDLERS` (endpoint → handler map), `chat_api_call()`, `PROVIDER_PARAM_MAP`, `chat_reply_text()` |
| `Chat/Chat_Deps.py` | The `ChatAPIError` hierarchy (`ChatAuthenticationError`, `ChatRateLimitError`, `ChatBadRequestError`, `ChatProviderError`, …) |
| `LLM_Calls/LLM_API_Calls.py` | Hosted provider handlers, one `chat_with_<provider>()` each; `get_openai_embeddings()` |
| `LLM_Calls/LLM_API_Calls_Local.py` | Local/self-hosted handlers over a shared OpenAI-compatible engine |
| `LLM_Calls/hosted_chat.py` + `hosted_chat_streaming.py` | Provider-neutral hosted transport and SSE handling shared by the strict adapters |
| `LLM_Calls/moonshot.py`, `zai.py`, `qwencloud*.py` | Strict Chat-Completions adapter-policy modules |
| `LLM_Calls/pricing_catalog.py` | Per-model $/Mtok pricing for the Console cost ticker |
| `Chat/usage_recorder.py` | Context-scoped token-usage recording seam |
| `tldw_chatbook/model_capabilities.py` | Package-root capability detection: vision, context windows, thinking/sampling predicates |
| `LLM_Provider_Catalog/` | Model list discovery + auto-refresh (ADR-020) |
| `Event_Handlers/LLM_Management_Events/` | Managed local-server lifecycle (per-backend modules + `server_lifecycle.py`) |

## Dispatch seam

There is no unified `chat_with_provider()`. Every caller goes through `chat_api_call(api_endpoint, messages_payload, …)`:

1. The endpoint is lowercased; a metrics attempt is logged with a redacted, cardinality-bounded label.
2. `API_CALL_HANDLERS` resolves the handler; unknown endpoints raise `ValueError` listing valid ones.
3. Ephemeral project-instruction origin tags are stripped per endpoint (preserved only for the endpoints that group them).
4. Generic params are derived from `chat_api_call`'s **own signature** (so the map cannot drift from the signature) and projected per provider via `PROVIDER_PARAM_MAP` (e.g. QwenCloud's `messages_payload` → `input_data`).
5. The handler executes and returns one of: a string, a normalized response dict, or an SSE generator.
6. **Dict responses pass through unchanged** — normalizing to strings here broke the Console gateway's tool-call/finish-reason/usage parsing.
7. Usage accounting runs against dict/string results through the active usage recorder (OpenAI-style token names win over provider names; absent usage falls back to ~4-chars/token estimates, with base64 image parts excluded). Accounting is wrapped — it must never break a call.
8. Errors map into the `ChatAPIError` taxonomy (below).

### Implemented providers

Hosted: openai, anthropic, cohere, groq, openrouter, deepseek, mistral(+`mistralai`), google, huggingface, moonshot, zai, qwencloud. Local/self-hosted: llama_cpp, koboldcpp, oobabooga, tabbyapi, vllm, local-llm (llamafile), ollama, aphrodite, custom-openai-api(+`-2`), mlx_lm — plus `local_*` aliases onto the same handlers.

## Handler internals (OpenAI as the reference)

- Config overlay: legacy `[openai_api]` + canonical `[api_settings.openai]`, with only `api_key`/`api_base_url` overlaid from canonical; a missing key raises `ChatConfigurationError`.
- Param resolution: arg > config > default.
- API selection: reasoning params switch the Responses API (`/responses`, payload key `input`); Responses stream events are translated back into chat-completions SSE.
- Capability gating via `model_capabilities` predicates: reasoning families drop `temperature`/`top_p`; models that require it swap `max_tokens` → `max_completion_tokens`.
- Prompt caching: `prompt_cache_key` on canonical OpenAI; Anthropic-side `cache_control` with degrade-retry including a 1-hour TTL tier.
- Streaming: handlers yield provider-shaped SSE `data:` lines; errors mid-stream are yielded as `data: {"error": …}` events; the `[DONE]` sentinel stays out of `finally` so a consumer's `GeneratorExit` (Console Stop) cannot skip cleanup. `stream_options.include_usage` is added with a 400-retry that drops it for providers that reject it.
- Non-streaming: urllib3 `Retry` (429/5xx, POST allowed) on an HTTPAdapter; retry count/delay/timeout from config.

## Error taxonomy

| HTTP outcome | Exception | Notes |
| --- | --- | --- |
| 401 | `ChatAuthenticationError` | |
| 429 | `ChatRateLimitError` | Carries the provider `Retry-After` when present |
| other 4xx | `ChatBadRequestError` | Real status preserved (402/403 credit-terminal distinction) |
| 5xx | `ChatProviderError` | |
| transport failure | `ChatProviderError(504)` | `RequestException` mapping |
| already-typed Chat errors | re-raised | Redacted when the request is sensitive |
| ValueError/TypeError/KeyError | `ChatBadRequestError` | "Configuration/Parameter Error" |

Retry layers: transport retries inside handlers (idempotent POST, 429/5xx) → semantic retry honoring `Retry-After` in `Agents/model_retry` → cross-provider fallback chain (ADR-110, credit-terminal detection) at the agent layer.

## model_capabilities

`ModelCapabilities` resolves per-model facts from `[model_capabilities]`: direct `models` mappings > per-provider `patterns` regex > models.dev gap-fill (opt-in, network-free) > defaults. Consumers: vision gating for attachments (`Chat/attachment_core`), context-window-based truncation (`Utils/token_counter`), per-provider tools payload builders, and thinking configuration (Anthropic enabled/disabled/adaptive budgets vs effort-style shapes; thinking budgets auto-grow `max_tokens` when needed; per-family predicates for moonshot/zai/deepseek too).

## Model catalog auto-refresh (ADR-020)

Providers refreshed: OpenAI, Anthropic, MistralAI, Moonshot, OpenRouter, QwenCloud, ZAI (`AUTO_REFRESH_PROVIDER_LIST_KEYS` — the code is authoritative; QwenCloud postdates the ADR prose).

- **Disk cache**: `model_catalog_cache.json` in the user data dir — IDs + timestamps only; bounded (2 MiB, 128 entries, 100 models/entry).
- **Consent-gated**: first startup shows a consent modal; allow persists consent, deny also disables auto-refresh. Refresh runs as an exclusive worker (`model-catalog-refresh` group).
- **Flow**: per provider — skip if disabled → resolve key (OpenRouter's catalog is public) → skip if disk entry fresh → discover models (`GET <base>/models`, metadata scrubbed) → compute new ids → append-only write-through to config **only** for opted-in providers and never on an oversized first fetch (that sets a baseline and appends nothing) → record to the disk cache atomically → prune cache to configured providers.
- **Selector merge**: capped at 50 merged entries; saved ids first, then discovered-only; for auto-refreshed cloud providers an endpoint-scoped snapshot is authoritative for new selector choices (retired ids drop out), while the active session model is preserved. One consolidated notification is forwarded down the screen stack (Textual messages only bubble up).

Config (`[model_catalog]`): `auto_refresh_enabled` (default true), `refresh_consent_recorded`, `stale_after_hours` (24), `auto_refresh_disabled` (provider list), `write_to_config` (providers), `use_models_dev` (default false).

## Local backends and managed servers

Local handlers read the immutable `[api_settings]` runtime snapshot. Managed server starts live in `Event_Handlers/LLM_Management_Events/`: per-backend command builders (`--model/-m --host --port` for llamacpp/llamafile), a claim-based lifecycle (`reserve_server_launch`, `run_server_subprocess`, bounded terminate) keyed to app-held `Popen` handles, and the Lab/Models UI for start/stop. Transformers/ONNX participate via local summarization/embedding paths rather than the chat handlers.

## Boundaries

- `LLM_Calls/*` — transport + payload assembly per provider; knows config sections, never UI.
- `Chat/Chat_Functions.py` — dispatch, param projection, error mapping, usage recording. `Chat/Chat_Deps.py` — the shared exception contract.
- `LLM_Provider_Catalog/` — model **list** discovery only; never builds chat requests.
- `model_capabilities.py` — per-model request-shape facts consumed by handlers and UI.
- Console layers (`console_provider_gateway`) consume dicts/SSE from handlers and must not reimplement dispatch.

## Verified gotchas

1. `MCP/tools.py::chat_with_provider` is a deliberate `NotImplementedError` stub referencing a removed upstream API — the real seam is `chat_api_call`.
2. Handlers return heterogeneous shapes (str | dict | SSE generator); `chat_api_call` must not normalize them.
3. Only OpenAI Responses stream events are translated to chat SSE; other providers' streams pass through in their own shape.
4. The `provider_name` parameter exists only for local handlers (`PROVIDERS_WITH_PROVIDER_NAME`).
5. Endpoint labels in logs/metrics are redacted and cardinality-bounded — never log raw endpoint strings with credentials.

## Related docs

- [chat-pipeline.md](./chat-pipeline.md), [console.md](./console.md) — the streaming consumers
- [rag.md](./rag.md) — embeddings share the provider config surface
- `backlog/decisions/020-automatic-model-catalog-refresh.md` (amends ADR-002)
