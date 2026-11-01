# LLM provider layer

This document describes how chat requests reach providers: the `chat_api_call` dispatch seam (there is **no** `chat_with_provider()` — see gotchas), the per-provider handlers and their payload surgery, streaming shapes, the error taxonomy, usage accounting, model capabilities, the model-catalog auto-refresh, and local backends.

## Authoritative files

| File | Role |
| --- | --- |
| `Chat/Chat_Functions.py` | THE dispatcher: `API_CALL_HANDLERS` (endpoint → handler map), `chat_api_call()`, `PROVIDER_PARAM_MAP`, `chat_reply_text()` |
| `Chat/Chat_Deps.py` | The `ChatAPIError` hierarchy (`ChatAuthenticationError`, `ChatRateLimitError`, `ChatBadRequestError`, `ChatProviderError`, …) |
| `provider_registry.py` (package root) | Single source of provider identity and engine-preset data (`ProviderRecord`, `ALL_RECORDS`, `RECORDS_BY_KEY`, `ENGINE_RECORDS`; ADR-179). Stdlib-only leaf module so `config.py` can import it |
| `LLM_Calls/hosted_provider_engine.py` | Generic strict Chat-Completions engine: `build_hosted_chat_handler(record)` builds one handler per preset record (resolve, payload, transport, response layers) |
| `LLM_Calls/LLM_API_Calls.py` | Hosted provider handlers, one `chat_with_<provider>()` each; `get_openai_embeddings()` |
| `LLM_Calls/LLM_API_Calls_Local.py` | Local/self-hosted handlers over a shared OpenAI-compatible engine |
| `LLM_Calls/hosted_chat.py` + `hosted_chat_streaming.py` | Provider-neutral hosted transport and SSE handling shared by the strict adapters |
| `LLM_Calls/moonshot.py`, `zai.py`, `qwencloud*.py` | Strict Chat-Completions adapter-policy modules |
| `LLM_Calls/pricing_catalog.py` | Per-model $/Mtok pricing for the Console cost ticker |
| `Chat/usage_recorder.py` | Context-scoped token-usage recording seam |
| `Chat/local_reasoning_metadata.py` | Bounded `/props` template and tool-capability reads (`read_local_reasoning_metadata`) |
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

### Provider registry and engine presets (ADR-179)

`provider_registry.py` holds one frozen `ProviderRecord` per provider (`ALL_RECORDS`, 56 records at `2b296e60b3`; `RECORDS_BY_KEY` also maps the legacy spellings in `ALIASES`, e.g. `mistralai`, `local_vllm`). Identity fields (key, `[providers]` config key, display name, cloud/local classification, env-var candidates, default base URL, native-tools and auto-refresh flags) cover every provider; parity tests in `Tests/test_provider_registry.py` compare them with `config.py`, the readiness sets and the other hand-kept lists. Behavior never lives in a record.

Records with `engine_driven` set (`ENGINE_RECORDS`) are *presets*: their preset fields (`settings_defaults`, `payload_flags`, `auth_scheme`, `base_url_suffix`, `reasoning_effort*`, `thinking_toggle_models`, `extra_body_fields`, `response_allowances`) drive the shared engine, so a preset provider has no `chat_with_<provider>()` function. `API_CALL_HANDLERS` registers each as `_LazyHostedChatHandler(<RECORD>)` (`Chat/Chat_Functions.py`), which imports the engine on first call so it stays off the UI-ready import budget, and `PROVIDER_PARAM_MAP` points the key at the shared `ENGINE_PROVIDER_PARAM_MAP`. The `custom-hosted` key is the same engine over user-registered custom endpoints (`[console] custom_endpoints_use_engine`, default on); those use the `bearer_optional` auth scheme so keyless endpoints execute. `moonshot`, `zai` and `qwencloud` are registry records but keep their own adapter modules (ADR-063).

Adding an OpenAI-compatible hosted provider is a record, a dispatch line and a param-map line, plus the tests; some auxiliary sites (endpoint maps, readiness, labels) still repeat record data by hand.

### Credential precedence (differs by resolver)

| Resolver | Order | Where |
| --- | --- | --- |
| Engine presets | explicit, then **env**, then stored `api_key` | `LLM_Calls/hosted_provider_engine.py::_resolve_api_key` |
| `moonshot.py`, `zai.py`, `qwencloud.py` | explicit, then **stored**, then env | `moonshot.py::_resolve_api_key`, `zai.py::_resolve_api_key`, `qwencloud.py::resolve_qwencloud_api_key` |
| Custom-endpoint registry | env, then stored | `Chat/custom_endpoint_registry.py::resolve_entry_credential` (ADR-146) |

Registry entry credentials end at none when neither source supplies a key. The
gateway preserves that keyless decision through health checks and chat dispatch;
a credential from the ordinary family slot never follows the entry's URL. Family
readiness contributes capabilities, while entry-owned ciphertext still blocks.

ADR-012's 2026-09-19 amendment rules that the stored `api_settings` key outranks the environment variable; the engine currently orders them the other way and the owner decided (2026-10-25) to keep ADR-012 for the hosted engine, so this row is expected to change under TASK-36941.3. Custom endpoints keep ADR-146's order by the owner's 2026-10-26 decision. A user with both set can therefore get a different key per provider family.

### Implemented providers

Hosted (bespoke handlers): openai, anthropic, cohere, groq, openrouter, deepseek, mistral(+`mistralai`), google, huggingface, moonshot, zai, qwencloud. Hosted (engine presets): every `ENGINE_RECORDS` entry (33, e.g. databricks, together, fireworks, cerebras, and `custom-hosted`). Local/self-hosted: llama_cpp, strata, polystrata, koboldcpp, oobabooga, tabbyapi, vllm, local-llm (llamafile), ollama, aphrodite, custom-openai-api(+`-2`), mlx_lm — plus `local_*` aliases onto the same handlers.

## Handler internals (OpenAI as the reference)

- Config overlay: legacy `[openai_api]` + canonical `[api_settings.openai]`, with only `api_key`/`api_base_url` overlaid from canonical; a missing key raises `ChatConfigurationError`.
- Param resolution: arg > config > default.
- API selection: reasoning params switch the Responses API (`/responses`, payload key `input`). GPT-5.6 with absent/`none` effort uses Chat Completions unless a nonblank summary/verbosity control is present. Exact unprefixed `gpt-6-luna` (case/outer whitespace normalized) also uses Chat Completions for explicit `none` effort and a nonempty list consisting entirely of canonical nested function tools with object parameter schemas; it forwards `reasoning_effort=none`, unchanged tools/choice/history and `max_completion_tokens`. Summary/verbosity presence (including summary `none`) still selects Responses. Empty/no tools, plain/fenced Luna calls, other IDs and absent/non-`none` effort retain their existing routing. This follows the [Luna model contract](https://developers.openai.com/api/docs/models/gpt-6-luna), which restricts Chat function calling to `none`; sampling remains a separate capability fact.
- OpenAI Responses limitation: its current normalization handles text only. Native function output items/argument events and canonical assistant/tool continuation are not translated end-to-end, and nested Chat function schemas are not converted to flat Responses tools. Provider-wide native-tool advertising does not establish support on those routes. The narrow Luna Chat exception does not implement Responses native tools.
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

The exact Haiku 5.5 family rejects `temperature`, `top_p`, `top_k` and fixed
`thinking.budget_tokens`, per its [migration guide](https://platform.claude.com/docs/en/models/haiku-5-5/migration-guide).
The shared Anthropic predicates omit sampling and reuse the existing adaptive
thinking/effort mapping; Haiku 4.5 keeps its sampling and fixed-budget behavior.
This is a documented capability correction, separate from the August 2026 live
probes for older families.

## Model catalog auto-refresh (ADR-020)

Providers refreshed: OpenAI, Anthropic, MistralAI, Moonshot, OpenRouter, QwenCloud, ZAI (`AUTO_REFRESH_PROVIDER_LIST_KEYS` — the code is authoritative; QwenCloud postdates the ADR prose).

- **Disk cache**: `model_catalog_cache.json` in the user data dir — IDs + timestamps only; bounded (2 MiB, 128 entries, 100 models/entry).
- **Consent-gated**: first startup shows a consent modal; allow persists consent, deny also disables auto-refresh. Refresh runs as an exclusive worker (`model-catalog-refresh` group).
- **Flow**: per provider — skip if disabled → resolve key (OpenRouter's catalog is public) → skip if disk entry fresh → discover models (`GET <base>/models`, metadata scrubbed) → compute new ids → append-only write-through to config **only** for opted-in providers and never on an oversized first fetch (that sets a baseline and appends nothing) → record to the disk cache atomically → prune cache to configured providers.
- **Selector merge**: capped at 50 merged entries; saved ids first, then discovered-only; for auto-refreshed cloud providers an endpoint-scoped snapshot is authoritative for new selector choices (retired ids drop out), while the active session model is preserved. One consolidated notification is forwarded down the screen stack (Textual messages only bubble up).

Config (`[model_catalog]`): `auto_refresh_enabled` (default true), `refresh_consent_recorded`, `stale_after_hours` (24), `auto_refresh_disabled` (provider list), `write_to_config` (providers), `use_models_dev` (default false).

## Local backends and managed servers

Local handlers read the immutable `[api_settings]` runtime snapshot. Managed server starts live in `Event_Handlers/LLM_Management_Events/`: per-backend command builders (`--model/-m --host --port` for llamacpp/llamafile), a claim-based lifecycle (`reserve_server_launch`, `run_server_subprocess`, bounded terminate) keyed to app-held `Popen` handles, and the Lab/Models UI for start/stop. Transformers/ONNX participate via local summarization/embedding paths rather than the chat handlers.

The shared local OpenAI-compatible sender fixes SSE decoding to UTF-8 before
iterating response lines, including when the response omits or misstates its
charset. Requests otherwise defaults charset-less `text/event-stream` to
Latin-1 and corrupts literal Unicode before JSON parsing. Console llama.cpp
requests with native tools use this shared sender; tool-free llama.cpp requests
use the separate HTTPX gateway path. `Tests/LLM_Calls/test_local_sse_encoding.py`
exercises real Requests response construction and byte decoding with fragmented
Unicode, escaped JSON, ASCII, normal exhaustion and consumer Stop. This corrects
new streamed replies; it does not reconstruct previously corrupted messages.

### External Strata and PolyStrata servers

`strata` and `polystrata` are separate canonical local providers, with independent
`api_settings` and saved model lists. They reuse `chat_with_llama(provider_name=...)`
and the shared compatible HTTP adapter, while staying out of the llama.cpp-only
direct Console path and server-management assumptions. Their optional credential
source is resolved through the existing provider credential contract; a resolved
keyless Console selection is never replaced by a stored key in the adapter.

Discovery uses `/v1/models`, including `meta.n_ctx` for the selected model's context
window. An optional bounded background `/props?model=...&autoload=false` probe
reads tool capability without requesting a model load. Missing properties leave
capabilities unknown. Native tools require both `supports_tools` and
`supports_tool_calls`; a template string alone grants nothing. Existing explicit
per-model settings and exact template replay policies still apply.

Their reasoning controls use top-level `reasoning_effort` (`none`, `low`, `medium`,
`high`) and `reasoning_budget_tokens`. The shared local stream adapter forwards
SSE `data:` lines and turns control lines into empty iteration checkpoints, which
Console ignores. Comments and `event`/`id`/`retry` fields never enter prose, and
keep-alive traffic returns control so Stop can settle the stream on its owner. This also fixes the llama.cpp path used when native tools select that
adapter. Legitimate assistant text containing "heartbeat" is unchanged.

The upstream contracts and client-only qualification scope are recorded in
[Settings](../User_Guide/settings.md#strata-and-polystrata) and
`Tests/Chat/test_strata_providers.py`.

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

## Native continuation identity

Provider engines produce canonical execution state; `console_agent_bridge` binds it to the exact raw selected provider before runtime custody. Continuation JSON V3 retains both identities and composes a nullable managed envelope in the existing private message column. The managed MAC covers the raw pin; legacy V1/V2 serialization and V2 signature input are unchanged. Host history/recovery validates execution provider, protocol, model, full normalized base and exact selection. Engine replay validates only canonical execution identity, after the host check. No display-origin inference or registry lookup can recreate missing durable selection authority.

The custom-hosted entry in `Chat_Functions.CUSTOM_PROVIDER_PARAM_MAP` forwards the existing `provider_continuations` argument, so a second native request includes the saved assistant call and completed result. This is the sole additional forwarding key relative to legacy custom endpoint maps. Moonshot, ZAI and the hosted engine retain their existing wire protocols. Private reasoning stays in the continuation/trace sidecar and is not projected into ordinary assistant text by the recovery translator. Session-only destinations remain ephemeral; their full bases are excluded from durable dispatch identity (ADRs 063, 147, 179).

Durable continuation is optional for a send. `resolve_console_destination` pins the continuation tuple (execution provider, protocol, full base) only when the dispatch checkpoint can record it; an unconfigured engine preset pins its record default, the URL the engine itself sends to. A blank, credentialed, query/fragment-bearing or oversized base leaves the tuple empty: the send proceeds and is not resumable. No default setup loses resume this way: only moonshot, zai, deepseek and engine presets produce continuations (legacy `chat_with_<provider>` handlers have none), every one of them that is ready under the shipped config resolves a durable tuple (pinned by `test_shipped_default_continuation_destinations_stay_durable`), the four workspace-URL presets block until a base is configured, and a credentialed or query/fragment base could never hold a continuation even before, because the checkpoint parser rejects it. The one deliberate loss is a tuple too large for the 2,048-byte dispatch checkpoint (each identity may otherwise reach 4 KiB): it stays non-durable for safety. `console_continuation_binding.continuation_can_be_durable` runs the native tool loop with ephemeral continuation durability for temporary sessions, session-only endpoints and dispatches whose checkpoint has no tuple (including pre-upgrade checkpoints, which dispatch recovery accepts on identity alone). A dispatch checkpoint constrains only the assistant message it owns: a regenerate, retry, continue or edit run that starts while another message owns dispatch recovery keeps its durable continuation (bound by its own host target), as on dev.

## Custom hosted streamed usage

`CUSTOM_HOSTED` requests `stream_options.include_usage` on streaming requests
through the existing preset flag (ADR-179); non-streaming payloads are unchanged.
The hosted parser still accepts a custom stream that omits usage. Console's
call-scoped usage tap reads top-level usage first, including its explicit
null/malformed presence, and otherwise accepts usage from one terminal choice
without rewriting the event. Automatic work still requires complete nonnegative
integer usage: missing or malformed counts retain the reservation and pause the
chain with `usage_unknown` (ADR-134). There is no estimate/refund fallback.

A custom endpoint that rejects the option is not a send failure:
`hosted_chat.owned_json_post` retries once without `stream_options` when a 400
(or FastAPI-style 422) body names it, as `chat_with_openai`/`chat_with_huggingface`
do. The engine enables this only for records whose stream may end without usage
(`tolerant_response_extras` or `stream_usage_optional`); a strict preset keeps the
clean 400, because a usage-free retry would bill a whole generation and then fail
for missing usage. Any other rejection (or a second one) surfaces as the ordinary
typed provider error, and the degraded stream simply carries no usage (so
automatic work pauses with `usage_unknown` rather than estimating).
The local synthetic two-request differential for TASK-35582 established that the
qualification server omitted usage without the option and supplied complete
usage with it. This diagnoses the current server behavior; the earlier retained
HTTP-status-only packets cannot establish their exact raw response fields.
