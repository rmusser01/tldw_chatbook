# Console Provider Execution Parity Design

Date: 2026-05-28
Status: Approved for implementation planning
Target branch: `dev`

## Purpose

Make the native Console able to use every chat provider already supported by `chat_api_call()`, instead of only `llama_cpp` and `local_llamacpp`.

Console is the primary agentic control surface. Provider selection that appears valid in Settings or Console must either send successfully or fail before submission with clear, actionable recovery text. Providers that do not support streaming may return one completed assistant message.

## Current Code Findings

Relevant files:

- `tldw_chatbook/Chat/console_provider_gateway.py`
- `tldw_chatbook/Chat/console_chat_controller.py`
- `tldw_chatbook/Chat/Chat_Functions.py`
- `tldw_chatbook/Chat/provider_readiness.py`
- `tldw_chatbook/Chat/console_session_settings.py`
- `tldw_chatbook/UI/Screens/chat_screen.py`
- `tldw_chatbook/UI/Screens/settings_screen.py`
- `tldw_chatbook/config.py`

Observed state:

- `Chat_Functions.API_CALL_HANDLERS` already dispatches many cloud and local providers.
- `Chat_Functions.PROVIDER_PARAM_MAP` already maps generic chat inputs to provider-specific call arguments.
- `config.py` already contains provider/model lists and `[api_settings.<provider>]` sections for many providers.
- `provider_readiness.py` can validate key-required and keyless providers without exposing secrets.
- `ConsoleProviderGateway.resolve_for_send()` currently resolves only `llama_cpp` and `local_llamacpp`.
- All other Console-native providers return a WIP blocked state.
- The current llama.cpp path has direct async HTTP support for `/v1/models`, `/health`, streaming chat, and non-streaming fallback.

## Goals

- Support end-to-end Console send for every provider already supported by `chat_api_call()`.
- Preserve the existing direct llama.cpp path.
- Allow non-streaming provider responses to render as a single completed assistant message.
- Keep provider execution behind the native Console gateway seam.
- Show blocked-state recovery before clearing the composer or appending the user message.
- Keep Settings as global config and Console Settings as active-session overrides.
- Maintain existing Console transcript, inspector, and composer behavior.
- Remove llama-only WIP blocking for every provider key already dispatched by `chat_api_call()`, with representative execution coverage by provider category.

## Non-Goals

- Do not rewrite all providers as native async HTTP implementations in this slice.
- Do not rebuild the full Settings provider catalog picker in this slice.
- Do not introduce new provider SDK dependencies.
- Do not store or display raw API key values.
- Do not move provider runtime process management into Settings.
- Do not remove the existing direct llama.cpp path.

## Supported Provider Scope

The supported Console provider set is the normalized key set exposed by `Chat_Functions.API_CALL_HANDLERS`, plus direct aliases already handled by Console:

- Cloud/API providers: `openai`, `anthropic`, `cohere`, `groq`, `openrouter`, `deepseek`, `mistral`, `mistralai`, `google`, `huggingface`, `moonshot`, `zai`.
- Local/OpenAI-compatible or local runtime providers: `llama_cpp`, `local_llamacpp`, `local_llamafile`, `ollama`, `local_ollama`, `vllm`, `local_vllm`, `koboldcpp`, `oobabooga`, `tabbyapi`, `aphrodite`, `local-llm`, `mlx_lm`, `local_mlx_lm`, `custom-openai-api`, `custom-openai-api-2`.

Provider keys must be normalized through the existing `provider_config_key()` behavior unless a legacy handler requires a specific hyphenated key. The design must avoid introducing a second normalization scheme.

### Provider Identity Contract

Implementation must add one provider-support helper used by all Console surfaces. The helper must resolve three identities:

- Display/config key: the value shown in Console Settings and used to locate `[api_settings.<provider>]`.
- Readiness key: the normalized key used by `get_provider_readiness()`.
- Execution key: the exact `api_endpoint` key passed to `chat_api_call()`.

Most providers use the same normalized key for readiness and execution. Non-identical mappings must be explicit:

| Display/config input | Readiness key | Execution key |
| --- | --- | --- |
| `Custom`, `custom` | `custom` | `custom-openai-api` |
| `Custom-2`, `custom_2`, `custom-2` | `custom_2` | `custom-openai-api-2` |
| `local-llm`, `local_llm` | `local_llm` | `local-llm` |
| `mlx_lm`, `local_mlx_lm` | `local_mlx_lm` | `local_mlx_lm` |
| `MistralAI`, `mistralai` | `mistralai` | `mistralai` |
| `llama_cpp`, `local_llamacpp` | direct llama key | direct llama path, not `chat_api_call()` |

The helper must also preserve exact execution keys already present in `API_CALL_HANDLERS`; for example, if a user enters `custom-openai-api`, that value resolves to readiness key `custom` and execution key `custom-openai-api`.

`mlx_lm` is not a distinct durable config identity in this slice. It is a user/provider alias of `local_mlx_lm` because readiness and default config are keyed as `local_mlx_lm`. Tests must prove both inputs resolve to readiness key `local_mlx_lm` and execution key `local_mlx_lm`.

The same helper must be used by:

- `ConsoleProviderGateway`.
- `build_console_provider_options()`.
- `build_console_settings_readiness()`.
- Console composer blocked-send preflight.
- Console control bar and Run Inspector readiness/status copy.

## Architecture

Use `ConsoleProviderGateway` as the single Console-owned execution seam.

The gateway has two execution paths:

1. Direct native llama.cpp path.
2. Generic legacy provider adapter path.

The direct path remains responsible for:

- Normalizing llama.cpp base URLs.
- Checking `/health`.
- Discovering `/v1/models` when no model is selected.
- Streaming `/v1/chat/completions`.
- Falling back to non-streaming completion if streaming emits no content.

The generic adapter path is responsible for:

- Resolving provider readiness through `get_provider_readiness()`.
- Resolving model and runtime parameters from `ConsoleProviderSelection`.
- Calling `chat_api_call()` with explicit Console message history and provider parameters.
- Converting provider responses into an async stream of transcript chunks.

This avoids duplicating provider-specific code while keeping Console independent from legacy UI widgets.

### Gateway Dependencies

`ConsoleProviderGateway` must accept injectable runtime dependencies:

- `config_provider`: callable returning the current read-only config mapping used for `[api_settings.*]`, `[chat_defaults]`, and provider-specific defaults.
- `environ`: optional mapping for deterministic API-key readiness tests.
- `chat_api_call_fn`: defaults to `Chat_Functions.chat_api_call` but is injectable for tests.
- Optional redaction/safe-error helper, defaulting to a local conservative redactor.

The gateway must not cache a stale config snapshot across Settings saves. It should call `config_provider()` during readiness/send resolution, or be rebuilt whenever the Console synchronizes active provider state. A static config copy is acceptable only inside unit tests that intentionally pass one.

Provider settings lookup must search `[api_settings]` by normalized key, matching the existing `_provider_settings()` behavior in `console_session_settings.py`; direct dictionary lookup alone is insufficient because configured TOML keys can differ in case or separators.

### Generic `chat_api_call()` Invocation

The generic adapter must pass these kwargs:

- `api_endpoint=<execution key>`
- `messages_payload=<Console message list>`
- `api_key=<readiness.api_key>`
- `model=<selected or configured model>`
- `streaming=<ConsoleProviderSelection.streaming>`
- `temp=<ConsoleProviderSelection.temperature>`
- `topp=<ConsoleProviderSelection.top_p>`
- `maxp=<ConsoleProviderSelection.top_p>`
- `topk=<ConsoleProviderSelection.top_k>`
- `minp=<ConsoleProviderSelection.min_p>`
- `max_tokens=<ConsoleProviderSelection.max_tokens>`

Optional fields may be omitted only when their value is `None`. The adapter should not invent provider-specific defaults when `chat_api_call()` or the provider handler already reads them from config.

Model rules:

- Providers with a selected or configured model pass that model.
- Providers whose existing handler can safely infer model from config may proceed with a blank explicit model only if `[api_settings.<provider>]` has a non-empty `model`, `api_model`, or `default_model`.
- Otherwise, readiness blocks with "Select a model before sending."

Base URL rules:

- Direct llama.cpp providers may keep using session `base_url` overrides because the direct path owns the HTTP URL.
- Generic adapter providers must use their existing persisted `[api_settings.<provider>]` endpoint/config path in this slice.
- If `ConsoleProviderSelection.base_url` is blank or matches the configured provider endpoint after normalization, proceed.
- If a generic URL-based provider has a non-blank session `base_url` that differs from the configured endpoint, block before send with `Provider blocked: save the endpoint in Settings before using it from Console.`
- Do not pass `api_url` or `base_url` kwargs to `chat_api_call()` in this slice because the current dispatcher does not expose a safe generic override contract.

Implementation must avoid false blocking when Console Settings pre-populates `base_url` from persisted config. The block applies only when the normalized session URL differs from the normalized configured endpoint.

## Data Flow

```text
+-------------------+      +----------------------+      +----------------------+
| Console composer | ---> | ConsoleChatController | ---> | ConsoleProviderGateway |
+-------------------+      +----------------------+      +----------------------+
                                      |                             |
                                      |                             +--> llama.cpp direct async path
                                      |                             |
                                      |                             +--> generic chat_api_call adapter
                                      |
                                      +--> ConsoleChatStore transcript updates
```

Submit flow:

1. User submits a Console composer draft.
2. `ConsoleChatController.submit_draft()` validates draft text and workspace policy.
3. Controller calls `ConsoleProviderGateway.resolve_for_send(selection)`.
4. Gateway blocks with visible recovery copy if provider/model/key/endpoint is not usable.
5. Only after resolution is ready does the controller append the user message.
6. Controller requests provider output through `stream_chat()`.
7. Streaming chunks or a single completed response update the assistant message.

## Provider Resolution

Resolution must produce:

- Provider key.
- Readiness key.
- Execution key.
- Effective model.
- API key if required and available.
- API key source label without raw value.
- Base URL or endpoint when needed.
- Runtime generation settings.
- `ready` boolean.
- User-facing blocked/recovery copy.

Resolution rules:

- Empty provider blocks with "Select a provider and model before sending."
- Provider not in the supported handler set blocks with "Provider blocked: `<provider>` is not available in Console yet."
- Missing model blocks when the provider cannot infer one safely.
- Missing API key blocks for key-required providers.
- Keyless local providers may proceed without credentials.
- Invalid local endpoint blocks with an example valid URL.
- Generic provider base URL overrides that are not persisted in config block with clear recovery copy.
- Secrets are never printed in resolution text, transcript copy, inspector copy, or test output.

The existing `NATIVE_CONSOLE_PROVIDER_KEYS` gate must not remain a llama-only blocker. It should be replaced or fed by the shared provider-support helper so generic supported providers return `native_send_supported=True` when readiness passes.

Surfaces that must use the same readiness result:

- Console Settings summary.
- Composer blocked-send preflight.
- Control bar provider/model labels.
- Run Inspector provider row.
- Transcript blocked-state recovery copy.

## Streaming And Non-Streaming Behavior

Providers may behave differently:

- If the provider returns a streaming iterator or generator, Console renders chunks as they arrive.
- If the provider returns a string, Console renders it as one completed assistant message.
- If the provider returns a dict, Console extracts the best user-visible content field where possible and otherwise renders a safe compact diagnostic.
- If the provider returns no usable content, Console marks the assistant message failed with recovery text.

The composer should clear only after the send is accepted. A readiness failure must preserve the draft.

### Sync-To-Async Bridge

The generic adapter must not iterate synchronous provider generators on the Textual event loop.

Required bridge behavior:

- Run the synchronous `chat_api_call()` invocation and any synchronous iterator consumption in a worker thread.
- Push ordered text chunks into an `asyncio.Queue`.
- Use a final sentinel to indicate completion.
- Push sanitized exception records through the same queue.
- Respect Console stop/cancel requests by stopping queue consumption and, where possible, signaling the worker to stop consuming further chunks.
- Preserve chunk order exactly as emitted by the provider.
- Never block the Textual event loop while waiting for a synchronous provider.

If a provider returns a complete string or dict, the worker should normalize it and enqueue exactly one content item plus the final sentinel.

Stop/cancel cannot forcibly terminate every synchronous provider call. The required user-facing contract is that Console stops consuming worker output immediately, marks the visible assistant response as stopped or failed according to existing Console semantics, and safely ignores any late chunks produced by a worker that finishes afterward.

### Response Normalization

Allowed streamed item types:

- `str`
- `bytes`, decoded as UTF-8 with replacement
- dict-like chunks

Dictionary extraction precedence:

1. `choices[0].delta.content`
2. `choices[0].message.content`
3. `choices[0].text`
4. `message.content`
5. `content`
6. `text`
7. `response`
8. `generated_text`

Unknown dictionary fallback:

- Do not dump the full dictionary.
- Emit `Provider returned an unsupported response shape.` as the visible copy.
- Include provider name and safe response type in logs only.

Empty content:

- If every chunk normalizes to empty content, mark the assistant message failed with `Provider returned no assistant content.`

Unsupported item types:

- Lists, response objects, SDK-specific objects, and other unsupported shapes must not be dumped into the transcript.
- Emit `Provider returned an unsupported response shape.` as the visible copy.
- Include only provider name, safe response type, and safe category in logs.

## Settings And Console Settings Relationship

Settings remains the durable global configuration surface:

- `[chat_defaults]` owns global default provider, model, streaming, and common generation defaults.
- `[api_settings.<provider>]` owns provider-specific keys, endpoints, and defaults.

Console Settings remains the active-session control surface:

- Session provider/model/base URL overrides are allowed.
- Session overrides affect the current Console session without rewriting global defaults.
- Console must list provider options from the configured provider/model registry and preserve custom values.

Full Settings provider catalog UX is a follow-up. This slice should not block on replacing the existing free-text Settings provider fields.

## Error Handling

User-visible failures must be actionable:

- Missing key: "Set `<ENV_VAR>` or add `api_key` under `[api_settings.<provider>]`."
- Missing model: "Select a model before sending."
- Invalid endpoint: "Use an http(s) URL such as `http://127.0.0.1:9099`."
- Unsupported provider: "Provider blocked: `<provider>` is not available in Console yet."
- Provider exception: summarize provider name and safe error category without leaking payloads or secrets.

The Run Inspector should show the same provider status as the control bar. The transcript should show blocked-state recovery when a send is rejected.

Safe error handling rules:

- Do not expose raw `str(exc)` directly in transcript, inspector, run-state, notifications, or tests.
- Classify provider exceptions as authentication, rate limit, bad request, provider unavailable/network, configuration, or unexpected provider error.
- Include provider name and status code when available.
- Redact API-key-like, token-like, password-like, bearer-token, and URL credential substrings before display.
- Logs may include traceback context but must still avoid raw API keys.

## Test Strategy

Focused tests should cover:

- A non-llama provider resolves through the generic adapter instead of WIP-blocking.
- Every provider key present in `API_CALL_HANDLERS` resolves as either supported for Console send or an explicit unsupported-provider recovery state; supported handler keys must not return the old llama-only WIP copy.
- A key-required provider with no key blocks before composer clear.
- A key-required provider with an injected env key proceeds without exposing the key.
- A non-streaming provider response renders as a completed assistant message.
- A streaming provider generator renders incrementally.
- Existing llama.cpp direct-path tests still pass.
- Provider/model labels in Console control bar and Run Inspector match the provider selection used for execution.
- Unsupported provider copy remains explicit and recoverable.
- Provider alias mapping covers `custom` to `custom-openai-api`, `custom_2` to `custom-openai-api-2`, and `local_llm` to `local-llm`.
- Provider alias mapping covers `mlx_lm` and `local_mlx_lm` as the same durable `local_mlx_lm` provider.
- Generic URL-based provider base URL overrides are rejected unless they match persisted config.
- A generic provider with a session URL matching persisted config after normalization proceeds and is not falsely treated as an override.
- `build_console_settings_readiness()` does not WIP-block a supported generic provider.
- A synchronous generator provider is consumed through the worker queue without blocking the async test loop.
- Stop/cancel stops UI consumption of a synchronous generic provider stream and ignores late worker chunks.
- Raw exception text containing secret-looking values is redacted before becoming visible UI copy.
- Unsupported list/object response shapes produce compact recovery copy rather than transcript dumps.

Tests should use fake gateway dependencies or monkeypatched `chat_api_call()` rather than making live network calls. The full provider-key sweep can be a readiness/dispatch contract test; live end-to-end sends should use representative fake providers by category rather than real external services.

## Rollout Plan

Recommended implementation sequence:

1. Add regression tests for generic provider resolution and the current WIP-blocking behavior.
2. Add a small provider-support helper that normalizes provider keys against `API_CALL_HANDLERS`.
3. Update Console Settings readiness/preflight gates to use the helper instead of llama-only native keys.
4. Extend `ConsoleProviderGateway.resolve_for_send()` for generic providers with injected config/env support.
5. Add the generic `chat_api_call()` adapter path in `stream_chat()` using a worker-thread-to-async-queue bridge.
6. Centralize response normalization and safe visible error copy.
7. Preserve and rerun existing llama.cpp tests.
8. Add mounted Console tests proving non-streaming completion renders, the composer clears only after acceptance, and blocked sends preserve drafts.
9. Capture actual CDP/textual-web screenshots for generic-provider success, missing-key blocked with composer preserved, and base-URL override blocked before PR review.

## Risks And Mitigations

- Risk: `chat_api_call()` is synchronous and may block the UI.
  Mitigation: run generic calls off the UI event loop and bridge results back into the async Console stream.

- Risk: Settings saves can make cached Console config stale.
  Mitigation: pass a `config_provider()` callable into the gateway or rebuild the gateway on Console provider-state synchronization.

- Risk: provider handlers return inconsistent shapes.
  Mitigation: centralize response-to-text/chunk normalization in the gateway and test string, dict, iterator, empty, list, and unsupported object results.

- Risk: provider key normalization can break legacy aliases.
  Mitigation: resolve against `API_CALL_HANDLERS` through the explicit provider identity table and test every non-identical alias.

- Risk: Console preflight blocks generic providers before the gateway can run.
  Mitigation: replace llama-only readiness gates with the shared provider-support helper and add mounted preflight regression tests.

- Risk: Settings can save a provider that is present in config but absent from dispatch.
  Mitigation: Console readiness must say unsupported rather than silently trying to send.

- Risk: API key diagnostics leak secrets.
  Mitigation: reuse `provider_readiness.py` source labels and redact any provider exception text before displaying it.

- Risk: Stop appears broken for sync providers that cannot be forcibly terminated.
  Mitigation: stop visible UI consumption immediately, ignore late worker chunks, and document that provider process cancellation is best-effort in this adapter slice.

## Acceptance Criteria

- Console can send through representative generic non-llama provider paths using the same gateway used by production sends.
- Every provider key already present in `API_CALL_HANDLERS` is recognized by Console provider support and no longer receives the old llama-only WIP block.
- Non-streaming provider output appears as a completed assistant message.
- Missing credentials and unsupported providers block before composer clear.
- Console Settings readiness and composer preflight no longer WIP-block supported generic providers.
- Provider alias mapping is explicit and tested for non-identical config/readiness/execution keys.
- Generic base URL overrides block only when they differ from persisted config after normalization.
- Stop/cancel behavior is safe for synchronous generic provider streams.
- Existing llama.cpp behavior is preserved.
- Tests cover readiness, non-streaming, streaming, unsupported-provider paths, response-shape normalization, redaction, and cancellation.
- Actual CDP/textual-web screenshots are captured for approval before merging implementation, including generic success and blocked-send recovery states.
