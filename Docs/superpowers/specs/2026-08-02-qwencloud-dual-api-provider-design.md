# QwenCloud Dual-API Provider Design

Date: 2026-08-02
Status: Independently reviewed; pending final user approval
Backlog task: [TASK-1336](../../../backlog/tasks/task-1336%20-%20Add-QwenCloud-dual-API-provider-support.md)

## Purpose

Add QwenCloud as a first-class Chatbook provider with an explicit API mode. Users can choose QwenCloud Responses or Chat Completions while retaining Chatbook's existing function-tool execution loop in either mode.

Responses is the default for new QwenCloud configurations. Chat Completions remains selectable for compatibility and for parameters or models that QwenCloud exposes only through that API.

## Source Findings

- QwenCloud's published skill index identifies `DASHSCOPE_API_KEY` as the standard credential environment variable.
- QwenCloud exposes an OpenAI-compatible base URL at `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`.
- The APIs use different request shapes for messages and function tools: Chat Completions nests function definitions under `function`; Responses uses flat function definitions and distinct `function_call` / `function_call_output` input items.
- Responses streaming emits typed events rather than Chat Completions `choices[].delta` chunks.
- QwenCloud documents that unrecognized Responses parameters may be ignored. Silent forwarding is therefore unsafe: Chatbook must use mode-specific allowlists.
- The current stable `qwen3.8-max` model page and the current Responses reference are not fully synchronized. The model page advertises stable `qwen3.8-max`, while the Responses reference currently enumerates `qwen3.8-max-preview`. Chatbook must not hard-code API compatibility by model name.
- Qwen text-only models may reject array-shaped content even when the array contains only text. Text-only structured content must be collapsed to a string before submission.

Primary references:

- [QwenCloud skills index](https://www.qwencloud.com/skills.md)
- [Qwen3.8-Max model and API reference](https://www.qwencloud.com/models/qwen3.8-max#api-reference)
- [Qwen through the OpenAI Chat Completions API](https://www.alibabacloud.com/help/en/model-studio/qwen-api-via-openai-chat-completions)
- [Qwen through the OpenAI Responses API](https://www.alibabacloud.com/help/en/model-studio/qwen-api-via-openai-responses)

## Goals

- Register `qwencloud` across provider dispatch, identity, readiness, Settings, Console, configuration, and model-catalog surfaces.
- Persist `api_mode` under `[api_settings.qwencloud]`.
- Default `api_mode` to `responses`; permit exactly `responses` and `chat_completions`.
- Support streaming and non-streaming text responses in both modes.
- Support only Chatbook's existing function tools in both modes, including multi-call turns and tool-result continuation.
- Normalize both external APIs into the OpenAI-style chat/tool shape consumed by both Console and the main Chat/CCP worker paths.
- Keep QwenCloud endpoint and credential overrides configurable, including Token Plan or future compatible endpoints.
- Add QwenCloud to the existing cached model-discovery pipeline.
- Fail before network I/O for invalid local configuration or unsupported request shapes.

## Non-Goals

- Do not expose or execute QwenCloud built-in web-search, code-interpreter, file-search, image-generation, or similar hosted tools.
- Do not add a QwenCloud SDK dependency.
- Do not generalize or migrate every OpenAI-compatible provider in this slice.
- Do not infer model/API compatibility from model-name patterns.
- Do not promise multimodal support for every Qwen model. The first slice supports text and user-role `image_url` parts only; audio, video, file, and unknown content parts fail locally and remain future work.
- Do not persist Responses server state, adopt `previous_response_id`, or change Chatbook's conversation storage contract.
- Do not make paid QwenCloud calls in the default automated test suite.

## Configuration Contract

The embedded/default configuration adds QwenCloud as follows:

```toml
[providers]
QwenCloud = ["qwen3.8-max"]

[api_settings.qwencloud]
api_key_env_var = "DASHSCOPE_API_KEY"
api_base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
api_mode = "responses"
model = "qwen3.8-max"
api_timeout = 120
api_retries = 2
api_retry_base_delay = 1.0
```

The effective API mode resolves from `[api_settings.qwencloud].api_mode`, then falls back to the hard default `responses`. There is no Console session or per-request API-mode override in this slice.

Mode values are normalized for surrounding whitespace and case, then validated against the two canonical values. Aliases are not accepted. An invalid value raises a typed provider-configuration error before any request is attempted.

Credential resolution follows the existing provider credential boundary:

1. Explicit API key passed by the trusted caller.
2. Environment variable named by `api_key_env_var`.
3. Default environment variable `DASHSCOPE_API_KEY`.

Credentials are never logged, returned in readiness copy, written to model-catalog caches, or embedded in exception text.

`api_timeout` is a per-attempt request timeout in seconds, valid from 1 through 600. `api_retries` is the number of additional attempts after the first request, valid from 0 through 5. `api_retry_base_delay` is the exponential-backoff base in seconds, valid from 0 through 30. Defaults therefore permit at most three total attempts with delays beginning at one second. Backoff is capped at eight seconds; a valid `Retry-After` value takes precedence but is capped at 30 seconds.

## Provider Identity And Registration

QwenCloud has one durable normalized identity: `qwencloud`.

| Surface | Value |
| --- | --- |
| Display label | `QwenCloud` |
| Config key | `qwencloud` |
| Readiness key | `qwencloud` |
| Dispatcher execution key | `qwencloud` |
| Metrics/error label | `qwencloud` |
| Default credential env var | `DASHSCOPE_API_KEY` |

Registration must use the existing provider identity helpers. It must not masquerade as OpenAI or Custom OpenAI, and OpenAI-specific configuration must not be used as fallback QwenCloud configuration.

## Architecture

Implement a dedicated QwenCloud adapter behind `chat_api_call()`.

```text
Chatbook messages + function tools
              |
              v
     chat_with_qwencloud()
              |
      +-------+--------+
      |                |
      v                v
 Responses mapper   Chat mapper
      |                |
 POST /responses   POST /chat/completions
      |                |
 Responses parser   Chat parser
      +-------+--------+
              |
              v
 Complete OpenAI-style chat chunks/messages/tool_calls
              |
      +-------+--------+
      |                |
      v                v
 Console gateway   Chat/CCP workers
      |                |
      +-------+--------+
              |
              v
      Existing tool executor
```

The adapter owns:

- Mode validation and effective-mode resolution.
- Endpoint construction.
- Mode-specific parameter filtering and request translation.
- Message and function-tool translation.
- Streaming and non-streaming response normalization.
- QwenCloud-specific error classification and safe retry policy.

The existing dispatcher, transcript stores, and tool executor continue to own provider selection, tool approval/execution, and conversation persistence. The two consumer paths require explicit compatibility work:

- Console already accumulates OpenAI-style tool-call deltas and preserves assistant/tool messages; QwenCloud output must remain compatible with it.
- Chat/CCP currently appends streamed tool fragments without merging them and its non-streaming parser misses `choices[0].message.tool_calls`. QwenCloud streaming therefore emits each complete normalized tool call exactly once, and the shared parser is extended to inspect the standard choices envelope.
- Chat/CCP's existing textual `[Tool Results]` continuation is insufficient for Responses. For QwenCloud only, continuation must append the normalized assistant `tool_calls` message plus one `role = "tool"` message per result to the in-memory request history, persist the existing tool-message records, and submit the continuation without creating a synthetic visible user message. Other providers retain their current continuation behavior in this slice.

The initial implementation should remain QwenCloud-focused. Pure helpers may be extracted only when their behavior is demonstrably identical and existing OpenAI tests protect the extraction. Adding QwenCloud must not change OpenAI request or response behavior incidentally.

## Endpoint Contract

The configured value is a base API URL, not a mode-specific endpoint. Endpoint resolution:

1. Trim whitespace and trailing slashes.
2. Remove one recognized terminal suffix, `/responses` or `/chat/completions`, if a user pasted a full endpoint.
3. Append `/responses` for Responses mode or `/chat/completions` for Chat Completions mode.

The canonical default base remains:

```text
https://dashscope-intl.aliyuncs.com/compatible-mode/v1
```

Arbitrary HTTP(S) compatible bases are preserved so Token Plan and future QwenCloud endpoints can be configured. The adapter does not rewrite custom hosts or invent version paths. Readiness rejects missing schemes, non-HTTP(S) URLs, credentials embedded in URLs, and already malformed endpoint paths.

Model discovery uses `GET {base}/models` after the same base normalization.

## Message Normalization

Chatbook's canonical conversation remains chat-shaped. Translation is pure and must not mutate the caller's message list.

### Common normalization

- Preserve supported roles and message order.
- Collapse a content array containing only text parts into one string. This prevents text-only Qwen models from rejecting array-shaped text input.
- Accept text parts and user-role `image_url` parts only. Never discard accepted image parts while simplifying adjacent text.
- For Chat Completions, preserve `{ "type": "image_url", "image_url": { "url": ... } }`.
- For Responses, convert it to `{ "type": "input_image", "image_url": ... }`; adjacent text becomes `input_text` when structured content is required.
- Reject images on non-user roles and reject audio, video, file, or unknown part types before network I/O.
- Reject malformed content parts with a configuration/request error that identifies the message index but does not include private content.
- Preserve empty assistant content when the assistant message carries tool calls.

### Chat Completions input

- Send ordinary `system`, `user`, and `assistant` messages in chat shape.
- Send assistant tool calls under `assistant.tool_calls`.
- Send tool results as `role = "tool"` with `tool_call_id` and string content.
- Retain validated user-role `image_url` parts for models that support them.

### Responses input

- Convert ordinary messages into documented Responses input messages.
- Keep simple text messages string-shaped. For mixed user text/image input, convert text to `input_text` and `image_url` to `input_image` without losing the URL or inline data URL.
- Convert every prior assistant tool call into a `function_call` item.
- Convert every prior Chatbook tool result into a `function_call_output` item whose `call_id` equals the originating Chatbook `tool_call_id`.
- Reject orphaned tool results whose `tool_call_id` cannot be associated with a prior function call.

Responses requests are stateless from Chatbook's perspective: each request sends the required conversation/tool history. `previous_response_id` is not used.

## Function Tool Contract

Only function tools are accepted in this feature.

Chatbook's canonical function definition is the existing Chat Completions-compatible shape:

```json
{
  "type": "function",
  "function": {
    "name": "get_current_time",
    "description": "Return the current time.",
    "parameters": {"type": "object", "properties": {}}
  }
}
```

For Chat Completions, the adapter validates and sends this nested shape.

For Responses, it flattens the function object:

```json
{
  "type": "function",
  "name": "get_current_time",
  "description": "Return the current time.",
  "parameters": {"type": "object", "properties": {}}
}
```

Validation rules:

- Reject non-function tool types, including QwenCloud built-ins.
- Require a non-empty function name and object-shaped JSON Schema parameters.
- Preserve descriptions and JSON Schema without mutating them.
- Reject duplicate function names.
- Accept an absent `tool_choice`, `auto`, or `none` in both modes. Reject `required`, forced-function objects, and every other string/object before network I/O in this first slice.

### Tool-call identity

For Responses output, QwenCloud's `call_id` is Chatbook's canonical tool-call `id`. The separate Responses output-item `id` is transport metadata and must not be used to associate the tool result.

```text
Responses function_call.call_id
        -> Chatbook tool_call.id
        -> tool executor tool_call_id
        -> Responses function_call_output.call_id
```

The round trip must support multiple calls in a turn, calls whose streaming argument fragments are interleaved, and empty JSON argument objects.

## Mode-Specific Parameter Policy

The adapter builds payloads from explicit allowlists based on current QwenCloud documentation. It never forwards a generic kwargs dictionary wholesale.

### Responses mode

Forward documented values when non-null and valid, including:

- `model`
- translated `input`
- `stream`
- `temperature`
- `top_p`
- function `tools`
- supported `tool_choice`
- `reasoning.effort` when selected and documented for the chosen API

Do not send Chatbook's generic maximum-token, seed, repetition/frequency/presence penalty, response-format, verbosity, reasoning-summary, or stop fields. Omission is intentional, not an error, even when generic saved defaults populated those fields. Supporting another field requires a later spec change rather than an implementation-time interpretation of this design.

### Chat Completions mode

Forward only documented Chat Completions parameters. Map Chatbook's maximum output setting to `max_completion_tokens`, not legacy `max_tokens`, where supported by the Qwen endpoint. Function tools use the nested chat shape.

### Settings communication

Changing API mode updates concise help text in Settings. Responses mode must explain that generation fields not documented by QwenCloud are not sent. QwenCloud may expose reasoning effort, but it must not be added to provider support lists for unsupported OpenAI-only reasoning summary or verbosity controls.

## Response Normalization

Both APIs normalize into the complete chat-shaped contract consumed by Console and Chat/CCP. Adapter tests alone are insufficient; each normalized path requires a consumer-level test.

### Non-streaming Chat Completions

Validate the response envelope, preserve assistant text, and preserve `message.tool_calls`. The shared Chat/CCP parser must recognize the complete `choices[0].message.tool_calls` envelope in addition to its existing direct-message and provider-specific shapes. A success response without usable text or tool calls is a malformed-provider-response error.

### Non-streaming Responses

Walk all output items in order:

- Concatenate supported assistant text segments.
- Convert every `function_call` item to an OpenAI-style `tool_calls[]` entry.
- Use `call_id` as the converted tool call `id`.
- Serialize arguments as the string expected by Chatbook's existing accumulator/executor.
- Set the normalized finish reason to `tool_calls` whenever one or more function calls are present; otherwise use a normal completed reason.

Text and tool calls may coexist in one response and neither may be dropped.

### Streaming Chat Completions

Preserve and emit text deltas incrementally. Accumulate each tool call's name, ID, and argument fragments inside the adapter, then emit each complete normalized tool call exactly once before the terminal finish chunk. Do not expose partial `delta.tool_calls` fragments to Chatbook consumers.

### Streaming Responses

Use a stateful event translator. It tracks each output item by output index and/or item ID and records:

- Function name.
- `call_id`.
- Tool-call index in normalized output.
- Incremental argument fragments.
- Completion state.

It converts documented QwenCloud events such as text deltas, output-item additions, function-argument deltas/done events, and response completion into ordered OpenAI-style chat SSE chunks. Text remains incremental; tool calls are buffered and each complete normalized call is emitted exactly once. A successful stream emits `[DONE]` exactly once.

A QwenCloud terminal provider-error event is not serialized as an SSE `{ "error": ... }` chunk because Chat/CCP can ignore an envelope without `choices`. Instead, the generator raises the project's typed `ChatProviderError(provider="qwencloud", ...)` with sanitized recovery text. It does not emit `[DONE]` after the error. If any response byte or event was already consumed, it does not retry the request.

Unknown informational event types are ignored only when they cannot affect text, tool-call identity, arguments, errors, or completion. A terminal response with an incomplete function-call identity or invalid arguments is a malformed-provider-response error.

## Readiness And Settings UX

Settings adds an `API mode` selector to the provider connection block.

- It is visible/enabled only for QwenCloud.
- Options are `Responses` and `Chat Completions`.
- An absent saved value displays `Responses`.
- Provider switching preserves unsaved QwenCloud mode draft state during the Settings session.
- Saving QwenCloud persists only the canonical value under `api_settings.qwencloud.api_mode`.
- Saving another provider does not create or alter an `api_mode` key.
- Invalid persisted values render a visible validation state and cannot be used for connection testing or chat submission.
- Mode-specific help explains parameter differences without implying that the selected model supports both APIs.

The shared `ProviderReadiness` contract retains its current credential-readiness meaning. QwenCloud-specific send validation is layered after shared credential readiness in Settings connection testing, Console send resolution, and Chat/CCP submission. That validation requires:

- A non-empty model.
- A valid base URL.
- A resolved API key.
- A valid API mode.

Readiness and connection-test copy identifies QwenCloud, never OpenAI, and never exposes credential values.

## Model Discovery

QwenCloud joins the existing ADR-002/ADR-020 model-catalog path:

- Discovery endpoint: authenticated `GET {normalized_base}/models`.
- Discovery eligibility and URL construction explicitly accept `/compatible-mode/v1`, `/compatible-mode/v1/responses`, and `/compatible-mode/v1/chat/completions`, normalizing all three to `/compatible-mode/v1/models` without constraining the host.
- Provider identity: `qwencloud`.
- Cache contents: model IDs and timestamps only.
- Existing TTL, capped selector merge, search popover, refresh notification, and optional append-only write-through behavior remain unchanged.
- Discovery failure falls back to configured and cached model IDs.

The adapter remains model-agnostic. A model rejected for the selected mode produces actionable recovery text: select a model compatible with the current API mode or switch API mode. The error must not claim a complete compatibility matrix.

## Error And Retry Policy

Classify failures into the project's existing typed provider error categories while retaining a safe QwenCloud label.

- Invalid mode, endpoint, tool schema, message history, or unsupported local request shape: fail before network I/O.
- Authentication/authorization errors: do not retry; point to the configured credential environment variable.
- Model/mode incompatibility or invalid parameter errors: do not retry; recommend changing model or API mode.
- Rate limits (`429`) and transient server errors (`500`, `502`, `503`, `504`): retry up to `api_retries` additional attempts with the configured capped backoff and bounded `Retry-After` handling.
- Connection establishment failures and timeouts: retry up to `api_retries` additional attempts where no response has been consumed.
- Other `4xx` responses do not retry.
- Streaming: never replay a request after the first response event or content byte has been consumed.
- Successful HTTP status with malformed body: raise a provider-response error rather than returning an empty assistant message.

Default automated tests assert retry counts and ensure secrets are absent from errors and logs.

## Testing Strategy

Follow test-driven development. Each behavior begins with a failing focused test.

### Registration and configuration

- Dispatcher registration and provider identity aliases/casing.
- Embedded config defaults.
- Config-to-default API-mode resolution and strict validation.
- Credential and endpoint resolution without cross-provider leakage.
- Readiness success and each blocked state.

### Request translation

- Responses and Chat Completions URLs, headers, timeouts, and payload allowlists.
- Text-only content-array collapse without source mutation.
- Exact user-role `image_url` conversion plus local rejection of unsupported roles/types.
- Flat versus nested tool definitions.
- Assistant tool-call and tool-result history conversion.
- Orphaned tool-result, duplicate-tool, unsupported built-in-tool, and invalid `tool_choice` failures before HTTP.
- Mode-specific parameter omission/mapping.

### Response normalization

- Text-only, tool-only, and mixed text/tool non-streaming responses in both modes.
- Multiple function calls and `call_id` round-trip.
- Chat tool-call fragment buffering and exactly-once complete-call emission.
- Responses text and interleaved function-argument events.
- `[DONE]` emitted exactly once.
- Provider error events and malformed terminal responses.
- Midstream provider-error propagation through Console and Chat/CCP, with no `[DONE]` and no retry after prior bytes/events.
- Chat/CCP choices-envelope parsing and QwenCloud native assistant/tool continuation.
- Console and Chat/CCP consumer-level function-call tests in both modes.

### Settings and model catalog

- QwenCloud-only API-mode visibility, draft preservation, save behavior, validation, and help copy.
- Other providers remain unchanged.
- Authenticated `/models` discovery, cache merge, refresh failure fallback, and exact normalization of all three `/compatible-mode/v1` endpoint forms.

### Reliability

- Exact retry defaults, bounds, statuses, attempt counts, backoff cap, and `Retry-After` cap.
- No retry after streaming begins.
- Timeout propagation and secret redaction.
- Regression tests for OpenAI plus existing Console and Chat/CCP function-tool execution.

### Optional live tests

Live tests are skipped by default and require explicit opt-in plus credentials. They use:

- `DASHSCOPE_API_KEY`
- `TLDW_LIVE_QWENCLOUD=1`
- Optional `TLDW_LIVE_QWENCLOUD_MODEL` override

The live suite contains the smallest possible text and function-call checks for each mode. It documents that model availability and billing are external and may change.

## Documentation And Delivery

Update provider-facing documentation with:

- QwenCloud setup and credential environment variable.
- Default international compatible base URL.
- `api_mode` values, config-to-default resolution, and default.
- Mode-specific parameter limitations.
- Function-tool support and built-in-tool non-goal.
- Configurable Token Plan/custom endpoint guidance.
- Optional live-test invocation.

No migrations are required because provider configuration is TOML-backed and accepts a new provider section/key.

## ADR Check

ADR required: yes, satisfied by existing ADRs

ADR paths:

- [ADR-006: Provider-Aware Generation Settings](../../../backlog/decisions/006-provider-aware-generation-settings.md)
- [ADR-020: Automatic model catalog refresh for cloud providers](../../../backlog/decisions/020-automatic-model-catalog-refresh.md)

Reason: This feature adds a provider/runtime boundary and a persisted provider-specific setting, but ADR-006 already assigns request-shape translation to adapters and provider-specific defaults to Settings. QwenCloud model discovery directly extends the ADR-020 pipeline. No new storage, ownership, security, or cross-module policy is introduced, so a duplicate ADR is not warranted.

## Acceptance Summary

The feature is ready only when QwenCloud can be configured and selected, both API modes support text and existing Chatbook function tools through Console and Chat/CCP in streaming and non-streaming flows, unsupported generic generation settings are intentionally omitted with visible mode guidance, invalid request shapes fail explicitly, model discovery uses the existing cache pipeline, Settings persists API mode safely, relevant documentation is updated, and the focused automated suite passes without paid network calls.
