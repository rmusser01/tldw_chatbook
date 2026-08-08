# QwenCloud Dual-API Provider Design

Date: 2026-08-02
Revised: 2026-08-07
Status: Architecture approved; pending final document review
Backlog task: [TASK-3603](../../../backlog/tasks/task-3603%20-%20Add-QwenCloud-dual-API-provider-support.md)
Architecture decision: [ADR-045](../../../backlog/decisions/045-qwencloud-dual-api-provider-boundary.md)

## Purpose

Add QwenCloud as a normal first-class Chatbook API provider, with the same
provider identity, Console, readiness, model-selection, model-catalog, error,
and native function-tool contracts used by providers such as OpenAI and
DeepSeek.

QwenCloud has one provider-specific transport choice: `api_mode`. Users can
select `responses` or `chat_completions` in Settings and configuration.
`responses` is the default. That choice changes only the QwenCloud adapter's
external request and response translation; it does not create a second
provider identity or a Qwen-specific Console/tool runtime.

## Source Findings

- QwenCloud uses `DASHSCOPE_API_KEY` as its standard credential environment
  variable.
- Its international OpenAI-compatible base URL is
  `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`.
- Chat Completions and Responses use different message, function-definition,
  function-result, and streaming event shapes.
- Responses may silently ignore unrecognized parameters, so the adapter needs
  an explicit mode-specific allowlist.
- QwenCloud model documentation and API compatibility lists can change
  independently. Chatbook must not infer API-mode compatibility from a model
  name.
- Some Qwen text-only models reject array-shaped content even when it contains
  only text. Pure text arrays must be collapsed before submission.

Primary references:

- [QwenCloud skills index](https://www.qwencloud.com/skills.md)
- [Qwen3.8-Max API reference](https://www.qwencloud.com/models/qwen3.8-max#api-reference)
- [Qwen through Chat Completions](https://www.alibabacloud.com/help/en/model-studio/qwen-api-via-openai-chat-completions)
- [Qwen through Responses](https://www.alibabacloud.com/help/en/model-studio/qwen-api-via-openai-responses)

## Goals

- Register `qwencloud` everywhere an existing hosted provider is registered:
  provider identity, dispatcher, parameter map, readiness, Settings, Console,
  configuration, metrics/errors, and model catalog.
- Persist `api_mode` under `[api_settings.qwencloud]`, default it to
  `responses`, and accept exactly `responses` or `chat_completions`.
- Support streaming and non-streaming text in both modes.
- Support Chatbook's existing function tools in both modes through the native
  Console agent path, including multiple calls and structured tool-result
  continuation.
- Normalize both QwenCloud APIs to the same internal OpenAI-style message and
  streaming contract already consumed by Console and `AgentService`.
- Keep endpoint and credential overrides configurable without borrowing
  another provider's settings.
- Join the existing cached cloud-provider model-discovery pipeline.
- Fail before network I/O for invalid local configuration and unsupported
  request shapes.

## Non-Goals

- Do not expose or execute QwenCloud-hosted web search, code interpreter, file
  search, image generation, or other built-in tools. Only existing Chatbook
  function tools are in scope.
- Do not add a QwenCloud SDK dependency; use the repository's HTTP stack.
- Do not create a Qwen-specific agent loop, stream accumulator, tool executor,
  transcript store, continuation dispatcher, or approval path.
- Do not restore the retired legacy Chat/CCP streaming pipeline. ADR-026 makes
  native Console the live interactive chat surface; CCP remains a conversation
  management/display surface.
- Do not use `previous_response_id` or persist QwenCloud server-side response
  state.
- Do not change the conversation database schema or add durable tool metadata.
- Do not infer model/API compatibility from model-name patterns.
- Do not make paid requests in the default automated test suite.

## Provider-Parity Principle

`qwencloud` is treated as a peer of `openai`, `deepseek`, and the other hosted
providers:

| Boundary | QwenCloud behavior |
| --- | --- |
| Provider selection | One selectable provider named `QwenCloud` |
| Execution identity | One normalized key: `qwencloud` |
| Dispatch | Dedicated handler behind `chat_api_call()` |
| Console | Uses normal provider resolution and streaming gateway |
| Native tools | Uses `NATIVE_TOOLS_PROVIDERS`, `AgentService`, and `agent_runtime` |
| Readiness | Uses the shared readiness contract plus Qwen mode validation |
| Settings | Uses the canonical F9 Settings provider surface |
| Models | Uses the existing provider catalog, cache, selector, and search flow |
| Errors/usage | Uses normal typed provider errors and provider label `qwencloud` |

No caller branches on QwenCloud to execute tools or continue a conversation.
Only the adapter branches on `api_mode` to map wire formats.

## Configuration Contract

The embedded/default configuration adds one provider entry and one provider
settings table using the same timeout/retry vocabulary as existing providers:

```toml
[providers]
QwenCloud = ["qwen3.8-max"]

[api_settings.qwencloud]
api_key_env_var = "DASHSCOPE_API_KEY"
api_base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
api_mode = "responses"
model = "qwen3.8-max"
timeout = 120
retries = 3
retry_delay = 1
streaming = true
```

`api_mode` resolves from `[api_settings.qwencloud].api_mode`, then the hard
default `responses`. Surrounding whitespace and case are normalized before
validation. Aliases and unknown values are rejected before a request.

This is a durable provider setting, not a Console session override. A Console
run resolves the selected provider/model/settings once through the existing
`ConsoleProviderResolution`; every model turn in that run therefore uses the
same QwenCloud mode. Credentials remain resolved by the established provider
credential boundary and are not copied into persisted run state.

Credential precedence:

1. Explicit key supplied by a trusted caller.
2. Environment variable named by `api_key_env_var`.
3. `DASHSCOPE_API_KEY`.
4. Existing config-backed credential fallback, if enabled by the shared
   credential policy.

QwenCloud must never fall back to OpenAI, DeepSeek, or Custom OpenAI endpoint
or credential settings.

## Provider Identity And Registration

| Surface | Value |
| --- | --- |
| Display label | `QwenCloud` |
| Normalized/config/readiness key | `qwencloud` |
| Dispatcher execution key | `qwencloud` |
| Metrics and error label | `qwencloud` |
| Default credential variable | `DASHSCOPE_API_KEY` |

Registration uses the existing identity helpers and provider inventory. It
must appear wherever an ordinary Console-sendable hosted provider appears.
It must not masquerade as OpenAI or Custom OpenAI merely because both external
APIs are OpenAI-compatible.

## Architecture

```text
Console / ordinary chat_api_call caller
                 |
                 v
        shared provider dispatcher
                 |
                 v
        chat_with_qwencloud()
                 |
        +--------+---------+
        |                  |
        v                  v
 Responses mapper   Chat Completions mapper
        |                  |
 POST /responses   POST /chat/completions
        |                  |
        +--------+---------+
                 |
                 v
   standard choices/message/delta contract
                 |
        +--------+---------+
        |                  |
        v                  v
 Console gateway     ordinary non-stream caller
        |
        v
 existing native accumulator -> AgentService -> agent_runtime
```

The QwenCloud adapter owns only:

- effective-mode validation;
- endpoint construction;
- mode-specific parameter filtering;
- message and function-tool translation;
- response and stream-event normalization;
- QwenCloud error classification and safe retry behavior.

The existing systems retain their current ownership:

- Console resolves provider/model/settings and owns streaming/cancellation.
- `console_provider_gateway` accumulates standard tool-call fragments by
  index.
- `AgentService` selects native tools, normalizes call IDs, and parses calls.
- `agent_runtime` performs approval/review, executes tools, appends assistant
  tool-call messages and paired `role="tool"` results, enforces budgets, and
  detects cycles.
- Existing stores and run logs own transcript and execution persistence.

## Endpoint Contract

The configured value is a base API URL. Resolution:

1. Trim whitespace and trailing slashes.
2. Remove one recognized terminal suffix, `/responses` or
   `/chat/completions`, if the user pasted a complete endpoint.
3. Append `/responses` for Responses mode or `/chat/completions` for Chat
   Completions mode.

Arbitrary HTTP(S) compatible bases are retained for Token Plan or future
compatible endpoints. Readiness rejects missing schemes, non-HTTP(S) URLs,
embedded credentials, and malformed endpoint paths. Model discovery uses
`GET {normalized_base}/models`.

## Canonical Message Contract

Chatbook remains chat-shaped internally. Translation is pure and never mutates
the caller's messages or tool definitions.

Common rules:

- Preserve supported roles and message order.
- Collapse a content array containing only text into one string.
- Accept text and user-role `image_url` parts in this slice.
- Reject images on other roles and reject audio, video, file, or unknown
  parts before network I/O.
- Preserve empty assistant content when it accompanies tool calls.
- Preserve assistant `tool_calls`, tool `tool_call_id`, and string tool-result
  content exactly through the internal contract.

Chat Completions mapping:

- Send system/user/assistant messages in chat shape.
- Send assistant calls under `assistant.tool_calls`.
- Send results as `role="tool"` plus `tool_call_id`.
- Retain accepted user `image_url` parts.

Responses mapping:

- Convert ordinary messages to Responses input messages/items.
- Keep simple text input string-shaped; convert mixed user text/image content
  to `input_text` and `input_image` parts.
- Convert prior assistant tool calls to `function_call` items.
- Convert paired tool results to `function_call_output` items using
  `tool_call_id` as `call_id`.
- Reject orphaned tool results before network I/O.

Responses requests are stateless: each continuation sends the necessary
canonical history. `previous_response_id` is not used.

## Existing Function Tools

The input contract is Chatbook's existing OpenAI-compatible function schema:

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

Chat Completions sends that nested shape. Responses flattens the inner
function fields into the documented Responses function-tool shape.

Validation requires a non-empty name, object-shaped parameters, unique names,
and `type="function"`. QwenCloud built-in tool types are rejected locally.
Absent `tool_choice`, `auto`, and `none` are supported; other values require
separate compatibility work.

For Responses output, `function_call.call_id` becomes the canonical OpenAI
tool-call `id`. The output item's separate transport `id` is never used to
pair results:

```text
Responses call_id -> canonical tool_call.id
                  -> AgentService ToolCall.call_id
                  -> role=tool tool_call_id
                  -> Responses function_call_output.call_id
```

Adding `qwencloud` to `NATIVE_TOOLS_PROVIDERS` is allowed only after a contract
test proves all three native-provider invariants: the dispatcher forwards
`tools`, responses preserve standard `message.tool_calls`, and the adapter
accepts the assistant/tool continuation history produced by `agent_runtime`.

## Parameter Policy

The adapter constructs payloads from explicit allowlists and never forwards a
generic kwargs dictionary wholesale.

Responses mode forwards only documented, supported values such as `model`,
translated `input`, `stream`, `temperature`, `top_p`, function `tools`, and
supported `tool_choice`. Reasoning effort is forwarded only when QwenCloud
documents it for this API. Unsupported generic max-token, seed, penalty,
response-format, verbosity, reasoning-summary, and stop fields are omitted.

Chat Completions mode forwards only documented chat parameters. Chatbook's
maximum output setting maps to `max_completion_tokens` where the endpoint
supports it. Function tools retain the nested chat shape.

Settings must explain intentional omissions without suggesting that every
Qwen model supports both modes.

## Response Normalization

Every successful non-streaming result normalizes to the same shape consumed by
existing providers:

```json
{
  "choices": [
    {
      "message": {
        "content": "...",
        "tool_calls": []
      }
    }
  ]
}
```

Text and tool calls may coexist and neither is discarded. Responses
`function_call.call_id` becomes the normalized call ID; arguments remain the
JSON string expected by the existing accumulator/parser.

Streaming Chat Completions preserves normal OpenAI `choices[0].delta`
fragments. It does not add a second adapter-level complete-call buffer.

Streaming Responses uses a stateful wire translator because typed Responses
events are not chat deltas. It tracks each output item and emits standard
`delta.content` and `delta.tool_calls` fragments with stable numeric indexes.
The existing Console accumulator remains the one owner that merges names,
IDs, and interleaved argument fragments into complete calls.

The translator must distinguish argument delta events from terminal events so
a final full-arguments value is not appended after its deltas. Duplicate or
replayed completion events cannot emit a call twice. A terminal response with
an incomplete call identity or arguments raises a malformed-provider-response
error.

Provider error events raise the project's typed `ChatProviderError` with the
provider label `qwencloud`; they are never emitted as a successful content
chunk. Once any response byte/event is consumed, the request is not retried.

## Continuation, Cancellation, And State

There is no QwenCloud continuation implementation. The existing native agent
loop owns continuation exactly as it does for OpenAI or DeepSeek:

1. The normalized assistant tool-call message is appended to loop history.
2. Existing approval/review policy runs before dispatch.
3. Each completed call is executed once through the existing registry.
4. Each result or tool error is appended as a paired `role="tool"` message.
5. The next provider call receives that canonical history; the Qwen adapter
   maps it to the selected external API.

Partial streamed calls are not exposed to the executor. Stop/cancellation,
tool timeouts, model-turn/step/wall/token budgets, cycle detection, result
truncation, and run logging remain existing runtime behavior. No Qwen-specific
round cap or session ledger is introduced.

Conversation persistence is unchanged. QwenCloud transport objects and
`previous_response_id` are not persisted, and credentials never enter the
conversation or run-log payload.

## Readiness And Settings UX

The canonical F9 Settings provider surface adds an `API mode` selector to the
QwenCloud connection settings:

- options: `Responses` and `Chat Completions`;
- absent value: `Responses`;
- saved values: `responses` and `chat_completions` only;
- visible/enabled only for QwenCloud;
- provider switching preserves the unsaved QwenCloud draft during that
  Settings session;
- saving another provider does not create or alter QwenCloud `api_mode`;
- invalid persisted values show a blocking validation state;
- help copy describes mode-specific parameter behavior.

QwenCloud otherwise follows ordinary provider readiness: model, endpoint,
credential, and provider identity resolve through shared contracts. Mode
validation is an additional Qwen adapter/readiness check, not a new readiness
system. Recovery copy names QwenCloud and never includes credentials.

## Model Discovery

QwenCloud joins the existing ADR-002/ADR-020 pipeline:

- authenticated `GET {normalized_base}/models`;
- provider identity `qwencloud`;
- model IDs and timestamps only in the cache;
- existing TTL, capped merge, searchable model popover, refresh notification,
  and append-only opt-in write-through;
- configured/cached fallback when discovery fails.

Base forms ending in `/compatible-mode/v1`, `/responses`, or
`/chat/completions` normalize to the same `/compatible-mode/v1/models`
endpoint without constraining the host.

If the selected model rejects an API mode, recovery copy recommends choosing
a compatible model or switching mode. Chatbook does not claim a static
compatibility matrix.

## Error And Retry Policy

- Invalid mode, endpoint, tool schema, message history, or local request shape
  fails before network I/O.
- Authentication/authorization and other non-transient `4xx` failures are not
  retried.
- Model/mode incompatibility is not retried and produces actionable recovery
  copy.
- Rate limits, transient server errors, connection-establishment failures,
  and timeouts use the existing bounded provider retry configuration.
- Streaming requests are never replayed after the first response byte/event.
- A successful HTTP status with no usable text or tool calls is a malformed
  provider response, not an empty successful answer.
- Errors and logs use `qwencloud` while excluding credential values and
  private message/tool payloads.

## Testing Strategy

Implementation follows test-driven development.

Provider-parity tests:

- identity, dispatcher, parameter-map, Console-sendable inventory, readiness,
  Settings, and model-catalog registration;
- a registry invariant proving every native-tool provider forwards `tools`,
  returns standard calls, and accepts canonical tool history;
- unchanged behavior for representative OpenAI and DeepSeek requests.

Adapter tests for both modes:

- exact URL, headers, timeout/retry settings, and parameter allowlists;
- text-array collapse and supported image conversion without input mutation;
- function schema and assistant/tool-history translation;
- text-only, tool-only, and mixed text/tool results;
- multiple and interleaved calls with exact `call_id` round-trip;
- Responses delta/done de-duplication;
- typed errors, malformed terminals, retry counts, and no retry after stream
  consumption;
- secret-redaction checks.

Joined consumer tests:

- Qwen adapter -> `ConsoleProviderGateway` -> existing tool accumulator ->
  `AgentService`/`agent_runtime` -> second Qwen request;
- run once for Responses and once for Chat Completions;
- cover streaming and non-streaming provider resolution, multiple calls, a
  tool validation/execution error, and cancellation before a partial call can
  execute;
- assert the second request contains the exact assistant/tool continuation and
  does not create a synthetic user message.

Settings/model-catalog tests cover selector visibility, defaulting,
persistence isolation, validation, help copy, `/models` normalization, cache
merge, and fallback.

Tests must use real production signatures and verbatim provider fixtures;
fakes may intercept network I/O but cannot redefine the caller/callee
contract. At least one test exercises the real joined Console entry path.
Existing failures are compared against the identical command on the baseline,
not dismissed by count alone.

Optional live tests are skipped by default and require:

- `DASHSCOPE_API_KEY`;
- `TLDW_LIVE_QWENCLOUD=1`;
- optional `TLDW_LIVE_QWENCLOUD_MODEL`.

The live gate uses an isolated scratch profile and the repository venv. It
performs the smallest text and function-call check in each mode and records
identifying response content rather than merely asserting no exception.

## Documentation And Delivery

Provider documentation records setup, credential variable, international base
URL, both `api_mode` values and default, parameter limitations, existing
function-tool support, built-in-tool exclusion, configurable compatible bases,
and optional live-test instructions.

No schema migration is required. Configuration remains TOML-backed.

## ADR Check

ADR required: yes

ADR path:
[backlog/decisions/045-qwencloud-dual-api-provider-boundary.md](../../../backlog/decisions/045-qwencloud-dual-api-provider-boundary.md)

Reason: exposing two external APIs under one durable provider identity and
normalizing both into the shared Console/native-tool runtime is a long-lived
provider boundary. ADR-045 records that decision. It follows ADR-006 for
Settings/adapter ownership, ADR-012 for credentials, ADR-020 for model catalog
refresh, and ADR-026 for native Console ownership.

## Acceptance Summary

QwenCloud is complete when it behaves like an ordinary selectable provider:
users configure it in Settings, select it in Console, stream or complete text
through either API mode, use existing Chatbook function tools through the same
native agent loop as OpenAI/DeepSeek, discover models through the standard
catalog, receive typed safe errors, and do not encounter provider-specific
continuation or persistence behavior outside the adapter's wire translation.
