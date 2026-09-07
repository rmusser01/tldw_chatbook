# Moonshot/Kimi and Z.ai/GLM Hosted Chat-Completions Design

Date: 2026-08-12
Status: Approved; written-spec review complete
Backlog task: [TASK-15676](../../../backlog/tasks/task-15676%20-%20Harden-Moonshot-Kimi-and-Z.ai-GLM-as-first-class-hosted-providers.md)
Foundation task: [TASK-15675](../../../backlog/tasks/task-15675%20-%20Add-durable-provider-tool-continuation-checkpoints.md)
Architecture decision: [ADR-063](../../../backlog/decisions/063-hosted-provider-wire-and-durable-tool-continuation.md)

## Purpose

Bring the existing `moonshot` and `zai` integrations up to the same first-class
reliability, security, streaming, usage, native-function-tool, Settings, and
model-discovery standard as the newest Chatbook providers.

Both services currently expose OpenAI-shaped Chat Completions APIs, not an
OpenAI Responses API. This design therefore supports Chat Completions only and
does not introduce `api_mode`. It implements the provider-neutral hosted
Chat-Completions wire boundary retained by ADR-063 and consumes TASK-15675's
durable continuation contract. This task migrates only Moonshot and Z.ai;
other provider behavior stays unchanged.

## Official Source Findings

### Moonshot AI / Kimi

- The production international base is `https://api.moonshot.ai/v1` and the
  documented generation route is `POST /v1/chat/completions`.
- The current official default family is Kimi K3; the OpenAPI model discriminator
  lists `kimi-k3` as the default and also lists Kimi K2.x and historical
  `moonshot-v1-*` IDs.
- Kimi K3 accepts `reasoning_effort` values `low`, `high`, and `max`; it does
  not expose the legacy Moonshot sampling fields on the K3 request schema.
- Streaming is SSE. `stream_options.include_usage=true` requests a terminal
  usage chunk before `[DONE]`.
- Function calls use standard assistant `tool_calls` and paired `role="tool"`
  results. Multiple calls are allowed and every call ID must have one matching
  result.
- Kimi thinking models return `reasoning_content`. Kimi K3 always enables
  Preserved Thinking and requires every historical assistant reasoning value
  to be kept as-is; other Kimi families follow their explicit `thinking.keep`
  policy. Multi-step tools also require the complete assistant message.
- Kimi accepts function tools plus provider-hosted/dynamic tools. Only existing
  Chatbook function tools are in scope here.
- The China regional base remains configurable as
  `https://api.moonshot.cn/v1`; custom compatible gateways remain possible.
- The published OpenAPI contains no `/responses` path.

Primary references:

- [Kimi documentation index](https://platform.kimi.ai/docs/llms.txt)
- [Kimi OpenAPI](https://platform.kimi.ai/docs/openapi.json)
- [Create Chat Completion](https://platform.kimi.ai/docs/api/chat)
- [Use Kimi API for tool calls](https://platform.kimi.ai/docs/guide/use-kimi-api-to-complete-tool-calls)
- [Using thinking models](https://platform.kimi.ai/docs/guide/use-kimi-k2-thinking-model)
- [Kimi tool use](https://platform.kimi.ai/docs/api/tool-use)

### Z.ai / GLM

- The general production base is `https://api.z.ai/api/paas/v4` and the
  documented generation route is `POST /paas/v4/chat/completions`.
- The current Chat Completion reference defaults to `glm-5.2`, which is the
  embedded default for this task. Historical GLM IDs remain selectable.
- Streaming is SSE. The terminal chunk carries `finish_reason` and `usage`,
  followed by `[DONE]`; no `stream_options.include_usage` request is required.
- Function tools use standard assistant `tool_calls` and paired tool-result
  messages. Z.ai documents `tool_choice="auto"` only for function tools.
- `thinking.type` accepts `enabled` or `disabled`. `clear_thinking=true` is the
  general-chat default, while official interleaved-tool guidance requires
  `clear_thinking=false` plus the complete unmodified `reasoning_content` for
  an active agent/tool run.
- `glm-5.2` accepts `reasoning_effort` values `none`, `minimal`, `low`,
  `medium`, `high`, `xhigh`, and `max`; Z.ai documents provider-side mappings
  for the lower and upper compatibility values.
- `tool_stream` is optional and defaults false. This task does not enable it
  automatically; the parser accepts complete or fragmented standard tool-call
  deltas without changing provider behavior.
- Provider-hosted web-search and retrieval tool types are out of scope.
- The coding base `https://api.z.ai/api/coding/paas/v4` is for supported coding
  tools and is not a substitute for the general API default.
- The published OpenAPI contains no `/responses` path.

Primary references:

- [Z.ai documentation index](https://docs.z.ai/llms.txt)
- [Z.ai OpenAPI](https://docs.z.ai/openapi.json)
- [Chat Completion API](https://docs.z.ai/api-reference/llm/chat-completion)
- [Streaming messages](https://docs.z.ai/guides/capabilities/streaming)
- [Thinking mode](https://docs.z.ai/guides/capabilities/thinking-mode)
- [Stream tool calls](https://docs.z.ai/guides/tools/stream-tool)

## Existing State And Problems

Chatbook already registers `moonshot` and `zai`, exposes both in Settings and
Console, and includes both in cloud model discovery. Their runtime handlers are
large independent functions in `LLM_API_Calls.py`, however:

- Moonshot tries to read an orphaned top-level `moonshot_api` mapping while
  Z.ai reads canonical `api_settings.zai`; direct calls, readiness, and Console
  can therefore resolve different values.
- The embedded defaults are stale (`kimi-latest` and `glm-4.5`).
- Streaming relays raw lines, fabricates `[DONE]`, embeds exception text into
  successful-looking error events, and does not reliably expose terminal usage.
- The `with Session()` scope ends before the returned generator is consumed;
  iterator/session ownership is therefore not explicit.
- Request validation and model-family allowlists are incomplete. Moonshot
  silently rewrites unsupported tool choice and injects legacy sampler defaults
  into every model; Z.ai omits standard continuation controls and tool choice.
- Non-streaming and streaming responses are validated differently.
- Z.ai is absent from `NATIVE_TOOLS_PROVIDERS`; Moonshot membership predates
  joined Console continuation/cancellation proof.
- Kimi reasoning needed for an active tool loop and K3 ordinary later turns is
  dropped by the current assistant-message handoff.

## Goals

- Preserve the stable provider keys `moonshot` and `zai`, public handler names,
  dispatcher identities, saved histories, and metrics/error labels.
- Default fresh/missing Moonshot models to `kimi-k3` and fresh/missing Z.ai
  models to `glm-5.2` without overwriting explicit user selections.
- Support strict streaming and non-streaming text, tools, errors, terminal
  states, usage, retries, cancellation, and resource ownership.
- Support existing Chatbook function tools through the ordinary Console agent
  runtime, including multiple calls and exact assistant/tool continuation.
- Preserve bounded invisible Kimi K3 reasoning for every retained assistant
  owner, and preserve Kimi-family/Z.ai tool-run reasoning under each model's
  exact durable checkpoint policy.
- Use canonical provider configuration, credential precedence, and endpoint
  normalization consistently in readiness, Console, direct adapters, and
  discovery.
- Reuse neutral Chat-Completions wire mechanics across Moonshot and Z.ai, and
  behavior-preservingly extract the already-tested QwenCloud Chat SSE primitive
  where practical.
- Leave every unrelated provider byte-for-byte unchanged in this task.

## Non-Goals

- No Moonshot or Z.ai Responses API compatibility mode.
- No vendor SDK dependency.
- No provider-hosted web search, retrieval, code runner, memory, dynamic tool
  loading, or other built-in execution.
- No new agent loop, tool executor, approval path, transcript store, model
  registry, or Settings destination. TASK-15675 owns the one conversation
  schema/checkpoint change used here.
- No provider conversation ID or server-side session ownership.
- No speculative preserved-thinking behavior beyond the documented model
  policy. Kimi K3 ordinary later turns replay retained reasoning because K3
  requires it; other Kimi families and Z.ai omit private reasoning from
  ordinary later requests unless their explicit policy says otherwise.
- No automatic Z.ai `tool_stream` opt-in.
- No complete implementation of Moonshot Flavored JSON Schema and no arbitrary
  vendor schema extensions. The strict common function schema and every
  Chatbook-generated function tool are supported.
- No migration of DeepSeek, Groq, Mistral, OpenRouter, local servers, or custom
  OpenAI providers in this PR.
- No paid request in the default test suite.

## Provider-Parity Principle

| Boundary | Moonshot/Kimi | Z.ai/GLM |
| --- | --- | --- |
| Stable provider key | `moonshot` | `zai` |
| Display identity | Moonshot AI, with Kimi guidance | Z.ai, with GLM guidance |
| API route | `/chat/completions` | `/chat/completions` |
| Default model | `kimi-k3` | `glm-5.2` |
| Credential env | `MOONSHOT_API_KEY` | `ZAI_API_KEY` |
| Console/runtime | Existing shared path | Existing shared path |
| Function tools | Existing Chatbook tools | Existing Chatbook tools |
| Built-in tools | Excluded | Excluded |
| Model discovery | Existing ADR-020 path | Existing ADR-020 path |
| Responses mode | None | None |

No caller branches on either provider to execute tools or continue a run.
Provider-specific code owns only configuration translation, request rules,
finish/error classification, and hidden-reasoning policy.

## Architecture

```text
Settings / direct caller / Console
               |
               v
 canonical provider resolution (provider, model, key, base, retry policy)
               |
               v
 chat_with_moonshot() / chat_with_zai() compatibility wrappers
               |
       +-------+--------+
       |                |
       v                v
 Moonshot builder   Z.ai builder
       |                |
       +-------+--------+
               |
               v
 neutral hosted Chat-Completions transport + strict response/SSE normalization
               |
               v
 OpenAI-shaped choices / deltas / usage / allowlisted assistant metadata
               |
       +-------+--------+
       |                |
       v                v
 ordinary caller   Console gateway -> AgentService -> agent_runtime
```

The neutral wire layer owns:

- one request/session lifecycle;
- bounded retry attempts and safe `Retry-After` handling;
- response/session close ownership;
- strict incremental UTF-8 and SSE framing;
- generic OpenAI Chat choice, delta, tool-fragment, usage, and finish-shape
  validation;
- common size, depth, event-count, and metadata limits;
- typed status/network/malformed-response errors with redacted copies.

It does not import config, Settings, Console, provider adapters, model catalogs,
or tool execution code. It receives immutable resolved transport values, an
already-built payload, and narrow provider callbacks/policies for finish-state
classification and allowlisted assistant metadata.

Provider-specific pure builders own:

- model/default and model-family request policy;
- exact payload keys and value validation;
- function-tool and tool-choice rules;
- reasoning/thinking request rules;
- provider finish/error classification;
- conversion from canonical Chatbook history without input mutation.

Existing public `chat_with_moonshot()` and `chat_with_zai()` signatures remain
compatible. They resolve direct-call inputs, delegate to their provider builder
and the neutral transport, and return the same OpenAI-shaped dict/iterator
contract expected by `chat_api_call()`.

## QwenCloud Compatibility Extraction

QwenCloud Responses remains entirely provider-specific. Only its strict Chat
Completions SSE/choice primitives may move into the neutral module.

The extraction is accepted only if the complete QwenCloud adapter/stream suite
proves unchanged:

- normalized chunks and order;
- text/tool coexistence;
- nullable fields and finish states;
- malformed-event errors and redaction;
- usage projection;
- bounded state;
- cancellation and exactly-once closure.

A captured-event compatibility corpus feeds both the old expected behavior and
the extracted path. At least one deliberate mutation to framing, tool index, or
finish handling must fail the relevant Qwen test. If clean parity requires
provider flags that do not benefit Moonshot/Z.ai, the extraction is narrowed to
lower-level framing/validation rather than weakening QwenCloud.

## Configuration Contract

Canonical durable owners remain:

```toml
[providers]
Moonshot = ["kimi-k3", "kimi-latest", "moonshot-v1-auto"]
ZAI = ["glm-5.2", "glm-5.1", "glm-4.6", "glm-4.5"]

[api_settings.moonshot]
api_key_env_var = "MOONSHOT_API_KEY"
api_base_url = "https://api.moonshot.ai/v1"
model = "kimi-k3"
timeout = 90
retries = 3
retry_delay = 1.0
streaming = true

[api_settings.zai]
api_key_env_var = "ZAI_API_KEY"
api_base_url = "https://api.z.ai/api/paas/v4"
model = "glm-5.2"
timeout = 90
retries = 3
retry_delay = 5
streaming = true
```

The exact static list may retain every existing historical ID; the examples
above show the required new leading defaults, not a deletion list.

Resolution for direct calls and Console uses one contract:

1. Explicit trusted-call argument.
2. Exact canonical `api_settings.<provider>` field.
3. Credential environment variable named by canonical configuration, falling
   back to `MOONSHOT_API_KEY` or `ZAI_API_KEY`.
4. Embedded provider default for non-secret fields.

Configured credentials outrank environment credentials, matching current
readiness behavior. Every candidate key passes the shared placeholder/blank
validator. A malformed exact canonical provider table blocks before any
environment fallback is read. Settings and adapter errors name the canonical
table and never expose a candidate credential. The old handler's injected
`moonshot_api` runtime mapping has no durable config producer and is removed
rather than promoted into a new legacy-config contract.

`api_region` remains a compatibility fallback only when no canonical base URL
is present. `international` selects `https://api.moonshot.ai/v1`; `china`
selects `https://api.moonshot.cn/v1`. An explicit saved or call-supplied base
always wins.

Console freezes provider, model, base, key, timeout, retries, and retry delay in
its normal resolution for the whole send/agent run. A Settings mutation during
the run cannot switch a later model turn to another endpoint or credential.
Resolved secrets are not persisted in run state.

## Endpoint Contract

The shared hosted Chat-Completions URL helper is pure and structural, not a host
allowlist.

For a configured base:

1. Require a bounded HTTP(S) URL with a valid authority.
2. Reject userinfo, query, fragment, raw control/whitespace/backslash, malformed
   percent escapes, decoded separators/dot segments, empty/doubled path
   segments, and ambiguous stacked request endpoints.
3. Strip at most one exact terminal lowercase `/chat/completions` suffix.
4. Preserve the remaining host/path and safe percent-encoded ordinary prefix
   data.
5. Build chat as `{base}/chat/completions` and discovery as `{base}/models`.

`/responses`, case/lookalike request suffixes, repeated request tails, and
unsafe encoded structural tokens fail before network I/O. Arbitrary valid
regional and compatible proxy hosts remain supported.

The same helper is used by readiness, direct adapters, Console resolution, and
model discovery. Chat and discovery therefore cannot silently target different
tenants or path prefixes.

## Canonical Message Contract

Builders deep-copy validated inputs and never repair/mutate caller history.

- Supported roles are system, user, assistant, and tool.
- A separately supplied system message is prepended only when history has no
  system row; conflicting duplicate ownership fails locally.
- String content is preserved. Supported multimodal content is preserved only
  for documented model families; unknown/unsupported parts fail rather than
  being stringified or silently dropped.
- Assistant turns retain `content` and the one allowlisted hidden
  `reasoning_content` field in TASK-15675's validated continuation checkpoint
  when provider policy requires it. Tool turns also retain complete
  `tool_calls` and paired results.
- Tool rows require a nonblank `tool_call_id` and string content.
- Every completed assistant call ID must be unique and matched by exactly one
  following tool-result row before the next unrelated conversational turn.
- Tool results remain in canonical call-batch order by identity. Execution may
  be sequential, but pairing is always by `tool_call_id`.
- Orphaned, duplicate, missing, cross-batch, or malformed calls/results fail
  before I/O.

## Existing Function Tools

Only standard function tools are accepted:

```json
{
  "type": "function",
  "function": {
    "name": "calculator",
    "description": "Evaluate an arithmetic expression.",
    "parameters": {"type": "object", "properties": {}}
  }
}
```

Validation requires exact top-level `type`/`function`, a unique function name
matching `^[A-Za-z_][A-Za-z0-9_-]{2,63}$`, a nonblank string description, and
object-shaped parameters. This is the strict outbound intersection of the two
providers' function contracts. Private top-level metadata and provider tool
types are rejected. Inputs are copied before use.

The supported JSON Schema surface is the strict common subset emitted by
Chatbook's tool catalog. Every built-in/local/MCP tool shape made available to
these providers receives compatibility coverage. The adapter does not claim to
validate every MFJS feature or accept arbitrary Kimi/Z.ai schema extensions.

Moonshot tool choice supports absent, `auto`, `none`, `required`, and an exact
forced-function object, as documented. Z.ai supports absent or `auto` only.
Unsupported choices fail locally; neither provider silently rewrites them.

Provider membership policy:

- Moonshot remains native-tool eligible only if the strengthened joined suite
  continues to pass.
- Z.ai is added to `NATIVE_TOOLS_PROVIDERS` as the final production change of
  its slice, after the unpatched joined tests prove forwarding, normalization,
  continuation, partial-call cancellation, and closure.

## Moonshot/Kimi Request Policy

The public generic `max_tokens` argument remains compatible but maps to the
current `max_completion_tokens` wire key.

### Kimi K3

Allowed request keys for this task are `model`, `messages`, `stream`,
`max_completion_tokens`, `stop`, `response_format`, `tools`, `tool_choice`,
`reasoning_effort`, and adapter-owned `stream_options`.

- `reasoning_effort` accepts only `low`, `high`, or `max`.
- K3 always uses Preserved Thinking. Every retained historical K3 assistant
  message includes exact bounded `reasoning_content`; callers cannot disable
  that replay independently of the model contract.
- Generic/config-backed temperature, top-p, penalties, seed, `n`, user ID,
  prediction, prompt-cache key, safety identifier, and unknown kwargs are not
  transmitted.
- Valid-but-unsupported generic defaults are omitted; invalid supplied values
  still fail at the local public boundary.
- Streaming sets `stream_options={"include_usage": true}`.

### Historical `moonshot-v1-*`

The curated legacy family retains its documented sampling surface:
`temperature`, `top_p`, `n`, `presence_penalty`, and `frequency_penalty`, with
documented ranges and relationships. Current common output/stop/format/tool
fields remain available where documented.

### Other configured/discovered Kimi IDs

Explicit curated current IDs receive their documented policy. An unknown model
ID uses the conservative common subset—messages, stream, maximum completion,
stop, response format, function tools/tool choice, and usage request—without
injecting model-family sampler or thinking fields. Discovery never grants
capabilities merely from a substring.

Response format accepts only exact documented text, JSON object, or JSON schema
shapes for model families that support them. Nested schema validation remains
bounded and type-strict.

## Z.ai/GLM Request Policy

Allowed request keys for the general Chat Completion slice are `model`,
`messages`, `do_sample`, `stream`, `thinking`, `temperature`, `top_p`,
`reasoning_effort`, `max_tokens`, `tools`, `tool_choice`, `stop`,
`response_format`, `request_id`, and the existing generic user identifier
mapped to `user_id` when valid.

- General/tool-free chat retains provider-default thinking and
  `clear_thinking=true`. An active or explicitly restored run that exposes
  Chatbook function tools sends `thinking.clear_thinking=false` and replays
  only the bounded reasoning owned by that run's checkpoint.
- Default `glm-5.2` accepts `reasoning_effort` values `none`, `minimal`, `low`,
  `medium`, `high`, `xhigh`, and `max`. Other model families omit this field
  unless their explicit curated policy documents support.
- `tool_stream` stays absent/false unless a future explicit feature adds it.
- Function `tool_choice` accepts absent or `auto` only.
- Only function tools are admitted; retrieval/web-search schemas are rejected.
- `stop`, request IDs, user IDs, sampler values, and response format receive
  exact local type/range/shape checks.
- Unknown generic kwargs are omitted rather than forwarded wholesale.

## Response And Finish-State Contract

Streaming and non-streaming paths normalize to the same internal OpenAI-shaped
contract:

```json
{
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "...", "tool_calls": []},
    "finish_reason": "stop"
  }],
  "usage": {}
}
```

Text and tool calls may coexist. The adapter never drops either. Streaming
function arguments arrive as string fragments. A Z.ai non-streaming response
may contain either a bounded JSON object or JSON string; an object is encoded
deterministically to a JSON string before entering the existing native
accumulator/parser. Other scalar/container shapes fail locally.

Moonshot valid finishes are `stop`, `tool_calls`, and `length`. Z.ai valid
usable finishes include `stop`, `tool_calls`, and `length`; `sensitive`,
`model_context_window_exceeded`, and `network_error` are typed terminal provider
errors with safe recovery copy, not successful empty responses. Blank, unknown,
contradictory, or tool-inconsistent finish states are malformed-provider errors.

A successful HTTP response with neither usable content nor complete calls is
malformed. Incomplete call identity/name/arguments, invalid JSON arguments,
conflicting terminal values, or call/finish disagreement fails before the
executor sees a call.

## Strict Streaming Contract

The closeable stream owns its `requests.Session` and `Response` until normal
completion, failure, cancellation, or explicit close. Close is idempotent and
best-effort; cleanup exceptions never mask the primary result/error.

Framing rules:

- strict incremental UTF-8;
- CR, LF, and CRLF support;
- dispatch only on a blank SSE record boundary;
- ignore comments and non-`data` fields;
- combine multiple `data:` fields per SSE rules;
- accept `[DONE]` only after a valid terminal provider state;
- reject truncated records, malformed JSON, deep/oversized structures,
  contradictory indexes/IDs, invalid nullable fields, post-terminal unseen
  events, and incomplete calls.

The parser uses list/segment accumulation and bounded digests rather than
quadratic string concatenation or retaining every event. Limits cover bytes,
lines, records, events, choices, tools, IDs/names, argument/text output, JSON
depth/nodes, and hidden reasoning. Provider-controlled state is released when
an item/choice finishes.

Moonshot requests terminal usage through `stream_options.include_usage`; Z.ai
accepts usage from its documented final choice chunk. A terminal usage-only
chunk produces no visible fallback text.

Streaming retries are allowed only before any response-body byte is consumed.
Once body consumption begins, malformed data, timeout, body-read failure, or
cancellation closes the stream and returns one typed error/cancellation without
replay. Partial tool calls are never emitted to execution.

## Non-Streaming Contract

Non-streaming JSON is read and validated inside the same owned response/session
lifecycle. Connection/timeout failures before a response and explicit
retryable status responses before their body is read use one global attempt
budget and close each failed response before the next attempt. Once any 2xx
response is received, its body-read/content-decoding/JSON/shape failure closes
and fails without replay, because the paid POST may already have completed.

No raw `requests`, urllib3, JSON, recursion, or cleanup exception escapes the
provider boundary. Error causes/contexts containing response or credential data
are not chained into user-visible errors.

## Retry And Error Policy

- `retries` means additional attempts after the first and is clamped to a
  nonnegative integer.
- `llm_retry_count()` remains the shared sensitive-request policy and can force
  retries to zero.
- Retryable POST statuses are 429, 500, 502, 503, and 504. Connection and
  timeout/read failures use the same overall budget under the lifecycle rules
  above.
- Other 4xx statuses are attempted once.
- Integer and HTTP-date `Retry-After` values are honored; malformed values fall
  back to exponential delay without exposing their contents.
- 401/403 map to authentication, 429 to rate limit, other 4xx to bad request,
  and exhausted transient/network failures to provider error.
- Error text uses the stable provider key and bounded generic/status/code data.
  Credentials, full URLs, response bodies, prompts, reasoning, tools/results,
  and request payloads are excluded from logs and exceptions.

## Usage Contract

Provider raw usage remains available to the existing Console usage signal, but
budget handoff trusts only strict nonnegative integers.

- Moonshot maps `prompt_tokens`, `completion_tokens`, `total_tokens`, and
  documented cached-token details.
- Z.ai maps the same top-level counts and
  `prompt_tokens_details.cached_tokens`.
- An exact positive `total_tokens` is authoritative when consistent; otherwise
  valid prompt/completion counts may produce the total under existing policy.
- Boolean, float, string, negative, malformed nested detail, or inconsistent
  counts are not coerced. The raw signal may retain them for diagnostics, but
  the agent budget uses its existing estimator.
- Absent terminal usage after cancellation/failure is “unknown,” never a
  fabricated zero.

## Durable Provider Reasoning Continuation

`reasoning_content` is a fixed allowlisted field inside TASK-15675's canonical
assistant-owned checkpoint, not an open metadata bag and not visible transcript
content.

- It must be a string and is charged to the shared output, checkpoint, sync,
  export, and context-budget limits.
- Streaming fragments are accumulated invisibly; non-streaming values pass the
  same validator.
- Before any complete function-call batch executes, the runtime force-persists
  the stable assistant generation plus its checkpoint and call states.
- Each call transitions `pending -> executing -> completed|failed` around the
  existing approval/execution seams. A restored `executing` call is ambiguous
  and is never automatically re-run.
- The next request expands exactly the reasoning/calls/results required by the
  active model policy.
- Restore is explicit, re-runs current approvals, uses the pinned
  provider/model/base, and resolves the credential at resume time.
- Each assistant variant owns its checkpoint; overlapping runs cannot see or
  overwrite one another.
- Tool-free reasoning is checkpointed only for Kimi K3, atomically with its
  final visible assistant row. Completed data may remain for private JSON
  export and provider replay; non-K3 Kimi/GLM ordinary later turns omit it.
- It never reaches visible chunks, run-log content, usage snapshots, error
  copies, human-readable exports, or persistent logs.

Kimi K3 replays every retained historical assistant reasoning value and counts
the private bytes atomically with the owning visible turn. Other curated Kimi
models send the complete assistant message only under their documented active/
restored preserved-thinking policy. Z.ai sends `clear_thinking=false` only for
that active or restored run and replays exact ordered `reasoning_content`;
ordinary/tool-free chat sends `clear_thinking=true`. QwenCloud remains governed
by ADR-045 and does not gain preserved-thinking behavior through the shared
wire parser.

## Native Continuation And Cancellation

There is no Moonshot- or Z.ai-specific continuation loop:

1. The normalized assistant tool-call message is returned through the ordinary
   gateway/bridge adapter.
2. `AgentService` parses complete calls and applies its native-provider rules.
3. The assistant generation and complete call batch are durably checkpointed
   before execution.
4. Existing approval/review policy runs before dispatch; the selected call is
   marked `executing` before the external side effect.
5. Existing tools execute through the shared registry; the exact result or
   structured failure is durably marked before continuation.
6. The next adapter request receives canonical visible history plus the
   provider-approved private expansion.

Cancellation before response retention, before iteration, during visible text,
during hidden reasoning, or during an incomplete call closes the underlying
iterator/response/session exactly once and executes no incomplete call. A
cleanup failure remains secondary to cancellation or provider failure.

## Readiness And Settings UX

The canonical F9 Settings Providers & Models surface remains the only Settings
owner. No API-mode selector is added.

- Provider identities remain Moonshot and Z.ai; Kimi/GLM appear in descriptive
  labels/help only, not as new provider keys.
- Fresh/missing model defaults show `kimi-k3` and `glm-5.2`.
- Existing explicit old models remain selected and editable.
- Static lists add current models without deleting historical values; runtime
  discovery remains authoritative under ADR-020.
- Credential, endpoint, model, sampler, streaming, and retry drafts remain
  provider-scoped and save atomically through existing Settings commit models.
- Malformed provider tables, placeholder keys, blank models, invalid endpoints,
  unsupported reasoning/tool choices, and invalid numeric fields visibly block
  Test/Save/send with provider-specific recovery.
- Field search can focus the relevant provider model, credential, endpoint, and
  reasoning controls.
- Moonshot help describes international/China/custom bases, K3 always-on
  Preserved Thinking, its private storage/export warning, and context cost.
- Z.ai help distinguishes the general API from the coding-only endpoint and
  explains durable but private reasoning preservation for active/restored
  function-tool runs.
- Built-in provider tools are not shown as available Chatbook tools.

The existing generic reasoning-profile control is used for Kimi K3 and accepts
only `low`, `high`, or `max` there. For default `glm-5.2` it accepts the exact
documented `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, and `max`
values. No separate Z.ai thinking selector is added in this task.

## Model Discovery

Both providers remain in the ADR-020 auto-refresh inventory:

- Moonshot uses authenticated `GET {normalized_base}/models`;
- Z.ai discovery remains best-effort because the current published OpenAPI
  does not advertise `/paas/v4/models`; unsupported/failing refresh falls back
  to configured and cached IDs without affecting chat readiness;
- the same resolved canonical endpoint and credential contract as chat;
- model IDs and timestamps only in the disk TTL cache;
- capped selectors plus the uncapped searchable picker;
- configured/cached fallback on failure;
- optional append-only write-through under existing policy.

A failed refresh preserves prior runtime/disk models and emits bounded provider
status without response bodies, URLs containing sensitive components, or keys.
Discovery does not change the active/saved model and does not infer capabilities
from a newly discovered name. The optional Z.ai live gate includes a bounded
model-endpoint probe whose failure is reported as discovery unavailable, not a
generation failure.

## Testing Strategy

Implementation follows strict test-driven development: every behavioral change
starts with a failing focused test, then the smallest production change, then
the focused and surrounding regressions.

### Pure/config/provider tests

- stable identities, dispatcher and parameter-map contracts;
- default model owners and preservation of explicit historical selections;
- exact credential/base/model precedence and source immutability;
- malformed canonical tables and invalid environment fallbacks;
- endpoint normalization parity across readiness/chat/discovery;
- provider/model-family allowlists, scalar/range/schema validation, and input
  copying;
- Kimi K3 tool-free reasoning persists with final content, replays on later K3
  turns, and is evicted atomically with its visible owner;
- function tools and exact tool-choice subsets;
- common function-name boundaries, including rejected leading digits/hyphens
  and one-/two-character names;
- no vendor built-in tool leakage;
- safe finish/error/usage classification.

### Shared wire tests

- exact URL, headers, timeout, retries, Retry-After, and sensitive zero-retry;
- streaming/non-streaming text, tool-only, and mixed results;
- fragmented/interleaved multiple calls with stable indexes and IDs;
- strict UTF-8/SSE record framing and terminal `[DONE]` rules;
- malformed/deep/oversized events and bounded/linear state;
- invalid JSON/content encoding/body reads and typed redaction;
- no stream retry after any body byte;
- no non-streaming replay after a 2xx response has been received;
- normal, error, cancellation, explicit/repeated close, and cleanup-failure
  lifecycle with exact once-only ownership;
- strict usage and estimator fallback;
- concurrent stream/metadata isolation.

### QwenCloud compatibility gate

- complete QwenCloud adapter + streaming suite before and after extraction;
- shared event corpus for nullable fields, choices, tools, finishes, usage,
  malformed events, limits, and closure;
- at least one relevant mutation must fail and be restored green.

### Joined application tests

For each provider, tests traverse real production seams:

```text
ConsoleAgentBridge -> AgentService/agent_runtime -> _StreamingModelAdapter
-> ConsoleProviderGateway -> chat_api_call -> provider wrapper
-> neutral transport -> temporary loopback HTTP server
```

They prove:

- one text turn and a second tool-influenced final answer;
- complete multi-call assistant batches and exact paired result history;
- no synthetic user continuation;
- provider-specific request fields and stable IDs;
- terminal usage reaches Console signals and agent budget accounting;
- partial-call cancellation after downstream parser observation executes zero
  tools and closes the live response exactly once;
- Kimi K3 later turns and Kimi/Z.ai active/restored tool continuations receive
  exact policy-required reasoning, the first complete call batch is persisted
  before execution, ambiguous restored `executing` calls never auto-run, and
  the data stays absent from visible/log/error/usage/human-export surfaces;
- a K3 tool call followed by its final answer stores that final reasoning-only
  round on the same assistant owner and replays it on the next K3 turn without
  creating a duplicate visible assistant message;
- repeated calls to the same function name remain valid while duplicate call
  IDs across an outbound history fail before I/O;
- Z.ai works before and after its final native-provider registry entry.

Local loopback fixtures prove application integration, not vendor availability.

### Settings/catalog tests

- real Pilot render, provider switching, invalid recovery, search/focus, save,
  revert, atomic failure, and second-save no-op behavior;
- fresh defaults versus explicit saved old models;
- shared `/models` URL/credential parity;
- cache fallback, cap/search, write-through privacy, and no unrelated provider
  drift.

### Optional paid live tests

Default collection makes no provider request. Live tests require both gates:

- Moonshot: `TLDW_LIVE_MOONSHOT=1` and nonblank `MOONSHOT_API_KEY`.
- Z.ai: `TLDW_LIVE_ZAI=1` and nonblank `ZAI_API_KEY`.

Each mode runs in a fresh subprocess with isolated `HOME`, XDG config/data,
`TLDW_CONFIG_PATH`, and `[paths].data_dir` set before Chatbook imports. The
harness removes log sinks, discards child stdout/stderr, uses randomized text
and arithmetic values, and proves exactly one real Calculator result influences
the final response. Secrets, prompts, responses, and tool results do not appear
in assertion/error text. Optional model overrides use provider-specific env
variables. Live tests remain skipped in ordinary CI.

## Documentation And Delivery

README, Settings guide, and Console guide document:

- stable provider names and credentials;
- current defaults and historical-model preservation;
- general/regional/custom endpoints;
- Chat-Completions-only scope;
- exact supported parameter/reasoning/tool-choice subsets;
- streaming/non-streaming usage behavior;
- existing Chatbook function tools and built-in-tool exclusion;
- durable private Kimi K3 later-turn reasoning, active/restored Kimi/Z.ai tool
  reasoning, and ordinary GLM clearing behavior;
- model discovery/cache and unknown pricing behavior;
- invalid config/endpoint recovery;
- optional isolated live gates.

TASK-15675 lands the database/sync/export foundation first. This task adds no
second provider-history store or schema. Configuration remains TOML-backed.

Delivery is one PR with review gates after:

1. neutral framing/transport and Qwen compatibility extraction;
2. Moonshot/Kimi migration;
3. Z.ai/GLM migration and native-tool eligibility;
4. Console metadata/usage integration;
5. defaults, Settings, catalog, docs, and live harness;
6. final full-suite/security/backlog review.

If the Qwen compatibility gate cannot stay clean without speculative provider
flags, the shared extraction is narrowed before later slices proceed.

## ADR Check

ADR required: yes

ADR path:
[backlog/decisions/063-hosted-provider-wire-and-durable-tool-continuation.md](../../../backlog/decisions/063-hosted-provider-wire-and-durable-tool-continuation.md)

Reason: this task implements a reusable hosted-provider transport/service
contract and consumes the privacy-sensitive durable assistant checkpoint across
provider, gateway, persistence, sync, export, and agent-runtime layers. ADR-063
records those boundaries and supersedes ADR-062.

## Acceptance Summary

The task is complete when Moonshot/Kimi and Z.ai/GLM behave like ordinary
first-class Chatbook providers through one strict hosted Chat-Completions wire
boundary: current defaults, canonical configuration, safe streaming and usage,
existing function tools, exact continuation/cancellation, actionable Settings,
shared discovery, and isolated optional live proof—without a speculative
Responses mode, vendor built-in tools, a second provider-history store, or
behavior changes to unrelated providers.
