# DeepSeek Dual-API Provider Design

Date: 2026-08-12
Status: Approved; written-spec review complete
Backlog task: [TASK-15677](../../../backlog/tasks/task-15677%20-%20Add-DeepSeek-dual-API-provider-support.md)
Foundation task: [TASK-15675](../../../backlog/tasks/task-15675%20-%20Add-durable-provider-tool-continuation-checkpoints.md)
Hosted Chat dependency: [TASK-15676](../../../backlog/tasks/task-15676%20-%20Harden-Moonshot-Kimi-and-Z.ai-GLM-as-first-class-hosted-providers.md)
Architecture decisions: [ADR-063](../../../backlog/decisions/063-hosted-provider-wire-and-durable-tool-continuation.md), [ADR-064](../../../backlog/decisions/064-deepseek-dual-api-provider-boundary.md)

## Purpose

Treat DeepSeek Chat Completions and the new Responses API as two strict wire
modes of the existing `deepseek` provider. Existing users keep Chat
Completions by default; users may explicitly select Responses without gaining a
second provider identity, credential owner, model catalog, agent loop, or tool
executor.

Both modes must behave like ordinary first-class Chatbook APIs: canonical
configuration, exact request allowlists, streaming and non-streaming parity,
native function tools, durable restart/sync/import continuation, terminal
usage, cancellation, Settings, readiness, and model discovery.

## Official Source Findings

- The production base is `https://api.deepseek.com`. The current model IDs are
  `deepseek-v4-flash` and `deepseek-v4-pro`; fresh/missing configuration uses
  `deepseek-v4-flash`.
- Chat Completions uses `POST /chat/completions` and ends streaming with
  `data: [DONE]`. `stream_options.include_usage=true` requests an empty-choice
  terminal usage chunk before `[DONE]`.
- Responses uses `POST /responses`. It is stateless: `previous_response_id` and
  `conversation` are unsupported, `store` is always false, and Chatbook must
  send explicit history.
- Responses streaming is semantic SSE. Each record names the event with the
  SSE `event:` field and JSON `type`, and its JSON contains an incrementing
  `sequence_number`; a terminal
  `response.completed`, `response.incomplete`, or `response.failed` replaces
  `[DONE]`.
- Both modes support function tools. Responses also supports server-side web
  search and the `apply_patch` custom tool, but those are outside this feature.
- Thinking defaults enabled. The distinct documented effort values are `low`,
  `high`, and `max`; compatibility values map onto them. Chatbook exposes
  provider default, `low`, `high`, and `max`, and rejects aliases that add no
  behavior.
- Thinking requests ignore or reject sampler controls. DeepSeek V4 thinking
  mode rejects `tool_choice`; tool-enabled thinking requests must omit it.
- For any assistant turn that performed a tool call, complete
  `reasoning_content`, assistant content, calls, and paired results must be
  passed in all later requests. Omitting that history can cause HTTP 400.
- Chat finish reasons are `stop`, `length`, `content_filter`, `tool_calls`, and
  `insufficient_system_resource`.
- Responses usage exposes `input_tokens`, cached input details,
  `output_tokens`, and reasoning output details. Chat usage exposes
  prompt/completion totals, prompt cache hit/miss, and reasoning details.

Primary references:

- [Using the Responses API](https://api-docs.deepseek.com/guides/responses_api/)
- [Create Response](https://api-docs.deepseek.com/api/create-response/)
- [Create Chat Completion](https://api-docs.deepseek.com/api/create-chat-completion)
- [Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode)
- [Tool Calls](https://api-docs.deepseek.com/guides/tool_calls)
- [Models and Pricing](https://api-docs.deepseek.com/quick_start/pricing)

## Existing State And Problems

Chatbook already has a `deepseek` provider, Settings identity, dispatcher path,
configuration, and model discovery. The current handler is an independent
Chat-Completions implementation in `LLM_API_Calls.py` with no API-mode field or
Responses translator. Its surrounding problems match the older hosted
providers: duplicated request/transport logic, loose parameter forwarding,
inconsistent stream ownership, no durable private reasoning history, and no
joined proof that restart-safe native tools preserve DeepSeek's later-turn
reasoning contract.

Routing Responses through OpenAI or QwenCloud is not correct. DeepSeek has its
own request omissions, thinking controls, event set, sequence rules, finish
states, and persistent tool-reasoning requirement.

## Goals

- Preserve stable provider key `deepseek`, labels, histories, metrics, and
  configuration ownership.
- Add explicit `api_mode` values `chat_completions` and `responses`, defaulting
  to `chat_completions` for missing settings.
- Default fresh/missing models to `deepseek-v4-flash` while preserving every
  explicit historical model.
- Support strict streaming and non-streaming text, reasoning, function calls,
  errors, usage, cancellation, retry, and resource ownership in both modes.
- Use only existing Chatbook function tools and the existing approval/execution
  path.
- Replay required private tool reasoning durably across restart, sync, import,
  and later same-provider turns under TASK-15675.
- Make Settings, readiness, Console pinning, direct calls, and model discovery
  resolve the same canonical credential/base/model/mode.
- Leave unrelated providers byte-for-byte unchanged.

## Non-Goals

- No second `deepseek-responses` provider identity.
- No provider SDK dependency.
- No DeepSeek web search, custom `apply_patch`, file search, code interpreter,
  MCP, computer use, or other provider-hosted tools.
- No provider conversation IDs, `previous_response_id`, server-side state, or
  response storage.
- No background responses, metadata, prompt templates, truncation, service
  tier, safety identifier, or prompt-cache control.
- No new agent loop, executor, approval system, transcript store, or durable
  schema beyond TASK-15675.
- No exactly-once claim for arbitrary tool side effects or cross-device
  takeover.
- No paid request in the default suite.

## Provider Identity And Mode

| Boundary | Contract |
| --- | --- |
| Stable key | `deepseek` |
| Display label | DeepSeek |
| Default model | `deepseek-v4-flash` |
| Credential | `DEEPSEEK_API_KEY` |
| Default base | `https://api.deepseek.com` |
| Default mode | `chat_completions` |
| Selectable mode | `responses` |
| Tools | Existing Chatbook function tools only |
| Provider built-ins | Excluded |

`api_mode` accepts only the two exact strings. An absent field resolves to
`chat_completions`; a present empty, non-string, or unknown value is invalid and
blocks readiness/save/send. Explicit call arguments override canonical config;
canonical config overrides the default. Lower-priority malformed sources are
not inspected after a valid higher-priority value wins.

Console freezes provider, model, normalized base, mode, credential, timeout,
and retry policy into its resolution snapshot. Later Settings mutation cannot
change a running turn or auxiliary/subagent call. Only DeepSeek receives the
mode argument; other providers' dispatcher kwargs remain unchanged.

## Architecture

```text
Settings / direct caller / Console
                |
                v
 canonical DeepSeek resolution + frozen api_mode
                |
                v
        DeepSeek provider adapter
          /                    \
 hosted Chat request/stream   Responses request/semantic SSE
          \                    /
           normalized ModelTurn + usage
                        |
                        v
        existing AgentService / native tools
                        |
                        v
        TASK-15675 durable continuation owner
```

The gateway carries the frozen mode and base but does not build provider wire
payloads. The adapter owns two pure builders and two response translators.
Chat mode consumes TASK-15676's neutral hosted Chat transport/parser; Responses
has a DeepSeek-specific translator and does not call the QwenCloud adapter.

## Configuration And Endpoint Contract

The canonical fresh configuration is conceptually:

```toml
[providers]
DeepSeek = ["deepseek-v4-flash", "deepseek-v4-pro"]

[api_settings.deepseek]
api_key_env_var = "DEEPSEEK_API_KEY"
api_base_url = "https://api.deepseek.com"
model = "deepseek-v4-flash"
api_mode = "chat_completions"
```

Credential precedence follows the repository contract: explicit usable key,
configured usable key, configured env-var name, default
`DEEPSEEK_API_KEY`. Placeholder, blank, malformed, and non-string values fail
closed without reading an unnecessary lower-priority source or exposing the
value.

One pure URL normalizer is shared by readiness, direct calls, Console, and
discovery. It accepts a safe absolute HTTP(S) base and strips at most one exact
terminal `/chat/completions` or `/responses` (with an optional trailing slash).
It rejects credentials, query, fragment, whitespace/control characters,
malformed authorities, unsafe percent-encoded structure, dot segments,
duplicate slashes, stacked/repeated request suffixes, or an empty host. The
adapter appends exactly one mode route; discovery appends `/models` to the same
normalized base. Inputs are never mutated.

## Canonical History And Durable Replay

Visible history supports system, user, assistant, and tool roles. Builders
deep-copy and strictly validate every row. A separately supplied system prompt
is prepended only when history has no system owner. Conflicting or malformed
system ownership fails locally.

Assistant tool turns require bounded string content (including the required
empty-string form), exact reasoning, complete function calls, and unique call
IDs. Every call has one paired result before an unrelated conversational turn.
Orphaned, duplicate, missing, cross-batch, or malformed calls/results fail
before I/O.

TASK-15675 owns private durable history. On the first complete call batch, the
runtime force-persists the stable assistant generation and checkpoint before
execution. Calls transition `pending -> executing -> completed|failed`; a
restored `executing` call is ambiguous and never auto-runs. Resume/takeover is
explicit, reruns approvals, uses the pinned provider/model/mode/base, and
resolves the current credential.

DeepSeek's replay policy is intentionally stricter than GLM and non-K3 Kimi
families. Every
retained tool-bearing round—assistant content, exact reasoning, call IDs/names/
arguments, and exact paired results—is expanded into every later DeepSeek
request while its owning visible turn remains in context. Tool-free private
reasoning is not persisted or replayed. A provider/model/mode/base switch
cannot silently translate an active checkpoint.

Private expansion counts against the request context budget. The visible owner
turn and all its private rounds are one eviction unit; the budgeter never keeps
the visible final answer while silently dropping required DeepSeek history.

## Existing Function Tools

Only exact function tools are accepted:

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

Top-level keys are exact. Names match the shared provider-safe function-name
contract and are unique. Descriptions are nonblank strings. Parameters are a
bounded object JSON Schema using the strict common subset emitted by Chatbook's
tool catalog. Private metadata and all non-function types fail locally.

In thinking mode DeepSeek V4 rejects `tool_choice`, so tool-enabled requests
omit it. Chatbook accepts only an absent/default or `auto` generic choice for
these requests and rejects `none`, `required`, and forced objects rather than
pretending they were honored. The Responses builder likewise intentionally
uses only the automatic function-tool path; provider web/custom choices never
enter the payload.

Parallel calls remain supported through the existing runtime and are paired by
call ID. Chatbook does not send `parallel_tool_calls` or `max_tool_calls` because
DeepSeek ignores them.

## Thinking And Generation Policy

Settings exposes four values:

- `provider_default`: omit `reasoning_effort`; thinking remains provider-
  default enabled;
- `low`: send enabled thinking with effort `low`;
- `high`: send enabled thinking with effort `high`;
- `max`: send enabled thinking with effort `max`.

Compatibility values `minimal`, `medium`, and `xhigh` are rejected locally
because they map to an already exposed value. There is no separate thinking-
disable setting in this feature.

Thinking-mode payloads omit `temperature`, `top_p`, `presence_penalty`, and
`frequency_penalty`. Valid generic settings are not transmitted; invalid
explicit scalar values still fail at the public boundary. Inputs are immutable.

Response format accepts only exact text or JSON-object forms currently shared
with Chatbook. JSON mode requires an explicit system/user instruction under the
existing UX guidance. Unsupported schema/format shapes fail locally rather
than passing through.

## Chat-Completions Request Policy

The exact feature allowlist is `model`, `messages`, `stream`, `max_tokens`,
`stop`, `response_format`, `tools`, `thinking`, `reasoning_effort`, and
adapter-owned `stream_options`.

- `model` is nonblank; booleans are not integers; numeric values are finite and
  receive documented bounds.
- `stop` is a copied string or at most 16 copied strings.
- `stream_options={"include_usage": true}` is sent only for streaming.
- `logprobs`, `top_logprobs`, `user_id`, prefix/FIM beta fields, strict-tool
  beta flags, deprecated penalties, and unknown kwargs are omitted.
- Assistant tool messages always carry string `content`, exact
  `reasoning_content`, and complete `tool_calls`.

The request route is `{normalized_base}/chat/completions`.

## Responses Request Policy

The exact feature allowlist is `model`, `input`, `instructions`, `stream`,
`max_output_tokens`, `tools`, `reasoning`, and `text`.

- Visible system ownership becomes `instructions`; remaining conversation and
  checkpoint expansion become an explicit `input` item list.
- Function-call items are immediately followed by their matching
  `function_call_output` item in canonical call order. Result input order may
  differ internally; pairing is always by call ID before serialization.
- Reasoning items use only supported plain text and are associated with the
  adjacent assistant/function-call round. Summary and encrypted content are
  never sent.
- Tools use flat Responses function shapes derived from the validated common
  schema.
- `reasoning` is omitted for provider default or contains exact effort
  `low`/`high`/`max`. `summary` is omitted.
- `text` contains only the supported format projection. Verbosity is omitted.
- `temperature`, `top_p`, `top_logprobs`, and `user` are intentionally omitted
  from this first thinking-enabled slice.
- `tool_choice`, `parallel_tool_calls`, `max_tool_calls`,
  `previous_response_id`, `conversation`, `store`, `background`, `metadata`,
  `include`, `prompt`, `truncation`, `service_tier`, `safety_identifier`,
  prompt-cache controls, `context_management`, and `stream_options` are absent.

The request route is `{normalized_base}/responses`. Unsupported provider fields
are omitted locally even though the provider says many are silently ignored.
The source history and tools remain unchanged.

## Response And Finish Contract

Both modes normalize into the existing internal model-turn shape containing
visible text, complete function calls, private reasoning for checkpointing,
finish reason, and raw usage.

Chat accepts one primary choice and exact finish states:

- `stop` -> `stop`;
- `length` -> `length`;
- `tool_calls` -> `tool_calls` only with one or more complete calls;
- `content_filter` -> typed provider refusal/error, never successful empty text;
- `insufficient_system_resource` -> typed transient provider error, never
  successful truncation.

Contradictory finish/tool states, arbitrary reasons, missing terminal state,
partial arguments, or malformed assistant fields fail closed.

Responses accepts only output items needed here: `reasoning`, `message` with
text parts, and `function_call`. Web-search/custom items are typed unsupported-
tool failures, not ignored. Every function call needs `call_id`; transport item
`id` is never substituted. Terminal `completed` maps text/tools normally;
`incomplete` with `max_output_tokens` maps to `length`; other incomplete,
failed, cancelled, malformed, or missing terminals become typed errors.

## Streaming Contract

### Shared SSE framing

Streaming uses strict incremental UTF-8 and bounded SSE records. It tolerates
LF, CRLF, split code points, comments/keepalive lines, and empty records. It
rejects invalid UTF-8, malformed JSON, oversized lines/records/events,
excessive nesting/nodes/output, incomplete data at EOF, or provider-controlled
unbounded state. No raw provider exception/body is exposed.

### Chat stream

Data-only Chat chunks preserve nullable `content`, `reasoning_content`,
`tool_calls`, and `usage` as absent. Choice/tool indexes, call identity/name,
argument fragment types, terminal finish, and complete call state are strict.
The stream must reach one valid terminal choice and then the usage chunk when
requested before `[DONE]`. `[DONE]` before completion, data after terminal, or
EOF without completion is an error.

### Responses stream

Each SSE record has an exact nonblank `event:` label; the decoded JSON has an
exact string `type` and strict nonnegative integer `sequence_number`. The label
and JSON type must match. A JSON `event` key is not required or used as the
discriminator. Sequence numbers strictly increase; duplicates, decreases, or
unseen post-terminal events fail. Increasing values may skip numbers because
the official examples are not contiguous.

Accepted events are:

- `response.created`, `response.in_progress`;
- `response.output_item.added`, `response.output_item.done` for reasoning,
  message, and function-call items;
- `response.content_part.added`, `response.content_part.done` for text;
- `response.reasoning_text.delta`, `response.reasoning_text.done`;
- `response.output_text.delta`, `response.output_text.done`;
- `response.function_call_arguments.delta`,
  `response.function_call_arguments.done`;
- `response.completed`, `response.incomplete`, `response.failed`.

Delta/done/full terminal representations are de-duplicated exactly once.
Output/call IDs, indexes, statuses, item types, call IDs/names, and done/full
values must agree. A terminal event owns full usage. `[DONE]` is invalid in this
mode. Provider web-search/custom-tool events are rejected before execution.

## Non-Streaming Contract

HTTP 2xx is not success until strict JSON parsing, response validation, output
normalization, finish-state validation, and usage extraction complete. Empty,
malformed, contradictory, deep, or oversized bodies produce typed redacted
errors. Once a 2xx response is received, malformed content is not retried.

## Transport, Retry, And Lifecycle

One shared requests transport owns the session/response lifecycle. It sends
Bearer authorization, JSON content type, the frozen timeout, and no secrets in
URLs. Streaming transfers ownership to an idempotently closeable iterator;
non-streaming closes response/session before returning. Cleanup failure never
masks the primary result/error/cancellation.

`llm_retry_count` means additional attempts. Only POST connection failures,
timeouts/read failures before body handoff, and HTTP
429/500/502/503/504 are retried. Other 4xx, successful-but-malformed bodies,
and any stream after the first response byte are not replayed. Retry-After
integer/date is bounded; invalid values fall back to bounded exponential delay.
Sensitive-request policy forces zero retries. Every failed response closes
before another attempt.

Cancellation before response retention, before iteration, during hidden
reasoning, visible text, or an incomplete call closes the live resource exactly
once and executes no incomplete tool. Cancellation races do not consume or feed
a late item after sealing.

## Errors, Usage, And Privacy

HTTP 401/403 map to authentication, 429 to rate limit, supported transient
statuses/network failures to provider errors, and invalid config/request/
response/stream/checkpoint data to provider-labelled typed errors. Public
errors contain safe provider/mode/status/recovery context only; credentials,
userinfo, URLs with query/fragment, prompts, reasoning, calls/results, raw
bodies, and canaries never enter logs or exception cause/context.

Raw usage reaches Console signals, but agent budgets trust only exact
nonnegative integers and structurally valid detail mappings.

- Chat maps prompt/completion/total, prompt cache hit/miss, and completion
  reasoning tokens.
- Responses maps input/output/total, cached input tokens, and reasoning output
  tokens.
- Inconsistent or malformed usage remains available only in the redacted raw
  diagnostic signal; budgeting uses the existing estimator.
- Cancellation/failure without terminal usage is unknown, not fabricated zero.

Checkpoint privacy is governed by TASK-15675: it is synced and present only in
versioned `.chatbook` or explicit active-path JSON private fields with warnings,
while text/Markdown,
rendering, FTS, clipboard-visible message copy, summaries, logs, errors, usage,
and telemetry exclude it.

## Settings, Readiness, And Discovery

F9 Settings Providers & Models remains the only Settings owner.

- DeepSeek shows an API Mode selector with exact options Chat Completions and
  Responses; missing config displays/saves Chat Completions.
- Present invalid values render no valid selection, block Test/Save/send, and
  can be explicitly repaired.
- Mode, endpoint, credential, model, retry, and reasoning drafts are provider-
  scoped, switch-safe, revertible, and atomically saved.
- Field search focuses API Mode by label, owned config key, or “mode”.
- Help explains current models, exact endpoint, both modes, default Chat mode,
  thinking values, sampler omissions, stateless Responses, durable private
  tool reasoning, existing tools, and excluded provider built-ins.
- A nonterminal checkpoint pins provider/model/mode/base; Settings offers
  restore/discard rather than silently changing it.

Readiness and direct/Console sends share the same fail-closed config resolver.
Malformed canonical provider tables, placeholder keys, invalid mode/base/model,
and incomplete restore requirements produce safe actionable recovery before
network I/O.

DeepSeek remains one ADR-020 discovery provider. Authenticated
`GET {normalized_base}/models` uses the same base/key contract; mode does not
change discovery identity. Cache contains only IDs/timestamps, preserves prior
data on failure, caps normal selectors, keeps the full searchable picker, and
never changes the saved active model or infers capability from an ID substring.

## Testing Strategy

Implementation follows strict test-driven development.

### Pure/config/request tests

- stable identity, fresh default, explicit historical model preservation;
- exact mode precedence/default/invalid values and Qwen/unrelated-provider
  isolation;
- credential/base/model/retry precedence, malformed tables, immutability, and
  secret-free errors;
- endpoint parity across readiness/direct/Console/discovery;
- exact Chat and Responses allowlists and every forbidden-field omission;
- thinking value matrix and sampler/tool-choice omission;
- exact function schema/name/choice matrices and built-in-tool rejection;
- history conversion, reasoning replay across later turns, atomic budget
  eviction, and provider/mode switch blocking.

### Response/transport tests

- streaming/non-streaming text, reasoning, tool-only, mixed, and usage;
- fragmented/interleaved parallel calls with stable identities;
- Chat nullable chunks, terminal usage, keepalive, `[DONE]`, finish matrix;
- Responses event allowlist, sequence/replay/conflict, item status, done/full
  recovery, call/output pairing, terminal matrix, and no `[DONE]`;
- explicit rejection of web/custom tool events;
- malformed/deep/oversized JSON/SSE/UTF-8 and bounded linear state;
- retry attempt cap, Retry-After, response close, no retry after body byte or
  2xx parse failure;
- cancellation and explicit/error/exhaustion close exactly once, including
  pre-retention and blocked-feed races;
- strict usage versus deterministic estimator fallback;
- privacy canaries absent from logs/errors/exception chains.

### Durable joined tests

Both modes traverse the real path:

```text
ConsoleAgentBridge -> AgentService/agent_runtime -> _StreamingModelAdapter
-> ConsoleProviderGateway -> chat_api_call -> DeepSeek adapter
-> temporary loopback HTTP server
```

They prove complete multi-call continuation, exact call/result pairing, no
synthetic user, a real tool-influenced final answer, and terminal provider usage
reaching the agent budget. Crash-point fixtures prove the assistant batch is
persisted before execution, each result before continuation, pending resume
requires fresh approval, completed/failed never re-run, and ambiguous executing
never auto-runs. Restart, sync, ordinary JSON import, branch/regeneration, and
later DeepSeek turns replay exact required reasoning in both modes. Partial-call
cancellation after a downstream parser checkpoint executes zero tools and
closes the live response exactly once.

Mutation checks remove the pre-execution checkpoint, later-turn reasoning,
call/output adjacency, Responses sequence guard, terminal usage, or close
forwarding and must fail before restoration.

### Settings/catalog tests

- real Pilot render/load/default/invalid repair, provider switching, search,
  save/revert, atomic failure, and second-save no-op;
- frozen Console resolution across Settings mutation and auxiliary calls;
- model discovery URL/key parity, cache fallback/cap/search/write-through
  privacy, and no unrelated-provider drift.

### Optional paid live tests

Default collection makes no request. Each mode requires both
`TLDW_LIVE_DEEPSEEK=1` and a nonblank `DEEPSEEK_API_KEY`. The two modes run in
fresh subprocesses with isolated HOME/XDG/config/data paths established before
Chatbook imports, no log sinks, discarded stdout/stderr, randomized prompts and
arithmetic values, and proof that one exact Calculator result influences the
final answer. Secrets, prompts, responses, and tool results never appear in
assertion/error text. Optional model/base overrides are provider-specific.

## Documentation And Delivery

README, Settings guide, and Console guide document:

- stable DeepSeek identity, credential, current models, and base;
- exact mode values and Chat default;
- mode-specific supported parameters and omissions;
- thinking default/low/high/max and sampler/tool-choice behavior;
- stateless explicit Responses history and no provider continuation IDs;
- existing function tools and provider built-in exclusions;
- durable private later-turn reasoning, explicit resume/takeover, and ambiguous
  execution recovery;
- streaming terminal/usage differences;
- model discovery/cache and unknown pricing behavior;
- invalid configuration recovery and optional live gates.

This is the third PR in the approved sequence:

1. TASK-15675: durable continuation foundation;
2. TASK-15676: hosted Chat wire plus Moonshot/Kimi and Z.ai/GLM;
3. TASK-15677: DeepSeek dual API using both foundations.

The DeepSeek PR does not duplicate checkpoint storage, native tool execution,
or hosted Chat transport. Any necessary change to a shared seam must preserve
its existing provider tests and remain provider-gated.

## ADR Check

ADR required: yes

ADR paths:

- [ADR-063](../../../backlog/decisions/063-hosted-provider-wire-and-durable-tool-continuation.md)
- [ADR-064](../../../backlog/decisions/064-deepseek-dual-api-provider-boundary.md)

Reason: this feature adds a provider API-mode contract, semantic stream
translation, strict durable private-history replay, and cross-layer Settings/
runtime/persistence boundaries. ADR-064 records the DeepSeek-specific choice;
ADR-063 owns the shared durable continuation boundary.

## Acceptance Summary

The task is complete when DeepSeek remains one ordinary first-class provider,
defaults safely to Chat Completions, can explicitly use Responses, supports the
existing native function tools in both modes, and preserves every provider-
required tool-reasoning round across restart/sync/import/later turns—without
provider built-in tools, server-side continuation state, hidden retries, secret
leakage, or behavior changes to unrelated providers.
