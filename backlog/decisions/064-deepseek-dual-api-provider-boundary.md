# ADR-064: Treat DeepSeek Chat Completions and Responses as one provider

Status: Accepted
Date: 2026-08-12
Related Task: [TASK-15677](../tasks/task-15677%20-%20Add-DeepSeek-dual-API-provider-support.md)
Related Spec: [DeepSeek Dual-API Provider Design](../../Docs/superpowers/specs/2026-08-12-deepseek-dual-api-provider-design.md)
Supersedes: N/A

## Decision

Chatbook will expose DeepSeek as one first-class provider with stable key
`deepseek` and one persisted `api_mode`. The setting accepts
`chat_completions` and `responses` and defaults to `chat_completions` so
existing behavior remains the default.

Console freezes the selected mode with provider, model, normalized base,
credential, timeout, and retry policy for the run. Shared application code
carries the frozen value but does not translate wire formats. The DeepSeek
adapter owns separate pure request/history builders and response translators
for the two modes. Chat Completions uses ADR-063's neutral hosted Chat wire;
Responses uses strict semantic SSE/event translation without routing through
the QwenCloud provider adapter.

Both modes normalize to Chatbook's existing model-turn and native function-tool
contracts. Only existing Chatbook function tools are supported. DeepSeek web
search, custom `apply_patch`, and other provider-hosted tools remain excluded.
Chatbook sends explicit stateless history and does not use provider
conversation IDs or `previous_response_id`.

Thinking uses provider defaults unless the user selects `low`, `high`, or
`max`. Compatibility aliases that add no distinct behavior are rejected locally.
Thinking-mode requests omit unsupported sampling fields. Tool-bearing
`reasoning_content` and paired tool history use ADR-063 checkpoints. DeepSeek's
documented requirement is stricter than GLM and non-K3 Kimi families:
completed tool-associated reasoning remains in later same-provider requests
while its owning visible turn remains inside the context window. Kimi K3 has a
separate broader always-on preserved-history policy under ADR-063.

Responses requests use an exact allowlist and omit unsupported stateful or
server-managed fields rather than relying on provider-side silent ignoring.
Streaming accepts only the documented reasoning, text, function-call, usage,
and terminal events needed by this feature. Sequence, output identity,
call/output pairing, replay, terminal, bounds, and lifecycle violations fail
closed before any incomplete call reaches the executor.

## Context

DeepSeek historically used an independent Chat-Completions handler in
`LLM_API_Calls.py`. The provider now publishes a Responses API with typed input
items and semantic stream events. Exposing a second provider identity would
split one credential/model/catalog/readiness owner, while treating Responses as
raw OpenAI Chat would lose its event and continuation semantics.

DeepSeek thinking-mode tool calls require complete `reasoning_content` in all
subsequent requests. Dropping it can produce a 400 response and breaks restored
conversations. ADR-063 supplies the durable private-history owner without
introducing server-side response state.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Make `DeepSeek Responses` a separate provider | Duplicates credentials, models, readiness, Settings, metrics, and catalog identity. |
| Default existing users to Responses | Changes established DeepSeek behavior without migration need; Chat Completions remains supported. |
| Route Responses through QwenCloud | Shares a broad protocol family but leaks provider-specific event, parameter, finish, and reasoning policy. |
| Use DeepSeek `previous_response_id` or conversations | The documented endpoint does not support them and Chatbook owns explicit portable history. |
| Enable DeepSeek built-in web/custom tools | Bypasses Chatbook's existing tool approval/execution boundary and needs a separate security decision. |
| Keep reasoning only until the final tool answer | Contradicts DeepSeek's documented later-turn replay requirement. |
| Accept `minimal`/`medium`/`xhigh` aliases | They map onto the exposed `low`/`high` values and add settings values with no distinct behavior. |

## Consequences

- DeepSeek gains one API-mode selector; unrelated providers remain unchanged.
- Existing saved configurations without `api_mode` resolve to
  `chat_completions`.
- The adapter has two provider-specific translations but one shared
  configuration, readiness, metrics, model catalog, Console, and tool runtime.
- DeepSeek private tool history becomes durable, synced, exportable, hidden,
  budgeted, and branch-owned under ADR-063.
- Switching provider, model, mode, or base cannot silently continue an active
  checkpoint; the user must finish/discard it or restore the pinned resolution.
- Optional paid tests remain explicitly gated and isolated; default tests make
  no provider request.

## Links

- [ADR-006: Provider-Aware Generation Settings](006-provider-aware-generation-settings.md)
- [ADR-020: Automatic Model Catalog Refresh](020-automatic-model-catalog-refresh.md)
- [ADR-045: QwenCloud Dual-API Provider Boundary](045-qwencloud-dual-api-provider-boundary.md)
- [ADR-063: Hosted Wire and Durable Tool Continuation](063-hosted-provider-wire-and-durable-tool-continuation.md)
