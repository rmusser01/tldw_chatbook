# ADR-045: Treat QwenCloud's dual APIs as one first-class provider boundary

Status: Accepted
Date: 2026-08-07
Related Task: [TASK-3771](../tasks/task-3771%20-%20Add-QwenCloud-dual-API-provider-support.md)
Related Spec: [QwenCloud Dual-API Provider Design](../../Docs/superpowers/specs/2026-08-02-qwencloud-dual-api-provider-design.md)

## Decision

Chatbook will expose QwenCloud as one first-class provider with the durable
normalized identity `qwencloud`, equivalent at shared application boundaries
to providers such as OpenAI and DeepSeek.

QwenCloud has one persisted provider-specific setting:
`[api_settings.qwencloud].api_mode`. It accepts `responses` and
`chat_completions` and defaults to `responses`. The selected mode changes only
the dedicated QwenCloud adapter's endpoint, request mapping, and response/event
translation. It does not create separate provider identities, model catalogs,
readiness systems, Console paths, or agent runtimes.

Console resolves and pins that value, with the effective QwenCloud base URL,
in its ordinary provider resolution before a run. Shared resolution and
dispatch code only carries those values; it never branches on the mode to map
wire formats or continue tools. Direct callers may pass an explicit mode, and
the adapter otherwise applies the same config/default resolution.

Both external APIs normalize to Chatbook's existing OpenAI-style internal
message contract. QwenCloud therefore uses the existing provider dispatcher,
Console gateway, native tool-call accumulator, `AgentService`, `agent_runtime`,
approval/execution policies, budgets, cancellation, run logs, and model-catalog
pipeline. Responses continuations are stateless translations of the canonical
assistant `tool_calls` and paired `role="tool"` history; Chatbook does not use
or persist `previous_response_id` or a QwenCloud conversation ID. The
Responses mapper emits each `function_call` immediately followed by its paired
`function_call_output`, as QwenCloud requires, even though canonical history
stores a call batch before its result batch.

Chatbook does not persist private reasoning content. The Chat Completions
adapter therefore disables preserved-thinking replay, which `qwen3.8-max`
otherwise enables by default and requires callers to echo exactly. Responses
requests include `store=false` where the compatible endpoint honors it, but
Chatbook makes no claim about provider operational retention.

Only existing Chatbook function tools are supported in this tranche.
QwenCloud-hosted built-in tools are excluded and require a separate decision
and feature.

## Context

QwenCloud provides both an OpenAI-compatible Chat Completions endpoint and a
Responses endpoint. Their wire formats differ substantially: Responses uses
typed input/output items, `function_call` and `function_call_output` objects,
and typed streaming events. Users still expect provider selection, readiness,
streaming, model discovery, and tool execution to behave like every other
hosted provider.

The repository previously contained a legacy Chat/CCP streaming and textual
tool-continuation pipeline. ADR-026 and TASK-577 retired that pipeline after it
had been unreachable since the native Console migration. Designing QwenCloud
around it would reintroduce a second chat runtime and make QwenCloud behave
differently from OpenAI and DeepSeek. The live native Console already has the
correct structured accumulator and assistant/tool continuation semantics.

ADR-006 assigns durable provider defaults to Settings, effective run settings
to Console, and wire translation to provider adapters. ADR-012 owns provider
credentials. ADR-020 owns cloud model-catalog refresh. This decision applies
those established boundaries to QwenCloud's two external APIs.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Expose `QwenCloud Responses` and `QwenCloud Chat Completions` as separate providers | Splits credentials, models, readiness, metrics, and user configuration even though they are two transports for the same service. |
| Register QwenCloud as OpenAI or Custom OpenAI | Causes identity and configuration leakage and cannot correctly translate Responses messages, tools, or events. |
| Add a Qwen-specific stream parser and continuation loop outside Console | Duplicates the live native runtime, bypasses shared approval/cancellation/budget behavior, and recreates the retired legacy architecture. |
| Restore the retired Chat/CCP pipeline to support another consumer | Reverses ADR-026 without a user-facing need; CCP is a management/display surface and Console is the interactive chat owner. |
| Use QwenCloud `previous_response_id` state | Introduces external state ownership and persistence/recovery questions that ordinary providers do not impose on Chatbook conversations. |
| Include QwenCloud built-in tools now | Mixes provider-hosted execution and permissions with Chatbook's existing local/MCP tool policy; it is an independent security and product feature. |

## Consequences

- Users see one QwenCloud provider and one additional API-mode selector.
- `responses` is the default, while `chat_completions` is available for model
  or parameter compatibility.
- Shared provider surfaces must register `qwencloud` exactly as they register
  other hosted providers.
- The adapter is more involved than a simple OpenAI-compatible wrapper because
  Responses needs bidirectional message/tool translation and stream-event
  normalization.
- The Responses mapper validates exact one-to-one tool call/results and
  reorders only within a canonical call/result batch to meet the external
  adjacency rule.
- Chat Completions deliberately gives up preserved historical reasoning until
  Chatbook has an explicit, privacy-reviewed reasoning-content contract.
- Tool behavior remains provider-neutral: the adapter never executes a tool or
  decides continuation policy.
- No schema migration or durable QwenCloud response/conversation state is
  introduced.
- Adding hosted QwenCloud tools, per-session API-mode overrides, or server-side
  Responses state later requires a new ADR check.

## Verification Consequences

Provider adapter tests alone are insufficient. At least one joined test for
each API mode must traverse the real Console gateway, native accumulator,
`AgentService`/`agent_runtime` continuation, and second adapter request. A
provider may enter `NATIVE_TOOLS_PROVIDERS` only after those tests prove it
forwards schemas, returns canonical calls, and accepts canonical tool history.

Optional paid live checks remain explicitly gated, credential-isolated, and
outside the default suite.

The contract suite also proves mode/base pinning through the shared resolution
and dispatcher, Responses call/output adjacency, Qwen SSE framing and
delta/terminal de-duplication, terminal usage/finish normalization, Chat
streaming usage opt-in, and the adapter-owned state/reasoning invariants.

## Links

- [ADR-006: Provider-Aware Generation Settings](006-provider-aware-generation-settings.md)
- [ADR-012: Provider Credential Settings Boundary](012-provider-credential-settings-boundary.md)
- [ADR-020: Automatic Model Catalog Refresh](020-automatic-model-catalog-refresh.md)
- [ADR-026: Retire the Chat-tab Conversation Entry Chain](026-retire-chat-tab-conversation-entry-chain.md)
