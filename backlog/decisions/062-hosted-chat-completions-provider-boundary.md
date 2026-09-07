# ADR-062: Use one strict hosted Chat-Completions wire boundary

Status: Superseded by ADR-063
Date: 2026-08-12
Related Task: [TASK-15676](../tasks/task-15676%20-%20Harden-Moonshot-Kimi-and-Z.ai-GLM-as-first-class-hosted-providers.md)
Related Spec: [Moonshot/Kimi and Z.ai/GLM Hosted Chat-Completions Design](../../Docs/superpowers/specs/2026-08-12-kimi-zai-hosted-chat-completions-design.md)

Superseded because ADR-063 retains the hosted Chat-Completions transport
boundary but replaces this ADR's ephemeral-only reasoning decision with a
durable, synced, branch-owned continuation checkpoint contract.

## Decision

Chatbook will use a provider-neutral, strict hosted Chat-Completions wire layer
for hosted providers whose external protocol is the OpenAI Chat Completions
message/choice/SSE family. Moonshot AI (`moonshot`) and Z.ai (`zai`) are the
first migrations. Existing public provider identities and handler entry points
remain stable.

The neutral layer owns only transport and generic wire correctness: HTTP
request/session lifecycle, bounded retries, strict SSE framing, OpenAI-shaped
choice/tool/usage validation, resource closure, limits, and redacted typed
errors. It is dependency-low and receives immutable resolved inputs; it does
not read configuration, import Console/Settings/provider adapters, select
models, execute tools, or decide provider-specific parameters.

Provider-specific pure builders remain responsible for exact model-family
payload allowlists, function/tool-choice rules, reasoning/thinking behavior,
finish-state classification, and canonical-history translation. A compatible
wire shape does not make two providers semantically identical.

The shared Console, `AgentService`, and `agent_runtime` continue to own native
tool disclosure, approval, execution, continuation, budgets, cancellation,
and persistence. Z.ai may enter `NATIVE_TOOLS_PROVIDERS` only after joined
application tests prove the existing registry invariants. Moonshot's existing
membership receives the same stronger proof.

Kimi and Z.ai `reasoning_content` may cross the provider/gateway/agent boundary
only as a fixed, bounded, string-valued, call-scoped field on the in-memory
assistant tool-call message. It is retained solely for an active tool
continuation and is never visible or durable. The interface is not an open
provider-metadata bag. Z.ai uses `clear_thinking=false` only for that active
tool run and `true` for ordinary chat. Broader Kimi/Z.ai preserved reasoning in
unrelated multi-turn chat is deliberately out of scope. ADR-045 continues to
govern QwenCloud and is not changed by this exception.

QwenCloud Responses remains provider-specific. Its Chat-Completions parsing
primitives may be extracted into the neutral layer only under complete
behavioral parity; provider-specific flags that weaken its existing contract
are not accepted.

This decision does not migrate every compatible provider at once. DeepSeek,
Groq, Mistral, OpenRouter, local servers, custom OpenAI providers, and OpenAI's
dual API retain their existing behavior until separately tested migrations are
justified.

## Context

Chatbook has more than fifteen provider handlers. Several hosted providers use
an OpenAI-like Chat Completions wire shape, but their implementations duplicate
HTTP sessions, retries, line-oriented SSE relays, errors, and response parsing.
The old Moonshot and Z.ai handlers demonstrate the cost: stale defaults,
different config sources, raw line relays, unclear stream ownership, missing
stream usage, and inconsistent function-tool continuation.

A core created permanently for only two vendors would be needless abstraction.
Conversely, rewriting every OpenAI-compatible provider in one task would turn a
two-provider reliability feature into a high-risk repository-wide transport
migration. The durable boundary is therefore neutral and migration-ready, but
adoption is incremental and evidence-gated.

External wire similarity also has limits. Kimi K3 omits legacy sampling fields
and needs exact reasoning continuation for some tool loops. Z.ai has its own
thinking, finish, and tool-choice rules. Local compatible servers intentionally
accept nonstandard parameters. A generic passthrough client would hide these
differences and recreate invalid-provider requests.

ADR-006 already assigns persisted generation defaults to Settings, effective
resolution to Console, and request translation to provider adapters. ADR-012
owns provider credential UX. ADR-020 owns cloud model catalog refresh. ADR-045
keeps QwenCloud's dual APIs under one provider-specific translation boundary.
This ADR adds only the missing common hosted Chat wire/lifecycle contract and
the narrow active-run reasoning handoff.

## Boundary Rules

- Stable provider keys remain `moonshot`, `zai`, and `qwencloud`; model family
  names do not become provider identities.
- There is no Moonshot or Z.ai `api_mode` until an official supported Responses
  contract exists and receives a new design/ADR check.
- Config/readiness resolves and freezes provider inputs before transport.
- The neutral transport has no global-config or UI dependency.
- Provider builders use explicit allowlists; unknown kwargs are not forwarded.
- Endpoint validation is structural and shared with discovery, not host-bound.
- Streaming is not replayed after any response-body byte is consumed.
- Non-streaming requests are not replayed after any 2xx response is received.
- The provider stream owns and closes response/session resources exactly once.
- Only complete normalized function calls reach the executor.
- Vendor built-in tools require a separate security/product decision.
- Hidden reasoning is bounded, call-local, ephemeral, invisible, and excluded
  from persistence/logs/errors/usage; it is replayed only in the immediate
  same-provider tool continuation.
- Optional paid tests are doubly gated and profile-isolated; default tests make
  no paid calls.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Keep independent Moonshot and Z.ai implementations | Continues duplicated lifecycle, retry, streaming, usage, and privacy defects and gives the next compatible provider no reusable contract. |
| Create a Moonshot/Z.ai-specific shared helper | Encodes a coincidental pair as architecture and offers no principled path for other hosted Chat providers. |
| Route both through the OpenAI provider/client | Leaks provider identity/configuration and cannot enforce Kimi/GLM model-family, thinking, finish, endpoint, and tool-choice rules. |
| Migrate every compatible provider now | Excessively broad blast radius; local and dual-API providers have materially different compatibility requirements. |
| Add a broad policy object with flags for every possible provider quirk | Speculative flexibility becomes a god object. Shared mechanics and provider-specific builders keep ownership explicit. |
| Persist Kimi/Z.ai reasoning in conversation rows | Expands private data ownership, schema, export, deletion, and sync obligations merely to continue an active tool loop. In-memory call-scoped echo is sufficient. |
| Drop provider reasoning like QwenCloud Chat | Can make official Kimi and Z.ai multi-step tool continuation invalid or degraded; both get the same narrow privacy-reviewed active-run exception instead. |
| Enable Z.ai `tool_stream` automatically | Changes provider behavior without a user need; ordinary complete or fragmented tool-call parsing is sufficient. |
| Add vendor built-in tools with function tools | Mixes provider-hosted execution and permissions with Chatbook's existing local/MCP policy and needs a separate decision. |

## Consequences

- Moonshot and Z.ai gain one tested lifecycle and normalized wire contract while
  retaining separate provider semantics.
- QwenCloud Chat may become a third consumer of lower-level primitives, but
  QwenCloud Responses and ADR-045 remain unchanged.
- Later hosted providers can migrate incrementally with provider-neutral
  contract tests rather than another transport rewrite.
- Some duplication remains intentionally in non-migrated providers until their
  own parity evidence exists.
- Kimi and Z.ai active tool runs carry additional in-memory private data;
  strict limits, call scoping, and negative persistence/log tests are
  mandatory.
- No database migration or durable provider state is introduced.
- Current default models change only for fresh/missing settings. Explicit saved
  models remain user-owned.

## Verification Consequences

Adapter unit tests alone are insufficient. Verification must include:

- pure provider/config/endpoint and exact payload tests;
- shared transport/SSE/retry/error/usage/lifecycle tests;
- QwenCloud before/after compatibility and mutation checks;
- joined Console-to-loopback-HTTP tool continuation and partial-call
  cancellation for Moonshot and Z.ai;
- negative tests proving Kimi/Z.ai reasoning never reaches durable or visible
  surfaces and concurrent runs do not cross-contaminate;
- real Settings Pilot and model-catalog endpoint/cache tests;
- optional paid provider tests only behind explicit gate plus credential.

Z.ai registry eligibility is the final production step after its joined test is
green. Any Qwen compatibility regression blocks later migration slices.

## Links

- [ADR-006: Provider-Aware Generation Settings](006-provider-aware-generation-settings.md)
- [ADR-012: Provider Credential Settings Boundary](012-provider-credential-settings-boundary.md)
- [ADR-020: Automatic Model Catalog Refresh](020-automatic-model-catalog-refresh.md)
- [ADR-045: QwenCloud Dual-API Provider Boundary](045-qwencloud-dual-api-provider-boundary.md)
- [Implementation design](../../Docs/superpowers/specs/2026-08-12-kimi-zai-hosted-chat-completions-design.md)
