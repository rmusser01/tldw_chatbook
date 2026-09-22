# ADR-175: One strict-JSON acceptance contract and one home for shared payload validators

Status: Accepted
Date: 2026-09-21
Related Task: [TASK-32855](../tasks/task-32855%20-%20One-strict-JSON-parser-and-one-home-for-hosted-validators.md)
Related Spec: cascade review 2026-09-19 (`qa/cascade-review-2026-09-19/report.md`)
Amends: [ADR-062](062-hosted-chat-completions-provider-boundary.md)

## Decision

Chatbook keeps exactly one strict-JSON acceptance rule, owned by
`Utils/input_validation.strict_json_loads`: malformed JSON, non-finite
constants (`NaN`/`Infinity`), duplicate object keys, non-string object keys,
non-finite floats, and shapes past the depth/node caps are all refusals.
Every provider wire boundary (hosted chat streaming/non-streaming, QwenCloud
chat streaming, QwenCloud Responses argument decoding) and every
continuation/checkpoint storage boundary (provider continuation, thinking
blocks) applies this one rule through the shared parser. A decoded
tool-arguments payload therefore can never be accepted on the wire and later
refused at its continuation checkpoint.

Moonshot and Z.ai additionally carried byte-identical (label-only) request
payload validators — numeric coercers, stop/response-format/call-batch
normalizers, and the bounded-shape checker — as two hand-maintained copies.
They live once in `hosted_chat.ProviderPayloadValidators`, parameterized by
the provider's wire identity (`provider`, `label`) exactly as the
cascade review scoped: mechanics shared, per-provider payload builders
unchanged (ADR-062's builders-stay-with-providers rule).

## Boundary Rules

- `strict_json_loads` is the only new strict-JSON entry point; family
  wrappers may narrow (pin their own caps) but not loosen it.
- Request-side history normalization (`_normalize_call_batch` and peers)
  keeps its existing plain `json.loads` + bounded-shape check: user-owned
  conversation history is not a wire surface, and last-wins duplicate keys
  there are a stored-format question, not an acceptance question.
- `resolve_provider_api_key` stays per provider: it resolves credentials
  from configuration, which the neutral layer never imports (ADR-062
  transport rule; ADR-012 owns credential settings).
- Metadata surfaces that reject only non-finite constants today
  (console dispatch checkpoints, generation-settings metadata, library
  activity/preparation, workspace tool protocol, Actor_Packs export,
  Workflows document service, MCP permission store) — and the
  duplicate-key-hook-only metadata readers (model artifacts service,
  visual identity, VisualIdentity_DB, vllm profiles) — keep their local
  rejectors. Their acceptance rules are storage-format
  contracts with their own size/trust guards; tightening them to full
  strict acceptance (caps + duplicate-key refusal) is a separate, surfaced
  decision per surface, not a rider on this one.

## Context

The 2026-09-17 core review found three drifted strict-JSON families: wire
parsers with depth/node caps but no duplicate-key rejection; storage parsers
with duplicate-key rejection but no caps; and ~ten parse-constant-only
sites. The drift was a live defect: a provider emitting a duplicate key in
tool arguments passed the wire family and threw an uncaught
`ContinuationValidationError` at the continuation checkpoint.
TASK-32805.5 landed the shared parser and wired the storage family plus the
hosted/qwencloud streaming decoders to it; this decision records the
contract, completes the QwenCloud Responses adoption, consolidates the
duplicated moonshot/zai validators, and enumerates the remaining
constant-only surfaces instead of silently tightening them.

## Consequences

- One acceptance rule to test: duplicate-key and depth-cap edges are pinned
  at the parser and replayed end-to-end through a wire tool-arguments
  payload.
- Provider error text is unchanged: the shared validators reproduce the
  exact per-provider messages (`"Moonshot stop is invalid."`,
  `"Z.ai stop is invalid."`).
- `hosted_chat` grows a pure, config-free validators seam; ADR-062's
  transport boundary is untouched.
- Future hosted Chat-Completions providers adopt the validators instead of
  copying them a third time.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Per-family strictness stays | Recreates the accepted-on-wire/refused-at-checkpoint defect class. |
| One global `json.loads` monkeypatch | Hidden coupling; acceptance must be a visible boundary decision. |
| Move `_resolve_api_key` into the shared validators | Imports configuration into the neutral layer (ADR-062) and crosses the ADR-012 credential boundary for zero duplication win beyond label strings. |
| Tighten every `parse_constant` site now | Changes storage-format acceptance for ten surfaces in one rider; each needs its own compatibility story. |

## Links

- [ADR-062: Hosted Chat-Completions provider boundary](062-hosted-chat-completions-provider-boundary.md)
- [ADR-012: Provider credential settings boundary](012-provider-credential-settings-boundary.md)
