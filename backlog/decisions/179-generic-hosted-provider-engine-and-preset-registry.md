# ADR-179: Generic hosted provider engine and preset registry

Status: Accepted
Date: 2026-09-23
Related Task: [TASK-33010](../tasks/task-33010%20-%20Generic-hosted-provider-engine-Phase-1-registry-engine-Databricks.md)
Related Spec: [Generic hosted provider engine and presets design](../../Docs/superpowers/specs/2026-09-23-generic-hosted-provider-engine-and-presets-design.md)
Related Plan: [Generic hosted provider engine Phase 1 implementation plan](../../Docs/superpowers/plans/2026-09-23-generic-hosted-provider-engine-phase1.md)

## Context

Adding a first-class hosted provider costs a ~1,000-line strict adapter plus
edits to ~20 scattered literal tables (Moonshot/ZAI: 39 files, +9,185/-1,052).
Goal: Hermes-style provider breadth without losing the strict hosted wire
boundary of ADR-062/063.

## Decision

1. `tldw_chatbook/provider_registry.py` (stdlib-only leaf) is the single
   source of provider identity; the scattered literal tables
   (`_cloud_provider_keys`, readiness key sets, display names, endpoint maps,
   `NATIVE_TOOLS_PROVIDERS`, `AUTO_REFRESH_PROVIDER_LIST_KEYS`,
   `SENSITIVE_AUXILIARY_AUDITED_ENDPOINTS`, continuation pairings) become
   registry-derived, each guarded by a parity test.
2. Engine-driven providers are preset records consumed by
   `LLM_Calls/hosted_provider_engine.py::build_hosted_chat_handler`,
   which delegates transport to `hosted_chat` (ADR-062 boundary unchanged).
3. Response strictness stays fail-closed; per-preset `response_allowances`
   (recorded from live-probe envelopes) are the only tolerated extras.
   The long-tail tolerant profile (Phase 2) is scoped to user-registered
   custom endpoints only.
4. moonshot.py and zai.py are not migrated (ADR-063 evidence-gated policy).
5. xAI/Grok support is deliberately excluded (maintainer decision).
6. Bedrock is in scope via its OpenAI-compatible endpoints + Bearer API keys
   (AWS, Dec 2025); native Converse wire and SigV4 stay out (fallback only).

## Consequences

Adding an OpenAI-compatible hosted provider becomes one registry record +
one dispatch registration line + tests. Sensitive-request audit and
cloud/local classification become registry-coverage tests instead of
hand-typed literals. Related: ADR-002, ADR-012, ADR-020, ADR-062, ADR-063,
ADR-146.

## Links

- [ADR-002: OpenAI-Compatible Model Discovery](002-openai-compatible-model-discovery.md)
- [ADR-012: Provider Credential Settings Boundary](012-provider-credential-settings-boundary.md)
- [ADR-020: Automatic Model Catalog Refresh](020-automatic-model-catalog-refresh.md)
- [ADR-062: Hosted Chat Completions Provider Boundary](062-hosted-chat-completions-provider-boundary.md)
- [ADR-063: Use a neutral hosted wire boundary with durable tool continuation](063-hosted-provider-wire-and-durable-tool-continuation.md)
- [ADR-146: Console custom endpoint registry](146-console-custom-endpoint-registry.md)

## Provider-owned response normalization amendment (2026-10-04, TASK-34367.1-.5)

Documented provider fields are qualified by primary schemas and actual-adapter
complete/SSE replays when paid captures are unavailable; this establishes an
offline contract, not a claim of live compatibility. The evidence source and
remaining live-capture boundary stay explicit beside each preset.

A hosted adapter may supply one call-owned wire normalizer after bounded JSON
validation and before strict response/event validation. The default supplies
none. The normalizer must preserve required content, tool, finish and accounting
fields, reject unknown/malformed fields that it consumes, and leave all other
fields for the existing closed parser. No arbitrary metadata is persisted.
Groq promotes only streamed x_groq.usage to ordinary usage, rejects conflicting
usage and nested x_groq.error, and keeps complete-response hardware-cache usage
separate. OpenRouter may reconcile its documented content-free final usage
choice repeating the already accepted finish state; changed finish/index,
content/tool/reasoning deltas, duplicate usage and error records fail closed.
Provider annotations remain scoped; unsupported vendor execution features do
not become allowed extras. This keeps ADR-062/063 ownership without teaching
the neutral parser provider names or loosening its default terminal contract.

Alternatives rejected: dropping all x_groq metadata loses accounting and hides
errors; universal extra fields or repeat-terminal tolerance weakens unrelated
providers; duplicated transports would bypass the shared resource/SSE guards.

## Tool-call shape clarification (2026-10-05, TASK-34364)

`ProviderRecord.tool_call_allowances` extends the existing record-owned
extra-field contract to non-streamed tool-call objects. Required id/type/function
fields and their validation remain strict; named extras pass the existing value
rule and are dropped. Fireworks allows only index/name, qualified by its retained
live tool fixture. No other record inherits this allowance. Provider-owned wire
normalization still runs between bounded JSON and closed shape validation.

For an established streamed call index, a null repeated id/type/function.name
claims no new value and is treated as omitted. A first delta must establish its
required identity; a changed non-null identity/type/name still refuses. This
shared continuation rule does not tolerate arbitrary extra keys or relax finish,
usage, argument, transport or resource limits.
