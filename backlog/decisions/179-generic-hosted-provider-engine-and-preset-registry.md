# ADR-179: Generic hosted provider engine and preset registry

Status: Accepted
Date: 2026-09-23
Related Task: [TASK-32917](../tasks/task-32917%20-%20Generic-hosted-provider-engine-Phase-1-registry-engine-Databricks.md)
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
