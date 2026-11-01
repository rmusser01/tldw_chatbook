# Versioned local-thinking response projection

User approved this contract repair on2026-10-18: “Approve versioned profile repair.” This is the concrete future-only versioned profile proposed in the archive recovery live QA; implementation remains within that approved intent.

## Problem and outcome

Native Qwen search completes, but its response profile freezes start_anchored_think while actual typed terminal reasoning is reasoning_content. Canonical equality refuses that source mismatch and settlement retains a safe artifact. A later completed-tool-turn successor correctly refuses the unverified terminal revision. New supported local captures must establish exact saved-response ownership and permit normal captured successors, including archive restore follow-ups.

## Contract

- New supported displayable local profiles use version2 in existing request-header adapter_defaults.response_projection. Keys are exactly version, thinking_stream_disposition, thinking_round_trip_version, provider, model, protocol, source_formats. thinking_round_trip_version stays1; disposition is displayable. No schema or migration.
- The closed adapter keys are llama_cpp, local_llamacpp, vllm and local_vllm. A nonblank string model is required. Canonical source_formats is [start_anchored_think, reasoning_content], in that order, except models containing kimi (case-insensitive): their existing structured parser explicitly refuses them, so the list is [start_anchored_think]. No other source or provider is admitted by v2. Both producer and reader consume one small shared rule in the existing semantic-revision module; no new service/module/dependency.
- This list records formats the existing selected adapter can parse; it does not assert a reviewed server template, native tools, replay permission or a particular response format. Preserve actual literal block.source_format, text and provenance. Do not relabel structured reasoning as tagged text.
- Reader validates the exact v2 shape, integer versions (bool rejected), closed provider/model-derived ordered list and unchanged provider/model/protocol ownership. It then accepts only the exact terminal block's source_format from that list, while preserving exact terminal round, single-block, status/type, raw equality HMAC, credential/PII policy and source checks.
- Existing v1 profiles keep their exact scalar source_format contract and prior behavior. Headers without a profile retain visible-only meaning. Proprietary hosted profiles remain v1, unchanged. Unsupported or malformed profiles/blocks remain refused or artifact-backed under existing handling.
- Existing calls, headers, response links, semantic revisions, saved messages and historical reconstructions are immutable. No migration, relinking old artifacts or reinterpretation of v1. Corrected ownership is prospective and must be qualified on fresh captures.
- Completed-tool range/source/lineage/policy/revision verification and factory/runtime guards remain unchanged. No arbitrary source matching, thinking/history disabling, replay override, capture bypass or authority expansion.

## Scope and validation

Production: console_trace_service.py profile producer and console_semantic_revision.py shared closed format rule/version-aware projection. Existing test_console_trace_thinking_settlement.py and test_console_archive_recovery_followup.py carry coverage; reuse _call in test_console_trace_settlement.py with optional provider/model keyword arguments, preserving its defaults and callers. Use an additional existing native-gateway test only if the connected parser boundary requires it. No production UI, provider/parser, schema, memory, permission, retention or Cancel/Discard change. The nonempty Persona system-provenance issue remains separate.

First establish RED with the actual Qwen reasoning_content shape through the real gateway/parser/SQLite settlement and a successor, fake only the documented external adapter/client seam with fail-closed network guard. Then require GREEN for tagged and structured formats, exact revision/verified_equal links, Direct/RAG native archive confirmation/restore, immutable predecessors and saved text, v1/legacy stability, malformed/unknown source/provider/model/protocol/round/status refusal, raw-equality/credential-mask collision and PII preservation. Reuse covering existing controls rather than duplicate them. No whole-suite sweep.

Root owns live verification, task/plan/ledger and endpoint budget. Source baseline fcca09d3f810e035f5b89b3df09b6bda1c228efd. Global model consumption17of20,3remain, no reset. Workers/reviewers make0app/model/network calls. Root may request a budget extension only after the correction and independent review are concrete; a fresh full run must fit its actual hard transport cap, covering requests and HTTPX/fallback/auxiliary paths.

## ADR and alternatives

ADR required: yes, amend existing ADR097 before implementation.
ADR path: backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md, amendment dated2026-10-18 for version2 local response formats; ADR090/166 still govern thinking/recovery.
Reason: the frozen response projection is a shared versioned service contract. Existing immutable-header storage suffices; no duplicate ADR or schema migration.

Rejected: widening v1 (changes old meaning), rewriting old artifact links (loses original proof), replacing all source_format values with tagged/structured (lies about parser provenance), dropping equality/terminal revision checks (grants unverified saved ownership), disabling capture/thinking (does not fix the requested flow), or changing server/template settings (not needed to support already parsed structured events).
