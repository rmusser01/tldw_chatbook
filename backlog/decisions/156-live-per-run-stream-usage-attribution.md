# ADR-156: Attribute bounded live model usage by run and call sequence

Status: Accepted  
Date: 2026-09-12  
Related Task: [TASK-18923](../tasks/task-18923%20-%20Agent-rail-live-per-run-status-line-elapsed-and-streaming-tokens.md)  
Related Spec: [Live per-run usage](../../Docs/superpowers/specs/2026-09-12-live-per-run-usage-design.md)

## Context

The Console already shows live elapsed/activity state, but final token accounting appears only after a run finishes. Primary and child calls share one streaming adapter and can overlap on different threads. Provider usage is intermittent and often terminal; retaining chunks or repainting for every delta would create unbounded memory or UI work. Adding exact run identity across `AgentService` and the adapter is an internal cross-module service contract, so it requires an ADR.

## Decision

`AgentService` accepts an optional `run_model_scope(run_id, agent_kind)` context-manager callback and enters it around each `run_agent_loop`. The Console adapter captures that thread-local attribution before crossing to its async loop and emits typed, sequence-numbered start/text/provider-usage/finish events to the bridge.

The bridge retains one scalar accumulator per active run. Provider output counts are authoritative when explicitly present and are valid integers at least zero; otherwise cumulative streamed UTF-8 bytes yield an explicitly local approximation of `ceil(bytes / 4)`. It publishes the first observable scalar immediately and later changes at most once per second. Sequence checks reject late events, and call finish plus terminal/prune/shutdown paths remove state.

Existing UI timers read the scalar into immutable snapshots. Final `RunOutcome.total_tokens`, persistence, budget weighting, pricing, and historical accounting do not change.

## Consequences

Live counts have honest provenance and exact run isolation without storing response text or adding a tokenizer, timer, database field, or per-chunk UI event. Provider zero can temporarily yield no visible segment, but remains more accurate than substituting a local estimate. The optional service callback is additive for non-Console callers, while its lifecycle and concurrency behavior become a tested contract.

## Alternatives rejected

- Use final `total_tokens` during streaming: it is unavailable and represents different budget/accounting semantics.
- Estimate each chunk independently: rounding inflates counts according to provider chunking.
- Store every chunk or usage sample: unnecessary and unbounded.
- Infer child identity from prompts or adapter call order: overlapping calls make attribution ambiguous.
- Add another UI timer or repaint per chunk: duplicates existing cadence and wastes work.

