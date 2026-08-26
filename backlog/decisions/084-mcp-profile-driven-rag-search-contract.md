# ADR-084: MCP profile-driven RAG search contract

Status: Accepted
Date: 2026-08-23
Related Task: [TASK-3500 - Align MCP perform_rag_search with profile-driven retrieval](../tasks/task-3500%20-%20Align-MCP-perform_rag_search-with-profile-driven-retrieval.md)
Amends: [ADR-003](003-settings-library-rag-defaults.md) by making the active
RAG profile an execution default for MCP media search as well as Library search

## Decision

MCP `search_rag` remains a media-search contract, but its default retrieval
behavior follows the active RAG profile rather than a separately constructed,
hardcoded semantic service.

The existing public `use_semantic: bool = True` request field remains unchanged:

- omitted or `true` follows the active profile's `plain`, `semantic`, or
  `hybrid` search mode;
- `false` remains an explicit media keyword-search override.

This is a deliberate compatible reinterpretation of `true`: it enables the
profile-driven RAG path instead of forcing semantic retrieval. No replacement
mode field or response-schema version is introduced.

Plain search uses the existing media keyword path and must not construct the
enhanced RAG runtime. Semantic and hybrid search resolve the process-wide shared
RAG service at request time. MCP does not own another enhanced service instance,
cache, profile generation, fusion implementation, or reranker.

The shared index may contain several Library source types, so every MCP semantic
or hybrid request applies the engine's existing `source_type=media` allowlist to
all retrieval legs. The existing media-type filter composes with that allowlist.
MCP must not route through the application-owned multi-source Library search
service or broaden its results to notes, conversations, or prompts.

Successful result rows retain the current top-level response keys. Complete
score provenance remains in each row's `metadata` and uses the shared Library
score-kind vocabulary. Vector similarity, hybrid fusion, preserved vector-leg
score, and reranker score are interpreted according to their actual scoring
path; keyword and FTS-only rows do not fabricate vector similarity.

Reranking is owned by the shared runtime. If reranker construction or execution
is unavailable, base retrieval results remain usable and carry the shared
`reranking_skipped` or `reranking_degraded` disclosure. MCP does not synthesize
a separate degradation marker. If the enhanced runtime itself is unavailable,
the existing keyword fallback remains available and its rows remain unscored.

## Context

ADR-003 placed durable RAG defaults under the application configuration boundary
and kept active search execution out of Settings. Library subsequently began
resolving its execution mode from the active profile and using the shared RAG
runtime. MCP `perform_rag_search` still eagerly constructed a separate service
and treated `use_semantic=True` as a hardcoded semantic route.

That divergence made the same profile and query select different retrieval
modes, model lifecycles, reranking behavior, and score interpretation depending
on whether the caller used Library or MCP. Simply sharing the runtime is not
sufficient because its derived index contains non-media records that are outside
the MCP tool's existing data contract.

This task changes a public request field's lasting meaning and establishes a
cross-module runtime and service boundary, so the contract belongs in a new ADR
rather than only in implementation notes or an amendment to accepted ADR-003.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Add a new `search_mode` request field | It would create two overlapping controls, require precedence rules, and expand the public API when the existing opt-out can remain compatible. |
| Keep `true` as forced semantic | Default MCP requests would continue to ignore plain and hybrid active profiles, preserving the parity bug. |
| Route MCP through Library's multi-source service | It would broaden a media-only MCP contract and couple standalone MCP execution to application-owned Library state. |
| Construct and cache a profile-aware MCP RAG service | It would duplicate shared runtime ownership and could remain stale after profile generation resets. |
| Filter only the hybrid keyword leg | Non-media records could still enter through vector retrieval. Confinement must apply at the engine allowlist used by every leg. |
| Recreate fusion, reranking, or score bands in MCP | Parallel implementations would drift and could again mislabel non-vector scores as vector similarity. |

## Consequences

Default MCP search changes when the active profile changes, while explicit
`use_semantic=False` callers retain keyword behavior. Plain profiles avoid model
startup cost. Enhanced requests share profile switches, service invalidation,
fusion, reranking, and degradation behavior with Library while remaining
media-confined.

MCP documentation and focused contract tests must explain and pin the boolean
compatibility rule. Test seams may inject a runtime, but production construction
must not create or retain an MCP-local enhanced service.

## Links

- [Approved design](../../Docs/superpowers/specs/2026-08-23-task-3500-mcp-profile-driven-rag-search-design.md)
- [ADR-003: Settings Library/RAG defaults boundary](003-settings-library-rag-defaults.md)
- [ADR-030: Local Library agent tool boundary](030-local-library-agent-tool-boundary.md)
- [TASK-3500](../tasks/task-3500%20-%20Align-MCP-perform_rag_search-with-profile-driven-retrieval.md)
