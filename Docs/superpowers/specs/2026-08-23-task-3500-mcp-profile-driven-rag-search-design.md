# TASK-3500: MCP profile-driven RAG search design

Date: 2026-08-23
Status: revised after second-pass audit; pending written-spec reapproval
Task: TASK-3500
Base: `origin/dev` at `2088b1bb0`

## Problem

Library RAG search follows the active profile's `plain`, `semantic`, or
`hybrid` search mode and carries score provenance through fusion and
reranking. MCP `perform_rag_search` instead constructs a separate RAG service
eagerly and treats `use_semantic=True` as a hardcoded semantic route. The two
surfaces can therefore use different runtime configuration, retrieval modes,
and match-strength semantics for the same query.

TASK-3500 aligns the MCP tool without routing it through Library's app-owned
four-source search service. MCP remains a media-search tool and retains its
existing request and response schema.

## Goals

- Make the default MCP RAG request follow the active profile's search mode.
- Reuse the process-wide RAG runtime used by ingestion and Library search.
- Keep semantic and hybrid MCP retrieval confined to media.
- Preserve fusion, reranker, and vector score provenance for the MCP
  inspector's existing weak-match interpretation.
- Preserve existing callers, including the legacy explicit keyword override.
- Avoid loading embedding or reranking machinery for a `plain` profile.

## Non-goals

- Do not route MCP through `LibraryLocalRagSearchService`; that would broaden
  MCP retrieval to notes, conversations, and prompts and introduce app-state
  coupling.
- Do not change the public `search_rag` parameter names, types, defaults, or
  top-level result keys.
- Do not add a second MCP-specific reranker, fusion implementation, score
  vocabulary, or service cache.
- Do not add Library-style source-coverage diagnostics. TASK-2540 owns that
  separate MCP coverage-note gap.
- Do not tune retrieval thresholds, fusion weights, models, or result depth.

## Compatibility contract

The public `use_semantic: bool = True` field remains in both standalone and
local MCP schemas.

| Request value | Behavior |
| --- | --- |
| omitted | Follow the active profile |
| `true` | Follow the active profile |
| `false` | Explicitly use the existing media keyword search |

This deliberately changes the meaning of `true` from "force semantic" to
"enable profile-driven RAG." The legacy opt-out remains stable, and callers do
not need to send a new field.

The response remains a list whose successful rows have exactly these
top-level keys:

- `id`
- `title`
- `content`
- `media_type`
- `source`
- `score`
- `metadata`

Score provenance stays inside `metadata`; no new top-level response field is
required.

## Architecture

### One mode rule

Add a small shared mode normalizer under
`RAG_Search/simplified/active_config.py`. It accepts only `plain`, `semantic`,
and `hybrid`; missing or unknown values resolve to `semantic`, preserving the
historical behavior.

Add a lightweight `resolve_active_rag_search_mode()` sibling to
`resolve_active_rag_top_k()`. It reads the active profile's stored
`default_search_mode`, applies the documented `RAG_SEARCH_MODE` environment
override, and normalizes the result without resolving the full RAG config or
probing optional model dependencies. Library's existing private mode resolver
delegates normalization to the same helper so MCP and Library cannot drift on
unknown-value handling.

### Lazy shared runtime

`SimplifiedRAGSearchService.__init__` stores the injected `MediaDatabase` and
sets `rag_service = None` to preserve the adapter's existing attribute shape.
It no longer reads the obsolete `rag_search.service.profile` section and no
longer calls `create_rag_service()`.

For semantic or hybrid requests, the adapter obtains
`get_shared_rag_service()` through `asyncio.to_thread()`. It does not cache the
resolved service locally. The shared singleton already owns construction,
generation checks, and invalidation on profile switches; a second MCP cache
would be stale after a reset and would duplicate lifecycle ownership.

The existing `rag_service` attribute remains only as a non-production
injection seam for focused tests. A normally constructed adapter leaves it
`None`; every real enhanced request consults the shared resolver.

### Routing and data flow

`MCPTools.perform_rag_search` keeps the explicit `use_semantic=False` branch.
The default/true branch calls a new profile-aware adapter method:

1. Resolve the active mode without constructing the enhanced runtime.
2. For `plain`, call the existing `keyword_search` directly.
3. For `semantic`, lazily resolve the shared runtime and call its
   `search(search_type="semantic")` path.
4. For `hybrid`, lazily resolve the same runtime and call its
   `search(search_type="hybrid")` path.
5. If the enhanced runtime is unavailable, retain the existing keyword
   fallback. Keyword rows remain unscored (`score is None`) rather than
   fabricating similarity.

The adapter's existing explicit `semantic_search` method remains semantic for
its internal/test callers, but it uses the same lazy runtime resolver and
shared result formatter. A single private enhanced-search helper handles
semantic and hybrid invocation and formatting; there is no duplicate format
loop.

### Media confinement

The shared vector collection can contain media, notes, and conversations.
Restricting only hybrid's keyword leg would still allow non-media vector rows
into MCP results. Both semantic and hybrid MCP calls therefore pass the
engine's existing metadata allowlist in the exact shape
`{"source_type": ("media",)}`. The engine applies this restriction to the
vector and FTS legs and skips unrelated FTS sub-legs fail-closed.

The existing `media_types` request continues to produce its current
`filter_metadata={"media_type": {"$in": media_types}}` constraint, composed
with the media source allowlist. The audit found that the engine currently
documents and implements equality-only metadata matching, so this already-
emitted `$in` shape rejects every enhanced row instead of filtering it. Close
that bug once in `RAGService` with one private value-matching helper used by
the existing semantic and keyword post-filter sites. The helper preserves
exact equality and adds only the already-used single-key `$in` membership
form, failing closed on malformed `$in` values; it does not introduce a
general query language. This keeps semantic and both hybrid legs consistent
for single- and multi-value media filters.

Enhanced retrieval treats the configured shared index and its configured
media database path as authoritative. Explicit keyword mode uses the injected
`MediaDatabase`. Production MCP construction obtains that database from the
same authoritative configuration resolver. The adapter must not mutate the
shared runtime's config to accommodate a custom test database; tests that need
a different enhanced corpus inject a runtime or configure the shared path.

## Reranking and degradation

The shared factory already constructs `EnhancedRAGServiceV2` from the active
`ProfileConfig`, including its reranking configuration. MCP adds no reranker
logic. Successful reranking and per-call failures therefore behave exactly as
they do for Library search, and the adapter preserves the returned metadata.

One shared-runtime honesty gap must be closed at its owner: if reranking is
enabled but reranker construction fails, the service currently logs the
failure and silently continues with `self.reranker = None`. The V2 service
will retain a credential-safe unavailability detail containing only the
exception type and, when more than one base result exists (the same condition
under which reranking would run), tag the first result with the existing
`reranking_skipped` metadata key. The setup state is cleared before
construction and profile-switch attempts so a
later successful or disabled profile cannot inherit a stale reranker or
failure reason. This minimal shared fix gives both Library and MCP the same
disclosure; the MCP adapter must not synthesize its own tag.

If retrieval itself raises, the exception continues to reach
`perform_rag_search`, whose existing compatibility error shape is
`[{"error": "..."}]`. Search failures are not converted into false empty
results.

## Score semantics and inspector behavior

Enhanced result formatting preserves the complete result metadata block.
That block already carries:

- no special marker for ordinary vector similarity;
- `hybrid_fusion` with the preserved vector-leg score for fused results;
- `rerank_score` or `_final_score_kind` when a reranker replaced the score;
- `reranking_skipped` / `reranking_degraded` when reranking did not fully
  succeed, without pretending the unchanged base score is a reranker score.

`mcp_inspector._ScoredRow` gains `score_kind` and `vector_score` slots alongside
`score`. `_extract_scored_rows` passes the nested `row["metadata"]` mapping
first and the top-level row second to the existing
`library_rag_result_score_kind()` helper. The helper intentionally does not
recurse into nested mappings, so passing only the complete outer row would
silently default every fused and reranked MCP result to vector similarity.
The inspector then continues to call `library_rag_all_matches_weak()`:

- vector scores use the existing similarity bands;
- hybrid rows use their preserved vector leg when present;
- FTS-only hybrid rows do not fabricate a similarity;
- reranker scores never use cosine-similarity thresholds;
- unscored keyword rows do not trigger an all-weak claim.

This reuses the existing Library score vocabulary rather than maintaining an
MCP copy.

## Documentation

Update the MCP tool documentation and argument descriptions so
`use_semantic` accurately states the compatibility rule: false forces keyword
search; true or omission follows the active RAG profile.

## Testing strategy

Focused tests will pin:

- stored and environment-overridden active modes, including unknown-to-
  semantic normalization;
- constructor purity: MCP adapter construction does not build a RAG runtime;
- `plain` routing never resolves the enhanced runtime;
- `semantic` and `hybrid` routing call the shared service with the correct
  search type, exact media allowlist, media-type filter, and limit;
- the existing `$in` media-type filter matches allowed values in both the
  semantic and hybrid keyword result processors while exact equality remains
  unchanged;
- each enhanced request consults the shared resolver, so a profile reset does
  not leave an MCP-local stale service;
- `use_semantic=False` remains the explicit keyword override;
- existing exact request and response shapes remain unchanged;
- reranking metadata survives the MCP formatter, and constructor/runtime
  reranker failures return multi-result base results with the shared skip
  disclosure without leaving stale setup state after a profile switch;
- inspector interpretation for vector, hybrid-with-vector, FTS-only hybrid,
  reranker, and unscored keyword rows, with provenance read from nested MCP
  metadata;
- focused mutation checks: changing each mode arm, removing the media
  allowlist, or restoring blind vector defaults must make its owning test
  fail.

Local targeted tests, formatting, lint, and diff checks provide verification.
Repository CI checks are not part of this workstream per user direction.

## ADR check

ADR required: yes.

ADR path: `backlog/decisions/084-mcp-profile-driven-rag-search-contract.md`

Reason: TASK-3500 changes the lasting public meaning of `use_semantic=True`
and makes MCP a profile-driven consumer of the process-wide shared RAG runtime.
ADR-084 records that service contract, cross-module runtime boundary, media
confinement, and shared score/degradation provenance. ADR-003 remains the
accepted Settings/defaults boundary that this decision amends.

## Rejected alternatives

### Route MCP through Library search

Rejected because Library search owns four source seams and application state.
It would broaden MCP's media contract and create a standalone-server dependency
on the UI/application layer.

### Build another profile-configured MCP runtime

Rejected because it would duplicate model, vector-store, profile-switch, and
reranker lifecycle ownership. It is also the source of the current eager-load
and stale-profile hazards.

### Reimplement fusion, reranking, or score bands in MCP

Rejected because all three already exist at shared seams. A second copy would
increase code and recreate the drift TASK-3500 exists to remove.
