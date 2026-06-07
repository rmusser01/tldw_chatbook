# ADR 003: Settings Library/RAG Defaults Boundary

Status: Accepted
Date: 2026-06-07
Related Task: [backlog/tasks/task-79 - Functionalize-Settings-Library-and-RAG-defaults.md](../tasks/task-79%20-%20Functionalize-Settings-Library-and-RAG-defaults.md)
Supersedes: N/A

## Decision

Settings may mutate only global Library/RAG defaults that belong to the application configuration boundary. These defaults live under the existing `AppRAGSearchConfig.rag` mapping:

- `rag.search.default_search_mode`
- `rag.search.default_top_k`
- `rag.search.score_threshold`
- `rag.search.include_citations`
- `rag.search.citation_style`
- `rag.search.snippet_max_chars`
- `rag.search.max_context_size`
- `rag.retriever.fts_top_k`
- `rag.retriever.vector_top_k`
- `rag.retriever.hybrid_alpha`

Library remains the owner of active search/query execution, source browsing, source selection, and result display. Console remains the owner of staged context and agentic chat execution. RAG indexing, embedding model lifecycle, chunking templates, collection management, and workspace eligibility remain outside this Settings slice.

## Context

The current Settings `Library & RAG` category is a read-only ownership contract. Users can see that Library and RAG exist, but they cannot configure basic search/RAG defaults from the application configuration hub.

The RAG service already reads most `AppRAGSearchConfig.rag.search` and `AppRAGSearchConfig.rag.retriever` values through `RAG_Search.simplified.config.RAGConfig.from_settings()`. `citation_style` and `snippet_max_chars` are new display defaults added under the existing `rag.search` boundary; they must be represented in the RAG config model before Settings exposes them. `max_context_size` already exists on the simplified RAG search config, but the implementation task must verify that it is loaded from persisted settings before treating it as user-editable. Library-native Search/RAG already renders snippets and citation labels, so Settings can expose durable defaults without inventing a second runtime owner.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Keep Library/RAG read-only in Settings | This preserves ownership clarity but leaves Settings incomplete for a major app configuration domain. |
| Add full RAG indexing, embeddings, and chunk-template management to Settings | This would flatten workflow ownership into Settings and produce a large, hard-to-review PR. Those areas need dedicated Library/RAG or embeddings tasks. |
| Store Library/RAG Settings in a new top-level section | A new section would duplicate existing `AppRAGSearchConfig.rag` state and increase the risk that Library, RAG, and Settings diverge. |
| Let Settings edit active Library query state | Active query state is workflow state, not a persisted default. Library owns query execution and selected results. |

## Consequences

The first functional Library/RAG Settings PR stays narrow: it edits global defaults only. It must update Settings, Library/RAG display copy, and focused tests so users can understand what changed and what remains owned elsewhere.

Future RAG indexing, embedding lifecycle, collection, workspace eligibility, and citation/snippet carry-through work can build on this boundary without treating Settings as the runtime execution surface.

## Links

- [Implementation plan](../../Docs/superpowers/plans/2026-06-07-settings-library-rag-defaults.md)
- [Backlog task TASK-79](../tasks/task-79%20-%20Functionalize-Settings-Library-and-RAG-defaults.md)
- [Settings configuration hub design](../../Docs/superpowers/specs/2026-05-24-settings-configuration-hub-design.md)
