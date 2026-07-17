# ADR-005: Invest in Local RAG, Mirroring the tldw_server Pipeline

Status: Accepted
Date: 2026-07-17
Related Task: [backlog/tasks/task-246 - Default-RAG-vector-store-to-persistent-ChromaDB-when-embeddings-deps-are-installed.md](../tasks/task-246%20-%20Default-RAG-vector-store-to-persistent-ChromaDB-when-embeddings-deps-are-installed.md)
Supersedes: N/A

## Decision

tldw_chatbook invests in a fully functional **local** semantic RAG stack, copying the tldw_server RAG pipeline design as closely as the TUI context allows. The alternative flagged in the 2026-07-12 RAG module audit — shipping honest FTS-only search and routing "RAG Answer" through the tldw server API — is rejected as the primary direction.

## Context

The RAG module audit (`backlog/docs/rag-module-audit-2026-07-12.md`, PR #610) found that every user-triggerable retrieval resolves to SQLite FTS5: nothing indexes into the vector store, the store defaults to in-memory, and only the chat sidebar initializes the RAG runtime. The audit left one product decision open: invest in local semantic RAG, or remove the pretense and depend on the server. The owner decided on 2026-07-17 to invest in local RAG.

## Target design (from the server reference)

The tldw_server invariants to mirror, per the audit's divergence table:

- Retrieval modes `fts` / `vector` / `hybrid`; hybrid fused via **RRF (k=60) + alpha-weighted blend, alpha=0.7 vector-weighted** (task-256).
- **Persistent ChromaDB** as the default vector store, with per-collection conventions and a single store shared by every embedding producer (tasks 246, 248).
- **Indexing at ingestion time** (chunk → embed → upsert) via a non-blocking background worker, plus bulk backfill (task-247).
- Default embedding model `all-MiniLM-L6-v2` @ 384 dims; reranking on by default where the runtime supports it.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Honest FTS-only + server-routed RAG Answer | Requires a running tldw_server and network access for any semantic retrieval; the chatbook is frequently used standalone/offline. |
| Keep the current state (semantic UI over an empty store) | Dishonest UX; the audit exists precisely because this misleads users and reviewers. |

## Consequences

Backlog tasks 246 → 247 (foundations), then 248, 249, 250, 256 proceed as sequenced in the audit. Cleanup tasks 251–255 proceed independently; removal of the legacy manual-embeddings UI (task-253) stands, since indexing moves to ingestion time and any future manual surface must be rebuilt Console-parity. The `embeddings_rag` optional-dependency boundary stays: without the extra, the app remains FTS-only and must say so honestly (task-250).
