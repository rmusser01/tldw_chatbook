# Legacy per-user Chroma storage: orphaned data and reset path (task-248)

**Context:** Until task-248 the codebase carried two disconnected Chroma stacks
(2026-07-12 RAG module audit, [rag-module-audit-2026-07-12.md](rag-module-audit-2026-07-12.md) §"Second store,
invisible to search"). The legacy stack — `Embeddings/Chroma_Lib.py`'s
`ChromaDBManager`, written to by the manual embeddings UI (removed in PR #669 /
task-253) and by the RAG_Admin local collection surface — was removed in
task-248. The sole surviving stack is `RAG_Search/simplified/vector_store.py`'s
`ChromaVectorStore`, shared process-wide via
`RAG_Search/ingestion_indexing.get_shared_rag_service()` (PR #667 / task-247);
it is what both ingestion-time indexing writes and RAG semantic search reads.

## What data may exist on disk

The legacy stack persisted to a **per-user** directory:

```
<USER_DB_BASE_DIR>/<user_id>/chroma_storage
# default: ~/.local/share/tldw_cli/<user_id>/chroma_storage
# e.g.:    ~/.local/share/tldw_cli/default_user/chroma_storage
```

The surviving stack persists to a different location with a different
collection convention:

```
~/.local/share/tldw_cli/chromadb        # RAG_Search default persist dir
```

Any `chroma_storage` directory left from the legacy stack is now **orphaned**:
no code reads or writes it. Embeddings stored there were never visible to RAG
search anyway (that was the bug this task fixes).

## Reset path (no automated migration)

There is deliberately **no automated migration** of legacy collections:
embeddings are a derived artifact — model- and dimension-dependent (legacy
collections may have been built with any user-configured model), and fully
regenerable from the source SQLite databases. Migrating vectors across
models/conventions would produce silently incomparable or wrong-dimension
entries.

To clean up and rebuild:

1. Delete the orphaned directory (safe; source content lives in the SQLite DBs):
   ```bash
   rm -rf ~/.local/share/tldw_cli/<user_id>/chroma_storage
   ```
2. (Re)build the unified semantic index from existing media/notes/conversations:
   ```bash
   python -m tldw_chatbook.RAG_Search.backfill
   ```
   New/updated content is indexed automatically at ingestion time (task-247),
   so the backfill is only needed once for pre-existing content.
