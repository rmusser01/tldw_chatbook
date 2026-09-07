# TASK-985 report

Worktree: `/Users/macbook-dev/Documents/GitHub/wt-985`, branch `fix/mcp-search-and-rag-resource`.

## Review findings

Four automated-reviewer findings on PR #1024. Investigated each against repo convention before acting;
two were declined with evidence rather than implemented.

### 1. "F-string SQL in `get_chunk_by_uuid`" — accepted (cosmetic only), security framing declined

The finding's SQL-injection framing is wrong: `target_table` was a hardcoded local literal
(`"UnvectorizedMediaChunks"`), the only user-supplied value (`chunk_uuid`) was already parameterized with
`?`, and the immediately preceding sibling `get_chunk_text` builds its query with the exact same
`f"...{target_table}..."` shape. There was no vulnerability.

What the f-string *did* do pointlessly was interpolate a constant, which is what trips scanners that flag
any f-string touching a `SELECT`. Fixed by inlining the literal table name and converting the query to a
plain (non-f) string, keeping the parameterized `?` for `chunk_uuid` untouched. Left the sibling
`get_chunk_text` alone — matching it would not have been a one-line change (it interpolates the table name
differently, on a single line) and the task said not to touch it unless the change was trivially safe.

### 2. "`get_chunk_by_uuid` lacks `transaction()`" — declined

This is a read, and every read helper in `Client_Media_DB_v2.py` — including the direct sibling
`get_chunk_text`, plus `get_specific_transcript`, `get_latest_transcription`, `get_specific_analysis`,
`get_media_prompts`, `get_unprocessed_media` — calls `db_instance.execute_query(...)` directly with no
`transaction()` wrapper. `transaction()` in this module is reserved for multi-statement writes needing
atomicity/rollback, not single-statement reads. `get_chunk_by_uuid` matches the established convention
exactly. No change made.

### 3. "`_seed_chunk` bypasses `transaction()`" (Tests/MCP/test_tools_resources_prompts_real_methods.py) — declined

Matches the sibling `_seed_transcript` helper in the same test file (same file, same pattern: direct
`media_db.execute_query(INSERT ..., commit=True)`, both with a docstring explaining that no public writer
API exists to seed a row with a caller-known UUID). It also matches the broader convention in
`Tests/Media_DB/test_media_db_v2.py`, which uses plain `execute_query(..., commit=True)` for fixture-row
inserts throughout and reserves `db.transaction()` specifically for tests asserting atomicity/rollback
behavior (e.g. its `Keywords` count-after-rollback test). No change made.

### 4. "`search_conversations` inputs unvalidated" (tldw_chatbook/MCP/tools.py) — accepted

`search_conversations` took a free-text `query` and integer `limit` straight from the MCP boundary with
zero validation, then handed `limit` directly into `search_conversations_by_content`'s `LIMIT ?` — a
negative value there is not just "no results," it's SQLite's `LIMIT -1` meaning *unbounded*, returning
every matching row.

Checked repo convention first: no MCP tool in `tools.py` (`chat_with_character`, `perform_rag_search`,
`get_conversation_history`, `export_conversation`) validates its inputs via `Utils/input_validation.py`
today, so this is a real, if scoped, gap rather than an established pattern this file already follows. But
the *same shape* of input — free-text search query plus a numeric result limit — is already validated
elsewhere in the codebase: `Library/library_rag_state.py`'s Library RAG search entry point uses
`validate_text_input` (bounded length, `allow_html=False`) on the query and `validate_number_range` on the
`top_k` limit. Brought `search_conversations` in line with that existing convention rather than closing the
whole file's gap: added `validate_text_input`/`validate_number_range` checks (query: non-empty, plain text,
≤2000 chars; limit: 1–100), returning the same `[{"error": ...}]` shape the function already used for
downstream exceptions. Deliberately left `perform_rag_search` (the sibling with the identical gap)
untouched — noted here rather than silently expanding scope; it did not use validation before this task,
and still doesn't. A follow-up task should be filed if that gap should also be closed.

## What changed

- `tldw_chatbook/DB/Client_Media_DB_v2.py` — `get_chunk_by_uuid`: dropped the pointless f-string, table name
  now inlined into a plain string.
- `tldw_chatbook/MCP/tools.py` — `search_conversations`: added `query`/`limit` validation via
  `Utils/input_validation.validate_text_input`/`validate_number_range`, new module-level
  `MAX_SEARCH_QUERY_LENGTH` (2000) / `MAX_SEARCH_RESULTS_LIMIT` (100) constants.

## Verification

- `Tests/MCP/` — 381 passed.
- `Tests/Media_DB/test_media_db_v2.py` — 38 passed.
- Rebased onto `origin/dev` (`1f00c19ea`) cleanly — no conflicts; upstream had not touched any of the
  files this branch modifies.

Commits: `fd5a5102d` (rewire, pre-existing), `a495d8045` (docs, pre-existing), `19548a1a6` (this review-fix
commit, post-rebase SHA).
