# ADR-223: Persistent embedding content-hash cache; batched ingestion skip-checks

Status: Accepted
Date: 2026-10-06
Task: [TASK-34420](../tasks/task-34420%20-%20Persistent-embedding-content-hash-cache-ADR-213.md)
Plan: [Non-console efficiency remediation](../../Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation.md), Task 8 (F6)
Numbered 223 at creation: the brief's provisional 213 was only a placeholder; the highest canonical ADR at authoring time was [ADR-222](222-provider-http-session-reuse.md) (verified per `backlog/docs/lessons-backlog-hygiene.md`).

## Decision

Chunk embeddings become content-addressed and persistent: the text's
`sha256` (not the model's handle, not the item's ID) decides whether a
provider/model embed call is needed, and the answer survives process
restarts in the same SQLite database that already tracks RAG indexing
state. Alongside, the per-item "is it unchanged?" skip check stops doing
N+1 primary-key lookups per batch and does one batched read instead.

### 1. Cache key: `(model_id, sha256(text))`

- `content_hash` = `sha256(text.encode("utf-8")).hexdigest()` — the full
  text, not a preview: the deleted "cache check" hashed only the first
  100 characters of each text (`embeddings_wrapper.py` pre-ADR), which
  would have conflated distinct chunks sharing a 100-char prefix.
- `model_id` = the wrapper's embedding model name
  (`EmbeddingsServiceWrapper.model_name`, e.g.
  `sentence-transformers/all-MiniLM-L6-v2`). The factory's internal
  model id (`"default"`) is deliberately NOT used: it is a config slot,
  not a model identity, and would key different models to one cache row.
- Same text under a different model is a different row: embeddings are
  model-specific, so a model switch must re-embed (and a model switch
  also re-fingerprints the Chroma collection, which re-indexes
  everything — this cache is what keeps that re-index from re-paying
  the provider for texts it has already seen under the new model).
- Collisions: for a local cache, sha256 collision risk is negligible
  (≈ 2^-128 birthday bound at realistic corpus sizes); the worst case is
  one wrong cached vector, self-healing on the next text change.

### 2. Storage: `embedding_cache` table in RAG_Indexing_DB

```sql
CREATE TABLE IF NOT EXISTS embedding_cache (
    model_id     TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    vector       BLOB NOT NULL,
    created_at   TEXT NOT NULL,
    PRIMARY KEY (model_id, content_hash)
);
CREATE INDEX IF NOT EXISTS idx_embedding_cache_created
ON embedding_cache(created_at);
```

- Why this DB: it is already the per-user, user-data-dir SQLite file the
  ingestion pipeline opens (`get_rag_indexing_db_path()`), already
  thread-local-held-connection managed, and already semantically "local
  RAG state that is rebuildable from source" — exactly what a cache is.
- **No version bump — deliberate deviation from the brief's "bump the
  RAG DB schema version per migrations convention"**: verified at
  implementation time, `RAG_Indexing_DB.py` has **no schema-version
  chain at all** — no `PRAGMA user_version`, no entry under
  `DB/migrations/` (that directory only serves DBs with their own
  version constants, e.g. `chachanotes_v*`, `workspaces_v*`), and
  `_initialize_schema()` runs idempotent `CREATE TABLE IF NOT EXISTS`
  statements on every open. The task instruction "follow THAT DB's own
  convention exactly" therefore yields: add the table to the
  `_initialize_schema` script. The change is purely additive — an old
  binary reading a new file ignores the table; a new binary reading an
  old file creates the table on open — so no migration or backfill is
  possible or needed (a cache is derivable, never authoritative).
- `created_at` is an ISO-8601 UTC string, matching how `indexed_items`
  stores its datetimes.

### 3. Vector serialization

`vector` = the embedding as little-endian raw float32 bytes
(`array("f").tobytes()`, with an explicit byteswap guard so a big-endian
host still writes little-endian). On read, length must be a multiple of
4 or the row is treated as absent (logged, skipped — a corrupt row must
never crash indexing). No format marker byte: the float32/LE choice is
recorded here and in the codec functions, and every vector in this
table is written by that one codec.

There is no prior vector-blob convention in this DB to match — the
vectors themselves live in ChromaDB; this table is the only blob it
stores. stdlib `array` (not numpy) keeps the DB layer dependency-free.

### 4. API (`RAGIndexingDB`)

- `get_cached_embeddings(model_id, content_hashes) -> dict[str,
  list[float]]` — deduplicates the requested hashes and reads them in
  chunked `IN (...)` queries of at most **500** placeholders (SQLite's
  default host-parameter ceiling is 999; 500 stays under it with room).
  Empty input returns `{}` without touching SQLite.
- `store_cached_embeddings(model_id, rows)` where each row is
  `(content_hash, vector: Sequence[float])` — ONE `executemany` of
  `INSERT OR REPLACE` inside ONE transaction, so a batch lands or
  doesn't as a unit (an interrupted backfill stores nothing partial).
  Duplicate hashes within one call collapse (last write wins) before
  the write.

### 5. Eviction: row-count cap, pruned in the insert transaction

- Cap = `EMBEDDING_CACHE_MAX_ROWS = 200_000` rows, a module constant
  (overridable per `RAGIndexingDB(..., embedding_cache_max_rows=...)`
  construction for tests). Size math for the default: a 384-dim vector
  is 1.5 KB of blob + ~140 B of key/timestamp, so 200k rows ≈ 340 MB —
  a defensible ceiling for a user-data-dir cache on a machine that
  libraries hundreds of thousands of chunks; 768-dim models double it,
  which is when the cap is doing its job.
- Immediately after the inserts, **in the same transaction**, if
  `COUNT(*) > cap`, delete the overflow oldest-first:
  `ORDER BY created_at ASC, rowid ASC LIMIT overflow`. `rowid` breaks
  `created_at` ties (a whole batch shares one timestamp) so "oldest"
  is deterministic — within a tie, earliest-inserted goes first.
- FIFO by insertion time (not LRU): the cache is rebuilt bottom-up by
  full re-indexes, where the oldest rows are exactly the ones the next
  re-index will re-request first; LRU bookkeeping (a write per read)
  would add write amplification to a read path.

### 6. Wrapper wiring (`embeddings_wrapper.py`)

Both `create_embeddings` and `create_embeddings_async` — the two choke
points every chunk and query embedding already flows through — gain the
same flow:

1. compute `sha256` **once per text** in the batch;
2. one `get_cached_embeddings` lookup for the batch's unique hashes;
3. embed ONLY the misses, through the existing circuit-breaker path
   (sync `factory.embed` / async `_async_factory_embed` unchanged);
4. merge hits + fresh vectors back into the caller's original order
   (duplicate texts within a batch share one lookup and one embed);
5. one `store_cached_embeddings` write for the misses after success.

Cache failures (lookup or store) are logged and degraded to
"no cache for this call" — a cache can never fail an embed. The store
is attached explicitly, never implicitly:

- constructor: `EmbeddingsServiceWrapper(..., embedding_cache_db=db)`;
- runtime: `set_embedding_cache_store(db)`, called by the ingestion
  seam — `index_entries()` in `ingestion_indexing.py` attaches the
  caller's `indexing_db` to `service.embeddings` (best-effort,
  re-attachable). That seam is chosen because it is the ONE place that
  already owns a `RAGIndexingDB` handle (both the post-ingest worker
  and `backfill_semantic_index` route through it). The wrapper itself
  must NOT default-open `get_rag_indexing_db_path()`: the wrapper (and
  `RAGService`) is constructed all over the test suite without a DB
  context, and an implicit open would point tests at the real
  user-data dir, breaking hermeticity for no functional gain.

**Query-side included**: a query is just a text at the same choke
point, so once any ingestion run has attached the store to the shared
service wrapper (`get_shared_rag_service`), query embeddings consult
and populate the same table. That is the "trivial and safe" case the
task brief called out: no second code path, no new key semantics, and
a query hit saves exactly one provider call. If the store was never
attached (service used only for search before any indexing), behavior
is exactly today's.

### 7. Metric semantics: `embeddings_cache_hit_rate` becomes real

The deleted check compared a hash of the batch against
`EmbeddingFactory._cache` — a dict keyed by **model id** — so it could
never hit, and the gauge was permanently 0%. Now `_cache_hits` /
`_cache_misses` count **per text** (not per batch) from actual store
lookups, `cache_hit_rate` in `get_metrics()` is their true ratio, and
`embeddings_cache_hit` / `embeddings_cache_miss` counters carry the
per-text counts as their value. With no store attached the counters
stay at zero requests — the rate is honestly 0.0 over zero samples,
and no fake hit/miss counters are emitted.

### 8. `EmbeddingFactory._lock` stays (explicitly out of scope)

The factory's global lock serializes every embed call to make model
handle reuse safe (an in-use model cannot be LRU-evicted mid-call).
Redesigning it is deferred: this cache removes most *contending* calls
(a full re-embed becomes zero factory calls), so the lock is mostly
uncontended after this ADR. Lock redesign would be its own task with
its own correctness argument.

### 9. Batched skip-check (`ingestion_indexing.index_entries`)

The per-entry loop calling `indexing_db.needs_reindexing(...)` — one
PK SELECT (plus metrics logging) per entry — is replaced by ONE
`get_indexed_items_by_type(item_type)` read per distinct `item_type`
in the batch (the read `reconcile_media_index` already uses), then an
in-memory comparison. Semantics are preserved exactly:

- item not in the tracking table → index (as `needs_reindexing`
  returned True for unknown items);
- `current_modified > stored last_modified` → index; otherwise skip;
- a naive `current_modified` is stamped UTC before comparing (same
  normalization `needs_reindexing` applied);
- a per-entry comparison error (e.g. mixed aware/naive datetimes)
  indexes that entry with a warning — as the per-entry `except` did;
- a failed batch read falls back to indexing every entry of that type
  with a warning — the same "fail open to re-indexing" the per-entry
  path had, and safe because indexing is idempotent.

## Context

`RAG_Search/simplified/embeddings_wrapper.py` computed a batch cache
key and checked it against `EmbeddingFactory._cache`, which is keyed by
model id only (`Embeddings_Lib.py`), so the check could never hit —
dead code feeding a permanently-0% `embeddings_cache_hit_rate` metric.
No persistence existed anywhere: any re-index (restart, interrupted
backfill resumed, rebuilt Chroma persist dir, re-chunking pass) re-embedded
every chunk through the provider/model. For local HF models that is
CPU-minutes; for `openai/*` models it is real money and rate limits.
Independently, the ingestion skip check did one `needs_reindexing` PK
lookup per entry where the batch read it needed already existed.

## Alternatives

- **Rely on ChromaDB to avoid re-embedding**: Chroma has no
  content-hash lookup; deciding "already embedded" means reading the
  collection's vectors back, which assumes the collection survived —
  the exact scenario (fresh/rebuilt persist dir) where re-embed pain
  happens. Rejected.
- **In-process LRU only** (fix the dead check, no disk): cheaper, but
  fails the actual requirement — restarts and interrupted backfills
  are the pain cases. Rejected.
- **Key by `(model_id, item_id, chunk_index)`**: breaks the moment
  chunking changes (same text, new index) or the same text appears in
  two items; content addressing is stable under both. Rejected.
- **Evict by total bytes**: honest about disk use but needs
  `length(vector)` SUM scans per store; row count is a constant-time
  proxy (every vector of a model has one length in practice) with a
  documented worst case. Deferred until a real corpus disagrees.
- **Implicit default-open of the indexing DB inside the wrapper**:
  rejected for test hermeticity and duplicate-handle reasons (§6).
- **Touch `EmbeddingFactory._lock`**: out of scope (§8).

## Consequences

- An unchanged-content full re-ingest after a restart performs **zero**
  provider/model embed calls (pinned by a reopen-the-DB test and the
  task's evidence run: first pass 100 texts embedded, reopened second
  pass 0).
- Re-embedding the same text under the same model costs one sha256 +
  one indexed SQLite read instead of a provider call; the skip-check
  N+1 (one PK lookup + metric log per entry) collapses to one read per
  item type per batch.
- `embedding_cache` shares the indexing DB's file, so its size is
  bounded by §5 and it is wiped by `clear_all()` alongside the other
  rebuildable tracking state.
- The `embeddings_cache_hit_rate` gauge now moves; anyone monitoring it
  should treat post-ADR values as per-text store hit rates (pre-ADR it
  was constant 0 and meaningless).
- The factory lock remains a serializing point for genuine misses; a
  corpus change large enough to miss constantly still serializes — that
  is the deferred §8 problem, unchanged by this ADR.
