# RAG Server-Port Programme — Hybrid-Fusion Defect Cluster (design)

Date: 2026-08-09
Status: approved-pending-user-review
Programme: RAG server-port (P0 merged #1428; P1 eval harness merged #1458)
Arc: defect cluster found by the P1 harness's first run — TASK-3994, TASK-3995,
TASK-3996, plus TASK-3998 (gate trustworthiness). One branch, one PR, measured
by the P1 gate.

## Background

The P1 eval harness's first run over the fixture corpus proved (task records
carry the verified mechanisms and probe evidence):

- **TASK-3994 (high):** hybrid never fuses. `RAGService._hybrid_search` fuses
  on `key=lambda r: r.id`, but the FTS leg emits doc-level ids (`media_15`)
  while the vector leg emits chunk ids (`media_15_chunk_0`) — the legs never
  match, an unfused FTS row cannot crack the top-10 (crossover at vector rank
  ~82 with alpha 0.7 / rrf_k 60), and hybrid ≡ semantic on 44/44 golden
  queries.
- **TASK-3995 (high):** `_escape_fts5_query` wraps the whole query in one
  pair of quotes, making every multi-token search an FTS5 *phrase* query
  (contiguous text required). The quoting is load-bearing injection safety —
  raw `Obsidian-3` raises `OperationalError('no such column: 3')`.
- **TASK-3996:** the engine's FTS leg joins `media_fts` only — notes and
  conversations are structurally unreachable in hybrid's keyword leg (28/48
  fixture docs invisible to it).
- **TASK-3998 (high, gate trustworthiness):** the baseline fingerprint
  records `sentence-transformers` (not on the load path) and omits the
  packages that produce the numbers (transformers, torch, chromadb).

Out of scope, declared: TASK-3997 (AND-vs-OR retrieval strictness — a filed
product investigation, not a defect); TASK-3999 (report persistence, low);
`pipeline_builder_simple.py`'s twin RRF merge (legacy path, TASK-3501).

## The measuring discipline (arc-wide)

The P1 harness is the acceptance instrument. Sequencing:

1. **TASK-3998 lands first** with its own same-commit re-stamp, so every
   later gate verdict is meaningful.
2. Each fix lands with RED-first unit tests; the gated harness run after each
   fix is recorded as **informational evidence** (per-category deltas in the
   task reports). The gate MAY fail mid-arc — hybrid cells can legitimately
   dip as FTS docs displace vector docs — that is the instrument working,
   not a defect.
3. **One deliberate final re-stamp** in the arc's closing task captures the
   cumulative honest numbers; the PR carries the per-fix progression table
   (P1 baseline → post-3995 → post-3994 → post-3996 → stamped).

Expected story: hybrid's keyword-exact category climbs from 0.135 (= today's
semantic, fusion never having happened) toward plain's 0.867 territory while
paraphrase stays at vector strength — the first measured proof hybrid earns
its name.

## Fix designs

### TASK-3998 — fingerprint the load-bearing stack

- Compared fingerprint keys become: embedding model id, **transformers**,
  **torch**, **chromadb** versions, corpus_sha256, platform.
- `sentence-transformers` moves to NON-compared stamp metadata
  (informational) — keeping it compared would reintroduce the spurious
  environment-changed re-stamp its own filing describes.
- Baselines re-stamped in the same commit, both fingerprints shown (its AC).

### TASK-3995 — per-token quoting in `_escape_fts5_query`

- Tokenize on whitespace; quote EACH token individually (embedded quotes
  doubled — per-token injection safety preserved); join with FTS5's implicit
  AND. Multi-token queries now match non-contiguous tokens; single-token
  behavior unchanged.
- Edge: a query that is empty after tokenization (all punctuation /
  whitespace) short-circuits to a no-match return — never an empty MATCH
  expression (FTS5 syntax error). Pinned by test.
- This makes the engine consistent with the Library seam's own AND grammar;
  TASK-3997's AND-vs-OR product question is unchanged by it.

### TASK-3994 — fuse on document identity

- The fusion key changes from `r.id` to document identity:
  `(metadata["source_type"], metadata["source_id"] or metadata["doc_id"])`,
  falling back to `r.id` when either component is absent — per-row fail-open
  to today's behavior, never a crash.
- **Fallback pinned, not assumed:** rows lacking the metadata (foreign
  indexes, direct `index_document` callers) keep today's no-merge behavior;
  a dedicated test pins the fallback so it cannot silently become the main
  path.
- Within-leg chunk collapse: `_leg_ranks` keeps the first occurrence per
  key (code-verified: `if k not in ranks` — earliest/best rank wins on
  dupes), so a document's best chunk carries its fusion rank (doc-level
  fusion, matching the harness's doc-level metrics). The winning chunk row
  is what displays.
- **Fusion-key vocabulary equality (silent-failure guard):** the key
  compares raw `source_type` strings, so cross-leg merging requires the
  EXACT ingestion vocabulary — `media` / `note` / `conversation`, singular.
  Every sub-leg (existing media, new notes/conversations) stamps those
  values, and a cross-leg merge test per source type pins it — a plural or
  variant spelling would leave rows present but never merging, silently
  reverting TASK-3996's purpose.
- **Display preference for genuinely-merged rows (never-run path goes
  live):** `FusedResult.item` historically prefers `fts_item` — dead code
  until this fix, because the legs never matched. Merged rows now prefer the
  **vector item** for display (the semantically-matched chunk, real
  similarity score, chunk metadata); the FTS contribution stays recorded in
  `hybrid_fusion.fts_score` / ranks. This decision is explicit and tested.
  P0's leg-score capture already runs before any mutation, so either
  preference is aliasing-safe, and P0's hand-built fusion tests (matching
  ids, empty metadata) remain green through the fallback path — protected
  oracles unmodified.
- **Second never-run path going live — cross-leg citation merge:**
  `_fuse_hybrid_results` contains an `include_citations and fts_item and
  vector_item` branch that combines both legs' citations; it has NEVER
  executed (the legs never matched). Post-3994 it runs with rows it was
  never exercised against: doc-level keyword citations merged with
  chunk-level vector citations. Its first-run behavior gets its own test
  (merged rows carry both legs' citations, no duplication, no crash) before
  the fix is considered done.
- P0's `hybrid_fusion.vector_score`/`fts_score` metadata now populates for
  genuinely merged rows; the score-kind bands improve with no further work.

### TASK-3996 — FTS leg covers notes + conversations

- Config-injected, path-validated `chachanotes_db_path` (mirrors P0's
  `media_db_path` precedent, same validation treatment).
- **Read-only raw connection, not the ORM:** the two chacha queries run over
  `sqlite3.connect("file:…?mode=ro", uri=True)` — structurally incapable of
  writing, no schema-touch, no client registration. (`CharactersRAGDB`'s
  constructor runs schema checks on open; the engine's search path must not
  open the app's main DB through it.) The SQL mirrors what `search_notes` /
  `search_conversations_by_content` already run — read them for the exact
  FTS table names and ranking expressions.
- The engine leg becomes three sub-queries (media as today + notes +
  conversation messages), each row stamped `source_type`/`source_id`/
  `title`, interleaved rank-fairly into one best-first FTS leg via fusion.py's
  existing `interleave_rankings`.
- A missing/unopenable DB degrades that sub-leg with a logged reason — never
  a write, never a crash; the other sub-legs still run.

## Error handling

- All P0 degradation rules unchanged (keyword-leg failure → semantic with a
  disclosed coverage note, etc.).
- New: per-sub-leg degradation in the FTS leg (one DB missing ≠ leg dead);
  the empty-after-tokenization short-circuit.

## Testing

- RED-first unit tests per fix; mutation checks on the fusion key function
  and the display-preference rule (each dropped → only its test reds).
- The gated harness run after each fix (informational mid-arc; final re-stamp
  deliberate) — per-category numbers pasted into task reports.
- P0's fusion/band tests must pass UNMODIFIED (protected oracles — the
  fallback path preserves their fixtures' behavior).
- Collection arithmetic vs merge-base; final whole-branch review.
- Live TUI check in the closing task: Library rag mode on the hybrid profile
  over real data — the first time hybrid visibly differs from semantic for a
  user; capture evidence (scratch profile, copied DBs, PR-2 recipe).

## Plan-phase verification items

1. Exact FTS table names/rank expressions in ChaChaNotes for notes and
   messages (read `search_notes` / `search_conversations_by_content`), and
   whether conversation matches should surface per-message or per-conversation
   rows (the Library seam's convention decides).
2. `FusedResult.item` consumers beyond `_fuse_hybrid_results` (grep) — the
   display-preference change must not surprise another caller.
3. The real chacha DB filename/path resolver (config.py) for the injected
   default, mirroring `get_media_db_path`.
4. Whether `_process_keyword_results_with_citations` needs sub-leg-specific
   citation handling for note/conversation rows or degrades gracefully —
   AND the exact merge body of the cross-leg citation branch (read it fully;
   its docstring predates any real execution).
5. How the harness's canonicalization treats the new note/conversation
   keyword rows (should already work via stamped provenance — verify, since
   the gate's numbers depend on it).
