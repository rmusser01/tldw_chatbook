# Hybrid-Fusion Defect Cluster Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make hybrid retrieval genuinely fuse (TASK-3994), let multi-token keyword queries match non-contiguous text (TASK-3995), extend the engine's FTS leg to notes+conversations (TASK-3996), and make the eval gate's fingerprint trustworthy first (TASK-3998) — all measured by the P1 harness with one deliberate final re-stamp.

**Architecture:** Engine-local fixes in `RAG_Search/simplified/rag_service.py` + one key-function change feeding `RAG_Search/fusion.py`'s existing machinery; read-only raw SQLite sub-legs for chacha; harness/baseline changes confined to `Tests/RAG_Eval/harness/`. Spec is authority: `Docs/superpowers/specs/2026-08-09-rag-port-hybrid-fusion-fixes-design.md`.

**Tech Stack:** Python 3.11+, pytest, SQLite FTS5, the P1 eval harness (`RAG_EVAL=1`).

## Global Constraints

- Spec (authority — read first): `Docs/superpowers/specs/2026-08-09-rag-port-hybrid-fusion-fixes-design.md`. The four backlog task files (3994/3995/3996/3998) carry the verified mechanisms — read your task's file before coding.
- Worktree `.worktrees/rag-fusion-fixes`, branch `fix/rag-hybrid-fusion-cluster`. **cwd silently resets between Bash blocks — start EVERY block with `cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/rag-fusion-fixes`.**
- Tests: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <paths>` with worktree cwd. pytest only (never `python -c` app imports; app-importing probes live UNDER Tests/). "no tests ran" = FAILED gate; read numeric counts.
- The gated harness (`RAG_EVAL=1 pytest Tests/RAG_Eval/`) runs after Tasks 3/4/5 as **informational evidence** — the gate MAY fail mid-arc (hybrid cells moving is the point); paste the per-category table into the task report and do NOT re-stamp. Only Task 2 (3998's own) and Task 6 (final) re-stamp, each deliberately with printed deltas.
- Fusion-key vocabulary: cross-leg merging requires EXACT singular `source_type` values (`media` / `note` / `conversation`) — the ingestion contract's vocabulary. Any sub-leg stamping a variant silently reverts 3996.
- P0's fusion/band tests and P1's harness always-on tests pass UNMODIFIED (protected oracles). If a change forces editing one, STOP and report the collision.
- Never `git stash`; Edit-based restores with unique markers; push after every task; commits reference the task's backlog ID and end `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

## Verified code anchors (line numbers drift; grep first)

- Fusion call site: `rag_service.py` `_fuse_hybrid_results` → `reciprocal_rank_fusion(keyword_results, semantic_results, key=lambda r: r.id, alpha=..., rrf_k=DEFAULT_RRF_K, max_results=top_k)` (~L989 on dev). `entry.item` consumed at ONE site (~L980); `FusedResult.item` property (fusion.py) prefers fts_item ("server parity") — leave the property; choose the display item at the call site.
- `_leg_ranks` (fusion.py) keeps the FIRST occurrence per key (`if k not in ranks`) — doc-collapse promotes each doc's best chunk. Verified.
- Cross-leg citation merge: the `include_citations and entry.fts_item and entry.vector_item` branch in `_fuse_hybrid_results` has NEVER executed — read its full body before Task 4; its docstring predates any real run.
- `_escape_fts5_query` (~L1289): wraps the WHOLE query in one quoted phrase; embedded quotes doubled. Raw `Obsidian-3` unquoted raises `OperationalError('no such column: 3')` — per-token quoting must preserve that safety per token.
- Keyword-leg metadata (post-P0): media rows stamp `doc_id` (str), `source_type: "media"`, `source: "media"`, `title`. Semantic rows (ingestion contract): `source_id` (str), `source_type` singular, `title`, per-chunk `chunk_id`.
- Chacha FTS shapes (read the methods for full SQL): notes → `notes_fts` JOIN `notes` ON rowid, `MATCH ?` (title+content); conversations → `messages_fts` JOIN `messages` JOIN `conversations`, `GROUP BY c.id`, `MIN(rank) as best_rank`, `ORDER BY best_rank` — PER-CONVERSATION rows (source_id = conversation id, matching ingestion's conversation docs).
- Chacha path resolver: `tldw_chatbook.config.get_chachanotes_db_path()` (config.py ~L5748; default `~/.local/share/tldw_cli/tldw_chatbook_ChaChaNotes.db`). Mirror `media_db_path`'s validation treatment (P0 precedent in `_keyword_search`: `validate_path_simple`/`lexical_path`, degrade-to-default + warning, never raise).
- P1 harness: `Tests/RAG_Eval/harness/baseline_io.py` (`current_fingerprint`, `compare_or_update`, `GATED_METRIC_KEYS`), committed baselines `Tests/RAG_Eval/baselines/*.json`; `RAG_EVAL_UPDATE_BASELINES=1` re-stamps with printed deltas. Canonicalization keys on stamped (source_type, source_id) — new sub-leg rows work if stamped correctly.
- P1 baseline numbers to beat (hybrid ≡ semantic today): hybrid keyword P@10 0.135 / plain 0.867; hybrid paraphrase recall 1.000.

---

### Task 1: Backlog bookkeeping

**Files:** the four existing task files (via CLI).

- [ ] **Step 1:** `backlog task edit <id> -s "In Progress"` + `--plan` (one line referencing this plan + the spec) for 3994, 3995, 3996, 3998. Read each task file first — their ACs are the per-task contracts.
- [ ] **Step 2:** Commit `chore(backlog): fusion-cluster tasks 3994-3996+3998 in progress` (+ trailer), push.

---

### Task 2: TASK-3998 — fingerprint the load-bearing stack (+ its re-stamp)

**Files:**
- Modify: `Tests/RAG_Eval/harness/baseline_io.py` (`current_fingerprint` and whatever helper feeds it)
- Modify: `Tests/RAG_Eval/baselines/*.json` (re-stamped)
- Test: `Tests/RAG_Eval/test_baseline_io.py` (extend)

**Interfaces:**
- Produces: compared fingerprint keys = `model`, `transformers`, `torch`, `chromadb`, `corpus_sha256`, `platform`; `sentence_transformers` moves to NON-compared stamp metadata (informational). Version strings via `importlib.metadata.version(...)` with a `"absent"` fallback (never raise if a package is missing — the extras gate already ensures presence in gated runs).

- [ ] **Step 1: RED tests** (always-on, extend test_baseline_io.py): fingerprint contains the three new keys with non-empty values; `sentence_transformers` NOT in the compared dict but present in the stamp's informational metadata; a baseline whose environment lacks the new keys compares as `environment_changed` listing them.
- [ ] **Step 2:** RED → implement → GREEN (run test_baseline_io.py + the full ungated Tests/RAG_Eval/; read counts).
- [ ] **Step 3: Re-stamp** (`RAG_EVAL=1 RAG_EVAL_UPDATE_BASELINES=1 pytest Tests/RAG_Eval/`): metric values must print all-zero deltas (numbers unchanged — only the fingerprint moved); paste the printout + both fingerprints into the report. Then a plain gated run → gate PASSED.
- [ ] **Step 4:** Tick 3998's ACs, Implementation Notes, `-s Done`. Commit `fix(rag-eval): fingerprint the load-bearing embedding stack (TASK-3998)` (+ trailer), push.

---

### Task 3: TASK-3995 — per-token FTS quoting

**Files:**
- Modify: `tldw_chatbook/RAG_Search/simplified/rag_service.py` (`_escape_fts5_query`; the no-match short-circuit in `_perform_fts5_search` or its caller)
- Test: `Tests/RAG_Search/test_fts5_query_escaping.py` (create)

**Interfaces:**
- Produces: `_escape_fts5_query(query) -> str` returns per-token quoted terms joined by spaces (FTS5 implicit AND), e.g. `"Obsidian-3" "lathe" "spindle"`; returns `""` for queries empty after tokenization, and the keyword search returns `[]` (no FTS call) on `""`.

- [ ] **Step 1: RED tests** — against a real in-memory FTS5 table (build it in the test with 2-3 docs; sqlite3 stdlib, no app DB):
```python
def test_multi_token_query_matches_non_contiguous_tokens():
    # doc: "The Obsidian-3 lathe shows spindle runout under load."
    # query "Obsidian-3 spindle runout" (tokens present, NOT contiguous)
    # old phrase form -> 0 rows; new per-token form -> 1 row
def test_hyphen_numeric_token_still_safe():
    # raw unquoted "Obsidian-3" raises OperationalError('no such column: 3')
    # the escaped form must execute cleanly and match
def test_embedded_quotes_are_doubled_and_safe():
    # query with a double-quote character executes without error
def test_single_token_behavior_unchanged(): ...
def test_all_punctuation_query_short_circuits():
    # _escape_fts5_query('!!! ...') == "" and _keyword_search returns [] without querying
```
Write the bodies against the real function; assertions above are the contract.
- [ ] **Step 2:** RED (phrase form fails the non-contiguous test) → implement → GREEN. Run the new file + `Tests/RAG_Search/test_keyword_leg_db_resolution.py` (P0's leg tests must stay green — their single-token seeds are unaffected).
- [ ] **Step 3: Informational gated run** (`RAG_EVAL=1`, do NOT re-stamp): expect plain unchanged (Library seam has its own grammar), hybrid still ≡ semantic (fusion not yet fixed) BUT the engine keyword leg now returns rows for multi-token queries — if the gate trips or numbers move, record the table; do not chase.
- [ ] **Step 4:** Tick 3995's ACs, Notes, `-s Done`. Commit `fix(rag): per-token FTS5 quoting — multi-token queries match non-contiguous text (TASK-3995)`, push.

---

### Task 4: TASK-3994 — fuse on document identity

**Files:**
- Modify: `tldw_chatbook/RAG_Search/simplified/rag_service.py` (`_fuse_hybrid_results`: the key lambda, the display-item selection, and — after reading it — the citation-merge branch if its first run needs hardening)
- Test: `Tests/RAG_Search/test_hybrid_doc_fusion.py` (create); `Tests/RAG_Search/test_hybrid_fusion_metadata.py` must pass UNMODIFIED.

**Interfaces:**
- Produces: a module-level pure function in rag_service.py:
```python
def _fusion_doc_key(result: Any) -> Hashable:
    """Document-identity fusion key: (source_type, source_id-or-doc_id).

    Falls back to the row id when either component is missing, preserving
    the pre-fix no-merge behavior for rows without the ingestion metadata.
    """
    md = getattr(result, "metadata", None) or {}
    source_type = md.get("source_type")
    source_id = md.get("source_id") or md.get("doc_id")
    if source_type and source_id:
        return (str(source_type), str(source_id))
    return result.id
```
  passed as `key=_fusion_doc_key`; merged rows (both legs present) display the VECTOR item.

- [ ] **Step 1: READ FIRST** — the citation-merge branch's full body in `_fuse_hybrid_results` (never executed; docstring untrustworthy) and the `entry.item` consumption site (P0's aliasing comment explains the capture-before-mutate rule — your display-preference change must keep leg scores captured BEFORE any mutation).
- [ ] **Step 2: RED tests** (hand-built rows with REAL metadata shapes — this is what P0's tests deliberately lack):
```python
def test_same_document_across_legs_merges():
    # keyword row id="media_15", metadata {"doc_id":"15","source_type":"media"}
    # vector row id="media_15_chunk_0", metadata {"source_id":"15","source_type":"media"}
    # -> ONE fused row; hybrid_fusion has BOTH fts_score and vector_score non-None
def test_merged_row_displays_the_vector_item():
    # the fused row's id/document are the CHUNK row's, not the doc row's
def test_vector_leg_chunks_of_one_doc_collapse_to_best_rank():
    # two chunks of doc 7 at vector ranks 1 and 3 + unrelated doc at rank 2
    # -> doc 7 fused once with vector_rank 1; unrelated doc vector_rank 2
def test_rows_without_metadata_keep_todays_no_merge_behavior():
    # empty-metadata rows with mismatched ids never merge (fallback pin)
def test_merged_citations_combine_without_duplication_or_crash():
    # SearchResultWithCitations on both legs (doc-level + chunk-level citations)
    # -> merged row carries both legs' citations, no exception
def test_fts_only_docs_can_enter_the_top_k():
    # an FTS-only doc with fts_rank 1 must appear in fused results when the
    # vector leg has < top_k distinct docs (the starvation fix, engine-level)
```
- [ ] **Step 3:** RED → implement (key fn + `key=_fusion_doc_key` + display preference at the `entry.item` site + whatever the citation branch's first run needs) → GREEN. Run: the new file + test_hybrid_fusion_metadata.py (UNMODIFIED, must pass) + Tests/RAG_Search/.
- [ ] **Step 4: Mutation checks** (Edit-based restores): revert key to `lambda r: r.id` → the merge tests red, fallback test stays green; flip display preference back to fts_item → only the display test reds.
- [ ] **Step 5: Informational gated run**: hybrid should now differ from semantic — paste the per-category table (expect keyword category movement; watch for paraphrase dips — record, don't chase).
- [ ] **Step 6:** Tick 3994's ACs (the re-stamp AC completes in Task 6 — note that in the task file), Notes, leave the re-stamp AC unticked, status stays In Progress until Task 6. Commit `fix(rag): hybrid fuses on document identity; merged rows display the matched chunk (TASK-3994)`, push.

---

### Task 5: TASK-3996 — notes + conversations sub-legs

**Files:**
- Modify: `tldw_chatbook/RAG_Search/simplified/rag_service.py` (`_keyword_search` becomes the three-sub-leg orchestrator; new `_chacha_notes_fts` / `_chacha_conversations_fts` helpers), `tldw_chatbook/RAG_Search/simplified/config.py` (add `chachanotes_db_path: Optional[Path] = None` beside `media_db_path`)
- Test: `Tests/RAG_Search/test_keyword_leg_chacha.py` (create)

**Interfaces:**
- Consumes: `get_chachanotes_db_path()` (lazy import, mirroring `get_media_db_path`); `interleave_rankings` from `..fusion`.
- Produces: the FTS leg returns media+note+conversation rows interleaved rank-fairly, each stamped `source_type` (`"note"`/`"conversation"` — EXACT singular), `source_id` (note id / conversation id as str), `title`, `source`; chacha queries run over `sqlite3.connect(f"file:{path}?mode=ro", uri=True)` — read-only, no ORM, no schema-touch. A missing/invalid chacha DB degrades those sub-legs with one warning; media continues.

- [ ] **Step 1: READ FIRST** — `search_notes` and `search_conversations_by_content` in DB/ChaChaNotes_DB.py for the exact FTS SQL (notes_fts JOIN notes ON rowid; messages_fts JOIN messages JOIN conversations GROUP BY c.id ORDER BY best_rank), including soft-delete filters (`deleted` columns — replicate them; returning deleted rows would be a data leak the ORM prevents). Mirror that SQL in the raw helpers with the same MATCH-expression escaping as Task 3.
- [ ] **Step 2: RED tests** — build REAL chacha + media DBs in tmp_path via the writer APIs (CharactersRAGDB(path) in the TEST is fine — tests may use the ORM; only the ENGINE must not):
```python
def test_keyword_leg_returns_note_and_conversation_rows(): ...
    # note + conversation + media docs seeded; a query matching all three
    # -> rows of all three source_types, stamped singular, with titles
def test_sub_legs_interleave_rank_fairly(): ...
def test_deleted_notes_and_conversations_are_excluded(): ...
def test_missing_chacha_db_degrades_to_media_only_with_warning(): ...
def test_chacha_connection_is_read_only(): ...
    # the helper's connection cannot write (attempt INSERT -> OperationalError)
def test_cross_leg_merge_per_source_type(): ...
    # THE VOCABULARY-EQUALITY PIN: for each of note/conversation, a keyword
    # sub-leg row and a semantic row of the same doc fuse into ONE row
    # (metadata source_type/source_id shaped exactly as ingestion stamps them)
```
- [ ] **Step 3:** RED → implement → GREEN. Run the new file + Task 3/4's files + test_keyword_leg_db_resolution.py.
- [ ] **Step 4: Informational gated run**: hybrid's FTS leg now sees all 48 fixture docs — paste the table.
- [ ] **Step 5:** Tick 3996's ACs, Notes, `-s Done`. Commit `fix(rag): engine FTS leg covers notes and conversations via read-only sub-legs (TASK-3996)`, push.

---

### Task 6: Final re-stamp, docs, live check, acceptance

**Files:**
- Modify: `Tests/RAG_Eval/baselines/*.json` (the deliberate re-stamp), `Tests/RAG_Eval/README.md` (known-defects list: 3994/3995/3996 fixed — describe what hybrid now does; keep 3997 open), `Docs/User_Guide/library/search-and-rag.md` if its hybrid description needs truth updates (read it), the 3994 task file (final AC + Done).

- [ ] **Step 1: The deliberate re-stamp** (`RAG_EVAL=1 RAG_EVAL_UPDATE_BASELINES=1`): paste the FULL delta printout; build the per-fix progression table (P1 baseline → post-3995 → post-3994 → post-3996 = stamped) from the informational runs' tables. Sanity: hybrid keyword category should be well above 0.135; semantic-mode numbers unchanged from P1 (fusion doesn't touch semantic); plain unchanged. If semantic or plain MOVED, STOP — something leaked outside hybrid; investigate before stamping.
- [ ] **Step 2:** Plain gated run → gate PASSED against the new baselines. Ungated battery (all Tests/RAG_Eval + Tests/RAG_Search files this branch touched, one run; read counts).
- [ ] **Step 3: Collection arithmetic** vs merge-base (baseline worktree `.worktrees/fusion-baseline` off `git merge-base origin/dev HEAD`, removed after; show the arithmetic).
- [ ] **Step 4: Live TUI check** (scratch profile, copied real DBs + chromadb, PR-2 recipe; tmux `-L fusion`): Library rag mode on Hybrid Basic — run a query whose answer lives in a note found by keyword only (pick from real data): pre-fix hybrid would miss it; now it appears with a `keyword match`-band or merged-band row. Capture evidence to the scratchpad. Teardown checklist: app quit, tmux kill-server, scratch deleted, live config untouched (hash), baseline worktree removed, `git status` clean.
- [ ] **Step 5:** README + guide updates; tick 3994's re-stamp AC, `-s Done`; lessons entry ONLY if something generalizable surfaced (the never-run-path-goes-live pattern is already recorded from P0 — extend that entry with the citation-branch incident only if its first run actually misbehaved).
- [ ] **Step 6:** Commit `chore(rag): fusion-cluster re-stamp + docs + closure` (+ trailer), push.

---

## Self-review (done at plan time)

- **Spec coverage:** 3998→Task 2; 3995 (incl. empty-tokenization edge)→Task 3; 3994 (key, fallback pin, display preference, citation first-run, vocabulary equality for media, mutation checks)→Task 4; 3996 (read-only sub-legs, path injection, interleave, degradation, deleted-row exclusion, vocabulary pins for note/conversation)→Task 5; gate choreography (informational runs, one final re-stamp, progression table)→Tasks 3-5 step-wise + Task 6; live check + docs→Task 6. Out-of-scope items respected (3997/3999/pipeline_builder untouched).
- **Placeholder scan:** test skeletons carry contracts with fill-against-real-API instruction (house pattern); no TBDs.
- **Type consistency:** `_fusion_doc_key` defined Task 4, referenced nowhere else by name; sub-leg helpers named once (Task 5); fingerprint keys (Task 2) match baseline_io's existing vocabulary.
- **Deliberate ordering:** 3995 before 3994 so the FTS leg produces multi-token rows BEFORE fusion starts merging them (otherwise 3994's informational run under-reports); 3996 last as the largest.
