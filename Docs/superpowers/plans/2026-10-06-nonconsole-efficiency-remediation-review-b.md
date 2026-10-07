# Non-Console Efficiency Remediation — Review B (Complement) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the 24 speed/efficiency defects unique to the 2026-10-06 review-B performance sweep (the complement of the sibling plan `2026-10-06-nonconsole-efficiency-remediation.md`): unbounded RAG content assembly, per-sample model reloads, O(B×F) sync hashing, serial LLM pipelines, missing browse index, legacy N+1 searches, per-keystroke DOM rebuilds, and per-save store walks — without changing user-visible behavior or search semantics.

**Architecture:** Independent targeted fixes grouped into 6 dependency-free waves (A–F). Each task is atomic, testable, and preserves existing interfaces unless the task explicitly introduces one (lazy singletons, batch helpers, paged grids). Fixes land on `origin/dev` and must not touch any file/region owned by the sibling PR (`perf/nonconsole-efficiency-remediation`) — see the exclusion list under Global Constraints.

**Tech Stack:** Python ≥3.12, Textual 8.x, SQLite (FTS5), numpy, httpx/requests, pytest (in-memory SQLite fixtures).

**Spec:** The findings index in this document (§Findings) is the spec. Line numbers refer to `origin/dev @ cddc89d3e7`; dev has since advanced (`1086b9f8aa`) so lines WILL drift — every task lists a grep anchor to relocate the site. Evidence excerpts live in the review-B session record.

## Global Constraints

- Base every branch on latest `origin/dev`, never on `chore/*` branches carrying console-maintenance WIP.
- **Sibling-PR exclusions (hard):** do NOT modify these files/regions — they are owned by `perf/nonconsole-efficiency-remediation`: `RAG_Search/pipeline_functions_simple.py` rerank function (anchor `Ranker(model_name=`) and `pipeline_builder_simple.py`; `DB/ChaChaNotes_DB.py` `search_library_conversations_page`/`_escape_library_conversation_like` region and the `get_conversations_for_character` / due-flashcards / character-card-visibility query regions; `RAG_Search/simplified/embeddings_wrapper.py`; `RAG_Search/ingestion_indexing.py`; `RAG_Search/simplified/vector_store.py` `delete`/`delete_documents` region (anchor `def delete`); `Character_Chat/world_info_processor.py`, `world_info_resolver.py`, `world_book_manager.py`, `Chat_Dictionary_Lib.py`; `Chat/chat_conversation_service.py`; `Chat/chat_persistence_service.py` `save_history` (anchor `def save_history`); `LLM_Calls/hosted_provider_engine.py`, `hosted_chat.py`, `legacy_line_stream.py`, and the `create_default_session()` call sites; `Widgets/Persona_Widgets/*`; `UI/Library_Modules/library_media_*`; `DB/character_conversation_search.py`; `UI/Watchlists_Modules/reader_item_snapshot.py`; `RAG_Search/reranker.py`; `Notes/file_notes_service.py`, `Notes/file_notes_session_owner.py`, `Widgets/Library/library_file_notes_workspace.py`; `Widgets/Chat_Widgets/chat_message_enhanced.py`, `Widgets/tool_message_widgets.py`, `app_speech.py`; `Chatbooks/chatbook_importer.py`, `Chatbooks/local_chatbook_service.py`; `Chat/Chat_Functions.py`.
- Per `AGENTS.md`: verify with targeted pytest runs of touched modules only; full sweeps only on explicit request.
- Performance claims need evidence per `backlog/docs/lessons-testing-evidence.md`: a before/after measurement (counted calls, spy counts, statement counts, or timers) in the task notes — "faster" is not evidence.
- Never weaken search semantics to win speed. Bounded content assembly (B1) must preserve per-conversation ordering (oldest-first) and stay documented as a retrieval-presentation bound, not a recall change; the FTS matching set is unchanged.
- No new settings surfaces outside `UI/Screens/settings_screen.py`; poll-interval constants stay module-level constants (no new config keys in this PR).
- Design-token rules (ADR-150) do not apply: no UI styling is changed; Python assigns existing classes only.
- Test runner: `cd <worktree-root> && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/... -q` (worktree code wins over the editable install because `python -m` puts cwd first on `sys.path`).

## ADR check (repo requirement)

```text
ADR required: yes (one)
ADR path: backlog/decisions/216-conversations-browse-order-index.md  (created in task B7, before the schema change)
Reason: an index addition is a storage/schema change. All other tasks are mechanical
performance fixes inside existing boundaries (caching, batching, off-loop hops,
debouncing) — no ADR per AGENTS.md ("routine bug fixes / mechanical refactors").
```

## Findings index (spec)

| ID | Sev | Finding | Anchor (grep) | File |
|----|-----|---------|---------------|------|
| B1 | HIGH | Hybrid/keyword conversation leg materializes every matching message, unbounded; citation regex scans full content | `messages_fts fts` inside `_chacha_conversations_fts` | `RAG_Search/simplified/rag_service.py` |
| B2 | MED | Scoped semantic search re-embeds identical query once per allowlist entry | `for allowlist in allowlists` | `Library/library_local_rag_search_service.py`, `RAG_Search/pipeline_functions_simple.py` (scoped-search region only) |
| B3 | MED | In-memory vector store: per-vector cosine w/ query norm recompute; O(n) list dedupe/LRU scans | `self._compute_similarity(query_embedding` | `RAG_Search/simplified/vector_store.py` (InMemoryVectorStore region) |
| B4 | HIGH | SentenceTransformer reloaded per scored eval sample | `SentenceTransformer(` | `Evals/eval_runner.py` |
| B5 | MED | Sync sqlite commit on event loop per eval sample | `progress_wrapper` / `store_result` | `Evals/eval_orchestrator.py` |
| B6 | MED | Evals rail recomputes failure counts over all eval_results per selection change | `run_group_cell_failure_counts` | `DB/Evals_DB.py` |
| B7 | HIGH | No index on `conversations(last_modified DESC, id DESC)`; every browse page temp-B-tree sorts | `ORDER BY last_modified DESC, id DESC` | `DB/ChaChaNotes_DB.py` |
| B8 | MED | Prompts_DB legacy searches: fetch-all FTS ids + IN-list + N+1 keyword attach, no LIMIT | `search_prompts_by_text` / `search_prompts_by_keyword` | `DB/Prompts_DB.py` |
| B9 | LOW | `search_media_db` DISTINCT over 1:1 joins forces dedup before LIMIT short-circuit | `COUNT(DISTINCT m.id)` | `DB/Client_Media_DB_v2.py` |
| B10 | HIGH | Notes sync identity fallback: O(B_miss × F) SHA-256 per pass, on the event loop | `identity_matches = tuple(` | `Notes/notes_sync_runtime.py` |
| B11 | LOW/MED | Sync plan + observation token computed 2–3× per pass, once on the event loop | `plan_reconciliation` in `observe_root` | `Notes/notes_sync_runtime.py` |
| B12 | MED | Sync watcher stat-walks every root every 1 s forever (idle ceiling 10 s) | `changed_root_ids` / `sync_watcher_interval_seconds` | `Notes/notes_sync_watcher.py`, `Notes/notes_sync_runtime.py`, `config.py` |
| B13 | MED | Docling `DocumentConverter` (and OCR models) re-created per PDF | `DocumentConverter()` | `Local_Ingestion/PDF_Processing_Lib.py` |
| B14 | HIGH | Ingestion "analysis" summarizes chunks strictly serially | `for i, chunk in enumerate(processed_chunks)` near `analyze(` | `Local_Ingestion/PDF_Processing_Lib.py`, `Local_Ingestion/Book_Ingestion_Lib.py`, `Local_Ingestion/Document_Processing_Lib.py` |
| B15 | MED | CLI batch ingest strictly serial per file | `for file_path in file_paths` in `batch_ingest_files` | `Local_Ingestion/local_file_ingestion.py` |
| B16 | HIGH | Research websearch gate: per-result serial sleep→LLM→scrape→sleep→summarize | `for idx, result in enumerate(search_results)` | `Web_Scraping/WebSearch_APIs.py` |
| B17 | MED | Activity log: per-keystroke teardown+remount of up to 1000×5 widgets; 60 s full rebuild | `_update_display` | `Widgets/activity_log.py` |
| B18 | MED | SmartContentTree: O(all-nodes) string rebuild + display writes per keystroke; sync load on mount | `_apply_filters` | `UI/Widgets/SmartContentTree.py` |
| B19 | MED | Config search: two full recursive DOM walks per keystroke | `_build_index` / `on_config_search_changed` | `UI/Widgets/config_search_widget.py` |
| B20 | MED | Repo tree: O(n) prefix scan + DOM query per descendant per directory toggle | `startswith(path + "/")` | `Widgets/Coding_Widgets/repo_tree_widgets.py` |
| B21 | MED | Trajectory timeline: O(lanes×records) render per frame incl. mouse-move drag | `_lane_line` / `on_mouse_move` | `UI/Widgets/trajectory_timeline.py` |
| B22 | MED | Chatbooks grid mounts one widget subtree per match, uncapped | `_update_content` mount loop | `UI/Chatbooks_Window_Improved.py` |
| B23 | MED | Video store walks whole store 3–4× per save; `RecoveredMedia` rebuilt per resolve (hundreds per slug allocation) | `_cleanup_orphan_stages_unlocked` / `RecoveredMedia(self._recovered_root)` | `Video_Generation/video_store.py` |
| B24 | MED | Console fork commit: 2–3 SELECTs per copied message | `resolve_console_fork_commit` / `_fork_source_parent` | `Chat/chat_persistence_service.py` (~2088–2238; do NOT touch `save_history` — sibling deletes it) |
| B25 | MED | Chat-history import replays up to 10k messages through the full per-message sidecar path | `for staged in staged_messages` | `Character_Chat/Character_Chat_Lib.py` |
| B26 | MED | `BUILTIN_PIPELINES` shallow-copied then nested dicts mutated (global state corruption); OpenRouter per-chunk INFO log | `BUILTIN_PIPELINES["` / `OpenRouter Stream: Content received` | `Event_Handlers/Chat_Events/chat_rag_events.py`, `LLM_Calls/Summarization_General_Lib.py` |
| B27 | LOW | Loop-invariant micro pack: per-char UTF-8 byte counting in think filter; EPUB spine linear scans; per-file `workspace.resolve()` in grep tool | `ord(text[index])` / `get_item_with_id` / `resolved_workspace = workspace.resolve()` | `Chat/llamacpp_think_filter.py`, `Local_Ingestion/Book_Ingestion_Lib.py`, `Tools/local_tool_impls.py` |

---

## Wave overview

| Wave | Tasks | Theme | Ships value |
|------|-------|-------|-------------|
| A | B1–B3 | RAG hot-path bounds + store internals | Default-profile search stops materializing unbounded text; fallback store stops O(n·m) |
| B | B4–B9 | Evals + DB layer | Eval runs lose 1000× model loads and per-sample loop stalls; browse stops sorting; legacy searches stop N+1 |
| C | B10–B12 | Notes sync | Renames stop squaring hashing; passes stop double planning; idle poller calms down |
| D | B13–B16 | Ingestion + research | Analyzed ingestion and research answers drop from minutes to tens of seconds |
| E | B17–B22 | UI keystroke hygiene | Legacy screens stop rebuilding the world per keystroke/click |
| F | B23–B27 | Stores, imports, micro packs | Per-save syscall multiplication, fork/import round trips, correctness-adjacent global-state fix |

Waves are independent — parallelizable across agents. Within a wave, tasks are file-disjoint except B10/B11/B12 (same module — land in order; B10 first).

---

### Task B1 (Wave A): Bound the hybrid/keyword conversation content leg

**Files:**
- Modify: `tldw_chatbook/RAG_Search/simplified/rag_service.py` — `_chacha_conversations_fts` (anchor `grep -n "_chacha_conversations_fts" RAG_Search/simplified/rag_service.py`; the second statement builds `messages_sql` with `FROM messages_fts fts`), and `_create_keyword_result_with_citations` (anchor `def _create_keyword_result_with_citations`) / `_keyword_citation_spans` (anchor `def _keyword_citation_spans`)
- Test: `Tests/RAG_Search/test_conversation_content_bound.py` (create)

**Interfaces:**
- Produces: module constants `_MAX_CONV_MESSAGES_PER_CONVERSATION = 80` and `_MAX_CONV_MESSAGES_TOTAL = 400` in `rag_service.py`; `_keyword_citation_spans` gains keyword param `text_limit: int | None = None` (existing callers unchanged — default None scans whole text as today; the conversation-content caller passes the preview length).

- [ ] **Step 1: Write failing tests** (adapt the fixture helpers to however neighboring rag_service tests build their in-memory DB — grep `Tests/RAG_Search` for an existing `messages_fts` fixture and reuse it):

```python
"""B1: conversation content assembly is bounded and citations scan the preview only."""
import re

import pytest

from tldw_chatbook.RAG_Search.simplified import rag_service as rs


@pytest.mark.asyncio
async def test_content_leg_caps_messages_per_conversation(seeded_conversations_db, rag_service_instance):
    # seeded fixture: 1 conversation, 500 messages all matching query "alpha"
    docs = await rag_service_instance._chacha_conversations_fts("alpha", limit=5)
    assert docs, "expected at least one document"
    doc = docs[0]
    joined_lines = doc["content"].split("\n")
    assert len(joined_lines) <= rs._MAX_CONV_MESSAGES_PER_CONVERSATION
    # oldest-first ordering preserved within the bound
    timestamps = [line for line in joined_lines]
    assert timestamps == sorted(timestamps, key=len) or True  # ordering asserted via SQL test below


@pytest.mark.asyncio
async def test_content_leg_caps_total_messages_across_conversations(seeded_many_conversations_db, rag_service_instance):
    docs = await rag_service_instance._chacha_conversations_fts("alpha", limit=5)
    total_lines = sum(len(d["content"].split("\n")) for d in docs)
    assert total_lines <= rs._MAX_CONV_MESSAGES_TOTAL


@pytest.mark.asyncio
async def test_citation_spans_scan_preview_only(rag_service_instance, monkeypatch):
    seen = {}
    real = rs._keyword_citation_spans

    def spy(text, *a, **k):
        seen["len"] = len(text)
        return real(text, *a, **k)

    monkeypatch.setattr(rs, "_keyword_citation_spans", spy)
    # seed one matching conversation whose messages exceed the preview size
    docs = await rag_service_instance._keyword_search.__wrapped__(rag_service_instance, "alpha") \
        if hasattr(rag_service_instance._keyword_search, "__wrapped__") else None
    assert seen.get("len", 0) <= 1000 + rs.MAX_QUERY_LENGTH, "citation scanning must run on the bounded preview"
```

  NOTE to implementer: the third test's exact hook point depends on where the preview (`[:1000]`) is currently computed relative to citation scanning — read `_create_keyword_result_with_citations` first and write the spy against the real call path. If citations already receive only the preview, keep the test as a regression guard and record that in notes. The ordering guarantee (oldest-first per conversation) must be asserted deterministically: seed messages with increasing timestamps and assert the first retained line's timestamp < the last retained line's timestamp.

- [ ] **Step 2: Run, expect FAIL** — `pytest Tests/RAG_Search/test_conversation_content_bound.py -q` (constants missing / unbounded counts).
- [ ] **Step 3: Implement.** In the second statement of `_chacha_conversations_fts`, replace the bare SELECT with a windowed bound (SQLite ≥3.25 ships with Python 3.12):

```sql
SELECT conversation_id, line FROM (
    SELECT m.conversation_id AS conversation_id,
           COALESCE(m.sender, 'unknown') || ': ' || COALESCE(m.content, '') AS line,
           ROW_NUMBER() OVER (
               PARTITION BY m.conversation_id
               ORDER BY m.timestamp ASC, m.rowid ASC
           ) AS rn
    FROM messages_fts fts
    JOIN messages m ON fts.rowid = m.rowid
    WHERE fts.messages_fts MATCH ? AND m.deleted = 0
      AND m.conversation_id IN ({placeholders})
)
WHERE rn <= ?
ORDER BY conversation_id, rn
```

  Then truncate the assembled per-conversation lists to `_MAX_CONV_MESSAGES_TOTAL` overall in Python (iterate conversations in their existing top-k order, accumulating until the cap). Add a comment: presentation bound for prompt assembly, FTS recall unchanged. In the citation path, pass `text_limit=1000` (or the existing preview constant — grep `[:1000]`) at the conversation-content call site so `_keyword_citation_spans` scans only the preview it will be cited against.
- [ ] **Step 4: Run** — `pytest Tests/RAG_Search/test_conversation_content_bound.py Tests/RAG_Search -k "keyword or hybrid or conversation" -q`. Existing keyword/hybrid behavior tests must stay green.
- [ ] **Step 5: Evidence + do not commit (orchestrator commits per wave).** Record: seeded 5,000-message fixture, rows fetched before (5,000) vs after (≤ 400) via `sqlite3.set_trace_callback` row counts.

**Acceptance criteria:** content assembly fetches ≤ caps regardless of match count; per-conversation oldest-first order preserved; citation regex never scans more than the preview; existing RAG keyword/hybrid tests green.

---

### Task B2 (Wave A): One embed per scoped semantic search

**Files:**
- Modify: `tldw_chatbook/Library/library_local_rag_search_service.py` (anchor `for allowlist in allowlists`)
- Modify: `tldw_chatbook/RAG_Search/pipeline_functions_simple.py` — ONLY the scoped-search allowlist loop at ~:524-535 (anchor `for allowlist in allowlists`); **do not touch the rerank function or any other region of this file** (sibling PR owns those)
- Test: `Tests/RAG_Search/test_scoped_search_single_embed.py` (create)

**Interfaces:** none new — callers pass the same `source_types`/scopes; the merge happens internally.

- [ ] **Step 1: Failing test** (monkeypatch the rag service so `search` counts embedding rounds — grep how existing tests fake `embeddings.create_embeddings_async`):

```python
"""B2: scoped semantic search performs exactly one query embedding per search."""
import pytest


@pytest.mark.asyncio
async def test_union_allowlist_single_embed(scoped_search_service, fake_rag_service):
    # fake_rag_service.search records metadata_allowlist args and embed call counts
    await scoped_search_service.search("test query", source_types=["media", "notes", "conversations"])
    assert fake_rag_service.embed_calls == 1, "k allowlist entries must not re-embed the query k times"
    assert fake_rag_service.search_calls == 1
    # the union allowlist must cover exactly the requested scopes
    passed = fake_rag_service.last_metadata_allowlist
    assert passed == fake_rag_service.expected_union_allowlist
```

- [ ] **Step 2: Run, expect FAIL** (embed_calls == k).
- [ ] **Step 3: Implement.** In both call sites, replace the `for allowlist in allowlists: await ...search(..., metadata_allowlist=allowlist)` loop with a single call passing the union list, exactly as `_search_hybrid` already does (grep `metadata_allowlist=allowlists` for the proven pattern). Preserve each caller's result-shaping code (the loop may zip results per source — collect the single result list and feed the same shaping logic; if the loop merges per-source results differently, replicate that merge over the single result set). Verify `_semantic_search_scoped` in `rag_service.py` (anchor `def _semantic_search_scoped`) accepts the multi-entry allowlist (the review confirmed it does).
- [ ] **Step 4: Run** — `pytest Tests/RAG_Search/test_scoped_search_single_embed.py Tests/Library -k "rag or search" -q`.
- [ ] **Step 5: Evidence** — record embed calls before (k, typically 2–4) vs after (1).

**Acceptance criteria:** one query embedding and one search call per scoped semantic search; returned results equivalent (same doc ids, tolerating order within score ties) to the merged loop output on a fixture.

---

### Task B3 (Wave A): In-memory vector store — O(1) lookups, matrix cosine, OrderedDict LRU

**Files:**
- Modify: `tldw_chatbook/RAG_Search/simplified/vector_store.py` — `InMemoryVectorStore` only (`add`/`_evict_lru`/`search`/`_compute_similarity`; anchor `class InMemoryVectorStore`). **Do not touch** `ChromaVectorStore` or the `delete` region (sibling PR owns `delete_documents`).
- Test: `Tests/RAG_Search/test_inmemory_vector_store_scaling.py` (create)

**Interfaces:**
- Produces: `InMemoryVectorStore` keeps its public surface (`add`, `search`, `clear`, size/capacity properties) and its eviction semantics (LRU by access, cap at the same configured max). Internals may change freely.

- [ ] **Step 1: Equivalence + behavior tests first** (these pin current behavior before the rewrite):

```python
"""B3: in-memory store results identical to naive cosine; LRU/dedupe semantics preserved."""
import numpy as np
import pytest

from tldw_chatbook.RAG_Search.simplified.vector_store import InMemoryVectorStore


def _naive_search(store, query, k):
    q = np.asarray(query, dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-12)
    scored = []
    for i, emb in enumerate(store.embeddings):
        e = np.asarray(emb, dtype=np.float32)
        e = e / (np.linalg.norm(e) + 1e-12)
        scored.append((float(np.dot(q, e)), i))
    scored.sort(reverse=True)
    return [i for _, i in scored[:k]]


def test_search_matches_naive_cosine(small_store_factory):
    store = small_store_factory(n=50, dim=8)
    query = [0.1] * 8
    got = [r["id"] for r in store.search(query, k=5)]
    want = [store.ids[i] for i in _naive_search(store, query, 5)]
    assert got == want


def test_lru_eviction_and_dedupe(tmp_store):
    store = tmp_store(cap=3)
    store.add("a", [1.0, 0.0]); store.add("b", [0.0, 1.0]); store.add("c", [1.0, 1.0])
    store.search([1.0, 0.0], k=3)          # touches "a" -> most recently used
    store.add("d", [0.5, 0.5])             # evicts LRU, which is "b" (not "a")
    assert set(store.ids) == {"a", "c", "d"}
    store.add("a", [0.9, 0.1])             # dedupe: replaces, does not duplicate
    assert store.ids.count("a") == 1 and len(store.embeddings) == 3
```

  Adapt accessor names (`store.embeddings`, `store.ids`) to the real attributes — read the class first. Keep the naive-cosine oracle's normalization matching whatever `_compute_similarity` currently does (cosine with epsilon — verify, don't assume).

- [ ] **Step 2: Run, expect PASS on old code** (golden baseline).
- [ ] **Step 3: Implement.**
  - Maintain `self._index_by_id: dict[Any, int]` alongside the parallel lists; dedupe check and replace become dict lookups.
  - Maintain `self._access_order: OrderedDict[Any, None]`; LRU touch = `move_to_end`, eviction = `popitem(last=False)`; `search` moves result ids to the end without list scans.
  - Maintain `self._matrix: np.ndarray | None` (shape `(n, d)`, float32) appended on add, rebuilt lazily on remove/replace (set to None and rebuild on next search); precompute row norms with the matrix; `search` = normalize query once, single `self._matrix @ q_normalized`, `argpartition` top-k then sort those. Delete the per-chunk `np.linalg.norm(query_embedding)` recomputation inside the loop.
  - Keep `clear()` invalidating all three structures. Keep the public cap/eviction config identical.
- [ ] **Step 4: Run** — `pytest Tests/RAG_Search/test_inmemory_vector_store_scaling.py Tests/RAG_Search -k "vector or in_memory or memory_store" -q`.
- [ ] **Step 5: Evidence** — index 5,000×16-dim vectors then search 100 queries: wall time before/after (expect ≫5× from the matmul alone); record numbers.

**Acceptance criteria:** results byte-equivalent to the naive oracle; LRU eviction order and dedupe-replace behavior identical; per-search numpy calls O(1) + one matmul; add path O(1) amortized (no `list.index`/`list.remove`).

---

### Task B4 (Wave B): SentenceTransformer process singleton

**Files:**
- Modify: `tldw_chatbook/Evals/eval_runner.py` — `calculate_semantic_similarity` (anchor `def calculate_semantic_similarity`; construction site at ~:820-827)
- Test: `Tests/Evals/test_semantic_model_singleton.py` (create)

**Interfaces:**
- Produces: `get_semantic_embedding_model() -> "SentenceTransformer"` and `_reset_semantic_model_for_tests() -> None` in `eval_runner.py`. All existing callers of `calculate_semantic_similarity` keep working unchanged (the `embedding_model=None` default stays).

- [ ] **Step 1: Failing test**:

```python
"""B4: the semantic-similarity model is constructed once per process."""
import sys
import types

import eval_runner_module  # placeholder — import the real module below

from tldw_chatbook.Evals import eval_runner


class _FakeST:
    instances = 0

    def __init__(self, *a, **k):
        type(self).instances += 1

    def encode(self, texts, **k):
        return [[float(len(t))] for t in texts]


def test_model_constructed_once(monkeypatch):
    eval_runner._reset_semantic_model_for_tests()
    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = _FakeST
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    try:
        eval_runner.calculate_semantic_similarity("hello world", "hello there")
        eval_runner.calculate_semantic_similarity("another", "pair")
        assert _FakeST.instances == 1
    finally:
        eval_runner._reset_semantic_model_for_tests()
```

- [ ] **Step 2: Run, expect FAIL** (instances == 2).
- [ ] **Step 3: Implement.** Module state `_SEMANTIC_MODEL = None`, `_SEMANTIC_MODEL_LOCK = threading.Lock()`; `get_semantic_embedding_model()` double-checked-locks the construction with the exact current args (`SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)`). Replace the in-function construction with `embedding_model = get_semantic_embedding_model()` when the arg is None. `_reset_semantic_model_for_tests()` clears both. Do not change the similarity math.
- [ ] **Step 4: Run** — `pytest Tests/Evals/test_semantic_model_singleton.py Tests/Evals -k "semantic or metric or similarity" -q`.
- [ ] **Step 5: Evidence** — record: 3 consecutive `calculate_semantic_similarity` calls → `SentenceTransformer` constructions before (3) vs after (1).

**Acceptance criteria:** one construction per process under concurrency (spawn two threads in a test variant if cheap); all existing eval metric tests green.

---

### Task B5 (Wave B): Off-loop eval result writes

**Files:**
- Modify: `tldw_chatbook/Evals/eval_orchestrator.py` (anchor `progress_wrapper` / `store_result`)
- Test: `Tests/Evals/test_orchestrator_offloop_writes.py` (create)

**Interfaces:** none new.

- [ ] **Step 1: Failing test** — drive one sample completion through the orchestrator with a fake `db` whose `store_result` records `threading.current_thread()`; assert it is not the thread running the event loop, and that the awaited wrapper still propagates db exceptions:

```python
"""B5: per-sample result persistence hops off the event loop."""
import asyncio
import threading

import pytest

from tldw_chatbook.Evals import eval_orchestrator as eo


@pytest.mark.asyncio
async def test_store_result_runs_off_loop(orchestrator_with_fake_db):
    seen_threads = []
    orchestrator_with_fake_db.db.store_result = lambda *a, **k: seen_threads.append(threading.current_thread())
    await orchestrator_with_fake_db._persist_sample_result(sample=fake_sample())  # adapt to real entry point
    assert seen_threads and seen_threads[0] is not threading.main_thread()
```

- [ ] **Step 2: Run, expect FAIL** (same thread).
- [ ] **Step 3: Implement.** Check whether the repo's `run_db_off_loop` helper (grep `def run_db_off_loop`) is importable here without an import cycle; if yes reuse it, else `await asyncio.to_thread(self.db.store_result, ...)`. Preserve the exact call arguments and error handling around the write.
- [ ] **Step 4: Run** — `pytest Tests/Evals/test_orchestrator_offloop_writes.py Tests/Evals -k "orchestrator" -q`.
- [ ] **Step 5: Evidence** — record blocking-write-per-sample count before (n on loop) vs after (0 on loop).

**Acceptance criteria:** no sqlite statements execute on the event loop during sample completion; failure in `store_result` still surfaces to the run's error path.

---

### Task B6 (Wave B): Cache run-group failure counts

**Files:**
- Modify: `tldw_chatbook/DB/Evals_DB.py` — `run_group_cell_failure_counts` (anchor `def run_group_cell_failure_counts`)
- Test: `Tests/Evals/test_failure_count_cache.py` (create)

**Interfaces:**
- Produces: `EvalsDB.invalidate_run_group_failure_cache() -> None` (public; called by every mutation path of `eval_results`).

- [ ] **Step 1: Failing test** — seed runs/results; call `run_group_cell_failure_counts` twice with a trace-callback counting `SELECT`s against `eval_results`; assert second call issues 0; call `store_result` (the real mutator — grep `def store_result` in this file); assert the next call re-scans and reflects the new row.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.** Instance-level cache `self._rg_failure_counts: dict | None = None`; the method returns the memo when set; `store_result` (and any other `eval_results` writer — grep `INSERT INTO eval_results` / `DELETE FROM eval_results`) sets it to None. Cache lives per DB instance; no cross-process pretensions. Size is naturally bounded by run-group count.
- [ ] **Step 4: Run** — `pytest Tests/Evals/test_failure_count_cache.py Tests/Evals -k "failure or run_group or view_model" -q`.
- [ ] **Step 5: Evidence** — record `json_extract` scans per compose before (1 per selection change) vs after (0 until next mutation).

**Acceptance criteria:** idle rail composes are O(1); counts never stale after a mutation; existing Evals DB tests green.

---

### Task B7 (Wave B): Conversations browse-order index (+ ADR-216)

**Files:**
- Create: `tldw_chatbook/backlog/decisions/216-conversations-browse-order-index.md`
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py` — schema block containing `CREATE INDEX IF NOT EXISTS idx_conversations_root` (~:893); consumers already exist (`search_conversations_page` ~:11561, `list_all_active_conversations` ~:10908, `locate_conversation_page` ~:11645 — do not modify the queries)
- Test: `Tests/ChaChaNotesDB/test_conversations_browse_index.py` (create)

**Interfaces:** none (pure index).

- [ ] **Step 1: Investigate the established index-addition pattern FIRST.** `git log --oneline -S "idx_notes_last_modified" -- tldw_chatbook/DB/ChaChaNotes_DB.py` and read the surrounding commit: determine whether new indices reach existing DBs via (a) the schema script re-run on every open (`CREATE INDEX IF NOT EXISTS`), or (b) a versioned migration step. Follow whichever pattern (a)/(b) the repo uses for indices; if (b), add the migration step in the same style as the most recent index migration and note the exact next schema version used (sibling PR may also bump versions — record yours in the PR description for conflict resolution).
- [ ] **Step 2: Write failing test**:

```python
"""B7: browse ordering is index-served (no temp B-tree per page)."""
import sqlite3

from tldw_chatbook.DB.ChaChaNotes_DB import ChaChaNotesDB


def test_browse_order_uses_index(tmp_path):
    db = ChaChaNotesDB(str(tmp_path / "t.db"))
    conn = db.connection  # adapt to however tests grab the raw connection
    plan = "\n".join(r[3] for r in conn.execute(
        "EXPLAIN QUERY PLAN SELECT id FROM conversations "
        "WHERE deleted = 0 ORDER BY last_modified DESC, id DESC LIMIT 20 OFFSET 0"
    ))
    assert "TEMP B-TREE FOR ORDER BY" not in plan.upper(), plan
    assert "idx_conversations_last_modified" in plan, plan
```

  (Adapt connection access + the exact WHERE to match `search_conversations_page`'s scope filter; the assertion that matters is "no TEMP B-TREE FOR ORDER BY".)
- [ ] **Step 3: Run, expect FAIL.** Add `CREATE INDEX IF NOT EXISTS idx_conversations_last_modified ON conversations(last_modified DESC, id DESC);` next to its sibling conversation indices (match their formatting), plus the pattern-appropriate migration wiring from Step 1.
- [ ] **Step 4: Run** — `pytest Tests/ChaChaNotesDB/test_conversations_browse_index.py Tests/ChaChaNotesDB -k "conversation or page or browse" -q`.
- [ ] **Step 5: Evidence + ADR.** Write `backlog/decisions/216-conversations-browse-order-index.md` (context: browse pages sort the full filtered set; decision: covering DESC index; alternatives: covering index incl. scope columns rejected because scope filters vary — record whatever EXPLAIN shows). Record rows-sorted before (n) vs after (page size) via EXPLAIN + trace on a 5k-conversation fixture.

**Acceptance criteria:** `EXPLAIN QUERY PLAN` shows the index serving the default browse ordering (no ORDER BY temp B-tree) for the page query; existing conversation tests green; ADR-216 written.

---

### Task B8 (Wave B): Port legacy Prompts_DB searches to the proven pattern

**Files:**
- Modify: `tldw_chatbook/DB/Prompts_DB.py` — `search_prompts_by_text` (anchor `def search_prompts_by_text`; IN-list at ~:4596, keyword loop ~:4608-4612), `search_prompts_by_keyword` (anchor `def search_prompts_by_keyword`; ~:4489-4511), `search_prompts` keyword attach (~:4011-4014). Do not touch anything else.
- Test: `Tests/PromptsDB/test_legacy_prompt_search_ports.py` (create)

**Interfaces:** none — same signatures and return shapes.

- [ ] **Step 1: Golden equivalence tests on CURRENT code** (run before changing anything): seed ≥ 30 prompts with keywords + FTS-indexed text (find the fixture helpers neighboring prompt-search tests use — grep `Tests/` for `search_prompts`); capture result tuples (id, name, keywords) for: content match, keyword match, combined, empty result. Save as the golden baseline (inline expected values or a snapshot dict in the test).
- [ ] **Step 2: Port `search_prompts_by_text`.** Replace the fetch-all-rowids + `IN ({placeholders})` with the subquery shape already shipped in `search_prompts` (~:3977-3981): `WHERE p.id IN (SELECT rowid FROM prompts_fts WHERE prompts_fts MATCH ?)`. Add LIMIT/OFFSET parameters (default the existing search page size — grep what `search_prompts` defaults to; keep backward-compatible defaults so existing callers see unchanged page sizes). Replace the per-row `fetch_keywords_for_prompt` loop with the batch helper `_library_keywords_for_prompts` (anchor `def _library_keywords_for_prompts`).
- [ ] **Step 3: Port `search_prompts_by_keyword`.** Same two changes: LIMIT/OFFSET (same defaults) + batch keyword attach.
- [ ] **Step 4: Port `search_prompts` page attach.** Swap its per-row keyword loop (~:4011-4014) for `_library_keywords_for_prompts` (as `list_library_prompts_page` already does).
- [ ] **Step 5: Run equivalence + new tests.** New test additions: with a trace callback, `search_prompts_by_text` on a page of 20 issues exactly 1 keywords query (not 20); LIMIT honored. `pytest Tests/PromptsDB/test_legacy_prompt_search_ports.py Tests/PromptsDB -q` plus any existing prompt-search test dirs (grep `Tests/ -name "*prompt*" -type d`).
- [ ] **Step 6: Evidence** — record per-search SQL statement counts before/after on the fixture (before: 1 + page-size or 1 + n_matches; after: constant).

**Acceptance criteria:** golden results identical for all fixture cases; statement counts bounded (no N+1); no `IN ({placeholders})` built from unbounded FTS match lists remains in the file's search paths; existing prompt tests green.

---

### Task B9 (Wave B): Drop redundant DISTINCT in `search_media_db`

**Files:**
- Modify: `tldw_chatbook/DB/Client_Media_DB_v2.py` — `count_select` (~:2751) and `final_select_stmt` (~:3146) (anchors `COUNT(DISTINCT m.id)` / `SELECT DISTINCT`)
- Test: extend the existing media-search equivalence tests (grep `Tests/` for `search_media_db` usage; add a case if none covers multi-join rows)

**Interfaces:** none.

- [ ] **Step 1: Prove 1:1 joins first.** Read every JOIN the builder can emit for the affected query shapes (media_fts, keyword joins, etc.). For each, write down the join cardinality in the test docstring. If ANY join can multiply rows, stop: keep DISTINCT, record why in notes, and skip to Step 4.
- [ ] **Step 2: Equivalence test** — golden result sets (ids + counts) for: text search, keyword filter, combined, empty. Run on current code, capture baseline.
- [ ] **Step 3: Remove DISTINCT** from both the COUNT and the SELECT. Re-run equivalence.
- [ ] **Step 4: Run** — `pytest Tests/ -k "search_media" -q` (targeted), plus the file's own test module if present.
- [ ] **Step 5: Evidence** — EXPLAIN/trace: temp B-tree dedup step present before, absent after, on a fixture with ≥ 1k matches.

**Acceptance criteria:** identical results; dedup temp-B-tree gone (if Step 1 proved 1:1); documented cardinality reasoning in notes.

---

### Task B10 (Wave C): Identity-fallback digest map (kills O(B×F) hashing)

**Files:**
- Modify: `tldw_chatbook/Notes/notes_sync_runtime.py` — the `observe_root` identity-fallback block (anchor `identity_matches = tuple(`)
- Test: `Tests/Notes/test_identity_fallback_map.py` (create)

**Interfaces:**
- Produces: `NotesSyncExecutor.identity_digest_index(discovered) -> dict[str, DiscoveredImportSource]` (module-level or method — match where `stable_identity_digest` lives) mapping `stable_identity_digest(item) -> item`. Uniqueness: if two files share a digest, keep the first in discovery order (the old code's `len(identity_matches) == 1` guard means ambiguous digests never matched anyway — preserve exactly that: store digests as `digest -> list[item]` and only resolve when the list has exactly one entry).

- [ ] **Step 1: Failing test**:

```python
"""B10: identity fallback computes each file digest once per pass, not once per missed binding."""
import pytest

from tldw_chatbook.Notes import notes_sync_runtime as rt


def test_digest_computed_once_per_file(monkeypatch, observe_fixture_factory):
    fixture = observe_fixture_factory(files=50, bindings=10, missing_bindings=3)  # 3 renamed files
    calls = []
    real = rt.NotesSyncExecutor.stable_identity_digest
    monkeypatch.setattr(rt.NotesSyncExecutor, "stable_identity_digest",
                        staticmethod(lambda item: (calls.append(item.observation.relative_path),
                                                   real(item))[1]))
    rt.run_observe_pass(fixture)  # adapt to the real entry (sync or async)
    per_file = {p: calls.count(p) for p in set(calls)}
    assert max(per_file.values()) == 1, f"digests recomputed: { {p: c for p, c in per_file.items() if c > 1} }"
    renamed = fixture.resolved_renamed_bindings()
    assert len(renamed) == 3  # fallback still resolves single-digest matches
```

- [ ] **Step 2: Run, expect FAIL** (renamed files' digests computed once per missed binding).
- [ ] **Step 3: Implement.** In `observe_root`, on first path miss, build the index once (store it in a local for the rest of the pass); resolve each miss via `index.get(binding.stable_identity_digest)` + the exactly-one-entry guard; delete the second digest computation at ~:1057-1059 by reusing the digest already computed for the matched file. Preserve the "no match on ambiguity" semantics and everything downstream (`claimed_paths` etc.).
- [ ] **Step 4: Run** — `pytest Tests/Notes/test_identity_fallback_map.py Tests/Notes -k "sync or observe or binding" -q`.
- [ ] **Step 5: Evidence** — digest computations per pass before (B_miss × F) vs after (B + F) on the fixture (record both numbers).

**Acceptance criteria:** per-pass digest computations ≤ B + F; rename resolution behavior identical (including ambiguity rejection); existing sync tests green.

---

### Task B11 (Wave C): Compute the reconciliation plan once per pass, off the event loop

**Files:**
- Modify: `tldw_chatbook/Notes/notes_sync_runtime.py` — the `plan_reconciliation`/`_observation_token` call sites in `observe_root` (~:1156, ~:2684-2685) (anchors `plan_reconciliation(request)` near `observation_token`, and `_observation_token(observations)`)
- Test: `Tests/Notes/test_single_plan_per_pass.py` (create)

**Interfaces:** none new.

- [ ] **Step 1: Failing test** — monkeypatch-count `plan_reconciliation` and `_observation_token` across one full `observe_root` pass: each must execute exactly once per pass. Also assert (via `threading.current_thread()` captured inside the plan spy) that the plan runs on a worker thread, not the event loop thread, when invoked from async context.
- [ ] **Step 2: Run, expect FAIL** (2–3 plan executions; one on the loop).
- [ ] **Step 3: Implement.** Read both call sites: at ~:1156 the full plan is built only to read `.observation_token` — replace with the cheap `_observation_token(observations)` helper if that is what the token needs (verify what `observation_token` derives from), or thread the already-computed plan through. At ~:2684 the token AND the plan are both computed — compute the plan once, derive the token from it (or from observations, whichever the reconciler's contract is — read `plan_reconciliation`'s return type first). Move whichever call remains in the async body into the existing `to_thread` offload pattern used by neighboring calls in the same function (grep `to_thread` in this file). Preserve the token's exact value (it feeds change detection — a changed token value would cause spurious syncs; the golden test must assert token equality before/after on a fixture).
- [ ] **Step 4: Run** — `pytest Tests/Notes/test_single_plan_per_pass.py Tests/Notes -k "observe or plan or token" -q`.
- [ ] **Step 5: Evidence** — plan executions per pass before (2–3) vs after (1); token value unchanged.

**Acceptance criteria:** exactly one plan computation per pass; no planning on the event loop; observation token byte-identical to before on fixtures.

---

### Task B12 (Wave C): Calm the sync watcher's idle polling

**Scope note (honest):** a per-directory mtime checkpoint that skips statting files is *unsound* — content edits change file mtime without touching any directory mtime, so pruning by dir stat would silently miss edits. The sound cheap fix is cadence, not checkpointing. OS-native watchers (FSEvents/inotify) are explicitly out of scope for this PR (recorded as follow-up candidate, matching the sibling plan's stance on `file_notes`).

**Files:**
- Modify: `tldw_chatbook/Notes/notes_sync_watcher.py` (anchor `poll_once` / backoff logic)
- Modify: `tldw_chatbook/Notes/notes_sync_runtime.py` — only if the backoff ceiling lives there (anchor `changed_root_ids`)
- Modify: `tldw_chatbook/config.py` — the watcher defaults block (anchor `sync_watcher_interval_seconds`) — constants only, no new keys
- Test: `Tests/Notes/test_watcher_idle_backoff.py` (create)

**Interfaces:** none new; existing config key keeps its name and active-poll behavior.

- [ ] **Step 1: Failing test** — drive `poll_once` with a stubbed discovery: with zero detected changes across N consecutive ticks, the effective sleep interval must grow to the new idle ceiling (30 s) and cap there; a single detected change resets to the base interval (1 s). Assert via a fake clock/timer recorder (follow how existing watcher tests fake time — grep `Tests/Notes` for watcher tests).
- [ ] **Step 2: Run, expect FAIL** (ceiling is 10 s).
- [ ] **Step 3: Implement.** Raise the idle-backoff ceiling 10 s → 30 s and smooth the growth curve (e.g. double per idle tick: 1→2→4→8→16→30). Keep the reset-on-change behavior byte-identical. Update the default constant in `config.py` only if the ceiling is expressed there.
- [ ] **Step 4: Run** — `pytest Tests/Notes/test_watcher_idle_backoff.py Tests/Notes -k "watcher or poll" -q`.
- [ ] **Step 5: Evidence** — walks per 10 idle minutes before (60) vs after (≈26).

**Acceptance criteria:** idle stat-walk frequency reduced ≥ 2×; active (changing) vaults poll exactly as today; reset-on-change verified.

---

### Task B13 (Wave D): Docling converter per-process singleton

**Files:**
- Modify: `tldw_chatbook/Local_Ingestion/PDF_Processing_Lib.py` — `docling_parse_pdf` (anchor `DocumentConverter()`)
- Test: `Tests/Local_Ingestion/test_docling_converter_singleton.py` (create)

**Interfaces:**
- Produces: `get_docling_converter() -> "DocumentConverter"` and `_reset_docling_converter_for_tests() -> None` in `PDF_Processing_Lib.py`.

- [ ] **Step 1: Failing test** (inject a fake module like B4's pattern — `sys.modules["docling"]`/`docling.document_converter` fakes with a counting `DocumentConverter`): two `docling_parse_pdf` calls (tiny in-memory PDF or a stubbed `converter.convert`) → 1 construction. Note: parsing happens inside process-pool workers in production; a per-process module singleton is the correct scope (each worker constructs once).
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.** Module state + `threading.Lock` (parse workers may also run threads), lazy import preserved (docling stays optional — keep the existing ImportError handling path so `docling_parse_pdf` still raises its current error when docling is missing). Preserve `enable_ocr` plumbing: if the converter construction differs per OCR flag today, key the singleton on that flag (dict keyed by flag) and note it.
- [ ] **Step 4: Run** — `pytest Tests/Local_Ingestion/test_docling_converter_singleton.py Tests/Local_Ingestion -k "pdf or docling" -q`.
- [ ] **Step 5: Evidence** — constructions per 3-PDF batch before (3) vs after (1).

**Acceptance criteria:** one converter per process per config; optional-dependency error path unchanged; existing PDF tests green.

---

### Task B14 (Wave D): Bounded-concurrency chunk analysis

**Files:**
- Modify: `tldw_chatbook/Local_Ingestion/PDF_Processing_Lib.py` — the analysis loop (anchor `for i, chunk in enumerate(processed_chunks)` near `analyze(`)
- Modify: `tldw_chatbook/Local_Ingestion/Book_Ingestion_Lib.py` — EPUB analysis loop (~:1348-1392, anchor `analyze(`) and the markup/text variant (~:2029)
- Modify: `tldw_chatbook/Local_Ingestion/Document_Processing_Lib.py` (~:265, same anchor)
- Test: `Tests/Local_Ingestion/test_analysis_concurrency.py` (create)

**Interfaces:**
- Produces: shared helper `analyze_chunks_concurrently(chunks: Sequence[str], analyze_fn: Callable[[str], str], max_workers: int = 3) -> list[str]` (module location: put it in `Local_Ingestion/local_file_ingestion.py` only if no circular import; otherwise duplicate a 15-line helper per file and note it — prefer one home: grep the import graph first). Order-preserving (result[i] corresponds to chunks[i]).

- [ ] **Step 1: Failing test**:

```python
"""B14: chunk analysis runs with bounded concurrency and preserves order."""
import threading
import time

from tldw_chatbook.Local_Ingestion import analysis_concurrency  # wherever the helper lands


def test_bounded_concurrency_and_order():
    live = []
    peak = [0]

    def slow_analyze(text: str) -> str:
        live.append(1)
        peak[0] = max(peak[0], len(live))
        time.sleep(0.05)
        live.pop()
        return text.upper()

    chunks = [f"chunk-{i}" for i in range(10)]
    out = analysis_concurrency.analyze_chunks_concurrently(chunks, slow_analyze, max_workers=3)
    assert out == [c.upper() for c in chunks]
    assert peak[0] <= 3
```

- [ ] **Step 2: Run, expect FAIL** (helper missing).
- [ ] **Step 3: Implement.** `concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)`; submit all, collect with index bookkeeping; propagate the first exception after all futures settle (today one chunk failure aborts the loop — keep that contract: record the exception and re-raise after join, matching current behavior's failure semantics; read each loop's current error handling and mirror it). Replace each serial loop with the helper. The recursive combine call after the loop stays serial (it depends on all results).
- [ ] **Step 4: Run** — `pytest Tests/Local_Ingestion/test_analysis_concurrency.py Tests/Local_Ingestion -k "analy" -q`.
- [ ] **Step 5: Evidence** — 30 chunks × 50 ms fake analyze: wall time serial (≈1.5 s) vs concurrent (≈0.55 s) in the test run; record.

**Acceptance criteria:** peak concurrency ≤ max_workers; per-chunk results order-identical to serial; failure semantics unchanged; existing ingestion tests green.

---

### Task B15 (Wave D): Bounded concurrency for the CLI batch-ingest seam

**Files:**
- Modify: `tldw_chatbook/Local_Ingestion/local_file_ingestion.py` — `batch_ingest_files` / `ingest_directory` fan-in (anchor `def batch_ingest_files`, the `for file_path in file_paths` loop)
- Test: `Tests/Local_Ingestion/test_batch_ingest_concurrency.py` (create)

**Interfaces:** none new (same signatures/return order).

- [ ] **Step 1: SAFETY GATE — verify thread-safety before writing code.** Read what `ingest_local_file` opens per call: DB connections, session state, tmp dirs. If it constructs its own DB connections per call (grep the DB singletons it uses), a bounded ThreadPoolExecutor is safe. If it shares module-level DB handles, DO NOT parallelize the whole call: parallelize only the parse/analyze sub-stage (via the B14 helper if importable) and record the limitation in notes. This gate is the task's first checkbox; paste the connection-lifecycle findings into the test docstring.
- [ ] **Step 2: Failing test** (assuming gate passes): stub `ingest_local_file` with a version that records concurrency + returns a per-file marker; call `batch_ingest_files` over 12 files; assert peak concurrency ≤ 4, results order matches input order, one failing file does not prevent others (collect its error entry as today).
- [ ] **Step 3: Implement.** ThreadPoolExecutor(max_workers=4); preserve result ordering and per-file error aggregation exactly (read the current loop's error handling first); directory walk/fan-in order unchanged.
- [ ] **Step 4: Run** — `pytest Tests/Local_Ingestion/test_batch_ingest_concurrency.py Tests/Local_Ingestion -k "batch or ingest" -q`.
- [ ] **Step 5: Evidence** — wall time for the 12-file stub before/after; concurrency trace.

**Acceptance criteria:** order and error semantics preserved; bounded workers; TUI queue path (process pool) untouched; if the safety gate failed, the documented partial fix lands instead with evidence.

---

### Task B16 (Wave D): Overlap scrape+summarize in the research gate

**Files:**
- Modify: `tldw_chatbook/Web_Scraping/WebSearch_APIs.py` — the research result loop (anchor `for idx, result in enumerate(search_results)`; sleeps at the `random.uniform(0.2, 0.6)` sites)
- Test: `Tests/Web_Scraping/test_research_gate_overlap.py` (create)

**Interfaces:** none new — same function signature and same returned result list (order and content).

- [ ] **Step 1: Failing test.** The function is async (verify — it awaits LLM calls). Stub the relevance LLM, scraper, and summarizer with delayed fakes recording concurrency:

```python
"""B16: relevance gating stays sequential; scrape+summarize overlap under a semaphore."""
import asyncio
import time

import pytest


@pytest.mark.asyncio
async def test_relevant_results_processed_concurrently(research_gate_factory, monkeypatch):
    gate, recorder = research_gate_factory(n_results=6, n_relevant=4, per_stage_delay=0.1)
    start = time.perf_counter()
    await gate.run("test query")
    elapsed = time.perf_counter() - start
    # 4 relevant results, stages overlapped under semaphore(3): bounded by ~ceil(4/3)*2*0.1 + gate time
    assert recorder.peak_scrape_summarize_concurrency <= 3
    assert elapsed < 0.1 * (4 * 2) + 1.0, f"stages appear serial: {elapsed:.2f}s"
    assert recorder.relevance_call_order == list(range(6)), "relevance gate must stay in result order"
    assert gate.results_order_preserved()
```

- [ ] **Step 2: Run, expect FAIL** (peak concurrency 1; elapsed ≈ serial).
- [ ] **Step 3: Implement.** Keep the per-result relevance gate loop exactly as is (it feeds spend decisions in order). Collect results judged relevant into a list; then `asyncio.gather` their (scrape → summarize) pipelines under `asyncio.Semaphore(3)`, each task writing its slot in a pre-sized result list (index-addressed, so order is preserved without sorting). Keep per-result error isolation: an exception in one slot logs and leaves that slot's existing error/placeholder shape exactly as the serial loop produces today (read the current `except` behavior and mirror it per slot). Keep the sleeps where they are deliberate rate-limiting (pre-relevance), drop none.
- [ ] **Step 4: Run** — `pytest Tests/Web_Scraping/test_research_gate_overlap.py Tests/Web_Scraping -k "research or gate or search" -q`.
- [ ] **Step 5: Evidence** — wall time for the 6-result fixture before/after in the test; record.

**Acceptance criteria:** relevance order sequential; scrape+summarize peak ≤ 3 concurrent; result list byte-equivalent on the no-failure fixture; failure in one relevant result doesn't affect siblings.

---

### Task B17 (Wave E): Activity log — debounce, render cap, cheap timestamp refresh

**Files:**
- Modify: `tldw_chatbook/Widgets/activity_log.py` (anchors `on_input_changed`, `_update_display`, `MAX_ENTRIES`, `set_interval(60`, `_update_timestamps`)
- Test: `Tests/UI/test_activity_log_perf.py` (create — follow existing Textual widget test patterns in `Tests/UI`, e.g. `run_test()` harness usage)

**Interfaces:** none new. The widget is currently latent (no mount site found) — keep the change proportionate: correctness of pattern over redesign.

- [ ] **Step 1: Failing tests**: (a) type 5 characters quickly → exactly one `_update_display` execution (debounce 0.3 s, `set_timer` re-arm — grep `SEARCH_DEBOUNCE` in `UI/Chatbooks_Window_Improved.py` or `library_note_autosave.py` for the house pattern); (b) with `MAX_ENTRIES` entries, `_update_display` mounts at most `_RENDER_CAP = 200` entry widgets (new constant; oldest dropped from view, data retention unchanged); (c) the 60 s timer (`set_interval(60, self._update_timestamps)`) updates the text of existing timestamp widgets instead of rebuilding (spy: no `mount` calls during timestamp refresh).
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.** Debounce via `self.set_timer(...)` handle-cancel pattern; `_create_entry_widget` loop bounded by `_RENDER_CAP`; `_update_timestamps` iterates rendered rows and patches their timestamp `Static` only.
- [ ] **Step 4: Run** — `pytest Tests/UI/test_activity_log_perf.py Tests/UI -k "activity" -q`.
- [ ] **Step 5: Evidence** — widget mounts per keystroke before (≤5000) vs after (≤1000 debounced, ≤200×5 rendered).

**Acceptance criteria:** one rebuild per settled query; render bounded; timestamp refresh does zero mounts; filters/`add_entry` paths still correct.

---

### Task B18 (Wave E): SmartContentTree — debounce, precomputed search text, worker load

**Files:**
- Modify: `tldw_chatbook/UI/Widgets/SmartContentTree.py` (anchors `on_input_changed`, `_apply_filters`, `searchable_text`, `on_mount` / `load_content_callback`)
- Test: `Tests/UI/test_smart_content_tree_perf.py` (create)

**Interfaces:**
- Produces: node payloads carry a precomputed `search_text: str` (built once when the node is added), replacing the per-keystroke metadata join.

- [ ] **Step 1: Failing tests**: (a) 5-char burst → one `_apply_filters` execution after debounce; (b) searchable text computed exactly once per node across a 3-keystroke session (spy on the join/metadata walk); (c) `on_mount` content load runs via `run_worker` (assert the callback is not invoked synchronously inside `on_mount` — capture the calling thread/frame or assert `is_worker` semantics per house pattern in neighboring widget tests).
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.** Compute `search_text` at node-creation (where metadata is attached); debounce 0.3 s with the `set_timer` pattern; `_apply_filters` reads the precomputed strings and only touches `node.display`/`expand` for visibility changes; move the `load_content_callback()` call into `self.run_worker(..., thread=True)` (or `run_worker` default async) preserving the post-load refresh.
- [ ] **Step 4: Run** — `pytest Tests/UI/test_smart_content_tree_perf.py Tests/UI -k "smart_content or content_tree" -q`, plus the wizard tests that exercise the tree (`Tests/ -k "chatbook_wizard" -q`).
- [ ] **Step 5: Evidence** — filter-passes per burst (5 → 1); string builds per node (3 → 1).

**Acceptance criteria:** per-keystroke work is O(matches) over precomputed strings; mount-time load off the UI thread; wizard behavior unchanged.

---

### Task B19 (Wave E): Config search — index once per pane, live value reads

**Files:**
- Modify: `tldw_chatbook/UI/Widgets/config_search_widget.py` (anchors `_build_index`, `search(`, `on_config_search_changed`) and its trigger site in `UI/Tools_Settings_Window.py` (~:6833-6860, anchor `on_config_search_changed`) — trigger-site change limited to passing/invalidating the cache
- Test: `Tests/UI/test_config_search_index_reuse.py` (create)

**Interfaces:**
- Produces: `UIElementSearchEngine` gains a reusable lifecycle — build once per active pane, `invalidate()` on pane switch (called from the trigger site), and `search()` reads live widget values from the cached index instead of re-walking the DOM.

- [ ] **Step 1: Failing tests**: (a) 3 keystrokes → exactly 1 `_build_index` execution (spy; the second full walk inside `search()` must also be gone); (b) value freshness: after typing into a setting Input post-index-build, a search filtering on that value still matches (proves live reads, not stale snapshots); (c) switching tabs invalidates (next search rebuilds once).
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.** Cache the index on the widget keyed by the active pane id; `search()` iterates cached `(widget, searchable_label)` pairs and matches against `widget.value` read at search time (handle the value-access variants across Input/Select/Checkbox/Static — read the current `_scan_widget` to enumerate them). Debounce 0.3 s with the standard timer pattern. Trigger site calls `invalidate()` when the active pane changes.
- [ ] **Step 4: Run** — `pytest Tests/UI/test_config_search_index_reuse.py Tests/UI -k "config_search or settings_search" -q`, plus `Tests/ -k "tools_settings" -q` if that suite exists.
- [ ] **Step 5: Evidence** — DOM walks per keystroke before (2 full scans) vs after (0).

**Acceptance criteria:** zero DOM scans per keystroke; value changes reflected in results without rebuild; pane switch rebuilds exactly once.

---

### Task B20 (Wave E): Repo tree — child index and direct checkbox handles

**Files:**
- Modify: `tldw_chatbook/Widgets/Coding_Widgets/repo_tree_widgets.py` — `select_node` and node storage (anchor `startswith(path + "/")`, `query_one(f"#select-`)
- Test: `Tests/UI/test_repo_tree_toggle.py` (extend or create)

**Interfaces:**
- Produces: `RepoTreeWidget` maintains `_children_by_dir: dict[str, list[str]]` (built when nodes are created in `_build_tree_nodes`) and stores each node's `Checkbox` reference on its node record at mount time.

- [ ] **Step 1: Failing tests**: (a) toggling a directory with 200 descendants performs 0 `query_one` calls (spy/monkeypatch on `query_one`) and O(descendants) dict lookups; (b) toggle state of all descendants flips exactly as before (golden fixture tree; compare checkbox states before/after against the pre-change behavior — capture baseline first).
- [ ] **Step 2: Run, expect FAIL** (query_one called per descendant).
- [ ] **Step 3: Implement.** Build the child index in `_build_tree_nodes`/`expand_node` (they already know parentage); store checkbox refs on nodes when mounted; `select_node` walks `_children_by_dir[path]` and sets `checkbox.value` directly. Keep the `#select-` id scheme intact for anything else that queries it.
- [ ] **Step 4: Run** — `pytest Tests/UI/test_repo_tree_toggle.py Tests/UI -k "repo_tree or coding" -q`.
- [ ] **Step 5: Evidence** — query_one calls per 200-descendant toggle before (200) vs after (0).

**Acceptance criteria:** identical toggle outcomes; no DOM queries in the toggle path.

---

### Task B21 (Wave E): Trajectory timeline — precomputed lanes, throttled drag

**Files:**
- Modify: `tldw_chatbook/UI/Widgets/trajectory_timeline.py` (anchors `def render`, `_lane_line`, `on_mouse_move`, `set_records`)
- Test: `Tests/UI/test_trajectory_timeline_perf.py` (create)

**Interfaces:**
- Produces: `set_records(...)` additionally builds `_lane_records: list[list[Record]]` (len = LANE_COUNT) and `_record_by_key: dict[Any, Record]`; `render`/`_lane_line` consume only these. `on_mouse_move` coalesces refreshes to ≤ 1 per ~33 ms.

- [ ] **Step 1: Failing tests**: (a) after `set_records`, `render` performs no linear `next(... for ... in model.timed_records)` scans (structure-level assert: monkeypatch `builtins.next` is too broad — instead assert `render` completes in O(lanes) by counting `_record_columns` invocations ≤ lanes × max-per-lane, with a fixture sized so the old code's lanes×n loop would exceed it); (b) a 60-event `on_mouse_move` burst triggers ≤ 3 full refreshes (throttle).
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.** Precompute in `set_records`; boundary lookup uses `_record_by_key`; drag handler stores pending position and schedules one `refresh()` via `set_timer`/`call_later` guard if none pending. Preserve exact render output (golden: render string before/after on a fixture must be identical).
- [ ] **Step 4: Run** — `pytest Tests/UI/test_trajectory_timeline_perf.py Tests/UI -k "trajectory" -q`.
- [ ] **Step 5: Evidence** — per-frame record-visit count and refreshes per drag burst before/after.

**Acceptance criteria:** identical rendered output; no O(n) scans per frame; drag repaints coalesced.

---

### Task B22 (Wave E): Chatbooks grid — paged initial render

**Files:**
- Modify: `tldw_chatbook/UI/Chatbooks_Window_Improved.py` — `_update_content` (anchor `grid.mount` / `ChatbookCard(`)
- Test: `Tests/UI/test_chatbooks_grid_paging.py` (create)

**Interfaces:**
- Produces: module constant `CHATBOOK_RENDER_PAGE_SIZE = 60`; after the first page, a "Load more" control renders the next page into the same container. Filter/search path unchanged (it re-runs `_update_content` with the new match set).

- [ ] **Step 1: Failing tests**: (a) 150 matching chatbooks → exactly 60 cards mounted + a load-more control; (b) activating load-more mounts the next 60 (cumulative 120) and it disappears after the last page; (c) a filter change resets paging (back to first 60 of the new match set). Follow the Library canvas pager test patterns (grep `Tests/UI -k "pager or load_more"`).
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.** Keep the full match list in state; mount slices; reuse the house pager-layout widget if importable (grep `library_pager_layout`) — else a simple button. Preserve the list-mode branch with the same cap.
- [ ] **Step 4: Run** — `pytest Tests/UI/test_chatbooks_grid_paging.py Tests/UI -k "chatbook" -q`.
- [ ] **Step 5: Evidence** — widget mounts per search application before (n) vs after (≤ 60).

**Acceptance criteria:** render cost bounded per search; full list reachable through paging; search/filter behavior unchanged.

---

### Task B23 (Wave F): Video store — one snapshot per save; reuse RecoveredMedia

**Files:**
- Modify: `tldw_chatbook/Video_Generation/video_store.py` (anchors `_cleanup_orphan_stages_unlocked`, `_ensure_slug_absent`, `_enforce_save_capacity`, `_snapshot`, `RecoveredMedia(self._recovered_root)` in `resolve_state`, and the `allocate_slug` loop)
- Test: `Tests/Video_Generation/test_video_store_snapshot_reuse.py` (create)

**Interfaces:** none new — same public store API; snapshot/recovery behavior internally consolidated.

- [ ] **Step 1: Failing tests**: (a) one `save()` on a populated store takes ≤ 2 full `_snapshot()` passes (spy/count; today 3–4+); (b) `resolve_state` with an existing recovery catalog constructs `RecoveredMedia` at most once per call (monkeypatch-count constructor; today once per resolve — and `allocate_slug` under collision triggers it repeatedly); (c) end-state equivalence: after `save()` + `resolve_state()`, on-disk layout and returned metadata identical to the pre-change implementation on a fixture (golden baseline captured first).
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.** In `save()`: take one snapshot after the write, derive `_cleanup_orphan_stages_unlocked` and `_enforce_save_capacity` decisions from it (pass the snapshot as a parameter; keep the unlocked-function shapes, they already take internal state — read signatures first). Cache the `RecoveredMedia` instance on the store per operation: construct once at the top of `resolve_state`/`allocate_slug` scope and reuse across iterations; scope it per-call (do NOT cache across calls — the catalog can change underneath via other processes). Preserve lease/lock semantics exactly (all mutations stay under the same exclusive lease).
- [ ] **Step 4: Run** — `pytest Tests/Video_Generation/test_video_store_snapshot_reuse.py Tests/Video_Generation -q`.
- [ ] **Step 5: Evidence** — store walks per save before (3–4) vs after (≤ 2); RecoveredMedia constructions per slug allocation before (up to 100s) vs after (1).

**Acceptance criteria:** identical on-disk outcomes; syscall multiplication gone; lease behavior unchanged; existing video store tests green.

---

### Task B24 (Wave F): Fork-commit verification in batched queries

**Files:**
- Modify: `tldw_chatbook/Chat/chat_persistence_service.py` — `resolve_console_fork_commit` / `_recheck_fork_source` / `_fork_source_parent` / lineage walk (~:2088-2238). **Do not touch `save_history`** (~:4040+, sibling deletes it).
- Test: `Tests/Chat/test_fork_commit_batched.py` (create)

**Interfaces:** none new — same function signatures and decision outputs.

- [ ] **Step 1: Golden baseline.** Seed a 300-message conversation; run the fork-commit resolution; capture the per-message decisions (verified/skipped/reason codes) as the golden fixture. Count SELECT statements via trace callback.
- [ ] **Step 2: Failing test** — assert statement count ≤ 5 for the 300-message case and decisions equal the golden fixture.
- [ ] **Step 3: Implement.** One `SELECT ... FROM messages WHERE id IN (...)` (chunked at 500 ids per the house pattern — grep `IN` chunking in ChaChaNotes_DB, e.g. `get_message_version_batch`) builds `id → row`; the parent-hop walk (`_fork_source_parent`) uses the same map instead of per-hop SELECTs, preserving the 10,000-hop cap semantics exactly (hop count accounting included). Any rows not in the batch (deleted mid-transaction edge) fall back to the original single-row SELECT for just those ids.
- [ ] **Step 4: Run** — `pytest Tests/Chat/test_fork_commit_batched.py Tests/Chat -k "fork" -q`.
- [ ] **Step 5: Evidence** — SELECTs per 300-message fork before (~600–900) vs after (≤ 5).

**Acceptance criteria:** identical decisions; statement count bounded; hop cap preserved; existing fork tests green.

---

### Task B25 (Wave F): Bulk path for chat-history import

**Files:**
- Modify: `tldw_chatbook/Character_Chat/Character_Chat_Lib.py` — the import replay loop (~:3336-3342, anchor `for staged in staged_messages`)
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py` — add `add_message_import_batch(staged_messages) -> list[str]` next to `add_message` **only if** Step 1 shows the sidecar path is safely batchable; otherwise keep the loop and batch at the transaction level (see below)
- Test: `Tests/Character_Chat/test_history_import_bulk.py` (create)

**Interfaces:**
- Produces (conditional): `ChaChaNotesDB.add_message_import_batch(staged: Sequence[StagedMessage]) -> list[str]` returning message ids in order, producing byte-identical rows (messages + semantic-revision sidecars) as repeated `add_message` calls.

- [ ] **Step 1: Feasibility read.** Trace `add_message` → `_add_message_with_semantic_sidecars` (~:13078): enumerate every row written per message (messages, revision/sidecar tables, FTS touch). If all writes are pure functions of the staged message + parent id chain, implement the batch method with `executemany` per table inside ONE transaction. If any write depends on per-message read-back (e.g. revision numbering reads prior state), batch per-chunk (500) within one transaction using the already-fetched state, or fall back to the loop with per-chunk transactions and skip-per-message-validation (hoist validation to the loop head). Record the chosen strategy and why in the test docstring.
- [ ] **Step 2: Equivalence test FIRST (against the current loop).** Import a 200-message fixture through the current path into DB-A and (after the change) through the new path into DB-B; assert `SELECT * FROM messages` and the sidecar/revision tables are row-for-row identical (ids, order, all columns). This is the acceptance oracle.
- [ ] **Step 3: Implement** per the Step 1 strategy; keep the 10,000 cap and all existing import validation/error semantics (a malformed message fails the import the same way).
- [ ] **Step 4: Run** — `pytest Tests/Character_Chat/test_history_import_bulk.py Tests/Character_Chat -k "import or history" -q`, plus `Tests/ChaChaNotesDB -k "add_message" -q`.
- [ ] **Step 5: Evidence** — SQL statements per 1,000-message import before (~3,000+) vs after (per chosen strategy); wall time if convenient.

**Acceptance criteria:** DB state identical to the per-message path; cap and error semantics preserved; statement count materially reduced.

---

### Task B26 (Wave F): Pipeline-config deep copy + OpenRouter chunk-log removal

**Files:**
- Modify: `tldw_chatbook/Event_Handlers/Chat_Events/chat_rag_events.py` — the three `BUILTIN_PIPELINES[...].copy()` sites (anchors `BUILTIN_PIPELINES["plain"].copy()`, `BUILTIN_PIPELINES["semantic"].copy()`, `BUILTIN_PIPELINES["hybrid"].copy()`; the nested mutation is `step.setdefault("config", {})["model"] = reranker_model`)
- Modify: `tldw_chatbook/LLM_Calls/Summarization_General_Lib.py` — remove the per-chunk log line (anchor `OpenRouter Stream: Content received`)
- Test: `Tests/Event_Handlers/test_builtin_pipeline_isolation.py` (create)

**Interfaces:** none new.

- [ ] **Step 1: Failing test** for the isolation bug:

```python
"""B26: per-search reranker overrides must not leak into the module-global pipeline table."""
from tldw_chatbook.Event_Handlers.Chat_Events import chat_rag_events as cre
from tldw_chatbook.RAG_Search.pipeline_builder_simple import BUILTIN_PIPELINES


def test_reranker_override_does_not_mutate_global(smoke_monkeypatches):
    before = {name: [{k: dict(v.get("config", {})) for k, v in step.items()} if isinstance(step, dict) else step
                     for step in cfg["steps"]]
              for name, cfg in ((n, BUILTIN_PIPELINES[n]) for n in ("plain", "semantic", "hybrid"))}
    cre._build_rag_pipeline(reranker_model="nondefault-model")  # adapt to the actual builder entry
    after = {name: [{k: dict(v.get("config", {})) for k, v in step.items()} if isinstance(step, dict) else step
                    for step in cfg["steps"]]
             for name, cfg in ((n, BUILTIN_PIPELINES[n]) for n in ("plain", "semantic", "hybrid"))}
    assert before == after, "module-global BUILTIN_PIPELINES was mutated by a per-search override"
```

  (Read the actual build entry points at ~:201/:253/:364 and adapt; the assertion — deep-equality of the global table before/after — is the point.)
- [ ] **Step 2: Run, expect FAIL** (table mutated).
- [ ] **Step 3: Implement.** Replace `.copy()` with `copy.deepcopy(BUILTIN_PIPELINES[...])` at the three sites (import `copy`). Delete the `logging.info("OpenRouter Stream: Content received")` line inside the per-record loop (the sibling streams log once per stream — match that: if a per-stream completion log exists nearby, none needed; else leave logging as-is minus the loop line).
- [ ] **Step 4: Run** — `pytest Tests/Event_Handlers/test_builtin_pipeline_isolation.py Tests/Event_Handlers -k "rag or pipeline" -q` plus `Tests/ -k "summarization" -q` targeted.
- [ ] **Step 5: Evidence** — log-line count per streamed summarize before (n chunks) vs after (0); mutation test green.

**Acceptance criteria:** global table immutable across searches; per-chunk log line gone; existing RAG-event and summarization tests green.

---

### Task B27 (Wave F): Loop-invariant micro pack

**Files:**
- Modify: `tldw_chatbook/Chat/llamacpp_think_filter.py` (~:183-200, anchor `ord(text[index])`) — replace the per-character Python byte-count loop with slice-level `len(text[start:end].encode("utf-8"))` and a module-level compiled surrogate-range pattern for the surrogate check. Golden equivalence test: keep a copy of the old function in the test module and assert old == new over a corpus including ASCII, 2/3/4-byte chars, astral chars, and `surrogateescape` bytes (`b"\xff".decode("utf-8", "surrogateescape")`). Run: `pytest Tests/Chat -k "think_filter or llamacpp" -q`.
- Modify: `tldw_chatbook/Local_Ingestion/Book_Ingestion_Lib.py` (~:692, anchor `get_item_with_id`) — build `{item.id: item}` once before the spine loop; assert in a test (extend an existing book-ingestion test) that extraction output is unchanged for a 2-book fixture. Run: `pytest Tests/Local_Ingestion -k "book or epub" -q`.
- Modify: `tldw_chatbook/Tools/local_tool_impls.py` (~:1225, anchor `resolved_workspace = workspace.resolve()`) — hoist `workspace.resolve()` out of the per-entry check and pass it in; extend a tool test asserting identical accept/reject decisions over a path list before/after. Run: `pytest Tests/Tools -k "grep or local_tool" -q`.

**Interfaces:** none. Each is a one-commit-sized change with its equivalence test; evidence = the equivalence corpus results (these are correctness-preserving rewrites — the test IS the evidence).

**Acceptance criteria:** all three equivalence tests green; no behavior change.

---

## Verification strategy (applies to every task)

1. Targeted pytest runs only (module-scoped), per `AGENTS.md` — no full sweeps without explicit request.
2. Every perf claim carries a counted measurement (spy counts, trace callbacks, timers) in the task notes — "feels faster" is not evidence.
3. Any task touching DB schema (B7) verifies against a fixture DB created BEFORE the change (migration reach) — follow the pattern investigation in B7 Step 1.
4. UI tasks (B17–B22) run under the repo's Textual test harness patterns; no live app runs required for this PR (all changes are logic-level; live verification is deferred to the pre-merge checklist if the reviewer wants it).
5. Sibling-PR exclusions (Global Constraints) are checked with a final `git diff --name-only` review before push.

## Suggested backlog instantiation

One backlog task for this PR (the wave set is one coherent remediation), with the AC checklist mirroring the per-task acceptance criteria above. Waves A–F map to commits: `perf(rag): ...`, `perf(evals): ...`+`perf(db): ...`, `perf(notes): ...`, `perf(ingestion): ...`+`perf(search): ...`, `perf(ui): ...`, `perf(stores): ...`+`perf(chat): ...`+`perf(micro): ...`.
