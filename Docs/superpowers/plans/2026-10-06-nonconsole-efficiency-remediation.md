# Non-Console Efficiency Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the 25 speed/efficiency defects identified in the 2026-10-06 non-Console efficiency review (Tiers 1–3): event-loop blocking, whole-corpus scans, per-message cold starts, per-chunk copy taxes, and unbounded polls — without changing user-visible behavior or retrieval semantics.

**Architecture:** Independent targeted fixes grouped into 7 dependency-ordered waves. Each task is atomic (one PR), testable, and preserves existing interfaces unless the task explicitly introduces a new seam (injection caches, session registry, embedding cache, immutable page records). Fixes land on `origin/dev`; the Console send-path seam (`Chat/console_chat_controller.py`) is touched by exactly one optional subtask (T3e) that must be sequenced with the in-flight console pipeline maintenance.

**Tech Stack:** Python ≥3.12, Textual 8.x, SQLite (WAL, FTS5) with versioned migrations, ChromaDB (optional), httpx/requests, pytest (in-memory SQLite fixtures).

**Spec:** The findings index in this document (§Findings) is the spec. All line numbers refer to `origin/dev @ cddc89d3e7` (worktree `/tmp/tldw-dev-review`); they WILL drift — each task lists a grep anchor to relocate the site. Evidence excerpts live in the review session record.

## Global Constraints

- Base every branch on latest `origin/dev`, never on `chore/*` branches carrying console-maintenance WIP.
- Per `AGENTS.md`: schema changes always bump the schema version and add a `migrations/` file; verify the current version by listing `migrations/` (review saw `chachanotes_v48_to_v49…`, so local-file claims of "v37" are stale).
- Per `AGENTS.md`: verify with targeted pytest runs of touched modules only; full sweeps only on explicit request.
- Performance claims need evidence per `backlog/docs/lessons-testing-evidence.md`: a before/after measurement (criterion-style timing or counted syscalls/queries) in the task notes, not just "faster".
- Never weaken search semantics to win speed unless the task explicitly says so (T2 keeps substring semantics; T15d keeps ranking order).
- No new settings surface outside `UI/Screens/settings_screen.py` if config knobs are added.
- Design-token rules (ADR-150) do not apply: no UI styling is changed.

## ADR check (repo requirement)

```text
ADR required: yes
ADR path: backlog/decisions/212-prompt-injection-cold-start-caches.md        (created in Wave 2, before T3)
          backlog/decisions/213-persistent-embedding-content-hash-cache.md   (created in Wave 4, before T8)
          backlog/decisions/214-provider-http-session-reuse.md               (created in Wave 3, before T6)
          backlog/decisions/215-conversation-timestamp-normalization.md      (created in Wave 6, before T15)
Reason: injection caches define cross-module invalidation contracts; the embedding cache adds
storage/schema; session reuse changes provider runtime resource lifecycle; T15 migrates stored
data format + schema. All other tasks are mechanical performance fixes inside existing
boundaries (no ADR).
```

## Findings index (spec)

| ID | Tier | Finding | Anchor file @ dev |
|----|------|---------|-------------------|
| F1 | 1 | FlashRank ranker rebuilt per query, sync on event loop | `RAG_Search/pipeline_functions_simple.py` (`Ranker(model_name=`) + `pipeline_builder_simple.py:107` |
| F2 | 1 | Library conversation search: correlated `content LIKE` EXISTS | `DB/ChaChaNotes_DB.py` (`_escape_library_conversation_like` region) |
| F3 | 1 | World-info full cold start per send; double `_process_entry`; per-key pattern rebuild; double collection/turn | `Character_Chat/world_info_resolver.py`, `world_book_manager.py`, `world_info_processor.py` |
| F4 | 1 | Dictionaries cold start per send (×2/turn); per-replacement `re.compile` | `Character_Chat/Chat_Dictionary_Lib.py` |
| F5 | 2 | `get_conversation_tree` unbounded full-conversation read | `DB/ChaChaNotes_DB.py` (`get_message_tree_rows_for_conversation`) |
| F6 | 2 | No embedding content-hash cache; dead cache check; global embed lock; N+1 `needs_reindexing` | `RAG_Search/simplified/embeddings_wrapper.py`, `Embeddings/Embeddings_Lib.py`, `RAG_Search/ingestion_indexing.py` |
| F7 | 2 | Legacy RAG-context store whole-file rewrite per message | `Chat/chat_conversation_service.py` (`_save_rag_context_store`) |
| F8 | 2 | Hosted streaming: 5–9 deepcopies + double JSON round-trip per chunk; full-payload deepcopy per attempt; no session pooling | `LLM_Calls/hosted_provider_engine.py`, `hosted_chat.py`, `legacy_line_stream.py`, `LLM_API_Calls.py` |
| F9 | 2 | Buddy modal 5 Hz O(session) poll + 64 KB rebuild-then-compare | `Widgets/Persona_Widgets/buddy_conversation_modal.py` |
| F10 | 2 | File Notes 1.5 s unbounded walk (2 lstats/file) + full replica read; unbounded session-change list | `Widgets/Library/library_file_notes_workspace.py`, `Notes/file_notes_service.py`, `Notes/file_notes_session_owner.py` |
| F11 | 3 | Non-sargable SQL: `julianday` keyset, `datetime(next_review)`, `json_extract` visibility + NOCASE sort | `DB/ChaChaNotes_DB.py:11210/21601/9177` regions |
| F12 | 3 | Recursive world-info rescan + O(matched²) dedup | `world_info_processor.py` (`recursive_scanning`) |
| F13 | 3 | Visual identity double full decode + 4× portrait re-hash | `Character_Chat/visual_identity.py`, `persona_visual_identity.py` |
| F14 | 3 | Personas conv-search: no debounce, write-txn per keystroke; media debounce 0.12 s | `Widgets/Persona_Widgets/personas_inspector_pane.py`, `UI/Library_Modules/library_media_controller.py`, `DB/character_conversation_search.py` |
| F15 | 3 | `ReaderItemSnapshot` deepcopies all cached pages per page turn | `UI/Watchlists_Modules/reader_item_snapshot.py` |
| F16 | 3 | Pairwise reranker sequential LLM comparisons | `RAG_Search/reranker.py` (`_merge_sort` region) |
| F17 | 3 | Chroma per-doc `where` delete → O(N×C) reindex | `RAG_Search/ingestion_indexing.py`, `simplified/vector_store.py` |
| F18 | 3 | Dead/landmined code: `save_history`, tool-widget dead branch + WrongType bug, legacy streaming entries, TTS full-DOM queries | `Chat/chat_persistence_service.py:4040+`, `Widgets/tool_message_widgets.py`, `app_speech.py` |
| F19 | 3 | Misc: fork double-fetch; CitationTraceBuilder re-encode; lore N+1; chatbooks registry re-parse; unconditional debug loop; post-gen dict disk re-read | see T20 anchors |

---

## Wave overview (dependency order)

| Wave | Tasks | Theme | Ships value |
|------|-------|-------|-------------|
| 1 | T1, T2 | Unblock the event loop + kill the worst scan | Immediately felt latency on every RAG send and Library search |
| 2 | T3, T4 | Prompt-injection cold-start caches (ADR-212) | Per-send latency for lorebook/dictionary users |
| 3 | T5, T6 | LLM streaming copy tax + connection reuse (ADR-214) | Streaming CPU + per-turn handshake |
| 4 | T7, T8, T9 | Scaling reads/writes: tree LIMIT, embedding cache (ADR-213), legacy store batching | Long-conversation open, re-ingest, chatbook import |
| 5 | T10–T14 | UI poll hygiene | Idle CPU, notebook battery, typing latency |
| 6 | T15–T18 | SQL sargability (ADR-215), identity decode, reranker concurrency, Chroma batch delete | Library/character scale behavior |
| 7 | T19–T21 | Small fixes, dead-code removal, docs/lessons | Debt reduction |

Waves 1–3 are independent of each other and can be parallelized on separate branches. Within a wave, tasks are independent except where an Interfaces block says otherwise (T3/T4 share the cache pattern; T15's migration is a prerequisite for none but touches the same DB file as T7 — land T7 first to avoid migration conflicts).

---

### Task 1 (F1): FlashRank ranker singleton + off-loop execution

**Files:**
- Modify: `tldw_chatbook/RAG_Search/pipeline_functions_simple.py` (rerank function; anchor `grep -n "Ranker(model_name" `)
- Modify: `tldw_chatbook/RAG_Search/pipeline_builder_simple.py:107` (anchor `grep -n "_execute_process_step(step_config, context)"`)
- Test: `Tests/RAG_Search/test_flashrank_ranker_reuse.py` (create)

**Interfaces:**
- Produces: `get_flashrank_ranker() -> "Ranker"` (lazy module singleton, thread-safe) and `_reset_flashrank_ranker_for_tests()` in `pipeline_functions_simple.py`; `async def _execute_process_step(...)` (becomes awaitable) in `pipeline_builder_simple.py`. All existing `_execute_process_step` call sites must be updated to `await` in the same task.

- [ ] **Step 1: Write failing tests** (no flashrank dependency needed — inject a fake via the factory seam)

```python
"""T1: FlashRank ranker is constructed once and rerank work runs off the event loop."""
import asyncio
import threading

import pytest

from tldw_chatbook.RAG_Search import pipeline_functions_simple as pfs


class _FakeRanker:
    instances = 0
    def __init__(self, *a, **k):
        type(self).instances += 1
    def rerank(self, req):
        return sorted(req.passages, key=lambda p: p["text"])


@pytest.fixture(autouse=True)
def _fake_flashrank(monkeypatch):
    monkeypatch.setattr(pfs, "_RANKER_FACTORY", _FakeRanker)
    pfs._reset_flashrank_ranker_for_tests()
    yield
    pfs._reset_flashrank_ranker_for_tests()


def test_ranker_constructed_once_across_calls():
    results = [{"title": "b", "content": "x", "score": 1.0}, {"title": "a", "content": "y", "score": 0.9}]
    pfs._rerank_results(results, query="q", model="flashrank")   # adjust to actual signature
    pfs._rerank_results(results, query="q", model="flashrank")
    assert _FakeRanker.instances == 1


def test_rerank_runs_off_event_loop_thread():
    seen_threads: list[threading.Thread] = []
    orig = _FakeRanker.rerank
    def spy(self, req):
        seen_threads.append(threading.current_thread())
        return orig(self, req)
    _FakeRanker.rerank = spy
    try:
        async def run():
            results = [{"title": "t", "content": "c", "score": 1.0}]
            return await pfs._rerank_results_async(results, query="q", model="flashrank")
        asyncio.run(run())
    finally:
        _FakeRanker.rerank = orig
    assert seen_threads and seen_threads[0] is not threading.main_thread()
```

- [ ] **Step 2: Run tests, expect FAIL** — `pytest Tests/RAG_Search/test_flashrank_ranker_reuse.py -v` → `AttributeError: _RANKER_FACTORY` / `ImportError` on the async wrapper.
- [ ] **Step 3: Implement.** In `pipeline_functions_simple.py`: add module state `_RANKER_LOCK = threading.Lock()`, `_CACHED_RANKER = None`, `_RANKER_FACTORY = None` (resolved lazily to the real `flashrank.Ranker` on first use so import stays optional); `get_flashrank_ranker()` returns the singleton under the lock, instantiating via `_RANKER_FACTORY or _real_flashrank_ranker`. Replace the per-call `ranker = Ranker(...)` with `ranker = get_flashrank_ranker()`. Change the cache dir from `/tmp` to the app cache dir (`platformdirs`-equivalent already used elsewhere — grep `cache_dir` in the repo and reuse that helper) so model weights survive reboot. Add `_rerank_results_async` = `await asyncio.to_thread(_rerank_results, ...)`. In `pipeline_builder_simple.py`: make `_execute_process_step` async (`async def`) and `await asyncio.to_thread` any CPU-heavy process step bodies (rerank at minimum); update line 107 call site to `results = await _execute_process_step(step_config, context)`.
- [ ] **Step 4: Run tests, expect PASS** — `pytest Tests/RAG_Search/test_flashrank_ranker_reuse.py Tests/RAG_Search -k "pipeline or rerank" -v`.
- [ ] **Step 5: Evidence + commit.** Time one rerank step twice in a scratch script (cold vs warm): record numbers in the task notes. `git commit -m "perf(rag): reuse FlashRank ranker and run rerank off the event loop"`.

**Acceptance criteria:** ranker constructed once per process; rerank/model-load never runs on the Textual event loop; existing pipeline tests green; warm-vs-cold measurement recorded.

---

### Task 2 (F2): Uncorrelate Library conversation content search

**Files:**
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py` — `search_library_conversations_page` (anchor `grep -n "EXISTS (SELECT 1 FROM messages m" DB/ChaChaNotes_DB.py`)
- Test: `Tests/ChaChaNotesDB/test_library_conversation_search.py` (extend existing file if present)

**Interfaces:** none (pure SQL rewrite, same signature/return shape).

- [ ] **Step 1: Write equivalence tests first** — build in-memory DB; seed 30 conversations × 40 messages each with known content; assert the same result set and `hit_N` projection semantics for: (a) title exact, (b) title substring, (c) message **mid-word substring** (`LIKE` semantics — this is the branch we must preserve), (d) FTS token match, (e) keyword match, (f) no-match → empty page, count 0.
- [ ] **Step 2: Run, expect PASS on old code** (golden baseline — capture actual result tuples as fixtures).
- [ ] **Step 3: Rewrite the branch.** Replace:

```sql
"EXISTS (SELECT 1 FROM messages m "
"WHERE m.conversation_id = conversations.id AND m.deleted = 0 "
"AND m.content LIKE ? ESCAPE '\\')"
```

with the uncorrelated shape already proven on the Console seam (`_conversation_search_filter`, ~line 11457):

```sql
"id IN (SELECT m.conversation_id FROM messages m "
"WHERE m.deleted = 0 AND m.content LIKE ? ESCAPE '\\')"
```

Keep parameter order and `message_hit_indexes` identical (branch index 2 unchanged). The COUNT, page query, and per-row `hit_2` projection all inherit the fix from the shared `branches` list. Add a header comment: one bounded pass over `messages` per search instead of a per-conversation correlated scan; state the rule ("never a leading-wildcard LIKE inside a correlated EXISTS") and cite TASK-34414 plus the sibling precedent task-33261/PERF-02 (whose ~70 s/150k measurement was on the correlated FTS-MATCH EXISTS variant — attribute it as such or use this branch's own measured numbers).
- [ ] **Step 4: Run tests** — equivalence suite must produce the golden baseline exactly. `pytest Tests/ChaChaNotesDB/test_library_conversation_search.py -v`.
- [ ] **Step 5: Evidence + commit.** Count statements/scans: run the search once against a seeded 100k-message fixture with `sqlite3` `set_trace_callback` and record the query text (no correlated EXISTS present). `git commit -m "perf(db): uncorrelate library conversation content search"`.

**Acceptance criteria:** byte-identical results/projections vs baseline; correlated EXISTS gone from the query plan; note-search LIKE branch intentionally untouched (single uncorrelated scan, deliberate substring semantics — record this decision in the task notes).

---

### Task 3 (F3): World-info injection cache (ADR-212 required first)

**Files:**
- Create: `tldw_chatbook/backlog/decisions/212-prompt-injection-cold-start-caches.md` (before code)
- Modify: `tldw_chatbook/Character_Chat/world_book_manager.py` (generation counter on every write method; anchor `grep -n "def get_world_books_for_conversation"`)
- Modify: `tldw_chatbook/Character_Chat/world_info_resolver.py:97-108` (processor cache)
- Modify: `tldw_chatbook/Character_Chat/world_info_processor.py` (precompiled keys; single `_process_entry`; id-set dedup — this also fixes F12)
- Test: `Tests/Character_Chat/test_world_info_injection_cache.py` (create)

**Interfaces:**
- Produces: `WorldBookManager.generation: int` (monotonic, bumped by every mutating method); `resolve_world_info(..., books=None)` accepts pre-collected books (used by T3e); entry dicts gain `compiled_primary_keys: tuple[re.Pattern, ...]` / `compiled_secondary_keys` (internal to processor).

Subtasks (each independently testable):

- [ ] **3a — Generation counter.** Add `self._generation = 0`; bump in every write path (create/update/delete book + entries). Test: read `generation`, mutate, assert increment; unchanged reads leave it stable.
- [ ] **3b — Resolver cache.** Key: `(conversation_id, character_id, manager.generation)`. Value: the constructed `WorldInfoProcessor` + fetched books. Invalidation falls out of the key. Bound: `functools.lru_cache`-style dict capped at 8 conversations (LRU evict). Test: two sends with no book edits → manager queried once (monkeypatch-count `get_world_books_for_conversation` calls); edit a book → refetched.
- [ ] **3c — Precompile keys once.** In `_process_entry`, store `re.compile(r"\b" + re.escape(k) + r"\b")` per key; `_keyword_in_text` uses the stored pattern (fall back to a module-level `@lru_cache(maxsize=4096) _compiled_keyword(keyword)` for any ad-hoc call sites, which also survives the 512-slot `re` module cache). Same for `world_info_regex.py` regex entries (compile once at process time).
- [ ] **3d — Kill double processing + O(matched²) dedup.** `_make_candidate` reuses the already-processed entry (no second `_process_entry`); recursion dedup becomes `seen = {e["uid"] for e in matched}` membership checks. Tests: entry-processing called exactly once per entry per build (spy); recursion with overlapping matches produces the same activation set as before (golden fixture with a small recursive book).
- [ ] **3e — (Optional, coordinate with console maintenance)** Pass the books already collected by `capture_prompt_transform_inputs` (`Chat/console_chat_controller.py:21581`) into the world-info applier via the new `books=` parameter so the turn collects once. This file is under console-pipeline maintenance — land only after coordinating, else defer to a follow-up task.

**Steps per subtask:** failing test → implement → targeted run (`pytest Tests/Character_Chat/test_world_info_injection_cache.py Tests/Character_Chat -k "world_info" -v`) → commit (`perf(character): cache world-info processor across sends` etc.).

**Acceptance criteria:** with an unchanged 1000-entry book, second send performs zero book queries and zero `re.compile` calls (assert via spies); activation results identical to pre-change golden fixtures; ADR-212 written and linked.

---

### Task 4 (F4): Chat-dictionary injection cache (same ADR-212)

**Files:**
- Modify: `tldw_chatbook/Character_Chat/Chat_Dictionary_Lib.py` — `_resolve_active_dictionaries` (1320–1383), `_compile_key_internal` (175–235), `match_whole_words` (618–647), `apply_replacement_once` (666–694); local/server dictionary services for generation counters (`local_chat_dictionary_service.py`, `server_chat_dictionary_service.py`)
- Test: `Tests/Character_Chat/test_chat_dictionary_injection_cache.py` (create)

**Interfaces:**
- Produces: compiled-key cache on `ChatDictionary` instances (`compiled_pattern` attribute, lazily built once); dictionary-store generation counters mirroring T3a; cached resolved-dictionary bundles keyed by `(conversation_id, store_generation)`.

- [ ] **4a — Store generation counters** on both dictionary services (same pattern as 3a).
- [ ] **4b — Cache `_resolve_active_dictionaries`** bundles (LRU 8); second send with no dictionary edits → zero DB loads and zero `ChatDictionary.from_dict` re-instantiations (spy-assert).
- [ ] **4c — Compile once:** `_compile_key_internal` result stored on the entry (regex validation is the expensive tail — it must run once per entry lifetime per generation, not per send). `match_whole_words` uses module-level `@lru_cache(maxsize=4096)` compiled literal patterns. `apply_replacement_once` hoists `re.compile` out of the `max_replacements` loop (compile once, then `pattern.subn` with count=1 in the loop, or track positions).
- [ ] **4d — Same console-seam note as 3e:** eliminate the double per-turn collection via the frozen-inputs bundle.

**Acceptance criteria:** unchanged dictionary set → second send does no DB reads, no JSON parsing, no regex compiles (spies); replacement output byte-identical on a golden fixture exercising literal keys, regex keys, `max_replacements`, and whole-word boundaries.

---

### Task 5 (F8a): Strip hosted-streaming deepcopy chain

**Files:**
- Modify: `tldw_chatbook/LLM_Calls/hosted_provider_engine.py:1749` (anchor `grep -n "deepcopy(next(self._stream))"`) and `_normalize_messages`/`_normalize_tools` (1146–1239)
- Modify: `tldw_chatbook/LLM_Calls/hosted_chat.py` — `_filtered_event`/`_filtered_choice`/`_filtered_tool_call` (407–472)
- Test: `Tests/LLM_Calls/test_hosted_streaming_copy_tax.py` (create)

**Interfaces:** none — internal representation unchanged.

- [ ] **Step 1: Mutation-isolation test.** Feed a synthetic event stream through the hosted engine; after the consumer mutates every yielded event deeply (set nested dict values), assert the *next* yielded event and the engine's internal state are unaffected. This pins the isolation contract that deepcopy currently provides.
- [ ] **Step 2: Replace deepcopies with per-level fresh shallow dicts.** In `_filtered_event`: `safe = {k: v for k in _KNOWN_TOP_LEVEL_KEYS if k in event}`; for `choices`/`delta`/`tool_calls`/`function` levels build `dict(...)` fresh copies exactly as the current code does, but with `dict()` instead of `deepcopy()` — leaf values at these levels are strings/ints/bools/None (assert this in the test with a type walk over a captured real stream fixture; if any mutable leaf exists, keep `deepcopy` for that key only and note it). Delete the engine-level `event = deepcopy(next(self._stream))` at 1749 entirely (upstream `_filtered_event` already returns freshly built dicts) — or downgrade to `dict(event)` if the isolation test fails without it. In `_normalize_messages`/`_normalize_tools`, replace `deepcopy(dict(raw))` with `dict(raw)` only where the source is caller-owned and mutation risk is real; otherwise pass through.
- [ ] **Step 3:** Run `pytest Tests/LLM_Calls -k "hosted" -v` + the new isolation test.
- [ ] **Step 4: Evidence + commit.** Micro-benchmark 10k synthetic chunks through the wrapper before/after (time.perf_counter); record in notes. `git commit -m "perf(llm): replace hosted-stream deepcopy chain with fresh shallow copies"`.

**Acceptance criteria:** isolation test green (downstream mutation cannot corrupt the stream); per-chunk deepcopy count drops from 5–9 to 0–1; hosted-provider streaming tests green.

**Out of scope (documented, not changed):** the `LegacyLineStream` json round-trip shim (`legacy_line_stream.py:44`) — it is a deliberate consumer contract for groq/deepseek/mistral/openrouter; removing it means rewriting four provider consumers. Record as a follow-up candidate with the measured chunk cost.

---

### Task 6 (F8b, ADR-214): Drop per-attempt payload deepcopy + provider session reuse

**Files:**
- Create: `tldw_chatbook/backlog/decisions/214-provider-http-session-reuse.md`
- Create: `tldw_chatbook/LLM_Calls/provider_sessions.py` (session registry)
- Modify: `tldw_chatbook/LLM_Calls/hosted_chat.py:798-807` (drop `deepcopy(dict(payload))` → pass `payload` directly; `requests` does not mutate the `json=` argument — pin with a test that the payload dict is unchanged after a call and after a retried call)
- Modify: `create_default_session()` call sites listed in the review: `LLM_API_Calls.py` (849, 948, 1820, 2802, 3643, 4385, 4473), `hosted_chat.py:781`, `qwencloud.py:1176`, `Summarization_General_Lib.py:1040`
- Test: `Tests/LLM_Calls/test_provider_session_reuse.py` (create)

**Interfaces:**
- Produces: `provider_sessions.get_session(key: str, factory: Callable[[], requests.Session]) -> requests.Session` — per-thread (`threading.local`) session cache; sessions keep the existing adapter/Retry config from `create_default_session()`; `provider_sessions.close_all_for_current_thread()` for worker teardown.

- [ ] **Step 1: Failing tests** — (1) two calls with the same key return the same session object within a thread; (2) different threads get different sessions (thread-safety posture: never share a `requests.Session` across threads); (3) session retains the mounted Retry adapter.
- [ ] **Step 2: Implement registry** and swap the ten call sites to `get_session(f"{provider}:{base_url}", create_default_session)`. Verify with the review's noted in-call retry reuse that nothing closes the session between attempts (keep behavior).
- [ ] **Step 3:** `pytest Tests/LLM_Calls -v` (targeted) + one live smoke against a provider if keys exist (per `lessons-live-verification.md`, record it).
- [ ] **Step 4: Evidence + commit.** Record TLS handshake count for 3 back-to-back calls before (3) vs after (1) using a local `httpx`/`requests` mock server or connection-log. `git commit -m "perf(llm): reuse provider HTTP sessions per thread; drop payload deepcopy"`.

**Acceptance criteria:** no cross-thread session sharing; retries still reuse the in-call session; payload passed by reference (mutation-probe test green); ADR-214 written (covers lifecycle, key granularity, teardown, timeout/keep-alive settings).

---

### Task 7 (F5): Bound `get_conversation_tree` reads

**Files:**
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py` — `get_message_tree_rows_for_conversation` (anchor `grep -n "def get_message_tree_rows_for_conversation"`)
- Modify: `tldw_chatbook/Chat/chat_conversation_service.py:1187-1211` (Python root paging → drives the new SQL) and `get_conversation_tree` totals
- Test: `Tests/Chat/test_conversation_tree_bounded_reads.py` (create)

**Interfaces:**
- Produces: `get_message_tree_rows_for_conversation(conversation_id, *, root_offset: int, root_limit: int, order_desc: bool)` → `(rows, total_roots)` — page of roots **and all their descendants** only.

- [ ] **Step 1: Golden equivalence test** — 300-message tree fixture (deep chains + wide branches); assert new paged API returns, for each `(offset, limit)`, exactly the rows the old full fetch + Python slice would render, and `total_roots` matches.
- [ ] **Step 2: Implement** as three bounded statements in one transaction: (a) `SELECT COUNT(*) FROM messages WHERE conversation_id=? AND deleted=0 AND parent_id IS NULL` (root count — confirm the actual root criterion from current assembly code and keep it identical); (b) root page via keyset/LIMIT-OFFSET on the same ordering the Python path uses; (c) descendants via one recursive CTE seeded with the page's root ids (`WITH RECURSIVE ... WHERE root IN (page ids)`), same column set as today. BLOB columns stay excluded (`has_image` flag unchanged).
- [ ] **Step 3: Wire the service** to consume `(rows, total_roots)` and delete the Python `root_rows[offset:offset+limit]` slicing.
- [ ] **Step 4: Row-count evidence.** Use `sqlite3.set_trace_callback` on the test fixture: old path fetches 300 rows for page 1 of 50 roots; new path fetches only the page's subtree. Record numbers. `pytest Tests/Chat/test_conversation_bounded_reads.py Tests/Chat -k "tree" -v`.
- [ ] **Step 5: Commit** — `perf(chat): bound conversation tree reads to the requested root page`.

**Acceptance criteria:** rendered page identical for all fixtures; rows fetched ≤ page subtree + count, regardless of conversation length; callers that genuinely need the full tree (fork path) still use the unbounded variant explicitly.

---

### Task 8 (F6, ADR-213): Persistent embedding content-hash cache + batched skip-checks

**Files:**
- Create: `tldw_chatbook/backlog/decisions/213-persistent-embedding-content-hash-cache.md`
- Modify: `tldw_chatbook/DB/RAG_Indexing_DB.py` — new `embedding_cache` table (schema + migration file; **bump the RAG DB schema version** per migrations convention), `get_cached_embeddings(model_id, hashes)`, `store_embeddings(rows)` batched
- Modify: `tldw_chatbook/RAG_Search/simplified/embeddings_wrapper.py:514-556` — wire the dead cache check to the real store (hits skip the provider call; fix or delete the bogus `embeddings_cache_hit_rate` metric so it measures reality)
- Modify: `tldw_chatbook/RAG_Search/ingestion_indexing.py:743-755` — replace the per-entry `needs_reindexing` loop with one batched `get_indexed_items_by_type` read (the media path at line 1541 already proves the pattern)
- Test: `Tests/RAG/test_embedding_content_hash_cache.py` (create)

**Interfaces:**
- Produces: `RAGIndexingDB.get_cached_embeddings(model_id: str, content_hashes: Sequence[str]) -> dict[str, list[float]]`; `store_cached_embeddings(model_id, rows: Sequence[tuple[str, list[float]]])` (one `executemany`, one transaction). Key = `(model_id, sha256(text))`. Eviction: cap table at a configured row count, prune oldest by `created_at` in the same transaction as inserts.

- [ ] **Steps:** (1) failing tests — cache hit on re-embed of identical text (fake factory counts provider calls), miss on changed text, isolation across model_ids, eviction under cap; (2) migration + table; (3) wrapper wiring (batch lookups before `embed_with_factory`, store misses after); (4) batched `index_entries` skip-check with a spy asserting one query per batch; (5) `pytest Tests/RAG Tests/RAG_Search -k "embedding or indexing" -v`; (6) evidence — re-run a backfill over an already-indexed fixture: provider embed calls before (N) vs after (0); (7) commit.
- Also note in ADR-213: the `EmbeddingFactory._lock` global serialization stays for now (correctness of model handle reuse); the cache makes the lock mostly uncontended. Lock redesign explicitly out of scope.

**Acceptance criteria:** unchanged-content re-ingest performs zero provider embed calls after restart (persistence proven by reopening the DB in the test); `needs_reindexing` N+1 gone; hit-rate metric reports true values; schema migration file added with version bump.

---

### Task 9 (F7): Batch legacy RAG-context store writes on import

**Files:**
- Modify: `tldw_chatbook/Chat/chat_conversation_service.py` — `record_message_rag_context` (1361–1400) + new `stage_rag_context_record(...)` / `flush_rag_context_store()`
- Modify: `tldw_chatbook/Chatbooks/chatbook_importer.py:2315` (recovery fallback path) — stage per message, flush once per import in a `finally`
- Test: `Tests/Chatbooks/test_import_rag_context_batching.py` (create)

**Approach:** keep `record_message_rag_context` semantics for genuine one-off recovery callers (immediate flush); importer uses staging so an N-message import writes the store file once instead of N times. Test: monkeypatch `_save_rag_context_store` counting invocations + `json.dumps` calls over the whole store — importer path: 1; single-message path: 1. Complexity note for reviewers: load-side memoization already exists; this closes the quadratic write side. Commit: `perf(chatbooks): flush legacy RAG-context store once per import`.

**Acceptance criteria:** import of a 500-cited-message chatbook triggers exactly one store serialize+write; recovery-mode single writes unchanged; store content byte-identical to per-message writes (fixture comparison).

---

### Task 10 (F9): Buddy modal cheap-gate + right-size polling

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_store.py` — add `session_fingerprint(session_id) -> tuple[int, int]` (message count, last-message content length) without full snapshot materialization
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/buddy_conversation_modal.py:127-216`
- Test: `Tests/Persona_Buddy/test_buddy_modal_poll_gating.py` (create)

**Approach:** per tick: (1) read fingerprint; if unchanged since last render → return (no snapshot, no 64 KB build, no `repr(payloads)`); (2) interval 0.2 s **only while a generation is active**, else 1.0 s (reset on fingerprint change); (3) build transcript from `[-60:]` only after the fingerprint moves (the snapshot call stays but is now change-gated); (4) drop the duplicate `coordinator.show_decisions(...)` (runs twice per tick today) and replace the `repr(payloads)` change-detection with a cheap tuple of ids/versions. Tests: idle modal performs zero `messages_for_session` calls across 5 ticks; active streaming performs ≥1 but no more than one per fingerprint change; decisions coordinator invoked once per tick. Commit: `perf(buddy): gate 5 Hz transcript poll on session fingerprint`.

**Acceptance criteria:** idle-open modal does O(1) work per tick; streaming updates still render within one poll interval; visual behavior unchanged.

---

### Task 11 (F10): File Notes poll — signature gate, bound, backoff

**Files:**
- Modify: `tldw_chatbook/Notes/file_notes_service.py` — `reconcile` signature short-circuit (anchor `grep -n "two stats per file" file_notes_service.py`) + `max_files` bound matching the watcher's discovery bounds
- Modify: `tldw_chatbook/Notes/file_notes_session_owner.py:1127` — cap the change list (keep newest 500 per binding, prune on commit) and coalesce on append
- Modify: `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py:2376, 3542-3557` — poll backoff 1.5 s → up to 6 s when the signature is unchanged (reset on change), and skip the per-tick coalesce+compare when no new changes were appended
- Test: `Tests/Notes/test_file_notes_poll_gating.py` (create)

**Approach:** the walk itself can't be skipped without fs events (note FSEvents/watchdog as a possible future task, out of scope), but the second lstat, the full replica `list_active_files` read, and the sort/diff run only when the (device, inode, size, mtime_ns) signature tuple changed. Tests: unchanged vault → replica read count 0 across ticks (spy); changed file → full reconcile runs once; 2000-file vault with no changes → tick cost O(walk stats) only, no DB rows materialized; change list capped at 500 after a burst of 600 records. Commit: `perf(notes): gate file-notes reconcile on discovery signature; bound session changes`.

**Acceptance criteria:** idle tab does no DB reads per tick after the first; reconcile still fires exactly once per real change; session-change memory bounded; existing sync tests green.

---

### Task 12 (F14): Debounce + read-only index check

**Files:**
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_inspector_pane.py:1260-1265` + `UI/Persona_Modules/personas_conversations_controller.py` — 0.2 s debounce mirroring `PERSONAS_SEARCH_DEBOUNCE_SECONDS`
- Modify: `tldw_chatbook/DB/character_conversation_search.py:872-905` — `ensure_keyword_index` does a read-only status SELECT first; opens the `immediate=True` write transaction only when status ≠ READY
- Modify: `tldw_chatbook/UI/Library_Modules/library_media_controller.py:2293-2299` + `Library/library_media_reader_state.py:26` — dedicated `MEDIA_FILTER_DEBOUNCE_SECONDS = 0.25` instead of reusing `SELECTION_SETTLE_SECONDS` (0.12)
- Test: extend `Tests/Library/` + `Tests/Persona_Buddy`/persona controller tests

**Approach:** three mechanical patches, one test each: (1) keystroke burst of 5 chars fires exactly one search (timer re-arm assertion — copy the library prompts debounce test pattern); (2) READY index → zero write transactions opened (connection spy); (3) media filter settles at 0.25 s (constant assertion + behavior test). Commit: `perf(ui): debounce personas conversation search; read-only keyword-index check; media filter 0.25 s`.

---

### Task 13 (F15): Structural sharing in `ReaderItemSnapshot`

**Files:**
- Modify: `tldw_chatbook/UI/Watchlists_Modules/reader_item_snapshot.py:146-203`
- Test: `Tests/Library/test_reader_item_snapshot_sharing.py` (create)

**Interfaces:**
- Produces: page payloads become shallow-frozen records: store each page as `tuple` of row dicts wrapped with `types.MappingProxyType` at construction (or a `@dataclass(frozen=True)` row — pick one; mapping-proxy keeps dict access syntax, minimizing caller churn). `with_continuation`/`with_pending_page` concatenate page tuples without copying rows.

**Approach:** (1) test: after N page turns, rows from page 1 are the *same objects* (`is`) as when first cached, and total copies are O(new page) (count `deepcopy` calls via monkeypatch — must be 0); (2) enforce immutability at the boundary (wrap once on construction); (3) audit the few read sites for mutation attempts (grep `\[.*\] *=` on page rows — fix any writer to rebuild the row instead). Commit: `perf(watchlists): structural sharing for reader page cache`.

**Acceptance criteria:** paging to the end of a 5k-item list does O(page) work per turn (asserted by copy counter); no behavior change in cached-page navigation; any code that previously relied on defensive copies is fixed to not mutate shared rows.

---

### Task 14 (F18d): TTS widget index instead of full-DOM queries

**Files:**
- Modify: `tldw_chatbook/app_speech.py:313-470` (four handlers)
- Modify: `tldw_chatbook/Widgets/chat_message_enhanced.py` + `Widgets/Chat_Widgets/chat_message.py` — register/unregister in an on_mount/on_unmount index
- Test: `Tests/App/test_tts_widget_index.py` (create)

**Approach:** module-level `_MESSAGE_WIDGETS: dict[str, list[Widget]]`; handlers do `_MESSAGE_WIDGETS.get(event.message_id, ())` → O(1); when empty (the production case today) handlers early-return before any DOM walk. Tests: handler with no registered widget performs zero `app.query` calls (spy); registered widget receives state update. Commit: `perf(app): index TTS message widgets; drop full-DOM scans`.

---

### Task 15 (F11, ADR-215): Sargable SQL trio (schema migration — riskiest task)

**Files:**
- Create: `tldw_chatbook/backlog/decisions/215-conversation-timestamp-normalization.md`
- Create: `tldw_chatbook/migrations/chachanotes_vNNN_to_vNNP_timestamp_normalization.sql` (determine current NNN by listing `migrations/`; **bump schema version**)
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py` — `get_conversations_for_character` (11210–11231), `get_due_flashcards`/`count_due_flashcards` (21601–21642), `_USER_VISIBLE_CHARACTER` + character list queries (9177–9327), plus every writer that emits non-canonical timestamps (grep `CURRENT_TIMESTAMP` across DB writes)
- Test: `Tests/ChaChaNotesDB/test_sargable_timestamps.py` (create)

**Subtasks:**
- [ ] **15a — Format audit (test-first, no code changes):** scan the conversations table for distinct `last_modified` formats (space-separated `YYYY-MM-DD HH:MM:SS` from SQLite `CURRENT_TIMESTAMP` vs ISO `...T...Z` from app code). This test documents the split and gates the migration.
- [ ] **15b — Normalize + fix writers:** migration rewrites all `conversations.last_modified` (and `messages` timestamps if audit shows the same split there) to canonical ISO-8601 `T...Z` via `strftime('%Y-%m-%dT%H:%M:%SZ', julianday(col))`; replace `CURRENT_TIMESTAMP` defaults/inserts in writers with the app's canonical timestamp helper so new rows can't regress (add a CHECK-pattern test asserting every inserted row matches the canonical regex). Then drop `julianday()` from `get_conversations_for_character` (raw `last_modified < ?` keyset + `ORDER BY last_modified DESC, id DESC`) and add `CREATE INDEX idx_conv_char_lm ON conversations(character_id, last_modified DESC, id DESC)` (or extend the existing `idx_conv_char` if compatible — verify with EXPLAIN QUERY PLAN in the test).
- [ ] **15c — Flashcards:** same treatment for `next_review`: normalize stored values, drop `datetime()` wrappers from WHERE/ORDER BY, keep/adapt `idx_flashcards_next_review`, verify with EXPLAIN QUERY PLAN assertion that the index is used.
- [ ] **15d — Character cards:** add `CREATE INDEX idx_character_cards_visible_name ON character_cards(name COLLATE NOCASE)` filtered `WHERE deleted = 0` (SQLite allows deterministic `json_extract` in partial-index WHERE; verify on the shipped SQLite version — if the expression index is rejected, fall back to a stored `is_user_visible` column maintained by triggers and index that). Verify NOCASE ordering is served.
- [ ] **15e — EXPLAIN assertions:** each query's test asserts the plan contains the expected index (no `SCAN conversations` / no full sort).

**Steps:** audit test (fails/documents) → migration + version bump → writer fixes → query rewrites → plan assertions → targeted runs (`pytest Tests/ChaChaNotesDB -v`) → restore/backup verification (test migration against a fixture DB containing both legacy formats; also test re-run idempotency) → evidence (EXPLAIN before/after pasted into notes) → commit.

**Acceptance criteria:** EXPLAIN QUERY PLAN shows index-driven range+order for all three paths; ordering results identical to pre-change on a mixed-format fixture; migration idempotent; schema version bumped; ADR-215 records the canonical-format policy.

---

### Task 16 (F13): Visual-identity decode memoization + version-based authority check

**Files:**
- Modify: `tldw_chatbook/Character_Chat/visual_identity.py` — memoize `_inspect_image_bytes` per `(path, size, mtime_ns)` (module LRU); have `resolve_visual_identity` return the prepared frames/durations it already decoded so `Chat/character_expression_playback.py:102 prepare_expression` reuses them instead of seeking/loading every frame a second time
- Modify: `tldw_chatbook/Character_Chat/persona_visual_identity.py:122-351` — store the portrait digest (version token) beside the persona record on first validation; `local_persona_visual_identity_is_current` compares version ints, re-hashing only on mismatch
- Test: `Tests/Persona_Visual/test_visual_identity_decode_memoization.py` (create)

**Approach:** spy tests — second resolution of the same asset performs zero `Image.load()` frame calls (resolver and playback combined: exactly one full decode per asset per stat-signature); persona revalidation performs zero sha256 calls when the version matches. Keep the corruption-check decode once per signature (it is deliberate). Commit: `perf(character): memoize visual-identity decode; version-token portrait authority`.

---

### Task 17 (F16): Parallelize pairwise reranker recursion

**Files:**
- Modify: `tldw_chatbook/RAG_Search/reranker.py:725-771` (anchor `grep -n "_merge" reranker.py`)
- Test: `Tests/RAG_Search/test_pairwise_reranker_concurrency.py` (create)

**Approach:** the two recursive sort halves are independent — `asyncio.gather` them under an `asyncio.Semaphore(max_concurrent_comparisons)` (default 4, constructor arg); the merge loop stays sequential (inherently order-dependent). Tests: with a fake comparator that records concurrency, sorting 8 items never exceeds the semaphore and total wall-clock ≈ critical-path (compare against a serial baseline with a controllable per-call delay). Results identical to the serial implementation on a golden fixture. Commit: `perf(rag): bounded concurrency for pairwise reranker recursion`.

---

### Task 18 (F17): Batch Chroma stale-chunk deletes

**Files:**
- Modify: `tldw_chatbook/RAG_Search/ingestion_indexing.py:760-773` (collect ids) + `tldw_chatbook/RAG_Search/simplified/vector_store.py:706-721` (new `delete_documents(doc_ids: Sequence[str])`)
- Test: `Tests/RAG_Search/test_batch_document_delete.py` (create)

**Approach:** one `collection.delete(where={"doc_id": {"$in": [...]}})` per batch (chunk the `$in` list at 500); runtime capability check on first use — if the installed Chroma rejects `$in` on delete, fall back to the per-doc loop (feature-detect once, log once). Tests: fake collection records delete calls — 5k changed docs produce ≤ ceil(5000/500) delete calls, not 5000. Commit: `perf(rag): batch Chroma document deletes on reindex`.

---

### Task 19 (F19a-c): Small correctness-adjacent perf fixes

Three one-file patches, one commit each, each with a targeted test:

- [ ] **19a — Fork double-fetch:** `Chat/chat_conversation_service.py:1277+1242` — add optional `rows=` parameter to `effective_active_leaf` (or a private `_leaf_from_rows`) so `copy_conversation_active_path` reuses its fetched rows; test asserts one `get_messages_for_conversation` call per fork (spy).
- [ ] **19b — CitationTraceBuilder incremental bytes:** `Chat/citation_trace_builder.py:723-754` — maintain a running `_governed_payload_bytes` counter; each `record_*` computes canonical bytes of only the new payloads once and adds them (port the pattern documented in `thinking_blocks.py:364-374`); cap-check compares the int. Test: N record calls perform exactly N canonical encodings (spy on `json.dumps`), and cap enforcement still trips at `GOVERNED_PAYLOAD_UTF8_BYTES_MAX`.
- [ ] **19c — Lore N+1 counts:** `UI/Screens/personas_screen.py:4378-4390` — add `WorldBookManager.count_entries_for_books(book_ids) -> dict[str, int]` (one `GROUP BY` query) and use it in `_list_world_books_with_counts`; test asserts one query for 10 books (spy).

---

### Task 20 (F19d-f): Remaining small fixes

- [ ] **20a — Chatbooks registry cache:** `Chatbooks/local_chatbook_service.py:162-337` — cache parsed registry keyed by `(path, mtime_ns, size)`; re-parse only on stat change; `list_chatbooks` filter/sort/paginate the cached parsed records without per-record pydantic re-validation (validate once at parse; `_record_copy` stays). Test: two `list_chatbooks` calls → one file read (spy).
- [ ] **20b — Debug-loop guard:** `Chat/Chat_Functions.py:2034-2051` — wrap the payload-summary loop in `if logging.isEnabledFor(logging.DEBUG):`. Test: with INFO level, a 500-message payload build performs zero join/format calls (spy on logging or refactor loop into `_debug_dump_payload(...)` and spy that).
- [ ] **20c — Post-gen dictionary mtime cache:** `Chat/Chat_Functions.py:2181-2204` — cache the parsed dictionary keyed by `(path, mtime_ns)`; re-read only on change. Test: two responses → one `parse_user_dict_markdown_file` call.
- [ ] **20d — Media scoped-search gather (optional):** the ≤4 sequential per-source-type store queries in `rag_service.py:1528-1539` could `asyncio.gather`; only if measurements justify it. Skip unless cheap.

---

### Task 21 (F18): Dead-code removal

**Files:**
- Modify/Delete: `Chat/chat_persistence_service.py:4040-4157` (`save_history` — zero callers verified; delete + grep-proof in test/notes)
- Modify: `Widgets/tool_message_widgets.py:258-276` — fix the swallowed `WrongType` (`query_one(".message-text", Static)` on a `Markdown` widget): query `Markdown` and call `update()` with the markdown source; remove the branch or delete the module if tests are its only consumers (verify with grep before deleting; keep the diff-remount worker logic)
- Modify: `Widgets/Chat_Widgets/chat_message_enhanced.py:688-691` — remove the no-op `update_message_chunk` dead streaming entry (and its test callers) **or** wire the missing `message_text` watcher if a resurrection is planned; decision recorded in the task notes (default: remove)
- Test: affected test files updated; `pytest Tests/Chat Tests/Widgets -k "save_history or tool_message or chat_message" -v`

**Approach:** each removal preceded by a caller-grep pasted into the task notes (evidence per lessons-testing-evidence). One commit per removal. If any removal turns out to have live callers, stop and re-scope rather than forcing it.

---

## Wave 7 wrap-up (mandatory before closing tasks)

- [ ] Update `backlog/docs/lessons-testing-evidence.md` (or the matching lessons file) with the recurring trap this review surfaced: **fixes that land on one seam while the identical pattern survives on a sibling seam** (Console vs Library search is the flagship; also debounce constants, N+1 counts). State the incidents, not just the rule.
- [ ] Each task's backlog file gets Implementation Notes with the before/after evidence numbers.
- [ ] ADRs 212–215 linked from their tasks; no other ADRs required (mechanical fixes).

## Verification strategy (applies to every task)

1. Targeted pytest runs only (module-scoped), per `AGENTS.md` — no full sweeps without explicit request.
2. Every perf claim carries a counted measurement (spy counts, trace callbacks, or timers) in the task notes — "feels faster" is not evidence.
3. Any task touching SQLite writers or migrations runs its test against a fixture DB with legacy-format data (mixed timestamps especially, for T15).
4. Live verification (real provider call / real app run) for T6 and T10 per `backlog/docs/lessons-live-verification.md`, recorded in notes.
5. Schema-version and migration-dir checks precede any DB change (T8, T15).

## Suggested backlog instantiation

One backlog task per T-number above (21 tasks), waves as dependency-ordered batches (`--dep` only to earlier tasks where the Interfaces block requires it: T3→T4 shares pattern but not code; T7 before T15 to avoid migration conflicts on the same file). Titles: use the task headings verbatim. When a task moves to In Progress, paste its §Approach into `backlog task edit <id> --plan` per repo workflow.
