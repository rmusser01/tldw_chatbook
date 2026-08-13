# RAG Server-Port P0 Foundations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the retrieval tldw_chatbook already owns reachable and honest: the live Library `rag` mode honors the active RAG profile (hybrid/semantic/plain + reranking), and the native Console send path gets opt-in RAG injection through the staged-evidence pipeline (TASK-406).

**Architecture:** Engine fixes first (fusion metadata, keyword leg, reranker factory), then the Library service's mode resolution, then score-kind-aware presentation, then the Console send path. No server code is lifted; the spec is authority: `Docs/superpowers/specs/2026-08-07-rag-port-p0-foundations-design.md`.

**Tech Stack:** Python 3.11+, Textual 8.x, pytest (venv-only), SQLite FTS5, chromadb (optional `embeddings_rag` extra).

## Global Constraints

- Spec: `Docs/superpowers/specs/2026-08-07-rag-port-p0-foundations-design.md` — read it before any task.
- Worktree: `.worktrees/rag-port-p0`, branch `feat/rag-port-p0-foundations`. **Background-Bash cwd silently resets to the main checkout — start every command block with `cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/rag-port-p0`.**
- Python: the worktree has NO venv. Run pytest as `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <paths>` with cwd in the worktree (cwd-first import resolution picks up the worktree's package).
- pytest is the ONLY python entry point — never `python -c "import tldw_chatbook..."` (a bare probe once wrote to the live config).
- "no tests ran" is a FAILED gate: read the numeric passed count every time.
- Targeted suites only (user ruling): run the test files this branch touches plus a `--collect-only -q` sweep; never full `Tests/UI`.
- Never `git stash` (repo-wide across 100+ worktrees). Never `git checkout HEAD -- <file>` to undo a mutation probe — use Edit-based restores with unique marker strings.
- Push after every task (`git push`) — durability begins at origin.
- Config honesty: retrieval changes must never present a fused/reranker score as a similarity; every degradation is disclosed, never silent.
- Commit messages end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

## Verified code anchors (read before relying on them; line numbers drift)

- `tldw_chatbook/RAG_Search/simplified/rag_service.py` — `RAGService.search` dispatch (~L627), `_keyword_search` (~L775: path-guess list, create-on-miss `MediaDatabase` side effect, `_perform_fts5_search` media-schema SQL), `_process_keyword_results_basic` / `_with_citations` (metadata keys: `doc_id, doc_title, media_type, url, author, ingestion_date, text_preview` — **no `source_type`**), `_hybrid_search` (~L914), `_fuse_hybrid_results` (fused score replaces leg scores; `metadata["hybrid_fusion"]` = alpha/rrf_k/ranks/RRF contributions only).
- `tldw_chatbook/RAG_Search/fusion.py::reciprocal_rank_fusion` (~L98) — `FusedResult` has `fts_item/vector_item/fts_rank/vector_rank/fts_rrf/vector_rrf/score`; original leg items retain their own `.score`.
- `tldw_chatbook/RAG_Search/simplified/enhanced_rag_service_v2.py` — `__init__` (~L97-103) calls `create_reranker(strategy=..., **self.reranking_config.__dict__)` → **TypeError: multiple values for 'strategy'** (reranking has NEVER constructed); second site ~L343; experiment site ~L238. `search()` applies `self.reranker.rerank` UNGUARDED (~L232-244).
- `tldw_chatbook/RAG_Search/reranker.py::create_reranker` (~L624) — only `pointwise|pairwise|listwise` (all LLM-based); raises ValueError otherwise.
- `tldw_chatbook/RAG_Search/config_profiles.py` — `hybrid_full_rerank = RerankingConfig(strategy="cross_encoder", ...)` (~L352) — **unsupported strategy**; profile field `search.default_search_mode` values `"plain"|"semantic"|"hybrid"` (~L239/261/283).
- `tldw_chatbook/RAG_Search/simplified/rag_factory.py` (~L55-83) — `enable_reranking = profile.reranking_config is not None`.
- `tldw_chatbook/Library/library_local_rag_search_service.py` — mode dispatch (~L178-182: `mode == "rag"` → `_search_semantic`, else `_search_keyword`); `_search_keyword` (~L184: four-seam, scope-aware via per-seam allowlists); `_SEMANTIC_SOURCE_TYPE_MAP` (~L54: note/notes→notes, media/media_chunk→media, conversation(s)/chat→conversations); `_search_semantic` (~L417: post-filters rows by canonicalized provenance `source_type`).
- `tldw_chatbook/Library/library_rag_state.py` — `LIBRARY_RAG_MATCH_STRONG_THRESHOLD = 0.5`, `MODERATE = 0.2` (~L95); `library_rag_score_suffix(score: float | None)` (~L601, pure); `library_rag_all_matches_weak(rows)` (~L1505).
- `tldw_chatbook/UI/Views/RAGSearch/search_handoff.py::_library_rag_score` (~L163) — clamps to [0,1]; feeds `EvidenceBundle` scores (~L397/469).
- `tldw_chatbook/UI/Screens/chat_screen.py` — chip manual run builds `LibraryRagSearchRequest(mode="rag", top_k=5, ...)` (~L12674); staging seam `_stage_console_library_rag_launch` (~L3207); send-path capture `await capture_console_staged_evidence_for_chat(...)` (~L4654).
- `tldw_chatbook/Chat/console_chat_controller.py::submit_draft` (~L1803).
- `tldw_chatbook/Widgets/Console/console_rag_settings_modal.py` — `ConsoleRagSettingsResult` (~L102), `ConsoleRagSettingsModal` (~L119).
- `tldw_chatbook/config.py::get_media_db_path()` (~L5712) — the real media DB is `~/.local/share/tldw_cli/tldw_cli_media_v2.db`; the keyword leg's guesses (`media_db.db`, `chacha_notes.db`) can NEVER match it.
- Cache: `simple_cache.py::_make_key(query, search_type, top_k, filters, metadata_allowlist)` — search_type in key; no stale-cache hazard on mode change (spec verification item 5: RESOLVED, no work needed).

---

### Task 1: Backlog filing + task-406 AC edit

**Files:**
- Create: `backlog/tasks/task-<ID> - RAG-port-P0-profile-honoring-retrieval-and-Console-send-path-injection.md` (via CLI)
- Modify: `backlog/tasks/task-406 - Wire-RAG-context-injection-into-the-native-Console-send-path.md` (via CLI)

**Interfaces:**
- Produces: the P0 backlog task ID, referenced by every later commit message and the final Done notes.

- [ ] **Step 1: Assign a safe ID.** Scan across ALL worktrees and origin/dev for the current max task ID, then leapfrog with headroom (house rule after ten+ collisions):

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/rag-port-p0
for d in /Users/macbook-dev/Documents/GitHub/tldw_chatbook /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/*; do ls "$d/backlog/tasks" 2>/dev/null; done | grep -oE '^task-[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -3
git fetch origin dev --quiet && git ls-tree -r --name-only origin/dev -- backlog/tasks | grep -oE 'task-[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -3
```

Pick max + 100 headroom (e.g. max 3022 → use 3122).

- [ ] **Step 2: Create the P0 task** with `backlog task create` (repeat `--ac` per criterion — comma form writes one run-on criterion): ACs = the spec's Workstream A bullets (profile mode honored; keyword leg replaced with config-injected path + no-write-on-search; score-kind-aware bands; reranker factory fixed + never fails a search; zero-results honesty) and Workstream B bullets (toggle default OFF; plain-text-sends only; staged-evidence routing; EMPTY-scope shared copy; 5s timeout with initializing-vs-failed notice; legacy path unchanged). Set `-s "In Progress"` and add the plan reference via `--plan`.

- [ ] **Step 3: Edit task-406's AC** per backlog rules (AC updated BEFORE implementation): note that enablement re-homes from the legacy sidebar checkbox to the Console RAG chip modal toggle, default OFF, and that injection routes through the staged-evidence strip (visible, consume-on-send), not invisible prompt injection.

- [ ] **Step 4: Commit and push**

```bash
git add backlog/ && git commit -m "chore(backlog): file RAG-port P0 task; re-home task-406 enablement AC

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" && git push
```

---

### Task 2: Fusion preserves original leg scores

**Files:**
- Modify: `tldw_chatbook/RAG_Search/simplified/rag_service.py` (`_fuse_hybrid_results`)
- Test: `Tests/RAG_Search/test_hybrid_fusion_metadata.py` (create)

**Interfaces:**
- Produces: every fused result's `metadata["hybrid_fusion"]` gains `"fts_score": float | None` and `"vector_score": float | None` (the ORIGINAL leg scores, `None` for the absent leg). Task 5's banding and Task 6's bundle rely on `vector_score`.

- [ ] **Step 1: Read** `_fuse_hybrid_results` in full. The `FusedResult` entries carry `fts_item`/`vector_item` (the original `SearchResult`s, each with its own `.score`) — the fused block currently records ranks + RRF contributions but discards those scores.

- [ ] **Step 2: Write the failing test**

```python
# Tests/RAG_Search/test_hybrid_fusion_metadata.py
"""Fusion must preserve original per-leg scores for score-kind-aware display.

RRF-fused scores max out at ~1/(rrf_k+1) ~= 0.016 — far below the UI's
similarity band thresholds — so the vector leg's original similarity is the
only honest banding input for hybrid rows (spec Workstream A item 3).
"""
import pytest
from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService
from tldw_chatbook.RAG_Search.simplified.citations import SearchResult


def _result(rid: str, score: float) -> SearchResult:
    return SearchResult(id=rid, score=score, document=f"doc {rid}", metadata={})


def test_fused_rows_preserve_original_leg_scores():
    keyword = [_result("m1", 0.001), _result("m2", 0.001)]
    semantic = [_result("m2", 0.83), _result("m3", 0.41)]
    fused = RAGService._fuse_hybrid_results(
        keyword_results=keyword, semantic_results=semantic,
        top_k=10, alpha=0.7, include_citations=False,
    )
    by_id = {r.id: r for r in fused}
    both = by_id["m2"].metadata["hybrid_fusion"]
    assert both["vector_score"] == pytest.approx(0.83)
    assert both["fts_score"] == pytest.approx(0.001)
    fts_only = by_id["m1"].metadata["hybrid_fusion"]
    assert fts_only["vector_score"] is None
    vec_only = by_id["m3"].metadata["hybrid_fusion"]
    assert vec_only["vector_score"] == pytest.approx(0.41)
    assert vec_only["fts_score"] is None
```

(If `SearchResult` lives elsewhere or has a different constructor, adjust the import/helper to the real dataclass — read `simplified/citations.py` first.)

- [ ] **Step 3: Run to verify it fails** (KeyError on `vector_score`). Command per Global Constraints; expect FAIL, read the output.

- [ ] **Step 4: Implement** — in `_fuse_hybrid_results`, where `metadata["hybrid_fusion"]` is assembled from each `FusedResult` entry, add:

```python
"fts_score": entry.fts_item.score if entry.fts_item is not None else None,
"vector_score": entry.vector_item.score if entry.vector_item is not None else None,
```

- [ ] **Step 5: Run the new test (PASS) plus the existing fusion tests** (`grep -rl "_fuse_hybrid_results\|reciprocal_rank_fusion" Tests/ | sort -u` and run those files). Read passed counts.

- [ ] **Step 6: Commit and push** (`feat(rag): preserve original leg scores in hybrid_fusion metadata`).

---

### Task 3: Replace the keyword leg's DB resolution (no guessing, no writes)

**Files:**
- Modify: `tldw_chatbook/RAG_Search/simplified/rag_service.py` (`_keyword_search`, `_process_keyword_results_basic`, `_process_keyword_results_with_citations`)
- Modify: `tldw_chatbook/RAG_Search/simplified/config.py` (add `media_db_path` to the search/service config — read the config dataclass first and follow its style)
- Test: `Tests/RAG_Search/test_keyword_leg_db_resolution.py` (create)

**Interfaces:**
- Consumes: `tldw_chatbook.config.get_media_db_path()` (returns `Path`; honors `TLDW_CONFIG_PATH` scratch profiles).
- Produces: `_keyword_search` resolves its DB as: explicit `config.search.media_db_path` if set → else `get_media_db_path()`; missing/unopenable DB returns `[]` after logging — **never creates a database**. Keyword rows gain `"source_type": "media"` and `"source": "media"` in metadata.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/RAG_Search/test_keyword_leg_db_resolution.py
"""The keyword leg must use the configured media DB and never write.

Today it guesses paths (media_db.db / chacha_notes.db) that can never match
the real tldw_cli_media_v2.db, opens the ChaChaNotes DB with media-schema
SQL, and on a total miss CREATES a MediaDatabase as a search side effect.
"""
import asyncio
from pathlib import Path
import pytest


def _make_service(tmp_path, media_db_path=None):
    # Build a RAGService with the in-memory vector store and mock embeddings;
    # read tests in Tests/RAG_Search/ for the existing construction helper
    # (several tests build a service with provider="mock") and reuse it,
    # overriding config.search.media_db_path when given.
    ...


def test_missing_media_db_returns_empty_and_creates_nothing(tmp_path):
    service = _make_service(tmp_path, media_db_path=tmp_path / "absent.db")
    results = asyncio.run(service._keyword_search("anything", top_k=5))
    assert results == []
    assert not (tmp_path / "absent.db").exists()          # no create-on-miss
    assert list(tmp_path.glob("*.db")) == []              # no rogue DB anywhere


def test_keyword_rows_carry_media_source_type(tmp_path):
    db_path = tmp_path / "tldw_cli_media_v2.db"
    # Create a real MediaDatabase and insert one item via its public API
    # (add_media_with_keywords) so media_fts is populated by triggers.
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
    db = MediaDatabase(db_path=str(db_path), client_id="test_keyword_leg")
    # read add_media_with_keywords' signature and insert a doc containing
    # the token "wombat"
    ...
    service = _make_service(tmp_path, media_db_path=db_path)
    results = asyncio.run(service._keyword_search("wombat", top_k=5))
    assert results, "FTS row expected"
    assert results[0].metadata["source_type"] == "media"


def test_default_resolution_uses_get_media_db_path(monkeypatch, tmp_path):
    sentinel = tmp_path / "sentinel_media.db"
    import tldw_chatbook.config as cfg
    monkeypatch.setattr(cfg, "get_media_db_path", lambda **kw: sentinel)
    service = _make_service(tmp_path)  # no explicit path configured
    results = asyncio.run(service._keyword_search("anything", top_k=5))
    assert results == []                                   # sentinel absent
    assert not sentinel.exists()                           # still no writes
```

Fill the `...` bodies against the real APIs (read the neighboring tests and `Client_Media_DB_v2.add_media_with_keywords` first) — the assertions above are the contract and must remain as written.

- [ ] **Step 2: Run to verify failures** — expect the no-writes test to FAIL today (the create-on-miss branch fires) and the source_type test to FAIL (key absent).

- [ ] **Step 3: Implement**
  - Delete the entire guess-list block and the `MediaDatabase(...)` create-on-miss branch in `_keyword_search`.
  - Resolve: `db_path = self.config.search.media_db_path or tldw_chatbook.config.get_media_db_path()`; if the file doesn't exist → `logger.warning(...)`, `return []`.
  - Add `"source_type": "media", "source": "media"` to the metadata dict in BOTH `_process_keyword_results_basic` and `_process_keyword_results_with_citations`.
  - Add `media_db_path: Optional[Path] = None` to the search config dataclass (import-cycle note: import `get_media_db_path` lazily inside the function, matching the existing lazy-import style there).

- [ ] **Step 4: Run new tests (PASS) + `Tests/RAG_Search/` targeted files touching rag_service.** Read counts.

- [ ] **Step 5: Commit and push** (`fix(rag): keyword leg uses configured media DB; never writes; stamps source_type`).

---

### Task 4: Fix the reranker factory seam (has never constructed)

**Files:**
- Modify: `tldw_chatbook/RAG_Search/reranker.py` (add `create_reranker_from_config`)
- Modify: `tldw_chatbook/RAG_Search/simplified/enhanced_rag_service_v2.py` (all three `create_reranker` call sites; guard `.rerank`)
- Modify: `tldw_chatbook/RAG_Search/config_profiles.py` (Hybrid Full's `strategy="cross_encoder"` → `"pointwise"`)
- Test: `Tests/RAG_Search/test_reranker_construction.py` (create)

**Interfaces:**
- Produces: `create_reranker_from_config(config: RerankingConfig) -> BaseReranker`; V2's `self.reranker` constructs for every built-in profile; a raising reranker degrades to unreranked results with `metadata["reranking_skipped"] = "<reason>"` on the first result and a warning log — the search NEVER fails because of reranking (spec error-handling rule).

- [ ] **Step 1: Write the failing tests**

```python
# Tests/RAG_Search/test_reranker_construction.py
"""Reranking profiles must construct and must never fail a search.

V2 called create_reranker(strategy=X, **config.__dict__) — the dict also
contains 'strategy', so EVERY reranking profile raised TypeError at
construction; Hybrid Full additionally requests the unimplemented
'cross_encoder' strategy. Reranking has never executed on any profile.
"""
import pytest
from tldw_chatbook.RAG_Search.reranker import RerankingConfig, create_reranker_from_config
from tldw_chatbook.RAG_Search.config_profiles import get_builtin_profiles  # read the real accessor name first


def test_create_reranker_from_config_does_not_double_pass_strategy():
    cfg = RerankingConfig(strategy="pointwise", top_k_to_rerank=5)
    reranker = create_reranker_from_config(cfg)
    assert reranker.config.strategy == "pointwise"


def test_every_builtin_profile_reranking_config_constructs():
    for profile in get_builtin_profiles():
        rc = getattr(profile, "reranking_config", None)
        if rc is not None:
            create_reranker_from_config(rc)  # must not raise


@pytest.mark.asyncio
async def test_raising_reranker_degrades_to_unreranked_results():
    # Build a V2 service (mock embeddings, in-memory store) whose profile has
    # reranking; monkeypatch self.reranker.rerank to raise RuntimeError; index
    # two docs; search must return results with reranking_skipped set, not raise.
    ...
```

- [ ] **Step 2: Run to verify failures** (ImportError on `create_reranker_from_config`; profile sweep raises for cross_encoder).

- [ ] **Step 3: Implement**
  - `reranker.py`: `def create_reranker_from_config(config: RerankingConfig) -> BaseReranker:` dispatching on `config.strategy` to the three classes (reuse `create_reranker`'s mapping; keep `create_reranker` for compatibility).
  - `enhanced_rag_service_v2.py`: replace all three call sites with `create_reranker_from_config(...)`; wrap the `__init__` construction in try/except (log + `self.reranker = None` — a broken reranker config must not kill service construction); wrap `await self.reranker.rerank(...)` in try/except → on exception log warning, tag `results[0].metadata["reranking_skipped"]`, keep unreranked results.
  - `config_profiles.py`: Hybrid Full `strategy="cross_encoder"` → `"pointwise"` with a comment stating cross_encoder is not implemented in chatbook (LLM rerankers only).
  - Note in the code where it's true, not in comments elsewhere: LLM reranking spends provider calls per search — the `_final_score_kind=reranker` provenance is the disclosure channel (Task 5).

- [ ] **Step 4: Run new tests (PASS) + files touching V2/reranker.** Read counts.

- [ ] **Step 5: Commit and push** (`fix(rag): reranker factory double-strategy TypeError; Hybrid Full unsupported strategy; rerank never fails a search`).

---

### Task 5: Library service honors the profile's search mode

**Files:**
- Modify: `tldw_chatbook/Library/library_local_rag_search_service.py`
- Test: `Tests/Library/test_library_rag_mode_resolution.py` (create; read `Tests/Library/` for the existing service-test fixtures and mimic them)

**Interfaces:**
- Consumes: `rag_service.config.search.default_search_mode` (`"plain"|"semantic"|"hybrid"`) from the shared service; `_search_keyword(query, source_types, top_k, scope=...)` (existing four-seam path).
- Produces: `_resolve_profile_search_mode(rag_service) -> str` (pure mapping, mutation-tested); `rag` mode dispatch:
  - `"plain"` → route to `_search_keyword` (four-seam, scope-aware); result carries `backend="local-fts"` plus a disclosed note that the active profile forced keyword ("Profile '<name>': keyword search (no vectors)").
  - `"semantic"` → current behavior, `backend="rag-semantic"`.
  - `"hybrid"` unscoped AND `"media" in source_types` → engine `search(search_type="hybrid")`, `backend="rag-hybrid"`.
  - `"hybrid"` scoped OR media deselected → semantic path + coverage note naming the reason ("scope active — semantic only until scope-aware hybrid lands" / "media excluded — semantic only").
- Zero-results honesty: the "Index empty" recovery state must NOT be claimed when hybrid returned keyword-leg rows; when the vector index is empty but FTS hit, the coverage note reads "semantic leg empty — keyword-only results".

- [ ] **Step 1: Read** the full `search`/`_search_semantic` flow (~L126-500) including how recovery states and `semantic_scope_coverage` diagnostics are assembled, and how the shared service is resolved (`_resolve_rag_runtime`). Read `Tests/Library/` for the fake-service fixtures used by existing mode tests.

- [ ] **Step 2: Write the failing tests** — one per dispatch arm so mutations isolate:

```python
# Tests/Library/test_library_rag_mode_resolution.py
"""rag mode must honor the active profile's default_search_mode.

The live path hardcoded search_type="semantic"; the engine's hybrid
(RRF k=60 + alpha, ADR-005 server parity) was unreachable. Routing rules
and disclosures per the P0 spec, Workstream A.
"""
# Tests (use the existing Library service fixtures with a fake rag_service
# whose config.search.default_search_mode is parameterized and whose
# search() records the search_type it was called with):

def test_hybrid_profile_unscoped_calls_engine_hybrid(): ...
    # fake service mode="hybrid"; source_types include "media"; no scope
    # -> fake.search called with search_type="hybrid"; backend == "rag-hybrid"

def test_hybrid_profile_with_active_scope_stays_semantic_and_discloses(): ...
    # scoped request -> search_type="semantic"; coverage note mentions scope

def test_hybrid_profile_with_media_deselected_stays_semantic_and_discloses(): ...

def test_plain_profile_routes_to_four_seam_keyword_path(): ...
    # fake mode="plain" -> _search_keyword called (spy); engine search NOT called;
    # disclosed note names the profile and "keyword search (no vectors)"

def test_semantic_profile_unchanged(): ...

def test_index_empty_not_claimed_when_keyword_rows_present(): ...
    # fake hybrid returns rows whose hybrid_fusion has fts_rank set but
    # vector_score None for all; vector count == 0
    # -> recovery state is NOT "index empty"; note says "semantic leg empty — keyword-only results"
```

Write each body concretely against the real fixtures; the behavioral assertions above are the contract.

- [ ] **Step 3: Run — all new tests FAIL** (dispatch doesn't exist).

- [ ] **Step 4: Implement.** Keep `_resolve_profile_search_mode` a small pure function:

```python
def _resolve_profile_search_mode(rag_service: Any) -> str:
    """Map the active profile's default_search_mode to an execution route.

    "plain" deliberately routes to the four-seam scope-aware keyword path,
    NOT the engine's media-only keyword leg (spec: plain-profile routing).
    Unknown values fall back to "semantic".
    """
    mode = getattr(getattr(getattr(rag_service, "config", None), "search", None),
                   "default_search_mode", "semantic")
    return mode if mode in ("plain", "semantic", "hybrid") else "semantic"
```

Then branch in the `mode == "rag"` dispatch, passing `search_type="hybrid"` only when unscoped and media-selected; thread the disclosure strings through the existing coverage-note mechanism (find where `semantic_scope_coverage` text is built and extend it — do not invent a second note channel).

- [ ] **Step 5: Run new tests (PASS) + existing Library RAG service tests.** Read counts.

- [ ] **Step 6: Mutation checks** (Edit-based restores with unique markers, never `git checkout`):
  - Mutate the `"plain"` arm to call semantic → ONLY the plain-routing test reds.
  - Mutate the scoped guard to allow hybrid → ONLY the scoped-disclosure test reds.
  - Restore both; rerun; read counts.

- [ ] **Step 7: Commit and push** (`feat(library): rag mode honors active profile search mode with disclosed routing`).

---

### Task 6: Score-kind-aware bands (UI state + Answer bundle)

**Files:**
- Modify: `tldw_chatbook/Library/library_rag_state.py` (`library_rag_score_suffix`, `library_rag_all_matches_weak`)
- Modify: the row-construction site that feeds them (find with `grep -rn "library_rag_score_suffix(" tldw_chatbook/` — the panel/state builds rows from service results; thread `score_kind` + `vector_score` from `metadata["hybrid_fusion"]` / reranker markers into the row)
- Modify: `tldw_chatbook/UI/Views/RAGSearch/search_handoff.py` (`_library_rag_score` consumers)
- Test: extend the existing state tests (find with `grep -rln "score_suffix\|all_matches_weak" Tests/`)

**Interfaces:**
- Consumes: Task 2's `hybrid_fusion.vector_score` / `.fts_score`; Task 4's reranker provenance.
- Produces: `library_rag_score_suffix(score, score_kind="vector_similarity", vector_score=None)`:
  - `vector_similarity` → band as today on `score`.
  - `hybrid_fusion` with `vector_score is not None` → band on `vector_score` (label unchanged).
  - `hybrid_fusion` with `vector_score is None` (FTS-only row) → `" | keyword match"` — never a fabricated similarity, never the fused 0.0x number.
  - `reranker` → `" | reranked"` (kind disclosed; no cosine banding of logits).
  - `library_rag_all_matches_weak` counts ONLY rows whose effective banding input is a vector similarity; keyword-only and reranked rows neither trigger nor suppress it.
- The Answer path: `EvidenceBundle` rows carry `score_kind`; whatever copy `_library_rag_score` feeds (locate its consumers) computes weakness only over vector-similarity kinds — a fused 0.016 must not read as "weak similarity" in answer coverage copy.

- [ ] **Step 1: Write the failing tests** (pure functions — exact):

```python
def test_fused_score_never_bands_on_cosine_thresholds():
    # A fused RRF score (~0.016) with a strong vector leg must band strong.
    assert library_rag_score_suffix(0.016, score_kind="hybrid_fusion",
                                    vector_score=0.83) == " | match: strong"

def test_fts_only_hybrid_row_reads_keyword_match():
    assert library_rag_score_suffix(0.0161, score_kind="hybrid_fusion",
                                    vector_score=None) == " | keyword match"

def test_reranker_scores_disclose_kind_not_band():
    assert library_rag_score_suffix(-3.2, score_kind="reranker") == " | reranked"

def test_all_matches_weak_ignores_non_similarity_kinds():
    # rows: one keyword-only hybrid row + one weak vector row -> True;
    # keyword-only rows alone -> False (no scored similarity rows).
    ...
```

Plus a row-threading test at the panel/state level: a service result with `hybrid_fusion.vector_score=0.83` renders the strong band in the composed row title (mimic the existing composited-text pins in `Tests/Library/`).

- [ ] **Step 2: Run — FAIL** (signature lacks kwargs).

- [ ] **Step 3: Implement** — extend the two pure functions (default kwargs keep every existing call site green); thread `score_kind`/`vector_score` where rows are built; update `search_handoff.py` so bundle rows carry `score_kind` and its weakness consumers filter on it.

- [ ] **Step 4: Run new + existing state/panel/handoff tests.** Existing suffix tests must pass UNMODIFIED (they call with similarity kinds by default) — if one must change, stop and re-check the design (protected-oracle rule).

- [ ] **Step 5: Commit and push** (`feat(library): score-kind-aware match bands; fused/reranker scores never presented as similarity`).

---

### Task 7: Console auto-retrieve toggle (config + modal)

**Files:**
- Modify: `tldw_chatbook/config.py` (default for `[chat_defaults] rag_auto_retrieve_on_send = false` — find the chat_defaults defaults block and follow its pattern)
- Modify: `tldw_chatbook/Widgets/Console/console_rag_settings_modal.py` (Switch + result field)
- Test: extend the modal's existing test file (find with `grep -rln "ConsoleRagSettingsModal" Tests/`)

**Interfaces:**
- Produces: `get_cli_setting("chat_defaults", "rag_auto_retrieve_on_send", False)` readable everywhere; `ConsoleRagSettingsResult` gains `auto_retrieve_on_send: bool`; toggling the Switch persists via the same config-write seam the modal (or Settings) already uses — read the modal's save path first and reuse it.

- [ ] **Step 1: Write the failing tests** — modal composes the Switch defaulted from config; dismiss result carries the flag; flag round-trips to config.
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** (Switch labeled "Auto-retrieve on send", default OFF; tooltip: "When on, each plain text send first retrieves library evidence into the staged-evidence strip").
- [ ] **Step 4: Run modal tests. Read counts.**
- [ ] **Step 5: Commit and push** (`feat(console): auto-retrieve-on-send toggle in RAG chip modal, default off`).

---

### Task 8: Console send-path injection (TASK-406)

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (send flow, before the `capture_console_staged_evidence_for_chat` call ~L4654) and/or `tldw_chatbook/Chat/console_chat_controller.py::submit_draft` — read both first; put the retrieval hook at the same layer that already owns staging (`_stage_console_library_rag_launch`) so the staged-evidence strip renders it exactly like a manual chip run.
- Test: `Tests/UI/test_console_auto_rag_on_send.py` (create; mimic the existing console staged-evidence tests — find with `grep -rln "capture_console_staged_evidence" Tests/`)

**Interfaces:**
- Consumes: Task 7's config flag; `_resolve_console_library_rag_scope` (existing); `run_library_rag_search`; `_stage_console_library_rag_launch`; the consume-on-send predicate from PR-4 (unchanged).
- Produces: `_maybe_auto_retrieve_for_send(draft_text) -> None` on the screen/controller seam with this exact gate, in this order (each clause mutation-tested):

```python
async def _maybe_auto_retrieve_for_send(self, draft_text: str) -> None:
    """Auto-retrieve library evidence for a plain text send (TASK-406).

    Fires only when ALL hold: the config toggle is on; the send is plain
    user text (not a slash command, tool approval, or regeneration); and
    nothing is already staged (manual staging always wins — no double
    spend). Retrieval failure or timeout NEVER blocks the send.
    """
    if not get_cli_setting("chat_defaults", "rag_auto_retrieve_on_send", False):
        return
    if not _is_plain_text_send(draft_text):     # reuse/extract the send-kind
        return                                  # classification the send path
    if self._has_staged_console_evidence():     # already has (read it first)
        return
    scope = ...  # resolve_effective_scope_for_chat via the existing seam
    if scope_is_empty(scope):
        self._notify_shared_empty_scope_copy()  # the EXISTING shared notice
        return
    query = draft_text[:AUTO_RAG_QUERY_MAX_CHARS]   # cap = 2000, a constant
    try:
        async with asyncio.timeout(AUTO_RAG_TIMEOUT_SECONDS):   # 5.0, constant
            outcome = await run_library_rag_search(...)  # profile top_k, mode="rag"
    except TimeoutError:
        self._notify_auto_rag_degraded(initializing=self._rag_service_still_initializing())
        return
    except Exception:
        self._notify_auto_rag_degraded(initializing=False)
        return
    self._stage_console_library_rag_launch(...)  # same launch shape as chip run
```

  The strip shows "Retrieving…" while in flight; the notice copy distinguishes "RAG service still initializing" from "retrieval failed" (reuse the recovery-state vocabulary from `library_rag_service`). The retrieval awaits INSIDE the existing send worker (exclusive) — a double-send cannot double-retrieve.

- [ ] **Step 1: Read** the send flow end-to-end (screen handler → `submit_draft` → capture at ~L4654) and the staged-evidence strip states before writing anything.

- [ ] **Step 2: Write the failing tests** — one per gate clause + the happy path:

```python
def test_toggle_off_means_no_retrieval_call(): ...
def test_slash_command_send_never_retrieves(): ...
def test_already_staged_evidence_skips_auto_retrieve(): ...
def test_empty_scope_short_circuits_with_shared_copy(): ...
def test_happy_path_stages_then_send_consumes(): ...
    # fake service returns 2 rows -> launch staged -> capture consumes ->
    # prompt contains the evidence block; strip shows count
def test_timeout_sends_without_evidence_and_notifies(): ...
def test_retrieval_exception_sends_without_evidence(): ...
```

- [ ] **Step 3: Run — FAIL.**
- [ ] **Step 4: Implement** per the produced interface above.
- [ ] **Step 5: Run new + the console staged-evidence suites.** Read counts.
- [ ] **Step 6: Mutation checks:** drop the already-staged guard → only its test reds; drop the plain-text gate → only its test reds; invert the toggle default → toggle-off test reds. Edit-based restores.
- [ ] **Step 7: Commit and push** (`feat(console): opt-in RAG auto-retrieve on send via staged evidence (task-406)`).

---

### Task 9: Chip manual run inherits profile top_k

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (~L12674: `top_k=5`)
- Test: extend the chip-run test (find with `grep -rln "LibraryRagSearchRequest" Tests/UI/`)

**Interfaces:**
- Consumes: the shared service's `config.search.default_top_k` (read the profile config dataclass for the exact field name — Settings' Search fold edits it).
- Produces: manual chip runs and Task 8's auto-retrieve use the same depth; fallback to the current literal 5 only when the service/config is unavailable, via one small helper both call sites share.

- [ ] **Step 1: Failing test** — with a fake service whose `default_top_k=12`, the chip run builds `LibraryRagSearchRequest(top_k=12)`; with no service, `top_k=5`.
- [ ] **Step 2: Run — FAIL. Step 3: Implement** (`_console_rag_top_k()` helper; both the chip run and Task 8 use it). **Step 4: Run tests, read counts. Step 5: Commit and push** (`fix(console): chip manual run inherits profile default_top_k`).

---

### Task 10: Docs, follow-up filing, backlog closure

**Files:**
- Modify: `Docs/User_Guide/library/search-and-rag.md`, `Docs/User_Guide/console/context-and-rag.md`, `Docs/User_Guide/settings/rag.md` (content + "Verified against" stamps — house rule: UI-changing PRs update the matching guide page)
- Create: follow-up backlog task (filed then as combined MCP/agent divergence; current scope is MCP-only because `LibraryRagToolProvider` owns fallback agent RAG retrieval when direct Library tools are off and `LibraryToolProvider` owns direct `library_search_notes` when they are on)
- Modify: the P0 backlog task + task-406 (Done, ticked ACs, Implementation Notes)

- [ ] **Step 1: Update the three guide pages**: rag mode is profile-driven (hybrid default; what each profile mode means; the plain-profile routing); match-band meanings incl. "keyword match" and "reranked"; the Console auto-retrieve toggle (default OFF, staged-evidence visibility, timeout behavior); reranking profiles spend provider calls per search (cost disclosure).
- [ ] **Step 2: File the follow-up task** (fresh cross-worktree ID scan): "Align MCP perform_rag_search + agent RAGSearchTool with profile-driven retrieval" — description states the P0 non-goal: Library and MCP disagreed about what a rag search was at ship time; reference TASK-694/TASK-1077. **Current disposition:** when direct Library tools are off, `LibraryRagToolProvider.search_library_rag` owns fallback agent RAG retrieval; when direct Library tools are on, `LibraryToolProvider` owns direct `library_search_notes`. TASK-3500 is narrowed to MCP `perform_rag_search` only, which is not yet aligned.
- [ ] **Step 3: Close the backlog tasks**: tick every AC that the merged work satisfies, add Implementation Notes (approach, decisions — notably the never-worked reranker finding and plain-profile routing — and modified files), `backlog task edit <id> -s Done`.
- [ ] **Step 4: Commit and push** (`docs(guide): profile-driven rag mode, score bands, console auto-retrieve; backlog closure`).

---

### Task 11: Gates + live TUI walkthrough

**Files:** none created — verification only. Evidence goes to the scratchpad directory, not the repo.

- [ ] **Step 1: Targeted suite battery** — every test file this branch created or modified, in one run; READ the numeric passed count. Then `--collect-only -q` over `Tests/` and compare against a fresh `origin/dev` baseline count with exact arithmetic (new total = old total + exactly the number of tests this branch added).
- [ ] **Step 2: Live TUI walkthrough** (PR-2 recipe): scratch config via `TLDW_CONFIG_PATH` with `[first_run] setup_started/setup_completed = true`, copy the real ChaChaNotes + media DBs AND the chromadb dir into the scratch profile BEFORE first launch; `tmux -L ragp0 new-session -d -x 235 -y 52`; verify live:
  1. Library rag mode with the default (Hybrid Basic) profile: results appear, bands render (no "weak (0.02)" wall), backend label reads `rag-hybrid`.
  2. Switch to BM25 Only in Settings → rag mode routes to keyword with the disclosed note.
  3. Switch to Hybrid Full → service constructs (the pre-fix TypeError would kill it); a search completes even with no reranking provider ready (skip-disclosed).
  4. Console: toggle auto-retrieve ON in the chip modal → plain text send shows "Retrieving…" → "Evidence sent · N" → reply arrives; a `/command` send does NOT retrieve.
  5. Scope a conversation → rag search discloses semantic-only.
  Delete the scratch profile after; verify the live config untouched.
- [ ] **Step 3: Fix-forward anything found** (each fix RED-first), rerun the battery, push.

---

## Self-review (done at plan time)

- **Spec coverage:** Workstream A items 1-5 → Tasks 2-6; Workstream B → Tasks 7-9; error handling → Tasks 4/5/8; non-goals → Task 10 step 2; testing section → per-task steps + Task 11. Verification items: 1 (Task 3, resolved: real DB is `tldw_cli_media_v2.db`), 2 (resolved: rerankers are LLM-based, no model download; the crash was the double-strategy TypeError — Task 4), 3 (resolved: stamp `source_type: "media"` — Task 3), 4 (Task 6 handoff changes), 5 (resolved: no work), 6 (Task 6 tests + Task 5's threshold audit during implementation).
- **Type consistency:** `library_rag_score_suffix(score, score_kind=..., vector_score=...)` used identically in Tasks 6; `hybrid_fusion.vector_score/fts_score` named identically in Tasks 2/5/6; `_console_rag_top_k()` shared by Tasks 8/9; `create_reranker_from_config` in Task 4 only.
- **Known unknowns pushed into read-first steps:** exact fixture helpers in `Tests/Library`/`Tests/UI`, the config-write seam for the modal, `add_media_with_keywords` signature, the coverage-note assembly point. Each task names the grep to find them.
