# Fusion Weighting & Keyword-Leg Budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Push the user's selected source types down into the hybrid keyword leg (TASK-14751), then choose — by measurement, under a pre-stated decision rule — and ship the fusion-weighting change that lets hybrid rescue keyword-only documents (TASK-4110), closing TASK-3994 #2b.

**Architecture:** Engine-local: a `source_types` parameter threads Library → `RAGService.search` → `_hybrid_search` → `_keyword_search`; new config knobs (`rrf_k`, hybrid pool multiplier) consume the existing-but-unwired validators; a gated comparison runner rides P1's harness machinery to produce the decision matrix; the winner ships at one authoritative site. Spec is authority: `Docs/superpowers/specs/2026-08-10-rag-fusion-weighting-design.md` — its **decision rule** and **two rescue senses** section binds Task 4.

**Tech Stack:** Python 3.11+, pytest, the P1 eval harness (`RAG_EVAL=1`), SQLite FTS5.

## Global Constraints

- Spec (read FIRST, especially "Diagnosis", "Decision rule", "Parameterization prerequisites"): `Docs/superpowers/specs/2026-08-10-rag-fusion-weighting-design.md`. Backlog tasks 4110 + 14751 carry the per-task AC contracts.
- Worktree `.worktrees/rag-fusion-weighting`, branch `fix/rag-fusion-weighting`. **cwd silently resets between Bash blocks — start EVERY block with `cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/rag-fusion-weighting`.**
- Tests: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <paths>` with worktree cwd; pytest only; app-importing probes under Tests/ only; "no tests ran" = FAILED gate; read numeric counts.
- Engine source-type vocabulary is EXACT singular: `media` / `note` / `conversation` (`SOURCE_TYPE_*` constants in rag_service.py); the Library's selection vocabulary is plural (`media`/`notes`/`conversations`/`prompts`). Translation is pinned by test — the singular/plural trap has bitten twice.
- Gated harness runs mid-arc are informational; deliberate re-stamps ONLY in Task 6 (`RAG_EVAL_UPDATE_BASELINES=1`, printed deltas). 14751 is expected to move NOTHING on the gate (harness selects all three types — say so, don't infer health from +0.000).
- Protected oracles: the fusion cluster's tests (`test_hybrid_doc_fusion.py`, `test_keyword_leg_chacha.py`, `test_fts5_query_escaping.py`, `test_hybrid_fusion_metadata.py`, `Tests/DB/test_private_sqlite_inventory.py`) pass UNMODIFIED — except where Task 5's shipped-default change legitimately moves a pinned constant; any such edit is a DISCLOSED oracle update in the report, never silent.
- Guarded-KIND rule (lessons-testing-evidence.md): every task's test battery includes `Tests/DB/test_private_sqlite_inventory.py` (cheap; this arc moves code near a registered connection owner).
- Never `git stash`; Edit-based restores with unique markers; push after every task; commits reference the relevant TASK id and end `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

## Verified code anchors (line numbers drift; grep first)

- `_keyword_search(query, top_k, filter_metadata, include_citations)` (rag_service.py ~L865): gathers `_media_keyword_subleg` + `_chacha_keyword_sublegs` via `asyncio.gather`, filters empty rankings, `interleave_rankings(rankings, key=_fusion_doc_key)[:top_k]`. `_chacha_keyword_sublegs` (~L1278) fills `rows[SOURCE_TYPE_NOTE]` and `rows[SOURCE_TYPE_CONVERSATION]`.
- `_hybrid_search` (~L1451): legs fetched at `top_k * SEARCH_RESULT_MULTIPLIER` (module constant = 2; ALSO used by `_semantic_search` — a hybrid pool knob must not touch semantic mode); fusion call `_fuse_hybrid_results` → `reciprocal_rank_fusion(..., alpha=resolve_hybrid_alpha(config.search.hybrid_alpha), rrf_k=DEFAULT_RRF_K)` (~L1532-1538, k HARD-CODED).
- `fusion.resolve_rrf_k(value) -> int` exists, validated, UNCONSUMED (fusion.py ~L264). `resolve_hybrid_alpha` shows the consumption pattern.
- The `hybrid_fusion` metadata block records `alpha`/`rrf_k` per row; P1's `local_citation_capture._reliable_rrf` RE-DERIVES the fused score from the recorded values — record the ACTUAL configured values or every hybrid row silently degrades to LEGACY score-kind. `Tests/RAG/test_local_citation_capture.py` hand-computes this arithmetic.
- Library hybrid arm: `_search_hybrid` (library_local_rag_search_service.py ~L668) calls `search(search_type="hybrid", ...)` (~L711); the caller `_search_rag` (~L323) holds `source_types` (plural vocabulary) and the FTS-servable gate `_FTS_SERVABLE_SOURCE_TYPES`.
- Comparison-runner seam: P1's `run_eval` flips `runtime.service.config.search.default_search_mode` in try/finally (Tests/RAG_Eval/harness/runner.py ~L265-280) — the comparison runner flips `rrf_k`/multiplier/alpha fields the same way. `build_eval_runtime` + `load_fixtures` are the setup; golden id `kw-plant-maintenance-record`, fixture slug `note-saltmarsh-hide`.
- P1 baseline facts: hybrid ≡ semantic on recall/MRR/NDCG (1.000 / 1.000 / 1.000 overall); `kw-plant-maintenance-record`: plain rank 1, semantic ABSENT from top-10 (present ~rank 22 in the index), hybrid absent (sorts 21st). Warn band 0.02, fail band 0.05.
- ADR-005: `backlog/decisions/005-invest-in-local-rag-mirroring-tldw-server.md` (k=60/α=0.7 provenance; Task 5 adds the addendum).

---

### Task 1: Backlog bookkeeping

**Files:** `backlog/tasks/task-4110 ...md`, `backlog/tasks/task-14751 ...md` (via CLI).

- [ ] **Step 1:** Read both task files (`backlog task 4110 --plain`, `backlog task 14751 --plain`). Set both `-s "In Progress"` with a one-line `--plan` referencing this plan + the spec. Report any AC/design conflict rather than editing silently.
- [ ] **Step 2:** Commit `chore(backlog): fusion-weighting arc tasks 4110+14751 in progress` (+ trailer), push.

---

### Task 2: TASK-14751 — source-type pushdown into the keyword leg

**Files:**
- Modify: `tldw_chatbook/RAG_Search/simplified/rag_service.py` (`search`, `_hybrid_search`, `_keyword_search`, `_chacha_keyword_sublegs`, `_media_keyword_subleg`)
- Modify: `tldw_chatbook/Library/library_local_rag_search_service.py` (`_search_hybrid` passes the translated selection)
- Test: `Tests/RAG_Search/test_keyword_leg_pushdown.py` (create)

**Interfaces:**
- Produces: keyword-only kwarg `keyword_source_types: Collection[str] | None = None` on `RAGService.search`, `_hybrid_search`, `_keyword_search` (None ⇒ all three; every existing caller unchanged — verify no positional callers first). Values are ENGINE vocabulary (`media`/`note`/`conversation`); unknown values ignored with a debug log (fail-open to fewer sub-legs, never a crash); empty collection ⇒ no sub-legs ⇒ leg returns [] (hybrid degrades to semantic via the existing disclosed path). Library `_search_hybrid` translates its plural selection via a small pure map `_ENGINE_KEYWORD_SOURCE_TYPES = {"media": "media", "notes": "note", "conversations": "conversation"}` (prompts absent — not FTS-servable).
- Behavior: only selected sub-legs RUN (skipped sub-legs are never queried — assert via spies, not just absent rows); single-type selection gives that sub-leg the FULL top_k in its natural best-first order; multi-type selections keep rank-fair interleaving.

- [ ] **Step 1: READ** `_keyword_search`'s orchestration + `_chacha_keyword_sublegs` (whether notes/conversations can be skipped independently) + `search()`'s kwarg plumbing + all `_keyword_search` callers (`grep -n "_keyword_search(" `) — confirm none pass positionally past top_k.
- [ ] **Step 2: RED tests** (real DBs via writer APIs, the test_keyword_leg_chacha.py pattern — NOT canned fakes; that blindness is why 14751 exists):

```python
def test_media_only_selection_gets_the_full_budget():
    # 12 matching media + 12 notes + 12 conversations seeded; top_k=20,
    # keyword_source_types={"media"} -> 12 media rows (all of them), zero others.
    # THE AC#2 PIN: reds if the budget silently reverts to a three-way split.

def test_unselected_sublegs_are_never_queried():
    # spies on _media_keyword_subleg/_chacha_notes_fts/_chacha_conversations_fts;
    # {"note"} selection -> media + conversations spies not called.

def test_multi_type_selection_keeps_rank_fair_interleaving():
    # {"media","note"}: first slots alternate media/note (AC#4 — no concatenation).

def test_none_means_all_three_unchanged():
    # None reproduces today's composition exactly (backward compat pin).

def test_empty_selection_returns_empty_leg():
    # set() -> [] without querying anything.

def test_library_hybrid_passes_translated_selection():
    # Library fixture with notes-only selected; spy on RAGService.search
    # asserts keyword_source_types == {"note"} (singular).  THE VOCABULARY PIN.
```

- [ ] **Step 3:** RED → implement → GREEN. Run: new file + `test_keyword_leg_chacha.py` + `test_hybrid_doc_fusion.py` + `test_library_rag_mode_resolution.py` + `Tests/DB/test_private_sqlite_inventory.py` (counts).
- [ ] **Step 4: Mutation:** drop the pushdown (ignore the kwarg) → the media-only and never-queried tests red, everything else green. Edit-restore.
- [ ] **Step 5: Informational gated run** (`RAG_EVAL=1 pytest Tests/RAG_Eval/ -q -p no:randomly`): expect +0.000 everywhere (harness selects all three) — record, don't celebrate.
- [ ] **Step 6:** Tick 14751 ACs #1/#3/#4 (+#2 if the full-budget pin covers its wording — read it), Implementation Notes, `-s Done`. Commit `fix(rag): push selected source types into the hybrid keyword leg (TASK-14751)` + `Refs TASK-14751.` + trailer. Push.

---

### Task 3: Config knobs + metadata honesty (the measurement prerequisites)

**Files:**
- Modify: `tldw_chatbook/RAG_Search/simplified/config.py` (SearchConfig: `rrf_k: int = 60`, `hybrid_pool_multiplier: int = 2` — follow the dataclass's existing field/TOML-round-trip style)
- Modify: `tldw_chatbook/RAG_Search/simplified/rag_service.py` (`_fuse_hybrid_results` reads `resolve_rrf_k(self.config.search.rrf_k)`; `_hybrid_search`'s TWO leg fetches use `top_k * self.config.search.hybrid_pool_multiplier`; `_semantic_search`'s own multiplier use UNTOUCHED; the `hybrid_fusion` metadata block records the ACTUAL alpha/rrf_k used)
- Test: `Tests/RAG_Search/test_fusion_config_knobs.py` (create)

**Interfaces:**
- Produces: `config.search.rrf_k` (validated via the existing `fusion.resolve_rrf_k`; invalid → default 60 with a warning, matching `resolve_hybrid_alpha`'s pattern) and `config.search.hybrid_pool_multiplier` (floor 1, sanity cap — read whether config.py has a clamp idiom; hybrid legs ONLY). Task 4's runner flips these fields per pass; Task 5 ships winners by changing DEFAULTS here.

- [ ] **Step 1: RED tests:**

```python
def test_rrf_k_config_reaches_the_fusion_call():
    # service with config.search.rrf_k=10; hand-run _fuse_hybrid_results on two
    # hand-built legs; a fused row's metadata["hybrid_fusion"]["rrf_k"] == 10
    # AND the fused score arithmetic matches 1/(10+rank) terms (not 1/(60+rank)).

def test_metadata_records_actual_values_and_rederivation_certifies():
    # THE METADATA-HONESTY PIN: run a fused row (rrf_k=10) through
    # local_citation_capture's normalization; assert the score-kind is RRF,
    # not LEGACY (the re-derivation must hold with non-default k).

def test_pool_multiplier_widens_hybrid_legs_only():
    # spies: _hybrid_search with multiplier=5, top_k=4 -> semantic+keyword legs
    # asked for 20; then _semantic_search directly (semantic MODE) still uses
    # the module SEARCH_RESULT_MULTIPLIER (2) — semantic-mode behavior unchanged.

def test_invalid_rrf_k_falls_back_to_default_with_warning(): ...
def test_defaults_unchanged():
    # fresh config -> rrf_k 60, multiplier 2; a default-config fused row's
    # arithmetic is byte-identical to pre-branch (protected-oracle insurance).
```

- [ ] **Step 2:** RED → implement → GREEN. Run: new file + `test_hybrid_fusion_metadata.py` UNMODIFIED + `Tests/RAG/test_local_citation_capture.py` + `Tests/RAG/test_fusion.py` + the inventory suite. Read counts.
- [ ] **Step 3: Informational gated run**: defaults unchanged ⇒ expect +0.000; record.
- [ ] **Step 4:** Commit `feat(rag): config-thread rrf_k and hybrid pool multiplier; fusion metadata records actual values` + `Refs TASK-4110.` + trailer. Push.

---

### Task 4: The comparison harness + THE MEASUREMENT

**Files:**
- Create: `Tests/RAG_Eval/harness/fusion_sweep.py` (the runner), `Tests/RAG_Eval/test_fusion_sweep.py` (env-gated entry)
- Test: always-on shape tests for the runner's pure aggregation only (no model).

**Interfaces:**
- Consumes: `build_eval_runtime`, `load_fixtures`, `run_eval`'s per-mode machinery or a hybrid-only variant (read runner.py first — reuse its report/latency helpers rather than duplicating; verification item 3), Task 3's config fields.
- Produces: `run_fusion_sweep(runtime, golden, strategies) -> SweepReport` where each strategy is a dataclass `(name, rrf_k, hybrid_pool_multiplier, hybrid_alpha)`; per strategy: per-category recall/MRR/NDCG/precision for hybrid mode + the rescue verdict for `kw-plant-maintenance-record` (present in top-10: yes/no, rank, merged-or-fts-only). `format_matrix() -> str` prints the decision table. Restores all config fields in finally.
- The strategy matrix (from the spec): control (60/2/.7); k-sweep {5,10,20} at 2/.7; multiplier {3,5} at 60/.7; the two best combined; alpha-combo ONLY if nothing else qualifies. Quota is NOT implemented in this task (YAGNI rule — only if the sweep fails).

- [ ] **Step 1:** Always-on RED tests for the pure parts (matrix formatting; strategy dataclass; a hand-built SweepReport's qualification check implementing the spec's decision rule: AC#4-satisfiable weighting present, no recall/MRR/NDCG cell regressing > 0.02, tie-break order rrf_k → quota → alpha-combo, pool widening as companion).
- [ ] **Step 2:** Implement runner + gated test. **Step 3: THE MEASUREMENT RUN** (`RAG_EVAL=1`, minutes): produce the full matrix; paste it VERBATIM into the task report AND `.superpowers` ledger. Apply the decision rule mechanically. If no strategy qualifies → STOP, report BLOCKED with the matrix (the spec says the owner chooses; do not improvise a quota in this task).
- [ ] **Step 4:** Commit `feat(rag-eval): fusion strategy sweep runner + the decision matrix` + `Refs TASK-4110.` + trailer. Push. Report states the WINNER + its numbers.

---

### Task 5: Ship the winner (TASK-4110)

**Files:**
- Modify: `tldw_chatbook/RAG_Search/simplified/config.py` (the winning DEFAULTS — one authoritative site) and/or profile configs if the winner is profile-scoped (read config_profiles.py; default-profile users must get the fix without touching Settings)
- Modify: `backlog/decisions/005-invest-in-local-rag-mirroring-tldw-server.md` (one-line addendum: constants refined by measurement for chatbook's ~20-row pools; date + PR ref)
- Test: `Tests/RAG_Search/test_fusion_rescue_pin.py` (create) + disclosed updates to any oracle pinning old defaults
- Test (gated): the AC#5 fixture guard added to `Tests/RAG_Eval/test_harness_run.py` or the sweep test file

**Interfaces:**
- Consumes: Task 4's winner (exact values in its report).
- Produces: shipped defaults = the winner; `test_fusion_rescue_pin.py`:

```python
def test_fts_only_row_outranks_a_vector_only_row_under_shipped_defaults():
    # AC#4: hand-built legs — FTS-only doc at fts rank 1; vector leg with >=
    # top_k distinct documents; under a DEFAULT config the FTS-only doc appears
    # in fused top-k, outranking at least one vector-only row.  Mutation:
    # reverting defaults to 60/2/.7 REDS this test.

def test_semantic_mode_still_misses_the_vector_blind_fixture():  # gated, AC#5
    # semantic mode does NOT return note-saltmarsh-hide for
    # kw-plant-maintenance-record (corpus keeps distinguishing coverage from noise).
```

- [ ] **Step 1:** Implement the winning defaults; sweep for oracles pinning old constants (`grep -rn "61\|/ 61\|0.0164\|DEFAULT_RRF_K" Tests/` + run the battery); each legitimate pin update DISCLOSED in the report with before/after.
- [ ] **Step 2:** RED→GREEN the two tests above; mutation check (revert defaults → rescue pin reds). **Shared-blend guard:** `reciprocal_rank_fusion`'s own DEFAULTS stay untouched (the winner lives in config, threaded at the call site) — verify `pipeline_builder_simple`'s behavior is byte-unchanged (its call sites still pass/inherit the old values; grep + its tests green); note on TASK-3501 ONLY if anything shared moved.
- [ ] **Step 3: Informational gated run:** hybrid's rescue cell flips; record the full table for Task 6's progression.
- [ ] **Step 4:** Tick 4110 ACs #1/#2/#4/#5 (re-stamp AC #3 awaits Task 6), Implementation Notes with the measured justification. Commit `fix(rag): ship the measured fusion weighting — hybrid rescues keyword-only documents (TASK-4110)` + trailer. Push.

---

### Task 6: Re-stamp, closure, live check, acceptance

**Files:** `Tests/RAG_Eval/baselines/*.json` (deliberate re-stamp), `Tests/RAG_Eval/README.md` (progression + the 4110 story; retire the "known defect" entry), `Docs/User_Guide/library/search-and-rag.md` (hybrid description + stamp), `backlog/tasks/task-3994 ...md` (#2b ticked with rescue evidence), task-4110 (#3 ticked, Done).

- [ ] **Step 1: Deliberate re-stamp** (`RAG_EVAL=1 RAG_EVAL_UPDATE_BASELINES=1`): full delta printout pasted; progression table (P1-era baseline → post-14751 (+0.000, expected-by-construction) → post-winner = stamped) with the mechanism note for every moved cell; `kw-plant-maintenance-record` hybrid cell miss → hit is the headline row. Plain gated run after → gate PASSED.
- [ ] **Step 2:** Ungated battery: every file this branch touched + the protected-oracle set + the inventory suite, one run, counts. Collection arithmetic vs merge-base (baseline worktree `.worktrees/weighting-baseline`, removed after, arithmetic shown).
- [ ] **Step 3: Live TUI check** (PR-2 recipe, tmux `-L weighting`, scratch profile, copied real DBs + chromadb): a query whose answer exists only as an exact keyword match in a note appears in hybrid results on the default profile — the user-visible rescue. Capture evidence; full teardown checklist (config hash, scratch deleted, kill-server, worktree list clean, git status clean).
- [ ] **Step 4:** README/guide updates; ADR addendum verified in place; TASK-3994 #2b ticked quoting its criterion; 4110 Done; lessons entry ONLY if something new generalized (the measurement-before-tuning discipline is already the programme's premise — don't restate it as a lesson).
- [ ] **Step 5:** Commit `chore(rag): weighting-arc re-stamp + closure` + `Refs TASK-4110.` + trailer. Push.

---

## Self-review (done at plan time)

- **Spec coverage:** pushdown → Task 2 (all four 14751 ACs); parameterization prerequisites + metadata honesty → Task 3; comparison harness, matrix, decision rule, YAGNI-quota, BLOCKED path → Task 4; winner + AC#4/#5 pins + ADR addendum + shared-blend guard → Task 5; re-stamp/progression/live/closure/3994 #2b → Task 6. Two rescue senses live in Task 4's qualification check + Task 5's separate pins. Out-of-scope list respected.
- **Placeholder scan:** test skeletons carry exact contracts with read-first fill instructions (house pattern); no TBDs.
- **Type consistency:** `keyword_source_types` (Tasks 2), `config.search.rrf_k`/`hybrid_pool_multiplier` (Tasks 3/4/5), strategy dataclass fields (Task 4 → 5 winner values); `kw-plant-maintenance-record`/`note-saltmarsh-hide` ids used consistently.
- **Sequencing:** 14751 before measurement (corrected leg), knobs before sweep, sweep before shipping, one re-stamp at the end — matches the spec's load-bearing order.
