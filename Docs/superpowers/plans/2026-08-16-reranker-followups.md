# TASK-3502: Reranker Follow-ups — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the now-real reranker honest: provider/model controls with a cost disclosure, robust copy/counter semantics, a visible failure disclosure, and no "| reranked" over-claim.

**Architecture:** Three tasks. T1 = engine (AC#3, AC#4, note-b) — tests + fixes at the reranker/service seam, all provider calls faked. T2 = UI (AC#1, AC#2, note-a) — Settings fold + Library panel notice. T3 = closure + the cross_encoder filing. Gate 105/105 in T1 (the stamping change touches the semantic path) and re-read in T3.

**Spec:** `Docs/superpowers/specs/2026-08-16-reranker-followups-design.md` — binds every task; the out-of-scope fences (no cross_encoder, no live LLM calls, no profile-default changes) are hard.

## Global Constraints

- Worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/rag-3502-reranker`, branch `feat/rag-3502-reranker-followups` (off dev `2b1d1817f`). **cwd resets every Bash block — cd first, EVERY block, especially before any push.**
- **VENV (none exists):** `uv venv .venv --python 3.12 && VIRTUAL_ENV=.venv uv pip install -e ".[dev,embeddings_rag]" "transformers==5.6.2" "torch==2.11.0" "chromadb==1.5.8"`; paste the in-worktree provenance line. ruff via `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff`.
- Never `git stash`; Edit restores; single foreground Bash (timeout 600000); do NOT run `Tests/UI/test_library_shell.py`. RED-first everywhere. **NO live provider calls** — the reranker's seam is `chat_api_call` (`RAG_Search/reranker.py:24`, invoked via executor at `:204`): fake it there.
- Gate: `RAG_EVAL=1 .venv/bin/python -m pytest Tests/RAG_Eval/ -q -p no:randomly` reads verbatim `PASSED: No regression. 105 metric(s)`.
- Commits reference TASK-3502, end `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; push after every task.

## Verified code anchors (grep first; lines drift)

- `RAG_Search/reranker.py`: `RerankingConfig` defaults `openai`/`gpt-3.5-turbo` (`:52-53`); `BaseReranker.last_rerank_failures/_total` instance state (`:99-107`, setter `_record_failure_counts`-style at `:104-107`); `PointwiseReranker._apply_scores` (`:387`); **the over-claim site: failed rows stamped `rerank_score=result.score` at `:384`** (success path `:371`); `PairwiseReranker:458`, `ListwiseReranker:596`.
- `RAG_Search/simplified/enhanced_rag_service_v2.py`: `_tag_first_result` (`:38`, copy-not-mutate docstring states the cache-by-reference hazard); `reranking_degraded` tag at `:328-330`; `reranking_skipped` at `:337`. The disclosure counters are read near the `:330` site — that read is what AC#4 re-scopes.
- Settings: `UI/Screens/settings_library_rag_defaults.py:56-68` (`enable_reranking`, `reranker_model` — blank = reranker default, `reranker_top_k=20` with the honesty-pin note); `UI/Screens/settings_rag_profile_adapter.py:136` (read), `:193-195` (write: `RerankingConfig()` then `model_name` if non-blank). **No provider field exists anywhere in the chain — that is AC#1's gap.** `RerankingConfig` constructors: adapter `:193`, `config_profiles.py:170` (deserialize), `:353`/`:404` (hybrid_full + accurate profiles).
- Score kinds: `Library/library_rag_score_kinds.py` (`:50` "what a REAL reranked row carries", `:80` the hide-similarity rule) — note-(b)'s contract comment lives here.
- Fold plumbing: `settings_screen.py:1132-1134` (fold map ids incl. `reranker-model`, `reranker-top-k`), `:1295` (toggle), `:1391-1396` (fold summary/"Saved as").

---

### Task 1: Engine honesty — AC#3, AC#4, note-(b)

**Files:** Modify `tldw_chatbook/RAG_Search/reranker.py`, `tldw_chatbook/RAG_Search/simplified/enhanced_rag_service_v2.py` (the counter-read site); Test `Tests/RAG_Search/test_reranker_degraded_paths.py` (new; check for an existing reranker test file first and extend it instead if one exists).

**Interfaces (produces):** the reranker's per-call failure counts are RETURNED to the caller (e.g. `rerank()` returns results plus counts, or a small result object) — T2 does not consume this, but the disclosure site in `enhanced_rag_service_v2.py` does; keep the change minimal and name the final shape in your report.

- [ ] **Step 1:** Venv; provenance pasted. `backlog task edit 3502 -s "In Progress"` + `--plan` (spec+plan paths).
- [ ] **Step 2 (AC#3, RED-first):** with a fake `chat_api_call`, drive the REAL `PairwiseReranker` and `ListwiseReranker` through a degraded run (some comparisons/permutations fail) against a results list held by a simulated cache reference; assert the ORIGINAL objects and list are byte-identical after (the `_tag_first_result` copy contract), and the returned list carries the tag. If either strategy mutates: fix with the same copy-not-mutate shape Pointwise got, minimally.
- [ ] **Step 3 (note-b, RED-first):** a 3-of-5-failed pointwise run: assert failed rows do NOT carry the reranked score kind (they keep their original kind/score; the row-level claim matches what happened) while succeeded rows do. Fix at `:384`: failed rows keep their original score WITHOUT the reranked stamp; update `library_rag_score_kinds.py`'s contract comment. Verify the Library score-kind tests still pass and extend the one that pins "what a REAL reranked row carries".
- [ ] **Step 4 (AC#4, scoping not locking):** make the per-call counts flow to the disclosure site without shared instance state: the rerank entry point returns counts; `enhanced_rag_service_v2.py`'s tag site uses the returned counts; the instance attributes are removed (grep consumers first — if anything else reads them, name it and decide). Pin structurally: a test asserting the disclosure string is built from the returned counts (fake two interleaved calls with different failure counts; each tag reflects its own).
- [ ] **Step 5:** Gate verbatim; `Tests/RAG_Search/` + `Tests/Library/` counts READ; ruff. Commit `fix(rag): reranker degraded-path honesty — copy semantics, scoped counters, no over-claim (TASK-3502)` + trailer. Push (cd in-block).

---

### Task 2: UI honesty — AC#1, AC#2, note-(a)

**Files:** Modify `UI/Screens/settings_library_rag_defaults.py` (+ `reranker_provider: str = ""` field), `UI/Screens/settings_rag_profile_adapter.py` (read/write `model_provider`), `UI/Screens/settings_screen.py` (the fold: provider Select + disclosure text + fold-map ids), the Library RAG panel notice site (locate via `recovery_state`/notice rendering in `Widgets/Library/library_search_rag_panel.py` — reuse the existing notice vocabulary); Tests: the settings-RAG suite files (grep `enable-reranking` under Tests/UI) + the panel's test file.

- [ ] **Step 1 (AC#1, RED-first):** adapter round-trip tests: a profile with `model_provider="anthropic"` reads into the new field; a save with the field set writes `reranking_config.model_provider`; blank = leave default (the `reranker_model` precedent at `:194` — same shape). Then the fold control: a Select with the providers the repo's provider registry actually supports for chat (enumerate from `API_CALL_HANDLERS` keys — do not hand-list), default entry labeled explicitly "openai (default)". Fold-map + summary entries updated.
- [ ] **Step 2 (AC#2):** the disclosure text adjacent to the toggle, stating one-call-per-candidate up to the configured top-k, visible without enabling; pinned by a compose test asserting the text exists and names the top-k value. Wording final: "Reranking scores each result with a separate {provider} call — up to {top_k} calls per search, billed at that provider's rates."
- [ ] **Step 3 (note-a, RED-first):** the tag consumer: when the search outcome's first result carries `reranking_skipped`/`reranking_degraded` metadata, the Library RAG panel renders one notice line (existing notice styling) naming which and the detail string; test drives the panel state with a tagged outcome. Verify the tag actually SURVIVES into what the panel receives (trace `raw_results` → outcome rows; if it does not survive, carry it on the outcome object — smallest honest plumbing, named in the report).
- [ ] **Step 4:** Batteries: the settings-RAG UI files + the panel file + `Tests/UI/test_console_library_tool_setting.py`; counts READ; ruff. Commit `feat(settings): reranker provider choice + cost disclosure; surface reranking degradation (TASK-3502)` + trailer. Push.

---

### Task 3: Closure + the measurement owner

- [ ] **Steps:** (1) File the cross_encoder task with FULL ID safety (sweep origin/dev + all worktrees + remote branches, leapfrog +100): "implement-or-retire `cross_encoder` reranking, with a gated-instrument measurement" — description cites this arc's out-of-scope reasoning (an LLM reranker cannot be measured deterministically; a local cross-encoder can), the `config_profiles.py` not-implemented comment, and pre-registers that retire is acceptable if the measurement shows nothing. (2) Close TASK-3502: 4 ACs + the two notes items against evidence, Implementation Notes, status Done (content by direct edit). (3) Gate re-read verbatim + collection sweep vs merge-base `2b1d1817f`. (4) Commit + push.

---

## Self-review (plan time)
- Spec coverage: AC#1→T2S1, #2→T2S2, #3→T1S2, #4→T1S4, note-a→T2S3, note-b→T1S3, out-of-scope filing→T3. No live calls anywhere (T1 fakes `chat_api_call`; T2 is UI-only). The one behaviour-visible engine change (note-b stamping) is gate-checked in T1 AND re-read in T3.
- Types: the counts-return shape is T1's to finalize and NAME (its only consumer is in the same task's diff); `reranker_provider` field name consistent across T2's three files.
- Ordering: T1 before T2 (the panel notice test may want the scoped-counts disclosure string final); T3 last.
