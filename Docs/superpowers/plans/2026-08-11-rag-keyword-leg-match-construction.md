# TASK-15400: Keyword-Leg MATCH Construction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the engine keyword leg's AND-of-every-token MATCH construction with the winner of a four-candidate sweep under a pre-registered mechanical rule — expected winner `and_then_or` (AND first, content-token OR only when AND returns zero rows), unblocking natural-language queries (40/60 return zero rows today) and the prompt category.

**Architecture:** One construction seam in `rag_service.py` (a `SearchConfig` field + a match-expressions builder + a zero-row fallback loop in each FTS sub-leg helper, with per-row provenance), keyed into the hybrid cache exactly as the fusion params are; the sweep extends `fusion_sweep.py`'s `Strategy` with a construction axis. Spec is authority: `Docs/superpowers/specs/2026-08-11-rag-keyword-leg-match-construction-design.md` — its pre-registered candidates, hard constraints, and decision rule bind every task. The winner is COMPUTED by the rule, never chosen by taste; a null result ships the table and nothing else.

**Tech Stack:** Python 3.11+, SQLite FTS5 (bm25 rank), the P2ab-renewed eval harness (`RAG_EVAL=1`), pytest.

## Global Constraints

- Spec (read FIRST): `Docs/superpowers/specs/2026-08-11-rag-keyword-leg-match-construction-design.md`. TASK-15400's file carries the measured attribution + Constraints section — read it too (`backlog task 15400 --plain`).
- Worktree `.worktrees/rag-keyword-match`, branch `fix/rag-keyword-leg-match` (off dev `3b1ad8eff`). **cwd silently resets between Bash blocks — start EVERY block with `cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/rag-keyword-match`.**
- Tests: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <paths>`; counts READ; never a directory AND files inside it in one invocation; "no tests ran" = FAILED; never `git stash`; Edit-based restores; single foreground Bash (timeout 600000). TCC "Operation not permitted" under ~/Documents = known transient fault: stop and report.
- INJECTION QUOTING IS LOAD-BEARING: every construction keeps each token individually quoted; `Tests/RAG_Search/test_fts5_query_escaping.py` (10 tests) must stay green and gets EXTENDED per construction, never weakened.
- Fixtures are P2ab's and FROZEN this arc: `Tests/RAG_Eval/fixtures/*` byte-identical until the final re-stamp touches only `baselines/` + README. A construction change flips CELLS, not the corpus hash — intermediate gated runs will read REGRESSION-or-improvement noise on hybrid cells until Task 5's re-stamp; that is expected, recorded, never "fixed" early.
- Protected oracles: the escaping suite; the prompts/chacha/media sub-leg suites; `Tests/DB/test_private_sqlite_inventory.py` in every battery; the gated prompt recall-0 pin and the census (both flip ONLY in Task 5, disclosed, if the winner earns it).
- Commits reference TASK-15400 and end `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; push after every task.

## Verified code anchors (line numbers drift; grep first)

- `rag_service.py::_escape_fts5_query` (~L3082): builds per-token-quoted implicit-AND MATCH; returns `""` = skip (callers already honor it). Tokenization in `_fts5_query_tokens` (SHARED with the citation builder — do not change its semantics). The docstring's "This construction is under review — TASK-15400" block is Task 5's to resolve.
- Consumers: `_media_keyword_subleg` (~L1590 → escaped at ~L1957-region for media SQL), `_chacha_fts_rows` (~L1957, builds once, passes `escaped_query` into `_chacha_notes_fts` ~L1999 and `_chacha_conversations_fts` ~L2070), `_prompts_fts` (~L2390), plus an early-exit check at ~L1521. The fallback loop wraps the SQL-execution helpers, not the tokenizer.
- Cache: **per-service-instance** (`self.cache = SimpleRAGCache(...)` ~L660 with the no-process-global comment). The sweep reuses ONE live runtime and mutates `SearchConfig` via `Strategy.apply` (`fusion_sweep.py:142-173`) — therefore the construction field MUST join the hybrid cache-key part exactly as rrf_k/alpha/pool did (`simple_cache.py`'s fusion-param tuple; the weighting arc's lesson). This resolves the spec's injection-seam question: **`SearchConfig` field, not TOML-wired, keyed**.
- Sweep: `Strategy` frozen dataclass (name/rrf_k/hybrid_pool_multiplier/hybrid_alpha, `apply()`, `changed_fields()`); `StrategyReport` (~L337); `run_fusion_sweep` calls `run_eval(runtime, golden, k, modes=(HYBRID_MODE,), ...)` per pass. `runner.UNAVERAGED_CATEGORIES`/`count_scored()` = the single scored predicate.
- Census + attribution: `Tests/RAG_Eval/test_fixture_authoring_probe.py` (~L131,151) documents the 40/60 and the 1/40 figures; the leg-level census = "target enters the keyword leg's top-10", 20/60 shipped. Negative probes: `runner.NegativeProbe` (docs_at_k/top_score/top_vector_score); hybrid provenance `metadata["hybrid_fusion"]` (`fts_rank`/`vector_rank`).
- Stopword trimming: NO list exists in the module — Task 2 introduces `_FTS5_STOPWORDS` (small fixed English set, module-level frozenset). It applies ONLY to OR-form construction; AND behavior for shipped default is byte-identical.

---

### Task 1: Backlog + the construction seam (engine)

**Files:**
- Modify: `tldw_chatbook/RAG_Search/simplified/config.py` (SearchConfig field), `tldw_chatbook/RAG_Search/simplified/rag_service.py` (builder + fallback loop + provenance), `tldw_chatbook/RAG_Search/simplified/simple_cache.py` (key part)
- Test: `Tests/RAG_Search/test_fts5_query_escaping.py` (extend), `Tests/RAG_Search/test_fts5_match_construction.py` (create)

**Interfaces (produces — Tasks 2-5 rely on these exact names):**
- `SearchConfig.fts_match_construction: str = "and"` — values `"and" | "and_stopword_trim" | "or" | "and_then_or"`; NOT wired to TOML/user config (no config-template entry, no resolver); invalid value → warn once + behave as `"and"` (fail-safe to shipped behavior).
- `RAGService._fts5_match_expressions(query: str) -> tuple[str, str | None]` — (primary MATCH expression, fallback expression or None), per the active construction: `and` → (AND, None); `and_stopword_trim` → (AND-over-content-tokens or full AND if trimming empties, None); `or` → (content-token OR or "" if trimming empties, None); `and_then_or` → (AND, content-token OR or None). Every token in every form individually quoted via the existing quoting; `""` primary = skip (existing contract).
- `_FTS5_STOPWORDS: frozenset[str]` — module-level, lowercase.
- Each raw FTS helper (`_media…`, `_chacha_notes_fts`, `_chacha_conversations_fts`, `_prompts_fts`) runs primary; on ZERO rows and a non-None fallback, runs the fallback ONCE; rows carry `metadata["fts_match"] = "and" | "or_fallback"` (Task 2's sweep + Task 5's prose read it).
- Hybrid cache key: the construction string joins the existing fusion-param tuple (same-order canonicalization; `"and"` must render the key BYTE-IDENTICAL to pre-arc keys — the legacy-invariant discipline from B1a).

- [ ] **Step 1:** `backlog task edit 15400 -s "In Progress"` + `--plan` referencing spec+plan. (The arc rides 15400 itself — no new arc task.)
- [ ] **Step 2:** READ the spec's candidate + constraint sections; `_escape_fts5_query` + `_fts5_query_tokens` + all four consumer sites + `_make_key`'s fusion tuple.
- [ ] **Step 3:** RED-first, in `test_fts5_match_construction.py` (real in-memory DBs, the chacha/prompts suites as fixtures templates): per-construction expression shape (AND unchanged; trim falls back to full AND when all-stopword; OR trims stopwords and returns "" when emptied; and_then_or returns both); fallback fires ONLY on zero primary rows (a one-row AND result never falls back — spy/counter); fallback rows stamped `or_fallback`, primary rows `and`; invalid construction warns + behaves as `and`; cache key: construction in the hybrid key, `"and"` byte-identical to the pre-arc rendering, different constructions → different keys; per-token quoting preserved in OR forms (extend the escaping suite with the OR/mixed cases — an injection attempt through the fallback path must still be inert).
- [ ] **Step 4:** Implement. Mutations: (a) fallback loop removed → its test reds; (b) quoting dropped from the OR form → escaping suite reds; (c) construction dropped from the key → the key test reds.
- [ ] **Step 5:** Battery: the new file + escaping suite + `Tests/RAG_Search/test_keyword_leg_prompts.py` + `test_keyword_leg_pushdown.py` + `test_hybrid_allowlist_pushdown.py` + `Tests/DB/test_private_sqlite_inventory.py`; counts. Default `"and"` ⇒ every existing count unchanged.
- [ ] **Step 6:** Commit `feat(rag): FTS5 match-construction seam — four candidates behind one keyed field (TASK-15400)` + trailer. Push.

---

### Task 2: The sweep's construction axis

**Files:**
- Modify: `Tests/RAG_Eval/harness/fusion_sweep.py` (Strategy axis + census + negative-composition record), `Tests/RAG_Eval/test_fusion_sweep.py` (always-on pins)
- Create: nothing (the axis lives in the existing module)

**Interfaces (produces):**
- `Strategy` gains `fts_match_construction: str = "and"`; `apply()` writes it onto the live `SearchConfig`; `changed_fields()` includes it. Existing strategy tuples unchanged (default keeps every current row meaning-identical — pin one).
- `CONSTRUCTION_STRATEGIES: tuple[Strategy, ...]` — exactly four rows at SHIPPED fusion params (rrf_k=5, pool 2, alpha 0.7): `and` (control), `and_trim`, `or`, `and_or` (names ≤10 chars per the dataclass's own rule).
- Per-row additions to the report: `census_hits: int` (golden queries whose target appears in the KEYWORD LEG's top-10 — leg-level, via a direct `_keyword_search` pass over the golden set, negatives/scoped excluded per `count_scored` semantics... NOTE: scoped queries DO count in the census — they hit AND today; use ALL non-negative queries, 53) and `negative_fallback_rows: int` (FTS-only rows inside hybrid top-10 across the 7 negatives, read off `metadata["hybrid_fusion"]` + `metadata["fts_match"] == "or_fallback"`).
- Control-row SELF-CHECK: `and`'s census MUST equal the shipped 20 — a mismatch raises before any other row runs (the cache-blindness alarm; the per-service cache makes this pass when passes rebuild/clear correctly — `apply()` callers must `runtime.service.cache.clear()` per pass, pinned by a test).
- NEAR/prefix probes: one function running the two variants over ONLY the 40 zero-row queries, report-only (promoted to a full row only if either beats `and_or`'s census — the spec's rule).

- [ ] **Step 1:** READ the sweep's pass loop, `StrategyReport`, and how Task-6's counterfactual cleared cache. RED-first always-on pins (synthetic outcomes): the axis round-trips through `apply`; the self-check raises on census mismatch; the negative-composition counter counts only `or_fallback` FTS-only rows; existing-rows-unchanged pin.
- [ ] **Step 2:** Implement; run the always-on sweep tests + `Tests/RAG_Eval` ungated (counts unchanged elsewhere). Mutation: cache-clear dropped from the pass loop → the self-check test reds (prove the alarm actually fires when blinded).
- [ ] **Step 3:** Commit `test(rag-eval): construction axis + census/negative-composition instrumentation for the 15400 sweep` + trailer. Push.

---

### Task 3: THE SWEEP RUN + mechanical decision

**Files:** report only (`.superpowers` ledger + task file note) — NO production change in this task.

- [ ] **Step 1:** Gated (`RAG_EVAL=1`): run the four-row construction sweep + the NEAR/prefix probes. Capture VERBATIM: per-row census, per-category gated cells (all three modes — plain/semantic must be byte-identical across rows: the zero-movement proof that only hybrid can move), negative probes + negative_fallback_rows, the vector-blind fixture's hybrid rescue per row, wall-time per row.
- [ ] **Step 2:** Apply the spec's decision rule MECHANICALLY, in writing: hard constraints (a) vector-blind hybrid rescue, (b) no gated cell regresses > 0.02, (c) escaping suite green; winner = max census subject to constraints; tie-break fewest extra FTS queries → smallest code delta; negative_fallback_rows recorded into the tie-break narrative. Expected: `and` census 20, `and_trim` ~21, `or` disqualified by (a), `and_or` ~30 and winner — but THE TABLE DECIDES; if the expectation breaks, follow the table. If NO candidate passes: STOP, report, and Task 5 becomes "ship the table + re-scope 15400" per the spec's null-result clause.
- [ ] **Step 3:** Report with the full table + the rule's application line-by-line. Nothing committed beyond the ledger/task note. (The reviewer re-runs the sweep.)

---

### Task 4: Ship the winner

**Files:**
- Modify: `tldw_chatbook/RAG_Search/simplified/config.py` (default flips to the winner), `rag_service.py` (the `_escape_fts5_query` docstring's under-review block resolves to the outcome + the winner's mechanism, stated from Task 3's TABLE, never paper arithmetic), `Tests/RAG_Eval/test_fixture_authoring_probe.py` + `Tests/RAG_Eval/fixtures/golden.toml`?? — NO: fixtures FROZEN; the 40/60 prose sites get updated in Task 5 with the re-stamp. This task: the default flip + docstring + the gated prompt recall-0 pin flip IF the winner earns it (DISCLOSED: docstring names both states, dates, the sweep row).
- Test: existing suites at the new default — the escaping suite, the construction suite (its default-behavior pins update as DISCLOSED flips), prompts/pushdown/allowlist suites, ungated `Tests/RAG_Eval`.

- [ ] **Step 1:** Flip `fts_match_construction`'s default to Task 3's winner. Enumerate every test that pins default behavior and flips — each is a disclosed oracle update named in the report (before/after).
- [ ] **Step 2:** Gated run (no re-stamp yet): capture the hybrid cells' movement + the census recount + the prompt pin. Battery counts.
- [ ] **Step 3:** Mutation: default reverted to `"and"` → the census/prompt pins red (the winner is load-bearing).
- [ ] **Step 4:** Commit `feat(rag): keyword leg ships <winner> — chosen by the 15400 construction sweep` + trailer. Push.

---

### Task 5: Re-stamp + closure + live check

**Files:** `Tests/RAG_Eval/baselines/*.json`, `Tests/RAG_Eval/README.md` (census prose + headroom table update; the 40/60 sites re-attributed to the measured outcome), `rag_service.py` 40/60 docstring sites, `Docs/User_Guide/library/search-and-rag.md` ("Prompts are keyword-only… keyword-shaped queries" paragraph updates if prompt behavior widened + stamp after the live check), backlog (15400 Done w/ the sweep table; one-line outcome note on TASK-3997), lessons if earned.

- [ ] **Step 1:** ONE deliberate re-stamp (`RAG_EVAL_UPDATE_BASELINES=1 RAG_EVAL=1`); full delta printout VERBATIM; reconcile every moved cell against Task 3's table (any surprise = STOP); fresh gated run reads `PASSED`.
- [ ] **Step 2:** README: census updated (20/60 → the winner's number, dated, sweep-cited); headroom table: prompt row re-attributed (unblocked or the residual bound stated); the negative-composition record published beside it.
- [ ] **Step 3:** LIVE CHECK (lessons-live-verification.md; scratch profile, config-hash before/after, teardown checklist): a natural-language prompt query through Library RAG Answer on the hybrid profile — the exact class that returns nothing today — finds a prompt (`| keyword match`). Stamp the User Guide page at the SHA that exists when you stamp.
- [ ] **Step 4:** 15400 Done (all ACs against evidence; Implementation Notes carry the table); 3997 note; grep the retired prose values ("40 of 60", "returns ZERO rows", the old census) — every site updated or explicitly historical.
- [ ] **Step 5:** Closing battery (Tests/RAG_Eval + Tests/RAG_Search + Tests/Library + inventory; counts; name pre-existing failures against their filed tasks) + collection sweep vs merge-base `3b1ad8eff`. Commit(s) + trailer. Push.

---

## Self-review (done at plan time)

- **Spec coverage:** seam+candidates → T1; sweep axis+census+negative-composition+self-check+probes → T2; the run+rule → T3; winner+disclosed flips → T4; re-stamp+headroom+live+closure → T5. Null-result clause routed (T3 Step 2 → T5 ships the table). Error handling in T1 (invalid value fail-safe; empty-after-trim; fallback inherits degrade paths — pinned in T1's suite).
- **Placeholder scan:** clean; the one deliberate open value is the winner (computed in T3 by the pre-registered rule — that is the arc's point, not a placeholder).
- **Type consistency:** `fts_match_construction` (T1 field = T2 Strategy axis = T4 default flip); `_fts5_match_expressions` (T1) consumed by sub-legs only; `metadata["fts_match"]` (T1) read by T2's counter and T5's prose; census=20 control (T2 self-check) = T3's control row.
- **Census denominator note:** 53 non-negative queries; the shipped census 20 counts hits among ALL golden queries' targets in the leg's top-10 — T2 Step 1 must reproduce the EXACT counting used by the Task-7 census (read the probe's method first) so the self-check compares like with like.
