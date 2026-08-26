# TASK-15700: Form-Tiered Sub-Leg Merge + Sweep Re-Run — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the keyword leg's cross-sub-leg displacement (fallback sub-legs tiered behind primary sub-legs in the merge), then re-apply TASK-15400's pre-registered decision rule under the fixed merge with two new pre-registered candidates (`prefix`, `and_then_prefix`) — shipping a widening winner if the table qualifies one, recording the null if not.

**Architecture:** Part A: a two-tier partition at `_keyword_search`'s gather site (sub-legs whose rows carry the construction's primary form wholly precede fallback sub-legs; rank-fair round-robin within tiers; `fusion.py` untouched). Part B: two new construction values through the Task-1(15400) seam, two new sweep rows, the rule re-applied verbatim. Spec + TASK-15700's task file are joint authority: `Docs/superpowers/specs/2026-08-13-rag-keyword-leg-tiered-merge-design.md`, `backlog task 15700 --plain`.

**Tech Stack:** Python 3.11+, SQLite FTS5, the P2ab eval harness (`RAG_EVAL=1`), pytest.

## Global Constraints

- Spec + task file first (the seven ACs bind). Worktree `.worktrees/rag-merge-15700`, branch `fix/rag-merge-interleave` (off dev `61f6ae575`). **cwd resets every Bash block — `cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/rag-merge-15700` first, EVERY block.**
- **VENV: this fresh worktree has NO local venv.** Task 1 builds it (`uv venv .venv && VIRTUAL_ENV=.venv uv pip install -e ".[dev,embeddings_rag]"`). Every measurement asserts `import tldw_chatbook; print(tldw_chatbook.__file__)` resolves INSIDE the worktree first (the shared venv path-hooks to the main checkout). The re-stamp alone runs in the MAIN venv (committed-fingerprint match) with `PYTHONPATH` forced to this worktree.
- pytest via the worktree venv; counts READ; no dir+files mixing; "no tests ran" = FAILED; never `git stash`; Edit-based restores; single foreground Bash (timeout 600000). TCC "Operation not permitted" = transient: stop, report.
- INJECTION QUOTING LOAD-BEARING: every construction keeps per-token quoting; `Tests/RAG_Search/test_fts5_query_escaping.py` (37) extends per new construction, never weakens. Prefix syntax: star OUTSIDE the quotes (`"tok"*` — proven in 15400's probes; `'"tok*"'` matches nothing).
- Fixtures FROZEN except where AC#6's one re-stamp touches baselines/README. Intermediate gated runs after Part B's flip read expected movement; after Part A ALONE they must read **105/105 at (+0.000) and control census 20** — either moving is a STOP (the spec's pre-registered intermediate gate).
- Mechanism prose is an ORACLE (three incidents in 15400): state mechanisms from run metadata/tables, never paper arithmetic; floats compared by bit, "tie" claims verified against the shipped expression.
- Commits reference TASK-15700, end `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; push after every task.

## Verified code anchors (line numbers drift; grep first)

- Gather site: `rag_service.py` ~L1655-1665 — `rankings = [r for r in (media_ranking, *chacha_rankings, prompts_ranking) if r]` → `interleave_rankings(rankings, key=_fusion_doc_key)[:top_k]`. The tier partition happens HERE (a partition + two interleave calls + concat + truncate; `fusion.py` untouched — its round-robin semantics stay correct within a tier).
- Form knowledge: every sub-leg row carries an `fts_match` key (`FTS_MATCH_AND="and"` / `FTS_MATCH_OR="or"`, row-level, pre-metadata-promotion — verify the key is still top-level at the gather site; if promotion already happened, read it from where it lives). All-or-nothing per sub-leg is pinned (`test_*_falls_back_independently` ×4).
- Fallback wrapper: `_fts_rows_with_fallback(run_expression, expressions)` ~L3529 — stamps the form; its docstring's "mixed-mode interleave" sentence gets updated by Part A (the mix is now tiered).
- Constants: `FTS_MATCH_CONSTRUCTION_*` ~L159-167, `FTS_MATCH_CONSTRUCTIONS` tuple; form stamps ~L170-190 region. Task 2 adds `FTS_MATCH_CONSTRUCTION_PREFIX = "prefix"`, `FTS_MATCH_CONSTRUCTION_AND_THEN_PREFIX = "and_then_prefix"`, form stamp `FTS_MATCH_PREFIX = "prefix"`.
- Expressions: `_fts5_match_expressions(query) -> (primary, fallback|None)`; term-key suppression exists for and_then_or (identical term sets + len==1). **Prefix fallback must NEVER suppress on term-set equality — a prefix form is semantically wider than its AND even over identical terms** (suppress only when the fallback expression is empty).
- Sweep: `CONSTRUCTION_STRATEGIES` (4 rows) `fusion_sweep.py:396`; `_validate_constructions` enforces membership in the engine vocabulary; `SHIPPED_CONTROL_CENSUS = 20` (control row runs `"and"` explicitly — stays); `prefix_probe_expression` ~L1178 (the probe form Task 2's real construction supersedes); negative-composition counter keys on the or-form — extends to any non-primary form.
- Twin: `pipeline_builder_simple.py:370-371` uses `interleave_rankings` on its own legs — Task 1 verifies mixed-form reachability; exemption comment or finding (no speculative refactor; TASK-3501 owns unification).
- 15400's census machinery: `keyword_leg_census` (53 non-negative queries, doc-level top-10, unscoped `_keyword_search`); rescue counting from query IDs.

---

### Task 1: Venv + backlog + Part A — the tiered merge

**Files:**
- Modify: `tldw_chatbook/RAG_Search/simplified/rag_service.py` (the gather site + a `_fts5_primary_form()` helper + the fallback wrapper's docstring), `tldw_chatbook/RAG_Search/pipeline_builder_simple.py` (exemption comment IF verification supports it)
- Test: `Tests/RAG_Search/test_keyword_leg_tiered_merge.py` (create)

**Interfaces (produces):**
- `RAGService._fts5_primary_form() -> str` — the active construction's primary form (`"and"` for and/and_stopword_trim/and_then_or; `"or"` for or; Task 2 extends for prefix forms). One definition; the partition and the negative counter both consume it.
- Gather-site rule, written down in a comment citing the incident (the displaced vector-blind fixture; the 3/7 scoped decomposition): `tier1 = [r for r in rankings if r[0].get("fts_match", primary) == primary]`, `tier2 = the rest`; `interleave(tier1) + interleave(tier2)`, then `[:top_k]`. Tier 2 only ever fills; cross-tier dedup structurally vacuous (sub-legs disjoint by source_type) — one comment, no machinery.

- [ ] **Step 1:** Build the venv (recipe above); assert import provenance (paste into report). `backlog task edit 15700 -s "In Progress"` + `--plan` (spec+plan paths).
- [ ] **Step 2:** READ the gather site + row shape at gather time (confirm the form key's location) + the twin's leg sources (can its lists carry mixed forms? its legs come from pipeline functions — trace whether any runs a fallback construction).
- [ ] **Step 3:** RED-first (`test_keyword_leg_tiered_merge.py`, real in-memory DBs per the chacha/prompts fixture pattern): (a) THE AC#2 PIN — under `and_then_or`, one sub-leg primary rank-1 row + another sub-leg many fallback rows → the primary row leads the merged output (RED on today's round-robin); (b) all-primary byte-identity — under `and_stopword_trim` and under `or`, merged order identical to `interleave_rankings` unpartitioned (pin by comparing against a direct call); (c) rank-fair-between-primaries kept (two primary sub-legs, many-vs-one — order unchanged from today); (d) tier-2-fills-never-displaces (tier 1 holds ≥ top_k → zero tier-2 rows); (e) no-tier-2-without-fallback (structural: all-primary constructions never produce a tier-2 entry).
- [ ] **Step 4:** Implement (partition + two interleaves + concat + truncate; the helper; the comments; the wrapper docstring update). GREEN; counts.
- [ ] **Step 5:** Mutations: tiering removed (single interleave restored) → pin (a) reds; tier order inverted → pins (b)+(d) red.
- [ ] **Step 6:** THE INTERMEDIATE GATE (gated, worktree venv): full run must read **105 metrics at (+0.000)** and the construction matrix's **control census 20** — either moving is a STOP (refutes byte-identity; report, do not reconcile).
- [ ] **Step 7:** Battery: new file + `test_fts5_match_construction.py` (30) + escaping (37) + prompts + pushdown + allowlist + inventory (21) + Tests/RAG_Eval ungated; counts. Commit `feat(rag): form-tiered sub-leg merge — fallback rows fill, never displace (TASK-15700 Part A)` + trailer. Push.

---

### Task 2: The prefix constructions + sweep rows

**Files:**
- Modify: `rag_service.py` (constants + `_fts5_match_expressions` + `_fts5_primary_form` + the prefix expression builder), `Tests/RAG_Search/test_fts5_match_construction.py` + `test_fts5_query_escaping.py` (extend), `Tests/RAG_Eval/harness/fusion_sweep.py` (2 new rows + negative-counter extension), `Tests/RAG_Eval/test_fusion_decision_rule.py` (pins)

**Interfaces (produces):**
- `FTS_MATCH_CONSTRUCTION_PREFIX = "prefix"` → expressions (content-token prefix OR-joined? NO — read 15400's probe: per-token `"tok"*` joined how? The probe joined with a space = implicit AND of prefixes... VERIFY the probe's join and reproduce ITS semantics — the 3-rescue lead was measured on that exact form; changing the join invalidates the lead), `""` when trimming empties. `FTS_MATCH_CONSTRUCTION_AND_THEN_PREFIX = "and_then_prefix"` → (AND, prefix-form or None; NEVER term-set-suppressed — semantically wider always).
- Form stamp `FTS_MATCH_PREFIX = "prefix"` on prefix-form rows; `_fts5_primary_form` returns `"prefix"` for prefix, `"and"` for and_then_prefix.
- Cache key: the two new values enter automatically (value-keyed; verify with the distinctness test pattern).
- `CONSTRUCTION_STRATEGIES` grows 4 → 6 (`"prefix"`, `"and_pfx"`, shipped fusion params, names ≤10 chars); `_validate_constructions` passes (vocabulary extended); the negative counter counts forms != the row's construction primary (or + prefix).
- Escaping suite: prefix-form injection cases (star placement; a token that is literally `*`; quoted operators through the prefix and fallback paths).

- [ ] **Steps:** READ `prefix_probe_expression`'s exact join FIRST (the lead's provenance). RED-first per construction (expression shapes incl. empty-after-trim; suppression NEVER fires for and_then_prefix — pin with an identical-term-set case; stamps; cache-key distinctness; escaping). Implement; mutations (suppression wrongly added → its pin reds; prefix star inside quotes → escaping reds). Battery + ungated Tests/RAG_Eval (counts; sweep pins updated for 6 rows as disclosed changes). Commit `feat(rag): prefix + and_then_prefix constructions — pre-registered rows for the 15700 re-run` + trailer. Push.

---

### Task 3: The sweep run + mechanical decision

**Files:** report only; backlog note on 15700.

- [ ] **Steps:** Gated, worktree venv (import provenance pasted first): the SIX-row construction matrix + the rule applied VERBATIM in writing (constraints (a) vector-blind hybrid rescue, (b) no cell regresses > 0.02, (c) escaping green; winner = max census subject to constraints; tie-breaks fewest-extra-queries → smallest-delta; negative-composition recorded — still corpus-vacuous, say so). Capture verbatim: per-row census/resc/zero-row, per-category hybrid cells, the vector-blind fixture's per-row provenance (fts_rank/vector_rank/slot), scoped recall per row, prompt cells, wall-time. EXPECTATIONS (the table decides): and 20/control, and_trim 21 (shipped), or ~28 still JOIN-fails, and_then_or ~29 NOW-QUALIFIES?, prefix ?, and_pfx ~23-24. If and_then_or's scoped holds 1.000 and the fixture keeps its rescue → likely winner at 29. AC#4 answered from the winner's fused table: the fixture's slot, gap to first excluded row, named next displacer. A null re-run (nothing qualifies) → record mechanisms per AC#3, `and_stopword_trim` stays, Task 4 becomes a no-op recording. Report + ledger + `backlog task edit 15700 --notes` one-liner (commit "docs(backlog): 15700 sweep outcome" + trailer). Push.

---

### Task 4: Ship the winner (or record the null)

**Files:** `config.py` (default flip IF the winner differs), `rag_service.py` docstring resolution, disclosed test flips, gated capture.

- [ ] **Steps:** If the winner ≠ `and_stopword_trim`: flip the default; enumerate EVERY default-pinning test that flips (battery at old default first — the 15400 discipline: every red enumerated, each flipped assertion comments both states + sweep row + date); the gated prompt/census pins flip disclosed; docstring states the outcome from the TABLE; mutation (default reverted → flips red); gated capture: the winner's predicted movers ONLY (an unpredicted mover = STOP). If null: update the docstring/README prose to record the re-run's outcome (the merge is fixed, the constructions still fail, mechanisms named) — no default change, no cell movement. Battery; counts. Commit + trailer. Push.

---

### Task 5: Re-stamp + closure + live check

**Files:** `Tests/RAG_Eval/baselines/*.json` (IF cells moved), `README.md` (census, headroom table, the AC#4 margin statement beside the 5/7-rank-9 note), `Docs/User_Guide/library/search-and-rag.md` (if user-visible behavior changed + stamp after live), backlog (15700 Done all 7 ACs; the 15400 backfill sites already carry 15700's id — verify), lessons if earned.

- [ ] **Steps:** ONE re-stamp IF Task 4 moved cells (MAIN venv + PYTHONPATH forced here + import provenance asserted in-run — the fingerprint-match method; reconcile cell-by-cell against Task 3's winner row; fresh gated run PASSED). AC#7: the residual zero-row census re-measured per category + the leg census, published in README + the task's Implementation Notes. AC#4's margin paragraph in the README. LIVE CHECK (lessons-live-verification.md; scratch profile; budget for the known RAG-Answer 4-min hazard; honest bounds if it recurs): the query class the winner unlocks through Library RAG Answer on hybrid. Closing battery + collection sweep vs merge-base `61f6ae575`; 15700 Done with the table + margin + residual verbatim; lessons (candidate: the intermediate-gate pattern — "a behavior-neutral refactor pre-registers its own zero-movement proof"; check for near-duplicates first). Commits + trailer. Push.

---

## Self-review (done at plan time)

- **Spec/AC coverage:** AC#1+#2 → T1; AC#3 → T3(+T2 rows); AC#4 → T3 answer + T5 README; AC#5 → T2 (real construction) + T3 (both-level measurement); AC#6 → T5 (conditional re-stamp, fingerprint method); AC#7 → T5. Intermediate gate → T1 Step 6. Twin exemption → T1 Step 2. Null path routed (T3 → T4 no-op → T5 without re-stamp).
- **Placeholder scan:** clean; the winner is computed (T3), the arc's point.
- **Type consistency:** `_fts5_primary_form` (T1) consumed by T1's partition + T2's counter; `FTS_MATCH_PREFIX`/construction names (T2) = T3's row names (`prefix`, `and_pfx`); `CONSTRUCTION_STRATEGIES` 6 rows (T2) = T3's matrix.
- **Open risk, named:** the prefix probe's exact join semantics (T2 Step 1 reads it FIRST — the 3-rescue lead is only valid for the probed form).

---

**Outcome addendum (2026-08-13, post-execution):** the sweep ran as planned;
the rule's tie-break selected `prefix` (240 vs 460 statements over the
measurement-identical qualifiers). **The owner ruled `and_then_prefix` ships**
(standing stability-over-quick-wins ruling, applied to the structural
self-displacement dimension the tie-break predates) — disclosed at every
outcome-recording site. The shipped default is the ruling's choice, not the
tie-break's.
