# P2c PRF Fail-First — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Probe PRF's premise on the 22 plain-failing golden queries under pre-registered admission criteria — build the plain-profile feature ONLY if the probe admits it, and record the null beside the three retired P2c premises if it doesn't.

**Architecture:** Phase A is harness-only (term derivation + expression composition as pure functions with always-on tests; the gated probe composing at the DB-level call sites the four-seam service itself uses). Phase B is CONDITIONAL and its tasks execute only on an ADMIT verdict. The spec is authority: `Docs/superpowers/specs/2026-08-13-rag-p2c-prf-fail-first-design.md` — its Step-0 fireability census, the ONE licensed variant, the admission bar, and the null path bind every task. The verdict is COMPUTED against the bar, never argued.

**Tech Stack:** Python 3.11+, SQLite FTS5, the P2ab eval harness (`RAG_EVAL=1`), pytest.

## Global Constraints

- Spec first. Worktree `.worktrees/rag-p2c-prf`, branch `feat/rag-p2c-prf-probe` (off dev `cb89f3ff6`). **cwd resets every Bash block — `cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/rag-p2c-prf` first, EVERY block.**
- **VENV: fresh worktree, NO local venv.** Task 1 builds it with the proven recipe: `uv venv .venv --python 3.12 && VIRTUAL_ENV=.venv uv pip install -e ".[dev,embeddings_rag]" "transformers==5.6.2" "torch==2.11.0" "chromadb==1.5.8"` (bare `uv venv` resolves py3.14 + drifted fingerprint packages — a false ENVIRONMENT_CHANGED hazard, twice proven). Assert `import tldw_chatbook; print(tldw_chatbook.__file__)` resolves IN the worktree before ANY measurement; paste the line.
- pytest via the worktree venv; counts READ; never `git stash`; Edit restores; single foreground Bash (timeout 600000). TCC "Operation not permitted" = transient: stop, report.
- Fixtures + baselines FROZEN through Phase A (the probe is read-only over the instrument). Phase B (if admitted) owns the ONE re-stamp.
- Injection quoting load-bearing in probes too: every composed expression keeps per-token quoting; probe expressions run through the same quoting helpers.
- Mechanism prose is an ORACLE (four incidents across two arcs): every mechanism sentence in reports/README states what a table/metadata showed. Gains AND losses by query id (the lost-column discipline).
- Commits reference the arc task (filed in Task 1), end `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; push after every task.

## Verified code anchors (line numbers drift; grep first)

- Four-seam call shape: `library_local_rag_search_service.py:548-557` — `fts_query = build_fts_match_query(query)` then `db.search_conversations_by_content(fts_query, top_k)` (and siblings per source type). THE PROBE HANDS ITS OWN EXPRESSIONS AT THESE DB-LEVEL CALL SITES — the product's SQL, not an approximation.
- `build_fts_match_query` (`Library/library_fts_query.py:78`): AND of plural/singular OR-groups — AND-strict across terms (why Step 0 exists).
- Row-content fact: four-seam media/conversation rows carry label snippets only ("Matched media · {type}"); notes carry `content`, prompts carry `user_prompt`/`details` (`library_local_rag_search_service.py:~1086-1135`). Term derivation FETCHES content for fed rows: `MediaDatabase.get_media_by_id` (~L6158), the chacha note/message read APIs (grep `get_note_by_id`/message fetch in ChaChaNotes_DB.py), prompts via its ORM. One read per fed row; count and report.
- Stopwords: `rag_service._FTS5_STOPWORDS` (67, pinned) — imported by the probe's term derivation.
- Harness idioms: `Tests/RAG_Eval/harness/fixture_probe.py` (per-query three-mode probe), `fusion_sweep.py` (grid + report + the lost-column/`lost_census_queries` pattern), `runner.py`/`goldenset.py` (golden loading, category filters), `harness_gate()` pytestmark for gated files.
- Golden populations (derive at probe time, never hardcode): 22 = paraphrase (13) + vocabulary_mismatch (9); guards = every plain query whose baseline pass hits (keyword's actual hitters — plain keyword recall 0.844 — plus scoped 7/7); negation 3; negatives 7.
- ID SAFETY for the arc task: FIVE collisions this programme (14913/15021/15401/15503/15703); last known high-water ~15810. Sweep ALL worktrees + origin/dev, leapfrog +100, create→mv→patch-frontmatter, verify with `backlog task <id> --plain`.

---

### Task 1: Venv + backlog + the probe machinery (pure functions + composition)

**Files:**
- Create: `Tests/RAG_Eval/harness/prf_probe.py`, `Tests/RAG_Eval/test_prf_probe.py` (always-on pure tests + the gated entry point)
- Modify: none in production (Phase A is harness-only).

**Interfaces (produces — Task 2 relies on these exact names):**
- `derive_expansion_terms(docs: Sequence[str], *, query_terms: Collection[str], n_terms: int, stopwords: Collection[str]) -> tuple[str, ...]` — TF over whitespace-alphanumeric runs, lowercased; excludes stopwords and query terms; deterministic tie-break (count desc, then alphabetical); returns ≤ n_terms.
- `compose_prf_expression(query: str, expansion_terms: Sequence[str]) -> str` — the query's content terms OR-extended with the expansion terms, every token individually quoted (reuse the engine's quoting helper via import, do not re-implement); `""` when nothing survives.
- `compose_feedback_expression(query: str) -> str` — the OR-of-content-terms first-pass form for the licensed variant (Step 0's fallback feedback pass).
- `ProbeQueryResult` dataclass: query_id, category, fireable (bool), first_pass_rows (int), fed_docs (int), content_fetches (int), target_rank_before (int|None), target_rank_after (int|None), rows_after (int) — the per-query table row Task 2 prints verbatim.
- The gated entry point runs NOTHING yet (Task 2 wires the passes); Task 1 ships machinery + tests only.

- [ ] **Step 1:** Build the venv (the pinned recipe); assert import provenance; paste. FILE THE ARC TASK (`backlog task create "P2c PRF fail-first probe" ...` — ID safety recipe above; ACs from the spec: fireability census run first; the bar applied mechanically; null recorded beside the retired premises OR Phase B built as scoped; guards derived at probe time; gains-and-losses by id). `-s "In Progress"` + `--plan`.
- [ ] **Step 2:** RED-first the pure functions: term derivation (stopword/query-term exclusion; N-cut; determinism incl. the tie-break; empty docs → ()); composition (quoting preserved — run a hostile token through and assert inertness against a scratch FTS5 table; OR shape; "" contract); the feedback form. Implement; GREEN; counts.
- [ ] **Step 3:** Mutations: stopword exclusion dropped → its test reds; quoting dropped → the hostile-token test reds.
- [ ] **Step 4:** Battery: the new file + Tests/RAG_Eval ungated (counts unchanged elsewhere). Commit `feat(rag-eval): PRF probe machinery — term derivation + composition, pure and pinned` + refs + trailer. Push.

---

### Task 2: THE PROBE RUN + the mechanical verdict

**Files:** `Tests/RAG_Eval/test_prf_probe_run.py` (the gated run — a NEW module), report + ledger + backlog note only.

> **Amended at Task 1 review (2026-08-13).** This line named
> `test_prf_probe.py`, which is now the always-on pure-function pin file (26+
> tests, no gate). Every gated module in that directory applies
> `pytestmark = harness_gate()` at module level, so a Task-2 implementer
> following BOTH the plan and the directory idiom would have silently gated
> the pins — the exact trap `harness/environment.py:harness_gate`'s own
> docstring names, and the reason it is per-module rather than directory-wide.
> The gated run goes in its own module.

- [ ] **Step 1 (STEP 0 FIRST — one command):** the fireability census over the 22: shipped first pass (the four-seam MATCH via `build_fts_match_query`, handed at the DB-level call sites), rows-returned per query. If fireability ≥ 5/22 → the base grid runs on the shipped first pass. If < 5/22 → the licensed variant (feedback pass = `compose_feedback_expression`) is activated, DISCLOSED as such in every table.
- [ ] **Step 2:** The grid (base point N=8/M=5; the full {4,8,16}×{3,5,10} ONLY if the base point shows signal — every point run is recorded; a null at every point run is the null): per query — first pass → fetch content for top-M fed rows (count fetches) → derive terms → second pass with `compose_prf_expression` → target rank. Guards in the same run: the derived currently-hitting population (gains AND losses by id — the lost-column discipline), negation (rows + junk delta, reported), negatives (structural-vs-live per the spec's honesty note).
- [ ] **Step 3:** THE VERDICT, computed against the spec's bar in writing, line by line: ≥5/22 rescued AND zero hitters lost AND zero negative rows AND the negation guard's report. ADMIT → Tasks 3-4 execute. NULL → Task 3 becomes the null recording and Task 4 is skipped.
- [ ] **Step 4:** Report (the per-query table VERBATIM — fireability, per-point results, guards; the content-fetch price; wall-time). Ledger + `backlog task edit <arc-id> --notes` one-liner. Commit `docs(backlog): PRF probe outcome` + refs + trailer. Push. (The reviewer re-runs the probe.)

---

### Task 3: ADMIT → build Phase B | NULL → record it

**If ADMITTED** — Files: `tldw_chatbook/RAG_Search/simplified/config.py` or the profile system per spec verification item 3 (the plain-only flag, OFF by default everywhere else), `tldw_chatbook/Library/library_local_rag_search_service.py` (the second pass through the four-seam seams + the route-note disclosure), `library_rag_state.py` (disclosure copy), tests per the four-seam patterns (RED-first; the hybrid/semantic-profile off-pin with its mutation; the disclosure pinned; the price counted).
**If NULL** — Files: `Tests/RAG_Eval/README.md` (the null recorded beside expansion/acronym/compositional with the fireability table and the licensed-variant result), backlog (the arc task Done with the table; the NEXT candidate's task filed — clarification gate — with the probe-machinery pointer; ID safety), lessons only if the probe surfaced a generalising trap.

- [ ] **Steps (ADMIT):** READ the profile surface first (verification item 3); RED-first; implement; mutation (PRF firing on a hybrid profile → its pin reds); battery + counts; commit + push.
- [ ] **Steps (NULL):** README + backlog + the next-candidate filing; ungated battery (nothing moved — counts identical); commit `docs(rag-eval): the PRF null, recorded — the fourth retired P2c premise` + refs + trailer; push.

---

### Task 4 (ADMIT only): Re-stamp + closure + live

- [ ] **Steps:** ONE re-stamp (the fingerprint-matching MAIN venv + PYTHONPATH forced here + provenance asserted in-run): plain-PRF cells move per Task 2's table, hybrid/semantic at +0.000 (the zero-movement proof; any unpredicted mover = STOP); README (the admitted evidence + the price); live check per lessons-live-verification.md (a BM25 Only profile paraphrase query finding its target; scratch profile; the TASK-15810 UI hazard budgeted — the engine two-arm A/B is the sanctioned fallback, said in the stamp); User Guide stamp if claims changed; arc task Done all ACs; closing battery + collection sweep vs merge-base `cb89f3ff6`; commits + refs + trailer; push.

---

## Self-review (done at plan time)

- **Spec coverage:** Step-0 census + variant → T2 Step 1; machinery + quoting + determinism → T1; the bar + verdict → T2 Step 3; guards-derived-at-probe-time → T2 Step 2; null path → T3-NULL (Task 4 skipped); Phase B constraints (plain-only, disclosed, priced, one re-stamp, 3997 note) → T3-ADMIT + T4; content-fetch pricing → T1 dataclass + T2 report; grid discipline (every point recorded) → T2 Step 2.
- **Placeholder scan:** clean; the verdict is computed (T2), the arc's point; T3's branch is conditional by design, not a placeholder.
- **Type consistency:** `derive_expansion_terms`/`compose_prf_expression`/`compose_feedback_expression`/`ProbeQueryResult` (T1) consumed only by T2; the arc task id threads T1→T4.
- **Open risk, named:** the chacha content-read API names are grepped at T1 (the anchor names the family, not exact signatures — the implementer reads before wiring).
