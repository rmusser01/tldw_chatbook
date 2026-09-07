# P2ab: Instrument Renewal + P0-Deferred Constraints — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the eval instrument's power to measure improvement (fail-first fixtures + scope machinery + ~150-doc corpus), then land the three P0-deferred constraints measured by it: scope-aware hybrid (B1), a prompts keyword sub-leg (B2), and the Library window honoring the profile (B3) — one deliberate re-stamp at the end.

**Architecture:** Half A extends `Tests/RAG_Eval/` (goldenset schema, runner scope machinery, corpus) with zero production changes; Half B is engine/Library work in `rag_service.py` + `library_local_rag_search_service.py` + one Library-state constant seam, in B1→B2→B3 order. Spec is authority: `Docs/superpowers/specs/2026-08-11-rag-p2ab-instrument-and-deferred-constraints-design.md` — its fail-first admission protocol, fail-closed allowlist semantics, and sequencing bind every task.

**Tech Stack:** Python 3.11+, pytest, the P1 eval harness (`RAG_EVAL=1`), SQLite FTS5 (`prompts_fts` confirmed present), the private-sqlite seam.

## Global Constraints

- Spec (read FIRST — the admission protocol, Half B fix designs, and sequencing): `Docs/superpowers/specs/2026-08-11-rag-p2ab-instrument-and-deferred-constraints-design.md`.
- Worktree `.worktrees/rag-p2a-instrument`, branch `feat/rag-p2a-instrument-renewal`. **cwd silently resets between Bash blocks — start EVERY block with `cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/rag-p2a-instrument`.**
- Tests: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <paths>` with worktree cwd; pytest only; app-importing probes under Tests/ only; "no tests ran" = FAILED gate; read numeric counts; never pass a directory AND files inside it in one invocation (silent under-collection trap).
- Engine source-type vocabulary EXACT singular: `media`/`note`/`conversation`, B2 adds `prompt`. Fusion-key/canonicalization/post-filter vocabulary pins extend accordingly.
- Existing fixtures stay BYTE-IDENTICAL (additions only); every gated run between the corpus change and Task 9's re-stamp reads `environment_changed` — expected, recorded, never "fixed" by intermediate re-stamps.
- Protected oracles: all existing Tests/RAG_Eval always-on tests, the fusion/rescue/pushdown/chacha suites, and `Tests/DB/test_private_sqlite_inventory.py` (in EVERY task battery — B2 adds a connection owner). Legitimate oracle flips (the scoped-route pin, the prompts recall-0 pin, disclosure copy) are DISCLOSED updates at the moment their fix lands, never silent.
- Never `git stash`; Edit-based restores with unique markers; push after every task; commits reference the relevant TASK id (assigned in Task 1) and end `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

## Verified code anchors (line numbers drift; grep first)

- Allowlist shape: `Chat/rag_scope.py::build_semantic_allowlists(eff) -> Optional[list[dict[str, set]]]` (~L371) — one dict per surviving source type, `{"source_type": {type}, "source_id": ids}`; `None` = unscoped. `EffectiveScope` is the seam's `scope=` type; the Library seam accepts `scope: Optional[EffectiveScope]` on `search` (~L179/185) and threads it to `_search_rag` (~L349).
- Engine allowlist guard: `rag_service.py` ~L803-805 (`metadata_allowlist and search_type != "semantic"` → ValueError). B1 removes the HYBRID arm only; keyword-mode raising stays (spec non-goal).
- Keyword-leg orchestration (post-weighting): `_keyword_search(query, top_k, filter_metadata, include_citations, *, keyword_source_types=None)` gathers `_media_keyword_subleg` + `_chacha_keyword_sublegs`, `interleave_rankings(...)[:top_k]`. Chacha sub-legs use `connect_private_sqlite("rag.chachanotes_keyword_leg", ..., read_only=True)`; the inventory suite asserts an EXACT census (B2 = new owner row + ratchet bump, the 3996 recipe).
- Notes id-filter precedent for B1's SQL: `ChaChaNotes_DB.search_notes` uses `AND main.id IN (SELECT value FROM json_each(?))` — mirror that form for every sub-leg allowlist filter (never unbounded IN lists).
- Prompts: `prompts_fts(name, author, details, system_prompt, user_prompt)` FTS5, rowid = prompt id (Prompts_DB.py ~L270); soft-delete via `deleted = 0` on `Prompts` (read the ORM's own prompt search for the exact query shape before mirroring); path resolver `config.py::get_prompts_db_path()` (~L5931, default `~/.local/share/tldw_cli/tldw_cli_prompts.db`); config injection mirrors `media_db_path`/`chachanotes_db_path` (`config.search.prompts_db_path`, path-validated, degrade-not-raise).
- Library window: `library_rag_state.py` `LIBRARY_RAG_DEFAULT_TOP_K = 5` (~L56), consumed via `_coerce_positive_int(top_k, LIBRARY_RAG_DEFAULT_TOP_K)` (~L1174), `LIBRARY_RAG_TOP_K_MAX = 50`. Console precedent: `_console_library_rag_profile_top_k()` in chat_screen.py (profile `default_top_k`, fallback 5) — a UI-screen seam, likely NOT import-reachable from Library state without a cycle → expect a twin + coupling test.
- Harness: `Tests/RAG_Eval/harness/` — goldenset.py (frozen dataclasses, validator collecting ALL defects), runner.py (mode flips in try/finally, per-category metrics, route-telemetry via `runtime_backends`), ingest.py (`build_eval_runtime`, `slug_to_source`, writers via real APIs), fixture files corpus.toml/golden.toml. Gated via `harness_gate()` pytestmark; `RAG_EVAL_UPDATE_BASELINES=1` re-stamps with printed deltas.
- Cache-key parts already present: `keyword_source_types` + hybrid fusion params + `metadata_allowlist`. Verification item 8: confirm hybrid+allowlist composes (the allowlist part predates hybrid reaching it — read `_make_key`).

---

### Task 1: Backlog filing

**Files:** new task file via CLI; TASK-14752 edited.

- [ ] **Step 1:** ID scan across ALL worktrees + origin/dev (CLI auto-assigns from local max — unsafe; recent maxima ~14752 here; leapfrog max+100, create→mv→patch-frontmatter recipe if needed). Create the P2ab task (`-s "In Progress"`, `--plan` referencing spec+plan) with one `--ac` per spec outcome: fail-first categories admitted by measured failure (or recorded unfailable); scoped category with routing before-pin; ~150-doc corpus with old-queries-unchanged evidence; scope-aware hybrid retiring the scoped→semantic disclosure family; prompts sub-leg (read-only, inventoried, vocabulary-pinned); Library default honors profile (user control untouched); ONE final re-stamp with per-category headroom table; live checks for the three user-visible changes.
- [ ] **Step 2:** `backlog task edit 14752 -s "In Progress"` + a plan line noting it lands inside B1 (its coverage-copy seams are B1's).
- [ ] **Step 3:** Commit `chore(backlog): file P2ab arc task; 14752 rides B1` (+ trailer), push.

---

### Task 2: Harness scope machinery (Half A infrastructure)

**Files:**
- Modify: `Tests/RAG_Eval/harness/goldenset.py` (schema + validator), `Tests/RAG_Eval/harness/runner.py` (scoped execution + telemetry), `Tests/RAG_Eval/harness/ingest.py` (if scope construction needs runtime ids)
- Test: `Tests/RAG_Eval/test_goldenset_integrity.py` (extend, always-on), `Tests/RAG_Eval/test_harness_scoped.py` (create, gated)

**Interfaces:**
- Produces: `GoldenQuery.scope_slugs: tuple[str, ...] | None` (loader tolerant, validator strict: only category `"scoped"` may carry it, non-empty when present, every slug exists); runner: scoped queries build a real `EffectiveScope` from `slug_to_source` ids (read rag_scope.py's constructor/tests for the canonical shape — state "scoped", allowlist per the real invariants) and pass `scope=` to the seam; per-scoped-query telemetry records the executed route (backend label + route notes). A gated BEFORE-PIN test: a scoped query today routes semantic (`rag-semantic` backend under the hybrid profile) — the disclosed-flip target for Task 6.
- The scoped category is EXCLUDED from cross-mode averages the way negatives are special-cased where routing makes modes incomparable — read how negatives are excluded and mirror the mechanism; scoped cells report under their own category.

- [ ] **Step 1:** READ rag_scope.py (EffectiveScope invariants, build_semantic_allowlists), the Library seam's scoped tests (canonical test-side EffectiveScope construction), runner.py's negative-category special-casing.
- [ ] **Step 2:** RED always-on validator tests (scope_slugs on wrong category; unknown slug; empty list; happy path). RED gated before-pin. Implement. GREEN; counts.
- [ ] **Step 3:** Mutation: validator rule dropped → its test reds. Commit `feat(rag-eval): golden-set scope schema + scoped runner machinery with routing before-pin` (+ TASK ref + trailer), push.

---

### Task 3: Fail-first fixture authoring + corpus scale-up (Half A core)

**Files:**
- Modify: `Tests/RAG_Eval/fixtures/corpus.toml`, `Tests/RAG_Eval/fixtures/golden.toml`, `Tests/RAG_Eval/test_goldenset_integrity.py` (quotas/guards)
- Create: `Tests/RAG_Eval/harness/fixture_probe.py` (the authoring helper: one query → three modes → ranks/scores printout; gated; rides existing machinery)

**Interfaces:**
- Produces: ~150-doc corpus (existing 49 byte-identical + additions incl. prompt docs); new golden categories `compositional`, `negation`, `acronym`, `scoped`, optionally `precision_pressure`; every ADMITTED fixture carries the `# admitted: <date> hybrid=<rank|miss> semantic=<rank|miss> plain=<rank>` comment; class-level outcomes (admitted N / unfailable) recorded for the README (Task 9 writes it). Scoped: ≥6 queries whose targets are keyword-findable but vector-poor in scope. Prompt docs + their golden queries (structural admission — recall 0 in all modes, pinned gated as the B2 before-pin). Quotas set from probe outcomes (hard floor only for scoped ≥6 and prompts ≥4).
- THE ADMISSION PROTOCOL IS THE TASK: author candidates → probe → keep only measured failures (target misses top-10 in hybrid AND semantic; keyword rank recorded). An unfailable class is a RECORDED outcome, not a failure — do NOT force-fit fixtures to fail (no nonsense text; the always-on authoring guards must pass and the docs must read as plausible corpus content).
- Old-queries-unchanged probe: every pre-existing golden query's top-10 (per mode) is IDENTICAL before/after the corpus addition, or the new doc is reworded — run it as part of this task (gated), keep the check as a test if cheap, else record the run's evidence.

- [ ] **Step 1:** Build fixture_probe.py first (RED-first for its pure formatting; gated smoke). **Step 2:** Author + probe class by class; record admissions. **Step 3:** Extend integrity guards (new categories in quotas; overlap/timeless/will-free guards over new docs; the composition test's arithmetic updated with reason comments). **Step 4:** Old-queries-unchanged evidence. **Step 5:** Gated full run — expect `environment_changed` (corpus hash moved) + the new categories' failing numbers recorded informationally. **Step 6:** Commit `feat(rag-eval): fail-first fixture classes + scoped/prompt categories + corpus scale-up` (+ refs + trailer), push. Report the per-class admission table VERBATIM.

---

### Task 4: B1a — engine: allowlists reach the FTS legs

**Files:**
- Modify: `tldw_chatbook/RAG_Search/simplified/rag_service.py` (the guard ~L803; `_hybrid_search` threading; each sub-leg's SQL gains the json_each id filter; cache-key composition verified)
- Test: `Tests/RAG_Search/test_hybrid_allowlist_pushdown.py` (create)

**Interfaces:**
- Produces: `search(search_type="hybrid", metadata_allowlist=...)` no longer raises — the allowlist (the per-source-type LIST shape from build_semantic_allowlists' consumers; read how the Library passes it for semantic today — match that calling convention exactly) threads to BOTH legs: semantic leg exactly as today, keyword leg via per-sub-leg id filters (`AND <id_col> IN (SELECT value FROM json_each(?))`, ids JSON-encoded); a sub-leg with NO entry for its source type is SKIPPED (fail-closed); a sub-leg with an entry runs filtered. Empty-for-every-sub-leg → keyword leg [] → existing degrade path. Keyword-mode + allowlist STILL raises (pin it).
- Cache key: verify `_make_key`'s allowlist part composes with fusion params + selection for hybrid (test: same query, different allowlists → different keys; same allowlist in different entry order → same key — read how the allowlist is canonicalized today).

- [ ] **Step 1:** READ the guard, the semantic leg's allowlist path, `_make_key`'s allowlist canonicalization, the notes json_each precedent. **Step 2:** RED tests: per-sub-leg filtering (real DBs: in-scope keyword doc found, out-of-scope keyword doc absent, per source type); fail-closed skip (scope names only notes → media/conversation sub-legs never queried — spies); keyword-mode still raises; cache-key composition; large-allowlist discipline (500 ids → parameter count stays bounded — assert the json_each form via SQL capture or a 1000+-id smoke). **Step 3:** Implement → GREEN; mutation (fail-closed flipped to unfiltered → its test reds). **Step 4:** Battery + inventory suite; counts. Commit `feat(rag): hybrid allowlists reach the FTS legs — fail-closed per-sub-leg id filters (B1a)` (+ refs + trailer), push.

---

### Task 5: B1b — Library: scoped hybrid routes hybrid; disclosure family retires; 14752

**Files:**
- Modify: `tldw_chatbook/Library/library_local_rag_search_service.py` (`_search_rag`'s scoped arm calls `_search_hybrid` with the allowlist; scoped-semantic fallback only when hybrid unavailable), `library_rag_state.py` (route-note vocabulary: `ROUTE_NOTE_HYBRID_SCOPED` retires; the 14752 coverage-copy fix: the note distinguishes "no semantic hits" from "no evidence at all" when the keyword leg supplied rows), `Docs/User_Guide/library/search-and-rag.md` (scoped behavior + stamp deferred to Task 8's live check)
- Test: `Tests/Library/test_library_rag_mode_resolution.py` (extend — the scoped-arm tests flip as DISCLOSED oracle updates), `Tests/Library/test_library_rag_state.py` (coverage-copy)

**Interfaces:**
- Consumes: Task 4's engine capability. Produces: scoped + hybrid profile → fused hybrid path with allowlists (backend `rag-hybrid`); the retired note's literal text grepped repo-wide (the stale-prose lesson — grep the COPY, and the VALUES); 14752's AC satisfied: coverage copy over keyword-sourced evidence no longer claims "found nothing" ambiguously (read 14752's AC for the exact contract).
- [ ] **Steps:** READ the scoped arm + the P0-era comments naming the constraint (they retire too — grep "scope-aware hybrid" / "semantic only"); RED-first the new routing tests + coverage-copy tests; implement; disclosed flips enumerated in the report; mutation (scoped arm reverted → routing test reds); battery incl. Task 4's file + mode-resolution + rag-state + gate16 file; commit `feat(library): scoped hybrid runs fused hybrid; scoped-semantic disclosures retire (B1b + TASK-14752)` (+ refs + trailer), push. Tick 14752 Done.

---

### Task 6: Harness scoped-category flip (the B1 measurement)

**Files:** `Tests/RAG_Eval/test_harness_scoped.py` (the before-pin flips — disclosed), report.

- [ ] **Steps:** Gated run: scoped queries now route hybrid (telemetry pin flips as a DISCLOSED update: before-pin asserted `rag-semantic`, now asserts `rag-hybrid` — keep a comment naming both states + dates); scoped category scores recorded (targets rescued — verify the expected mechanism: in-scope keyword rank-1, FTS-only, fused in by the k=5 weighting); paste the before/after scoped table into the report. NO re-stamp (Task 9). Commit `test(rag-eval): scoped category measures B1 — routing flip + rescue evidence` (+ refs + trailer), push.

---

### Task 7: B2 — prompts keyword sub-leg

**Files:**
- Modify: `tldw_chatbook/RAG_Search/simplified/rag_service.py` (fourth sub-leg `_prompts_keyword_subleg` + `_prompts_fts` raw helper; `config.search.prompts_db_path`; `_fusion_doc_key`/`KEYWORD_LEG_SOURCE_TYPES` gain `prompt`), `tldw_chatbook/DB/private_sqlite.py`-adjacent owner registry (new owner `rag.prompts_keyword_leg`, read-only URI), `Tests/DB/test_private_sqlite_inventory.py` (row + ratchet — the 3996 recipe), `tldw_chatbook/Library/library_local_rag_search_service.py` (+`library_rag_score_kinds`/canonicalize maps as needed: `prompt` vocabulary through post-filter + `_FTS_SERVABLE_SOURCE_TYPES` + the narrowed no-keyword-sources note), `Tests/RAG_Eval/harness/ingest.py` (prompts writer via the real Prompts_DB API + `config.search.prompts_db_path` injection), `Tests/RAG_Eval/harness/canonicalize.py` (prompt provenance)
- Test: `Tests/RAG_Search/test_keyword_leg_prompts.py` (create — the chacha suite is the template: read-only pin, soft-delete pin w/ FTS-rebuild discipline, symlink/parent-dir refusal via the seam, degrade-on-missing, cross-vocabulary pins, interleave), plus the gated prompts before-pin FLIP (recall 0 → scored; disclosed)

**Interfaces:**
- Produces: hybrid (and the four-seam? NO — engine leg only; the Library four-seam path already searches prompts its own way — do not touch it) returns prompt rows stamped `source_type: "prompt"`, source_id (str prompt id), title (prompt name), rescued via the weighting (every prompt row is FTS-only — pin one end-to-end: a prompt doc enters hybrid top-k). Soft-delete: mirror the ORM's `deleted = 0` + rebuild-before-search test discipline (the vacuous-guard lesson). B1's allowlist filtering extends to the prompts sub-leg IF scope can name prompts (read the scope vocabulary — if scope never carries prompt ids, the sub-leg is skipped under any scope: fail-closed default, pinned).
- [ ] **Steps:** READ Prompts_DB's own search SQL + the chacha suite; RED-first the full pin set; implement; inventory + ratchet; harness writer + canonicalization; the gated recall-0 pin flips DISCLOSED; mutation (deleted-filter dropped → its test reds post-rebuild); battery incl. inventory suite; commit `feat(rag): prompts keyword sub-leg — read-only, inventoried, rescue-reachable (B2)` (+ refs + trailer), push.

---

### Task 8: B3 — Library window honors the profile + live checks

**Files:**
- Modify: `tldw_chatbook/Library/library_rag_state.py` (the default seam: profile `default_top_k` resolution, fallback 5; user control untouched), `tldw_chatbook/UI/Screens/library_screen.py` if the control's placeholder/initial value surfaces the default (read it), `Docs/User_Guide/library/search-and-rag.md` (+ stamp after the live check)
- Test: `Tests/Library/test_library_rag_state.py` (extend: unset→profile 15; explicit user value wins; invalid→profile; profile-unresolvable→5; coupling test with the Console seam's semantics — a twin is acceptable, a silent divergence is not)

- [ ] **Steps:** READ the Console precedent + all `LIBRARY_RAG_DEFAULT_TOP_K` consumers; RED-first; implement; mutation (resolution dropped → its test reds). Then THE LIVE CHECK (PR-2 recipe, tmux -L p2ab, scratch profile, copied real DBs + chromadb): (1) scoped hybrid search returns keyword-found in-scope evidence (B1 live); (2) a prompt hit surfaces in hybrid results (B2 live); (3) the Library list at depth 15 remains usable (B3 live; capture scroll behavior). Capture evidence; full teardown checklist (config hash, scratch deleted, kill-server, git status clean). Commit `feat(library): rag depth honors the active profile default (B3) + live evidence` (+ refs + trailer), push.

---

### Task 9: The re-stamp + closure

**Files:** `Tests/RAG_Eval/baselines/*.json`, `Tests/RAG_Eval/README.md`, backlog tasks, lessons if earned.

- [ ] **Steps:** ONE deliberate re-stamp (`RAG_EVAL_UPDATE_BASELINES=1`): full delta printout; the per-category HEADROOM TABLE replaces the at-ceiling warning (each failing category's numbers = P2c's admission targets; scoped + prompts categories' post-B numbers; the old categories unchanged — verify via the old-queries-unchanged evidence); gate PASSED after. README: admission-protocol documentation, per-class outcomes (incl. unfailable classes as P2c evidence), the retired Library-window bound. Backlog: P2ab task Done with the full evidence; ungated battery + collection arithmetic vs merge-base (baseline worktree, removed after); final `git status` clean; push.

---

## Self-review (done at plan time)

- **Spec coverage:** scope machinery → T2; fail-first + scale-up + prompt/scoped fixtures + old-queries probe → T3; B1 engine → T4; B1 Library + disclosure retirement + 14752 → T5; B1 measurement → T6; B2 → T7; B3 + live → T8; re-stamp + headroom + README → T9. Error-handling items land in their fix's task (fail-closed T4, prompts degrade T7, validator T2). Non-goals respected (keyword-mode guard stays, pinned T4; four-seam untouched T7; no semantic prompts indexing).
- **Placeholder scan:** contracts + read-first instructions per the house pattern; no TBDs.
- **Type consistency:** `scope_slugs` (T2→T3), `_prompts_keyword_subleg`/`rag.prompts_keyword_leg` (T7 only), `config.search.prompts_db_path` (T7), category names compositional/negation/acronym/scoped/precision_pressure (T2 validator ↔ T3 fixtures ↔ T9 README).
- **Sequencing:** A before B (T2-T3 before T4-T8); B1a→B1b→measurement; B2 after B1 (extends the scope-aware leg); B3 + live last; ONE re-stamp (T9). Matches the spec's load-bearing order.
