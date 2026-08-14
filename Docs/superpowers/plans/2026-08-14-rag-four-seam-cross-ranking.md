# TASK-16071: Rank-Fair Cross-Seam Merge — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the plain four-seam path's fixed-order concatenation with the engine's rank-fair `interleave_rankings`, measured by the gated instrument's pre-registered movers, with the PRF probe's oracle table re-run as a bound-recalibrating control.

**Architecture:** One merge-site change in `library_local_rag_search_service.py` (all-primary regime — no tiering; the rule + incident written at the site), always-on pins RED on today's concatenation, then the gated capture with pre-registered movers/zero-movement and the conditional re-stamp. Spec + TASK-16071's task file are joint authority: `Docs/superpowers/specs/2026-08-14-rag-four-seam-cross-ranking-design.md`, `backlog task 16071 --plain`.

**Tech Stack:** Python 3.11+, SQLite FTS5, the eval harness (`RAG_EVAL=1`), pytest.

## Global Constraints

- Spec + task file FIRST. Worktree `.worktrees/rag-16071-crossseam`, branch `fix/rag-four-seam-cross-ranking` (off dev `7b122c702`). **cwd resets every Bash block — `cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/rag-16071-crossseam` first, EVERY block.**
- **VENV (fresh worktree, none exists):** `uv venv .venv --python 3.12 && VIRTUAL_ENV=.venv uv pip install -e ".[dev,embeddings_rag]" "transformers==5.6.2" "torch==2.11.0" "chromadb==1.5.8"` — bare `uv venv` = false-ENVIRONMENT_CHANGED hazard, thrice proven. Assert `import tldw_chatbook; print(tldw_chatbook.__file__)` in-worktree before ANY measurement; paste the line.
- pytest via the worktree venv; counts READ; never `git stash`; Edit restores; single foreground Bash (timeout 600000). TCC "Operation not permitted" = transient: stop, report.
- Fixtures FROZEN; baselines move only via the ONE re-stamp (Task 2, in the fingerprint-matching MAIN venv + PYTHONPATH forced here + provenance asserted in-run), reconciled cell-by-cell against the pre-registered movers — an unpredicted mover is a STOP.
- Mechanism prose is an ORACLE (five incidents, three arcs): every mechanism sentence states what a table/metadata showed; a control that holds a second variable fixed measures the PAIR (the PRF lesson) — the oracle re-run states its selector.
- ID SAFETY: EIGHT collisions this programme; latest true max ~16072-region + whatever other sessions minted. Sweep ALL worktrees + origin/dev, leapfrog +100, create→mv→patch-frontmatter, verify.
- Commits reference TASK-16071, end `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; push after every task.

## Verified code anchors (line numbers drift; grep first)

- THE MERGE SITE: `library_local_rag_search_service.py:449-452` — `rows.extend(outcomes[source_type][1])` over `_KNOWN_KEYWORD_SOURCE_TYPES` (`:67`). Replace with `interleave_rankings([...per-seam lists...], key=...)` imported from `RAG_Search/fusion.py`; NO truncation added (the change is ORDER only; consumers cut).
- Row identity (verified): rows carry `source_id` + `provenance.source_type` (`_note_row`/`_media_row`/`_conversation_row`/`_prompt_row`, ~L1080-1136) — key = `(row["provenance"]["source_type"], row["source_id"])`.
- Callers (verified): `_search_keyword` consumed only in-service (`:268`, `:331`); outcome-dict consumers order-insensitive; DISPLAY consumption = Task 1's verification (the panel's "per source" is a heading suffix at `library_search_rag_panel.py:705` — establish whether rows render in list order; if a visible Search-mode regrouping results, it is a DISCLOSED change with the User Guide updated, not silently shipped).
- The two plain-keyword misses (0.875 = 14/16): Task 1 measures WHICH two and WHY (seam-burial vs genuine) BEFORE the fix lands — this decides the recall prediction Task 2 reconciles against.
- The PRF control: `Tests/RAG_Eval/test_prf_probe_run.py`'s oracle table (PRE-REGISTERED-selector guard already prevents verdict re-entry); re-run it under the new merge, report both selector rows before/after.
- Engine precedent for the pins: `Tests/RAG_Search/test_keyword_leg_tiered_merge.py` (the 15700 pin idioms — displacement, rank-fairness, byte-identity, mutations).

---

### Task 1: Venv + backlog + the fix + pins

**Files:** Modify `tldw_chatbook/Library/library_local_rag_search_service.py` (the merge site + the written rule); Create `Tests/Library/test_library_keyword_cross_seam.py`.

- [ ] **Step 1:** Venv (pinned recipe); provenance pasted. Backlog: `backlog task edit 16071 -s "In Progress"` + `--plan`; ALSO file the agentic-document-expansion roadmap task (the owner's capability question, 2026-08-14: chunk→document expansion tool + agent policy + P3-style evaluation — description cites RAGSearchTool, the doc_id linkage, sibling/parent-inclusion's P3 deferral; ID safety recipe; To Do; label rag,agents).
  > **AMENDED 2026-08-14 (Task 1 review).** Two errors in the line above, both
  > carried faithfully into the filing and both now fixed in TASK-16174. (1)
  > Sibling/parent inclusion is **P2 Retrieval intelligence**, not P3
  > (`Docs/superpowers/specs/2026-08-07-rag-port-p0-foundations-design.md:37`);
  > only the answer-level EVALUATION question is P3-grader territory, and this
  > line used "P3" for both. (2) It is not "deferred" — it is an **inert,
  > user-reachable surface**: `SearchConfig.include_parent_docs` /
  > `parent_size_threshold` / `parent_inclusion_strategy`
  > (`RAG_Search/simplified/config.py:559-561`) are switched on by three
  > shipped profiles (`config_profiles.py:310-312`, `:347-349`, `:524-526`) and
  > read by nothing in `tldw_chatbook/` (grep-verified). Filed as TASK-16174
  > AC#7: wire it or retire it, never leave it as a third overlapping surface.
- [ ] **Step 2:** MEASURE FIRST (gated): which two plain-keyword queries miss, and their mechanism (seam-burial vs genuine — run the four-seam pass per query, read where the targets sit). Record in the report; set the recall prediction for Task 2.
- [ ] **Step 3:** Display-order verification: how the panel consumes row order (grep the evidence-list rendering); state the Search-mode visual consequence; if visible, enumerate as a disclosed change.
- [ ] **Step 4:** RED-first (`test_library_keyword_cross_seam.py`, real DBs per the service's existing test patterns): (a) displacement pin — a media-seam rank-1 row precedes notes-seam rank-5 rows in the merged output (RED on today's concatenation); (b) rank-fairness (equal seams alternate); (c) single-seam byte-identity; (d) no-truncation contract (4×top_k rows in → all out); (e) prompts-seam participation (the buried fourth seam interleaves — the 16071 filing's point).
- [ ] **Step 5:** Implement (the interleave + the written rule citing the worked examples + the no-tiering comment with the 15700 pointer). GREEN; counts. Mutations: extend restored → pin (a) reds; seam order permuted inside a position → pin (b) catches or is documented as order-within-position semantics (state which).
- [ ] **Step 6:** Battery: new file + Tests/Library rag files + Tests/RAG_Eval ungated + inventory; counts. Commit `fix(library): rank-fair cross-seam merge — the four-seam path stops privileging seam order (TASK-16071)` + refs + trailer. Push.

### Task 2: The gated capture + control re-run + re-stamp

- [ ] **Steps:** Gated full run (worktree venv): reconcile EVERY moved cell against the pre-registered movers (plain keyword/scoped MRR/NDCG may shift; plain keyword recall per Task 1's prediction; paraphrase/vocab CANNOT move; hybrid/semantic MUST be +0.000) — unpredicted mover = STOP. The PRF oracle control re-run (both selector rows, before/after table; PRF stays retired — the bound recalibrates). IF cells moved: the ONE re-stamp (main venv method), fresh gated run PASSED. Report with tables verbatim. Commit + push.

### Task 3: Closure + live

- [ ] **Steps:** README (the 16071 fix recorded; the four-seam bound's recalibrated numbers replace the old oracle rows WITH selector attribution; the retired-premise sections' pointers updated if they cite the old bound). User Guide if display changed (+ stamp after live). LIVE CHECK (lessons-live-verification.md; scratch profile; TASK-15810 budgeted; engine A/B fallback sanctioned): a media/conversation-target query surfacing above notes seam-fill. 16071 Done all ACs; lessons only if earned; closing battery + collection sweep vs merge-base `7b122c702`; commits + refs + trailer; push.

## Self-review (plan time)

- Spec coverage: fix+rule → T1; movers/zero-movement/control/re-stamp → T2; README/UG/live/close → T3; the agentic roadmap filing → T1 Step 1; the misses-mechanism measurement BEFORE the fix → T1 Step 2 (ordering matters: the prediction must precede the change).
- Placeholders: none; the recall prediction is measured-then-registered by design.
- Types: the interleave key `(provenance.source_type, source_id)` consistent T1↔T2.
