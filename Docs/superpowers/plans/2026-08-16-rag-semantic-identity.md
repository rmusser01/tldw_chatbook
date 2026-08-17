# TASK-16588: Semantic-Route Identity — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Emit the `note_id`/`doc_id` identity fallbacks in the Library payload (AC#2), then measure `expand_document` on the semantic and hybrid routes against BOTH a canonical and a non-canonical index (AC#1-residue/#3), with byte cost and the gate re-proven (AC#4-#6).

**Architecture:** Two tasks. Task 1 is the code change + unit pins (Task 3b's exact shape, one file). Task 2 is the mechanical route probe (no LLM, no spend), run pre-fix and post-fix, plus closure. The pre-registered expectations differ BY INDEX KIND — a nonzero pre-fix `not_found` on the canonical index would be a NEW finding, not the expected one.

**Tech Stack:** Python 3.11+, pytest, chromadb (pinned), the eval-harness/QA seeding patterns.

**Spec:** `Docs/superpowers/specs/2026-08-16-rag-semantic-identity-design.md` (as corrected at `279619d86`) — binds every task.

## Global Constraints

- Worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/rag-16588-route`, branch `feat/rag-16588-semantic-identity` (off dev `bab84f7d9`). **cwd resets every Bash block — cd first, EVERY block.**
- **VENV (none exists):** `uv venv .venv --python 3.12 && VIRTUAL_ENV=.venv uv pip install -e ".[dev,embeddings_rag]" "transformers==5.6.2" "torch==2.11.0" "chromadb==1.5.8"`; assert `.venv/bin/python -c "import tldw_chatbook; print(tldw_chatbook.__file__)"` resolves IN the worktree; paste the line. ruff via the MAIN repo's `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff`.
- Never `git stash`; Edit restores; single foreground Bash (timeout 600000); do NOT run `Tests/UI/test_library_shell.py`. The app regenerates `css/tldw_cli_modular.tcss` on boot — restore before committing after any live run. TCC "Operation not permitted" = stop and report.
- Gate in BOTH tasks: `RAG_EVAL=1 .venv/bin/python -m pytest Tests/RAG_Eval/ -q -p no:randomly` reads verbatim `PASSED: No regression. 105 metric(s) within 0.05 of baseline` — a moved cell is STOP.
- Commits reference TASK-16588, end `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; push after every task.

## Verified code anchors (line numbers drift; grep first)

- `Agents/library_rag_tool_provider.py`: `_chunk_start` helper `:117-131`; `_project_row` attaches identity under the hint precondition around `:385-392` (`projected["chunk_start"]` at `:392`). The sealing loop bounds `serialized_size(payload)` to `MAX_RESULT_BYTES` (32 KiB); identity keys are deliberately outside the shrink order.
- `Library/library_local_rag_search_service.py` `_semantic_row:~1230`: `source_id = metadata.source_id || metadata.document_id || chroma point id`; everything else in metadata lands in `provenance` (minus the popped `source_type`/`item_type`/`type`).
- `Tools/document_expansion_tool.py`: `execute(..., note_id=None, media_id=None, doc_id=None, **_provenance)`; the resolver tries literal AND prefix-stripped (`note_<uuid>`/`media_12`); `DEFAULT_MAX_CHARS = 8000`.
- Canonical entry builders write `source_id`/`source_type` (`RAG_Search/ingestion_indexing.py:592/:637`, media `:548`); `store_documents_batch` (`RAG_Search/simplified/indexing_helpers.py:181`, meta spread at `:~258`) adds `doc_id` (PREFIXED entry id) + `doc_title` + `chunk_start`/`chunk_end`/`chunk_index`. **`media_id` appears in no builder — do not emit it.**
- Non-canonical precedent to copy for the probe's second index: `Docs/superpowers/qa/2026-08-14-rag-answer-first-query-hang/seed_profile.py:64-72` (hand-built `IndexEntry` with `{"type","note_id","title"}` metadata — no `source_id`, so `_semantic_row` falls to the point id).
- Headless drive precedent: `Docs/superpowers/qa/2026-08-14-rag-answer-first-query-hang/probe_headless.py` (env-isolate FIRST, import inside `_load_app_modules()`); the provider's tool path is `Agents/library_rag_tool_provider.py` (invoke with `mode="rag"`); direct engine routes via `rag_service.search(search_type="semantic"|"hybrid")`.

---

### Task 1: The payload addition + pins (AC#2, #4, #5, #6)

**Files:**
- Modify: `tldw_chatbook/Agents/library_rag_tool_provider.py` (`_project_row` — emit `note_id` and `doc_id` from provenance, same precondition block as `chunk_start`)
- Test: extend `Tests/Agents/test_library_tool_provider.py`

**Interfaces (produces):** projected rows additionally carry `"note_id"`/`"doc_id"` (strings, only when present in provenance and non-empty, only under the hint precondition). Task 2 consumes these exact key names.

- [ ] **Step 1:** Venv (pinned recipe; paste provenance). `backlog task edit 16588 -s "In Progress"` + `--plan` (spec+plan paths).
- [ ] **Step 2 (RED):** tests, following the file's existing fixture style:

```python
def test_rows_carry_note_id_and_doc_id_fallbacks()      # provenance {"note_id": "n1", "doc_id": "note_n1"} -> both projected, values verbatim
def test_fallbacks_absent_when_provenance_lacks_them()  # canonical row w/o note_id -> key absent (not None/"")
def test_fallbacks_ride_the_hint_precondition()         # no-hint row (text-bearing note w/ full snippet... use an UNSUPPORTED source_type row) -> no identity, no fallbacks
def test_empty_string_fallbacks_are_dropped()           # provenance {"note_id": ""} -> key absent
def test_media_id_is_never_projected()                  # provenance {"media_id": "7"} -> "media_id" not in row
def test_sealed_payload_survives_fallbacks()            # 10-row payload, strip-and-reserialize byte cost STATED in the assertion message; returned == 10
```

- [ ] **Step 3:** RED run (KeyError/absence failures pasted). **Step 4:** implement — a small helper mirroring `_chunk_start`'s shape (string-coerce, drop empty), called in the same `if hint is not None:` block. **Step 5:** GREEN; provider file + `Tests/Agents/` counts READ; ruff on touched files.
- [ ] **Step 6:** Gate: verbatim `PASSED: No regression. 105 metric(s)`; paste. Commit `feat(agents): payload rows carry note_id/doc_id fallbacks for semantic identity (TASK-16588)` + trailer. Push.

---

### Task 2: The route probe, before/after, closure (AC#1-residue, #3 + close)

**Files:**
- Create: `Docs/superpowers/qa/2026-08-16-rag-semantic-identity/route_probe.py` + `report.md` (+ `probe-artifacts.json`)
- Modify: the TASK-16588 task file (ACs + notes + Done); `Tests/RAG_Eval/README.md` (arc paragraph)

**Probe design (pre-registered):**
- Scratch profile per live-run rules (scratch HOME/TLDW_CONFIG_PATH; real config sha256 before/after; env-isolate BEFORE any tldw import, `probe_headless.py` pattern; HF offline; teardown; CSS restore if any app boot happened — the probe itself must NOT boot the TUI).
- Seed TWO indexes: (1) CANONICAL — ≥12 notes + ≥4 media + ≥4 conversations through `note_document`/`media_document`/`conversation_document` + `index_entries`, including ≥4 documents > 8000 chars with a distinctive mid-document marker sentence placed > 8000 chars in (so a head-window CANNOT contain it); (2) NON-CANONICAL — the same notes hand-built as `IndexEntry` with `{"type","note_id","title"}` metadata (the 15810 script's shape, `seed_profile.py:64-72`).
- Drive BOTH routes per index: the Library provider path (`mode="rag"`) AND direct `rag_service.search(search_type="semantic")` + `search_type="hybrid"` — the provider path is the production projection; the direct path is the control that shows what metadata arrived.
- Per returned row record: hint verdict, identity keys present (`source_id`/`chunk_id`/`chunk_start`/`note_id`/`doc_id`), and for every declared-expandable row a DIRECT `ExpandDocumentTool().execute(**row-identity)` call recording `status`, whether `chunk_start` was passed, `truncated`, and window-contains-marker (substring of the row's own snippet AND, for the long docs, the planted marker).
- Counted per (index kind × route): `not_found` on declared-expandable rows; long-doc windows missing their marker; canonicalization-VARIANT rows receiving no hint (count only — fix belongs to 16688 unless it is a one-line allowlist broadening AND the count is nonzero).

**Pre-registered expectations (a reading outside these is a FINDING to report, not to quietly absorb):**
- Canonical index, pre-fix AND post-fix: `not_found` = 0 (doc_id was already resolvable via metadata `source_id`); nonzero = NEW finding.
- Non-canonical index, PRE-fix (probe run once at Task 1's parent commit via `git stash`-free checkout of the pre-change file — simplest honest method: run the probe with the payload addition DISABLED via a probe-side flag that strips `note_id`/`doc_id` from rows before the expand call): nonzero `not_found` expected — the (b) evidence. POST-fix: 0.
- Long-doc chunked rows with `chunk_start` carried: window contains the marker; a head-window that still contains it means the corpus failed its own design (fix the corpus, not the claim).

- [ ] **Step 1:** Write `questions`/corpus builder + probe; ruff; run PRE-fix mode then POST-fix mode; write `report.md` with the per-(index×route) table and every count.
- [ ] **Step 2:** If any pre-registered expectation is violated: STOP, report the finding, and wait for review before touching production code (the ledger records what happened).
- [ ] **Step 3:** Gate re-read verbatim; batteries `Tests/Agents/ Tests/Library/ Tests/Tools/` counts READ; collection sweep vs merge-base `bab84f7d9`.
- [ ] **Step 4:** Closure: task file — all 6 ACs against evidence (#3 cites the probe table; #4 the byte line from Task 1's sealing test; #6 the gate line), Implementation Notes, status Done (direct file edit). `Tests/RAG_Eval/README.md` gains the arc paragraph with the table. If the variant-row count was nonzero and not fixed here, add the count to TASK-16688's notes (direct edit).
- [ ] **Step 5:** Commit (`qa(rag): semantic-route identity probe — before/after (TASK-16588)` then `docs(backlog): close TASK-16588 on evidence` + trailers). Push.

---

## Self-review (plan time)

- **Spec coverage:** AC#2 → T1; AC#1-residue+#3 → T2 (both index kinds, both routes, before/after); AC#4 → T1 Step 2 sealing test + T2 report; AC#5 → T1's precondition/absence tests + untouched tool; AC#6 → gate in both tasks. Out-of-scope fences repeated in T2's variant-row rule.
- **Placeholder scan:** clean; every test named with its assertion; probe expectations pre-registered per (index × route).
- **Type consistency:** `note_id`/`doc_id` key names identical in T1 Interfaces, T1 tests, T2's expand-call kwargs; `chunk_start` consumed as shipped by 16174's fix wave.
- **Ordering:** T1 before T2 so the post-fix probe measures shipped code; the pre-fix arm uses a probe-side strip flag instead of a checkout dance (no stash, no re-commit).
