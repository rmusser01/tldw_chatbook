# TASK-16174: Agentic Document Expansion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retire the inert parent-inclusion knobs, ship one gated `expand_document` tool whose contract works from exactly what retrieval rows carry, wire an expansion policy into the payload the agent actually sees, and answer the evaluation question with a run, not an assumption.

**Architecture:** Four phases in dependency order: K (knob truth — retire + compat), T (the tool), P (the wired policy), E (the oracle measurement + closure). No retrieval behaviour changes anywhere: the gated suite must read 105/105 (+0.000) at the end.

**Tech Stack:** Python 3.11+, dataclass config, the `Tool` ABC + `_GATEABLE_BUILTINS` catalog, pytest, the RAG eval harness corpus.

**Spec:** `Docs/superpowers/specs/2026-08-15-rag-agentic-expansion-design.md` — binds every task; read first.

## Global Constraints

- Worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/rag-16174-expansion`, branch `feat/rag-16174-agentic-expansion` (off dev `8727a2861`). **cwd resets every Bash block — cd first, EVERY block.**
- **VENV (none exists yet):** `uv venv .venv --python 3.12 && VIRTUAL_ENV=.venv uv pip install -e ".[dev,embeddings_rag]" "transformers==5.6.2" "torch==2.11.0" "chromadb==1.5.8"`. Assert `.venv/bin/python -c "import tldw_chatbook; print(tldw_chatbook.__file__)"` resolves IN the worktree before any test run; paste the line. If dev added a dep, install it — never rebuild.
- Never `git stash`; Edit-based restores; single foreground Bash (timeout 600000); do NOT run `Tests/UI/test_library_shell.py` (monolith, failure-to-exit). TCC "Operation not permitted" = stop and report.
- The app regenerates `css/tldw_cli_modular.tcss` on boot — restore before committing after any live run.
- Gate: `RAG_EVAL=1 pytest Tests/RAG_Eval/` must read **PASSED 105/105 (+0.000)** in Tasks 1 and 4. A moved cell = STOP and report.
- Commits reference TASK-16174 and end `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; push after every task.
- API keys in the repo root are FOR AGENT USE (Task 4's live run); never echoed to logs. Pytest runs are network-blocked by default — the live run is a SCRIPT under `Docs/superpowers/qa/`, never a test.

## Verified code anchors (line numbers drift; grep first)

- Knobs: `RAG_Search/simplified/config.py:559-561` (`include_parent_docs`, `parent_size_threshold`, `parent_inclusion_strategy`); profile writes `RAG_Search/config_profiles.py:310-312`, `:347-349`, `:524-526`. Grep-verified 4 occurrences = 1 definition + 3 writes, 0 reads.
- **Compat hazard (verified):** `SearchConfig(**search_data)` at `config.py:710` (`from_dict`) is a plain dataclass kwargs call over a user-editable dict — a retired key arriving from saved config raises `TypeError`. Phase K MUST filter `search_data` to `dataclasses.fields(SearchConfig)` names with a warn-once log for dropped keys (precedent for hostile-dict defence: `rag_service.py:3689` comment).
- Row identity: keyword-leg rows carry `source_id` + `provenance.source_type`, media/conversation with `chunk_id: ""` and label snippets (`library_local_rag_search_service.py:1187/:1202`). **Semantic rows (`_semantic_row:1230`): `source_id` = metadata `source_id` || `document_id` || the chroma point id — the real document identity may live in `provenance` extras (`note_id`/`doc_id` are chroma metadata fields). The tool must accept provenance extras as identity fallbacks.**
- Fetch APIs: `Client_Media_DB_v2.get_media_by_id:6158`, `ChaChaNotes_DB.get_conversation_by_id:7575` (+ messages via the conversation's message APIs), `ChaChaNotes_DB.get_note_by_id:11575`, `Prompts_DB.get_prompt_by_id:2918`.
- Catalog: `Agents/tool_catalog.py:467` `GateableTool(gate_key, module, class_name, tool_name)`; rows at `:492-516`. Tool ABC + `risk_tags` in `Tools/tool_executor.py` (a risk-tagged tool is floored to `ask`). DB-handle precedent: `Tools/note_management_tools.py`.
- Provider payload: `Agents/library_rag_tool_provider.py` `_project_row` + the sealing loop bounding `serialized_size(payload)` to `MAX_RESULT_BYTES` — an added per-row key adds bytes; the sealing loop already degrades gracefully, but Task 3 adds a test proving hints survive sealing on a normal payload.
- Eval corpus: `Tests/RAG_Eval/fixtures/corpus.toml` (172 docs; source_type counts include media 19+8, conversation 13, prompt 6) seeded through production APIs by `Tests/RAG_Eval/harness/ingest.py` (`add_media_with_keywords`, `add_note`, `add_conversation`/`add_message`).

---

### Task 1: Venv + Phase K — retire the knobs, survive saved configs

**Files:**
- Modify: `tldw_chatbook/RAG_Search/simplified/config.py` (delete 3 fields; filter unknown keys in `from_dict`'s `search_data` path)
- Modify: `tldw_chatbook/RAG_Search/config_profiles.py` (delete the 3×3 profile writes)
- Test: `Tests/RAG_Search/test_search_config_compat.py` (new)

**Interfaces (produces):** `SearchConfig` without the three fields; `from_dict` tolerant of unknown search keys (drops + warns once per key). Later tasks rely on nothing else from here.

- [ ] **Step 1:** Build the venv (pinned recipe above); paste import provenance. `backlog task edit 16174 -s "In Progress"` and add `--plan` pointing at spec+plan paths.
- [ ] **Step 2:** Write the failing tests:

```python
def test_saved_config_with_retired_parent_keys_loads(caplog):
    data = {"search": {"include_parent_docs": True, "parent_size_threshold": 5000,
                       "parent_inclusion_strategy": "size_based", "top_k": 7}}
    cfg = RAGConfig.from_dict(data)          # must NOT raise TypeError
    assert cfg.search.top_k == 7
    assert not hasattr(cfg.search, "include_parent_docs")
    assert any("include_parent_docs" in r.message for r in caplog.records)

def test_unknown_search_key_is_dropped_with_notice(caplog):
    cfg = RAGConfig.from_dict({"search": {"never_a_field": 1}})
    assert not hasattr(cfg.search, "never_a_field")

def test_no_profile_sets_parent_inclusion():
    import tldw_chatbook.RAG_Search.config_profiles as m
    src = inspect.getsource(m)
    assert "include_parent_docs" not in src
```

- [ ] **Step 3:** Run them: first two FAIL (TypeError / field exists), third FAILS. Paste output.
- [ ] **Step 4:** Implement: delete the three fields from `SearchConfig`; delete the nine profile lines; in `from_dict`, before `SearchConfig(**search_data)`, filter to `{f.name for f in dataclasses.fields(SearchConfig)}` and `logger.warning` each dropped key once. Apply the same filter pattern ONLY to search_data (other sections are out of scope).
- [ ] **Step 5:** GREEN. Then the no-reads proof: `grep -rn "include_parent_docs\|parent_size_threshold\|parent_inclusion_strategy" tldw_chatbook/ Tests/` → expect ZERO in tldw_chatbook/, only this task's compat test in Tests/. Fix any test fixture that set them.
- [ ] **Step 6:** Gate: `RAG_EVAL=1 pytest Tests/RAG_Eval/ -q` → must read PASSED 105/105 (+0.000); plus `pytest Tests/RAG_Search/ -q` counts READ.
- [ ] **Step 7:** Commit `feat(rag): retire inert parent-inclusion knobs; saved configs survive (TASK-16174)` + trailer. Push.

---

### Task 2: Phase T — the `expand_document` tool

**Files:**
- Create: `tldw_chatbook/Tools/document_expansion_tool.py`
- Modify: `tldw_chatbook/Agents/tool_catalog.py` (one `GateableTool` row)
- Test: `Tests/Tools/test_document_expansion_tool.py` (new)

**Interfaces (produces):** class `ExpandDocumentTool(Tool)`, tool name `expand_document`, gate key `expand_document_enabled`; async `execute(source_type: str, source_id: str, chunk_id: str = "", offset: int = 0, max_chars: int | None = None) -> dict`. Return shape (all types): `{"status": "ok"|"not_found"|"unsupported", "source_type", "source_id", "title", "text", "total_size", "window": {"start", "end"}, "truncated": bool, "next_offset": int | None}`. Task 3 consumes the tool name and this shape.

- [ ] **Step 1:** Read `Tools/tool_executor.py` (Tool ABC + `risk_tags`) and `Tools/note_management_tools.py` (DB-handle acquisition precedent). Record in the report how DB paths are resolved there and mirror it.
- [ ] **Step 2:** Write failing contract tests against real in-memory/scratch DBs (the repo's standard), one per branch:

```python
async def test_note_expands_to_full_body()            # seeded note; status ok, text == body, truncated False
async def test_media_label_only_row_expands()         # ONLY source_type+source_id (chunk_id "") — the AC#4 case
async def test_conversation_returns_role_prefixed_transcript()
async def test_prompt_expands()
async def test_over_budget_returns_window_and_next_offset()  # body > max_chars; window head; next_offset set
async def test_chunk_centred_window_when_chunk_id_known()    # window contains the chunk text, not the head
async def test_offset_continuation_walks_the_document()      # second call with next_offset returns the next window
async def test_unknown_id_is_not_found()              # status not_found, no raise
async def test_semantic_identity_fallbacks()          # provenance extras: note_id/doc_id accepted when source_id is a chroma point id
def test_tool_is_gated_off_by_default()               # catalog: gate key absent/false -> tool not offered
def test_tool_carries_risk_tags()                     # floored to ask
```

- [ ] **Step 3:** Run → all FAIL (module missing). Paste.
- [ ] **Step 4:** Implement `ExpandDocumentTool`: per-type fetch via the anchors (`get_note_by_id` / `get_media_by_id` / `get_conversation_by_id`+messages / `get_prompt_by_id`); identity resolution order = explicit `source_id`, then provenance-style fallbacks the caller may pass (`note_id`, `doc_id`, `media_id`) — accept them as optional kwargs so an agent can paste a row's provenance verbatim; budget default from a module constant (`DEFAULT_MAX_CHARS = 8000`, cap `HARD_MAX_CHARS = 32000`); truncation window logic exactly as the tests pin. `description` states the contract AND the policy sentence (Task 3 finalises wording). `risk_tags` set (reads user data).
- [ ] **Step 5:** Catalog row: `GateableTool("expand_document_enabled", "document_expansion_tool", "ExpandDocumentTool", "expand_document")`. Confirm the Settings ▸ Tools screen derives the switch (it derives from `_GATEABLE_BUILTINS` — assert via the existing catalog tests' pattern, and say so in the report).
- [ ] **Step 6:** GREEN; whole-file ruff; `pytest Tests/Tools/ Tests/Agents/ -q` counts READ (name pre-existing reds if any).
- [ ] **Step 7:** Commit `feat(tools): gated expand_document tool — chunk-to-document expansion (TASK-16174)` + trailer. Push.

---

### Task 3: Phase P — the policy, wired into what the agent sees

**Files:**
- Modify: `tldw_chatbook/Agents/library_rag_tool_provider.py` (`_project_row` gains `expand_hint`)
- Create: `tldw_chatbook/Library/library_expand_policy.py` (pure helper)
- Modify: `tldw_chatbook/Tools/document_expansion_tool.py` (final `description` policy wording)
- Test: `Tests/Library/test_library_expand_policy.py`, extend `Tests/Agents/test_library_rag_tool_provider.py`

**Interfaces:**
- Consumes: Task 2's tool name/shape.
- Produces: `expand_hint(row: Mapping) -> dict | None` in `library_expand_policy.py` returning `{"expandable": bool, "reason": "label_only"|"truncated_snippet"|"text_bearing"}`; provider rows gain `"expand_hint": {...}` when the hint is not None.

- [ ] **Step 1:** Failing tests:

```python
def test_media_label_row_hint_is_label_only()         # snippet 'Matched media · pdf' -> expandable True, reason label_only
def test_conversation_label_row_hint_is_label_only()
def test_text_bearing_note_row_hint_is_text_bearing() # expandable False
def test_truncated_semantic_snippet_hint()            # snippet ends mid-sentence at the projection cap -> truncated_snippet
def test_provider_rows_carry_expand_hint()            # end-to-end through _project_row
def test_sealed_payload_survives_hints()              # normal 10-row payload with hints stays under MAX_RESULT_BYTES and keeps hints
```

- [ ] **Step 2:** RED → paste. **Step 3:** Implement: the helper is pure (string/shape inspection only — label detection by the `provenance.source_type` + empty-`chunk_id` + label-prefix triple, never by parsing the label text alone); `_project_row` attaches it; tool `description` final wording: "Expand a retrieval hit into its document. Use when a high-ranked hit is label-only (media/conversation rows) or its snippet is truncated and the answer needs the content. Re-query instead if the hit itself looks irrelevant. Never expand the same source twice — reuse the earlier result." **Step 4:** GREEN; targeted suites counts READ. **Step 5:** Commit `feat(agents): per-row expand hints wire the expansion policy into the payload (TASK-16174)` + trailer. Push.

---

### Task 4: Phase E — the oracle run, the gate, AC#6, closure

**Files:**
- Create: `Docs/superpowers/qa/2026-08-15-rag-agentic-expansion/oracle_run.py` + `questions.toml` + `report.md`
- Modify: the TASK-16174 task file (ACs + Implementation Notes + Done); `Tests/RAG_Eval/README.md` (arc section)

- [ ] **Step 1:** Author ≥6 questions with fact oracles from `corpus.toml`'s media/conversation docs — each oracle string verified (grep) to appear ONLY in that doc's body, never in titles/labels/notes. Record the verification in `questions.toml` comments.
- [ ] **Step 2:** `oracle_run.py`: scratch profile seeded via the harness ingest path; the real agent loop (AgentService + the Library retrieval tool + `expand_document`) with a repo-root API key (cheap model, temperature 0); run each question tool-OFF then tool-ON; score = oracle-fact inclusion (case-insensitive substring/regex); print per-question table + total spend. Config-isolation per `lessons-live-verification.md`; CSS bundle restored after.
- [ ] **Step 3:** Run it. The deliverable is the TABLE, whatever it says — an OFF≥ON result is a finding, not a failure; report honestly either way. Commit the artifacts.
- [ ] **Step 4:** Gate + batteries: `RAG_EVAL=1 pytest Tests/RAG_Eval/ -q` → PASSED 105/105 (+0.000); `pytest Tests/Library/ Tests/Tools/ Tests/Agents/ Tests/RAG_Search/ -q` counts READ; collection sweep vs merge-base `8727a2861`.
- [ ] **Step 5:** Close the task file: all 7 ACs against evidence (#5 cites the run artifact; #6 one recorded sentence: independent of reranking, 3502 unpresumed; #7 cites Task 1's grep-zero). Implementation Notes per house style. `Tests/RAG_Eval/README.md` arc section states what shipped + the oracle table. Status Done.
- [ ] **Step 6:** Commit `docs(backlog): close TASK-16174 on evidence (TASK-16174)` + trailer. Push.

---

## Self-review (plan time)

- **Spec coverage:** K→T1, T→T2, P→T3, E+closure→T4; AC#1/#2 (T2), #3 (T3), #4 (T2 label-only tests + T3 hints), #5 (T4 run), #6 (T4 sentence), #7 (T1). Out-of-scope items untouched.
- **Placeholder scan:** clean — every test named with its assertion, every decision pre-registered.
- **Type consistency:** `expand_document`/`ExpandDocumentTool`/`expand_document_enabled` and the return shape identical in T2's Interfaces and T3/T4's consumption; `expand_hint` shape identical in T3's Interfaces and tests.
- **Ordering:** T1 first so no dead surface coexists with the new tool even transiently; T2 before T3 (hints reference the tool's existence in description wording); T4 last (measures the assembled loop; gate proves zero retrieval change).

---

### Task 3b (added 2026-08-15, after Task 3's report): close the identity loop in the payload

**Why (T3's disclosed concern):** the provider row carries `expand_hint`
but not the `source_type`/`source_id` that `expand_document` REQUIRES —
for label-only rows `result_id` merely happens to equal `source_id`, and
the seam is only inferable from label prose. The policy must be
actionable BY CONTRACT, or Task 4's tool-ON arm measures the model's
inference ability, not expansion's value.

**Files:** Modify `Agents/library_rag_tool_provider.py` (`_project_row`);
extend `Tests/Agents/test_library_tool_provider.py`.

- [ ] **Step 1 (RED):** tests: every projected row carries `source_type`
  and `source_id` matching the service row's `provenance.source_type` /
  `source_id`; a semantic row additionally carries its non-empty
  `chunk_id`; a label-only row's `source_id` equals what the row already
  exposed as `result_id`; the sealed-payload test still passes with the
  added keys (re-measure the per-row byte cost and state it).
- [ ] **Step 2:** implement (payload ADDITION only — `expand_hint`'s
  pinned `{expandable, reason}` interface and all existing keys
  unchanged). **Step 3:** GREEN + battery counts read. **Step 4:** commit
  `feat(agents): payload rows carry the identity expand_document requires (TASK-16174)`
  + trailer; push.
