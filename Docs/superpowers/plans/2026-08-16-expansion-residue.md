# TASK-16688 + TASK-16788: Expansion Residue Batch — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the five 16174-review residue findings (pins + one fetch fix + one QA walk + two recordings) and record+pin the `allowed_tools` runtime-tool contract.

**Architecture:** Two tasks, both mechanical — every decision is pre-registered in the spec. Task 1 = TASK-16688 (all five decisions). Task 2 = TASK-16788 + both task-file closures.

**Spec:** `Docs/superpowers/specs/2026-08-16-expansion-residue-design.md` — binds every step; the decisions there are not to be re-litigated by the implementer.

## Global Constraints

- Worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/rag-16688-residue`, branch `chore/rag-16688-16788-residue` (off dev `c2f30862c`). **cwd resets every Bash block — cd first, EVERY block, especially before any push.**
- **VENV (none exists):** `uv venv .venv --python 3.12 && VIRTUAL_ENV=.venv uv pip install -e ".[dev,embeddings_rag]" "transformers==5.6.2" "torch==2.11.0" "chromadb==1.5.8"`; paste the in-worktree import-provenance line. ruff via `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff`.
- Never `git stash`; Edit restores; single foreground Bash (timeout 600000); do NOT run `Tests/UI/test_library_shell.py`. RED-first for every new pin.
- Gate (Task 1 only — AC#4 changes a fetch path): `RAG_EVAL=1 .venv/bin/python -m pytest Tests/RAG_Eval/ -q -p no:randomly` reads verbatim `PASSED: No regression. 105 metric(s)`.
- Commits reference the task ids, end `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; push after every task.

## Verified code anchors (grep first; lines drift)

- Twin literals: `Tools/document_expansion_tool.py` `SUPPORTED_SOURCE_TYPES` / `PROMPT_BODY_COLUMNS`; `Library/library_expand_policy.py` `EXPANDABLE_SOURCE_TYPES`; `RAG_Search/simplified/rag_service.py` `PROMPT_DOCUMENT_COLUMNS`.
- Conversation fetch: `Tools/document_expansion_tool.py` `_fetch_conversation` → `get_messages_for_conversation` (default `include_image_data=True`).
- Long-doc corpus pattern: `Docs/superpowers/qa/2026-08-16-rag-semantic-identity/route_probe.py` (`_filler`/`build_long_document`, env-isolation header, `_validated_scratch_path`).
- 16788 facts for the docstring: `allowed_tools` filters the CATALOG (`agent_service.py:1501` + the Q7 find_tools guard + `invoke_tool`'s `:1041` name check); run-log schemas append under `log_active` (`:1564-1580`); **runtime CALLS dispatch by name in `agent_runtime.py:1383-1400`'s dedicated `elif` branches BEFORE the generic tool fallback, so `allowed_tools` structurally never sees them** — same family as spawn/find_tools/load_tools/skill-file. Oracle confound note: `Docs/superpowers/qa/2026-08-15-rag-agentic-expansion/report.md`.
- Task files on dev: `ls backlog/tasks/ | grep -E "16688|16788"` (16688's title has an em-dash-free slug; do NOT `backlog task edit` content — direct edits only; status changes via CLI are fine).

---

### Task 1: TASK-16688 — five decisions landed

**Files:** Modify `Tools/document_expansion_tool.py` (one kwarg), `Library/library_expand_policy.py` (docstring note); Create `Tests/Tools/test_expansion_twin_literals.py`, `Docs/superpowers/qa/2026-08-16-expansion-residue/continuation_walk.py` + `report.md`; Modify `Tests/Tools/test_document_expansion_tool.py` (kwargs pin), `Tests/Library/test_library_expand_policy.py` (variant-exclusion pin), `Docs/User_Guide/mcp.md` (consent paragraph + stamp), `Tests/RAG_Eval/README.md` (one variant-count sentence if not already there).

- [ ] **Step 1:** Venv; provenance pasted. `backlog task edit 16688 -s "In Progress"`; same for 16788 (Task 2 will close it).
- [ ] **Step 2 (AC#1, RED-first):** `test_expansion_twin_literals.py`:

```python
def test_source_type_allowlists_cannot_drift():
    assert set(SUPPORTED_SOURCE_TYPES) == set(EXPANDABLE_SOURCE_TYPES)

def test_prompt_body_columns_match_rag_service():
    assert tuple(PROMPT_BODY_COLUMNS) == tuple(PROMPT_DOCUMENT_COLUMNS)  # order matters: the join rule depends on it
```
RED by mutation (temporarily add a bogus entry to one side, observe red, revert via Edit) — paste both directions.
- [ ] **Step 3 (AC#2):** variant-exclusion pin in the policy tests (a `media_chunk` row → `expand_hint(...) is None`), docstring note citing the 16588 probe's 0/340 with positive control; the README sentence.
- [ ] **Step 4 (AC#4, RED-first):** kwargs pin (wrap the real `get_messages_for_conversation` on a seeded conversation, assert `include_image_data is False` in recorded kwargs) → RED → add the kwarg → GREEN; existing transcript tests untouched and green.
- [ ] **Step 5 (AC#5):** `continuation_walk.py` (env-isolate FIRST; `_validated_scratch_path` precedent; ONE >20,000-char note; expand at default budget → `truncated is True` + `next_offset`; walk to exhaustion; assert concatenated windows == the document text; write `report.md` with windows/coverage/call-count). Run it; commit the report.
- [ ] **Step 6 (AC#3):** the mcp.md consent paragraph + "Verified against" stamp refresh; the same trade-off paragraph goes in the 16688 task file's notes (Task 2 finalizes the file).
- [ ] **Step 7 (AC#6):** gate verbatim + `Tests/Tools/ Tests/Agents/ Tests/Library/` counts READ; ruff on touched files. Commit `chore(rag): 16688 residue — pins, image-blob fix, continuation walk, consent record (TASK-16688)` + trailer. Push (cd in the SAME block).

---

### Task 2: TASK-16788 — document + pin; close both tasks

**Files:** Modify `tldw_chatbook/Agents/agent_service.py` (the `allowed_tools` docstring on `AgentRunConfig` or wherever the param is documented — grep `allowed_tools` docstrings), `Tests/Agents/` (the pin; follow the file layout of existing agent_service tests); the two task files (closure).

- [ ] **Step 1 (docstring):** state the contract where `allowed_tools` is documented: it filters the CATALOG; runtime tools (spawn_subagent, wait/check agents, find_tools/load_tools, the skill-file tool, install_skill, run_skill_script, search_run_log/run_log_stats/run_log_slice) are offered under their OWN gates regardless, and their calls dispatch in `agent_runtime.py`'s dedicated branches before the catalog fallback; reference the oracle confound note path.
- [ ] **Step 2 (pin, RED-first against a mutated expectation):** a test that runs the schema-assembly path (or the narrowest existing harness for `_run_one`'s disclosure — follow existing tests' fixtures) with an EMPTY `allowed_tools` and `log_active` conditions satisfied, asserting the three run-log schemas ARE offered. Mutation: filter runtime_schemas by allowed_tools locally → test reds → revert.
- [ ] **Step 3:** batteries `Tests/Agents/` counts READ; ruff. Close BOTH task files (direct edits: ACs → `[x]` with evidence, Implementation Notes, status Done via CLI). Commit `docs(agents): allowed_tools governs the catalog — runtime-tool contract recorded and pinned (TASK-16788)` + closure commit + trailers. Push.

---

## Self-review (plan time)
- Spec coverage: 16688 AC#1→T1S2, #2→T1S3, #3→T1S6, #4→T1S4, #5→T1S5, #6→T1S7; 16788 all three ACs→T2. No placeholders; anchors verified this session (runtime dispatch at agent_runtime.py:1383-1400 read directly). Ordering: the gate runs in T1 where the only behaviour change lives; T2 is docs+pin only.
