# Propositions Vendoring & Program Descope Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land `strategies/propositions.py` (the 39th vendored file — heuristic/spacy/llm engines, zero new shims) so the already-routed `propositions` method works, and record the permanent descope rulings for `auto_boundary_assistant`, `async_chunker`, and telemetry — closing the program.

**Architecture:** A manifest move + sync (the #2/#3 pattern); one `_KNOWN` prompt mapping in the existing `prompt_loader` shim; the sync script's skip-table rewritten so the two formerly-deferred test files carry terminal dispositions (the un-skipped propositions suite revives; the templates suite's reason gets its final wording); descope rulings written into the manifest's excluded comments and pinned by tests; fixtures + docs.

**Tech Stack:** Python ≥3.11, the vendoring machinery (manifest/sync/rewrite rules), pytest with the `production_path` marker for fixtures.

**Spec:** `Docs/superpowers/specs/2026-08-23-propositions-vendoring-design.md` — §4 the descope rulings, §5 vendoring + the LLM-contract precedent, §6 testing, §7's 9 ACs, §8's 5 rulings.

## Global Constraints

- Never move the pin (`dev` @ `385afa951922c8a9dc2002c675bb6cad65e4ac23`); `strategies/propositions.py` is a MOVE from `excluded` to `vendored`, never both lists.
- Vendored files are never hand-edited (sync-script rewrite rules only).
- Zero new shims; zero new dependencies (spacy stays optional — no pyproject change).
- The LLM contract: the shim's adapter translates chatbook's payload-dict callback to the engine's positional `llm_call_func(api_name, prompt, None, api_key, system_message, temp, False, False, False, model_override=…)` — the #1 rolling_summarize precedent.
- Fallback-to-heuristics on LLM failure and the guarded server-config keys are PARITY, preserved as-is (never "fixed" to fail-close).
- Golden fixtures cover the heuristic engine only; the LLM engine's coverage is stub-based.
- Repo-wide grep-clean of "deferred to #6" residue when done (the manifest, the sync script's table, the regenerated test files, and any spec prose that still says it — #1's spec's historical record is exempt as history).
- Repo rule: targeted test runs only; venv `.venv/bin/python`; exclude `Tests/Chunking/test_sync_script.py` (network hang — use `/tmp/tldw_server_sync` at the pin when the sync must run).

---

### Task 1: Vendor propositions + the descope ledger

**Files:**
- Modify: `Helper_Scripts/sync_chunking_engine.py` (VENDORED list gains `strategies/propositions.py`; `TESTS_MODULE_SKIPPED` — the propositions entry REMOVED, the templates entry's reason reworded terminal), `tldw_chatbook/Chunking/engine/VENDOR_MANIFEST.toml` (the move; `excluded` comments carry the not-vendored rulings), `tldw_chatbook/Chunking/_shims/Utils/prompt_loader.py` (`_KNOWN` gains the propositions mapping)
- Create (via sync): `tldw_chatbook/Chunking/engine/strategies/propositions.py`; the regenerated `Tests/Chunking/test_propositions_strategy.py` (skip block gone) + `test_upstream_chunking_templates.py` (terminal reason)
- Test: `Tests/Chunking/test_sync_script.py` (extend), a new descope-pin test

**Interfaces:**
- Produces: `from tldw_chatbook.Chunking.engine.strategies.propositions import PropositionChunkingStrategy` (Task 2 consumes); the manifest's excluded comments as the descope ledger.

- [ ] **Step 1 — failing tests:** the manifest/import pins (propositions in vendored + not-in-excluded + count 39; the module importable; `engine.strategies.propositions` referenced by no shim) AND the descope pins (the manifest's `excluded` entries for `auto_boundary_assistant.py`/`async_chunker.py` contain "not vendored" ruling text — a comment-parsing assertion; `grep -ri "deferred to #6"` over `Helper_Scripts/`, `tldw_chatbook/`, `Tests/` returns zero hits after the sync). Red.
- [ ] **Step 2 — manifest + script edits + sync:** the move in both lists; the skip-table rewrite; the excluded-comments rulings; the `_KNOWN` mapping (read the vendored propositions.py's `load_prompt(...)` call — likely `("chunking", "Proposition-based Chunking")` per the pin, but use what the call actually passes); run the sync against `/tmp/tldw_server_sync` at the pin.
- [ ] **Step 3 — green:** the sync-contract suite (39), the un-skipped `test_propositions_strategy.py` (10 upstream tests), the descope pins; full `Tests/Chunking/ -q --ignore=Tests/Chunking/test_sync_script.py` no regressions.
- [ ] **Step 4 — the AC #2 execution check:** `improved_chunking_process(text, {"method": "propositions", "max_size": 5})` returns chunks (heuristic) — the exact call verified failing in discovery.
- [ ] **Step 5 — commit:** `feat(chunking): vendor propositions (39th file, zero shims); descope ledger for the not-vendored files`

### Task 2: LLM-contract adapter + fallback pin + fixtures

**Files:**
- Modify: `tldw_chatbook/Chunking/Chunk_Lib.py` (the adapter: the method's chunk_text/process_text path passes the engine a positional-wrapping callable when the caller supplied the payload-dict `llm_call_function` — locate how rolling_summarize's adapter does it and mirror; guarded so no-callback → engine heuristic default), `Tests/Chunking/test_auto_planner_parity_fixtures.json` (propositions heuristic cases) + the fixture generator/test if separate
- Test: `Tests/Chunking/test_chunk_lib_shim.py` (extend)

**Interfaces:**
- Consumes: `PropositionChunkingStrategy` (Task 1).
- Produces: `propositions` honoring `llm_call_function_for_chunker` (payload-dict) through the adapter.

- [ ] **Step 1 — failing tests (§6.4-6.5):** a stubbed positional callback receives the translated call (payload-dict caller → adapter → positional, kwargs intact) and its strings become proposition chunks; the stub RAISES → heuristic output returned, no raise (the fallback pin); no-callback call → heuristic chunks (the default). Red.
- [ ] **Step 2 — implement** the adapter translation in the shim (mirror the rolling_summarize adapter's shape).
- [ ] **Step 3 — fixtures:** propositions heuristic cases join the golden corpus (generate with test mode off; assert byte-equality).
- [ ] **Step 4 — green + commit:** `feat(chunking): propositions LLM adapter (payload-dict to positional) + heuristic fallback pin + parity fixtures`

### Task 3: Docs + close-out

**Files:**
- Modify: `CHANGELOG.md`; the user-guide method-list page(s) (locate where #2/#3 added methods — the ingest docs' chunking-method list)
- Test: close-out run

- [ ] **Step 1 — docs:** CHANGELOG (the method now works); the method list gains `propositions`.
- [ ] **Step 2 — targeted close-out:** `pytest Tests/Chunking/ Tests/Library/test_agent_chunk_student_story.py Tests/Performance/test_app_import_weight.py -q --ignore=Tests/Chunking/test_sync_script.py` — zero new failures.
- [ ] **Step 3 — the repo-wide descope-residue grep** (the Task 1 pin re-run at final state; #1's spec history exempt).
- [ ] **Step 4 — commit:** `docs(chunking): propositions method live; program-close docs`

## Self-Review (run at save)

1. **Spec coverage:** AC 1→T1; AC 2→T1 (the execution check); AC 3→T1 (un-skip + terminal wordings + residue grep); AC 4→T2; AC 5→T1 (no pyproject change — verify no diff); AC 6→T1 (the ledger + drift-obligation note rides the spec, already written); AC 7→T2; AC 8→T3; AC 9→T3 close-out. All 5 §8 rulings: 1→T1's ledger, 2→T2's adapter, 3→T2's fallback pin, 4→T2's fixture scope, 5→T1's no-wiring fact (the AC 2 check proves it).
2. **Ordering:** T1 (vendored file) before T2 (consumes the strategy); T3 last.
3. **Type consistency:** `PropositionChunkingStrategy(language, llm_call_func, llm_config)`; the positional call shape quoted verbatim from the pin in T2's test; the `_KNOWN` pair decided by reading the actual call at execution (stated, not guessed).
4. **Placeholders:** none — every code-bearing step has its contract; the two read-at-execution facts (the `_KNOWN` pair, the adapter's mirror-site) are named with where to look.
