# Reranking Residue Batch — Implementation Plan (TASK-17265, TASK-17365, TASK-17600)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the three tasks the reranking thread left behind — the system prompt that never reaches two providers, the cloned-profile token trap, and a middleware surface that declares eleven names and implements four.

**Architecture:** Three tasks. T1 = TASK-17265 (engine + payload-boundary tests). T2 = TASK-17365 (a token floor) + TASK-17600 (the guard, the deletions, the docs correction). T3 = closure + the lesson. Every decision is pre-registered in the spec; do not re-litigate them.

**Spec:** `Docs/superpowers/specs/2026-08-18-reranking-residue-design.md` (amended at `16fbb3d8a` with the 8-name finding).

## Global Constraints

- Worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/rag-rerank-residue`, branch `chore/rag-reranking-residue` (off dev `22d156155`). Venv EXISTS (pinned; provenance verified) — `.venv/bin/python`, never rebuild. ruff via `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff`.
- **EVERY Bash block starts with its own `cd <worktree>` and echoes `pwd` + branch before any mutating git op. After ANY abnormal agent/tool death, run `git status` before trusting a test result** (two incidents: an abandoned mutation and a half-finished fix wave both read as regressions).
- Never `git stash`; Edit-based restores; RED-first; do NOT run `Tests/UI/test_library_shell.py`. **NO live provider calls anywhere in this arc** — fake at the `chat_api_call`/transport seam.
- The gate (`RAG_EVAL=1 … Tests/RAG_Eval/`) is **vacuous for the reranker** (no gated cell runs one). Run it, report it verbatim, and repeat that caveat rather than implying coverage.
- Commits reference the task ids + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; push after every task.

## Verified this session — do not re-derive

- **All 29 providers map `system_message`** in `PROVIDER_PARAM_MAP`, and **all 29 handlers reference their mapped parameter** in their source. So moving the prompt out of the in-band message cannot strand a provider — the one regression risk is cleared at source level (the payload-boundary test still runs; source evidence is not wire evidence).
- The reranker's system turn is built at `reranker.py:315-326`; `max_tokens` is consumed at **exactly one** site (`reranker.py:346`), so the reasoning floor lands in one place.
- **Middleware: 11 declared by pipelines, 4 implemented.** Unimplemented and silently no-op: `abstract_extractor`, `citation_formatter`, `citation_parser`, `code_formatter`, `code_syntax_enhancer`, `result_clustering`, `table_renderer`. `technical_docs` and `research_papers` are made ENTIRELY of unimplemented middleware; three of the seven have no `[middleware.*]` block at all. Handlers: `pipeline_loader.py:429` / `:451`.
- **`reranking_strategy` reads nothing, and TASK-16965's own docs point at it.** `RAG-DESIGN.md:2371` says the strategy "stays selectable as `reranking_strategy = "cross_encoder"`" — false. The REAL path works: `RAGConfigProfile` serialises `asdict(reranking_config)` (`config_profiles.py:145`) and deserialises `RerankingConfig(**…)` (`:169`), and `strategy` is a dataclass field, so a saved/cloned profile round-trips it. The owner's "keep it selectable" ruling is honoured in substance; only the documented lever is wrong.

---

### Task 1: TASK-17265 — the system prompt reaches every provider

**Files:** Modify `tldw_chatbook/RAG_Search/reranker.py`; Test `Tests/RAG_Search/test_reranker_system_prompt.py` (new).

- [ ] **Step 1:** `backlog task edit 17265 -s "In Progress"` + `--plan`. Read `reranker.py:299-350` and one handler that maps `system_message` to a differently-named parameter (`PROVIDER_PARAM_MAP` shows both `system_message` and `system_prompt` targets) — the mapping, not the name, is what must be exercised.
- [ ] **Step 2 (RED):** tests at the **assembled-payload boundary**, not the reranker's call site (AC#1 is explicit). Fake the transport under `chat_api_call` so the real dispatcher and the real handler run, and inspect what the handler would send:
  - `test_anthropic_receives_the_system_prompt` — it arrives wherever anthropic puts system text, NOT dropped.
  - `test_google_receives_the_system_prompt` — same for google's shape.
  - `test_openai_receives_exactly_one_system_instruction` — AC#2: no duplicate/conflicting system text.
  - `test_a_local_provider_receives_exactly_one_system_instruction` — the keyless family.
  - `test_no_in_band_system_role_is_sent` — the messages list carries only the user turn.
- [ ] **Step 3:** Implement: drop the in-band `{"role": "system"}` entry; pass `system_message=(system_prompt or self.config.system_prompt)` to `chat_api_call` (omit when falsy). Keep the existing seam guard green.
- [ ] **Step 4:** GREEN; `Tests/RAG_Search/` counts READ; ruff. Commit `fix(rag): the reranker's system prompt reaches every provider (TASK-17265)` + trailer. Push.

---

### Task 2: TASK-17365 (token floor) + TASK-17600 (the guard and the deletions)

**Files:** Modify `reranker.py` (the floor), `RAG_Search/pipeline_loader.py` + `Config_Files/rag_pipelines.toml` (deletions), `Docs/Development/RAG/RAG-DESIGN.md` + `config.py` + `Helper_Scripts/rag_config_examples/rag_v2_example.toml` (the `reranking_strategy` correction); Tests: extend the reranker tests, and a new middleware-contract test.

- [ ] **Step 1 (17365, RED):** `test_reasoning_raises_the_token_floor` — a config with `include_reasoning=True, max_tokens=100` yields an effective budget ≥ the floor at the single consumption site; `test_a_deliberate_large_max_tokens_is_untouched` — 4000 stays 4000 (it is a FLOOR, not an assignment); `test_no_floor_without_reasoning` — 100 stays 100. Implement at `reranker.py:346`'s source value. **No migration** — the spec's reason: a floor cannot guess wrong about a user's deliberate value.
- [ ] **Step 2 (17600, RED — the guard is the deliverable):** `test_every_declared_middleware_name_is_implemented` — parse `rag_pipelines.toml`'s pipeline `before`/`after` lists, compare against the `middleware_id ==` branches in `pipeline_loader.py`, and FAIL on any declared-but-unimplemented name. It must be red NOW (7 names). Add the reverse direction too (implemented-but-undeclared) per the spec's both-directions rule.
- [ ] **Step 3 (17600, the deletions):** remove `result_reranking` and the seven unimplemented names from the pipelines that declare them, and any `[middleware.*]` block left with no implementation and no consumer. **Do not wire reranking on** — TASK-16965 measured it net-harmful, so switching it on for `high_accuracy` users is exactly what the measurement forbids; say so in the diff. If removing a pipeline's whole middleware list leaves it empty, remove the key rather than leaving `[]` implying a feature.
- [ ] **Step 4 (17600 F3, the docs correction):** `reranking_strategy` reads nothing. Delete it from the commented config examples, and correct `RAG-DESIGN.md:2371` to name the mechanism that actually works — a saved/cloned profile's `reranking_config.strategy`, which round-trips through `RAGConfigProfile`. This is TASK-16965's own claim being made true.
- [ ] **Step 5:** GREEN; batteries (`Tests/RAG_Search/`, `Tests/RAG/` if it covers pipelines) counts READ; ruff. Commit `fix(rag): a reasoning token floor; delete the middleware promises (TASK-17365, TASK-17600)` + trailer. Push.

---

### Task 3: Closure + the lesson

- [ ] **Step 1:** Close all three task files — every AC against evidence. 17265's AC#1 cites the payload-boundary tests by name; 17600's AC#2 cites the both-directions guard and lists the eight names removed; 17365 records the floor decision and why not a migration.
- [ ] **Step 2 (the lesson, if earned — check `lessons-testing-evidence.md` for a near-duplicate FIRST and extend rather than duplicate):** the incident is that a config surface can be *declared, enabled and documented* while implementing nothing, and that three separate arcs each found one instance (parent-inclusion knobs, `result_reranking`, `reranking_strategy`) before anyone looked for the CLASS. The rule: when you find one inert declared surface, enumerate its whole namespace in both directions before closing — the second instance is usually adjacent to the first.
- [ ] **Step 3:** Gate verbatim (with the vacuity caveat); collection sweep vs merge-base `22d156155`; batteries; commit + push.

## Self-review (plan time)
- 17265 AC#1→T1S2 (payload boundary, not call site), #2→the two exactly-one tests, #3→the new file alongside the existing seam guard, #4→fake transport everywhere.
- 17365's ACs→T1... no: →T2S1, with the migrate-vs-floor decision pre-registered and the deliberate-value test pinning the floor semantics.
- 17600 AC#1→T2S3 (delete arm), #2→T2S2 (both directions), #3→n/a under the delete arm (recorded in closure), #4→T2S2's guard.
- No live calls; the gate's vacuity is stated rather than implied; the ordering puts the guard RED before the deletions so the deletions are what turn it green.
