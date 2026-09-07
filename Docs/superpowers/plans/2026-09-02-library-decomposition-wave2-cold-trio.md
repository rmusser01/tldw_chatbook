# Library Decomposition Wave 2 — Cold Trio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the export, collections, and search subsystems from `LibraryScreen` following the recipe the conversations exemplar established, after first closing the census anti-slack guard gap (task-27019).

**Architecture:** Identical to the foundation series: per-subsystem state object + controller(s) under the byte-for-byte canon, recipe-doc mechanics, both ratchets lowered per move PR. This plan adds no new mechanism — `backlog/docs/library-decomposition-recipe.md` §1–§11 is the how; this plan is the task boundaries, wave-specific decisions, and gates.

**Spec:** `Docs/superpowers/specs/2026-09-01-library-screen-decomposition-design.md` (as corrected 2026-09-02). **Recipe (mandatory reading for every task):** `backlog/docs/library-decomposition-recipe.md`. **Worked example:** the conversations series — `tldw_chatbook/UI/Library_Modules/library_conversations_state.py`, `library_conversation_reader_controller.py`, `library_conversations_controller.py`, `Tests/Architecture/test_library_conversations_wiring.py`.

## Global Constraints

- Everything in the foundation plan's Global Constraints applies verbatim (pure moves at the canon's strictness; interleaved churn awareness; monkeypatch-routing; worktree venv; `-p no:randomly`; blame-ignore per move commit; xdist paired-baseline sweep protocol per recipe §7 with its documented pre-existing failures list).
- **Every move PR lowers the library `_BUDGETS` row to its own post-move measurement** (spec as corrected; ceiling AND slack must be green at every task boundary).
- One subsystem series fully lands before the next starts. Series order: export → collections → search.
- Wave-2 measured baselines (2026-09-02, post-foundation, file at 43,965/1,282): export 51 methods / 12 init-fields; collections 67 / 28; search 23 / 4. These are snapshots; every task re-derives with the recipe §2 script.
- **Search/RAG/collections boundary is an execution decision:** `_library_rag_searched_query` and `_library_collections_saved_searches*` sit on cluster boundaries. The ownership script + recipe rules decide; a method or field that is genuinely half-owned is a BLOCKED report with specifics, not an improvisation. RAG itself is NOT in this wave.

---

### Task 1: Census anti-slack guard (task-27019) + settings-row follow-up filing

**Files:** Modify `Tests/UI/test_library_recompose_ratchet.py`; create one backlog task file via CLI.

- [ ] Read task-27019 (`backlog/tasks/`) and the size ratchet's `test_budget_is_not_left_slack_after_a_wave` (the model, incl. its tolerance rationale).
- [ ] Write the failing guard first: with the current census equal to the pin, temporarily raise the pin by more than the chosen tolerance in a scratch copy to prove the new test fails; then restore. Tolerance: mirror the size ratchet's documented looseness rationale; document the chosen number in the test docstring.
- [ ] Implement `test_census_pin_is_not_left_slack` in the recompose ratchet file; mutation-test both directions (headroom injected → fails; exact pin → passes).
- [ ] `backlog task edit 27019 -s Done` with implementation notes per repo DoD.
- [ ] File the settings-screen follow-up the spec's non-goals promised: `backlog task create "settings_screen.py needs a size-ratchet budget row" -d "Spec 2026-09-01 non-goal follow-up: the ratchet that let library_screen triple also has no settings row; settings_screen.py was 15,922 lines at the 2026-08-02 doctrine baseline." --ac "Budget row added at measured values" --ac "Mutation-checked (dummy method -> fails)"`.
- [ ] Run the recompose ratchet file + preflight; commit `test(library): census anti-slack guard (task-27019); file settings ratchet-row task`.

### Tasks 2–4: Export series (state → controller → cleanup)

Recipe §1–§6 verbatim, conversations series as the template. Wave-specific notes:
- Ownership analysis first (recipe §2 script with the wave's prefix set); expected ~12 exclusive fields → `LibraryExportState`; shared fields (e.g. `_library_export_origin_row_id` is written by rail-switch shell code — verify) stay per the ≥2-subsystems rule.
- One controller: `LibraryExportController` (`library_export_controller.py`). The existing `_open_library_export_canvas` interaction with rail/`@on` surfaces: delegators stay per canon.
- Wiring tests extend `Tests/Architecture/` with a new `test_library_export_wiring.py` following the conversations wiring-test final shape (full-cluster loop, same-name forwarding regex, state-field property coverage, `_safe_text`-style binding pins if used).
- Characterization spot-check precedes the state PR (recipe rule; Task-5-of-foundation shape) — export's `@on` surface is small; pins only for genuinely unpressed handlers.
- Cleanup lowers the pin, prunes zero-reference delegators by census, retargets tests assertions-byte-for-byte, appends blame-ignore hashes (running `git rev-parse` — never typed from memory; recipe records the incident).

### Tasks 5–7: Collections series

Same shape. Wave-specific notes:
- 28 fields / 67 methods — the biggest series so far; the move may split into TWO controllers only if the ownership analysis shows a clean seam (e.g. capture/sync vs browse-actions); otherwise one controller, one move PR. The existing `library_collections_browse_controller.py` stays untouched (foundation non-goal).
- `_library_collections_saved_searches*` fields: decide collections-vs-search ownership by consumer census; record in recipe per-subsystem table.
- `_library_collections_capture_controller` field holds an existing controller instance — it is WIRING, not state; it stays on the screen (like `_conversation_reader_controller` does).

### Tasks 8–9: Search series

Same shape, smallest. Wave-specific notes:
- 4 init-fields, at least one (`_library_search_history`) possibly exclusive; `_library_rag_searched_query` likely RAG-owned (stays for the RAG wave) — census decides.
- If the analysis shows search is too entangled with RAG to extract alone (>1/3 of its methods calling or called by rag-prefixed methods), STOP and report: the right answer may be a combined search+rag wave-3 series, and that is a controller ruling, not an implementer improvisation.

### Task 10: Wave close

- [ ] Re-measure; recipe §11 trajectory updated with wave-2 numbers; per-subsystem ownership tables appended.
- [ ] Full library xdist sweep + paired baseline against the wave-2 branch base (the foundation tip), zero branch-unique failures required.
- [ ] Final probe run recorded (pure moves: numbers unchanged within noise).

## Self-review record

- Guard-first ordering protects every later pin-lowering in this wave and beyond.
- All mechanics delegated to the recipe by reference — this plan only pins boundaries, order, and the two known entanglement decisions (saved-searches, searched-query) plus the search/RAG merge escape.
- Foundation lessons encoded: per-move pin lowering, rev-parse-not-memory, census-before-prune, characterization-before-state.
