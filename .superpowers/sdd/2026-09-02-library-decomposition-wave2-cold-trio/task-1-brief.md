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

