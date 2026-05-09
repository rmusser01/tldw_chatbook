# Library Screenshot QA Notes

Date: 2026-05-09
Branch: `codex/screen-qa-library`
Backlog task: TASK-14.3
Commit: pending
Screen: Library
Viewport: 2048x1280 via Chromium / textual-web
Launch method: `tldw-serve --host 127.0.0.1 --port 8816`
Screenshot method: Playwright browser screenshot against `http://127.0.0.1:8816`
Fallback reason: None

## Baseline Screenshot

- Path: `Docs/superpowers/qa/product-maturity/screen-qa/library/baseline-2026-05-09-playwright-library.png`
- Defects: Library could remain in an indefinite loading state, empty local Library data still reported `Ready`, the source browser lacked a direct Collections affordance, and Search/RAG source-browser action exited to the legacy route instead of staying in Library-native Search/RAG mode.

## Interaction Smoke

- Goal: Verify Library Source Browser Search/RAG stays inside the Library workbench.
- Steps: Launched the app through textual-web, opened Library, clicked the Source Browser `Search/RAG` action, and captured the resulting screen.
- Result: Library remained selected and mounted the in-place Search/RAG panel with blocked empty-source recovery copy.
- Path: `Docs/superpowers/qa/product-maturity/screen-qa/library/interaction-2026-05-09-playwright-library-search-rag-mode.png`

## Fixes

- Summary: Added off-main-thread Library source snapshot loading with a mounted fallback timeout, stable empty-state next-action copy, terminal workbench pane framing, direct Source Browser `Collections` mode switching, direct Source Browser `Search/RAG` mode switching, and a contextual right-pane inspector that shows an empty `Inspector` state until a source/evidence item is selected.

## Final Screenshot

- Path: `Docs/superpowers/qa/product-maturity/screen-qa/library/review-2026-05-09-playwright-library-contextual-inspector.png`
- User approval: approved in Codex chat on 2026-05-09

## Verification

- Commands:
  - `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_destination_visual_parity_correction.py::test_library_mode_strip_is_compact_and_workbench_visible Tests/UI/test_destination_visual_parity_correction.py::test_library_mode_strip_keeps_all_mode_chips_visible Tests/UI/test_destination_visual_parity_correction.py::test_library_workbench_renders_terminal_borders Tests/UI/test_destination_visual_parity_correction.py::test_library_empty_state_reports_empty_with_next_action Tests/UI/test_destination_visual_parity_correction.py::test_library_inspector_uses_empty_state_until_item_selected Tests/UI/test_destination_visual_parity_correction.py::test_library_source_browser_collections_action_switches_to_collections_mode Tests/UI/test_destination_visual_parity_correction.py::test_library_source_browser_search_action_switches_to_search_mode Tests/UI/test_destination_visual_parity_correction.py::test_library_source_snapshot_timeout_handles_blocking_async_services Tests/UI/test_destination_visual_parity_correction.py::test_library_source_snapshot_times_out_to_stable_error Tests/UI/test_destination_visual_parity_correction.py::test_library_loading_state_fails_safe_when_snapshot_never_applies Tests/UI/test_product_maturity_phase3_library_contract_layout.py Tests/UI/test_product_maturity_gate16_library_search_rag.py --tb=short`
  - `git diff --check`
- Results: 26 focused Library visual/state, Library contract, and Gate 1.6 Search/RAG tests passed; diff hygiene passed. Pytest emitted the existing requests dependency warning.
- Note: A broad full-file visual sweep also surfaced two existing Console compact-viewport failures unrelated to this Library change; they were not addressed in this Library PR candidate.

## Residual Risks

- PR merge is still pending; do not start the next screen unless the user explicitly overrides.
- Vertical pane resizing is feasible but should be a separate cross-screen workbench task rather than added to this Library polish PR.
