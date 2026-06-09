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

## 2026-06-09 Content Hub Landing Page

- Branch: `codex/console-workspace-server-readiness-v3`
- Backlog task: TASK-89.2
- Screen: Library default route / content hub landing page
- Screenshot method: CDP/Playwright browser screenshot against local textual-web.
- Approved screenshot: `Docs/superpowers/qa/product-maturity/screen-qa/library/content-hub-buttons-spaced-cdp-2026-06-09.png`
- User approval: approved in Codex chat on 2026-06-09 after the left-column Library module buttons were made taller and visually separated.

## 2026-06-09 Verification

- Commands:
  - `python -m pytest -q Tests/UI/test_library_content_hub.py Tests/UI/test_destination_shells.py -k library Tests/UI/test_product_maturity_phase3_library_contract_layout.py Tests/UI/test_post_release_workspaces_library_depth.py --tb=short`
  - `python -m pytest -q Tests/UI/test_destination_visual_parity_correction.py -k library --tb=short`
  - `python tldw_chatbook/css/build_css.py`
  - `git diff --check`
- Results: Focused Library hub, destination shell, layout contract, workspace-depth, and visual-parity regressions passed. CSS build completed with the existing optional missing `features/_evaluation_v2.tcss` warning.

## 2026-06-09 Conversations Browser

- Branch: `codex/console-workspace-server-readiness-v3`
- Backlog task: TASK-89.3
- Screen: Library Conversations route / saved conversation browser
- Screenshot method: CDP/Playwright browser screenshot against local textual-web at `http://127.0.0.1:8901/?fontsize=12`.
- Approved screenshot: `Docs/superpowers/qa/product-maturity/screen-qa/library/conversations-browser-polish-cdp-2026-06-09.png`
- User approval: approved in Codex chat on 2026-06-09 via `continue` after the polished screenshot was presented.

## 2026-06-09 Conversations Verification

- Commands:
  - `python -m pytest -q Tests/UI/test_library_content_hub.py::test_library_conversations_action_opens_native_browser_without_legacy_exit Tests/UI/test_library_content_hub.py::test_library_conversations_selection_shows_metadata_and_handoff_actions Tests/UI/test_library_content_hub.py::test_library_conversations_empty_state_is_honest_and_blocks_actions Tests/UI/test_library_content_hub.py::test_library_conversation_use_as_source_hands_off_selected_conversation --tb=short`
  - `python -m pytest -q Tests/UI/test_library_content_hub.py Tests/UI/test_destination_shells.py -k library Tests/UI/test_product_maturity_phase3_library_contract_layout.py Tests/UI/test_post_release_workspaces_library_depth.py --tb=short`
  - `python -m pytest -q Tests/UI/test_destination_visual_parity_correction.py -k library --tb=short`
  - `git diff --check`
- Results: 4 focused Conversations tests passed, 34 broader Library-focused tests passed, 17 Library visual parity tests passed, and diff hygiene passed. Pytest emitted the existing requests dependency warning.

## 2026-06-09 Import/Export Workflow

- Branch: `codex/console-workspace-server-readiness-v3`
- Backlog task: TASK-89.4
- Screen: Library Import/Export route / source acquisition handoff workflow
- Screenshot method: CDP/Playwright browser screenshot against local textual-web at `http://127.0.0.1:8901/?fontsize=12`.
- Approved screenshot: `Docs/superpowers/qa/product-maturity/screen-qa/library/import-export-workflow-cdp-2026-06-09.png`
- User approval: approved in Codex chat on 2026-06-09 after the actual rendered screenshot was presented.

## 2026-06-09 Import/Export Verification

- Commands:
  - `python -m pytest -q Tests/UI/test_library_content_hub.py::test_library_import_export_opens_native_workflow_with_clear_boundaries Tests/UI/test_library_content_hub.py::test_library_import_export_dedicated_actions_emit_handoff_routes Tests/UI/test_destination_shells.py::test_library_import_export_action_switches_to_native_mode_without_route_handoff Tests/UI/test_destination_shells.py::test_library_import_export_dedicated_import_action_emits_ingest_route --tb=short`
  - `python -m pytest -q Tests/UI/test_library_content_hub.py Tests/UI/test_destination_shells.py -k library Tests/UI/test_product_maturity_phase3_library_contract_layout.py Tests/UI/test_post_release_workspaces_library_depth.py --tb=short`
  - `python -m pytest -q Tests/UI/test_destination_visual_parity_correction.py -k library --tb=short`
  - `git diff --check`
- Results: 4 focused Import/Export regressions passed, 37 broader Library-focused tests passed, 17 Library visual parity tests passed, and diff hygiene passed. Pytest emitted the existing requests dependency warning.

## 2026-06-09 Collections Workflow

- Branch: `codex/console-workspace-server-readiness-v3`
- Backlog task: TASK-89.5
- Screen: Library Collections route / reusable source-set workflow
- Screenshot method: CDP/Playwright browser screenshot against local textual-web at `http://127.0.0.1:8901`.
- Approved screenshot: `Docs/superpowers/qa/product-maturity/screen-qa/library/collections-workflow-cdp-2026-06-09.png`
- User approval: approved in Codex chat on 2026-06-09 after the actual rendered selected Collection screenshot was presented.

## 2026-06-09 Collections Verification

- Commands:
  - `python -m pytest -q Tests/UI/test_library_content_hub.py::test_library_collections_selection_explains_membership_workspace_and_actions Tests/UI/test_library_content_hub.py::test_library_collections_empty_state_keeps_global_browse_rule_and_blocks_wip_actions --tb=short`
  - `python -m pytest -q Tests/UI/test_library_content_hub.py Tests/UI/test_product_maturity_phase39_library_collections.py Tests/UI/test_post_release_workspaces_library_depth.py Tests/UI/test_product_maturity_phase3_library_contract_layout.py --tb=short`
  - `python -m pytest -q Tests/UI/test_destination_visual_parity_correction.py -k library --tb=short`
  - `git diff --check`
- Results: 2 focused Collections regressions passed after red/green, 33 broader Library/Collections/workspace tests passed, 17 Library visual parity tests passed, and diff hygiene passed. Pytest emitted the existing requests dependency warning.

## 2026-06-09 Compact Source Column

- Branch: `codex/console-workspace-server-readiness-v3`
- Backlog task: TASK-89.9
- Screen: Library content hub / source browser column
- Screenshot method: CDP/Playwright browser screenshot against local textual-web at `http://127.0.0.1:8901/?fontsize=12`.
- Approved screenshot: `Docs/superpowers/qa/product-maturity/screen-qa/library/source-column-compact-cdp-2026-06-09.png`
- User approval: approved in Codex chat on 2026-06-09 after the actual rendered Library screenshot was presented.

## 2026-06-09 Compact Source Column Verification

- Commands:
  - `python -m pytest -q Tests/UI/test_destination_visual_parity_correction.py::test_library_source_browser_stays_content_fit_at_wide_viewport --tb=short`
  - `python -m pytest -q Tests/UI/test_destination_visual_parity_correction.py -k library --tb=short`
  - `python -m pytest -q Tests/UI/test_library_content_hub.py::test_library_hub_module_actions_are_visually_separated --tb=short`
  - `python tldw_chatbook/css/build_css.py`
  - `git diff --check`
- Results: Focused compact-column regression, Library visual parity, source-action spacing regression, CSS build, and diff hygiene passed. Pytest emitted the existing requests dependency warning; CSS build emitted the existing optional missing `features/_evaluation_v2.tcss` warning.
