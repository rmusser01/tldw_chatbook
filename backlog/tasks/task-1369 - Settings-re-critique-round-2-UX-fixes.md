---
id: TASK-1369
title: Settings re-critique round 2 UX fixes
status: Done
assignee: []
created_date: '2026-08-05 21:11'
updated_date: '2026-08-05 21:37'
labels:
  - settings
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remaining items from the post-fix re-critique (.impeccable/critique/2026-08-05T16-56-50Z__tldw-chatbook-ui-screens-settings-screen-py.md, 30/40): (1) P2 Overview is the wrong front door - 7+ concerns on the landing card incl. a mutating sync control and ownership table behind a Loading wall; cut to 3-4 status rows with Open-category affordances and collapse/relocate the rest. (2) P2 invalid-field highlight disappears on focus (.settings-invalid-input:focus restyles to normal surface) - keep an error tint or text marker while focused. (3) P3 dotted Python module paths rendered as user-facing row content (Sync_Interop..., AppRAGSearchConfig...) violating the task-181 user-language rule - human labels in rows, module paths to tooltip/debug. (4) Theme editor polish: Apply re-themes the whole app instantly with no instant-apply label (splash viewer has one), and the 40 preset swatches are mouse-only (on(Click)) - keyboard users cannot apply presets. (5) Sync dialog polish: friendlier fallback copy when preview counts are not loaded, and avoid the confirm-resume row flicker overwriting the running status.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Overview landing shows at most 4 primary status rows each with an Open-category affordance; remaining content collapsed or relocated,Invalid fields keep a visible error indicator while focused,No dotted module paths in user-facing rows (human labels; paths in tooltip or debug expander),Theme Apply carries an instant-apply label consistent with the splash viewer and presets are keyboard-activatable,Sync dialog fallback copy reads cleanly when counts are unloaded,Relevant suites green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Restructure Overview card to <=4 status rows with Open-category buttons; fold rest behind disclosure/relocate; fold manual sync into sync summary row
2. Fix .settings-invalid-input:focus to keep error indicator; regenerate modular CSS
3. Replace dotted module paths in rows with human labels; move paths to tooltip/debug
4. Theme editor: add instant-apply hint; make preset swatches focusable + Enter/Space
5. Sync dialog: readable fallback copy; skip resume refresh while sync run worker in flight
6. Run targeted settings test suites; update task AC/notes
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
ADR required: no (presentation-only changes).

Approach: minimal presentation restructure per the re-critique; no data-loading changes.

1. Overview front door (settings_screen.py _render_overview_detail): landing card now leads with 4 primary status rows (Provider readiness, Config path/storage, Privacy, Sync summary via new _overview_sync_summary) - the first three carry Open <category> buttons (settings-overview-open-*, handled by handle_overview_open_category -> _select_category); the sync row folds in the Preview/Run manual sync controls. The 8-row server/sync/workspace/handoff table, manual sync detail rows and Switch Source / Server moved into a collapsed Collapsible (#settings-overview-sync-details); Console paste collapse, Diagnostics and the ownership table into a second collapsed Collapsible (#settings-overview-ownership-details). Deviation: the sync row's affordance is its Preview/Run controls (there is no sync category).
2. Invalid-field focus (_agentic_terminal.tcss): .settings-invalid-input:focus now keeps an error-tinted background (-status-error 28%) instead of restyling to the normal surface; modular CSS regenerated (tldw_cli_modular.tcss). Existing CSS-pinning test updated to the new contract.
3. Dotted paths: Library & RAG Save targets rows now read 'Search: search behavior and result defaults' / 'Retriever: keyword/vector retrieval and blend defaults' with the AppRAGSearchConfig paths on tooltips (_detail_row gained a tooltip kwarg); LIBRARY_RAG inspector 'Affected config' copy now 'search and retriever defaults under AppRAGSearchConfig'. The domain-contract constant and SETTINGS_SERVER_SYNC_WORKSPACE_SOURCE_CONTRACTS keep their paths - they are not user-facing (defensive/unreachable render path) and are test-pinned.
4. Theme editor (settings_theme_editor.py): added #settings-theme-apply-hint using the splash viewer's phrasing ('Apply applies immediately - no Save needed'); preset swatches are now focusable with tooltips + focus CSS (:focus border) and apply on Enter/Space via on_key -> shared _apply_preset_swatch helper. Also fixed a latent bug: str(styles.background) produced 'Color(r, g, b)' instead of hex (now normalized via Color.hex), which affected the existing click path too.
5. Sync dialog: confirm dialog falls back to 'Sync will push all pending changes.' when pending counts are Loading/Refreshing/unknown instead of interpolating them; new _manual_sync_run_in_flight flag (set in the confirm callback, cleared in the run worker's finally) makes on_screen_resume skip its sync-rows refresh while a run is in flight, ending the confirm->resume flicker that overwrote the 'running' rows.

Tests: new tests for all five items in Tests/UI/test_settings_configuration_hub.py and Tests/UI/test_settings_theme_editor.py. Suites green: test_settings_configuration_hub.py (254), test_settings_theme_editor.py (12), category_sweep/footer_hints/narrow_layout/save_commit_models/model_catalog_layout (30), test_settings_library_rag_defaults.py (18), settings subsets of test_destination_shells.py (12) and test_destination_visual_parity_correction.py (10). Ruff findings on touched files are pre-existing on HEAD.

Files: tldw_chatbook/UI/Screens/settings_screen.py, tldw_chatbook/Widgets/settings_theme_editor.py, tldw_chatbook/css/components/_agentic_terminal.tcss, tldw_chatbook/css/components/_settings_splash_theme.tcss, tldw_chatbook/css/tldw_cli_modular.tcss (regenerated), Tests/UI/test_settings_configuration_hub.py, Tests/UI/test_settings_theme_editor.py

Follow-up (code review, CHANGES NEEDED light - all 5 addressed): (1) Overview disclosure expanded/collapsed state now persists across recompose=True rebuilds via instance flags (_overview_sync_details_collapsed / _overview_ownership_details_collapsed) fed by Collapsible.Toggled handlers and read at compose time - an expanded disclosure no longer snaps shut when a sync row changes mid-run; regression test test_settings_overview_disclosures_stay_expanded_across_sync_row_recompose added. (2) _manual_sync_run_in_flight clearing is now gated by a monotonic _manual_sync_run_token so a cancelled (exclusive-group) stale worker's finally cannot clear a newer run's flag. (3) The confirm callback wraps the _manual_sync_run_worker() call so a synchronous raise clears the flag and shows a 'could not be started' row instead of stranding the flag True. (4) First-paint sync summary reads 'Loading sync status...' instead of 'loading; pending outgoing: unknown'. (5) Theme editor copy: swatch hint mentions Enter or Space; apply hint now 'Apply takes effect immediately - no Save needed' (instant-apply vocabulary kept). Verification: test_settings_configuration_hub.py (256), test_settings_theme_editor.py (12), test_settings_category_sweep.py + test_settings_model_catalog_layout.py - all green (one transient focus-race flake in the full-suite run passed standalone and on rerun).
<!-- SECTION:NOTES:END -->
