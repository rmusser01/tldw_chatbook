# Phase 4.5 W+C Service Adoption

Task: `TASK-5.5`
Branch: `codex/unified-shell-phase4-wc-service`

## Goal

Turn the top-level W+C destination from static explanatory copy plus active-run follow into a service-backed snapshot that tells users whether local monitored sources and saved collection items are available and can be staged into Console.

## Implementation Summary

- Loaded local monitored sources through `watchlist_scope_service.list_watch_items(runtime_backend="local")`.
- Loaded local saved collection items through `media_reading_scope_service.list_read_it_later(mode="local")`.
- Rendered loading, available, empty, service-unavailable, and service-error states with stable selectors.
- Preserved the existing `Open current Watchlists` route and the existing latest active W+C run Console follow behavior.
- Added `Stage W+C Context in Console`, disabled until concrete local watchlist or collection context exists.
- Built `ChatHandoffPayload` from actual local watchlist and collection counts plus sample titles instead of generic placeholder copy.

## Verification

- Baseline focused command before changes: `.venv/bin/python -m pytest Tests/UI/test_destination_shells.py::test_watchlists_collections_uses_compact_title_and_clear_sections Tests/UI/test_console_live_work_handoffs.py::test_watchlists_destination_keeps_console_follow_disabled_without_active_run -q`
- Baseline focused result: `2 passed, 3 warnings in 10.84s`.
- Red command: `.venv/bin/python -m pytest Tests/UI/test_destination_shells.py::test_watchlists_collections_lists_local_snapshot_from_services Tests/UI/test_destination_shells.py::test_watchlists_collections_empty_state_disables_console_attach Tests/UI/test_destination_shells.py::test_watchlists_collections_service_failure_uses_recovery_copy Tests/UI/test_destination_shells.py::test_watchlists_collections_attach_to_console_uses_listed_context -q`
- Red result: `4 failed, 1 warning in 15.45s` because the W+C destination had no service snapshot selectors or Console context handoff.
- First green behavior command: `.venv/bin/python -m pytest Tests/UI/test_destination_shells.py::test_watchlists_collections_lists_local_snapshot_from_services Tests/UI/test_destination_shells.py::test_watchlists_collections_empty_state_disables_console_attach Tests/UI/test_destination_shells.py::test_watchlists_collections_service_failure_uses_recovery_copy Tests/UI/test_destination_shells.py::test_watchlists_collections_attach_to_console_uses_listed_context -q`
- First green behavior result: `4 passed, 1 warning in 7.04s`.
- Focused regression command after preserving existing W+C live-work behavior: `.venv/bin/python -m pytest Tests/UI/test_console_live_work_handoffs.py::test_watchlists_destination_logs_adapter_failure_and_disables_follow Tests/UI/test_console_live_work_handoffs.py::test_watchlists_destination_click_uses_item_promised_by_button_label -q`
- Focused regression result: `2 passed, 1 warning in 8.69s`.
- Final focused command: `.venv/bin/python -m pytest Tests/UI/test_destination_shells.py Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_shell_product_model_visibility.py Tests/Subscriptions/test_watchlist_scope_service.py::test_scope_service_routes_local_and_server_actions_with_watchlists_action_ids Tests/Media/test_media_reading_scope_service.py::test_scope_service_list_read_it_later_normalizes_local_saved_state -q`
- Final focused result: `116 passed, 8 warnings in 89.31s`.

## QA Walkthrough Notes

- Environment: focused Textual mounted-window tests using the repo virtualenv.
- Entry path: top navigation `W+C` destination.
- Visual check: W+C keeps the compact title, Watchlists and Collections ownership copy, existing Watchlists route, and active-run Console follow state while adding a `Local W+C snapshot` section.
- Available-state result: local service responses render watchlist and collection counts plus visible sample titles and enable `Stage W+C Context in Console`.
- Empty-state result: empty local service responses render `No local Watchlists or Collections are available yet.` and disable Console staging with add-context recovery copy.
- Service-error result: scope-service exceptions render `W+C services unavailable; retry W+C later.` and disable Console staging with retry-oriented recovery copy.
- Functional result: Console handoff stages `wc-context` with watchlist and collection counts, sample titles, runtime/source ownership metadata, and no generic placeholder body.
- Regression result: existing active-run Console follow keeps the run promised by the visible button even after the asynchronous local W+C snapshot refresh recomposes the screen.

## Residual Risk

- This slice adopts local list and context staging only; full W+C detail, create/edit/delete, import/export, feed WebSub, retry/backoff controls, alert-rule editing, and server collections feed UX remain future work.
- The walkthrough uses focused mounted-window QA, not a full clean-HOME running app session.
- Local watchlist service returns a limited list without a total count, so the UI labels watchlists as a bounded snapshot rather than a complete inventory.
