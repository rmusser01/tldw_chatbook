---
id: TASK-27012
title: Clean Ruff formatter debt for ruff-watchlists-subscriptions
status: To Do
assignee: []
created_date: '2026-08-31 18:31'
updated_date: '2026-08-31 18:31'
labels:
  - maintenance
  - formatting
  - quality
dependencies:
  - TASK-26000
references:
  - Docs/superpowers/specs/2026-08-30-task-26000-ruff-formatter-debt-design.md
  - Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json
priority: medium
---

<!-- TASK-26000-BATCH: ruff-watchlists-subscriptions -->
<!-- TASK-26000-PATHS-SHA256: c801621b78449067be80db86d379f886144f34af30d7d7aca3bad7d0a5e4e33c -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-watchlists-subscriptions` Ruff formatter batch at the owner boundary recorded as: Watchlists/subscriptions services and direct tests.. The focused test surface recorded by TASK-26000 is `["Tests/Subscriptions", "Tests/Watchlists"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/Subscriptions/test_app_watchlists_db_wiring.py",
  "Tests/Subscriptions/test_briefing_audio_db.py",
  "Tests/Subscriptions/test_briefing_audio_pipeline.py",
  "Tests/Subscriptions/test_briefing_audio_synthesis.py",
  "Tests/Subscriptions/test_briefing_cadence_db.py",
  "Tests/Subscriptions/test_briefing_cast.py",
  "Tests/Subscriptions/test_briefing_export_markdown.py",
  "Tests/Subscriptions/test_briefing_feed.py",
  "Tests/Subscriptions/test_briefing_feed_export.py",
  "Tests/Subscriptions/test_briefing_feed_query.py",
  "Tests/Subscriptions/test_briefing_keep.py",
  "Tests/Subscriptions/test_briefing_presets_db.py",
  "Tests/Subscriptions/test_briefing_selection.py",
  "Tests/Subscriptions/test_briefing_service.py",
  "Tests/Subscriptions/test_daily_report_demo.py",
  "Tests/Subscriptions/test_daily_reports_view.py",
  "Tests/Subscriptions/test_feed_server.py",
  "Tests/Subscriptions/test_fts_backfill.py",
  "Tests/Subscriptions/test_html_text.py",
  "Tests/Subscriptions/test_item_dates.py",
  "Tests/Subscriptions/test_item_persist.py",
  "Tests/Subscriptions/test_local_watchlists_service.py",
  "Tests/Subscriptions/test_site_config_manager.py",
  "Tests/Subscriptions/test_subscription_egress_wiring.py",
  "Tests/Subscriptions/test_watchlist_bundle_service.py",
  "Tests/Subscriptions/test_watchlist_check_now_source_id.py",
  "Tests/Subscriptions/test_watchlist_content_alert_service.py",
  "Tests/Subscriptions/test_watchlist_content_kind_producer.py",
  "Tests/Subscriptions/test_watchlist_failure.py",
  "Tests/Subscriptions/test_watchlist_feed_api_in_flight_guard.py",
  "Tests/Subscriptions/test_watchlist_filter_service.py",
  "Tests/Subscriptions/test_watchlist_noise_not_volume.py",
  "Tests/Subscriptions/test_watchlist_normalizers.py",
  "Tests/Subscriptions/test_watchlist_opml_entity_expansion.py",
  "Tests/Subscriptions/test_watchlist_opml_service.py",
  "Tests/Subscriptions/test_watchlist_preview_service.py",
  "Tests/Subscriptions/test_watchlist_scope_service.py",
  "Tests/Subscriptions/test_watchlist_snapshot_pruning.py",
  "Tests/Subscriptions/test_watchlists_db_instance_and_off_loop.py",
  "Tests/Subscriptions/test_watchlists_operation_coordinator.py",
  "Tests/Subscriptions/test_watchlists_service_no_blocking_db_io.py",
  "Tests/Subscriptions/test_watchlists_service_off_loop.py",
  "Tests/Watchlists/test_kept_briefings_modal.py",
  "Tests/Watchlists/test_no_side_effecting_predicates.py",
  "Tests/Watchlists/test_reader_item_snapshot.py",
  "Tests/Watchlists/test_region_layout.py",
  "Tests/Watchlists/test_region_layout_store.py",
  "Tests/Watchlists/test_snapshot_view_modal.py",
  "Tests/Watchlists/test_startup_reconcile_scheduler_race.py",
  "Tests/Watchlists/test_watchlist_scope_service.py",
  "Tests/Watchlists/test_watchlist_tree.py",
  "Tests/Watchlists/test_watchlists_artifacts_pane.py",
  "Tests/Watchlists/test_watchlists_artifacts_refresh_states.py",
  "Tests/Watchlists/test_watchlists_artifacts_script_selection_in_place.py",
  "Tests/Watchlists/test_watchlists_backend_controller.py",
  "Tests/Watchlists/test_watchlists_briefing_presets_ui.py",
  "Tests/Watchlists/test_watchlists_bulk_source_authoring.py",
  "Tests/Watchlists/test_watchlists_cold_open_layout.py",
  "Tests/Watchlists/test_watchlists_collections_screen.py",
  "Tests/Watchlists/test_watchlists_demo_banner.py",
  "Tests/Watchlists/test_watchlists_items_pane.py",
  "Tests/Watchlists/test_watchlists_layout_hysteresis_probe.py",
  "Tests/Watchlists/test_watchlists_notifications_pane.py",
  "Tests/Watchlists/test_watchlists_overview_pane.py",
  "Tests/Watchlists/test_watchlists_pagination.py",
  "Tests/Watchlists/test_watchlists_responsive_layout.py",
  "Tests/Watchlists/test_watchlists_scoped_rebuilds.py",
  "Tests/Watchlists/test_watchlists_sources_pane.py",
  "Tests/Watchlists/test_watchlists_workbench.py",
  "tldw_chatbook/Subscriptions/__init__.py",
  "tldw_chatbook/Subscriptions/baseline_manager.py",
  "tldw_chatbook/Subscriptions/briefing_export.py",
  "tldw_chatbook/Subscriptions/briefing_selection.py",
  "tldw_chatbook/Subscriptions/briefing_service.py",
  "tldw_chatbook/Subscriptions/briefing_voices.py",
  "tldw_chatbook/Subscriptions/daily_report_demo.py",
  "tldw_chatbook/Subscriptions/feed_server.py",
  "tldw_chatbook/Subscriptions/fts_backfill.py",
  "tldw_chatbook/Subscriptions/html_text.py",
  "tldw_chatbook/Subscriptions/local_watchlists_service.py",
  "tldw_chatbook/Subscriptions/watchlist_bundle_service.py",
  "tldw_chatbook/Subscriptions/watchlist_content_alert_service.py",
  "tldw_chatbook/Subscriptions/watchlist_failure.py",
  "tldw_chatbook/Subscriptions/watchlist_filter_service.py",
  "tldw_chatbook/Subscriptions/watchlist_normalizers.py",
  "tldw_chatbook/Subscriptions/watchlist_opml_service.py",
  "tldw_chatbook/Subscriptions/watchlist_preview_service.py",
  "tldw_chatbook/Subscriptions/watchlist_scope_service.py",
  "tldw_chatbook/Subscriptions/watchlists_operation_coordinator.py"
]
```

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] After rebasing onto current `origin/dev`, reproduce and reconcile every TASK-26000 assigned path; if upstream deleted, renamed, modified, or already formatted it, record that lineage and amend ownership mechanically without silently dropping it or absorbing an unassigned path. <!-- TASK-26000-CONTRACT: rebase-reconcile --><!-- TASK-26000-CONTRACT: drift-reconciliation -->
- [ ] Run Ruff 0.15.22 formatting on only the assigned paths, with no unassigned Python path changed. <!-- TASK-26000-CONTRACT: assigned-paths-only -->
- [ ] Before and after formatting, parse each assigned file on Python 3.12.11 with `ast.parse(..., type_comments=True)`, normalize only `TypeIgnore.lineno`, and require equal `ast.dump(..., include_attributes=False)`. <!-- TASK-26000-CONTRACT: ast-type-comments -->
- [ ] Preserve ordered comment-token text; anchor inline `# noqa`, `# type: ignore`, and single-target Ruff directives to the same deepest AST-node path and significant-token position, preserve standalone file directives between the same adjacent statement paths, and require each `# fmt: off` / `# fmt: on` range to enclose the same ordered AST-node interval. <!-- TASK-26000-CONTRACT: comment-directives -->
- [ ] Ruff lint and `ruff format --check` pass on every touched Python path. <!-- TASK-26000-CONTRACT: ruff-checks -->
- [ ] Implementation Notes record the focused-test rationale and every exact test command/result. <!-- TASK-26000-CONTRACT: focused-tests -->
- [ ] `git diff --check` and `Tests/CI/test_backlog_task_id_uniqueness.py` pass. <!-- TASK-26000-CONTRACT: governance -->
- [ ] The diff contains no hand-written production behavior change. <!-- TASK-26000-CONTRACT: no-handwritten-behavior -->
<!-- AC:END -->
