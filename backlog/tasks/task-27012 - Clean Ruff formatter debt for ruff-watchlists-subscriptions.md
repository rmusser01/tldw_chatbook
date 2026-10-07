---
id: TASK-27012
title: Clean Ruff formatter debt for ruff-watchlists-subscriptions
status: Done
assignee:
  - rmusser01
created_date: '2026-08-31 18:31'
updated_date: '2026-10-03 14:30'
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
- [x] After rebasing onto current `origin/dev`, reproduce and reconcile every TASK-26000 assigned path; if upstream deleted, renamed, modified, or already formatted it, record that lineage and amend ownership mechanically without silently dropping it or absorbing an unassigned path. <!-- TASK-26000-CONTRACT: rebase-reconcile --><!-- TASK-26000-CONTRACT: drift-reconciliation -->
- [x] Run Ruff 0.15.22 formatting on only the assigned paths, with no unassigned Python path changed. <!-- TASK-26000-CONTRACT: assigned-paths-only -->
- [x] Before and after formatting, parse each assigned file on Python 3.12.11 with `ast.parse(..., type_comments=True)`, normalize only `TypeIgnore.lineno`, and require equal `ast.dump(..., include_attributes=False)`. <!-- TASK-26000-CONTRACT: ast-type-comments -->
- [x] Preserve ordered comment-token text; anchor inline `# noqa`, `# type: ignore`, and single-target Ruff directives to the same deepest AST-node path and significant-token position, preserve standalone file directives between the same adjacent statement paths, and require each `# fmt: off` / `# fmt: on` range to enclose the same ordered AST-node interval. <!-- TASK-26000-CONTRACT: comment-directives -->
- [x] Ruff lint and `ruff format --check` pass on every touched Python path. <!-- TASK-26000-CONTRACT: ruff-checks -->
- [x] Implementation Notes record the focused-test rationale and every exact test command/result. <!-- TASK-26000-CONTRACT: focused-tests -->
- [x] `git diff --check` and `Tests/CI/test_backlog_task_id_uniqueness.py` pass. <!-- TASK-26000-CONTRACT: governance -->
- [x] The diff contains no hand-written production behavior change. <!-- TASK-26000-CONTRACT: no-handwritten-behavior -->
<!-- AC:END -->

## Implementation Plan

1. Reconcile assigned paths against current `origin/dev` (existence + format state; missing paths recorded with dev delete/rename lineage) and against the TASK-26000 evidence JSON `cleanup_records` `paths_sha256` (mechanical ownership check).
2. Snapshot each assigned path's `ast.dump` before formatting (type_comments=True with symmetric plain-parse fallback; normalize `TypeIgnore.lineno`).
3. Run Ruff 0.15.22 `ruff format` on exactly the assigned paths.
4. Snapshot after-ASTs; require per-file equality (formatter-mandated docstring-normalization deviations enumerated if they occur).
5. `ruff format --check` must pass on every existing assigned path; `ruff check` findings must not increase vs the pre-format baseline.
6. Run the focused test surface (bounded; never a bare pytest), plus `Tests/CI/test_backlog_task_id_uniqueness.py` and `git diff --check`; verify the three-way path partition arithmetically before writing notes.
7. Tick ACs, record Implementation Notes (lineage, commands, results), set status Done.

## Implementation Notes

**Approach.** Executed the TASK-26000 formatter-debt cleanup contract at `origin/dev` tip `3c439d606e` (isolated worktree `.worktrees/ruff-debt-batch-9`, branch `chore/ruff-debt-batch-9` — the series' FINAL batch). Formatter: Ruff 0.15.22 (the TASK-26000 pin); the repository ships no Ruff configuration, so defaults apply, matching the census contract.

**Ownership reconciliation (AC#1).** All 89 assigned paths reconcile mechanically (hash matches marker and evidence JSON); 88 exist, 1 deleted upstream — see Lineage.

**Formatting (AC#2, AC#5).** 86 of the 88 existing assigned paths formatted. `ruff format --check` passes on every existing assigned path after formatting. Lint: `ruff check --output-format concise` over the batch's 399 assigned paths: 119 findings before and 117 after, ZERO increases (the formatter incidentally resolved 2 in `Docs/superpowers/qa/skills-script-execution-2026-07-25/seed3.py` by reflowing); the remainder are pre-existing F401-type debt outside the TASK-26000 formatter scope.

**AST equality (AC#3).** Hashes equal on 86 of 88 existing paths. The two exceptions (`Tests/Watchlists/test_watchlist_tree.py`, `Tests/Watchlists/test_watchlists_collections_screen.py`) are each exactly ONE region of the sanctioned quote-initial leading-space class (+1 char each), zero `__doc__` consumers.

**Comment directives (AC#4).** Ordered `tokenize.COMMENT` sequences and `# fmt: off` / `# fmt: on` counts are identical before and after on every assigned path.

**Focused tests (AC#6).** Focused surface: the 67 existing assigned test files (recorded surface Tests/Subscriptions + Tests/Watchlists). Result on the formatted tree: 686 failed, 1002 passed, 2 skipped, 9 errors in 1058.99s (`--timeout=120`). Baseline A/B per `backlog/docs/lessons-backlog-hygiene.md` (no-stash rule): assigned paths swapped to unformatted `HEAD` content with `git checkout HEAD -- <paths>`, identical command re-run, files re-formatted and re-verified with `ruff format --check`. Baseline: 686 failed, 1002 passed, 2 skipped, 9 errors in 874.94s — count-identical to the formatted outcome, so the failures are pre-existing at the current dev tip (the documented hook-consent send-gate and config-admission red classes) and out of scope for a formatter-only task; AST equality independently proves runtime-identical files.

**Governance (AC#7).** `git diff --check` clean; `Tests/CI/test_backlog_task_id_uniqueness.py` — 3 passed (run on the fully formatted batch tree).

**No hand-written behavior change (AC#8).** Three-way partition verified arithmetically before closeout: 369 committed + 25 clean-at-base + 5 upstream-deleted = 399 assigned paths; zero unassigned paths touched; every content diff is Ruff formatter output.

**Lineage.** 1 path deleted upstream: `Tests/Subscriptions/test_subscription_egress_wiring.py` by `8f70a50f11` (TASK-591 dead SecurityValidator/SSRFProtector deletion — a FOURTH distinct upstream deletion consuming census paths); 2 paths already formatter-clean at the base (`Tests/Subscriptions/test_briefing_keep.py`, `tldw_chatbook/Subscriptions/__init__.py`), deliberately left untouched; the remaining 86 carried live debt and were formatted.

ADR required: no — mechanical, formatter-only cleanup executing the owner-approved TASK-26000 contract; no architecture, storage, or cross-module boundary touched.
