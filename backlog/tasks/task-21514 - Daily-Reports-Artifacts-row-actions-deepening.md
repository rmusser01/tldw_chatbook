---
id: TASK-21514
title: Daily Reports - Artifacts row actions deepening
status: Done
assignee: []
created_date: '2026-08-30 00:27'
updated_date: '2026-08-30 07:07'
labels: []
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deepen the Artifacts screen Reports rows: deep-link each row to the owning watchlist's artifacts pane, in-place text preview, keep/export affordances, and a kept badge (requires a cross-DB lookup into ChaChaNotes kept_briefings - keep state is not in SubscriptionsDB, so the spec's single-join premise does not hold). Follow-up to TASK-21513 / ADR-079.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria (the what)

- [x] Each Reports row carries a " · kept" badge when (and only when) a ChaChaNotes kept_briefings row exists for that briefing id; the badge flips in place after a Keep without reopening the screen.
- [x] Pressing a row's View button previews that report in `#artifacts-detail-pane` (full `get_briefing` row fetched off the UI thread), rendering the untrusted body via `Markdown(body, hyperlinks=False)` grouped under a literal header for `complete`, and a status body for failed/empty/generating; a Clear preview control restores the previous pane.
- [x] Pressing a row's Open button posts `NavigateToScreen("watchlists_collections", screen_context={section: "artifacts", backend: "local", briefing_id: "local:briefing:{id}"})` — the same context contract the Console's watchlists operation cards post — with no changes on the Watchlists side.
- [x] Keep and Export buttons (under the reports list) act on the previewed report: Keep runs the real `keep_briefing` (origin="manual", guard claimed pre-worker, KeepRefused surfaced as a warning, created/re-keep toasts) and is disabled without a complete preview or a ChaChaNotes handle; Export pushes the vendored FileSave picker and writes `briefing_markdown_document` output through `validate_path_simple` off the event loop.
- [x] Every new Button carries a tooltip; all DB work stays off the UI thread in named workers; a failed action notifies and never crashes the app.

## Implementation Plan (the how)

1. Extend `Tests/UI/test_artifacts_screen_reports.py` first (RED): badge on/off, View preview + clear, Open deep-link context, Keep end-to-end + disabled paths, Export dialog push + direct write-path, using a file-backed ChaChaNotes DB (per `test_briefing_keep.py`'s "never `:memory:`" rule — CharactersRAGDB hands each thread its own connection).
2. Annotate `_refresh_daily_reports`'s thread worker with kept ids from one `list_kept_briefings(limit=200)` call; render " · kept" on the row label.
3. Add per-row View/Open buttons dispatched in the existing `on_button_pressed` prefix-match; View drives a generation-guarded thread worker (group `artifacts-report-preview`) that fetches `get_briefing` and merges `watchlist_name`/`kept` from the list row.
4. Render the preview in `compose_content`'s detail pane with the artifacts_pane renderable convention (literal header Text + `Markdown(..., hyperlinks=False)` in a `Group`; status bodies for non-complete); a Clear button restores the Chatbook branch.
5. Copy the Watchlists Keep trio (`keep_briefing` via `asyncio.to_thread`, in-flight guard claimed pre-worker, honest toasts, badge-refreshing `finally`) and the export pair (`FileSave` dialog + `_write_*_export_file` validate→build→write shape) against the previewed row.

ADR required: no
ADR path: backlog/decisions/079-daily-reports-surface-and-demo-seeding.md (governing ADR, linked not duplicated)
Reason: direct implementation of ADR-079's Daily Reports surface — no new schema, sync policy, provider boundary, or cross-module contract; every seam (keep service, export service, navigation-context keys, render conventions) already exists and is reused as-is.

## Implementation Notes (the PR description)

- **Approach**: all new behavior lives in `tldw_chatbook/UI/Screens/artifacts_screen.py`; every service seam is reused, none extended. The kept badge is computed inside the existing `artifacts-daily-reports` thread worker (one `list_kept_briefings(limit=200)` call when a ChaChaNotes handle exists; missing/failing handle degrades to no badges). View/Open are per-row buttons dispatched in the existing `on_button_pressed` prefix-match beside Play. The preview is a generation-guarded thread worker (group `artifacts-report-preview`) fetching `get_briefing` and merging `watchlist_name`/`kept` from the list row; `compose_content` renders it in `#artifacts-detail-pane` with precedence over the Chatbook branch plus a Clear button.
- **Seams reused verbatim (adapted only in notify target)**: `Subscriptions.briefing_keep.keep_briefing` + the Watchlists Keep trio (guard claimed in the sync handler before `run_worker`; `asyncio.to_thread`; `KeepRefused` → warning; "Kept with N scripts" / "Already kept — added N new scripts" toasts; badge-flipping refresh in `finally`); `Subscriptions.briefing_export.briefing_markdown_document` / `default_briefing_filename` + the `_push_export_briefing_dialog`/`_write_briefing_export_file` shape (vendored `FileSave`, `validate_path_simple`, off-loop write, guard cleared in `finally`); the artifacts_pane renderable convention (`Group(header Text, Markdown(body, hyperlinks=False))`, status bodies for failed/empty/generating, `strip_control_characters` on header values); the navigation-context contract keys in `Constants` (section/backend/briefing_id) posting `local:briefing:{n}` receipts, post-only.
- **Test-infra extension**: `DestinationHarness` (Tests/UI/test_destination_shells.py) now records `message.screen_context` into a parallel `seen_contexts` list (backward-compatible; `seen_routes` unchanged) so the deep-link test can assert the exact context dict.
- **Trade-off**: toast bodies go through a new `_notify` helper (`markup=False` default, `getattr` degrade) mirroring `_notify_watchlists`, because several new toasts embed app-unauthored text (KeepRefused messages, path-validation errors). Keep disables (not just refuses) without a complete preview or ChaChaNotes handle, matching the Watchlists pane's gating.
- **Files modified**: `tldw_chatbook/UI/Screens/artifacts_screen.py`, `Tests/UI/test_artifacts_screen_reports.py` (9 new tests; existing 4 untouched and green), `Tests/UI/test_destination_shells.py` (harness context capture only).
- **Verification**: `Tests/UI/test_artifacts_screen_reports.py` 14/14 green after a confirmed RED run (9 failed pre-implementation); regression: `Tests/Watchlists/test_watchlists_demo_banner.py`, `Tests/Subscriptions/test_daily_report_demo.py`, `Tests/Subscriptions/test_daily_reports_view.py`, `Tests/UI/test_ui_responsiveness_artifacts.py`, `Tests/UI/test_destination_shells.py` — all green except 3 pre-existing Library recovery-copy failures in `test_destination_shells.py` that fail identically on clean HEAD 4459776fe (verified via stash; unrelated to this change). Ruff + py_compile clean.
