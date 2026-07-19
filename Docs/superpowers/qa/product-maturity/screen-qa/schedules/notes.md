# Schedules Workbench Screenshot QA

## Branch / commit under test

- Branch: `feature/scheduling-module-screen`
- Commit: `1204e8b3` (`fix(scheduling): remove remaining TaskInspector CSS duplication and test race`)
- Previous commits in this phase:
  - `63d8a553` — `fix(scheduling): address code-quality review blockers for UX polish`
  - `2967ff49` — `fix(scheduling): address residual UX polish upgrades`
  - `5ba331c3` — `fix(scheduling): address UX polish review feedback`
  - `9afdd539` — `feat(scheduling): UX polish for Schedules workbench`
  - `c18439b0` — `feat(scheduling): route schedules destination to new workbench`
  - `c39f24de` — `fix(scheduling): address Console-follow seam review feedback`
  - `d9919bce` — `feat(scheduling): preserve Console-follow seam and update destination tests`
  - `66fb5513` — `test(scheduling): update navigation test for SchedulesWorkbench route`
  - `a14d9992` — `docs(scheduling): add schedules workbench screenshot QA evidence`

## Viewport size

- Baseline screenshot: 1600 × 900 px
- Final screenshot: 1600 × 900 px
- Terminal content rendered at the default Textual screenshot size.

## Launch method

Because the full `tldw_chatbook.app` requires user config, API keys, and an existing database, these screenshots were captured from minimal wrapper apps that mount the relevant screen classes directly. This isolates the Schedules destination chrome and content while preserving the real screen implementations used in production.

- Baseline: `BaselineApp` mounts legacy `tldw_chatbook.UI.Screens.schedules_screen.SchedulesScreen`
- Final: `ScreenshotApp` mounts new `tldw_chatbook.UI.Screens.scheduling.schedules_workbench.SchedulesWorkbench`

Capture pipeline:

1. `textual run --screenshot 2 --screenshot-path ... <app.py>` produced a faithful SVG rendering of the running terminal UI.
2. The SVG was converted to PNG via macOS `sips` (Chrome headless was unavailable in this runtime).
3. Resulting PNGs are stored below.

## Screenshot paths

- `baseline-2026-07-18T22-43-17.png` — legacy `SchedulesScreen` empty state
- `final-2026-07-18T23-23-00.png` — polished `SchedulesWorkbench` with populated reminder queue

## Baseline defects found

The baseline (legacy `SchedulesScreen`) showed:

- Filter counts (`Next Run 0`, `Paused 0`, `Failed 0`, `Retry 0`, `History 0`) dominate the left pane and do not match the new data-first workbench model.
- Action buttons (`Retry run`, `Pause run`) are always present but disabled in the empty state; their purpose is unclear without an active run.
- Console-follow recovery copy is verbose and split across multiple panes.

## UX polish pass (`9afdd539`)

The final `SchedulesWorkbench` addresses the baseline defects and adds a senior UX/HCI polish pass:

- **Human-readable labels**: `schedule_kind` renders as `Recurring` / `One-time`; `last_status` renders as capitalized text (`Waiting`, `Running`, `Paused`, etc.); cron expressions summarize to `Daily at 09:00 UTC` / `Weekly on Monday at 10:00 UTC`; next-run datetimes append the timezone (`2026-07-20 09:00 UTC`).
- **Status color encoding**: Status badges appear in both the task list `DataTable` and the detail pane, using design-system semantics (primary/success/warning/error/muted).
- **Detail pane reorganization**: Metadata is grouped compactly at the top; lifecycle actions (`Enable`, `Disable`, `Delete`) sit directly above the `Follow in Console` button; the button is disabled with an honest tooltip when no active run is available.
- **Delete confirmation**: The Delete button opens a `ModalScreen` confirmation dialog before any deletion.
- **Empty state improvements**: Friendly copy guides users to select a task or press `Ctrl+C` to create one; an empty queue explains how to create a first reminder.
- **Inspector cleanup**: Duplicate status/next-run lines removed; focus is on sync state (`version 0 (local)` / `version N (server <id>)`), last run, owner, and a conflict card.
- **TCSS updates**: Badge, header, empty-state, lifecycle-row, and follow-button styles added.

## UX polish fixes (`5ba331c3`)

This follow-up commit addresses blocking code-quality issues found in the UX polish review:

- **Ctrl+D shortcut**: Now triggers deletion for the currently selected task using the same confirmation modal flow as the Delete button.
- **Disabled Console-follow guard**: The handler returns early when the `Follow in Console` button is disabled.
- **`load_tasks()` error handling**: Service failures now surface an error notification and consistent empty-state copy.
- **Shared delete dialog**: The custom `DeleteTaskModal` was replaced with the existing `DeleteConfirmationDialog` widget.
- **Duplicate CSS removed**: Status badge styles now live only in `_scheduling.tcss`.
- **Expanded test coverage**: Added tests for delete modal, empty/no-selection states, disabled Console-follow, status badge classes, inspector metadata, conflict card, cron humanization, and the service-error path.

Visual changes are subtle (behavior and code-quality fixes), so the existing final screenshot remains representative.

## Residual UX polish upgrades (`2967ff49`)

This commit addresses minor issues identified during a final review pass over the workbench:

- **Dedicated status badge classes**: `TaskStatus.COMPLETED`, `FOUND_RESULTS`, `ARCHIVED`, and `MISSED` now map to their own CSS classes (`completed`, `found-results`, `archived`, `missed`) instead of aliasing to `running` / `needs-attention` / `disabled`. This lets the design system style each status independently.
- **Public delete request API**: `TaskDetail._request_delete()` was renamed to `request_delete()` and the workbench `Ctrl+D` binding now calls the public method.
- **Stale-row cleanup on service error**: When `list_reminders()` fails, the workbench now clears the `DataTable` and the internal task list so stale rows do not linger after a refresh failure.
- **Regression tests**: Added coverage for the dedicated badge-class mapping and for clearing stale rows on a service error.

## Code-quality review blockers (`63d8a553`)

This commit addresses blocking issues raised by a code-quality review of the workbench:

- **Missing `Text` import**: `schedules_workbench.py` now imports `Text` from `rich.text` for its `DataTable` row annotation.
- **Type annotations**: `app_instance` and `_latest_console_follow_item_from_adapter` now carry type hints.
- **Async console-follow refresh**: `_refresh_console_context` is now a plain async helper awaited by `load_tasks`. It handles both sync and async service methods (`build_dashboard_input`, `list_reading_digest_outputs`) so it no longer silently receives coroutines inside a `thread=True` worker.
- **Duplicate CSS removed**: Styles for headers, lifecycle rows, follow button, empty state, and conflict card were removed from `TaskDetail.DEFAULT_CSS` and `TaskInspector.DEFAULT_CSS`; they now live only in `_scheduling.tcss`.
- **Delete end-to-end tests**: Added tests for the confirmation → `DeleteTaskRequested` → `delete_reminder` success path and the delete-reminder failure path.

## Final CSS/test cleanup (`1204e8b3`)

This commit addresses the last code-quality review finding:

- **Remove `TaskInspector.DEFAULT_CSS`**: All inspector styles (metadata, labels, values, conflict card) now live exclusively in `_scheduling.tcss`.
- **Remove unused import**: `Label` is no longer imported in `task_detail.py`.
- **Robust delete tests**: Delete-flow tests now use `pilot.app.workers.wait_for_complete()` instead of racing the background worker with an explicit `load_tasks()` call.

## User approval status

✅ Approved by user on 2026-07-19. Task 4.8a is complete.

## Tests run

```bash
.venv/bin/python -m pytest Tests/UI/test_schedules_workbench.py Tests/UI/test_destination_shells.py -v
```

Result: **122 passed, 1 skipped** (the single skip is an unrelated Personas tooltip audit).

## Residual risks

- The screenshots were generated from SVG renderings of the running app, then rasterized to PNG. They faithfully represent the terminal layout but are not direct pixel captures of a physical monitor.
- The full production app was not launched, so splash-screen, config-error, or first-run states are not exercised here.
- Compact-size layout was not explicitly verified; the workbench uses `fr` widths and should reflow, but a dedicated compact screenshot is recommended before the screen is marked fully mature.
