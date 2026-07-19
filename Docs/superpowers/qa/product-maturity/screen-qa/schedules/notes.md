# Schedules Workbench Screenshot QA

## Branch / commit under test

- Branch: `feature/scheduling-module-screen`
- Commit: `5ba331c3` (`fix(scheduling): address UX polish review feedback`)
- Previous commits in this phase:
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

## User approval status

Pending user review of the attached `final-2026-07-18T23-23-00.png`.

## Tests run

```bash
.venv/bin/python -m pytest Tests/UI/test_schedules_workbench.py Tests/UI/test_destination_shells.py -v
```

Result: **117 passed, 1 skipped** (the single skip is an unrelated Personas tooltip audit).

## Residual risks

- The screenshots were generated from SVG renderings of the running app, then rasterized to PNG. They faithfully represent the terminal layout but are not direct pixel captures of a physical monitor.
- The full production app was not launched, so splash-screen, config-error, or first-run states are not exercised here.
- Compact-size layout was not explicitly verified; the workbench uses `fr` widths and should reflow, but a dedicated compact screenshot is recommended before the screen is marked fully mature.
