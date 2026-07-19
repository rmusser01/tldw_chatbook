# Schedules Workbench Screenshot QA

## Branch / commit under test

- Branch: `feature/scheduling-module-screen`
- Commit: `66fb5513` (`test(scheduling): update navigation test for SchedulesWorkbench route`)
- Previous commits in this phase:
  - `c18439b0` — `feat(scheduling): route schedules destination to new workbench`
  - `c39f24de` — `fix(scheduling): address Console-follow seam review feedback`
  - `d9919bce` — `feat(scheduling): preserve Console-follow seam and update destination tests`

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
2. The SVG was wrapped in a minimal HTML page and rasterized to PNG via Chrome headless (`--headless --screenshot ...`).
3. Resulting PNGs are stored below.

## Screenshot paths

- `baseline-2026-07-18T22-43-17.png` — legacy `SchedulesScreen` empty state
- `final-2026-07-18T22-42-14.png` — new `SchedulesWorkbench` with populated reminder queue

## Baseline defects found

The baseline (legacy `SchedulesScreen`) showed:

- Filter counts (`Next Run 0`, `Paused 0`, `Failed 0`, `Retry 0`, `History 0`) dominate the left pane and do not match the new data-first workbench model.
- Action buttons (`Retry run`, `Pause run`) are always present but disabled in the empty state; their purpose is unclear without an active run.
- Console-follow recovery copy is verbose and split across multiple panes.

The final `SchedulesWorkbench` addresses these by:

- Replacing count filters with a `DataTable` showing actual reminder tasks (`Title`, `Kind`, `Status`, `Next Run`).
- Showing lifecycle actions (`Enable`, `Disable`, `Delete`) only in the detail pane for the selected task.
- Consolidating status, sync, and conflict metadata in the inspector pane.
- Preserving the `#schedules-follow-in-console` Console-follow seam with honest disabled copy when no active run is available.

## User approval status

Pending user review of the attached `final-2026-07-18T22-42-14.png`.

## Tests run

```bash
.venv/bin/python -m pytest Tests/UI/test_schedules_workbench.py Tests/UI/test_screen_navigation.py::test_lazy_screen_registry_resolves_visible_shell_destinations Tests/UI/test_destination_shells.py -v
```

Result: **107 passed, 1 skipped** (the single skip is an unrelated Personas tooltip audit).

## Residual risks

- The screenshots were generated from SVG renderings of the running app, then rasterized to PNG. They faithfully represent the terminal layout but are not direct pixel captures of a physical monitor.
- The full production app was not launched, so splash-screen, config-error, or first-run states are not exercised here.
- Compact-size layout was not explicitly verified; the workbench uses percentage widths and should reflow, but a dedicated compact screenshot is recommended before the screen is marked fully mature.
- The `Follow in Console` button is partially clipped at the bottom of the final screenshot because the detail pane content exceeds the rendered viewport. The button is present and functional; widening the terminal or scrolling the detail pane would reveal it fully.
