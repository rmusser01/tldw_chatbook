# Watchlists Screenshot QA Notes

Date:
Branch: `codex/screen-qa-watchlists`
Backlog task: TASK-14.6
Commit: pending
Screen: Watchlists
Viewport: 1960x1240 browser viewport via textual-web
Launch method: `tldw-serve --host 127.0.0.1 --port 8841` with isolated Watchlists default-tab config
Screenshot method: Playwright Chromium capture of actual textual-web runtime
Fallback reason: none

## Baseline Screenshot

- Path: `Docs/superpowers/qa/product-maturity/screen-qa/watchlists/baseline-2026-05-09-watchlists.png`
- Defects: Header consumed excessive vertical space, workbench started too low, panes were not clearly titled as list/detail/inspector columns, future-resizable dividers were absent, and the detail pane could remain stuck in a loading state.

## Interaction Smoke

- Goal: Verify blocked Watchlists recovery path remains usable when local Watchlists services are unavailable.
- Steps: Opened Watchlists as the default destination in the running app and inspected the disabled Console staging/follow actions in the status inspector.
- Result: The screen renders a recoverable service-unavailable detail state, disabled stage/follow actions, and an always-visible route to open the current Watchlists surface.

## Fixes

- Summary: Converted Watchlists into the approved compact control-plane layout with a status/filter header, explicit list/detail/status-inspector column titles, future-resizable dividers, Watchlists-only copy, and a bounded service recovery state instead of an indefinite loading dead-end.

## Final Screenshot

- Path: `Docs/superpowers/qa/product-maturity/screen-qa/watchlists/final-2026-05-09-watchlists.png`
- User approval: approved in chat on 2026-05-09 with "yes good for now"

## Verification

- Commands: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_destination_shells.py -k watchlists --tb=short`
- Commands: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_destination_visual_parity_correction.py -k watchlists --tb=short`
- Results: Focused Watchlists shell and visual-parity regressions passed before PR creation.

## Residual Risks

- The actual local Watchlists data service may still be unavailable in the isolated QA runtime; the screen now fails closed into a visible recovery state rather than an indefinite loading state.
