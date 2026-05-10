# Personas Screenshot QA Notes

Date: 2026-05-09
Branch: `codex/screen-qa-personas`
Backlog task: TASK-14.5
Commit: pending
Screen: Personas
Viewport: 2050x1240 textual-web capture
Launch method: `tldw-serve --host 127.0.0.1 --port 8830` with isolated QA HOME/XDG paths
Screenshot method: Playwright Chromium screenshot of the running textual-web session
Fallback reason: none; actual rendered screenshot captured

## Baseline Screenshot

- Path: `Docs/superpowers/qa/product-maturity/screen-qa/personas/baseline-2026-05-09-personas.png`
- Defects: Personas lacked a clearly named three-column destination model, did not expose visible divider rails for future resizing, and the inspector/actions column did not read as a bounded workbench column.

## Interaction Smoke

- Goal: Verify the Personas destination can load local behavior context and expose a Console attachment path without hanging.
- Steps: Open Personas from a clean textual-web session, wait for local character/persona snapshot resolution, verify counts/actions render, and confirm the final screenshot shows stable column state.
- Result: Local behavior snapshot resolved with character/profile counts, inspector readiness copy, and `Open Personas` / `Attach to Console` actions visible.

## Fixes

- Summary: Added explicit Personas mode strip, three named workbench columns, full-height column borders, divider rails between panes for later resizable handles, and deterministic service timeout behavior for stalled Personas snapshot loads.

## Final Screenshot

- Path: `Docs/superpowers/qa/product-maturity/screen-qa/personas/final-2026-05-09-personas.png`
- User approval: approved in-session on 2026-05-09 after reviewing the actual rendered screenshot.

## Verification

- Commands: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_destination_visual_parity_correction.py -k personas --tb=short`
- Commands: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_destination_shells.py -k personas --tb=short`
- Commands: `git diff --check`
- Results: `7 passed, 67 deselected, 1 warning`; `13 passed, 63 deselected, 1 warning`; diff check clean. The warning is the existing Requests dependency warning from the local environment.

## Residual Risks

- The divider rails are explicit visual/structural affordances only; click-and-drag resizing is intentionally deferred to a later resizable pane implementation.
