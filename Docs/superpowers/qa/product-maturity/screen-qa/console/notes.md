# Console Screenshot QA Notes

Date: 2026-05-09
Branch: `codex/screen-qa-console`
Backlog task: TASK-14.1
Commit:
Screen: Console
Viewport: 2048x1280
Launch method: `tldw-serve --host 127.0.0.1 --port 8765` with `PYTHONPATH` pointed at the Console QA worktree.
Screenshot method: Actual textual-web rendering captured through headless Chromium / Playwright after a 45-second post-load wait for the splash screen to clear.
Fallback reason: Codex in-app Browser lost its active pane after tab refresh, so repeatable captures used Playwright against the same running textual-web server.

## Baseline Screenshot

- Path: `Docs/superpowers/qa/product-maturity/screen-qa/console/baseline-2026-05-08-playwright-2048x1280.png`
- Defects: No blocking Console layout defect in the 2048x1280 actual browser capture. The earlier in-app Browser capture retained a stale canvas size after viewport changes and was not used as approval evidence.

## Interaction Smoke

- Goal: Verify the Console composer supports visible typed input and a longer pasted prompt without hiding the text.
- Steps: Focused the textual-web terminal textarea, typed a short prompt, then pasted a long multi-sentence prompt.
- Result: Short typed text is visible in the composer. Long pasted text expands the composer to multiple visible rows and keeps the active end of the prompt visible.
- Evidence:
  - `Docs/superpowers/qa/product-maturity/screen-qa/console/interaction-2026-05-08-playwright-composer-textarea-focus.png`
  - `Docs/superpowers/qa/product-maturity/screen-qa/console/interaction-2026-05-08-playwright-composer-long-text.png`

## Fixes

- Summary: Updated stale Console live-work source readiness regression expectations to match the current compact copy shipped on `dev`; no screenshot-backed Console layout code change was required in this pass.

## Final Screenshot

- Path: `Docs/superpowers/qa/product-maturity/screen-qa/console/final-2026-05-08-playwright-console.png`
- User approval: approved by user in Codex thread on 2026-05-09 after recapture with splash cleared.

## Verification

- Commands:
  - From the repository root with the repo virtualenv active: `python -m pytest -q Tests/UI/test_console_internals_decomposition.py Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py::test_console_core_loop_exposes_agentic_shell_regions --tb=short`
- Results: `105 passed, 1 warning in 45.63s`

## Residual Risks

- Browser capture required a fixed post-load wait because textual-web exposes the terminal as a canvas rather than semantic app widgets.
