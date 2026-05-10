# Schedules Screenshot QA Notes

Date: 2026-05-10
Branch: `codex/screen-qa-schedules`
Backlog task: TASK-14.7
Commit: pending
Screen: Schedules
Viewport: 2050x1240 browser viewport via textual-web
Launch method: `tldw-serve --host 127.0.0.1 --port 8843` with isolated Schedules default-tab config
Screenshot method: Playwright Chromium capture of actual textual-web runtime
Fallback reason: none

## Baseline Screenshot

- Path: `Docs/superpowers/qa/product-maturity/screen-qa/schedules/baseline-2026-05-09-schedules.png`
- Defects: Header consumed excessive vertical space, filter and ownership copy were visually detached from the workbench, panes were not clearly titled as list/detail/status inspector columns, future-resizable dividers were absent, and the status inspector had only a disabled action without persistent state/retry/next-action context.

## Interaction Smoke

- Goal: Verify the blocked Schedules-to-Console recovery path remains understandable when no active schedule run or digest output exists.
- Steps: Opened Schedules as the default destination in the running app and inspected the disabled Console recovery action and detail-pane recovery copy.
- Result: The final screen renders a blocked but recoverable state with a clear owner, next action, retry/backoff status, and disabled Console handoff action.

## Fixes

- Summary: Converted Schedules into the approved compact control-plane layout with a status/filter header, explicit schedule queue/detail/status-inspector columns, future-resizable dividers, and visible state/retry/next-action inspector copy while preserving the existing Console follow and reading-digest launch behavior.

## Final Screenshot

- Path: `Docs/superpowers/qa/product-maturity/screen-qa/schedules/final-2026-05-10-schedules.png`
- User approval: approved in chat on 2026-05-10 with "approved"

## Verification

- Commands: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_destination_shells.py -k schedules --tb=short`
- Commands: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_destination_visual_parity_correction.py -k schedules --tb=short`
- Commands: `git diff --check`
- Results: Focused Schedules shell and visual-parity regressions passed before PR creation; whitespace check was clean.

## Residual Risks

- The isolated QA runtime has no active schedule run or reading digest output, so this pass approves the empty/blocked recovery state. Populated run and digest-output states remain covered by mounted Console handoff tests.
