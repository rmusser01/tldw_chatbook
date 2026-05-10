# Screen Screenshot QA Campaign

This folder tracks actual rendered screenshot approval for the 12 top-level destination screens. Geometry dumps and SVG exports are regression evidence only. A screen is approved only after the user approves an actual screenshot from the running app.

## Rules

- Capture screenshots from the running app only.
- Do not use rendered SVGs, generated mockups, or code layouts for approval.
- Keep one screen per branch and PR.
- Do not open a screen PR until the final screenshot has explicit user approval.
- Record rejected screenshots and follow-up fixes in the target screen's `notes.md`.
- Prefer `textual-serve` plus browser automation; if using a terminal screenshot fallback, record the reason.
- Use `textual-web-cdp-debugging.md` for the approved outside-sandbox textual-web/CDP capture workflow.

## Tracker

| Order | Screen | Backlog Task | Branch | Baseline | Final | Approved | PR | Merged |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Console | TASK-14.1 | `codex/screen-qa-console` | `console/baseline-2026-05-08-playwright-2048x1280.png` | `console/final-2026-05-08-playwright-console.png` | approved | #287 | merged |
| 2 | Home | TASK-14.2 | `codex/screen-qa-home` | `home/baseline-2026-05-08-playwright-home.png` | `home/final-2026-05-09-playwright-home-polish2.png` | approved | #288 | pending |
| 3 | Library | TASK-14.3 | `codex/screen-qa-library` | `library/baseline-2026-05-09-playwright-library.png` | `library/review-2026-05-09-playwright-library-contextual-inspector.png` | approved | pending | pending |
| 4 | Artifacts | TASK-14.4 | `codex/screen-qa-artifacts` | `artifacts/baseline-2026-05-09-artifacts.png` | `artifacts/final-2026-05-09-artifacts-columns.png` | approved | #292 | pending |
| 5 | Personas | TASK-14.5 | `codex/screen-qa-personas` | pending | pending | pending | pending | pending |
| 6 | Watchlists | TASK-14.6 | `codex/screen-qa-watchlists` | pending | pending | pending | pending | pending |
| 7 | Schedules | TASK-14.7 | `codex/screen-qa-schedules` | pending | pending | pending | pending | pending |
| 8 | Workflows | TASK-14.8 | `codex/screen-qa-workflows` | pending | pending | pending | pending | pending |
| 9 | MCP | TASK-14.9 | `codex/screen-qa-mcp` | pending | pending | pending | pending | pending |
| 10 | ACP | TASK-14.10 | `codex/screen-qa-acp` | pending | pending | pending | pending | pending |
| 11 | Skills | TASK-14.11 | `codex/screen-qa-skills` | pending | pending | pending | pending | pending |
| 12 | Settings | TASK-14.12 | `codex/screen-qa-settings` | pending | pending | pending | pending | pending |
