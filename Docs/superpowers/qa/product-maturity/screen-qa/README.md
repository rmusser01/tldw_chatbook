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
| 2 | Home | TASK-14.2 | `codex/screen-qa-home` | `home/baseline-2026-05-08-playwright-home.png` | `home/final-2026-05-09-playwright-home-polish2.png` | approved | #288 | merged |
| 3 | Library | TASK-14.3 | `codex/screen-qa-library` | `library/baseline-2026-05-09-playwright-library.png` | `library/review-2026-05-09-playwright-library-contextual-inspector.png` | approved | pending | pending |
| 4 | Artifacts | TASK-14.4 | `codex/screen-qa-artifacts` | `artifacts/baseline-2026-05-09-artifacts.png` | `artifacts/final-2026-05-09-artifacts-columns.png` | approved | #292 | merged |
| 5 | Personas | TASK-14.5 | `codex/screen-qa-personas` | pending | pending | approved | #294/#295 | merged |
| 6 | Watchlists | TASK-14.6 | `codex/screen-qa-watchlists` | `watchlists/baseline-2026-05-09-watchlists.png` | `watchlists/final-2026-05-09-watchlists.png` | approved | #296 | merged |
| 7 | Schedules | TASK-14.7 | `codex/screen-qa-schedules` | `schedules/baseline-2026-05-09-schedules.png` | `schedules/final-2026-05-10-schedules.png` | approved | #297/#298 | merged |
| 8 | Workflows | TASK-14.8 | `codex/screen-qa-workflows` | `workflows/baseline-2026-05-10-workflows.png` | `workflows/final-2026-05-10-workflows.png` | approved | #299 | merged |
| 9 | MCP | TASK-14.9 | `codex/screen-qa-mcp` | `mcp/baseline-2026-05-10-mcp.png` | `mcp/final-2026-05-10-mcp.png` | approved | #301 | merged |
| 10 | ACP | TASK-14.10 | `codex/screen-qa-acp` | `acp/baseline-2026-05-10-acp-rerun.png` | `acp/final-2026-05-10-acp-columns.png` | approved | #304 | merged |
| 11 | Skills | TASK-14.11 | `codex/screen-qa-skills` | `skills/baseline-2026-05-11-skills.png` | `skills/final-2026-05-11-skills.png` | approved | #305 | merged |
| 12 | Settings | TASK-14.12 | `codex/screen-qa-settings` | `settings/baseline-2026-05-11-settings.png` | `settings/final-2026-05-11-settings.png` | approved | pending | pending |
