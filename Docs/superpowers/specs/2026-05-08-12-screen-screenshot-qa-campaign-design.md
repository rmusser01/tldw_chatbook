# 12-Screen Screenshot QA Campaign Design

Date: 2026-05-08
Status: draft for review
Source branch: `origin/dev`

## Purpose

The current Backlog.md tracker marks the 12 top-level destination screens as visually verified through mounted geometry and text evidence. That evidence is useful for regression coverage, but it is not sufficient for approval. Going forward, a screen is not approved until the user reviews an actual rendered screenshot of the running application and explicitly approves it.

This spec defines a staged campaign to validate and correct each top-level screen independently before continuing broader product-maturity work.

## Screens In Scope

The 12 top-level screens marked visually verified by the tracker are:

1. Console
2. Home
3. Library
4. Artifacts
5. Personas
6. Watchlists
7. Schedules
8. Workflows
9. MCP
10. ACP
11. Skills
12. Settings

Console remains first because it is already in flight and is the primary agentic control surface. Home and Library follow because they anchor the default status/control experience and source/RAG/product-loop workflows.

## Approval Rule

Each screen must be handled as its own isolated PR.

A screen is not approved when:

- A Textual mounted test passes.
- A geometry dump matches expected selectors.
- An SVG export looks acceptable.
- A code-level layout contract passes.

A screen is approved only when:

- The app is launched from latest `origin/dev` plus that screen's focused branch.
- An actual rendered screenshot is captured from the running app.
- The screenshot shows the target screen in the expected viewport.
- The screenshot is shown to the user.
- The user explicitly approves that screenshot.

## Per-Screen Workflow

Each screen follows the same sequence:

1. Start from latest `origin/dev`.
2. Create a dedicated branch and worktree for the screen.
3. Capture a baseline actual screenshot before changes.
4. Compare the screenshot against the approved ASCII/design contract and current product role.
5. Record defects as P0, P1, P2, or P3.
6. Fix only that screen unless a shared-shell bug blocks the screen.
7. Run focused automated verification for the changed seams.
8. Capture a final actual screenshot from the running app.
9. Ask the user to approve the screenshot.
10. Open one PR against `dev` only after screenshot approval.
11. Address review comments for that PR.
12. Continue to the next screen only after merge or an explicit user decision to pause.

## Branch And PR Naming

Use predictable branch names:

- `codex/screen-qa-console`
- `codex/screen-qa-home`
- `codex/screen-qa-library`
- `codex/screen-qa-artifacts`
- `codex/screen-qa-personas`
- `codex/screen-qa-watchlists`
- `codex/screen-qa-schedules`
- `codex/screen-qa-workflows`
- `codex/screen-qa-mcp`
- `codex/screen-qa-acp`
- `codex/screen-qa-skills`
- `codex/screen-qa-settings`

Each PR title should use:

`Screen QA: <Screen Name>`

## Screenshot Evidence

Screenshots must be actual captures of the running app. Do not use generated SVGs, synthetic HTML mockups, or code-rendered diagrams as approval evidence.

Store evidence under:

`Docs/superpowers/qa/product-maturity/screen-qa/<screen>/`

Each screen folder should contain:

- `baseline-<timestamp>.png`
- `final-<timestamp>.png`
- `notes.md`

The notes file should record:

- Branch under test
- Commit under test
- Viewport size
- Launch method
- Screenshot paths
- User approval status
- Tests run
- Residual risks

## Required Checks Per Screen

Every screen pass must verify:

- The top navigation is visible and the active screen is clear.
- The screen identity/header matches the product model.
- Primary regions are visible without awkward clipping.
- Primary actions are visible or honestly disabled with reason copy.
- Empty, loading, setup, blocked, or error states are understandable.
- Keyboard focus does not trap the user.
- The screen fits at default desktop terminal size.
- The screen has at least one compact-size check unless the user explicitly scopes it out.
- The Textual footer/status affordance remains visible unless intentionally replaced.

## Screen-Specific Focus

| Screen | Primary UX Question |
| --- | --- |
| Console | Can the user read, type, send, stage context, inspect run state, and recover from blocked provider/RAG/tool states? |
| Home | Can the user understand system status, active work, notifications, and next-best actions at a glance? |
| Library | Can the user find sources, use Search/RAG, manage Collections, import/export, and stage evidence to Console? |
| Artifacts | Can the user find generated outputs and reopen Chatbooks or artifacts into Console? |
| Personas | Can the user understand characters/personas/profiles and attach behavior context to Console? |
| Watchlists | Can the user monitor sources/runs/alerts without Collections confusion? |
| Schedules | Can the user understand timing, paused/running/failed jobs, and recovery actions? |
| Workflows | Can the user understand reusable procedures, run state, inputs/outputs, and handoff to Console? |
| MCP | Can the user understand server-first MCP configuration, tools, auth, permissions, and unavailable states? |
| ACP | Can the user understand ACP runtimes/sessions and why setup is required when blocked? |
| Skills | Can the user discover local Agent Skills, inspect readiness, and stage/attach skills? |
| Settings | Can the user understand global defaults without hiding destination-owned setup like ACP or MCP? |

## Tracking Table

| Order | Screen | Branch | Baseline Screenshot | Final Screenshot | User Approved | PR | Merge Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Console | `codex/screen-qa-console` | pending | pending | pending | pending | pending |
| 2 | Home | `codex/screen-qa-home` | pending | pending | pending | pending | pending |
| 3 | Library | `codex/screen-qa-library` | pending | pending | pending | pending | pending |
| 4 | Artifacts | `codex/screen-qa-artifacts` | pending | pending | pending | pending | pending |
| 5 | Personas | `codex/screen-qa-personas` | pending | pending | pending | pending | pending |
| 6 | Watchlists | `codex/screen-qa-watchlists` | pending | pending | pending | pending | pending |
| 7 | Schedules | `codex/screen-qa-schedules` | pending | pending | pending | pending | pending |
| 8 | Workflows | `codex/screen-qa-workflows` | pending | pending | pending | pending | pending |
| 9 | MCP | `codex/screen-qa-mcp` | pending | pending | pending | pending | pending |
| 10 | ACP | `codex/screen-qa-acp` | pending | pending | pending | pending | pending |
| 11 | Skills | `codex/screen-qa-skills` | pending | pending | pending | pending | pending |
| 12 | Settings | `codex/screen-qa-settings` | pending | pending | pending | pending | pending |

## Risk Controls

- Keep PRs screen-scoped so review and approval remain unambiguous.
- If a shared layout/token bug affects multiple screens, fix the shared seam in the first affected screen PR and document the downstream impact.
- Do not port stale local work wholesale. Start each screen from latest `origin/dev` and bring over only the focused, reviewed changes needed for that screen.
- Do not let automated geometry evidence replace screenshot approval.
- Do not proceed to the next screen while the current screen has unresolved P0 or P1 visual/usability issues unless the user explicitly accepts the residual risk.

## Done Definition

The 12-screen campaign is complete when every in-scope screen has:

- A dedicated merged PR.
- Baseline and final actual screenshot evidence.
- Explicit user approval of the final screenshot.
- Focused automated verification for changed behavior or layout seams.
- A `notes.md` file with residual risks and launch/test evidence.
