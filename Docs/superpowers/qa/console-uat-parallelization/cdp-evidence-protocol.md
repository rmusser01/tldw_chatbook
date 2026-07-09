# Console CDP Evidence Protocol

Date: 2026-06-21

## Purpose

Provide one durable Console evidence protocol for the parallel Console UAT streams. Historical plans reference a `textual-web-cdp-debugging.md` runbook, but that file is not present on current `dev`; this task therefore treats the durable Console CDP runbook as part of the harness work.

## Required Evidence

Each Console UAT PR must include:

- A screenshot or recording from the actual running app through Textual-web/CDP.
- Real user-path interaction for the evidence state: navigate/click/type through the running app UI and use the app's real local/server services.
- A live provider/API response for any state that claims completed assistant output. If no provider is reachable, capture the real blocked/unavailable state and mark provider-response UAT as blocked.
- A short note with the branch, commit, port, isolated config/data paths, and target Console state.
- The focused regression or UAT command used for that PR.
- A user approval checkpoint: approved, not approved, or blocked.

## Prohibited Evidence

Do not use these as approval evidence:

- Generated SVGs.
- ASCII mockups.
- Static code layout diagrams.
- Screenshots of test doubles or non-running stand-ins.
- Seeded or hand-authored assistant transcripts presented as if they were provider responses.
- Mocked provider responses, fake scope services, or fixture-only data presented as UAT acceptance evidence.
- Terminal logs without a rendered screen capture.

## Regression Tests Versus UAT Evidence

Automated regressions may use fakes, seeded fixtures, or deterministic service doubles
when needed for stable CI coverage. Those tests prove code paths and edge cases, but
they are not user acceptance evidence. UAT approval requires the real running app,
real UI interactions, real persistence/services, and real provider/API responses when
the workflow includes assistant generation.

## Launch Requirements

The harness PR must establish a reusable launch path with these properties:

- Runs from a clean worktree based on current `origin/dev`.
- Uses an isolated HOME/XDG config and data directory under `${TMPDIR:-/tmp}`.
- Uses a stable local port documented in the PR.
- Records the app log path.
- Can be rerun with escalation if local port binding is blocked by sandboxing.
- Avoids writing secrets, raw API keys, or full `.env` contents into committed artifacts.

## Screenshot Naming

Use this pattern:

```text
Docs/superpowers/qa/console-uat-parallelization/<task-id>-<state>-cdp-YYYY-MM-DD.png
```

Examples:

```text
Docs/superpowers/qa/console-uat-parallelization/task-129-blocked-send-cdp-2026-06-21.png
Docs/superpowers/qa/console-uat-parallelization/task-130-provider-dropdown-cdp-2026-06-21.png
Docs/superpowers/qa/console-uat-parallelization/task-131-message-actions-cdp-2026-06-21.png
```

## Minimum Console States

The parallel streams should capture at least these states before final closeout:

- Baseline Console with Default workspace visible.
- Composer focused with visible typed text.
- Blocked-send recovery feedback.
- Completed assistant response.
- New chat tab and compact close button.
- Provider/model selector with selected value visible.
- Model settings focused input with text visible.
- Selected message with action row visible.
- Edit message modal.
- Save as modal with unavailable destinations labeled.
- Workspace switcher or workspace rail with saved conversations.
- Resumed saved conversation after leaving and returning to Console.

## Final Integration Requirement

The final integration PR should replay the matrix from `acceptance-matrix.md` and update every row with:

- branch/PR number;
- screenshot or recording path;
- regression/UAT command;
- approval state;
- residual risk, if any.
