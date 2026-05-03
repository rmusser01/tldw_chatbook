# Unified Shell Maturity Roadmap

Date: 2026-05-03
Status: Phase 2 in progress
Source Branch: `origin/dev` at `726f4954` plus `codex/unified-shell-phase2-home-dashboard-snapshot`

## Purpose

Track remaining Unified Shell work in one place so rendered screens, clickable buttons, and real user workflow completion are not confused with each other.

## Current Verified Baseline

- PR #204 added the master shell UX foundation.
- PR #205 hardened shell action affordances and changed the visible Watchlists/Collections shell label to `W+C`.
- `Docs/Design/master-shell-route-inventory.md` maps top-level shell destinations and deferred surfaces.
- `Docs/Design/master-shell-design-system-contract.md` defines shared shell classes, state labels, and testing hooks.
- `Docs/superpowers/specs/2026-05-02-agentic-terminal-design-system-design.md` defines the agentic terminal visual and interaction grammar.
- `Docs/superpowers/specs/2026-05-03-unified-shell-maturity-tracking-design.md` defines this maturity-gate tracking model.

## Baseline Tracking Gaps Captured At Phase 0 Start

- The detailed master-shell implementation plan exists locally in the stale root checkout but is not present on `origin/dev`.
- The UX remediation plan is useful history but is no longer the canonical tracker for Unified Shell completion.
- Backlog.md was not initialized before Phase 0. Resolved by `TASK-1.1`.
- Current shell wrappers expose some honest unavailable states rather than complete product workflows.
- Phase-completion evidence was not centralized around manual QA walkthroughs.

## Known Product Gaps

- Home active-work controls, detail routing, Console launch requests, and item identity now route through an explicit adapter boundary; real service-backed adapters still need implementation.
- Workflows has no wired workflow service in the shell wrapper.
- W+C, Schedules, Workflows, and ACP now use honest unavailable Console states until actionable payloads exist.
- ACP launch is disabled until an ACP-compatible runtime is configured.
- MCP management is not embedded in the top-level MCP wrapper.
- Skills local/server services exist, but the top-level Skills shell still leaves import disabled and lacks list/detail/import UX adoption.
- Library still links into legacy Notes, Media, Ingest, Search/RAG, and conversation screens rather than fully Library-native views.
- Product workflows remain unverified until running-app QA evidence is added in later phases.

## Definition Of Done

A task or phase is not complete just because UI renders or a control is clickable.

Every completed implementation task must include:

- Automated regression evidence for the changed seam.
- Manual QA walkthrough evidence from the running app.
- Repo-tracked QA summary under `Docs/superpowers/qa/unified-shell/`.
- Visual usability notes for layout, labels, focus, and reachable controls.
- Functional workflow evidence proving the user can complete the relevant task or recover from the blocked state.
- Residual-risk notes for live server/API paths, optional dependencies, or environment limits not exercised.

Every completed phase must include:

- Linked Backlog tasks.
- Linked PRs or commits.
- Focused test output.
- QA walkthrough artifact location.
- Screenshots, logs, or equivalent app-use evidence when relevant.
- Defect severity summary for anything found during QA.
- A short "what remains" statement.

## Backlog Task Hierarchy

Parent phase tasks:

- Phase 0: Canonical Tracking - `TASK-1`
- Phase 1: Shell Contract Complete - `TASK-2`
- Phase 2: Home Operational Control - `TASK-4`
- Phase 3: Console Live Work Hub - `TASK-3`
- Phase 4: Destination Service Adoption - `TASK-5`
- Phase 5: Capability And Recovery System - `TASK-6`
- Phase 6: Audit Replay And Closeout - `TASK-7`

Initial child tasks:

- Phase 0.1: Initialize Backlog.md and roadmap docs - `TASK-1.1`
- Phase 0.2: Reconcile current dev state and merged evidence - `TASK-1.2`
- Phase 1.1: Create shell QA walkthrough harness and evidence template - `TASK-2.1`
- Phase 1.2: Audit destination action functionality beyond render/click tests - `TASK-2.2`
- Phase 1.3: Remove false Console-launch affordances from skeletal destinations - `TASK-2.3`
- Phase 1.4: Replay shell contract and close Phase 1 - `TASK-2.4`
- Phase 2.1: Add Home active-work adapter contract - `TASK-4.1`
- Phase 2.2: Route Home detail and Console actions through active-work adapter - `TASK-4.2`
- Phase 2.3: Bind Home controls to active-work item context - `TASK-4.3`

## QA Evidence Index

| Phase | Evidence Path | Status |
| --- | --- | --- |
| Phase 0 | `Docs/superpowers/qa/unified-shell/phase-0/` | verified |
| Phase 1 | `Docs/superpowers/qa/unified-shell/phase-1/` | verified |
| Phase 2 | `Docs/superpowers/qa/unified-shell/phase-2/` | in-progress |
| Phase 3 | `Docs/superpowers/qa/unified-shell/phase-3/` | not-started |
| Phase 4 | `Docs/superpowers/qa/unified-shell/phase-4/` | not-started |
| Phase 5 | `Docs/superpowers/qa/unified-shell/phase-5/` | not-started |
| Phase 6 | `Docs/superpowers/qa/unified-shell/phase-6/` | not-started |

## Phase Overview

| Phase | Goal | Status | Backlog Tasks | QA Evidence | Residual Risk |
| --- | --- | --- | --- | --- | --- |
| Phase 0: Canonical Tracking | Make remaining work trackable. | verified | `TASK-1`, `TASK-1.1`, `TASK-1.2` | `phase-0/` | Product UI workflows are out of scope for Phase 0. |
| Phase 1: Shell Contract Complete | Remove false shell affordances and prove shell usability. | verified | `TASK-2`, `TASK-2.1`, `TASK-2.2`, `TASK-2.3`, `TASK-2.4` | `phase-1/` | Live service workflows remain intentionally deferred to Phases 2-6. |
| Phase 2: Home Operational Control | Make Home a real dashboard/control surface. | in-progress | `TASK-4`, `TASK-4.1`, `TASK-4.2`, `TASK-4.3` | `phase-2/` | Real active-run, schedule, and agent-service adapters still need implementation. |
| Phase 3: Console Live Work Hub | Make Console the live-agent control surface. | not-started | `TASK-3` | `phase-3/` | Live work event sources need explicit contracts. |
| Phase 4: Destination Service Adoption | Turn wrappers into useful product surfaces. | not-started | `TASK-5` | `phase-4/` | Service coverage varies by destination. |
| Phase 5: Capability And Recovery System | Systematize unavailable and blocked states. | not-started | `TASK-6` | `phase-5/` | Shared taxonomy not yet extracted. |
| Phase 6: Audit Replay And Closeout | Prove shell works for first-time and power users. | not-started | `TASK-7` | `phase-6/` | Depends on prior phases and running-app QA. |

## Phase 1.1 QA Harness Evidence

`TASK-2.1` owns the first reusable Phase 1 QA harness assets:

- `Docs/superpowers/qa/unified-shell/phase-1/walkthrough-protocol.md`
- `Docs/superpowers/qa/unified-shell/phase-1/walkthrough-template.md`
- `Docs/superpowers/qa/unified-shell/phase-1/2026-05-03-phase-1-protocol-smoke.md`

## Phase 1.2 Destination Action Audit Evidence

`TASK-2.2` records action ownership and usability status for every top-level destination:

- `Docs/superpowers/qa/unified-shell/phase-1/2026-05-03-destination-action-audit.md`

Initial audit result: W+C, Schedules, Workflows, and ACP had false Console-launch affordances. `TASK-2.3` resolves those as honest unavailable states.

## Phase 1.3 False Affordance Evidence

`TASK-2.3` resolves the Phase 1.2 false Console-launch finding by turning skeletal Console follow/launch actions into disabled unavailable states with recovery copy:

- `Docs/superpowers/qa/unified-shell/phase-1/2026-05-03-phase-1-3-false-affordance-fix.md`

## Phase 1.4 Shell Contract Closeout Evidence

`TASK-2.4` replays the Phase 1 shell contract and closes the phase as verified:

- `Docs/superpowers/qa/unified-shell/phase-1/2026-05-03-phase-1-shell-contract-closeout.md`

## Phase 2.1 Home Adapter Evidence

`TASK-4.1` adds an explicit Home active-work adapter boundary for dashboard state and lightweight controls:

- `Docs/superpowers/qa/unified-shell/phase-2/2026-05-03-home-active-work-adapter-contract.md`

## Phase 2.2 Home Detail And Console Adapter Evidence

`TASK-4.2` moves Home detail and Console actions behind the active-work adapter boundary:

- `Docs/superpowers/qa/unified-shell/phase-2/2026-05-03-home-detail-console-adapter-actions.md`

## Phase 2.3 Home Active-Work Item Context Evidence

`TASK-4.3` binds Home controls to explicit visible active-work item context:

- `Docs/superpowers/qa/unified-shell/phase-2/2026-05-03-home-active-work-item-context.md`

## Phase 0: Canonical Tracking

Done when:

- Backlog.md is initialized and smoke checks pass without interactive prompts.
- This roadmap exists and links generated Backlog task IDs.
- `backlog/docs/unified-shell-maturity-roadmap.md` points Backlog users to the canonical roadmap.
- QA evidence directories exist.
- Phase parent tasks and initial child tasks exist.
- Phase 0 QA summary records commands run, outputs, deviations, and residual risk.

## Phase 1: Shell Contract Complete

Done when every destination has honest status/action ownership, tests, and a manual shell walkthrough proving navigation, layout, focus, labels, and primary actions are usable.

## Phase 2: Home Operational Control

Done when Home controls route to real services or explicit adapters, and QA verifies approve, reject, pause, resume, retry, and open-detail workflows from the running app.

## Phase 3: Console Live Work Hub

Done when Console receives live work from workflows, schedules, ACP, MCP, RAG, and artifacts, and QA verifies the flows are completeable rather than merely clickable.

## Phase 4: Destination Service Adoption

Done when destination wrappers expose real list/detail/action flows where services exist, and QA verifies each destination supports at least one meaningful end-to-end user workflow.

## Phase 5: Capability And Recovery System

Done when missing dependency, auth, server, runtime, and policy states use shared recovery patterns, and QA verifies blocked states are understandable and recoverable.

## Phase 6: Audit Replay And Closeout

Done when first-time and power-user walkthroughs are replayed in the actual app with screenshots, logs, test evidence, and residual-risk notes.

## QA Walkthrough Protocol

Each QA walkthrough must record:

- Environment: branch, commit, Python version, runtime source, config/home directory.
- Entry path: clean Home launch, direct route, command palette, or destination link.
- Visual check: layout, clipping, focus indication, labels, disabled states, and information hierarchy.
- Keyboard path: whether the workflow can be completed without mouse-only assumptions.
- Mouse/click path: whether clickable elements behave as promised.
- Functional result: completed workflow, blocked workflow with recovery, or defect.
- Defect severity: `blocker`, `workflow-degradation`, `recoverability`, or `polish`.
- Evidence: screenshots, logs, probe JSON, or concise notes linked from the roadmap.
- Residual risk: untested live server/API/dependency paths.

## Update Rules

- Roadmap changes must happen through PRs.
- Backlog task IDs must be linked from this roadmap.
- Backlog tasks must not be marked `Done` until acceptance criteria, automated verification, implementation notes, and QA walkthrough evidence are complete.
- Phase status must be one of: `not-started`, `in-progress`, `implemented`, `qa-needed`, `qa-failed`, `verified`, or `blocked`.
- `implemented` means code or docs landed but app-level QA has not passed.
- `qa-needed` means automated checks passed and manual walkthrough is pending.
- `qa-failed` means the running app walkthrough found a blocker or workflow-degradation issue.
- `verified` means both automated checks and manual QA evidence are complete.
- Render-only, mount-only, and click-event-only tests do not count as workflow completion.
- Preserve historical evidence without pretending stale plans are current.
