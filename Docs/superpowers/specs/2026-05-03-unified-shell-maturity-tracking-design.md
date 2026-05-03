# Unified Shell Maturity Tracking Design

Date: 2026-05-03
Status: Draft
Source baseline: `origin/dev` at `534cf226` (`Harden master shell action affordances (#205)`)

## Purpose

Create a durable tracking system for the remaining Unified Shell work after the foundation PRs. The goal is to prevent the current failure mode where screens render, routes exist, or buttons emit events, but the running app is still visually broken, confusing, or functionally incomplete.

The tracking system must answer four questions quickly:

- What has already landed?
- What work remains?
- Which PR-sized task owns the next slice?
- What evidence proves a phase is actually usable in the running app?

## Current Verified Baseline

The Unified Shell foundation is partially landed on `dev`:

- PR #204 added the master shell UX foundation.
- PR #205 hardened shell action affordances and changed the visible Watchlists/Collections shell label to `W+C`.
- `Docs/Design/master-shell-route-inventory.md` maps top-level shell destinations and deferred surfaces.
- `Docs/Design/master-shell-design-system-contract.md` defines shared shell classes, state labels, and testing hooks.
- `Docs/superpowers/specs/2026-05-02-agentic-terminal-design-system-design.md` defines the visual and interaction grammar.
- The UX remediation stream has substantial merged evidence, including the local audit replay work and many disabled-action/recovery-tooltip fixes.

Known tracking gaps:

- The detailed master-shell implementation plan exists locally in the stale root checkout but is not present on `origin/dev`.
- The UX remediation plan is useful history but is no longer an accurate canonical tracker for Unified Shell completion.
- Backlog.md is installed locally, but `backlog/` is not initialized in the repository.
- Current shell wrappers expose some honest unavailable states rather than complete product workflows.

Known product gaps from current `origin/dev`:

- Home active-work controls still use placeholder notification hooks for approve, reject, pause, resume, and retry.
- Workflows has no wired workflow service in the shell wrapper.
- ACP launch is disabled until an ACP-compatible runtime is configured.
- MCP management is not embedded in the top-level MCP wrapper.
- Skills local/server services exist, but the top-level Skills shell still leaves import disabled and lacks list/detail/import UX adoption.
- Library still links into legacy Notes, Media, Ingest, Search/RAG, and conversation screens rather than fully Library-native views.
- Phase-completion evidence is not centralized around manual QA walkthroughs.

## Tracking Strategy

Use a maturity-gate hybrid:

- The roadmap is the executive source of truth.
- Backlog.md tasks are PR-sized execution units.
- The roadmap tracks phase status, QA evidence, residual risk, and links to task IDs.
- Backlog tasks track implementation details, acceptance criteria, and per-PR completion notes.

Canonical roadmap path:

`Docs/superpowers/trackers/unified-shell-maturity-roadmap.md`

Durable QA evidence path:

`Docs/superpowers/qa/unified-shell/`

QA evidence may reference bulky temporary artifacts, but every completed task or phase must have a repo-tracked summary in this directory so evidence does not disappear when `/private/tmp` is cleaned.

Backlog setup:

- Initialize `backlog/` if absent.
- Use `backlog/docs/` for a short pointer or mirror note that references the canonical roadmap.
- Use `backlog/tasks/` for atomic task files.
- Create one parent Backlog task per maturity phase.
- Create child Backlog tasks for PR-sized implementation slices.
- Link child tasks to phase parents and use explicit dependencies when a later slice relies on an earlier contract or QA harness.
- Use labels such as `unified-shell`, `phase-0`, `home`, `console`, `workflow`, `capability-state`, and `audit-replay`.

## Global Definition Of Done

No phase or task is complete just because UI renders, a route resolves, or a button is clickable.

Every completed task must include:

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

## Maturity Gates

| Phase | Goal | Done When |
| --- | --- | --- |
| Phase 0: Canonical Tracking | Make remaining work trackable. | Backlog is initialized, post-init Backlog smoke checks pass, the roadmap exists, current `dev` state is reconciled, phase parent tasks and initial child tasks exist, and the tracker defines mandatory QA evidence for every later phase. |
| Phase 1: Shell Contract Complete | Remove false shell affordances and prove the shell is usable. | Every destination has honest status/action ownership, tests, and a manual shell walkthrough proving navigation, layout, focus, labels, and primary actions are usable. |
| Phase 2: Home Operational Control | Make Home a real dashboard/control surface. | Home controls route to real services or explicit adapters, and QA verifies approve, reject, pause, resume, retry, and open-detail workflows from the running app. |
| Phase 3: Console Live Work Hub | Make Console the single live-agent control surface. | Console receives live work from workflows, schedules, ACP, MCP, RAG, and artifacts, and QA verifies the flows are completeable rather than merely clickable. |
| Phase 4: Destination Service Adoption | Turn wrappers into useful product surfaces. | Destination wrappers expose real list/detail/action flows where services exist, and QA verifies each destination supports at least one meaningful end-to-end user workflow. |
| Phase 5: Capability And Recovery System | Systematize unavailable and blocked states. | Missing dependency, auth, server, runtime, and policy states use shared recovery patterns, and QA verifies blocked states are understandable and recoverable. |
| Phase 6: Audit Replay And Closeout | Prove the shell works for first-time and power users. | First-time and power-user walkthroughs are replayed in the actual app with screenshots, logs, test evidence, and residual-risk notes. |

## Roadmap Document Structure

`Docs/superpowers/trackers/unified-shell-maturity-roadmap.md` should use this structure:

```markdown
# Unified Shell Maturity Roadmap

Date:
Status:
Source Branch:

## Purpose

## Current Verified Baseline

## Definition Of Done

## Backlog Task Hierarchy

## QA Evidence Index

## Phase Overview

| Phase | Goal | Status | Backlog Tasks | QA Evidence | Residual Risk |

## Phase 0: Canonical Tracking
## Phase 1: Shell Contract Complete
## Phase 2: Home Operational Control
## Phase 3: Console Live Work Hub
## Phase 4: Destination Service Adoption
## Phase 5: Capability And Recovery System
## Phase 6: Audit Replay And Closeout

## QA Walkthrough Protocol

## Update Rules
```

## Backlog Task Structure

Backlog tasks must be PR-sized and outcome-oriented.

Backlog hierarchy:

- Parent tasks represent maturity phases and should not contain implementation details.
- Child tasks represent PR-sized implementation or QA slices.
- Child tasks must link to their parent phase task.
- Dependencies should be explicit when one child task creates a contract, harness, adapter, or service seam needed by later tasks.
- Later phase parent tasks may exist immediately as planning anchors, but later child tasks should stay drafts or uncreated until they can be scoped to one reviewable PR.

Example:

```markdown
# task-N - Wire Home active-work controls

## Description

Make Home controls operate through explicit service/adapters instead of placeholder notifications.

## Acceptance Criteria

- [ ] User can approve or reject a pending approval from Home.
- [ ] User can pause, resume, or retry active work from Home when a real item exists.
- [ ] Empty or unavailable states remain honest and non-clickbait.
- [ ] Manual QA walkthrough confirms controls work in the running app.
- [ ] Focused automated tests cover the state and event seams.

## Implementation Plan

Added only after task starts.

## Implementation Notes

Added after implementation with PR, test, and QA evidence.
```

## Initial Backlog Task Set

Create parent tasks for every phase immediately. Create initial child tasks for Phase 0 and Phase 1 immediately. Create later phase child tasks only when enough information exists to keep them PR-sized.

Initial parent tasks:

- Phase 0: Canonical Tracking.
- Phase 1: Shell Contract Complete.
- Phase 2: Home Operational Control.
- Phase 3: Console Live Work Hub.
- Phase 4: Destination Service Adoption.
- Phase 5: Capability And Recovery System.
- Phase 6: Audit Replay And Closeout.

Initial child tasks:

- Phase 0.1: Initialize Backlog.md and roadmap docs.
- Phase 0.2: Reconcile current `dev` state and merged evidence.
- Phase 1.1: Create shell QA walkthrough harness and evidence template.
- Phase 1.2: Audit destination action functionality beyond render/click tests.

Planned follow-on task groups:

- Phase 2.1: Define Home active-work adapter contract.
- Phase 2.2: Wire Home controls to real action handlers.
- Phase 3.1: Define Console launch/follow payload contract.
- Phase 3.2: Render Console live-work status cards.
- Phase 4.1: Adopt Skills services in the top-level Skills shell.
- Phase 4.2: Adopt MCP management in the top-level MCP shell.
- Phase 4.3: Adopt Workflows and Schedules services.
- Phase 4.4: Adopt ACP runtime/session services.
- Phase 4.5: Adopt Library Search/RAG/import-export entry flows.
- Phase 5.1: Create shared capability-state taxonomy.
- Phase 5.2: Roll out capability states for remaining blockers.
- Phase 6.1: Replay first-time user QA.
- Phase 6.2: Replay power-user QA.
- Phase 6.3: Final Nielsen heuristic closeout.

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

QA walkthroughs are not optional. A task or phase that only passes unit, route, or render tests remains in progress.

Durable QA evidence files should live under:

```text
Docs/superpowers/qa/unified-shell/
├── README.md
├── phase-0/
├── phase-1/
├── phase-2/
├── phase-3/
├── phase-4/
├── phase-5/
└── phase-6/
```

Each phase directory should contain a concise markdown summary for each QA walkthrough. Large screenshots, videos, or probe artifacts may stay outside git when necessary, but the repo-tracked summary must include enough detail to understand the result without opening temporary files.

## Backlog Initialization Requirements

If `backlog/` is absent when Phase 0 starts, initialize it from a clean worktree:

```bash
backlog init --defaults --agent-instructions agents --integration-mode cli --backlog-dir backlog
```

Then validate the initialized tracker before creating many tasks:

```bash
backlog task list --plain
backlog overview
find backlog -maxdepth 2 -type f | sort
```

Expected:

- `backlog/` exists with task, doc, decision, and config files expected by the installed Backlog.md version.
- Backlog commands run without interactive setup prompts.
- Generated files are visible in `git status`.
- No generated file conflicts with existing project docs.

After validation, create tasks with `backlog task create` and use `--plain` for reads in agent workflows.

Do not initialize Backlog.md in a stale or dirty root checkout.

## Update Rules

- Roadmap changes must happen through PRs.
- Backlog task IDs must be linked from the roadmap.
- Backlog tasks must not be marked `Done` until acceptance criteria, automated verification, implementation notes, and QA walkthrough evidence are complete.
- Phase status must be one of: `not-started`, `in-progress`, `implemented`, `qa-needed`, `qa-failed`, `verified`, or `blocked`.
- `implemented` means code or docs landed but app-level QA has not passed.
- `qa-needed` means automated checks passed and manual walkthrough is pending.
- `qa-failed` means the running app walkthrough found a blocker or workflow-degradation issue.
- `verified` means both automated checks and manual QA evidence are complete.
- Render-only, mount-only, and click-event-only tests do not count as workflow completion.
- The roadmap should preserve historical evidence without pretending stale plans are current.

## Non-Goals

- Do not redesign the whole app visually.
- Do not replace existing route IDs.
- Do not rewrite already-merged Chat handoff architecture.
- Do not create phase-sized mega tasks in Backlog.md.
- Do not mark wrappers complete until they support meaningful workflows or honest recovery states in the running app.
