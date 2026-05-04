# Unified Shell Maturity Roadmap

Date: 2026-05-04
Status: Phase 2, Phase 3, and Phase 4 in progress
Source Branch: `origin/dev` at `45fa942c` plus `codex/unified-shell-phase4-library-source-service`

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

- Home active-work controls, detail routing, Console launch requests, item identity, local unread notification counts, notification review routing, local W+C watchlist run snapshots, and local W+C run detail routing now route through explicit adapter boundaries; schedule and agent-service adapters still need implementation.
- Workflows has no wired workflow service in the shell wrapper, but can launch Console when the active-work adapter already has actionable workflow-run context.
- W+C, Schedules, and Workflows now expose Console follow or launch when the existing active-work adapter has actionable run context; Schedules also exposes Console launch for the latest local reading-digest output when no active run is available; Search/RAG result cards can stage selected retrieved evidence into Console; Artifacts can launch the latest local Chatbook artifact into Console; ACP still uses honest unavailable Console states until actionable payloads exist.
- Console now has a typed app-owned launch contract, reusable status-card display seam, source-readiness summary, Home W+C active-work source producer, W+C, Schedules, and Workflows destination producers, a Schedules reading-digest output fallback producer, a RAG search-result producer, an Artifacts Chatbook producer, and W+C run-detail action routing for staged live-work payloads; additional source-specific live event producers still need implementation.
- ACP launch is disabled until an ACP-compatible runtime is configured.
- MCP now adopts the existing Unified MCP management panel in the top-level MCP wrapper; Console MCP live-work launch and deeper service expansion remain future work.
- Skills now lists local Agent Skills through `skills_scope_service` and can stage local skill context into Console; server skills, import, detail, edit, validation, and execution UX remain future work.
- Library now surfaces a local source snapshot from notes, media, and conversations and can stage concrete source context into Console; full Library-native detail views, embedded Import/Export, and embedded Search/RAG remain future work.
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
- Phase 2.4: Wire Home to local notification snapshot adapter - `TASK-4.4`
- Phase 2.5: Route Home notification review to the notifications inbox - `TASK-4.5`
- Phase 2.6: Surface local watchlist runs in Home active work - `TASK-4.6`
- Phase 2.7: Open Home local watchlist run details - `TASK-4.7`
- Phase 3.1: Add Console live-work launch contract - `TASK-3.1`
- Phase 3.2: Add Console live-work status card seam - `TASK-3.2`
- Phase 3.3: Open Home W+C active work in Console - `TASK-3.3`
- Phase 3.4: Route Console W+C live-work actions - `TASK-3.4`
- Phase 3.5: Launch latest W+C run from W+C into Console - `TASK-3.5`
- Phase 3.6: Show Console live-work source readiness - `TASK-3.6`
- Phase 3.7: Launch active Schedules run from Schedules into Console - `TASK-3.7`
- Phase 3.8: Launch RAG search result from Search/RAG into Console - `TASK-3.8`
- Phase 3.9: Launch latest Chatbook artifact from Artifacts into Console - `TASK-3.9`
- Phase 3.10: Launch active Workflows run from Workflows into Console - `TASK-3.10`
- Phase 4.1: Adopt Unified MCP panel in MCP destination - `TASK-5.1`
- Phase 4.2: Adopt Skills services in Skills destination - `TASK-5.2`
- Phase 4.3: Adopt Library source services in Library destination - `TASK-5.3`

## QA Evidence Index

| Phase | Evidence Path | Status |
| --- | --- | --- |
| Phase 0 | `Docs/superpowers/qa/unified-shell/phase-0/` | verified |
| Phase 1 | `Docs/superpowers/qa/unified-shell/phase-1/` | verified |
| Phase 2 | `Docs/superpowers/qa/unified-shell/phase-2/` | in-progress |
| Phase 3 | `Docs/superpowers/qa/unified-shell/phase-3/` | in-progress |
| Phase 4 | `Docs/superpowers/qa/unified-shell/phase-4/` | in-progress |
| Phase 5 | `Docs/superpowers/qa/unified-shell/phase-5/` | not-started |
| Phase 6 | `Docs/superpowers/qa/unified-shell/phase-6/` | not-started |

## Phase Overview

| Phase | Goal | Status | Backlog Tasks | QA Evidence | Residual Risk |
| --- | --- | --- | --- | --- | --- |
| Phase 0: Canonical Tracking | Make remaining work trackable. | verified | `TASK-1`, `TASK-1.1`, `TASK-1.2` | `phase-0/` | Product UI workflows are out of scope for Phase 0. |
| Phase 1: Shell Contract Complete | Remove false shell affordances and prove shell usability. | verified | `TASK-2`, `TASK-2.1`, `TASK-2.2`, `TASK-2.3`, `TASK-2.4` | `phase-1/` | Live service workflows remain intentionally deferred to Phases 2-6. |
| Phase 2: Home Operational Control | Make Home a real dashboard/control surface. | in-progress | `TASK-4`, `TASK-4.1`, `TASK-4.2`, `TASK-4.3`, `TASK-4.4`, `TASK-4.5`, `TASK-4.6`, `TASK-4.7` | `phase-2/` | Schedule and agent-service adapters still need implementation; local watchlist retry/pause/resume remain recoverable rather than fully controllable. |
| Phase 3: Console Live Work Hub | Make Console the live-agent control surface. | in-progress | `TASK-3`, `TASK-3.1`, `TASK-3.2`, `TASK-3.3`, `TASK-3.4`, `TASK-3.5`, `TASK-3.6`, `TASK-3.7`, `TASK-3.8`, `TASK-3.9`, `TASK-3.10` | `phase-3/` | ACP, MCP, server Artifacts, and deeper source-specific live event streams still need implementation. |
| Phase 4: Destination Service Adoption | Turn wrappers into useful product surfaces. | in-progress | `TASK-5`, `TASK-5.1`, `TASK-5.2`, `TASK-5.3` | `phase-4/` | Service coverage varies by destination; MCP now adopts the existing Unified MCP panel, Skills now lists/stages local Agent Skills, and Library now lists/stages local source snapshots, but remaining destinations and deeper Library/Skills flows still need adoption or verified recovery. |
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

## Phase 2.4 Home Local Notification Snapshot Evidence

`TASK-4.4` wires Home to the local unread notification queue without creating false active-work controls:

- `Docs/superpowers/qa/unified-shell/phase-2/2026-05-03-home-local-notification-snapshot.md`

## Phase 2.5 Home Notification Review Routing Evidence

`TASK-4.5` makes Home notification review actions lead into the existing notifications inbox:

- `Docs/superpowers/qa/unified-shell/phase-2/2026-05-03-home-notification-review-routing.md`

## Phase 2.6 Home Local Watchlist Run Snapshot Evidence

`TASK-4.6` surfaces queued, running, and failed local W+C watchlist runs as Home active work:

- `Docs/superpowers/qa/unified-shell/phase-2/2026-05-03-home-local-watchlist-run-snapshot.md`

## Phase 2.7 Home Local Watchlist Run Details Evidence

`TASK-4.7` makes Home local W+C watchlist run detail controls open the selected run in the W+C runs surface:

- `Docs/superpowers/qa/unified-shell/phase-2/2026-05-03-home-local-watchlist-run-details.md`

## Phase 3.1 Console Live-Work Launch Contract Evidence

`TASK-3.1` adds the normalized Console launch payload contract and renders pending launch details in Console:

- `Docs/superpowers/qa/unified-shell/phase-3/2026-05-03-console-live-work-launch-contract.md`

## Phase 3.2 Console Live-Work Status Card Evidence

`TASK-3.2` extracts reusable Console live-work status card state and stable render selectors:

- `Docs/superpowers/qa/unified-shell/phase-3/2026-05-03-console-live-work-status-card-seam.md`

## Phase 3.3 Home W+C Active-Work Console Launch Evidence

`TASK-3.3` makes Home W+C active-work rows produce a real Console live-work launch context:

- `Docs/superpowers/qa/unified-shell/phase-3/2026-05-03-home-wc-active-work-console-launch.md`

## Phase 3.4 Console W+C Action Routing Evidence

`TASK-3.4` makes supported Console W+C live-work actions route to existing W+C run details:

- `Docs/superpowers/qa/unified-shell/phase-3/2026-05-03-console-wc-action-routing.md`

## Phase 3.5 W+C Destination Console Launch Evidence

`TASK-3.5` makes W+C itself expose Console follow for the latest active W+C run when adapter context exists:

- `Docs/superpowers/qa/unified-shell/phase-3/2026-05-03-wc-destination-console-launch.md`

## Phase 3.6 Console Source Readiness Evidence

`TASK-3.6` shows W+C and Schedules as connected in Console while keeping planned future live-work sources honestly unavailable:

- `Docs/superpowers/qa/unified-shell/phase-3/2026-05-03-console-live-work-source-readiness.md`

## Phase 3.7 Schedules Destination Console Launch Evidence

`TASK-3.7` makes Schedules itself expose Console follow for active schedule runs when adapter context exists:

- `Docs/superpowers/qa/unified-shell/phase-3/2026-05-03-schedules-console-launch.md`

## Phase 3.7 Schedules Reading Digest Console Launch Evidence

`TASK-3.7` also lets Schedules stage the latest local reading-digest output in Console when no active schedule-run context exists:

- `Docs/superpowers/qa/unified-shell/phase-3/2026-05-03-schedules-digest-console-launch.md`

## Phase 3.8 RAG Search Console Launch Evidence

`TASK-3.8` lets Search/RAG result cards stage selected retrieved evidence in Console while preserving existing Use in Chat behavior:

- `Docs/superpowers/qa/unified-shell/phase-3/2026-05-03-rag-search-console-launch.md`

## Phase 3.9 Artifacts Chatbook Console Launch Evidence

`TASK-3.9` lets Artifacts stage the latest local Chatbook artifact in Console while preserving the existing Chatbooks destination route:

- `Docs/superpowers/qa/unified-shell/phase-3/2026-05-03-artifacts-chatbook-console-launch.md`

## Phase 3.10 Workflows Destination Console Launch Evidence

`TASK-3.10` makes Workflows itself expose Console launch for active workflow runs when adapter context exists:

- `Docs/superpowers/qa/unified-shell/phase-3/2026-05-03-workflows-console-launch.md`

## Phase 4.1 MCP Destination Service Adoption Evidence

`TASK-5.1` embeds the existing Unified MCP management panel in the top-level MCP destination while preserving the legacy `tools_settings` MCP alias:

- `Docs/superpowers/qa/unified-shell/phase-4/2026-05-04-mcp-destination-service-adoption.md`

## Phase 4.2 Skills Destination Service Adoption Evidence

`TASK-5.2` makes the top-level Skills destination list local Agent Skills through `skills_scope_service` and stage concrete local skill context into Console:

- `Docs/superpowers/qa/unified-shell/phase-4/2026-05-04-skills-destination-service-adoption.md`

## Phase 4.3 Library Source Service Adoption Evidence

`TASK-5.3` makes the top-level Library destination list local notes, media, and conversations through existing source services and stage concrete local source context into Console:

- `Docs/superpowers/qa/unified-shell/phase-4/2026-05-04-library-source-service-adoption.md`

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
