# Product Maturity Phased Roadmap Design

Date: 2026-05-05
Status: User-approved design; spec review approved; pending implementation planning
Primary Repo: `tldw_chatbook`
Source Baseline: `dev` at `9c4b3bf0` (`Close Phase 6 Nielsen shell audit (#246)`)
Related Specs:

- `Docs/superpowers/specs/2026-05-02-new-user-first-run-shell-ux-design.md`
- `Docs/superpowers/specs/2026-05-02-agentic-terminal-design-system-design.md`
- `Docs/superpowers/specs/2026-05-03-unified-shell-maturity-tracking-design.md`
- `Docs/superpowers/trackers/unified-shell-maturity-roadmap.md`

## Summary

The Unified Shell maturity roadmap is verified through Phase 6. The next workstream should move beyond shell correctness into product-depth maturity: making each visible module useful for real workflows while preserving the verified shell guardrails.

The roadmap should use a hybrid maturity-gate model:

- 6 major phases for executive-level tracking.
- PR-sized child gates for execution.
- mandatory QA walkthrough evidence for every phase.
- explicit "usable, not merely rendered" exit criteria.

The first phase must establish a stronger QA baseline before new feature depth. This prevents repeating the earlier failure mode where screens render, controls are clickable, but the app is visually broken, confusing, or unusable.

## Current Baseline

The current `dev` branch has verified Unified Shell work:

- Home is the default dashboard/status surface.
- Console is the primary live agentic control surface.
- Top-level destinations exist for Home, Console, Library, Artifacts, Personas, W+C, Schedules, Workflows, MCP, ACP, Skills, and Settings.
- Destination wrappers expose honest unavailable or recovery states instead of false affordances.
- Console has an app-owned launch contract and multiple staged-context producers.
- Local Library, Personas, Skills, and W+C snapshots can stage context into Console.
- MCP adopts the existing Unified MCP management panel.
- ACP remains intentionally unavailable until an ACP-compatible runtime is configured.
- Phase 6 first-time user, power-user, and Nielsen heuristic evidence is recorded under `Docs/superpowers/qa/unified-shell/phase-6/`.

This baseline proves the shell direction. It does not prove that every product module has deep, complete, local/server-parity workflows.

## Problem Statement

The app now has a coherent shell, but the remaining work is fragmented across product depth, local/server parity, and reliability:

- Library still needs native detail, import/export, and embedded Search/RAG workflows.
- Console needs deeper live-work producers, durable run context, and stronger agent-control ergonomics.
- Artifacts and Chatbooks need stronger persistence, reopen, export, and reuse paths.
- Personas need detail, edit, import/export, archetypes, exemplars, dictionaries, and lore workflows.
- Skills need Agent Skills discovery, import, detail, validation, edit, and execution workflows.
- W+C needs detail, edit, import/export, feed/WebSub, alert rules, retry/backoff, and server collection-feed UX.
- Schedules and Workflows need service adapters and usable execution/recovery controls.
- MCP needs deeper live-work integration with Console.
- ACP needs its own runtime/session model, separate from MCP.
- Workspaces, flashcards, and quizzes need explicit placement in the product model.
- Live server/API paths and optional dependency states still need broader verification.

The next roadmap must make this work trackable without turning it into one large, unreviewable plan.

## Goals

- Preserve the verified Unified Shell operating model.
- Prioritize actual workflow completion over visual redesign.
- Keep Console as the agentic programming/control interface, similar in role to Claude Code, Codex, or Gemini CLI.
- Keep Home useful as a notification center, status page, and lightweight control dashboard.
- Mature visible modules into real workflows in small, independently shippable gates.
- Keep Workspaces, flashcards, quizzes, sources, Artifacts/Chatbooks, Personas, Skills, W+C, Schedules, Workflows, MCP, ACP, Search/RAG, media, notes, and handoffs explicit in planning.
- Add beginner orientation without slowing power users.
- Require QA walkthrough evidence that proves the running app is usable, not just rendering.
- Maintain clear local parity targets with the adjacent `tldw_server2` checkout, representing the server capability model previously discussed as `tldw_server`.

## Non-Goals

- Do not reopen the completed Unified Shell Phase 0-6 work unless a regression is found.
- Do not implement concept images as literal screen requirements.
- Do not redesign for aesthetics alone.
- Do not collapse MCP and ACP.
- Do not turn every destination into an alternate live-agent surface.
- Do not create broad rewrites or large task batches before PR-sized gates are defined.
- Do not treat automated tests alone as proof of UX completion.

## Operating Model

Each phase must define:

- **Product outcome**: what becomes meaningfully usable for a real user.
- **Scope boundary**: what is explicitly deferred.
- **PR-sized gates**: independently reviewable slices.
- **Acceptance criteria**: user-visible outcomes, not implementation steps.
- **QA walkthrough**: clean-user and power-user checks proving workflow completion.
- **Regression coverage**: focused tests for the changed seams.
- **Evidence artifact**: repo-tracked QA note under `Docs/superpowers/qa/`.
- **Exit gate**: the phase is not done until the intended workflows can be completed end-to-end or blocked states are understandable and recoverable.

Backlog.md should track execution after this spec is accepted:

- one parent task per phase.
- child tasks only for PR-sized gates.
- child tasks should include "QA walkthrough verifies the running app is usable" as acceptance criteria.
- phase completion requires evidence, not just task status.

## Severity Mapping

Product-maturity QA should keep the existing Unified Shell defect taxonomy and map P0/P1 closure rules to it:

| Priority | Taxonomy | Meaning | Phase Exit Rule |
| --- | --- | --- | --- |
| P0 | `blocker` | Prevents basic use, traps the user, corrupts or loses user work, or makes a required workflow impossible. | Must be fixed before the phase or gate can close. |
| P1 | `workflow-degradation` | Breaks or seriously slows a core workflow but leaves a workaround. | Must be fixed before phase close unless explicitly accepted with owner, rationale, and follow-up task. |
| P2 | `recoverability` | A blocked or error state exists, but recovery copy, ownership, or next action is unclear. | Can remain only with documented residual risk and a scoped follow-up. |
| P3 | `polish` | Visual, wording, density, or minor interaction issue that does not block completion. | Can remain as backlog polish if it does not hide status, source authority, or recovery action. |

QA evidence should use one taxonomy label per defect and may also include the P-level when useful for release decisions.

## Product Model

The roadmap should preserve this product model:

- **Home**: cross-module status, notifications, active work, next-best actions, and lightweight approve/pause/retry/open controls.
- **Console**: live chat, agentic programming/control, RAG answers, staged context, tool calls, approvals, MCP tool use, ACP sessions, run logs, recovery, and artifact creation.
- **Library**: notes, media, conversations, imports/exports, Search/RAG, source browsing, metadata, and evidence.
- **Artifacts**: generated outputs, reusable deliverables, Chatbooks, reports, datasets, drafts, and exports.
- **Personas**: behavior configuration, characters, profiles, archetypes, exemplars, dictionaries, and lore.
- **W+C**: Watchlists and Collections, with local parity to server watch/collection concepts.
- **Schedules**: when work runs, health, retries, pause/resume, and execution history.
- **Workflows**: what procedure runs, steps, inputs, outputs, approvals, and recovery.
- **MCP**: external tool/resource protocol management and readiness.
- **ACP**: separate agent/session protocol runtime and live agent collaboration model.
- **Skills**: Agent Skills-compatible capability packs built around `SKILL.md`, bundled resources, validation, import, and attachment to Console/workflows.
- **Settings**: global configuration, providers, privacy, runtime policy, optional dependencies, and diagnostics.

Workspaces and Collections must not become duplicate grouping concepts:

- **Workspaces** define broad user context and scope across sources, chats, notes, artifacts, study outputs, settings, and long-running work.
- **Collections** define reusable source sets inside W+C/Library that can feed RAG, study generation, schedules, workflows, and monitoring.
- A source can belong to a workspace and one or more collections, but implementation plans must identify which surface owns create, edit, and review actions for each grouping type.

## Phase Roadmap

| Phase | Product Outcome | Done When |
| --- | --- | --- |
| Phase 1: QA Baseline And Usability Guardrails | The app can be launched, navigated, visually inspected, and exercised from a clean user state with reliable evidence. | A fresh user can navigate every top-level module without broken layout, dead controls, confusing empty states, or unclear blocked states; at least one narrow core loop is exercised end-to-end. |
| Phase 2: Core Agentic Loop | The main value loop works: source/question -> grounded Console interaction -> saved/reopened Artifact or Chatbook. | User can start from Library/RAG or Console, produce grounded output, save it, reopen it from Artifacts/Home, and recover from missing-source or missing-runtime states. |
| Phase 3: Knowledge And Study Workflows | Knowledge workflows become coherent: ingest, organize, retrieve, study, and reuse. | User can move through Library, Import/Export, notes, media, Search/RAG, Workspaces, flashcards, quizzes, and collections without losing context or hitting placeholder UI. |
| Phase 4: Agent Configuration And Execution | Agent setup and run control become coherent across Personas, Skills, MCP, ACP, Schedules, and Workflows. | User can configure an agent run, understand tool/runtime readiness, approve/pause/retry work, and diagnose failure states. |
| Phase 5: Server-Parity And Live Integrations | High-value local/server parity gaps are closed where they materially improve local use. | Local Chatbook has documented parity coverage for priority `tldw_server2` workflows, live adapters/events where needed, and explicit residual gaps. |
| Phase 6: Release Hardening And Documentation | The product reaches release-candidate usability. | Full first-time, power-user, accessibility/focus, visual, recovery, docs, and regression replays pass or produce tracked defects. |

## Phase 1: QA Baseline And Usability Guardrails

Purpose:

Create the quality floor for all later product-depth work.

PR-sized gates:

- Clean first-run launch and configuration walkthrough.
- Top-level navigation smoke for Home, Console, Library, Artifacts, Personas, W+C, Schedules, Workflows, MCP, ACP, Skills, and Settings.
- Keyboard/focus sweep for navigation, command palette, forms, tables, and major actions.
- Visual broken-state audit for clipped panels, unreadable labels, low-contrast states, and misleading disabled controls.
- Empty/error/setup-state coverage for missing API keys, optional dependencies, server connections, and local runtime blockers.
- Narrow core-loop proof: one source or query can reach Console and produce a recoverable output or honest blocker.

First execution boundary:

- The first implementation plan should cover roadmap task setup plus **Phase 1.1: Canonical Product-Maturity QA Harness** only.
- Phase 1.1 should create the reusable clean-run protocol, QA evidence template, severity mapping, selected terminal-size matrix, and one smoke command or test entry point.
- Phase 1.1 should not attempt to complete the full first-run, focus, visual, empty-state, and core-loop audits in one PR.
- Later Phase 1 child tasks should each own one independently reviewable gate from the list above.

Done when:

- A fresh-user walkthrough can be repeated deterministically.
- The QA artifact records what was tested, what passed, what failed, and what remains uncertain.
- Focused regression tests cover the highest-risk async/loading paths.
- Any P0 or P1 usability defect discovered during the walkthrough is fixed before the phase closes.

## Phase 2: Core Agentic Loop

Purpose:

Make the central product loop complete enough to be useful every day.

PR-sized gates:

- Library/RAG search and source selection can stage evidence into Console.
- Console can ask/answer with visible grounded context and source authority.
- Console can create or update an Artifact/Chatbook.
- Artifacts can reopen the result and route it back into Console.
- Home can show recent/active work and provide resume/open controls.
- Recovery states cover missing source, missing provider, missing model, missing runtime, and failed generation.

Done when:

- A user can complete source/question -> grounded chat -> saved Artifact/Chatbook -> reopen/resume without manual state reconstruction.
- Power-user flow supports keyboard-first repeated use.
- Beginner flow exposes enough labels/help text to understand what happened and what to do next.

## Phase 3: Knowledge And Study Workflows

Purpose:

Turn Library and study-adjacent surfaces into a coherent knowledge workbench.

PR-sized gates:

- Library-native detail views for notes, media, conversations, and imported sources.
- Import/Export entry points under Library with progress, validation, and recovery.
- Search/RAG as both deliberate Library workflow and Console-integrated retrieval mode.
- Workspaces as scope containers for sources, chats, notes, artifacts, and study outputs.
- Flashcards from selected sources or Console outputs.
- Quizzes from selected sources or Console outputs.
- Collections as reusable source sets for RAG, study, schedules, and workflows.

Done when:

- A user can ingest or select material, organize it in a workspace or collection, retrieve from it, generate study material, and reuse it in Console.
- The UI distinguishes source material, generated outputs, and study derivatives.
- Import/export failures are diagnosable and recoverable.

## Phase 4: Agent Configuration And Execution

Purpose:

Make agent configuration, tools, skills, schedules, and workflows understandable and controllable.

PR-sized gates:

- Personas detail/edit/import/export workflows.
- Skills discovery/import/detail/validation/edit workflow following the Agent Skills `SKILL.md` model.
- Skills attachment to Console, Workflows, or Personas where appropriate.
- MCP readiness, available tools/resources, and Console launch/follow integration.
- ACP runtime/session readiness, separate from MCP.
- Schedules service adapter with pause/resume/retry/history.
- Workflows service adapter with step status, inputs, outputs, approvals, and recovery.
- Home active-work controls connected to real schedule/workflow/agent items.

Done when:

- A user can configure the behavior, capabilities, tools, schedule, and procedure for an agent-assisted task.
- Runtime readiness is visible before launch.
- Failures show cause, impact, and recovery action.
- Console remains the live execution surface.

## Phase 5: Server-Parity And Live Integrations

Purpose:

Close the highest-value local/server parity gaps without losing local-first usability.

PR-sized gates:

- Compare Chatbook modules against adjacent `tldw_server2` capabilities.
- Prioritize parity gaps by user value and local feasibility.
- Wire server-backed Library/Search/RAG capabilities where useful.
- Expand W+C watchlist, collection-feed, alert-rule, feed/WebSub, retry/backoff, and history behavior.
- Add server/local sync or import paths for Personas, Skills, sources, and artifacts where appropriate.
- Add live event producers for Console and Home where services support them.
- Record explicit parity coverage and residual gaps.

Done when:

- Highest-value server-backed workflows have local equivalents, server-backed paths, or honest unavailable states.
- Users can see whether local, server, workspace, remote-only, or dry-run authority applies.
- Remaining parity gaps are documented and not hidden behind misleading UI.

## Phase 6: Release Hardening And Documentation

Purpose:

Convert the matured product into a release-candidate state.

PR-sized gates:

- Full first-time user walkthrough.
- Full power-user replay across at least 5 core workflows.
- Nielsen heuristic audit replay.
- Keyboard/focus and accessibility sweep.
- Visual polish against the agentic terminal design system.
- Error/recovery copy review.
- Docs/onboarding/help updates.
- Regression suite hardening and CI review.

Done when:

- Core workflows complete in the running app.
- Layout is coherent across supported terminal sizes.
- Failure states are actionable.
- Docs match current behavior.
- P0 and P1 defects are closed or explicitly accepted with owner and follow-up.

## Workflow Matrix For Planning

Each implementation plan should preserve these workflow families:

| Workflow | Primary Modules | Expected Result |
| --- | --- | --- |
| First-run orientation | Home, Settings, Console, Library | User knows what is configured, what is missing, and where to start. |
| Grounded answer | Library, Search/RAG, Console, Artifacts | User asks against sources, sees evidence, and saves/reopens output. |
| Study loop | Library, Workspaces, flashcards, quizzes, Console | User turns source material into study artifacts and can reuse context. |
| Agent run | Personas, Skills, MCP, ACP, Schedules, Workflows, Console | User configures, launches, approves, monitors, and recovers agentic work. |
| Monitoring loop | W+C, Schedules, Home, Console, Artifacts | User watches sources, reviews changes, acts on alerts, and saves outputs. |
| Recovery loop | Home, Console, Settings, affected module | User understands blocker cause and can retry, configure, pause, or route elsewhere. |

## QA And Evidence Rules

Every child task should include:

- focused automated tests for the changed seam.
- manual walkthrough of the running app.
- keyboard/focus check when controls or navigation change.
- visual check for layout, label clarity, and disabled/recovery states.
- clean-environment or empty-state check when setup/configuration is affected.
- QA note under `Docs/superpowers/qa/product-maturity/phase-N/`.

Suggested QA artifact structure:

```markdown
# Phase N Gate QA

Date:
Branch/Commit:
Task:
Workflow:

## What Was Verified

## Automated Evidence

## Manual Walkthrough

## Visual/Focus Notes

## Defects Found

## Residual Risk

## Exit Decision
```

## Backlog Planning Model

After this spec is accepted, create Backlog.md tasks with this shape:

- parent task for each phase.
- child task for each gate only when it can fit in one PR.
- each child task must be atomic, testable, and independently reviewable.
- child tasks must not depend on future uncreated tasks.
- every task acceptance criteria should include a QA walkthrough proving functionality works in the running app.

Suggested phase labels:

- `product-maturity`
- `phase-1-qa-baseline`
- `phase-2-core-agentic-loop`
- `phase-3-knowledge-study`
- `phase-4-agent-execution`
- `phase-5-server-parity`
- `phase-6-release-hardening`

## Risks And Mitigations

- **Risk: roadmap becomes too broad to execute.** Mitigate by creating only PR-sized child tasks and keeping future child tasks uncreated until scoped.
- **Risk: QA becomes documentation-only.** Mitigate by requiring running-app walkthrough evidence and focused regression tests before completion.
- **Risk: Console absorbs too much configuration.** Mitigate by keeping destinations responsible for setup/preparation and Console responsible for live execution.
- **Risk: local and server authority blur.** Mitigate with explicit source-authority labels and parity notes.
- **Risk: beginner guidance slows power users.** Mitigate with progressive disclosure, command palette discoverability, and keyboard-first flows.
- **Risk: ACP and MCP collapse conceptually.** Mitigate by keeping MCP as tools/resources and ACP as agent/session runtime.

## Open Questions For Implementation Planning

- Which clean-run harness should become canonical for product-maturity QA?
- Which single core-loop proof should anchor Phase 1?
- Which `tldw_server2` parity inventory should Phase 5 use as the source of truth?
- Which terminal sizes should be mandatory for visual QA?
- Which five power-user workflows should become recurring release replay tests?

## Planning Decisions Required Before Phase 1 Execution

Resolve these before creating implementation tasks beyond the Phase 1 parent and Phase 1.1 harness task:

| Decision | Recommended Default | Why It Matters |
| --- | --- | --- |
| Canonical clean-run harness | Fresh `HOME`/`XDG_*` run using the existing Textual test pilot where possible, with manual terminal replay when visual/focus behavior cannot be proven in tests. | Prevents app-state leakage and makes first-run regressions reproducible. |
| Phase 1 core-loop proof | Library or Search/RAG source/query stages context into Console and either produces a grounded answer path or an honest missing-runtime blocker. | Anchors QA in the product center without overloading Phase 1 with full feature depth. |
| Terminal-size matrix | Minimum supported compact size, common laptop terminal, and large power-user workspace. | Prevents layouts that only work on the developer's current terminal dimensions. |
| Recurring power-user workflows | Grounded answer, study loop, agent run, monitoring loop, and recovery loop from the workflow matrix. | Keeps later QA tied to repeated expert usage, not shallow navigation. |
| Severity policy | Use the P0/P1/P2/P3 mapping in this spec. | Ensures the team knows which findings block phase completion. |

## Transition Criteria To Implementation Planning

Proceed to implementation planning only after:

- this spec is reviewed and accepted.
- any spec-review issues are resolved.
- the first Backlog.md parent/child task set is approved.
- Phase 1 scope is narrowed to roadmap task setup plus Phase 1.1 only.
