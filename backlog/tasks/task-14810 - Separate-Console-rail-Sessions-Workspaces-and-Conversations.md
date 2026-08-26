---
id: TASK-14810
title: 'Separate Console rail Sessions, Workspaces, and Conversations'
status: In Progress
assignee: []
created_date: '2026-08-10 06:01'
updated_date: '2026-08-10 06:05'
labels:
  - console ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Redesign the Console left context rail so sessions remain first, followed by distinct collapsible Workspaces and Conversations sections, using a compact desktop-sidebar hierarchy while preserving current context switching behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Sessions remains the first rail section.
- [x] #2 Workspaces renders as its own collapsible section below Sessions.
- [x] #3 Conversations renders as its own collapsible section below Workspaces.
- [x] #4 Workspace and conversation actions preserve existing behavior and active-state semantics.
- [x] #5 Focused tests and rendered UI evidence verify section order, labels, collapse behavior, and geometry.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Characterize the existing rail state, workspace tray, conversation browser, styles, and mounted tests.
2. Add independent persisted Workspaces and Conversations disclosure state while retaining Sessions first.
3. Project the shared workspace-context snapshot into scoped Sessions, Workspaces, and Conversations trays.
4. Update Console styling, compatibility seams, and focused interaction/geometry tests.
5. Rebuild the CSS bundle, capture rendered evidence, run mutation checks, and document verification.

ADR required: no

ADR path: `backlog/decisions/017-console-left-rail-usability.md`

Reason: This is a direct UX refinement of the existing Console rail architecture and persistence model; it does not introduce a new storage, runtime, service, security, or cross-module boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Split the former mixed Session context into three peer disclosure sections in the requested order: Sessions, Workspaces, then Conversations. Added independent persisted open/closed preferences for the two new sections, scoped the shared context widget so each section owns only its relevant controls, and kept the long-standing conversation-browser id as the compatibility seam for search/resume behavior. Multi-tray refresh now lives in `ConsoleLeftRail`, reducing `chat_screen.py` by seven net lines while keeping all three projections synchronized under the existing run-time flicker guard.

Updated Console component CSS and rebuilt the modular bundle. Added state, mount-order, ownership, independent-collapse, placeholder, and run-tick synchronization coverage. Mutation verification confirmed the ordering test fails if the Workspaces label regresses. Rendered evidence: `Docs/superpowers/qa/task-14810/console-left-rail.svg`.

Verification from an isolated `origin/dev` PR worktree: 117 Console tests passed in the full focused run; its sole fixed-deadline startup timeout passed when rerun alone. Nine CSS build-integrity tests passed, Ruff passed for all task-owned modules and tests, and `chat_screen.py` compiles. Repository-wide collection reaches the broader tree but stops on two Confluence tests because the optional `playwright` dependency is not installed. The architecture ratchet remains red because `HEAD` is already 12 lines over its budget; this change reduces that screen by seven net lines. Three unrelated, pre-existing rail-width-budget assertions also remain red against current `HEAD`. The task remains In Progress rather than Done until repository-level DoD blockers are resolved.

After the PR's first rebase, `dev` advanced with TASK-14807 and introduced its MCP Tools checkbox override directly in the generated CSS bundle. The hosted bundle guard exposed that source/bundle drift on the virtual merge. The final rebase preserves the override in `_agentic_terminal.tcss` and regenerates the bundle so the latest `dev` behavior remains reproducible.

The final Qodo review findings were addressed by migrating legacy `session_open` collapse state into the two new section flags when they are absent, documenting the new public projection APIs, and removing the unused provisional `#console-conversation-context` selector.

The latest-`dev` compatibility run also exposed two stale MCP Workbench test assumptions from TASK-14807's default-on tool switch. The test that specifically exercises the explicit off-to-on save path now seeds an off state, and the empty diagnosis expectation now reflects the eight gates left off plus Tools-mode guidance. Product behavior is unchanged.

The same Textual-floor run exposed a pre-existing deferred-canvas mount race: Textual 8.2.8 may invoke the first load callback before the Workbench's `ContentSwitcher` is queryable. The guarded reload now queues a replacement load one message-pump turn later when the deferred Tools canvas is still absent, preserving the deferred-load design while preventing `NoMatches` from stranding initial state.

The task was renumbered from the provisional TASK-14801 to TASK-14810 after the required all-remote sweep found TASK-14801 already claimed on `origin/codex/roleplay-chat-identity`.

ADR: existing `backlog/decisions/017-console-left-rail-usability.md`; no new ADR required.

TASK-19638 status reconciliation (2026-08-20, latest `origin/dev` at
`a1d6df3f89244e918a1fb12facbd4ed0d927c24c`): the two former Confluence
collection blockers no longer reproduce in the repository development
environment (31 tests collect), and the focused rail-width budget file now
passes all eight tests. The repository architecture gate remains red:
`chat_screen.py` measures 21,292 lines against its unchanged 17,727-line
one-way budget (+3,565). TASK-14810 therefore remains **In Progress**. Its
checked acceptance criteria and shipped implementation are not being
misrepresented as repository-level completion; closing it still requires an
honest resolution of that recorded Definition-of-Done blocker.
<!-- SECTION:NOTES:END -->
