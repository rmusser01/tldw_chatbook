---
id: TASK-21507
title: Add Research shell authority and responsive Workspace foundation
status: To Do
assignee: []
created_date: '2026-08-24 05:54'
updated_date: '2026-08-24 05:54'
labels:
  - research
  - workspace
  - ux
  - architecture
dependencies: []
references:
  - Docs/superpowers/specs/2026-08-23-research-workspace-design.md
  - Docs/superpowers/plans/2026-08-23-research-workspace-foundation.md
  - backlog/decisions/078-research-workspace-authority-and-screen-boundaries.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the durable Research destination and a real Workspace screen whose Local/Server authority, responsive pane layout, and device-only presentation state are explicit and safe foundations for later Sources, Chat, and Studio workflows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Research is a fourteenth shell destination after Library and before Artifacts; its primary route is `research_workspace`, direct `research` callers still mount Research Runs, existing destination shortcuts do not shift, and F10 opens Research Workspace.
- [ ] #2 Workspace and Runs mount one shared Research mode bar and navigate between separate real screens while preserving each screen's own saved state.
- [ ] #3 The Workspace screen exposes an explicit `Workspace data: Local | Server` selector and authority-qualified workspace identities; unavailable Server state fails closed with recovery and never reads, displays, or mutates Local as fallback.
- [ ] #4 Local and Server adapters implement one normalized, capability-aware read/lifecycle contract without reusing `WorkspaceAuthority` as the data-source discriminator or merging results from both owners.
- [ ] #5 Sources and Studio use exact `<---` / `--->` collapse/reveal labels, deterministic focus relocation, separate stored preference versus effective responsive state, and Chat maximizes when both side panes are closed.
- [ ] #6 Wide, medium, narrow, and short-height layouts meet the approved pane-count and minimum-content contracts at the exact verification sizes, with hidden panes removed from the focus cycle and responsive overrides restoring preferences when width returns.
- [ ] #7 A private, bounded, atomically written device-overlay store keys pane preferences by qualified authority/profile/principal/workspace identity, recovers per-record corruption without blocking the canonical workspace, and stores no secrets or canonical content.
- [ ] #8 Targeted unit, mounted Textual, command-palette, navigation, persistence, inverse, and geometry checks pass; generated CSS is rebuilt from source and no full-suite claim is made.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR

ADR path: `backlog/decisions/078-research-workspace-authority-and-screen-boundaries.md`

Reason: This task directly implements the accepted shell, authority, adapter, overlay, and responsive-layout boundaries in ADR-078. A new ADR is required only if implementation changes those owners or allows automatic cross-authority fallback.

Follow `Docs/superpowers/plans/2026-08-23-research-workspace-foundation.md` task-by-task with test-first checkpoints and one scoped commit per completed plan task.
<!-- SECTION:PLAN:END -->
