---
id: TASK-21507
title: Add Research shell authority and responsive Workspace foundation
status: Done
assignee:
  - '@codex'
created_date: '2026-08-24 05:54'
updated_date: '2026-08-24 10:08'
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
- [x] #1 Research is a fourteenth shell destination after Library and before Artifacts; its primary route is `research_workspace`, direct `research` callers still mount Research Runs, existing destination shortcuts do not shift, and F10 opens Research Workspace.
- [x] #2 Workspace and Runs mount one shared Research mode bar and navigate between separate real screens while preserving each screen's own saved state.
- [x] #3 The Workspace screen exposes an explicit `Workspace data: Local | Server` selector and authority-qualified workspace identities; unavailable Server state fails closed with recovery and never reads, displays, or mutates Local as fallback.
- [x] #4 Local and Server adapters implement one normalized, capability-aware read/lifecycle contract without reusing `WorkspaceAuthority` as the data-source discriminator or merging results from both owners.
- [x] #5 Sources and Studio use exact `<---` / `--->` collapse/reveal labels, deterministic focus relocation, separate stored preference versus effective responsive state, and Chat maximizes when both side panes are closed.
- [x] #6 Wide, medium, narrow, and short-height layouts meet the approved pane-count and minimum-content contracts at the exact verification sizes, with hidden panes removed from the focus cycle and responsive overrides restoring preferences when width returns.
- [x] #7 A private, bounded, atomically written device-overlay store keys pane preferences by qualified authority/profile/principal/workspace identity, recovers per-record corruption without blocking the canonical workspace, and stores no secrets or canonical content.
- [x] #8 Targeted unit, mounted Textual, command-palette, navigation, persistence, inverse, and geometry checks pass; generated CSS is rebuilt from source and no full-suite claim is made.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR

ADR path: `backlog/decisions/078-research-workspace-authority-and-screen-boundaries.md`

Reason: This task directly implements the accepted shell, authority, adapter, overlay, and responsive-layout boundaries in ADR-078. A new ADR is required only if implementation changes those owners or allows automatic cross-authority fallback.

Follow `Docs/superpowers/plans/2026-08-23-research-workspace-foundation.md` task-by-task with test-first checkpoints and one scoped commit per completed plan task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

Completed the ADR-078 foundation with stable Research routing, normalized Local
and Server adapters, qualified controller state, private device-only pane
preferences, responsive Workspace/Run screens, and an explicit mounted
`Workspace data: Local | Server` selector. Task 5 late-binds only the active
foundation services; unavailable Server remains selected with typed recovery
and makes no Local fallback call. The Task 5 file-list deviation is limited to
the existing controller/header/screen/CSS modules required to make AC #3 a real
interactive selector, plus the user-guide index needed to expose the new page.

Focused Task 1–5 verification passed (`308 passed`, one known
`RequestsDependencyWarning`); all six required inverse mutations went RED, CSS
reproduction, Ruff, format, compile, diff, detector, and rendered geometry
checks passed. The detector's brief-relative `.agents` path is absent from the
isolated worktree, so the identical repository detector was run from the main
checkout with no findings. Known unrelated Library failures in the broad
navigation file were excluded by the approved selected-node boundary, and
full pytest was not run. No new ADR was required; ADR-078 remains the governing
decision.

### Fix Round 1

Added explicit Workspace snapshot/restore for selected authority, qualified
workspace intent, active responsive pane, and pane preferences, verified through
the real Workspace → Runs → Workspace navigation boundary without changing
Runs' independent state. Catalog loads now carry mandatory monotonic generation
and controller revision metadata; stale Local results are rejected at both
controller storage and screen paint across Local → Server → Local ABA changes.

Qualified pane preferences reset before a different workspace's overlay loads,
and overlay application is fenced by returned ref, selected ref, controller
revision, and overlay generation. Medium companion changes now persist through
the same optimistic overlay store. Production CSS distinguishes inactive focus
from active authority selection, and rendered geometry directly proves Chat
uses the wide grid minus two fixed four-cell reveal handles when both side panes
are closed.

Fix Round 1 verification passed `318 passed` with only the accepted
`RequestsDependencyWarning`; all six inverse guards passed separately, and CSS
reproduction, Ruff, format, compile, diff, detector, and affected rendered
geometry checks passed. ADR-078 remains the governing decision, no new ADR was
required, the known unrelated Library failures remain out of scope, and full
pytest was not run.

### Fix Round 2

Replaced cancellation-based overlay persistence with one bounded, coalescing
save drain per mounted Workspace screen. Threaded writes now finish in commit
order; each queued write captures the current qualified owner, pane preferences,
and optimistic revision only after entering serialization. Revision results are
accepted only while both qualified ref and monotonic owner generation still
match, so an authority switch cannot apply an old owner's completion to the new
workspace.

Deterministic mounted coverage pauses the first save after its atomic commit,
then makes a second medium companion choice. The final UI preference persists
at revision 2 without a conflict warning. A separate authority-switch case
proves queued work recaptures Server ownership and does not rewrite the prior
Local record. The production Workspace → Runs → Workspace regression now waits
for catalog/overlay reconciliation and verifies Server identity, selector state,
pane arrangement/status, saved presentation context, and independent Runs
state.

Fix Round 2 verification passed `320 passed` with only the accepted
`RequestsDependencyWarning`; the amended slice passed `125 passed`, all six
inverse guards passed, and CSS reproduction, Ruff, format, compile, diff,
detector, and affected rendered geometry checks passed. ADR-078 remains the
governing decision, no new ADR was required, known unrelated Library failures
remain out of scope, and full pytest was not run.
