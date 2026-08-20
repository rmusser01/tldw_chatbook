---
id: TASK-397
title: Command palette fast Down+Enter can dismiss without running the command
status: In Progress
assignee: []
created_date: '2026-07-20 18:45'
updated_date: '2026-08-20 15:25'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Observed during live TUI verification (2026-07-20, tmux-driven session): open the command palette (Ctrl+P), type a query that still has SEVERAL matching commands (e.g. "logs"), then press Down and Enter in quick succession (~1s apart) — the palette closed without running the highlighted command and the app stayed on the current screen. Retyping a narrower query that left exactly ONE match, then Down+Enter, worked reliably every time. Likely a race between the palette's async result refresh and selection state (Textual's built-in CommandPalette), but worth reproducing under pilot control to determine whether it is upstream Textual behavior or something in our providers (e.g. commands being re-yielded and resetting the highlight). Fast keyboard users will hit this as "the palette ate my command".
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Reproduced (or ruled out) under a pilot-driven test: type a multi-hit query, Down+Enter while results are still refreshing
- [ ] #2 If ours: highlighted command runs even when selection races the result refresh; if upstream: issue filed/linked and any feasible mitigation noted on this task
- [ ] #3 A narrow app-side mitigation freezes an actionable visible command list on keyboard navigation or Enter selection, so the acted-on command runs exactly once while provider results are still arriving without blocking initial or replacement-query results
<!-- AC:END -->

## Implementation Plan

Follow [the reviewed executable plan](../../Docs/superpowers/plans/2026-08-20-task-397-command-palette-selection-race.md): characterize the stock Textual refresh race with a deterministic mounted harness, add the minimal actionable-snapshot compatibility subclass, wire `TldwCli` to that palette, run only related regression/static checks, file or link the upstream Textual issue, obtain cumulative review, and close the task only when every AC and DoD item has evidence.

ADR required: no.

ADR path: N/A.

Reason: this localized compatibility shim preserves the existing provider and application boundaries and changes no storage, sync, security, dependency, or service contract.
