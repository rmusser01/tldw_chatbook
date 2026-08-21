---
id: TASK-397
title: Command palette fast Down+Enter can dismiss without running the command
status: Done
assignee: []
created_date: '2026-07-20 18:45'
updated_date: '2026-08-20 17:05'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Observed during live TUI verification (2026-07-20, tmux-driven session): open the command palette (Ctrl+P), type a query that still has SEVERAL matching commands (e.g. "logs"), then press Down and Enter in quick succession (~1s apart) — the palette closed without running the highlighted command and the app stayed on the current screen. Retyping a narrower query that left exactly ONE match, then Down+Enter, worked reliably every time. Likely a race between the palette's async result refresh and selection state (Textual's built-in CommandPalette), but worth reproducing under pilot control to determine whether it is upstream Textual behavior or something in our providers (e.g. commands being re-yielded and resetting the highlight). Fast keyboard users will hit this as "the palette ate my command".
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reproduced (or ruled out) under a pilot-driven test: type a multi-hit query, Down+Enter while results are still refreshing
- [x] #2 If ours: highlighted command runs even when selection races the result refresh; if upstream: issue filed/linked and any feasible mitigation noted on this task
- [x] #3 A narrow app-side mitigation freezes an actionable visible command list on keyboard navigation or Enter selection, so the acted-on command runs exactly once while provider results are still arriving without blocking initial or replacement-query results
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow [the reviewed executable plan](../../Docs/superpowers/plans/2026-08-20-task-397-command-palette-selection-race.md): characterize the stock Textual refresh race with a deterministic mounted harness, add the minimal actionable-snapshot compatibility subclass, wire `TldwCli` to that palette, run only related regression/static checks, file or link the upstream Textual issue, obtain cumulative review, and close the task only when every AC and DoD item has evidence.

ADR required: no.

ADR path: N/A.

Reason: this localized compatibility shim preserves the existing provider and application boundaries and changes no storage, sync, security, dependency, or service contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a narrow compatibility boundary around Textual 8.2.8's asynchronous
command-palette refresh behavior. A deterministic mounted Pilot test with provider
gates and a fake batch clock proves that a late result rebuild resets a visible
Down selection from the second command to the first and makes Enter run the first
callback exactly once. The controlled ordering did not reproduce the original
palette-dismisses-without-a-callback symptom, so the confirmed defect is reported
as the narrower wrong-command race.

- Added `StableCommandPalette`, which cancels gathering only when a keyboard list
  action targets a visible, non-empty command list whose first option is not
  Textual's disabled no-matches placeholder, then delegates to Textual unchanged.
- Made `TldwCli.action_command_palette()` open that compatibility subclass with
  Textual's canonical `--command-palette` ID and existing enabled/open guards.
- Covered the stock reset, pending and settled exactly-once selection, gather
  cancellation, pre-result navigation, stale no-matches replacement results,
  Escape cancellation, and app construction/duplicate guards.
- Reported the upstream framework behavior with a standalone deterministic
  reproducer and the app-side workaround: [Textual issue #6701](https://github.com/Textualize/textual/issues/6701).
- Final related matrix: `90 passed, 1 warning in 9.13s` across the race, basic,
  provider, and shell-route palette tests. The warning is the environment's
  pre-existing Requests dependency-version warning. Focused Ruff, Ruff format,
  MyPy, compileall, and `git diff --check` all passed.
- Core commits: `42ed24265` / `dc34d5ec3` (deterministic characterization),
  `95ac8cd18` (stable palette), and `86fdebe88` (application integration).

The deliberate tradeoff is that results arriving after the user starts navigating
are omitted from that palette snapshot; changing the query or reopening the palette
starts a fresh gather. No user-guide change was needed because the intended palette
contract is unchanged. No lessons entry was warranted: the task followed the
existing deterministic/non-vacuous testing guidance and did not uncover a new
repository-wide trap.

ADR required: no.

ADR path: N/A.

Reason: the exactly pinned framework compatibility shim preserves existing
provider/application ownership and changes no durable architectural boundary.

Independent cumulative review over merge base `1bf7f234e` through the implementation
HEAD approved the protected Textual seam, guard behavior, deterministic tests,
exactly-once callback contract, app construction, upstream-report honesty, scope,
and this evidence record. Its sole P2 was a stale literal `origin/dev..HEAD` range in
the closeout plan after `origin/dev` advanced by 93 commits; the plan now uses the
merge-base-scoped range. Re-review approved the correction with no open P0-P2
findings.
<!-- SECTION:NOTES:END -->
