---
id: TASK-19601
title: >-
  Backlog Guard is red on dev: seven colliding task IDs, four unresolvable
  under the current rule
status: Done
assignee: []
created_date: '2026-08-21 18:20'
labels:
  - ci
  - backlog-hygiene
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The "No duplicate backlog task IDs" guard is FAILING on `dev` (runs
32509983450, 32510107746 and every run since). Seven ids are each held by
two task files, so the guard fails for every PR in the repo -- and a guard
that is permanently red gets ignored, which is how the previous collisions
went unnoticed.

This is separate from TASK-18060, which was the eighth collision and was
fixed in PR #1872; these seven are a different, larger batch spanning two
active programmes (Console AGENTS.md context vs local-research engine, and
Console Context/Inspect UX vs Library pager work).

The blocker is that **four of the seven pairs are Done vs Done**, and this
repo's standing rule is that a Done task NEVER moves (its id is already
cited in merged commits, ADRs, and code comments). Under the current rule
those four have no legal resolution at the file level:

| id | side A | side B | resolvable? |
|----|--------|--------|-------------|
| 16320 | Add-startup-AGENTS.md-project-context (Done) | Trajectory-import-open-shared-traces (Done) | NO -- both Done |
| 16322 | Add-nested-AGENTS.md-activation (Done) | Build-the-local-research-execution-engine (Done) | NO -- both Done |
| 16323 | Enforce-run-budgets-reserve-settle-ledger (Done) | Verify-and-roll-out-Console-AGENTS.md (Done) | NO -- both Done |
| 18912 | Reconcile-Console-Context-and-Inspect-UX (Done) | Standardize-Library-pager-display (Done) | NO -- both Done |
| 16324 | Add-iterative-gap-driven-replanning (Done) | Atomically-pin-local-tool-workspace (To Do) | yes -- To Do side moves |
| 18913 | Align-Library-Prompt-browsing-to-20-item-pages (Done) | Keep-Console-workspace-geometry-at-100-columns (In Progress) | yes -- In Progress side moves |
| 18915 | Add-an-Inspector-overflow-fold-hint (To Do) | Page-Library-Skills-with-source-wide-trust-recovery (To Do) | yes -- either moves |

Fixing only the three resolvable pairs does not turn the guard green, so a
partial fix buys nothing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 The Backlog Guard passes on `dev`.
- [x] #2 Whatever is decided for the four Done-vs-Done pairs is written down as the rule to apply next time (this is the 8th-15th collision; the pattern is not stopping on its own).
- [x] #3 No id cited by already-merged code, ADRs, or commit messages silently changes meaning without a provenance note (the pattern used for TASK-19300).
<!-- AC:END -->

## Notes

<!-- SECTION:NOTES:BEGIN -->
Owner decision required -- three options, deliberately NOT chosen here:

1. **Relax "Done never moves" for the tie case** and renumber one side of
   each Done-vs-Done pair (newer `created_date` moves, say), each with a
   TASK-19300-style provenance note. Restores a green guard; costs
   traceability churn on merged work.
2. **Teach the guard to distinguish historical from NEW collisions** --
   e.g. fail only when a PR ADDS a duplicate, with an explicit
   grandfathered-ids allowlist for the four unresolvable pairs. Keeps the
   guard's real purpose (catch collisions before merge) and stops it being
   permanently red. This is arguably what the guard should have done all
   along.
3. **Renumber both sides** of each unresolvable pair to fresh ids so
   neither inherits the ambiguous one. Cleanest end state, largest churn.

Self-demonstrating evidence for option 2: Qodo's review of the very PR
that FILED this task flagged that the PR trips the guard, because adding
any file under `backlog/tasks/**` runs a guard that hard-fails on
pre-existing duplicates. A guard that fails the ticket reporting the guard
is failing is a guard that only ever measures history, never the change in
front of it.

The deeper cause is unchanged and worth fixing regardless: ids are minted
by reading the highest id on a branch, so two agents branching from the
same base mint the same id. A mint-time reservation (or minting from a
monotonically increasing counter file that conflicts loudly on merge) would
end the recurrence.
<!-- SECTION:NOTES:END -->

## Implementation Notes (2026-08-21)

**Owner rule decided** (recorded here per AC#2): *the older arrival keeps
the id; the younger task renumbers — regardless of Done status — with a
Renumbering provenance section naming the old id.* This supersedes the
"Done tasks never move" reading that made the four Done-vs-Done pairs
unresolvable; the provenance note (TASK-19300 pattern, AC#3) is what keeps
old citations honest instead of immobility.

**Executed:** the seven younger arrivals renumbered to TASK-19634..19640
(each with a Renumbering provenance section): 16320→19634 (startup
AGENTS.md), 16322→19635 (nested AGENTS.md), 16323→19636 (AGENTS.md
rollout), 16324→19637 (pin local tool workspace), 18912→19638 (Context/
Inspect baseline), 18913→19639 (100-column geometry), 18915→19640
(Inspector overflow hint). Younger-side references updated: the
agents-md-support and context-inspect-phase0 plans, the 18913 containment
plan+spec (renamed to task-19639), and three code comments
(chat_screen.py ×2, console_rail_state.py — now "TASK-19639 (formerly
TASK-18913)"). Keeper-side citations (trajectory, research engine,
library pagination programmes) untouched; historical collision reports
(18802 report, 19052 plan) deliberately left as records. Guard workflow
guidance message updated to state the new rule. Guard passes locally on
this branch: 0 filename dupes, 0 frontmatter dupes.
