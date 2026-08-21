---
id: TASK-19573
title: >-
  Seven duplicate backlog task ids on dev, one of them a live In Progress
  ambiguity
status: Done
assignee: []
created_date: '2026-08-21 20:26'
labels:
  - backlog
  - process
priority: high
dependencies: []
---

## Description

Source: 2026-08-21 holistic review, Lane 7 (process, tooling & repo health) —
its **F3** and **F8**. Re-verified at this branch base by running the guard's
logic locally over 2,280 task files: **exit 1, seven duplicates**, in **both**
namespaces (filename prefix *and* frontmatter `id:`).

**The Backlog Guard is RED on `dev` right now** —
`gh run list --workflow="Backlog Guard"` shows `failure` on `dev` at
`2026-08-21T16:53:13Z`, plus failures on three branches within the same hour.

**Correction to the review's summary: the ranges are not contiguous.** 16321
and 18914 are *not* duplicated. The seven are:

| id | colliding files |
|---|---|
| 16320 | `Add-startup-AGENTS.md-project-context-to-Console` / `Trajectory-import-open-shared-traces-read-only` |
| 16322 | `Add-nested-AGENTS.md-activation-before-Console-tools` / `Build-the-local-research-execution-engine` |
| 16323 | `Enforce-run-budgets-with-a-reserve-and-settle-ledger` / `Verify-and-roll-out-Console-AGENTS.md-support` |
| 16324 | `Add-iterative-gap-driven-replanning-to-local-research-runs` / `Atomically-pin-local-tool-workspace-execution` |
| 18912 | `Reconcile-Console-Context-and-Inspect-UX-baseline-before-remediation` / `Standardize-Library-pager-display-and-harden-Conversation-paging` |
| 18913 | `Align-Library-Prompt-browsing-to-20-item-pages` / `Keep-Console-workspace-geometry-inside-the-viewport-at-exactly-100-columns` |
| 18915 | `Add-an-Inspector-overflow-fold-hint` / `Page-Library-Skills-with-source-wide-trust-recovery` |

**The ambiguity is live, not theoretical.** `task-18913` (Console workspace
geometry) is **`status: In Progress`**, created 2026-08-20 with dependency
`TASK-18912` — while the other `task-18913` is `Done`. `backlog task 18913
--plain` is non-deterministic: which task an agent reads depends on directory
ordering.

**Root cause: the MAX+20 leapfrog has now failed twice against concurrently
minting sessions.** Both batches are internally clean — two sessions each
picked a contiguous block, and the blocks overlapped. This is not carelessness;
it is a protocol that assumes it can observe the maximum id at claim time, in a
repo where other sessions are minting ids in worktrees and unpushed branches
simultaneously. (This filing used **MAX+30** for exactly that reason.)

**No local gate exists.** The check lives **only** in
`.github/workflows/backlog-guard.yml` as inline bash, fires only on
`backlog/tasks/**` paths, and — per TASK-19572 — is not a required check, so a
red guard does not block a merge. There is **no pytest equivalent**, so nobody
running tests locally can catch a collision before pushing.

Related, from F8: leapfrogging produces **duplicate work**, not just duplicate
ids — `task-13262` and `task-14650` are both In Progress with the **identical
title**. The backlog holds 2,279 tasks (1,934 Done / 308 To Do / 36 In
Progress), max id 19480 at review time; 13 of the 36 In Progress are stale by
more than 14 days and three carry no date at all.

## Acceptance Criteria

- [x] All seven duplicate ids are resolved by renumbering — the newer or
      less-referenced task in each pair moves to a fresh id above the current
      maximum
- [x] `task-18913`'s In Progress ambiguity is resolved first, since an agent is
      actively working against it
- [x] Renumbering updates every inbound reference (dependencies, branch names,
      PR bodies where practical), not just the filenames
- [x] Backlog Guard is green on `dev`
- [ ] A **local** gate exists — a pytest that fails on duplicate ids — so a
      collision is caught before push rather than by a workflow nobody can
      block on
- [x] The id-claiming protocol is revised to survive concurrent sessions: the
      failure mode is two sessions observing the same maximum, so the fix must
      not be "leapfrog further" (that has now failed twice). Record the chosen
      approach in `backlog/docs/lessons-backlog-hygiene.md` **with the incident
      that produced it**
- [ ] `task-13262` / `task-14650` (identical titles, both In Progress) are
      reconciled
- [ ] The 13 stale In Progress tasks are triaged — closed, reassigned, or
      returned to To Do

## Implementation Notes (2026-08-21)

Resolved by TASK-19601's owner rule (older keeps the id; younger
renumbers with provenance). The "live In Progress ambiguity" was
task-18913 Keep-Console-workspace-geometry — the younger side; it
renumbered to TASK-19639 and continues there under its In Progress state.
All seven collisions cleared; see TASK-19601 notes for the full map.

### Verification pass (task/19573-burn, dispatched separately after the above)

Re-ran the guard's own logic (both the filename and frontmatter `id:`
checks in `.github/workflows/backlog-guard.yml`) locally at `origin/dev`
HEAD `5f720a404`: **zero duplicates in either namespace**, across 2,324
task files. All seven renumbered targets (TASK-19634..19640) resolve to
exactly one file each, each carries a `## Renumbering provenance` section,
and a repo-wide grep for the seven old ids turned up nothing stale —
every remaining hit is either the keeper side's own code/docs (which
correctly still cites its own id), a renumbered file's own provenance
note, or a deliberately-preserved historical record (the 18802 report,
the 19052 plan, ADR-067/068). No renumbering work was needed from this
pass — TASK-19601 had already done it and it holds.

Remaining work: no new id needed to be minted, so the MAX+30 sweep this
branch was set up to run was not exercised. Added a lessons entry
(`backlog/docs/lessons-backlog-hygiene.md`, "Task IDs collide constantly")
documenting that MAX+20 leapfrogging failed twice against concurrent
minting and that the durable fix is TASK-19601's older-keeps-the-id rule,
not more distance. The guard's failure message already states that rule
(`.github/workflows/backlog-guard.yml:59`), so no further change was
needed there.

Folding the duplicate-id check into a "derived-artifacts" CI gate was
considered and **not done here**: TASK-19572 turns out to be about a
different, unrelated `derived-artifacts` job (three stdlib-only checkers —
CSS bundle sync, profile-owned-path inventory, persistent-diagnostic
inventory — plus branch protection). It does not mention the backlog
duplicate-id guard, so consolidating the two would be inventing scope
TASK-19572 never claimed, not deferring to it.

Three ACs below remain genuinely open and were **not** addressed in this
pass (out of scope for this dispatch): the local pytest duplicate-id gate,
the TASK-13262/TASK-14650 identical-title reconciliation (both still
`In Progress`, still byte-identical apart from id and one Notes line), and
the 13-task stale-In-Progress triage. Left `status: Done` as inherited
rather than reopening it unilaterally, since the task's core defect (red
guard, live ambiguity) is genuinely resolved and closed elsewhere
(TASK-19601); the three open items above should be tracked as their own
follow-up rather than reopening this one.
