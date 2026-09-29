---
id: TASK-32772
title: Design agent-assisted archived conversation recovery
status: Done
assignee:
  - '@codex'
created_date: '2026-09-18 05:55'
updated_date: '2026-09-27 15:30'
labels: []
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define how a user can ask a Console agent to find a missing archived conversation without knowing its name or workspace, identify it from bounded results and explicitly restore it.
<!-- SECTION:DESCRIPTION:END -->

## Renumbering provenance

Backlog CLI assigned TASK-32746. A fresh fetch plus all-ref object-path and existing-worktree sweep found a task ceiling of 32771 before filing. Renumbered this newly created design task to TASK-32772 before creating inbound references; no existing task was changed. The ADR ceiling was 165.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The design defines cross-workspace archive discovery and truthful archive-time behavior.
- [x] #2 The design defines exact-target confirmation and state-dependent chat or workspace restoration through both user entry points.
- [x] #3 The architecture decision defines Library authority, RAG-only recovery, local storage and Console runtime boundaries.
- [x] #4 The spec defines bounded results, stale and partial outcomes, cancellation ownership and targeted verification.
- [x] #5 The written spec is reviewed and approved before implementation planning.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Design-only task; entered In Progress before writing artifacts.

ADR required: yes

ADR path: backlog/decisions/166-agent-assisted-archive-recovery.md

Reason: Console recovery contracts, Library authority/retrieval-mode amendment, archive chronology and interactive mutation ownership.

1. Inspect existing archive/search/restore, Library authority and Console confirmation flows.
2. Confirm scope and compare approaches with the user; review edge cases before writing.
3. Write the agreed spec and proposed canonical ADR; link both from this task and the ADR index.
4. Self-review for placeholders, contradictory authority/lifecycle rules, boundedness, errors and testable outcomes; validate document links and whitespace.
5. Obtain user review of the written spec before invoking writing-plans. Application implementation and its atomic tasks belong to that later plan.

Spec: [Agent archive recovery design](../../Docs/superpowers/specs/2026-09-17-agent-archive-recovery-design.md)

ADR: [ADR-166](../decisions/166-agent-assisted-archive-recovery.md)
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Designed agent-assisted recovery of local archived chats across workspaces and documented the approved contracts in the linked specification and ADR-166. The user approved the written spec and both review rounds before planning. ADR-166 is Accepted; ADR-030/079 now identify only the narrow recovery amendment in metadata, preserving their original decision bodies.

The design separates real archive dates from workspace concurrency revisions, matches either applicable archive event, bounds search and result retention, and binds confirmation to an immutable target. It also defines interactive-origin admission, RAG-only recovery under Library authority, role-restricted search, cancellation ownership, deduplication, cross-store partial outcomes and separate Preview/Open actions.

Implementation plan: [Agent archive recovery](../../Docs/superpowers/plans/2026-09-27-agent-archive-recovery.md). The plan defines seven independently testable delivery tasks, concrete interfaces, targeted red/green checks and disposable-data live qualification. No application code or migration was implemented by this design task.

Verification: reviewed spec-to-plan coverage and interface consistency; 19 plan/spec/ADR/task links resolve, all 12 Python examples parse, placeholder checks pass, and all seven new delivery tasks pass the scoped Backlog ID/path guard with unique IDs and earlier-only dependencies. Document whitespace is checked separately before commit. Application tests and Python lint/format are not applicable to these Markdown-only changes; implementation tests remain prospective.

Task closeout was performed through Backlog CLI in a temporary Git workspace to avoid repeated branch-history scanning in the shared checkout. The returned ID, filename, approval criterion and status were verified before copying the updated record back. Unrelated records and application changes were preserved.
<!-- SECTION:NOTES:END -->
