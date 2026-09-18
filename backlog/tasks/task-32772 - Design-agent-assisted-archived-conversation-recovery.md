---
id: TASK-32772
title: Design agent-assisted archived conversation recovery
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-18 05:55'
labels: []
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define how a user can ask a Console agent to find a missing archived conversation without knowing its name or workspace, identify it from bounded results and explicitly restore it.
<!-- SECTION:DESCRIPTION:END -->

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

## Renumbering provenance

Backlog CLI assigned TASK-32746. A fresh fetch plus all-ref object-path and existing-worktree sweep found a task ceiling of 32771 before filing. Renumbered this newly created design task to TASK-32772 before creating inbound references; no existing task was changed. The ADR ceiling was 165.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The design defines cross-workspace archive discovery and truthful archive-time behavior.
- [x] #2 The design defines exact-target confirmation and state-dependent chat or workspace restoration through both user entry points.
- [x] #3 The architecture decision defines Library authority, RAG-only recovery, local storage and Console runtime boundaries.
- [x] #4 The spec defines bounded results, stale and partial outcomes, cancellation ownership and targeted verification.
- [ ] #5 The written spec is reviewed and approved before implementation planning.
<!-- AC:END -->

## Implementation Notes

Design artifacts only: wrote the linked spec and proposed ADR-166 and added its index entry. The user approved both recovery entry points, explicit chat/workspace choices and the review correction allowing recovery in RAG-only mode under Assistant Library access. The written spec still requires user review; this task remains In Progress and implementation planning has not started.

Self-review made unknown-time inclusion an explicit search argument, bounded result/confirmation/receipt retention, prevented result renumbering from changing targets, separated runtime-owned restore from exact resume, and specified transaction-level workspace lifecycle checks and honest cross-store partial completion. Search excludes private control content and labels inactive-branch matches.

Verification: document-link, placeholder and whitespace checks pass; the new task passes the scoped Backlog ID/path guard and its ID is unique in the checkout. The repository-wide ID guard reports existing duplicate task IDs unrelated to TASK-32772; those records were not changed. No application tests were run because this change contains only design documentation. ADR-166 amends the relevant parts of ADR-030/079 and preserves ADR-147; existing accepted ADR bodies were not rewritten.
