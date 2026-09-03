# Collections Follow-up Backlog Creation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a reviewed, dependency-correct set of six `tldw_server` prerequisite tasks and seven `tldw_chatbook` follow-up records for the Collections work stream.

**Architecture:** The Server records own additive, versioned capability contracts; Chatbook records remain fail-closed and reference the exact Server task identities without cross-repository Backlog dependencies. A Chatbook tracking parent groups the three management workflows, while the legacy lifecycle remains a decision-only ADR task. Task IDs are allocated only after rebasing both filing branches on their latest `origin/dev` state.

**Tech Stack:** Backlog.md CLI, Markdown, Git

---

## Source design

- Approved design: `Docs/superpowers/specs/2026-09-01-collections-followup-backlog-design.md`
- Chatbook foundation: `TASK-18919` and `backlog/decisions/107-collections-capture-authority-and-legacy-boundary.md`
- Filing rule: create every record with the Backlog.md CLI; do not edit generated task files manually.

## Task 1: Freeze the reviewed design and refresh repository bases

**Files:**

- Modify: `Docs/superpowers/specs/2026-09-01-collections-followup-backlog-design.md`
- Create: `Docs/superpowers/plans/2026-09-02-collections-followup-backlog-creation.md`

1. Verify the design contains thirteen records, explicit filing metadata, measurable acceptance criteria, stable cross-repository reference rules, and no optional real-user data probe.
2. Run `git diff --check` and inspect the final documentation diff.
3. Commit the reviewed design and this execution plan on `codex/collections-followup-backlog`.
4. Fetch Chatbook's `origin/dev` and rebase the Chatbook branch onto it.
5. Fetch `origin/dev` separately in `tldw_server2`, then create an isolated Server worktree on branch `codex/collections-followup-backlog` from that refreshed ref; do not touch the dirty primary checkout.
6. Before filing in each repository, build an occupied-ID census from every registered worktree plus every `refs/remotes/*` tree's `backlog/tasks`, `backlog/completed`, and `backlog/archive/tasks` buckets. Also search those trees for normalized title duplicates. Save the census outside the repositories, confirm the candidate branch itself passes its local duplicate-ID guard, and treat every newly allocated ID as provisional until the post-creation and pre-merge census repeats.

## Task 2: File the six Server capability prerequisites

**Files:**

- Create through Backlog.md CLI: `tldw_server2/backlog/tasks/task-<allocated> - Add atomic revision-guarded hard delete for Reading items.md`
- Create through Backlog.md CLI: `tldw_server2/backlog/tasks/task-<allocated> - Add coherent scoped tag and domain aggregates for Reading items.md`
- Create through Backlog.md CLI: `tldw_server2/backlog/tasks/task-<allocated> - Establish safe Collections output-template ownership and deletion.md`
- Create through Backlog.md CLI: `tldw_server2/backlog/tasks/task-<allocated> - Attest bounded Reading digest schedule and output management.md`
- Create through Backlog.md CLI: `tldw_server2/backlog/tasks/task-<allocated> - Add complete restart-safe Reading export jobs.md`
- Create through Backlog.md CLI: `tldw_server2/backlog/tasks/task-<allocated> - Add Server-native Reading export re-import.md`

1. Check the available tools for the official Backlog MCP workflow required by the Server repository. If no dedicated Backlog MCP tool is callable, record that result and use the documented Backlog.md CLI fallback.
2. Create S1 through S5b in the design's order with status `To Do`, the exact priority and labels from the filing table, and the corresponding description and acceptance criteria from the approved design.
3. Work around Backlog CLI 1.44's collapsed repeated-`--ac` behavior: create each record with exactly its first criterion, then append each remaining criterion with a separate `backlog task edit <id> --ac <criterion>` call. Verify the resulting criterion counts are S1=6, S2=5, S3=5, S4=6, S5a=6, and S5b=6.
4. Add stable textual references to `tldw_chatbook:TASK-18919` and the Chatbook design path in each description.
5. Capture each CLI-assigned task ID immediately; make S5b depend on the actual S5a ID and do not add dependencies among the other Server tasks.
6. Inspect every generated record with `backlog task <id> --plain`; verify title, description, acceptance criteria, labels, priority, status, and dependency.
7. Repeat the all-remote-ref/all-worktree occupied-ID and normalized-title census. If any provisional ID or title collides, renumber/recreate it before commit using the repository's documented owner rule.
8. Run the repository's task-ID/metadata checks when present, run `git diff --check`, inspect the complete Server diff, and commit only the six generated records and any Backlog CLI index metadata.

## Task 3: File the seven Chatbook follow-ups

**Files:**

- Create through Backlog.md CLI: `backlog/tasks/task-<allocated> - Enable atomic Server capture hard delete.md`
- Create through Backlog.md CLI: `backlog/tasks/task-<allocated> - Present complete Server capture tag and domain facets.md`
- Create through Backlog.md CLI: `backlog/tasks/task-<allocated> - Complete Server Collections management workflows.md`
- Create through Backlog.md CLI: `backlog/tasks/task-<allocated> - Manage Server Collections output templates.md`
- Create through Backlog.md CLI: `backlog/tasks/task-<allocated> - Manage Server Reading digest schedules and outputs.md`
- Create through Backlog.md CLI: `backlog/tasks/task-<allocated> - Manage Server Reading import and export workflows.md`
- Create through Backlog.md CLI: `backlog/tasks/task-<allocated> - Decide legacy Collections migration or retirement.md`

1. Create C1, C2, and C3 with status `To Do`, exact design metadata, and `TASK-18919` as their same-repository prerequisite.
2. Capture C3's allocated ID, then create C3a, C3b, and C3c as children of C3. Also give each child the completed `TASK-18919` prerequisite; do not make children depend on the tracking parent.
3. Put the exact allocated `tldw_server:TASK-<id>` prerequisite identities in C1, C2, C3a, C3b, and C3c descriptions. Do not use same-repository dependency fields for those external tasks.
4. Create C4 as a low-priority, decision-only task depending on `TASK-18919`; require a new ADR and prohibit production lifecycle mutation or real-user data inspection.
5. Work around Backlog CLI 1.44's collapsed repeated-`--ac` behavior exactly as for the Server records. Verify criterion counts are C1=5, C2=5, C3=4, C3a=5, C3b=6, C3c=6, and C4=6.
6. Preserve the design's descriptions and acceptance criteria, reference ADR-107 and the approved design, and leave implementation plans absent while all records remain `To Do`.
7. Inspect all seven records with `backlog task <id> --plain`; verify C3 child links, dependencies, external references, priorities, labels, and statuses.
8. Repeat the all-remote-ref/all-worktree occupied-ID and normalized-title census. Resolve any provisional collision before commit.
9. Run `python scripts/check_backlog_task_ids.py`, run `git diff --check`, inspect the complete Chatbook diff, and commit only the generated backlog records and Backlog CLI index metadata.

## Task 4: Cross-repository closeout verification

1. Re-read all thirteen records from the CLI and compare them against the approved design's filing table and acceptance criteria.
2. Verify every Chatbook capability task names the correct allocated Server task, S5b depends only on S5a, the three C3 children have the correct parent, and no record references an uncreated future ID.
3. Repeat the occupied-ID and normalized-title census against every current remote ref and registered worktree in both repositories. Record that the IDs remain provisional until this same check is run immediately before merge.
4. Confirm both worktrees are clean after their commits and report the two branch names, thirteen task IDs/titles, dependency graph, and any intentionally deferred implementation planning.
5. Do not push, open pull requests, or merge unless the user requests those repository mutations separately.

## Verification

- `git diff --check` passes in both filing worktrees before commit.
- Repository-provided Backlog duplicate-ID/metadata checks pass where available, and the broader remote-ref/worktree census finds no conflicting provisional ID or normalized title.
- `backlog task <id> --plain` shows all thirteen tasks as `To Do`, unassigned, and without implementation plans.
- The Server branch contains exactly six new capability task records; the Chatbook branch contains exactly seven new follow-up records plus the reviewed design and this plan.

## ADR check

**ADR required:** no for this filing change.

**ADR path:** N/A

**Reason:** This plan records backlog metadata only and makes no production architecture decision. The future S1, S2, S3, S5a/S5b, and C4 records explicitly require the applicable ADR work before their implementation begins.
