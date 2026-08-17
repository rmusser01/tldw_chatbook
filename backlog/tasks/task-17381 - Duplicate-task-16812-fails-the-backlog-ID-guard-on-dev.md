---
id: TASK-17381
title: Duplicate task-16812 fails the backlog ID guard on dev
status: To Do
assignee: []
created_date: '2026-08-17 08:05'
labels:
  - backlog-hygiene
  - ci
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two different task files claim TASK-16812 on dev — the Console local-provider thinking controls task and the research category-lane baselines task — so the duplicate-ID CI guard fails on dev itself and every pull request against dev inherits that red. This is the seventh ID collision in this repo, and it is the second one created by a renumber landing on an ID that was already taken: the Console file was added by a commit titled "chore(backlog): resolve duplicate task IDs".

Resolving it needs a decision rather than a mechanical rename, because the usual rule (never move a Done task) cannot break the tie: both tasks are Done, and both IDs are referenced from live artifacts. The Console ID is linked by filename from ADR-066 plus a plan and a QA script; the research ID is cited from source comments, a test section header, the eval baseline doc, and another task's justification. Whichever file moves, the references that point at it need to move with it, and the historical plan/QA artifacts that record the work under its old number need a decision about whether they are rewritten or left as history.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The duplicate-ID guard passes on dev
- [ ] #2 Exactly one task file claims each of the two affected IDs, in both the filename and the frontmatter
- [ ] #3 Every live reference to the renumbered task resolves to it, including the ADR link that names the file
- [ ] #4 The chosen treatment of historical plan and QA artifacts naming the old ID is recorded with its reason
- [ ] #5 The lessons entry on ID collisions records this incident, since a renumber caused it
<!-- AC:END -->
