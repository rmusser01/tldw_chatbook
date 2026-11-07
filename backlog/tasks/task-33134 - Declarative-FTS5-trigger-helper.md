---
id: TASK-33134
title: Declarative FTS5 trigger helper
status: To Do
assignee: []
created_date: '2026-09-27 19:46'
labels: [refactor, db]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Six hand-rolled FTS5 trigger farms (ChaChaNotes, Evals, Subscriptions, Client_Media, Prompts x2, Library_Collections) each re-implement the same insert/update/delete trigger triplet, plus a bespoke backfill module that exists only for one of the six. A helper that builds the trigger triplet, rebuild, and backfill from a table description collapses the farms; only ChaChaNotes' guarded, undelete-aware messages_fts behavior needs an escape hatch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One helper builds trigger triplets from a table, FTS-table, and column description.
- [ ] #2 The guard and undelete-aware messages_fts behavior keeps its escape hatch and its behavioral witnesses.
- [ ] #3 Backfill is covered by the helper instead of the bespoke single-store module.
- [ ] #4 FTS behavior on every touched store is unchanged, verified by targeted tests.
<!-- AC:END -->

## Renumbering provenance

Filed 2026-09-27 as task-33089. origin/dev minted its own task-33089 before this branch merged, so per the landed-keeps-id rule this task moved to task-33134; every inbound reference moved with it.
