---
id: TASK-2750
title: Renumber 34 duplicate task IDs resident on dev (repo-wide dup sweep)
status: To Do
assignee: []
created_date: '2026-08-06 23:50'
labels:
  - hygiene
  - backlog
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The post-merge duplicate scan after PR #1379 (2026-08-06) found **34 task IDs each carried by two distinct files on origin/dev**: 400-403, 506-519, 1333, 1338-1347, 1360-1361, 1373-1375, 2231. Spot-checks confirm they are real (e.g. two `task-400` files: "Fix MCP navigation crash…" vs "Move Console Context staged sources…"; two `task-1373`: Settings j/k navigation vs blocking-IO guard; two `task-2231`: Library ingest round 7 vs Roleplay redesign R2).

This is the recurring silent-pileup failure: the backlog-guard workflow's post-merge check runs on dev pushes whose CI is intentionally cancelled, and per-PR checks cannot see sibling branches, so duplicates accumulate until a dedicated sweep (precedent: PR #760 cleaned five in 2026-07).

Resolution rules that held previously: the **Done/older/load-bearing side keeps the number** (close-out commits, PR titles and code comments reference it); the To Do/newer side renumbers to fresh IDs claimed with headroom past every open branch's claims; cross-references (`dependencies:`, doc/code citations of the moved ids) are fixed in the same pass; frontmatter `id:` and filename must both move. Re-run the two-namespace scan (filename prefix + frontmatter id, `os.listdir` + regex — not `git ls-tree | sed`, which chokes on quoted paths) on the merged view immediately before AND after the cleanup PR merges.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The repo-wide two-namespace duplicate scan on origin/dev reports zero duplicate filename IDs and zero duplicate frontmatter ids
- [ ] #2 Every renumbered task keeps its content, status and history; the keeper side of each pair is chosen by the Done/older/load-bearing rule and the choice is recorded in the cleanup PR body
- [ ] #3 All cross-references to moved IDs (task dependencies, docs, code comments) resolve to the new numbers
- [ ] #4 The scan is re-run on the merged view after the cleanup PR lands (concurrent sessions mint IDs by the hour)
<!-- AC:END -->
