---
id: TASK-32625
title: >-
  Library Notes: the sync review spends three rows per file and reports skips at
  a different granularity from Import once
status: To Do
assignee: []
created_date: '2026-09-15 06:44'
labels:
  - library
  - notes
  - critique-4
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 (dev 77eb2601a6), assessor A heuristic 8 and section 11, persona Alex, Obsidian workflow.

What happened. Wave 4 (task-32535, PR #2679) made the sync review honest -- it names files, folders and effects now, instead of 'Safe item N' -- and that is a real improvement both assessors confirm. What it did not get is Import once's density: the sync review spends about three screen rows per file row (A cap 49) where Import once fits 65 sources on one page with groups, counts, a 45-file run collapsed to a single openable row, and a reason on every skipped row (A cap 21).

The two also now disagree on granularity. Import once reports skips at FOLDER level ('vault/.trash', 'vault/Templates'); the sync review reports the same vault at FILE level ('.trash/Old idea.md'). Same content, two mental models, one screen apart -- a side effect of the per-file item_skips the wave added.

Cause PROVEN by capture. The right shape is A's improvement idea 7: one dense review component shared by both paths, which would fix the granularity split for free.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The two review surfaces share a row renderer, or state why they differ
- [ ] #2 The sync review fits a 54-file vault without three rows per file
- [ ] #3 Skips are reported at one granularity across both paths
<!-- AC:END -->
