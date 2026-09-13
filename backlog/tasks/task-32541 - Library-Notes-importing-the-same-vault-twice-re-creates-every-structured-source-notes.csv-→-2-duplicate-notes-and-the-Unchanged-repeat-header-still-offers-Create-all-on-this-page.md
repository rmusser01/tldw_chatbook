---
id: TASK-32541
title: >-
  Library Notes: importing the same vault twice re-creates every structured
  source (notes.csv → 2 duplicate notes), and the Unchanged repeat header still
  offers "Create all on this page"
status: To Do
assignee: []
created_date: '2026-09-13 06:46'
labels:
  - library
  - notes
  - critique-3
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 (dev 5fd502dbac), both assessors, personas Riley and Jordan, Obsidian import workflow. A P2 #7 / B D5.

**What happened.** Second Import once of the unchanged vault → folder-collision block with "Create a unique sibling" pre-selected ("vault (2)") → every `.md` is "Unchanged repeat · Content: no change · Folder placement: no change" with Skip pre-selected — but `notes.csv` is under New again: "create 2 new notes: CSV note one, CSV note two · keywords csv · Create in vault (2)" (A 61; B 36). `meta.yaml` and `scratch.txt` follow the same path. The Unchanged repeat group's header also carries "Create all on this page", which would re-create 56 notes in one click (A 61). Captures: A 61; B 36.

**Cause.** INFERRED: structured sources (CSV rows, YAML) are not fingerprinted per produced record, so repeat detection — which works for Markdown — never classifies them. No test in Tests/Notes or Tests/UI names CSV or structured repeats (B grep). Task-32130 / 32176 fixed the first-import CSV count only. Docs contradicted: notes.md's Unchanged repeat rows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A second Import once of an unchanged vault lists notes.csv, meta.yaml and scratch.txt under Unchanged repeat with Skip pre-selected
- [ ] #2 The Unchanged repeat group header offers no "Create all" action
- [ ] #3 A test imports a CSV source twice through the review and asserts the second review classifies it as an unchanged repeat
<!-- AC:END -->
