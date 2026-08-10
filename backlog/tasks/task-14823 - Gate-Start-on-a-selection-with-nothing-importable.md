---
id: TASK-14823
title: >-
  Gate Start on a selection with nothing importable
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-10 21:00'
labels:
  - library
  - ingest
  - ux
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
P1 of the 2026-08-10 re-critique. Staging a folder containing no importable files leaves Start ENABLED with an empty gate line, and pressing it manufactures a permanent failure receipt the preflight had already diagnosed.

Observed live: an empty directory produced `0 files` in the preflight summary, `0 will import` in the commit line, an EMPTY start-quiet-line, and an enabled `#library-ingest-start`. Pressing it produced `✗ failed · emptydir · No files to import were found in this folder.` plus a toast `Import finished — 1 failed`, permanently moving the queue tally and polluting Recent imports with a failure that was predictable before the click.

DESIGN.md's dense-form convention says an inert action carries its reason at the control. Here the action is live and the reason arrives as a failure receipt. The surface already has the right pattern one branch away: the not-found case gates Start with a named reason and offers `Choose a file…`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A staged selection with nothing importable gates Start with a stated reason instead of allowing a doomed run
- [ ] #2 The gate distinguishes an empty folder from a folder whose files are all unsupported, since the recovery differs
- [ ] #3 No failure receipt is created for a selection the preflight already knew could not import anything
<!-- AC:END -->
