---
id: TASK-27021
title: 'Console @-references: wire expansion into the send path + composer completion'
status: To Do
assignee: []
created_date: '2026-09-02 00:38'
labels:
  - ux
  - console
dependencies:
  - TASK-26020
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-26020 shipped the tested @-reference engine (Chat/console_references.py): parser (emails/decorators left untouched), resolver reusing the file tools' allowed-roots + sensitive-path authority, binary/size refusal, folder listing, and git diff/staged. What remains is app-context integration not verifiable headless: (1) apply expand_references(build_console_reference_resolver(), run_git_reference) to the user's draft at the single console_turn_preparation seam BEFORE send, so references actually expand (AC#1/#2); (2) surface the ReferenceRecords in the transcript so the user sees what was expanded (AC#6); (3) offer completion for reference targets in the composer, reusing the suggestion surface like the $-skill mentions (AC#5). Wire with the app running (textual-serve) and verify a real @file/@folder/@diff expands and an outside-roots/binary ref is refused in the transcript.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The user's draft is expanded via console_references before send (file/folder/diff/staged), at one preparation seam
- [ ] #2 The transcript shows what was expanded/refused (the ReferenceRecords)
- [ ] #3 The composer offers completion for reference targets, reusing the existing suggestion surface
- [ ] #4 Live-verified with the app running: @file expands, @/outside-root and a binary file are refused visibly
<!-- AC:END -->

## Renumbering provenance

This task previously held id TASK-26044, colliding with the
"Console-Workspace-Files-isolated-Git-decoration" task that arrived on origin/dev first (dev max was 27018 at
renumber time, 2026-09-02; range swept across all remote branches and local
worktrees). Per the owner rule decided 2026-08-21 in TASK-19601 (**older id
keeps it; the younger task renumbers with a provenance note, regardless of
status**), it renumbered to TASK-27021. Citations to TASK-26044 in this
branch's commit messages or implementation notes written before 2026-09-02
refer to THIS task; the dev-resident TASK-26044 holder is the Workspace
Files task.
