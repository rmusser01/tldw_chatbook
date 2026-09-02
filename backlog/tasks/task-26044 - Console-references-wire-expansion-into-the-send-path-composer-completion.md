---
id: TASK-26044
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
