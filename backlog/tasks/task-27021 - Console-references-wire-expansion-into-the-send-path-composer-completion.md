---
id: TASK-27021
title: 'Console @-references: wire expansion into the send path + composer completion'
status: In Progress
assignee: []
created_date: '2026-09-02 00:38'
updated_date: '2026-09-02 13:50'
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
- [x] #1 The user's draft is expanded via console_references before send (file/folder/diff/staged), at one preparation seam
- [x] #2 The transcript shows what was expanded/refused (the ReferenceRecords)
- [ ] #3 The composer offers completion for reference targets, reusing the existing suggestion surface
- [ ] #4 Live-verified with the app running: @file expands, @/outside-root and a binary file are refused visibly
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Send-path expansion + transcript records shipped; composer completion and the live app verification remain (this stays In Progress).

- AC#1 DONE: expansion runs at the ONE preparation seam (_submit_draft_inner): after draft validation, candidates are detected cheaply inline and expanded off-loop (asyncio.to_thread) via build_console_reference_resolver + run_git_reference; the store echo keeps the RAW draft; the outgoing provider payload swaps the just-echoed last user message for the expanded text BEFORE the dictionaries/world-info transforms (accepted hazard noted in-code: dictionary keywords inside included file content will also match). AGENT_WAKE drafts (machine-composed) never expand; any expansion failure sends the raw draft rather than blocking (test-pinned).
- AC#2 DONE: a compact '@-references:' system row adjacent to the user echo names each included/refused reference with its reason (26020 AC#6).
- AC#3 residue: composer completion for reference targets (suggestion surface).
- AC#4 residue: live verification with the app running (real @file expands; outside-root/binary visibly refused).

Tests: 3 seam tests in Tests/Chat/test_console_chat_controller.py (expanded-payload-vs-raw-echo + system row; email at-sign untouched with resolver never built; expansion-failure sends raw). Controller suite: 238 pass, only the 12 dev-inherited baseline failures. console_references suite 22 green.

Files: tldw_chatbook/Chat/console_chat_controller.py, Tests/Chat/test_console_chat_controller.py.
<!-- SECTION:NOTES:END -->
