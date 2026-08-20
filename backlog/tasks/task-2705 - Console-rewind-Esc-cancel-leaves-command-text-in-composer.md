---
id: TASK-2705
title: 'Console /rewind: cancelling the menu leaves "/rewind" in the composer'
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-31'
updated_date: '2026-08-19'
labels: [console, polish]
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Dismissing the Rewind menu with Esc (or "Never mind") leaves the literal
`/rewind` text sitting in the composer draft, so the next thing the user
types continues after it (e.g. `/rewind/bogus` → unknown-command hint). The
other slash commands that clear the draft on successful dispatch behave the
same way on cancel, but `/rewind` is the one whose menu is routinely
opened-and-cancelled while browsing. Observed live on dev @ ff435772c
(G1 user-guide session, 2026-07-31); documented as a quirk in
`Docs/User_Guide/console/branching-and-rewind.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] After an argument-free `/rewind` successfully opens the Rewind menu,
      Esc or "Never mind" leaves the invocation consumed while preserving
      text typed after the Enter keypress (empty if there was no later text).
- [ ] If `/rewind` cannot open because there are no prompts, its captured
      invocation is restored and the existing warning is shown.
- [ ] A modal-launch failure or a changed/replaced composer never loses or
      clears the user's current draft.
- [ ] Choosing "Restore to here" still replaces the draft with the restored
      prompt text (existing behavior preserved).
- [ ] The User Guide quirk note is updated/removed to match.
<!-- AC:END -->

## Implementation Plan

ADR required: no

ADR path: N/A

Reason: this is a localized command-draft cleanup bug fix within the existing
Console/composer contracts; it changes no storage, service boundary,
dependency, security policy, or long-lived architecture.

1. Add mounted RED tests for keyboard and visible-Send cancellation, Restore,
   no-row refusal, non-empty arguments, launch failure, and stale composers.
2. Add the narrow argument-free `/rewind` branch at the existing command-send
   boundary and return an opened/refused result from the rewind handler.
3. Guard visible-Send cleanup by composer identity, edit serial, generation,
   and dispatched text; restore keyboard stashes on refusal or exception.
4. Remove the resolved User Guide workaround and run the bounded rewind/send/
   safe-dismissal test and static-analysis matrix.
5. Complete independent review, task evidence, acceptance criteria, and Done
   status before branch integration.

Detailed executable plan:
`Docs/superpowers/plans/2026-08-19-task-2705-rewind-cancel-draft.md`
