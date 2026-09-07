---
id: TASK-25827
title: Workspace conversation load failure shows a truncated message with no cause
status: Done
assignee: []
created_date: '2026-08-31 05:09'
updated_date: '2026-08-31 06:36'
labels:
  - console
  - ux-review
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When the workspace conversation list fails to load, the rail shows a clipped string and a bare Retry control. Two distinct root causes are collapsed into one generic sentence, the real exception is swallowed into a debug log, and the visible text is cut off by the rail width so the user cannot read even the generic message.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The rail error is legible at the rail's actual width
- [ ] #2 Distinct failure causes produce distinct user-facing messages
- [ ] #3 The underlying exception is recorded at a level the user can be pointed to
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Two unrelated failures shared one sentence: membership-token-unknown and a swallowed list_conversations exception both rendered 'Workspace conversations are unavailable.', which the rail then clipped to 'Workspace conversations a...' -- naming neither cause. Split into WORKSPACE_CONVERSATIONS_ACCESS_UNKNOWN ('Workspace access unknown.') and WORKSPACE_CONVERSATIONS_LOAD_FAILED ("Couldn't load conversations."), both short enough to read at rail width, and raised the swallowed exception log from debug to warning so 'check the app log' actually leads somewhere. Four pinned tests updated to their real cause. Verified against baseline: the 13 failures in test_console_workspace_controller.py are pre-existing on clean dev, unchanged by this.
<!-- SECTION:NOTES:END -->

## Correction to the implementation note

I wrote that raising the swallowed exception from `debug` to `warning` makes
"check the app log" lead somewhere. That is half right and the distinction
matters, because TASK-25816 established the other half: the PERSISTENT log file
admits only records marked by `persist_event` (`PersistentDiagnosticFilter`),
so a plain loguru `warning` does NOT reach it.

What the level change actually buys: the cause now surfaces in the terminal and
in the in-app Logs screen (F8), where a `debug` record was filtered out. Those
are real surfaces and the change is worth keeping, but anyone chasing this in
the persistent file will still not find it. Making it durable would need a
metadata-only `persist_event` alongside, the way TASK-25816 did for the database
failure -- deliberately not done here, since the rail error already names its
cause on screen and the two distinct messages were the actual defect.

Surfaced by the Derived Artifacts security gate, which pins every production
diagnostic statement: the level change registered as a statement rewrite and
forced this review. Confirmed safe -- identical message and argument, and
`workspace_id` was already interpolated before the edit; no user content,
secret, path or URL was added.
