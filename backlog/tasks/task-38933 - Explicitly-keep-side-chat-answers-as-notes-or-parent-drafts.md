---
id: TASK-38933
title: Explicitly keep side-chat answers as notes or parent drafts
status: To Do
created_date: 2026-11-05 16:08
labels:
- console
- side-chat
- notes
priority: medium
references:
- https://uxplanet.org/anthropic-just-shipped-claude-2-0-bbb46a9c1383
- TASK-17167
documentation:
- Docs/Architecture/notes-sync.md
- Docs/Architecture/console.md
- backlog/decisions/068-console-text-selection-and-annotations.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users retain a useful side-chat answer through deliberate actions while keeping ordinary side chats disposable. Make the distinction between saving an answer, continuing the parent conversation, and changing Personal Context clear.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A completed or visibly partial side-chat answer offers explicit Save as note and Add to parent draft actions, with partial answers labeled.
- [ ] #2 Saving produces a real note through the existing note authority and records only valid source provenance; success is acknowledged only after persistence succeeds.
- [ ] #3 Adding to the originating parent stages a user-reviewable draft without sending it, overwriting an existing draft, switching workspaces, or interrupting active work.
- [ ] #4 Closed, deleted, changed-authority, or unavailable parent/note destinations produce a clear refusal or recovery choice rather than selecting a different destination.
- [ ] #5 Dismissal alone persists no side-chat content, and neither action promotes content into Personal Context or copies the entire parent context.
- [ ] #6 Targeted checks cover retained content accuracy, destination errors, existing parent drafts, duplicate activations, and late completion after navigation.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Implementation planning records ADR required: yes/no, ADR path, and reason; any required new or amended ADR is accepted and linked before implementation begins.
- [ ] #2 Automated tests (unit + integration) cover any new logic, the linter and formatter pass, and `./scripts/preflight.sh` passes, per CLAUDE.md's Definition of Done.
- [ ] #3 Relevant user guidance and living Architecture documentation describe the delivered behavior and its limits.
<!-- DOD:END -->
