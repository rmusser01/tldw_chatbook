---
id: TASK-42311
title: Act on current conversation outputs through their owners
status: To Do
created_date: 2026-11-22 15:55
labels:
- console
- artifacts
- library
priority: high
dependencies:
- TASK-38935
references:
- https://uxplanet.org/anthropic-just-shipped-claude-2-0-bbb46a9c1383
- TASK-38935
documentation:
- Docs/Architecture/artifacts.md
- Docs/Architecture/canvas.md
- Docs/Architecture/console-file-authority.md
- backlog/decisions/172-library-artifacts-browse-and-navigation.md
- backlog/decisions/121-local-versioned-canvas-artifacts-and-browser-sandbox.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users continue working with a deliverable from the current conversation's Outputs list (TASK-38935) through the item's own owner. Split out of TASK-38935 on 2026-11-22 so the read-only list can land as one PR and the actions as another.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Open, revise, export, and locate-in-Library actions appear on an Outputs entry only when the item's actual owner supports and authorizes them, and each action runs through that owner.
- [ ] #2 An action on a missing, revoked, or unreadable item, or on an entry whose status changed since it was listed, refuses with the reason and never acts on a different item.
- [ ] #3 Actions follow the entry's exact conversation, branch/revision, workspace, and file authority, and none creates a duplicate content store or changes retention.
- [ ] #4 Mounted and targeted owner-integration checks cover each action per output kind, absent unsupported actions, owner failures, and stale entries after session switching.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Implementation planning records ADR required: yes/no, ADR path, and reason; any required new or amended ADR is accepted and linked before implementation begins.
- [ ] #2 Automated tests (unit + integration) cover any new logic, the linter and formatter pass, and `./scripts/preflight.sh` passes, per CLAUDE.md's Definition of Done.
- [ ] #3 Relevant user guidance and living Architecture documentation describe the delivered behavior and its limits.
<!-- DOD:END -->
