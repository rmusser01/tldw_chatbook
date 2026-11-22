---
id: TASK-38938
title: Organize workspace conversations into custom sidebar sections
status: To Do
created_date: 2026-11-05 16:09
updated_date: 2026-11-22 15:51
labels:
- console
- workspaces
- organization
priority: medium
references:
- https://uxplanet.org/anthropic-just-shipped-claude-2-0-bbb46a9c1383
- TASK-38936
documentation:
- Docs/User_Guide/console/sessions-tabs-workspaces.md
- backlog/decisions/010-console-conversation-local-marks.md
- backlog/decisions/083-console-edge-rails-and-workspace-tree-ownership.md
- backlog/decisions/219-console-chat-destinations-and-bounded-starts.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Help users organize several workstreams within one workspace without creating additional workspaces or changing conversation context and access. Make section membership an understandable presentation preference.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Users can create, rename, reorder, and remove named sidebar sections within an explicit workspace using keyboard-accessible controls.
- [ ] #2 A conversation has one section placement within its existing workspace, with an unsectioned fallback; removing a section never removes its conversations.
- [ ] #3 Section membership and ordering survive restart, with documented local/server/sync behavior and unambiguous handling of deleted, retired, or moved conversations.
- [ ] #4 Sections do not create shared context/memory, change workspace membership, alter file/provider permissions, or confer execution authority.
- [ ] #5 Stars, search, child-chat lineage (where TASK-38936's display is present), and active-run indicators retain consistent single-placement behavior; temporary search disclosure does not overwrite saved section state.
- [ ] #6 Targeted persistence and mounted navigation checks cover reorder, restart, removal, workspace changes, narrow layouts, and unavailable storage.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Implementation planning records ADR required: yes/no, ADR path, and reason; any required new or amended ADR is accepted and linked before implementation begins.
- [ ] #2 Automated tests (unit + integration) cover any new logic, the linter and formatter pass, and `./scripts/preflight.sh` passes, per CLAUDE.md's Definition of Done.
- [ ] #3 Relevant user guidance and living Architecture documentation describe the delivered behavior and its limits.
<!-- DOD:END -->
