---
id: TASK-38935
title: Find and reopen current conversation outputs with provenance
status: To Do
created_date: 2026-11-05 16:08
labels:
- console
- artifacts
- library
priority: high
references:
- https://uxplanet.org/anthropic-just-shipped-claude-2-0-bbb46a9c1383
documentation:
- Docs/Architecture/artifacts.md
- Docs/Architecture/canvas.md
- Docs/Architecture/console-file-authority.md
- backlog/decisions/172-library-artifacts-browse-and-navigation.md
- backlog/decisions/121-local-versioned-canvas-artifacts-and-browser-sandbox.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give users one reliable place to find deliverables created during a conversation and continue working with them. Preserve the distinction between temporary output, saved content, and generated files that are no longer available.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A current-conversation Outputs section lists supported generated files, Canvas revisions, and saved notes/artifacts only when their existing owners establish a valid relationship to that conversation.
- [ ] #2 Each entry identifies its type, actual source turn/run where known, and temporary/saved/missing status; unlinked or legacy provenance is labeled unknown.
- [ ] #3 Open, revise, export, and locate-in-Library actions appear only when the item's actual owner supports and authorizes them.
- [ ] #4 Items retain their canonical storage and lifecycle owners; the feature creates no aggregate artifact database, duplicate content store, or automatic retention promise.
- [ ] #5 Entries follow exact conversation, branch/revision, workspace, and file authority; missing files, revoked bindings, and owner read failures remain explicit.
- [ ] #6 Mounted and targeted owner-integration checks cover mixed output kinds, revisions, empty/unavailable states, and session switching without stale actions.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Implementation planning records ADR required: yes/no, ADR path, and reason; any required new or amended ADR is accepted and linked before implementation begins.
- [ ] #2 Automated tests (unit + integration) cover any new logic, the linter and formatter pass, and `./scripts/preflight.sh` passes, per CLAUDE.md's Definition of Done.
- [ ] #3 Relevant user guidance and living Architecture documentation describe the delivered behavior and its limits.
<!-- DOD:END -->
