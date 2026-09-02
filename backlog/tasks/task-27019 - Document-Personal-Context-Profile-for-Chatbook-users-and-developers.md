---
id: TASK-27019
title: Document Personal Context Profile for Chatbook users and developers
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-01 14:45'
updated_date: '2026-09-02 03:57'
labels: []
dependencies: []
references:
  - Docs/superpowers/specs/2026-08-31-personal-context-documentation-design.md
  - >-
    backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Publish accurate, discoverable Chatbook documentation for using and extending the Personal Context Profile while clearly separating shipped synchronization behavior from planned capabilities.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The canonical user guide includes a quick start, task-oriented workflows, the full synchronization boundary, and the seven common troubleshooting states.
- [ ] #2 A developer guide maps Shared Core, encrypted local storage, interviews, agent authority, context injection, Sync-v2 integration, current limitations, the ten-item extension checklist, and targeted tests.
- [ ] #3 User and developer indexes link to the guides, and stable links connect to merged server documentation.
- [ ] #4 Documentation does not advertise unshipped delete-everywhere, purge acknowledgement, post-link conflict resolution, or server-origin publication behavior.
- [ ] #5 Targeted UI, Console, first-link, dispatcher, client, contract, link, and diff checks pass after the final rebase.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Rebase/inventory shipped behavior.
2. Task-oriented user guide.
3. Focused developer guide.
4. Discovery/server links.
5. Final targeted contract/link/diff verification.
6. Complete notes/open docs-only PR.

ADR required: no new ADR required; existing ADR applies
ADR path: backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md
Reason: Documentation only; the existing Personal Context authority, Sync, and encryption ADR applies.
<!-- SECTION:PLAN:END -->

## Renumbering provenance

- Previous ID: TASK-26835
- Current ID: TASK-27019
- Reason: current `origin/dev` contains the older `task-26835 - Textual-batch-updates-leave-the-screen-frozen-until-the-next-input-event.md` record (created 2026-09-01 14:27); this documentation record was created at 2026-09-01 14:45 and therefore moved under the younger-task-renumbers rule.
- Inbound references: the filename, frontmatter ID, and Chatbook documentation implementation plan references moved together to TASK-27019; no older task was changed.
