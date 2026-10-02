---
id: TASK-33805
title: Console workspace or casual chats with bounded background starts
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-02 23:27'
updated_date: '2026-10-02 23:35'
labels:
  - console
  - agents
dependencies:
  - TASK-33804
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let a primary Console agent create a fresh chat in its workspace or casual scope, prepare a durable draft or start one bounded background turn, and preserve user control and recoverable outcomes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 new_chat supports both destinations and draft/start modes with strict validation and compatible omitted arguments; fork_chat retains its existing contract.
- [ ] #2 Fresh chats persist destination assistant and generation defaults, explicit instructions overrides, global/workspace ownership, fresh scratch/project state, and no source staged inputs or grants.
- [ ] #3 Both approval caches separate mode and resolved destination under live source ownership; the mounted card shows complete prompts and overrides without focus changes.
- [ ] #4 Version-2 drafts retain edits and explicit clears across activation/restart; legacy handoffs keep their decoding behavior and stale writes cannot restore old text.
- [ ] #5 Authorized machine-origin starts use shared allowance and wake/start capacity, preserve provenance through durable checkpoint migration, and return truthful draft/not_started/started/review_required results.
- [ ] #6 Manual Send wins while a start is prepared, source cancellation has the documented cutoff, paused preflight retains a draft, and duplicate/crash callbacks cannot repeat dispatch or consume newer input.
- [ ] #7 Targeted tests, lint/format/token checks and real Console/provider verification cover both destinations/modes, recovery and view detachment before completion.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/211-console-chat-destinations-and-bounded-starts.md
Reason: Implements the approved chat creation, machine-origin submission, durable draft and two-store recovery contract.

Planning phase only; no application implementation or acceptance criterion has been completed.
Detailed plan: Docs/superpowers/plans/2026-10-02-console-chat-destinations-and-starts.md, Task 2.
Spec: Docs/superpowers/specs/2026-10-02-console-chat-destinations-and-starts-design.md.

1. Add failing validation, scope/grant and durable destination-default tests.
2. Extend new_chat schema, trusted preparation and both approval caches; preserve fork_chat.
3. Persist destination identity/settings and revision-fenced version-2 drafts through existing runtime/store owners.
4. Migrate conversation dispatch checkpoints for native start origin and exact-attempt receipts, consuming matching drafts atomically.
5. Add runtime-owned one-attempt starts sharing allowance and fleet capacity; audit every origin-dependent submission branch.
6. Arbitrate manual intent before busy guards, enforce acceptance/cancellation cutoff, and make view completion an observer.
7. Verify targeted SQLite/Chat/mounted-UI tests, lint/format/token checks, documentation and real Console/provider behavior before closure.
<!-- SECTION:PLAN:END -->
