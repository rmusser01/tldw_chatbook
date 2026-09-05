---
id: TASK-31552
title: llama.cpp manual prompt-cache snapshot manager
status: To Do
assignee: []
created_date: '2026-09-05 01:15'
labels: []
dependencies: []
references:
  - backlog/decisions/119-llamacpp-prompt-cache-snapshot-ownership.md
documentation:
  - Docs/superpowers/specs/2026-09-04-llamacpp-slot-snapshots-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users manually preserve and reload processed llama.cpp context, including supported image and audio context, from a server launched inside Chatbook. Provide predictable private storage and configurable retention without implying conversation recovery or guaranteed cache reuse.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Users can save a selected slot with an automatically generated timestamp name and restore a saved snapshot to an eligible slot on a Chatbook-launched server.
- [ ] #2 The manager retains the newest 10 complete snapshots per profile by default, supports a validated configurable count, and prunes only after a fully committed successful save.
- [ ] #3 Snapshot operations honor launch identity, endpoint readiness, compatibility evidence, private file ownership, and uncertain operation outcomes across navigation and restart.
- [ ] #4 The UI explains cache-only restore semantics, exposes actionable failure and partial-success states, and remains keyboard usable in the production Models screen.
- [ ] #5 Targeted automated checks and an isolated real-server save/restart/restore test prove persistence and actual same-image prefix reuse with an eligible model.
<!-- AC:END -->

## Design status

The user selected manual management before automatic per-conversation persistence,
Chatbook-launched servers, timestamp-generated names, and configurable retention
with a default of 10. The reviewed specification and ADR are linked above.
Implementation has not started; acceptance criteria remain unchecked.

ADR required: yes

ADR path: backlog/decisions/119-llamacpp-prompt-cache-snapshot-ownership.md

Reason: new private snapshot files, automatic deletion, and a llama-server
management boundary. Existing ADR-029 and ADR-036 also apply.

ID allocation: the CLI offered 31429; refs and 64 worktrees contained task IDs
through 31551, so this record was moved to 31552 before linking it elsewhere.
Recheck allocation before integration.
