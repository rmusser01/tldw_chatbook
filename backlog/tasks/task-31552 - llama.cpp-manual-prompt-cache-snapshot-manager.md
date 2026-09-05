---
id: TASK-31552
title: llama.cpp manual prompt-cache snapshot manager
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-05 01:15'
updated_date: '2026-09-05 01:43'
labels: []
dependencies: []
references:
  - backlog/decisions/119-llamacpp-prompt-cache-snapshot-ownership.md
documentation:
  - Docs/superpowers/specs/2026-09-04-llamacpp-slot-snapshots-design.md
  - Docs/superpowers/plans/2026-09-04-llamacpp-slot-snapshots.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users manually preserve and reload processed llama.cpp context, including supported image and audio context, from a server launched inside Chatbook. Provide predictable private storage and configurable retention without implying conversation recovery or guaranteed cache reuse.

### Design status

The user selected manual management before automatic per-conversation persistence,
Chatbook-launched servers, timestamp-generated names, and configurable retention
with a default of 10. The reviewed specification and ADR are linked above.
Implementation has not started; acceptance criteria remain unchecked.

The user approved integration of the follow-up review: compatibility-gated save
publication, integrity checks before restore, proxy-free loopback transport,
terminal working-file cleanup, separate probe/operation deadlines, and visible
cross-model retention wording. These are requirements for implementation, not
claims of tested application behavior. ADR-119 records the accepted amendments.

ADR required: yes

ADR path: backlog/decisions/119-llamacpp-prompt-cache-snapshot-ownership.md

Reason: new private snapshot files, automatic deletion, and a llama-server
management boundary. Existing ADR-029 and ADR-036 also apply.

ID allocation: the CLI offered 31429; refs and 64 worktrees contained task IDs
through 31551, so this record was moved to 31552 before linking it elsewhere.
Recheck allocation before integration.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Users can save a selected slot with an automatically generated timestamp name and restore a saved snapshot to an eligible slot on a Chatbook-launched server.
- [ ] #2 The manager retains the newest 10 complete snapshots per profile by default, supports a validated configurable count, and prunes only after a fully committed successful save.
- [ ] #3 Snapshot operations honor launch identity, endpoint readiness, compatibility evidence, private file ownership, and uncertain operation outcomes across navigation and restart.
- [ ] #4 The UI explains cache-only restore semantics, exposes actionable failure and partial-success states, and remains keyboard usable in the production Models screen.
- [ ] #5 Targeted automated checks and an isolated real-server save/restart/restore test prove persistence and actual same-image prefix reuse with an eligible model.
- [ ] #6 Save is disabled without complete required compatibility evidence; evidence invalidated before publication prevents retaining the new snapshot or pruning older ones, including with keep count 1.
- [ ] #7 Restore verifies staged byte length and SHA-256 before any Restore POST; truncated or same-length corrupted input leaves the destination slot untouched.
- [ ] #8 All management and readiness traffic uses a validated numeric loopback destination with proxies and redirects disabled; proxy environment variables and redirect responses cannot forward requests or credentials elsewhere.
- [ ] #9 Successful and acknowledged terminal operations and proven pre-submission failures release safe working files; repeated restores do not accumulate copies, cleanup failures expose residual bytes, and uncertain operations retain files until safe.
- [ ] #10 Five-second probe deadlines are separate from explicit ten-minute Save/Restore submission deadlines; preparation and elapsed operation status remain visible and slow valid operations are not failed at the probe deadline.
- [ ] #11 The Save area visibly states the effective newest-N retention limit across all models, including count changes and narrow terminal layouts.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/119-llamacpp-prompt-cache-snapshot-ownership.md. Reason: accepted snapshot ownership and retention contract; ADR-029 and ADR-036 also apply. Execute Docs/superpowers/plans/2026-09-04-llamacpp-slot-snapshots.md in six reviewed units: (1) strict settings and effective launch admission; (2) private transactional storage and integrity; (3) bounded loopback-only management HTTP; (4) app-owned operation and subprocess lifecycle; (5) manual Models widget and canonical F9 settings; (6) isolated real-server reuse evidence and closeout. Use targeted RED/GREEN tests per unit and keep all acceptance criteria open until their evidence exists. No application implementation has started.
<!-- SECTION:PLAN:END -->
