---
id: TASK-31384
title: >-
  Console unified interrupt surface: one round helper and one card host for the
  five blocking prompt kinds
status: To Do
assignee: []
created_date: '2026-09-04 19:29'
labels:
  - console
  - refactor
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Console controller now carries FIVE hand-cloned copies of the same blocking round loop -- MCP approvals, skill install, skill script, worktree merge, and (since PR #2379) ask_user questions -- each with its own registry, lock, retained-payload map, marshal, remount-at-activation trio, and revocation leg, and ChatTaskCards routes each kind to its own bespoke card. Every new kind is a fourth or fifth copy of ~150 lines and three activation-site edits, and every bug fix (PR #1836's round-keying, M2's atomic busy check) has to be applied per copy. Sub-project C of the design spec (2026-08-19-console-user-interaction-design.md section 4) is the extraction: one _run_pending_round helper the kinds parameterise, and one card host routing by kind. A first spine (C1) was designed and implemented on PR #1903 and closed unmerged on 2026-09-04 after dev's approvals-verdict rewrite outran it; its design doc (2026-08-20-console-interrupt-host-design.md) and plan are recoverable from that branch's history and remain valid. The extraction is a refactor with a parity oracle: the existing interrupt battery must pass unchanged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One shared round helper backs all five kinds; the per-kind request_* methods keep their names and return shapes
- [ ] #2 One host in the task-card slot routes payloads to cards by kind, including lazy mounting for kinds that stay off the boot path
- [ ] #3 The interrupt battery (approval, skill-install, skill-script, worktree-merge, ask_user suites and the concurrency suites) passes with the same failure set as clean dev
- [ ] #4 Per-kind behaviour differences (FIFO queueing for approvals vs busy for questions; verdict keying; revocation) are expressed as parameters, not copies
<!-- AC:END -->
