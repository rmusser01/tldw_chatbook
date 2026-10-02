---
id: TASK-33663
title: Preserve hook review admission before Console Resend
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-02 20:15'
updated_date: '2026-10-02 20:53'
labels:
  - agents
  - console
  - integration
dependencies: []
documentation:
  - Docs/superpowers/plans/2026-09-29-agent-orchestration-burndown.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Independent PR2918 integration review reproduced persisted Resend dispatch while normal Send is refused for hook review. The existing common Resend route must respect the same authority before clearing or retrying a broken turn.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Normal Send and every Resend shape refuse the same unavailable or unreviewed hook authority before provider dispatch or destructive clearing.
- [ ] #2 Real controller and SQLite regressions fail before the repair and prove failed and stopped replies remain unchanged under refusal, then permit one explicit retry after authority is restored.
- [ ] #3 Existing exact wake refund, accepted nonreplay and saved-close custody remain valid; targeted static checks and independent review approve.
- [ ] #4 Granted required v2 initialization and input checkpoints execute before any Resend clear; refusal or cancellation leaves the original rows, no provider dispatch, and no leaked turn scope, process or submit-task ownership.
- [ ] #5 Resend preserves validated hook context and existing configuration/session currentness; a permitted attempt runs once with exact in-place lineage and normal lifecycle retirement.
- [ ] #6 Public Retry and Continue use the same admission, preparation and retirement boundary as Resend; queued Retry keeps its existing queue authority and scheduled input semantics.
- [ ] #7 Transcript polling survives the initial awaited hook admission read for an existing replay worker and publishes incremental replies before completion.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/163-expanded-console-hook-runtime.md; backlog/decisions/197-console-hook-configuration-review.md; backlog/decisions/098-visible-bounded-console-prompt-queue.md; existing ADR-126/199 apply.
Reason: restore accepted hook lifecycle, queue reservation ownership and existing worker-driven publication; no new permission, scheduler or lifecycle owner.
1. Preserve independent real SQLite hook-review and required-initialization RED probes; trace every Retry/Continue/Resend and normal Send caller.
2. Preserve the existing real hook/context/currentness/cancellation regressions. Add the independently reproduced runtime-bound queued Retry and mounted initial-hook-read polling regressions before further implementation.
3. Reuse normal submission admission, initialization/input/currentness and finally retirement before row clearing. Extend the existing coordinator-owned slot predicate only for explicitly requested authorized recovery with no claimed entry; retain HELD, terminal, no accepted live turn, currentness and global-cap checks. Keep transcript publication alive for existing unfinished console-run workers, covering sibling replay actions without another owner.
4. Qualify affected Send/Resend/queue/wake/close and mounted polling consumers, static and unchanged performance guards. Obtain immutable independent runtime and UI review, record every non-green limit and close through CLI.
<!-- SECTION:PLAN:END -->
