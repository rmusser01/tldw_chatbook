---
id: TASK-33663
title: Preserve hook review admission before Console Resend
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-02 20:15'
updated_date: '2026-10-02 20:28'
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
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/163-expanded-console-hook-runtime.md; backlog/decisions/197-console-hook-configuration-review.md; existing ADR-126/199 apply.
Reason: repair a bypass of the accepted admission and hook lifecycle contract by reusing its producers, currentness and retirement owners; no new permission or lifecycle owner.
1. Retain the independent real SQLite RED probes for hook review and saved/granted required SessionStart, and trace every Retry/Continue/Resend caller plus normal Send producers and consumers.
2. Add focused real-controller regressions for failed/stopped shapes, required initialization/input refusal, unchanged rows, accepted context/lineage and cancellation cleanup before implementation.
3. Reuse the existing normal submission admission, initialization/input scope and finally retirement at the shared execution seam before destructive clearing. Preserve exact maintenance-wake refund, close fencing and physical custody; avoid a consent-only patch or separate hook owner.
4. Qualify affected Send/Resend/wake/close consumers, static and unchanged performance guards, obtain immutable independent review, record limits and close through CLI.
<!-- SECTION:PLAN:END -->
