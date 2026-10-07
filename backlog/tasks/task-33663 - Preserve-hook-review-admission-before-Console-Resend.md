---
id: TASK-33663
title: Preserve hook review admission before Console Resend
status: Done
assignee:
  - '@codex'
created_date: '2026-10-02 20:15'
updated_date: '2026-10-02 21:38'
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
- [x] #1 Normal Send and every Resend shape refuse the same unavailable or unreviewed hook authority before provider dispatch or destructive clearing.
- [x] #2 Real controller and SQLite regressions fail before the repair and prove failed and stopped replies remain unchanged under refusal, then permit one explicit retry after authority is restored.
- [x] #3 Existing exact wake refund, accepted nonreplay and saved-close custody remain valid; targeted static checks and independent review approve.
- [x] #4 Granted required v2 initialization and input checkpoints execute before any Resend clear; refusal or cancellation leaves the original rows, no provider dispatch, and no leaked turn scope, process or submit-task ownership.
- [x] #5 Resend preserves validated hook context and existing configuration/session currentness; a permitted attempt runs once with exact in-place lineage and normal lifecycle retirement.
- [x] #6 Public Retry and Continue use the same admission, preparation and retirement boundary as Resend; queued Retry keeps its existing queue authority and scheduled input semantics.
- [x] #7 Transcript polling survives the initial awaited hook admission read for an existing replay worker and publishes incremental replies before completion.
- [x] #8 Transcript text-generation actions honor the existing shared provider-readiness refusal before worker launch or destructive replay; restored readiness permits one explicit replay, while image/video actions retain their existing authority.
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
5. Retain the corrected real mounted readiness RED on candidateb887 and exact incoming185c; route Retry/Resend/Continue/text Regenerate/edit-resend worker admission through the existing screen provider refusal and controller activity gate using the established message-controller callable wiring. Keep image/video routes unchanged; qualify real refusal, unchanged rows and one permitted replay. Existing ADR012/033 apply; no new evidence owner or ADR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Shared normal Send hook admission, required initialization/input, captured context/currentness and final retirement across public Retry/Continue/Resend before destructive clearing. Narrow coordinator recovery reuses only the authorized held slot; existing unfinished console-run workers retain transcript publication during initial awaited hook admission. Existing constructor callable wiring combines current activity then provider-readiness refusal before Retry/Resend/text Regenerate/Continue/edit-resend worker launch; image/video keep separate gates. No new owner, schema, permission, dependency or cap. Existing ADR012/033/098/126/163/197/199 apply.
Real hook/runtime/SQLite and mounted RED regressions are retained. Root40neighbors,35readiness,10wake/close and5final readiness/media controls pass; final independent runtime/UI reviews approve2cfbb01c76. Original five storage/import/UI/CSS budgets pass85.975s;94patchPython fatal/added-line,tennewRuff/format and artifact guards pass. Raw configuration/bare-shell failures reproduce on incoming source; adapted bootstrap ports and corrected scratch controls are recorded distinctly. Detailed reviewed positive/non-green evidence, scope limits and exact source manifest are in Docs/superpowers/reviews/2026-09-29-agent-orchestration-burndown.md. Production changes are confined to existing controller/replay/coordinator and UI worker/callable boundaries, with durable tests and explicit fixture ports. Fresh-head Qodo/CI and protected merge remain delivery gates.
<!-- SECTION:NOTES:END -->
