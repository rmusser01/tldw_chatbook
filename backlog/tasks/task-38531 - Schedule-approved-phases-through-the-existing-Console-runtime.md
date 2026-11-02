---
id: TASK-38531
title: Schedule approved phases through the existing Console runtime
status: In Progress
created_date: 2026-11-01 17:31
dependencies:
- TASK-38524
- TASK-38525
- TASK-38526
- TASK-38527
- TASK-38528
- TASK-38529
- TASK-38530
labels:
- console
- agents
- workflows
references:
- Docs/superpowers/plans/2026-11-01-native-phased-agent-workflows-implementation.md#task-8
documentation:
- Docs/superpowers/specs/2026-11-01-native-phased-agent-workflows-design.md
- backlog/decisions/285-native-phased-agent-workflows.md
assignee:
- '@codex'
updated_date: 2026-11-02 01:36
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver implementation slice 8 of the accepted native shared-worktree workflow design so schedule approved phases through the existing Console runtime. Keep existing execution, permission and persistence ownership; this task does not authorize unrelated changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Read-only planning and approved task phases dispatch through typed native authority with per-session/global capacity and whole-chain accounting.
- [ ] #2 Host settlement events drive only approved eligible tasks; workflow-managed children do not independently trigger ordinary fleet wakes.
- [ ] #3 Ordinary/native opaque activity honors worktree claims; Pause/Stop/replan revoke launch authority and preserve user files and durable results.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect native bridge/controller, activity, instruction discovery, physical settlement and startup recovery seams; resolve source-backed integration gaps before production edits.
2. Add real native integration rig and failing dispatch/scheduling/admission/settlement journeys.
3. Deliver and independently gate internal slice 8A: recovery denial/release events and separate physical guidance footprints; retain whole-task RED journeys and all unrelated work.
4. Deliver and gate focused 8B: preserve ordinary run_reply signature/decorators and share its preparation/finalizers without live native/SQLite/contextvar scopes across the handoff. Split existing registry composition once before exact task authorization and Task7 seed, feeding the same planner/service. Retain original session/provider/review/instruction and client owners. Use finite run_preparation_read ownership; task mode omits primary generation/marker/run_turn and refuses capture without phase snapshot ownership. Retire original resources only after actual child drain and required durable settlement; prove missing-producer RED/GREEN, ordinary parity, duplicate/stale/failure/cleanup and exact-byte preservation before exact-path commit. Review fix round 1: reproduce actual provider-worker admission refusal and close-versus-launch interleavings; deliver typed terminal refusal through the existing stream queue and synchronize retirement/native launch under the original task authorization lock, without revoking admitted work on a refused close.
5. Deliver and gate focused 8C: derive complete metadata-only host freshness read scopes on original task custody, preserving model permission scope; validate after complete activity claim and actual native owner before child row/reservation commit/provider. Include existing native preparation external-shell provenance reads in its complete finite preflight footprint before acquisition; preserve consent, original source identity, known no-dispatch retry and native cleanup. Deliver subsequent narrowly scoped snapshot/outer-admission and phase-authority/continuation foundations, then event-driven coordinator integration. Preserve actual Change Review consent/phase windows and physical publication lifetimes; current Resume targets/tools and exact Pause/Stop semantics remain required.
6. Run the real dispatch/scheduling/admission/settlement journeys and named affected native tests; final all-commit Task8 review uses original BASE 1b3f64aa59ae2196d92bf0d486df55e2b30c1677. No partial gate completes parent AC.
7. Scoped static/preservation/self-review and final report. Leave parent status/AC pending full independent integration gate.
ADR required: no
ADR path: backlog/decisions/285-native-phased-agent-workflows.md
Reason: Direct implementation with source-backed private producer/physical/continuation rulings recorded in task-8-context.md; decomposition reduces uncertain combined refactoring without reducing accepted scope.
Internal 8D1 before D2: preserve existing coordinated workers and queues; acquire finite complete OPAQUE activity and actual native tool ownership before discover/B/E/survivor callbacks through original publication return; expose unresolved publication separately from physical retirement. Retry only actual capacity/activity release events; verify original roots/source and refuse native coordinator mismatches. Qualify native Git >=2.36, refuse configured clean/process/smudge filters before entry, and pin hooks/signing/fsmonitor/GC/maintenance and diff helpers per invocation. Preserve ignores, consent, historical and standalone legacy semantics. Amend ADR285/084 before production, then real native targeted RED/GREEN, parity, static/preservation and exact-path commit. Capture-enabled workflow tasks remain refused until D2. ADR required: no new ADR; paths: backlog/decisions/285-native-phased-agent-workflows.md and backlog/decisions/084-change-review-consent-and-asynchronous-finalization.md. Reason: approved narrow qualification of existing ownership/capture boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Internal slice 8B only: ordinary run_reply consumes the shared native preparation/finalizer frame. The private workflow producer composes one native catalog, obtains original task custody, builds the complete Task7 seed, and captures the genuine Task6 child preparation without a primary turn/message/model call. Finite root-read claims, original source/cancellation checks and durable-checkpoint/child-drain retirement retain native resources. The existing provider worker re-enters activation on its own thread with the original workflow source. ADR-285 applies; Console and agent-runtime architecture docs describe the private handoff.

Targeted producer/Task5/Task6, ordinary bridge/Goal/capture parity, static and preservation evidence is recorded in `.superpowers/sdd/2026-11-01-native-phased-agent-workflows-implementation/task-8b-report.md`. Capture-enabled task preparation refuses until genuine phase snapshot ownership is supplied. Launch freshness read admission, Pause/Stop and public event-driven coordinator integration remain pending; parent status and acceptance criteria remain incomplete.

8B review fix round 1 delivers actual provider-worker admission failures through the existing typed stream queue. Original task authorization now serializes native launch and the retirement decision; refused premature close preserves admitted work, and failed finalization cannot reopen launch. Bounded native refusal/race tests and affected gateway/child cleanup checks pass; the same report records RED/GREEN and exact preservation evidence. Independent slice re-review is still required.

Internal slice 8C derives complete host-only freshness scopes from the actual native checkpoint collector, adds them to original task authorization without expanding model scope, and validates after atomic activity/native owner admission before child row creation or allowance commit. Predispatch refusal positively drains and settles the finite claim while retaining unconsumed original reservation custody. The private producer also claims external native shell reads from predecessor review/dependencies before its finite callback; consent stays outside claims. Missing/failed predecessor checkpoints refuse before task custody/child reservation, retaining the existing reason and deferred phase-approval semantics. ADR-285 applies.

Targeted evidence: initial real held-collector RED (2 failures), final focused 13 checks passing, final duplicate-after-drain check passing, and affected Task5/6/7/8B gate 128 passing plus the 2 approved earlier-refusal boundaries subsequently corrected and passing. Static checks introduce no diagnostics; all 106 baseline/WIP file hashes and original staged patch are preserved. Details, commands, exact source/API/successor obligations and independent-review handoff: .superpowers/sdd/2026-11-01-native-phased-agent-workflows-implementation/task-8c-report.md. Capture/outer admission/coordinator/human-review and phase-settlement integration remain pending; parent status and AC remain incomplete.
Task 8D1 adds finite native ownership to the existing app-owned Change Review queues/workers/publisher: complete opaque admission, event-driven capacity/activity retry, original source/root guards, borrowed original opaque claim support, and physical worker-plus-publication return separate from typed observation/publication/unresolved outcomes. Discover failures retain original roots. Native Git >=2.36 is qualified before each command: configured clean/process/smudge filters and promisor/partial-clone remotes refuse before helper-capable entry; hooks/signing/fsmonitor/gc/maintenance and external diff/textconv are suppressed without changing global ignores, history, restore, attributes, or user config. ADR-285/084 and Console architecture updated. Final native tests: 34 passed; affected existing tests: 69 passed; selected raw ordinary bridge checks: 3 passed/8 config-source profile-blocked (only 2 have exact prior 8B failure evidence); exact eight-node supported bootstrap-profile diagnostic: 8 passed. Zero introduced Ruff diagnostics; original 106 dirty-file hashes and cached c5 patch preserved. Task 8D2 still owns ordinary/phase admission integration, window/durable settlement, and the separate consent root initializer's native ownership/helper qualification; capture-enabled workflow preparation remains refused. Parent remains In Progress with AC unchecked; fresh root review gate required.
Final D1 self-review added two native overload/degraded no-entry cases using a real held discovery and baseline timeout: both pass, with no further production changes. Final evidence totals 34 native cases plus these 2 focused additions; exact commands, final-byte hashes, raw/qualified profile attribution and preservation audit are retained in task-8d1-report.md.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
