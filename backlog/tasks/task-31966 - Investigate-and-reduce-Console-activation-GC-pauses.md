---
id: TASK-31966
title: Investigate and reduce Console activation GC pauses
status: To Do
assignee: []
created_date: '2026-09-07 18:45'
labels:
  - console
  - performance
  - follow-up
dependencies: []
references:
  - Docs/QA/task-31245/README.md
  - Tests/Benchmarks/console_character_switcher_latency.py
  - >-
    backlog/decisions/120-character-conversation-navigation-and-local-semantic-search.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow up separately on the main-thread garbage-collection pause found during TASK-31245 Character switcher qualification. Determine lifecycle/allocation ownership and deliver an evidence-backed bounded correction without expanding the current feature PR or disguising its failed latency measurements.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A disposable-profile reproduction identifies allocation and lifecycle ownership, compares a frozen baseline, and distinguishes observer overhead from automatic GC costs.
- [ ] #2 An evidence-backed correction preserves exact conversation activation, focus, cancellation, and terminal resource ownership; any global GC or cache policy change has an approved ADR before implementation.
- [ ] #3 The corrected real-owner latency matrix at 52x20 and 120x50 meets the existing 50 ms event-loop and 100 ms busy-paint limits, with raw timings and corpus/source provenance retained.
- [ ] #4 Targeted regressions and repeated terminal resource checks pass without warning suppression, threshold increases, real-profile access, or replacing native evidence with mocks.
<!-- AC:END -->

## Evidence and scope

On 2026-09-07 the user explicitly chose a separate follow-up for this investigation.
This task does not reopen the reviewed TASK-31245 activation/rollback/resource or
measurement-harness corrections, and does not defer its unrelated native checks.

- Diagnostic source: f5b943c8f361c05cf53e61a67c3fb440b685cf4e.
- Harness correction: 61adfe72399cf2ed37f5bec843d4311a3d499bd1.
- Attempt7: activation maximum event-loop interval 87.732709 ms, containing an
  automatic main-thread generation-2 collection of 79.256875 ms wall time and
  79.092125 ms CPU time. Collection reclaimed 13,497 objects; none uncollectable.
- Exact conversation OPENED, Console exposure, transcript identity and modal
  registry removal succeeded. Preparation maximum interval was 30.153542 ms.
- Collected-object ownership is unknown. Bounded observers dropped 71 GC and 269
  owner-detail records; instrumentation allocations may shift collection timing.
  This is not proof of a leak or a baseline Textual regression.
- No speculative global GC/cache policy, forced collection or threshold increase
  was applied. Existing transcript reconciliation already prunes removed row
  references, reuses unchanged rows and batches mount/removal.

Durable summary: Docs/QA/task-31245/README.md. Local raw evidence is retained in
.superpowers/sdd/2026-09-05-character-keyword-release-isolation/ui-latency-task5/attempt-7/runtime/evidence/ui-latency-evidence.json;
the associated task-5-gc-scope-assessment.md records causal limits. These ignored
artifacts must be preserved before worktree cleanup; their paths alone are not
portable evidence. Reconfirm the problem on the current baseline before designing
a fix. The existing standalone 300-query retrieval pass is not UI latency proof.

ADR review is required when taking this task In Progress. ADR120 governs existing
activation authority; a new global GC/cache/lifecycle policy requires its own
approved architectural decision. No implementation plan or remedy is approved yet.
