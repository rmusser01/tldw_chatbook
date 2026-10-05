---
id: TASK-31966
title: Investigate and reduce Console activation GC pauses
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-07 18:45'
updated_date: '2026-10-05 00:56'
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Retain the frozen baseline, exact identity outcomes, GC timings, observer costs and all failed runs without relabelling them.
2. Attribute stable control-refresh substeps in a small disposable real-app profile; distinguish nested work, instrumentation effects and GC from exclusive function cost.
3. Obtain design approval for the newly identified bounded recovery-bar height-class correction before changing production code.
4. After approval, add genuine mounted regressions for unchanged refreshes, recovery transitions, unrelated classes and inline reset behavior, then implement atomic idempotent class replacement in the existing owner.
5. Freeze clean source and rerun the relevant targeted tests, strict resource checks, artifact guards and fresh real-owner scale matrix against unchanged timing limits. Preserve residual failures; do not change GC policy to force a pass.
6. Keep native, Windows, participant and final application-owner retirement gaps explicit; do not start the dependent semantic subsystem or mark qualification complete without its required evidence.
ADR required: no new ADR for the proposed local, behavior-preserving class correction.
ADR path: N/A; existing backlog/decisions/120-character-conversation-navigation-and-local-semantic-search.md, backlog/decisions/150-design-token-system-and-design-language.md, backlog/decisions/161-component-pattern-library.md and backlog/decisions/198-gc-policy-freeze-boot-heap.md govern the work.
Reason: existing geometry, tokens, activation, authority and lifetime contracts are preserved; any later global GC/cache policy requires separate approved ADR review.

### Historical evidence and scope — 2026-09-07 checkpoint

Retained from the original task. This subsection is inside the CLI-owned plan
marker so future task serialization does not discard its evidence.

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
approved architectural decision. At the original checkpoint, no implementation
plan or remedy was approved. The current bounded plan above does not authorize a
global policy change or claim qualification completion.

### Approved Send-reason correction — 2026-10-04

1. Add mounted RED regressions for unchanged visible/hidden/empty Send reasons and atomic state transitions; retain real width budgets and class restyling.
2. Replace only the shared Send-reason owner's size-class mutations atomically; preserve unrelated classes, inline resets, escaped copy/setup link and full-width voice preparation.
3. Verify mounted geometry, resize, conflicting overrides and voice/disabled-state contracts with targeted tests and strict resource observations. Do not bundle voice/attachment candidate fixes.
4. Commit clean source and rerun the identical small restyle observer with freshly prepared exact-head source, then the real-owner scale matrix under the unchanged limits. Preserve residual failures and external qualification gaps.
ADR required: no new ADR.
ADR path: N/A; existing ADR120, ADR150, ADR161 and ADR198 above apply.
Reason: a local rendering-idempotence correction preserves every authority, sizing, ownership and GC/cache policy boundary. User approved this bounded design after the ed124 restyle evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented and committed bounded recovery-height idempotence at 0ab325187e: mounted RED/GREEN, six final regressions pass with strict DB retirement, eleven artifact guards pass. Broader targeted run: 117 pass, four inherited config-admission failures reproduced without the fix. Same real-owner diagnostic measured five height updates at 0.375 ms summed versus 94.219 ms baseline. Fresh 10k/250k Keyword check passes: 300 exact queries, P95 240.416 ms, zero owned DB descriptors after cleanup. Full narrow/wide UI matrix still fails all eight 50 ms activation intervals (max 93.817 ms); a current-head GC trace proves collections contribute but do not explain every stall. Native/Windows/participant and terminal app-owner retirement gaps remain unwaived; all AC remain open. Existing ADR120/150/161/198 apply; no global GC/cache policy change. See Docs/QA/task-31245/fixture-rebuild-2026-10-04.md for source IDs, raw timings, failed runs and remaining work.

At clean ed124369f1, untimed small-profile heap traversal reached 6,389 of 6,617 unfrozen Strips from widget caches after two actual saved-chat resumes; boot pre-import freezing changes generation membership, so this is not leak or latency proof. Separate real-node observer counted 40 Send-reason, ten voice-status and ten attachment-indicator restyles in five unchanged control refreshes, zero dropped records or app exceptions. Shared Send-reason width/height remove-and-readd is the next bounded candidate; all four callers traced. Await its short design approval before production implementation, preserving budget/copy/voice behavior and existing ADR120/150/161/198. No GC policy change or qualification waiver. Raw source-bound receipts, failed observer setup and causal limits recorded in Docs/QA/task-31245/fixture-rebuild-2026-10-04.md.

Implemented the approved shared Send-reason atomic size-class correction, preserving budgets, copy safety and voice suppression. Canonical-ID mounted RED repeats without the fix; final new mounted, private-profile Send-disabled and token check passes 25 tests in 164.77s, no warnings, strict zero retained database files at all 25 teardown observations. Independent review found no production blocker. Representative config admission failures, retry-thread warning and unchanged CSS dimension failure reproduce with the fix removed; broader interrupted covering run is not qualified. New regression joins the UI census. Existing ADR120/150/161/198 apply; fresh clean-head restyle/scale measurement and external qualification remain pending, all AC open. Full evidence and failure limits appended to Docs/QA/task-31245/fixture-rebuild-2026-10-04.md.
<!-- SECTION:NOTES:END -->
