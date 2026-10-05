---
id: TASK-31966
title: Investigate and reduce Console activation GC pauses
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-07 18:45'
updated_date: '2026-10-05 05:09'
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

### Approved Model-value scoped lookups — 2026-10-04

1. Add mounted regressions proving the three Model-value updates do not evaluate whole-screen queries while preserving structured values, hidden-rail updates, absent rows and remounted children/containers.
2. Resolve each existing row container with native ID lookup and query only that container for its value; preserve original update, missing-row, focus and config/ownership behavior. Do not bundle recovery lookup, memo, update equality or GC policy changes.
3. Verify RED/GREEN, affected Model-section contracts, design-token and artifact guards, static analysis and strict resource observations with targeted runs only.
4. Freeze clean source and remeasure native-query attribution and the real-owner scale matrix under unchanged limits; retain residual failures and external qualification gaps.
ADR required: no new ADR.
ADR path: N/A; existing ADR120, ADR150, ADR161 and ADR198 apply.
Reason: the approved change only narrows a DOM lookup to its current mounted owner, using native caches and preserving all freshness, authority, visual and lifetime contracts.

### Bounded measurement-input correction — current-source architecture checkpoint

1. Retain ac3a4f1 normal and diagnostic failures. Native GC-trigger evidence identifies a 73.944541 ms collection inside Pilot’s per-widget Enter barrier; the measured setup queues up to 751 callbacks. This identifies test-driver contribution, not every production stall.
2. Add a real mounted regression proving measured Enter delivers the native Input.Submitted event without a per-widget Pilot barrier. Observe legacy RED before replacing only that extra barrier with the installed native driver key-dispatch path; retain all exact-ready, modal registry removal, focus, transcript, source, corpus and busy/loop checks.
3. Run targeted measurement regressions, forwarding/exception checks as applicable, static analysis and independent review. Freeze new source and prepare fresh source-bound corpus/Keyword receipts before remeasuring the unchanged 52x20/120x50 matrix. Preserve residual failures.
4. Do not infer native qualification, allocation ownership, terminal resource retirement or a production latency pass from this harness correction; native/Windows/participant and global GC/cache gates remain unwaived.
ADR required: no new ADR.
ADR path: N/A; existing ADR120 and ADR198 apply.
Reason: this test-only correction removes artificial descendant callback fan-out while preserving the same installed native key dispatch and every real-owner acceptance boundary. It changes no application authority, lifecycle, UI or GC/cache policy. The user’s standing approval to fix scale verification covers this bounded correction, not any new architectural policy.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented and committed bounded recovery-height idempotence at 0ab325187e: mounted RED/GREEN, six final regressions pass with strict DB retirement, eleven artifact guards pass. Broader targeted run: 117 pass, four inherited config-admission failures reproduced without the fix. Same real-owner diagnostic measured five height updates at 0.375 ms summed versus 94.219 ms baseline. Fresh 10k/250k Keyword check passes: 300 exact queries, P95 240.416 ms, zero owned DB descriptors after cleanup. Full narrow/wide UI matrix still fails all eight 50 ms activation intervals (max 93.817 ms); a current-head GC trace proves collections contribute but do not explain every stall. Native/Windows/participant and terminal app-owner retirement gaps remain unwaived; all AC remain open. Existing ADR120/150/161/198 apply; no global GC/cache policy change. See Docs/QA/task-31245/fixture-rebuild-2026-10-04.md for source IDs, raw timings, failed runs and remaining work.

At clean ed124369f1, untimed small-profile heap traversal reached 6,389 of 6,617 unfrozen Strips from widget caches after two actual saved-chat resumes; boot pre-import freezing changes generation membership, so this is not leak or latency proof. Separate real-node observer counted 40 Send-reason, ten voice-status and ten attachment-indicator restyles in five unchanged control refreshes, zero dropped records or app exceptions. Shared Send-reason width/height remove-and-readd is the next bounded candidate; all four callers traced. Await its short design approval before production implementation, preserving budget/copy/voice behavior and existing ADR120/150/161/198. No GC policy change or qualification waiver. Raw source-bound receipts, failed observer setup and causal limits recorded in Docs/QA/task-31245/fixture-rebuild-2026-10-04.md.

Implemented the approved shared Send-reason atomic size-class correction, preserving budgets, copy safety and voice suppression. Canonical-ID mounted RED repeats without the fix; final new mounted, private-profile Send-disabled and token check passes 25 tests in 164.77s, no warnings, strict zero retained database files at all 25 teardown observations. Independent review found no production blocker. Representative config admission failures, retry-thread warning and unchanged CSS dimension failure reproduce with the fix removed; broader interrupted covering run is not qualified. New regression joins the UI census. Existing ADR120/150/161/198 apply; fresh clean-head restyle/scale measurement and external qualification remain pending, all AC open. Full evidence and failure limits appended to Docs/QA/task-31245/fixture-rebuild-2026-10-04.md.

Frozen ca793ee190 measurements: identical small real-node observer eliminates all 40 Send-reason restyles (zero dropped records/app exceptions), while voice/attachment remain ten each, intentionally separate. Fresh full corpus and 300 exact Keyword queries pass: P95 172.685959 ms, loop 17.772709 ms, zero DB descriptors after cleanup. Full UI matrix still fails all eight activation loop intervals (73.400–187.347 ms), both preparation windows, one search window and one busy paint (127.125750 ms); observer max 75.940500 ms retained as limitation. All 60 exact searches/eight exact opened activations complete; post-unmount retained app has 18 registered handles, not terminal retirement. No policy change or qualification waiver. Raw roots /tmp/task31966-send-measure-v0Z4Cs and full failure receipt documented in Docs/QA/task-31245/fixture-rebuild-2026-10-04.md. Residual attribution and native/Windows/participant/resource gaps remain, all AC open; no final combined PR or semantic work.

Same measured ca793ee190 source/corpus: existing GC observer retained 30 paired gen-2 collections, zero drops, callback max 0.021166 ms, unchanged thresholds. Several wide collections last 66–79 ms, but a 58.262666 ms wide activation window contains no gen-2 collection. Existing synchronous observer retained 66 >=20ms spans with no drops: shared config/control boundary has 20–60ms spans without gen-2 overlap; selected reflows include 59–104ms gen-2 overlap. Paint observers are <=3.245ms in these activation diagnostics. First-activation cProfile tables have inconsistent cumulative/internal and caller accounting, so are not used to choose authority/cache changes. All failed diagnostics retained; original branch/head restored after each exact-source detached run. Next bounded probe design separates native config scope entry/render/exit without skipping any checks, awaits or cross-pass caches; no production remedy yet selected. Existing ADR120/150/161/198 and all open qualification gaps remain. Full limits/receipts in Docs/QA/task-31245/fixture-rebuild-2026-10-04.md.

Approved throwaway native config entry/render/exit and rendering-substep probes completed at exact clean ca793ee190 against its original verified 10k/250k corpus. Original isolated branch/head restored after every failed matrix; all 60 exact searches/eight opened activations complete, no app exception, clean source/unchanged corpus, no drops or policy changes. Entry 2.2485–29.051625ms, render 2.925458–46.569666ms, exit 0.032–1.292667ms inside first-probe activation refreshes; none overlap gen-2. Summary extension attributes most summary cost to widget application, not state building. Final native query observer proves four screen-wide queries consume 6.501790–15.040541ms per rooted application; retained Static updates <0.24ms. Proposed bounded correction: scope only the three Model-value lookups to mounted row containers with native queries, preserving missing/remounted rows, live hidden-rail updates and all authority checks; await design approval before production edits. All eight activation loop intervals still fail every diagnostic; GC and other work remain separate contributors. No terminal resource/native/Windows/participant qualification or semantic gate waiver, all AC open. Digests, raw roots, failed matrices and scope limits retained in Docs/QA/task-31245/fixture-rebuild-2026-10-04.md.

Implemented the approved three Model-value scoped native lookups; recovery query, config/ownership boundaries, visual geometry and GC/cache policy unchanged. Mounted RED at both widths observes four whole-screen query evaluations rather than one; missing/remounted coverage already passes. Initial GREEN four pass. Final complete Model-section plus design-token check: 15 passed in 55.98s, no pytest warnings, required read-only descriptor gates pass in BOTH parent and private test children after reusing existing opt-in Console constructor ownership. Every mounted child census is empty; parent admission/lease remain. Earlier parent-only retirement claims are explicitly limited, not terminal lifetime proof. Eleven artifact guards green; changed test Ruff/format and production-range format clean; same 200 inherited full-file Ruff diagnostics, whitespace clean. No new ADR (existing ADR120/150/161/198); fresh clean-head scale/query attribution remains pending, all qualification AC open. See Docs/QA/task-31245/fixture-rebuild-2026-10-04.md for retained RED/failed strict run/child receipts and limits.

Frozen correction a0772b7c2a75cb1127f6138e9360402c6765f5b7 independently reviewed with no findings. Fresh full 10k/250k corpus and all 300 exact Keyword queries pass; P95 128.731708ms, loop 15.789042ms, zero owned DB descriptors/registered handles after cleanup. Normal 52x20/120x50 UI completes all 60 exact searches/eight exact OPENED activations, all busy/preparation limits pass, but seven activation intervals and one search exceed 50ms (activation max 111.111ms). Ordinary observer max 52.958875ms is explicitly retained as an overhead limit, not pure production cost. Unchanged native-query observer proves exactly one remaining root scan in all 15 rooted summary applications versus four baseline; query 1.958292–3.975500ms and apply 3.442042–9.092125ms, no drops/threshold changes. Diagnostic still has six failed activations (max 124.529209ms) with small paint observer 1.382459ms. Gen-2 explains part of wide stalls; narrow failed intervals also occur without GC. Actual post-unmount app owners remain live (16/18 handles), not terminal retirement. Every measured source/corpus guard is clean/exact and unchanged. After three bounded corrections, stop before a fourth production fix for architecture discussion of Console refresh/GC and measurement overhead; no global policy/cache change approved. Native/Windows/participant/resource and semantic gates remain open, all task AC unchecked; final combined PR not ready. Full receipt paths, raw intervals and limitations in Docs/QA/task-31245/fixture-rebuild-2026-10-04.md.

Current-dev integration at ac3a4f1: 288 affected tests pass, strict parent/private-file gates contain no database files. Fresh 10k/250k Keyword passes all 300 queries (P95 117.960333ms); normal UI retains seven activation and one search failures. Native trigger attribution proves a 73.944541ms gen-2 pause inside Pilot whole-screen Enter barrier with up to 751 callbacks, not scanned-object ownership or every production stall. Implemented bounded test-only same-native App._press_keys dispatch, retaining exact OPENED, modal unregister, exposure, focus, transcript, corpus/source and unchanged limits. Mounted RED receives native submission but detects four barrier callbacks; GREEN 48 affected tests in 10.84s, no warnings, strict private-file gate clean. Independent review finds no actionable issue. Existing ADR120/198; no new production, GC/cache policy or waiver. Fresh corrected-source scale/UI measurements and native/Windows/participant/terminal retirement remain pending; all AC open. Raw failures, hashes and causal limits retained in Docs/QA/task-31245/fixture-rebuild-2026-10-04.md.

Pre-commit correction gate: all eleven derived-artifact guards pass (/tmp/task31966-pilot-preflight.log); both changed benchmark Python files Ruff/format clean and git diff whitespace clean. No full sweep run. The reviewed helper and settlement checks are unchanged since the 48-test strict GREEN.
<!-- SECTION:NOTES:END -->
