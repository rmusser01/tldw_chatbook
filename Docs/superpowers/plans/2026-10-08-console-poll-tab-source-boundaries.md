# Console tab source-boundary characterization

Task: TASK-34563.38. Extends Phase 1 of the existing Console polling plan.

ADR required: existing ADR.
ADR path: backlog/decisions/226-console-polling-and-full-state-reconciliation.md (Proposed), with ADR-126/222.
Reason: characterize existing full-refresh effects before selecting the already proposed polling route. No product API or scheduling change is authorized by these controls.

## Ownership and scope

This lane owns only `Tests/UI/test_console_poll_tab_source_boundaries.py` and this plan. The integration owner owns the task, product changes, review, native execution and final integration. Preserve all other drafts. Source parsing, formatting and linting here must not import the app or run native tests.

## Controls

1. Reuse the actual durable mounted Console fixture and original full-refresh entry. Hold the real Surface session lock and observe the original tab helper suspended in `Surface.sync_sessions`. No replacement tab, ensure, full-refresh, timer or publication body.
2. Intervene only on real state while that original await is held: add membership while preserving the active session's established settings; activate another real session that needs settings; reset the runtime's store/controller handles through their original reconstruction seams. Preserve the outgoing store/draft and observe the exact current target of the loop's repeated ensure.
3. Issue an overlapping original FULL request and require its retained demand to reach an original immediate/coalesced replay worker. Require the held helper to publish the captured membership and then the current owner in the same invocation. Keep normal defaults and any actual refusal visible.
4. Observe original settings ensure, native raw operations and descriptor retirement under the actual full-refresh ancestry. Check current store/session identity at ensure return, physical descriptor closes, issued leases/operations, exact draft behavior, task/worker retirement and source bindings. Use the existing fixture's real database and runtime disposal; explicitly retire a displaced original controller after reconstruction.
5. Freeze bounded source controls for integration review and sequential native execution. These are baseline characterization, not passing evidence for a new narrow API or a latency improvement. Root must distinguish fixture/observer failures from product findings.

## Limits

The explicit original FULL call makes the tab boundary deterministic without fabricating a run or receipt. Existing TASK-34563.34 controls separately qualify natural Preparing polling and real received-turn replay. This task does not replace them. H3/speech prelude, roleplay persistence and the proposed routing API remain unchanged. A missing-settings intervention is source-state fault injection; it must retain real creation-time provenance rather than invent a generation or readiness flag.

## First native baseline and observer correction

`575-tab-source-boundaries-original-1` reports three failures at the shared observer's final assertion: each case recorded three unrecognized FULL ancestry lookups that continued beyond the 40-frame bound. Independent XML/log review finds no earlier case assertion in the recorded exception chains. Six workspace-browser warning traces separately end in `ValueError: Chat conversation local service is unavailable`; they are a fixture coverage limitation, not the reported test failures. The after-fixture worker report is empty. Root reports frozen sources/HEAD and normal retirement (53.92 seconds pytest, 64.156 seconds driver). This run does not qualify routing or the post-fixture in-test native oracle.

The narrow source correction keeps the 40-frame bound and reports unrelated versus truncated lookups without attributing either to the test driver. Initial and overlap ownership still require the actual marker/task and held Surface Future; original replay workers retain their independent source/task checks. A failure property now records the test body's first exception before shared observer finalization can mask it. No body, deadline, product or native-lifetime assertion is relaxed. Root reviews and runs the correction sequentially.

## Second native baseline and membership-only correction

`575-tab-source-boundaries-original-2` passes active-session and store reconstruction. Each publishes the old and current membership through the same original tab invocation, performs one direct current-owner ensure and one original replay, preserves drafts/current source, and proves issued native/worker retirement. The after-fixture worker report is empty; root reports frozen source/HEAD and normal driver retirement in 63.657 seconds.

Membership fails at fixture exit when Textual propagates an original credential-readiness timer exception; the test body records no first-body failure. That display path reaches `_maybe_refresh_stale_default_console_settings`, which returns existing settings unchanged under a presentation projection. The artificial `settings=None` gap can therefore return `None` and raise the generic provenance error. That error alone does not establish an older generation or absent baseline.

The narrow correction gives membership its independent purpose: add the real inactive session without clearing the active session's established settings. Require zero direct tab-loop ensures, exact old-to-current membership publication, unchanged active settings/baseline/generation, drafts, pending FULL replay and original physical retirement. The qualified active-session and reconstruction cases retain their setup and assertions. No provenance is fabricated, readiness timer suppressed, deadline expanded or product changed. Missing-settings recovery remains covered by the existing active-session case; this membership result must not be reported as missing-settings recovery.

## Third baseline and exact request-task attribution

`575-tab-source-boundaries-original-3` again passes active-session and store reconstruction. Membership has no recorded first-body failure or provenance exception, but the observer rejects a second FULL start attributed to `initial`. The original FULL can eagerly start its replay worker from `finally`, retaining the original driver's stack ancestry while the executing Task belongs to the replay worker. An ancestry marker alone therefore cannot identify the issuing task.

The observer now requires the executing Task's actual coroutine to be `_request_original_full` and the discovered marker frame to be that exact coroutine frame before attributing `initial` or `overlap`. Other callers remain explicitly unrelated. The separately captured original replay Worker/Task/body checks remain unchanged. This tightens request attribution without changing product bodies, source interventions, normal timers, deadlines or native-retirement assertions. The old run remains observer-invalid for membership; root reviews and reruns the frozen correction sequentially.

## Fourth baseline and selected-invocation identity

`575-tab-source-boundaries-original-4` qualifies membership (zero direct ensures) and store reconstruction (one), each with two same-helper publications, one original replay, preserved drafts/current source and exact issued-resource retirement. Active-session fails only the observer's task equality check: an executing replay `Worker._run` is compared with the already-finished original request task. No first-body failure is recorded, and the after-fixture worker report is empty.

The original Surface/ensure selectors still compared numeric frame IDs before checking the issuing task. A frame number can be reused after the original task completes; stack ancestry can also cross eager task creation. Selection now requires the retained exact request Task together with the FULL/tab frame identity. Immediate replay creation uses that same compound identity, while coalesced replay keeps its independent original-worker proof. The selected original task/Surface Future, positive publication/ensure counts, source, drafts, native custody and physical retirement assertions remain unchanged. No production change, widened timeout or normal timer suppression is involved.

## Qualified original boundary evidence

At saved integration HEAD `575307709976fc92e100e2295b7e20ca5b934886`, `575-tab-source-boundaries-original-6` passes all three mounted original-body cases (39.754 seconds pytest, 48.187 seconds owned driver). Source and HEAD remain unchanged; after-fixture worker census is empty. Each branch records two publications by the same original tab loop and one completed pending FULL replay, with exact current owner objects, outgoing/current drafts and recorded native resources physically retired. Membership makes zero direct settings ensures; active-session and reconstruction each make one for their current owner. AST, Ruff lint/format and diff checks pass.

This closes Task34563.38 characterization only. The routine route must escalate before initial store creation and before a repeated ensure after the Surface await. It must also classify coachmark/membership effects. The complete Phase1 gate in the polling plan and Proposed ADR226 remains open; no production narrowing or latency success is implied.
