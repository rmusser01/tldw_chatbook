# Console context publication during Preparing

Task: TASK-34601, OPT-10.

ADR required: no new ADR.
ADR path: backlog/decisions/226-console-polling-and-full-state-reconciliation.md (bounded context-publication amendment); preserves ADR-225.
Reason: route an existing disposable presentation publication through the existing qualified manual Preparing refresh. Preserve every eligibility, source, lifetime, custom callback and FULL fallback rule; introduce no cache, lifetime owner, service contract or freshness policy.

## Evidence and scope

At e3ee73c9d2, two qualified original action-to-provider observers see 4/3/3 and 4/2/4 context reads. The second resolves the exact resident ConsoleRunStatus enum: one warm status-only invalidation at age0.625s awaits106.184ms. Other reads cross payload/display revisions; some results are correctly discarded. Clipped await totals of763.010/841.152/1073.460ms in the second run include scheduling and overlapping work, not removable critical-path savings. Exact changed publications schedule FULL through ConsoleContextReadSnapshot. Receipts remain under context-refresh-attribution/context-refresh-{1,2}; no key or TTL change is selected.

The existing _sync_console_poll_display_ui already narrows only a source-current manual received intent, delegates to the original full owner, and retains the no-keyword callback for custom FULL implementations. Its narrowed pass still publishes all original presentation surfaces and skips only already-qualified chat-core/roleplay reconciliation. Reuse that route for the context memo's default publication callback when available; legacy/custom screens retain the originally captured no-argument callback. A publication arriving during an existing sync must call the captured FULL callback, which preserves trailing demand; the ordinary poll overlap branch alone would drop it. Select and enter that route without an intervening await. Do not build another coordinator, change the key or TTL, skip display reads, or relax custody.

## Implementation and ownership

1. Controller/provider lane reviews the existing route for differences caused by this caller. Root integrates and is the only local native test/timing owner. No product change until the source review is complete.
2. Controller/provider lane owns the small callback selection in UI/Console_Modules/console_spend_projection.py and focused context-publication tests. All other product/test files stay frozen unless root explicitly transfers ownership.
3. Establish RED with the real context-publication callback during a qualified held manual Preparing attempt. Count original FULL-only reconciliation entry, verify real transcript/controls publication, and retain native retirement. Add an overlap control that publishes after a running FULL consumed old context and requires trailing publication. Preserve legacy/custom captured callbacks and independent FULL requests.
4. Make the minimum callback-routing change. Re-run the regression and existing context presentation, Preparing, polling, source-change, deferral, teardown and native lifetime cases selected by the actual delta. Preserve deadlines.
5. Freeze both implementations before final integrated checks. Run quiet, sequential, full-profile A/B/B/A against e3ee73c9d2, with unchanged source/overlap/provenance guards. Adopt only a demonstrated whole-Send benefit or independently justified behavior correction; otherwise remove the experiment and retain its evidence in the optimization ledger.

## Limits

This route only removes reconciliation from context publications during the existing narrow Preparing interval. Other publications still use FULL, and the narrowed pass still includes some native display reads. No saving is promised from phase-wide helper counts, which include response settlement. The one-second application and sub-100ms rendered-feedback targets remain open.


## Implemented and locally qualified (2026-10-09)

The snapshot captures FULL and the optional existing poll-display callback once. Its small wrapper retains captured FULL when a sync is running or the screen has replaced that callback; exact bound receiver/body identity recognizes ordinary repeated attribute access without invoking custom equality. Otherwise it enters the existing guarded manual-Preparing route. No intervening await, context key/TTL change, new coordinator or native lifetime owner was added.

Original source: context-routing-red-1 fails the two intended routing checks after successful original transcript/control publication; four compatibility/overlap cases pass. The temporary direct-swap negative control fails exactly on lost _console_sync_requested demand (context-routing-overlap-negative-1). Final integrated selection passes51 cases in167.63s, including changed runtime configuration, FULL replay, final deferral, owner changes, exact context-host retirement and both corrected plugin controls. The primary publication test was then strengthened to require successful True completion; all six final routing cases pass in24.98s. Source/HEAD stable in each run. Ruff checks pass; independent review found no actionable issue.

Quiet full-profile sequential ABBA plus one BA confirmation, cold/warm/warm seconds:

| Run | Seconds |
| --- | --- |
| baseline A1 | 4.601554 / 2.763011 / 3.315037 |
| candidate B1 | 4.463632 / 2.873153 / 2.757184 |
| candidate B2 | 3.975742 / 2.817478 / 2.755896 |
| baseline A2 | 4.110389 / 15.515540 / 4.042185 |
| candidate B3 | 4.433306 / 2.731259 / 2.867563 |
| baseline A3 | 4.329425 / 3.398638 / 2.629956 |

All18 saved turns complete their traces with zero checkpoints, unchanged sources and NO_DETECTED_OVERLAP. The only normalized loaded product difference is console_spend_projection.py; raw_participants.py, storage_admission.py and server.py differ only in line endings. Candidate source is the uncommitted, fingerprinted change above c9fb3fbc6a; baseline is e3ee73c9d2, with identical product code to c9fb before this experiment.

The unexplained15.515540s baseline outlier is retained. Its broad stage slowdown prevents treating the initial56.3% mean difference as a causal speedup. The separate confirmation pair is3.014297s baseline versus2.799411s candidate warm mean (7.13%); all six candidate warm samples are2.731259–2.873153s. Six-sample warm medians are3.356838/2.787331s. This small headless/stub-provider sample supports retaining the bounded work reduction, not a percentile, cold-speed, consistent heartbeat or physical-terminal claim. One-second Send and sub100ms rendered-feedback acceptance remain open. Receipt context-routing-comparison.json sha256 2a887cd1214ed79e4e3a06506ec321e06524664fdceae55b9c4bec0d24b0924e.

The existing three-OS Preparing job now includes the six publication cases and the two empty-ceiling plugin controls with unchanged case/job budgets. Exact final-source remote qualification is pending. Broader context-only routing outside manual Preparing, narrower invalidation dependencies and longer TTLs are not implemented; the wider source/transition proof obligations in ADR226 remain in force.
