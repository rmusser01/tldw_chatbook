# Console Send phase attribution

Status: original-body attribution and finite run-log/admission consolidation; overall performance acceptance remains open.

Attribution product source: `f70cacae3b225378573e9e0903422fe0e607553f` on `codex/console-send-preparation-plan`. The prior [shared preparation qualification](2026-10-06-console-shared-preparation-verification.md) retains matched observer-free samples and the failing original native regression.

## Method

`Tests/Performance/console_send_phase_spans.py` selects original Python code objects through local `sys.monitoring` events. It retains bounded scalar names, timestamps, thread/task identifiers, and source hashes. It does not replace product functions or retain arguments, results, receivers or frames. Global monitoring events are zero. Original application bodies, enabled tools, timers, private-profile guards and completion deadlines remain intact; only final provider network I/O is immediate.

Each sample executes three Sends sequentially, then verifies three replies and complete linked traces, zero pending dispatch checkpoints, source equality and diagnostic process/monitor retirement. Runs are sequential. The separate UAT chat's last confirmed state was native-idle with its agents paused; renewed requests received no fresh acknowledgement. Therefore these runs are attribution only, including that coordination limitation.

All timings below are inclusive elapsed wall time. Nested spans include children and waits and must not be added as predicted savings. These diagnostics do not qualify terminal paint or prove ordinary production-native cleanup.

## First two passes

| Run | Send 1 | Send 2 | Send 3 |
| --- | ---: | ---: | ---: |
| spans-1, action to adapter | 11.137 s | 10.754 s | 10.537 s |
| spans-2, action to adapter | 12.353 s | 11.419 s | 9.276 s |

The following original stage intervals do not overlap. Each cell lists Send 1 / 2 / 3, in seconds.

| Interval | spans-1 | spans-2 |
| --- | --- | --- |
| UI action to UI submit | 1.565 / 1.911 / 1.850 | 1.900 / 2.567 / 1.515 |
| UI submit to controller | .028 / .101 / .103 | .040 / .176 / .106 |
| Controller to provider resolution | 1.314 / 1.351 / 1.676 | 1.544 / 1.459 / 1.431 |
| Resolution to commit start | .661 / .738 / .776 | .508 / .693 / .733 |
| Durable commit stage | .319 / .150 / .147 | .411 / .119 / .118 |
| Commit success to trace reservation | 6.956 / 6.266 / 5.583 | 7.350 / 6.173 / 4.978 |
| Trace reservation to actual adapter | .295 / .238 / .402 | .600 / .233 / .394 |

The partitions equal action-to-adapter time within 0.00015 seconds of the probe's separately sampled origin. Saving is a small part of these totals; the evidence does not support blaming six seconds on the acceptance transaction or removing the save-before-dispatch barrier.

Postcommit function spans across these passes locate awaited prompt history at .726–1.206 seconds, a later hook-admission check at .434–.968 seconds, and shared provider composition at .828–.913 seconds. None is a savings estimate. Checkpoint completion to trace reservation remains 2.065–3.518 seconds.

Pass 2 divides that last interval:

| Interval | Send 1 | Send 2 | Send 3 |
| --- | ---: | ---: | ---: |
| Checkpoint end to bridge body | .796 | .552 | .508 |
| Bridge body to run_turn body | .505 | .377 | .417 |
| run_turn to _run_one | .355 | .260 | .307 |
| _run_one to model adapter body | 1.269 | 1.440 | .380 |
| Adapter body to trace factory | .560 | .573 | .447 |
| Trace factory to trace reservation | .033 | .004 | .005 |

These boundaries alone cannot separate filesystem admission, database work, thread scheduling or contention. Original source maps the middle intervals to nested agent admission, run creation/lifecycle/run-log writes, and independent model-loop admission. A narrower pass observes admission entry separately from its full lifetime.

Warm personal-context and budget lookups are poor optimization targets in these samples: all budget calls are at most .213 ms, warm personal service lookup .008 ms, warm profile-tool composition .033 ms, and profile snapshots .044 ms. Cold personal service construction is .140 seconds. Repeated static call sites alone do not establish a bottleneck.

## Narrowed third pass

The final attribution pass completed three Sends in 11.033 / 8.064 / 10.112 seconds. It records 526 scalar events: 254 complete start/return pairs and 18 original admission-generator yields. All 18 first-entry pairs are complete, the event cap did not overflow, and the local monitor retired. Source manifests match; three replies and linked complete traces remain, with zero checkpoints. Diagnostic Job retirement is positive with no forced cleanup, identity overflow or PID lookup race. General production-native cleanup remains explicitly unqualified.

The observer measures the original `RecoveryAdmissionGuard.execution` generator from entry to its first yield, separating admission from the subsequent work held inside that context. Six such entries precede each adapter call; their measured entry costs range from .184 to .568 seconds. This is now direct evidence of admission cost, rather than attributing a whole context-manager lifetime to its checks.

| Original operation | Send 1 | Send 2 | Send 3 |
| --- | ---: | ---: | ---: |
| Admission immediately before `_run_one` | .230 s | .195 s | .213 s |
| Admission immediately before model-loop `_consume` | .414 s | .353 s | .565 s |
| Run-log binding | 1.091 s | .907 s | .211 s |
| Run creation | .014 s | .015 s | .014 s |
| Both context lifecycle rows, inclusive callback | .034 s | .029 s | .043 s |

The first two admission rows account for almost all elapsed time in their corresponding previously unexplained gaps. The observed DB row writes are a much smaller consolidation candidate than admission/run-log preparation. Neither the six admission durations nor overlapping parent spans are advertised as achievable savings: required fresh checks remain required.

Source inspection identifies one already-supported finite-operation reuse opportunity inside run-log binding: its base-directory and run-directory containment checks independently resolve the same sensitive-path context. `is_within(..., context=...)` already supports sharing that context within one synchronous invocation while continuing to resolve/check each candidate. This is a bounded candidate for targeted controls; it does not justify a global path cache or skipping admission gates.

## Evidence custody and limits

Pass 1 has 262 scalar events/131 paired spans; pass 2 has 368/184. There are no malformed rows, unmatched starts/returns, timestamp reversals or observer overflows. Both 7,741-file before/after source manifests match. Selected production sources matched their retained hashes at audit, before the follow-up change below.

Both passes prove containment Job emptiness at parent exit, release of the Job identity, pipe/identity-monitor task retirement, unchanged containment source, no forced retirement, private-profile removal and local monitoring retirement. The receipt explicitly leaves `ordinary_app_native_cleanup_proven=false`; diagnostic containment is not a general proof of every production resource lifetime.

Pass 2 has one PID diagnostic lookup race and therefore fails a strict zero-race timing qualification criterion despite positive Job retirement. Pass 1 retains the inherited cosmetic source-receipt kind `original_composition_counts`; the actual plugin/output clearly records function spans. Do not alter either raw receipt to hide these limits.

Evidence is retained under `.superpowers/sdd/2026-10-06-console-shared-tool-preparation/native-pairs/candidate-spans-{1,2,3}.*` in the candidate worktree. The original subsecond Send target, actual 100 ms paint target, native regression and other-host qualification remain open.

## Next architecture work

The [retained ordinary commit plan](../superpowers/plans/2026-10-06-console-native-commit-ownership.md) now resolves the source-review findings about accepted cancellation, Close/dispose ownership and identical draft retyping. TASK-34563.8 owns implementation: an exact native outcome, existing ACCEPTED settlement, and targeted retention through the existing lifecycle. Eleven unchanged baseline controls pass; two real held-SQLite regressions then reproduced the original detached-cancellation failure before source edits. This prerequisite does not move durability or history later and does not claim save-speed gains. Atomic admission/promotion, screen-free capture and the runtime initial-hook-review bridge still precede enabling receipt. Warm personal/budget lookups remain excluded from speculative optimization.

ADR: [ADR-222](../../backlog/decisions/222-console-send-preparation-and-io-ownership.md).

## Finite run-log consolidation

TASK-34563.6 applies the existing per-invocation path-context API inside the existing synchronous storage scope. Ordinary stock binding now resolves one sensitive-path context for its base and run-directory checks instead of two. Both candidate paths still resolve/check independently. No context survives the bind. Legacy migration keeps its separate fresh observation. Sharing requires the original writer and unchanged function, body and defaults for containment, its sensitive predicate and its context resolver. Custom routes retain their original two-argument checks without eager preparation or retry. These definition-time anchors select call shape only; they grant no execution authority.

Original-source RED: six expected new failures exposed duplicate resolutions or missing supplied context; 56 checks passed, including all 51 original writer cases. Review of the first implementation found that changed containment defaults, a custom predicate or a custom resolver could lose their ordinary refusal semantics when supplied an explicit context. Three additional controls reproduced that defect before correction. The final 15 new cases cover those routes, both path denials, fresh subsequent binds, disabled/idempotent behavior and independent legacy migration.

Final integrated verification passed **89 targeted cases, zero failures/errors/skips**, covering the new tests, original writer behavior, sandbox/workspace isolation, actual scoped source admission and same-identity body drift. The four source/test hashes matched before/after. Scoped Ruff checks pass. The new test's final mixed newline was then normalized by Ruff with identical parsed AST; production bytes remain the tested bytes. Run-log/file-tool formatting passes. Two existing formatting findings in `sensitive_paths.py` are outside the changed anchors and remain unchanged from the committed base.

The combined run exceeded the bounded diagnostic process-history recorder (144 overflow events, one PID lookup race). Repeating the same unchanged scope in 78-case behavior and 11-case native batches passed again. The behavior batch had no overflow and two lookup races; the native batch still exceeded historical capacity (146 events) and had one race. These diagnostic limitations remain explicit; there is no strict complete PID-history claim. Every run independently proved its contained Job empty at parent exit, released the native identity, retired pipe/monitor tasks, kept the containment source current, required no forced retirement and removed the private profile. The native tests retain their own physical-owner assertions. None of this substitutes for general production-resource qualification.

A final observer-free private app check on the unchanged product completed three Sends in **11.092 / 8.207 / 9.989 seconds** to actual adapter entry. It produced three linked complete replies/traces and zero dispatch checkpoints. All 7,742 source-manifest entries matched; Job/identity/pump retirement was normal with zero recorder overflow/races, and the private profile was removed. This is one candidate diagnostic, not a new matched comparison; the inherited runner receipt kind `matched_observer_control` does not change that fact. UAT's last confirmed native-idle state remained unchanged, but no fresh acknowledgement arrived. The original subsecond target and actual 100 ms rendered feedback remain unmet/unqualified respectively. No end-to-end gain is claimed from the proven 2-to-1 context count reduction.

Evidence: `checks/runlog-path-red.*`, `checks/runlog-custom-red.*`, `checks/runlog-final*`, and `native-pairs/runlog-candidate-1.*` under the retained worktree evidence directory. The next performance boundary is repeated synchronous admission preparation; independent task/thread entry and live owner/path checks remain required.


## Finite activation preparation

TASK-34563.7 now prepares validated control records and the registry once during one stock `execution_scope` admission. The existing observation closes and verifies named identities, stamps and ancestors before an allowed decision reaches the guarded body. Owner approvals, exact paths, source selection and live native leases remain independently checked. Startup projection stays per owner because it can inspect live enrolled roots. No observation data survives into the guarded body or crosses admissions, tasks or threads. Standalone permission calls and custom readers keep their original route, argument shapes and short-circuit behavior. The shared route accepts only bounded stock owner tuples; larger/custom inputs take the ordinary route without eager preparation.

The original one/two-owner controls measured **3/2 and 5/4 control-record/registry reads** and failed the new requirement while successfully executing the original guarded bodies. The final controls prove **1/1** in both cases and fresh observations on later entries. Review also moved the final source/selector/lease check after observation completion and bounded new tuple qualification work. Mechanical extraction was checked separately: public validation/error behavior, owner projection and execution scope preserved their parsed structure; 65 original controls passed before the shared route was applied.

The final selected integrated scope has **129 passing cases**, including 27 new preparation cases, 86 original activation/observation cases and 16 native Console/MCP/RAG controls. Two original POSIX cases skip on Windows. Two original controls are explicitly excluded from the final repeat: a raw `chmod(0755)` privacy expectation already failed on unchanged Windows source, and a passing DACL mutation control left its synthetic ancestor inaccessible to the outer cleaner. Its one verified-scope cleanup retry also failed; the fixture residue and original evidence remain recorded, without changing ACLs or claiming removal.

Three initial MCP/RAG checks failed before their intended body because their fixtures embedded raw Windows paths in TOML double-quoted strings (`\U` caused an invalid hex escape). The identical checks failed on unchanged baseline source with an identical runner. Encoding just the two selected setup paths using the existing JSON-to-TOML convention let all three reach and pass their original permission/pause/cancel assertions. Production config guards and test deadlines remain unchanged. The separate unselected RAG `_LAZY_TRACKING` setup has the same existing path-encoding defect and remains outside this correction.

Every final batch proved normal contained Job emptiness, native identity release, pipe/monitor retirement and current containment source, with zero recorder overflow or lookup races. Their private profiles were removed. Product and new-test hashes match the pre-integration manifest. Ruff reports no new source diagnostics (one activation and thirteen bootstrap findings pre-exist); all three test files pass Ruff. Both product files and the new/RAG tests pass formatting; MCP's existing unrelated formatting diff is byte-for-byte identical to its HEAD formatter diff. These baseline limits and other-host qualification remain open, so this is scoped evidence, not a blanket clean-suite claim.

The final observer-free candidate sample completed three Sends in **10.304 / 9.710 / 8.964 seconds** to actual adapter entry, with three linked complete replies/traces and zero dispatch checkpoints. Its full source manifests match, native Job/identity/pump retirement is normal, diagnostic overflow/races are zero, and the private profile was removed. The coordinated window is closed. As before, UAT's last confirmed native-idle state had no fresh acknowledgement; this single candidate diagnostic does not establish a matched speed gain. UI action to submit still took approximately 1.60 / 2.30 / 1.99 seconds. The subsecond adapter target remains unmet and actual 100 ms terminal feedback remains unqualified. The next substantial work is the accepted Send preparation/early-feedback lifecycle, not more broad profiling or removal of durability.

Evidence: `checks/activation-*` and `native-pairs/activation-candidate-1.*` in the retained worktree evidence directory; unchanged baseline comparison is `checks/activation-sibling-baseline.*` in the baseline worktree. Plan: [finite activation preparation](../superpowers/plans/2026-10-06-console-finite-activation-preparation.md). Governing decision: [ADR-222](../../backlog/decisions/222-console-send-preparation-and-io-ownership.md).


## Ordinary save ownership before immediate feedback

TASK-34563.8 implements the retained native-outcome prerequisite in [the reviewed plan](../superpowers/plans/2026-10-06-console-native-commit-ownership.md), under [ADR-222](../../backlog/decisions/222-console-send-preparation-and-io-ownership.md). A single exact owner now survives queued issuance, repeated Stop, Close and dispose until the original save and any required ACCEPTED settlement actually finish. The finite worker captures its original persistence/database and callback; source changes cannot dispatch or publish into a successor. Required publication/history order remains, failed saves keep the draft, and cancelled/error paths preserve newer or identically retyped input. Normal completion releases the native owner before unrelated provider response work.

A real two-second SQLite lock exposed a 2.078-second event-loop stall in the initial accepted-terminal-write implementation. Moving that finite settlement through the named retained native boundary made its unchanged 500 ms heartbeat control pass. Other meaningful review regressions caught queued source replacement, early dispose cancellation, an error after actual terminal settlement, and trace-error returns that released or cleared the wrong owner/input. All are covered by final passing checks; they do not establish an ordinary Send latency improvement.

Final scoped verification: **45 passing cases**, including 32 new native/store/controller tests and 13 existing save, provider, history, draft and lifecycle controls. Three extra original controls failed. The exact unchanged pre-change commit `8fff47b36c` reproduces the whole-Send heartbeat failure (2.031 s vs 500 ms; candidate 718 ms) and the human continuation setup timeout. The unchanged Windows command-hook executor deliberately refuses launch, so neither human nor revoked command fixture can produce the continuation being awaited. A separate platform-independent checked-proposal test keeps the actual scheduler, live revocation, SQLite gate/rollback and cleanup; it passes. Original command tests and heartbeat thresholds are unchanged. These host/latency limitations remain open.

Every final candidate batch and the exact-head comparison proves normal contained Job emptiness, identity release, pipe/monitor retirement, current containment source, zero diagnostic overflow/races and private-profile removal. This is scoped diagnostic custody, not general production cleanup qualification. The final source hashes match. Ruff adds no diagnostics: 169 store and 61 controller findings pre-exist. Six files fully pass formatting; the other two retain exactly their baseline formatting transformations. Final read-only review has no actionable findings. No full suite was run.

The next performance work is shared screen-free configuration capture, followed by atomic received-intent promotion and a runtime-owned initial hook-review bridge. These preserve one admission owner while allowing Preparing feedback before slow work. The actual 100 ms feedback and subsecond adapter targets remain unqualified/unmet.

Evidence: `checks/commit-owner-*` under the retained candidate evidence directory; exact-head comparison `checks/commit-owner-heartbeat-head.*` in the attached `console-commit-baseline` worktree. No new observer-free timing was run solely for this ownership prerequisite.

### Original native trace-timeout interpretation

The original candidate native probe remains failed and incomplete. Its retained Send 2 stages show UI action at 503948.2643377 and controller completion at 503968.1553042: **19.8909665 seconds before UI sync and trace waiting**. The original 15-second deadline starts before the action; thus the budget was already at least 4.891 seconds overdue before the trace check. The check asserts immediately if any trace work remains after that deadline, explaining why this run failed there while the baseline reached the later send-duration assertion. This is a lower bound derived from the original recorded stages and unchanged test ordering, not evidence that candidate trace work subsequently retired. Preserve the candidate's incomplete result; no trace hang or successful cleanup is inferred from it. Preparation still exceeded the original budget.


## Shared screen-free configuration capture

TASK-34563.9 implements one Chat-owned configuration producer and thin mounted/runtime adapters under [ADR-222](../../backlog/decisions/222-console-send-preparation-and-io-ownership.md) and the [reviewed capture plan](../superpowers/plans/2026-10-07-console-configuration-capture.md). Common Library, review-admission, character, prompt, skill, MCP, capability and payload capture now has one owner. Existing RAG/tool-policy/presentation differences remain explicit adapter inputs. Default asynchronous runtime capture no longer imports the Console session adapter; supported custom synchronous providers retain call shape, affinity, errors and live lookup. Six relocated helper bodies retain identical ASTs and direct-call compatibility imports. Arbitrary rebinding of relocated aliases at their former module is outside the documented ADR-220 relocation contract.

The meaningful unchanged-source RED caught the actual unconditional UI import. Final verification passed **31 targeted cases**: eight new controls, 22 existing capture/context/wiring controls and one real mounted Send journey. These establish detached complete values, owning-session selection, custom callbacks, held-worker cancellation/source drift and collection of a detached view. One additional mounted Send-button test fails during fixture configuration load before Send; unchanged 8fff47b36c reproduces that same `raw_source_selection_changed` failure. Its assertions and deadlines remain unchanged. The passing neighboring mounted test uses its existing isolated-profile marker and ordinary-send policy with exchange capture disabled. Nonempty dictionary/world-info DB capture is not directly qualified by these tests.

All candidate and comparison runs retired their contained Job normally, released the native identity, drained pipe/monitor tasks, kept source current and removed private profiles, with zero diagnostic overflow/races and no forced cleanup. Product/test hashes remained frozen. Scoped lint introduces no diagnostics and existing formatter transformations are unchanged; new files pass both. Independent final review found no material capture-boundary issue. No full suite or new timing run was used for this prerequisite.

This removes 375 lines from the duplicate adapters into a 372-line shared producer, a net reduction of three production lines. It does not remove required I/O or prove lower elapsed latency. The controller still exceeds its pre-existing size ratchet and whole-app cold/module-census qualification remains open. App-owned initial hook review and atomic received-intent promotion are still needed before enabling immediate Preparing feedback. Actual100ms feedback and subsecond adapter targets remain unqualified/unmet.

Evidence: candidate `checks/capture-final-*` and `checks/capture-static-final.json`; unchanged comparison `checks/capture-mounted-baseline.*` in the retained `console-commit-baseline` worktree.


## Resident initial hook review

TASK-34563.10 implements the runtime review prerequisite under [ADR-222](../../backlog/decisions/222-console-send-preparation-and-io-ownership.md) and the [reviewed plan](../superpowers/plans/2026-10-07-console-initial-hook-review.md). The stock waiting-for-Send review now lives in the existing resident interrupt host. A disposable modal claims an exact review token and attachment generation; unmount releases presentation, while explicit Cancel, Settings, Stop, Close and disposal settle the resident request. A cancelled waiter cannot dispatch, and completing consent after waiter loss requires a new Send. Manual review and supported injected three-argument callbacks retain their routes.

Review actions use the existing live HookPermissions owner; Ready runs a fresh checked snapshot before settlement. One private executor Future retains each finite native action, with a submission-success seal and exact original-host completion. All-owned-Task cancellation, queued submission failure, cancellation before driver entry and controller replacement each produced meaningful failing controls before correction. Real native writes now remain owned through repeated cancellation, Close and disposal; completed consent is preserved without reviving a cancelled Send. The first all-task RED also cancelled an unrelated Canvas reader; the final fixture captures only the action task cohort and explicitly excludes that independent reader. This is scoped hook-operation qualification, not universal runtime Task-cancellation coverage.

The actual mounted integration caught a readiness check that rejected the Console's own covering modal. Existing-token actions now validate token, attachment and current modal; opening a new modal still requires a reconciled answerable view. Two new fixture assertions incorrectly used Textual's sticky widget `is_mounted`; they now verify removal from the stack, DOM detachment and stopped message processing. The unchanged eager Allow-all control passes and dispatches exactly once.

Final nonduplicated verification: **71 passing targeted cases**, including 13 new runtime cases, two new mounted cases, the original eager mounted control, original interrupt-host, callback, held-native-read and stale-consent controls, and manual revoke/disable. Two additional original worker-route controls still fail at their unchanged five-second pre-modal waits (test_console_hooks_review.py lines 196 and 249); unchanged c00e010424 reproduces the same failures. No deadline or original assertion was weakened. The last relocation batch passed 57/57. All batches retired their contained Jobs and native identities, drained pipe/monitor tasks and removed private profiles without force. Diagnostic identity lookup races occurred once each in compatibility, native-boundary RED and final relocation batches; overflow remained zero. These receipts do not prove general production cleanup.

The 17 hook-specific host methods and private operation record moved, with identical ASTs, to a stateless `console_hook_review_host.py` mixin. It creates no constructor, separate state, lock or registry; InterruptRoundHost remains the sole owner. The general host ends at 6520 lines, versus 6499 before and 6998 before extraction. Final source hashes match, independent implementation/packaging reviews are clean, and scoped lint adds no diagnostics. Existing controller/model lint findings remain 60/15; existing controller/runtime formatter transformations remain 15/38. Size gates remain unsatisfied: controller 30212 lines against 29299 and host 6520 against 6471; neither cap was raised. The task stays In Progress for these static and wider qualification limits.

No new timing run or full test sweep was performed. This prerequisite does not claim a latency gain: actual rendered 100 ms feedback and under-one-second adapter entry remain unqualified/unmet. Next is atomic received-intent reservation and promotion through the existing admission owner, with slow checked capture behind immediate Preparing feedback. Saved-turn failure still keeps the draft; WAL/NORMAL, required dispatch effects and awaited history are unchanged.

Evidence: `checks/hook-review-*` in the retained candidate evidence directory and `.superpowers/sdd/2026-10-07-console-initial-hook-review/` (source fingerprints, AST relocation proof, complete failed-run history and final audit).


### Follow-up review: task startup and remount ownership

Post-commit review found a configured task factory could create an eager Task, issue the original native consent write, then raise before returning the Task. The held-original-body regression failed with no retained retirement while the write remained live. Constructing this private driver directly with the lazy stdlib Task preserves its handle before native issuance; public callbacks and the private executor Future remain unchanged.

The first follow-up integrated run passed 16 cases but exposed one remount Cancel failure. Textual posts ScreenResume before the popped modal finishes Unmount. The projector reused that old modal's token for its replacement, and late old Unmount then invalidated the replacement. A deterministic actual-projector regression reproduced identical tokens. New-modal creation now requests a fresh token; same-modal refresh retains its existing token. Late cleanup therefore cannot cancel or release a successor presentation.

The final affected batch passed **18/18** (15 runtime, two mounted and the original eager Allow-all control); both new regressions fail meaningfully before their respective corrections. This adds two distinct passing controls to the previous 71, rather than a fresh rerun of all 73. Existing unrelated baseline failures and static/module-size limits remain as recorded above. Independent source review confirmed both causes and the minimal fixes; final self-review and scoped Ruff pass, with the runtime's 38 existing formatter transformations unchanged. No deadline or original mounted assertion changed.

All four follow-up runs, including both REDs and the 16-pass/one-failure intermediate batch, retired their contained Jobs, native identities, pipes/monitors and private profiles normally. Diagnostic overflow and lookup races were zero in these four runs; this does not revise earlier recorded races or establish general production cleanup. Source fingerprints match. Evidence: checks/hook-review-eager-factory-{red,green}.*, checks/hook-review-remount-token-red.*, checks/hook-review-followup-final.*, and followup-{static,audit}.json in the retained evidence directories. No timing or full sweep was run. Immediate rendered feedback and subsecond adapter entry remain unqualified/unmet; the received-intent admission work is next under ADR-222.


## Guard-level preparation continuation, 2026-10-07

At candidate `7d150a5cf5`, the existing original-function observer completed
three actual file-backed saved turns in 10.040486 / 9.150090 / 8.538632 seconds
from Send action to immediate adapter entry. All three replies/traces completed;
source hashes were unchanged, the monitoring slot retired, and overflow and
unfinished admission entries were zero. The contained run retired normally in
68.703 seconds and removed its private profile. This 60-target observer is a
diagnostic, not a matched performance acceptance run.

Its six separate recovery-guard pre-yield entries total 1.103263 / 1.056454 /
1.089280 seconds per Send. Configuration capture spans are 1.0033 / 1.1197 /
1.3124 seconds; tool-provider composition is .9007 / .8057 / .9851 seconds.
Run-log binding is .9040 / .7897 / .1196 seconds. Inclusive spans overlap and
must not be summed into hypothetical savings. Request serialization remains
small; the measured remaining issue is repeated application preparation.

TASK-34563.7 now also covers sharing control-record/registry data across the
related sources within one guard's finite admission. Its original-body RED
(`activation-guard-preparation-red`) performs two independent nested admissions
on the exact retained original lease. Both guarded bodies and lease checks
succeed, but each admission rereads control records and registry twice and
opens 216 native handles. The intended 1/1 observation assertion fails on
unchanged product source. The 6.969-second contained run retired normally,
with zero forced retirement, identity overflow or lookup races. This is count
evidence; it does not establish the whole-Send target.

Raw receipts remain under
`.superpowers/sdd/2026-10-06-console-shared-tool-preparation/native-pairs/agent-preparation-spans-current-1.*`
and `checks/activation-guard-preparation-red.*`.

### Guard-level integrated verification

The guard now lazily shares one original control observation per bootstrap root
inside its existing synchronous admission walk. Source leases remain separate;
every source and owner projection still runs in order. A normal helper completes
all observations and final source/selector/lease checks before execution state
is published. No observation or prepared permission survives the guarded yield.
Custom resolve/admit/finalize callbacks retain the ordinary route.

Review caught a mixed-route omission: an empty-owner source admitted through the
ordinary fallback was absent from the final witness list. The new regression
retired that exact second lease as the original shared observation completed;
it reproduced entry into the guarded body before correction. The corrected
route records and rechecks successful fallback leases with the shared sources.
Two independent final source reviews found no further actionable issues.

Final integration (`activation-guard-final`) passed **65 targeted cases** in
201.61 seconds (207.172 seconds including the diagnostic driver). This includes
17 new guard controls, original finite activation/identity controls, six stock
nested-agent controls, and ten original provider/MCP/RAG native controls. Real
control-record/registry reads fell from 2/2 to 1/1 per entry; native opens fell
from 216 to 145 in both fresh nested entries. Two distinct enrolled files also
use one observation while retaining their separate leases.

The existing nested-agent observer now observes the extracted original
permission projection under the actual guard ancestry. Its fault injection
occurs on a positive projection return before completed admission, not after an
allowed guard yield. Original provider/publication, source-count and physical
cleanup assertions remain the outcome oracle; the new completion controls
separately mutate sources and leases at observation retirement.

All 65 cases pass on the final source. Ruff, formatting and whitespace checks
pass for all four changed Python files. The contained Job was empty at normal
parent exit, its native identity and monitor/pipes retired, and the private
profile was removed without force. The bounded historical PID recorder filled
its 16-entry capacity (862 overflow events and one lookup race); this is an
explicit diagnostic-history limitation, not a complete process-history claim.
The independent contained-Job emptiness proof and individual native ownership
assertions passed. No deadlines or recorder capacity were widened and no full
suite was run. Whole-Send timing remains a separate acceptance result below.

### Guard continuation whole-Send result

The unchanged 60-target diagnostic (`activation-guard-spans-final-1`) completed
three actual saved Sends in **14.716631 / 8.966272 / 12.691052 seconds** to
adapter entry. The six finite guard entries total .718731 / .544679 / .709525
seconds, versus 1.103263 / 1.056454 / 1.089280 before this continuation. Counts
and guard attribution improved; the overall target remains unmet and these raw
samples do not demonstrate an overall latency gain.

Remaining spans include configuration capture 1.531 / 1.245 / 1.924 seconds,
provider composition 1.504 / .802 / 1.338 seconds, and run-log binding 1.686 /
1.001 / .253 seconds. Initial controller hook admission costs .728 / .532 /
.908 seconds, hook preparation .719 / .670 / .957 seconds, and the postcommit
fresh hook check 1.550 / .522 / .772 seconds. These inclusive measurements
overlap; they are not an additive savings estimate. Send 3 used the older UI
preparation route before custody, requiring an exact fallback-boundary check.

All replies and linked traces completed, with zero remaining dispatch
checkpoints. Send heartbeat maxima were 420 / 148 / 374 ms and typing max was
1.174 seconds; universal immediate feedback remains unqualified. The 96.235-
second diagnostic retained identical source manifests, current function
bindings, zero event overflow/unfinished admission entries, and retired its
monitor. Its contained Job, native identity and pipes/monitor retired normally;
private profile was removed, with zero identity overflow or lookup races.
No full sweep was run. Work continues on the remaining measured preparation
and the observed early-receipt fallback under ADR-222.

### Narrow hook and receipt attribution

The optional detail mode of the same observer adds six original function
boundaries, bounded raw-check aggregates and source-line receipts. Its first
run (`hook-receipt-detail-1`) completed 13.382601 / 8.823169 / 7.930431-second
Sends with three linked complete traces and no checkpoints. All three calls
returned a received-intent ID at wiring.py line 561. The prior Send-3 legacy
fallback did not recur; a stale composer display flag remains a hypothesis,
so neither the production gate nor the fixture was changed.

Five HookPermissions._current entries occur before each adapter call. Ordinary
warm entries observe 254 original raw._check returns, taking about .27 seconds
in a .52-.60-second inclusive operation. One warm entry records376 checks.
Default hook-path resolution costs about .04 seconds in ordinary warm entries;
removing just that one resolver cannot account for the whole delay. Run-log
setup also calls the original user-directory resolver repeatedly, about
.036-.056 seconds per ordinary warm call. The existing sensitive-input bundle
runs original path accessors inside one operation but still recomputes their
shared directory; its original native test intentionally counts20 calls on a
cold/rebuilt input bundle. This identifies actual repeated preparation beyond
merely sharing a native scope.

Source manifests/bindings and the monitor remained current. Both event and
aggregate capacity overflow, unmatched starts and unfinished entries were zero.
The64-frame ancestry bound was reached190 times: raw counts are scoped observed
counts, not a complete native-work census. Raw elapsed is inclusive and must
not be added to parent spans. The87.891-second contained run retired normally,
removed its private profile and had zero process-history overflow/lookup races.
A bounded caller-location refinement will distinguish repeated discovery checks
from actual native effect validation before selecting the next change.

### Original-caller refinement

The 67-target opt-in diagnostic (`hook-caller-detail-1`) completed all three
saved Sends in **10.554331 / 9.535932 / 8.773512 seconds** to adapter entry,
with three replies, linked complete traces and no remaining checkpoints.
All received-intent calls returned an accepted ID. Both observed stock
sensitive-input bundles were selected, confirming that the repeated directory
work occurs inside the existing shared operation rather than an unsupported
fallback.

Of 254 checks in an ordinary warm HookPermissions entry, 223 came from
`raw._runtime_operation`: 71 in the config route and 152 in the hook route.
The remaining 31 came from the existing scope/effect boundaries. These are
immediate-caller counts; they do not distinguish open from close callers of
runtime discovery. The targeted retirement regression supplies that ancestry
count before implementation. The first cold entry recorded 356 checks; one
later entry recorded 376. No new whole-Send observer is needed for this slice.

The 74.735-second run retained identical source manifests and current original
bindings. The monitor retired with no event/caller overflow, unmatched starts
or unfinished entries. The existing 64-frame ancestry limit was reached 190
times, so aggregate counts remain scoped observations. The contained Job and
native identity retired without force, pipes/monitor retired and the private
profile was removed; process-history overflow and lookup races were zero.
Send heartbeat maxima were 949 / 159 / 183 ms and typing max was 769 ms.
Neither the one-second Send target nor universal immediate feedback is met.

### Owned handle retirement and shared-directory regressions

The original owned-close regression (`raw-retirement-count-red`) closed 21
tracked descriptors, including 15 during the original parent walk. Each close
reentered `raw._check` (21 total). Before its intended zero-check assertion,
the test verified actual EBADF, descriptor removal and operation retirement.
The 5.703-second contained run retired normally. An earlier fixture-only
attempt expected the base directory instead of its configured user subfolder;
that setup failure is retained as `raw-retirement-red`, not counted as RED.

The next directory-preparation regression (`sensitive-directory-red`) ran the
existing actual-source sensitive-input fixture before selection changes. All
13 expected database paths and the file/trust/container deny lists matched;
the new count assertion failed at **20 directory resolutions versus 4**.
This baseline, including the owned-close candidate, recorded 2,603 native opens
and 81 raw checks. The 8.750-second contained run retired normally. The target
retains initial/final original directory validation plus both independent RAG
readers, and shares only directory data within the existing finite operation.

### Owned retirement integrated result

Task 34563.17 adds a private exact-current-operation/actor/tracked-integer-fd
lookup, then delegates close and uncertain-outcome retention to the original
close owner. Voice and visual precedence and ordinary untracked fallback stay
unchanged. Source/parent/permission validation remains on every new effect.

All nine new ownership controls and 11 existing source/native controls passed
in `raw-retirement-integration`. The real count changed from 21 closes / 21
checks to **21 closes / 0 checks**, including 15 parent-walk closes. Tests prove
EBADF before descriptor-number reuse, ownership-map retirement, cleanup after
source revocation, refusal of subsequent effects, closed-admission drain,
foreign/inactive actor refusal, copied-token nonparticipation, ordinary
untracked close and retained uncertainty without retry.

That batch also retained eight failures plus one teardown error in older
fixtures, so it is not reported as an all-green batch. Four raw uncertainty
children failed to report readiness (then the fixture's failure-handler wait
timed out), two alias cases required unavailable Windows symlink privilege or
assignment to read-only WindowsOS.supports_dir_fd, and two loose-voice cases
failed before their target operation because their HOME-only setup inherited
USERPROFILE and reused the enclosing profile. A separate exact-HEAD comparison
of five representative cases reproduced the raw-before, both alias and both
voice failures. The other three raw variants remain unqualified, not claimed
as individually baseline-reproduced. Candidate source bytes were restored
exactly in finally. No fixture deadlines or product guards were relaxed.

The existing native Windows junction-refusal and pinned-directory-publication
controls both passed separately (`raw-retirement-windows`, 3.828 seconds).
Four passing original config uncertainty cases cover close-before/after at
lock/final-parent boundaries. Voice-specific failed-fixture evidence remains a
limitation; unrelated fixture repair is outside this preparation change.
The 98.641-second integration Job retired normally without force; its bounded
PID history recorded 67 overflow events. The separate baseline and Windows
runs also retired normally with their private profiles removed.

The final unchanged 67-target Send probe (`raw-retirement-spans-1`) returned
**9.197429 / 5.789172 / 4.819775 seconds** versus the immediately preceding
10.554331 / 9.535932 / 8.773512-second samples. A typical hook read now makes
**150** raw checks instead of **254**; ordinary late warm entries take about
.30-.32 seconds. All replies and three linked traces completed; no dispatch
checkpoints remained. This single sequential comparison is an observed
improvement, not a distribution or the subsecond acceptance result.

All source manifests and original bindings remained current. No event/caller
capacity overflow, unfinished entry or unmatched start was observed; the
bounded ancestry walk missed 122 times. The 58.843-second contained run retired
its Job, native identity, pipes and monitor without force and removed its
private profile, with no PID-history overflow or lookup race. Send heartbeat
maxima were 258 / 131 / 134 ms and typing max was 571 ms; responsiveness remains
above the target. Scoped review found no new lint issues; raw_participants.py
has unchanged whole-file formatting debt and private_paths.py retains its
existing E721 at line 132. The new helper and test file pass formatting.
Existing ADR-222 and ADR-126 govern the change. Work continues under task
34563.18 on the measured repeated directory inputs; no full suite was run.

### Shared directory preparation integration

Task 34563.18 centralizes the 13 sensitive database selectors in config's
private `_database_path` helper while preserving every public signature and
annotation. Default paths may use the directory already verified by the
current sensitive-input operation; explicit custom paths retain their original
validation, short-circuit and scheduled-task expansion rules. Stock file,
skill-trust and container projections receive the same explicit directory.
Both independent RAG readers and the original initial/final directory reads
remain. No new owner, registry, cache or native lifetime is introduced.

Definition-time function and defaults checks qualify the prepared route.
Custom functions or changed default values use their ordinary public readers;
source/reader changes during a stock build refuse its publication. Observed
environment/CWD changes use ordinary readers without changing the additive
error contract. Their sticky invalidation is checked again at publication,
including after a changed value returns to its original value. Review caught
that this flag must be checked after final admission as well as when computing
eligibility; two real-body mutation controls verify that correction.

The final integrated run (`sensitive-directory-integration`) passed **46/46**
targeted cases in 236.95 seconds (240.562 seconds including the driver).
This includes original custom/failure/source/body/guarded-body/tuple controls,
nine new parity/default/helper/late-invalidation cases, run-log path preparation
and the original ordinary scoped run-log admission. Cold directory calls fall
from **20 to 4**; the warm hit still makes one directory call and two raw
checks. All 13 default and explicitly overridden database selections, reset
behavior, scheduled validation, file/trust coverage and database sidecar denial
match the ordinary readers. Source changes and failed preparation never
publish the raw-input memo. Required physical resource retirement passed.

The contained Job was empty at normal parent exit; its native identity,
pipes and monitor retired and the private profile was removed without force.
The bounded historical PID recorder overflowed 1,262 times (zero lookup races),
so this is not a complete process-history census. Independent Job emptiness
and the tests' native retirement assertions passed. No deadlines or diagnostic
capacities were widened and no full sweep was run.

Both product diffs received root and independent API review without remaining
findings. AST checks and all 13 public signature comparisons pass. Changed
functions and the test file pass formatting. Config's 56 existing Ruff findings
are unchanged; sensitive_paths.py has no lint findings and retains two existing
whole-file wrapping differences. The helper is also included in the existing
run-log source qualification records. ADR-222 and ADR-126 continue to apply.

The separate two-case count run (`sensitive-directory-counts`, 14.078 seconds
including the driver) passed with successful original-body stdout retained.
Cold native opens fall from **2,603 to 1,003**, raw checks from **81 to 17**,
and user-directory calls from **20 to 4**. The warm hit remains **356 opens,
two raw checks and one user-directory call**. Both final operation censuses
were zero; the existing startup owner lasts until process exit. The contained
Job retired normally, its private profile was removed, and PID history had
no overflow or lookup race.

The original three-Send probe (`sensitive-directory-spans-1`) measured
**9.002798 / 8.157567 / 7.945963 seconds** to the adapter. Against task 17's
9.197429 / 5.789172 / 4.819775-second sample, this establishes **no overall
latency improvement**. All three saved turns completed, with three replies,
three linked traces and zero dispatch checkpoints. Both sensitive-input
bundles selected the qualified prepared route; all received intents were
accepted. Config capture still took 1.335 / 1.412 / 1.283 seconds, and provider
composition 1.074 / .760 / .904 seconds. Typical hook reads still performed
150 raw checks. Local operation-count savings are established independently
of the variable whole-Send samples; the subsecond target remains unmet.

Send heartbeat maxima were **364 / 128 / 192 ms**, typing max **587 ms**,
and idle max **2.386 seconds**. The source stayed unchanged throughout the
68.640-second driver run. No span or caller capacity overflow, unmatched
start or unfinished entry occurred; the bounded ancestor walk recorded 122
misses, so scoped call ancestry is not a complete global census. The Job,
native identity, pipes and monitor retired without force, the private profile
was removed, and PID history recorded no overflow or lookup race.

Task 18's four acceptance criteria are verified, but its status remains
In Progress because inherited whole-file static debt and earlier platform
qualification gaps remain explicitly unresolved. This is a local checkpoint,
not completion of the Console responsiveness work. Existing ADR-222 and
ADR-126 remain the governing decisions.
