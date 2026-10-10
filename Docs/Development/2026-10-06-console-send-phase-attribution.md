# Console Send phase attribution

Deferred and unimplemented options: [optimization review list](console-optimization-review-list.md).

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

ADR: [ADR-225](../../backlog/decisions/225-console-send-preparation-and-io-ownership.md).

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

Evidence: `checks/activation-*` and `native-pairs/activation-candidate-1.*` in the retained worktree evidence directory; unchanged baseline comparison is `checks/activation-sibling-baseline.*` in the baseline worktree. Plan: [finite activation preparation](../superpowers/plans/2026-10-06-console-finite-activation-preparation.md). Governing decision: [ADR-225](../../backlog/decisions/225-console-send-preparation-and-io-ownership.md).


## Ordinary save ownership before immediate feedback

TASK-34563.8 implements the retained native-outcome prerequisite in [the reviewed plan](../superpowers/plans/2026-10-06-console-native-commit-ownership.md), under [ADR-225](../../backlog/decisions/225-console-send-preparation-and-io-ownership.md). A single exact owner now survives queued issuance, repeated Stop, Close and dispose until the original save and any required ACCEPTED settlement actually finish. The finite worker captures its original persistence/database and callback; source changes cannot dispatch or publish into a successor. Required publication/history order remains, failed saves keep the draft, and cancelled/error paths preserve newer or identically retyped input. Normal completion releases the native owner before unrelated provider response work.

A real two-second SQLite lock exposed a 2.078-second event-loop stall in the initial accepted-terminal-write implementation. Moving that finite settlement through the named retained native boundary made its unchanged 500 ms heartbeat control pass. Other meaningful review regressions caught queued source replacement, early dispose cancellation, an error after actual terminal settlement, and trace-error returns that released or cleared the wrong owner/input. All are covered by final passing checks; they do not establish an ordinary Send latency improvement.

Final scoped verification: **45 passing cases**, including 32 new native/store/controller tests and 13 existing save, provider, history, draft and lifecycle controls. Three extra original controls failed. The exact unchanged pre-change commit `8fff47b36c` reproduces the whole-Send heartbeat failure (2.031 s vs 500 ms; candidate 718 ms) and the human continuation setup timeout. The unchanged Windows command-hook executor deliberately refuses launch, so neither human nor revoked command fixture can produce the continuation being awaited. A separate platform-independent checked-proposal test keeps the actual scheduler, live revocation, SQLite gate/rollback and cleanup; it passes. Original command tests and heartbeat thresholds are unchanged. These host/latency limitations remain open.

Every final candidate batch and the exact-head comparison proves normal contained Job emptiness, identity release, pipe/monitor retirement, current containment source, zero diagnostic overflow/races and private-profile removal. This is scoped diagnostic custody, not general production cleanup qualification. The final source hashes match. Ruff adds no diagnostics: 169 store and 61 controller findings pre-exist. Six files fully pass formatting; the other two retain exactly their baseline formatting transformations. Final read-only review has no actionable findings. No full suite was run.

The next performance work is shared screen-free configuration capture, followed by atomic received-intent promotion and a runtime-owned initial hook-review bridge. These preserve one admission owner while allowing Preparing feedback before slow work. The actual 100 ms feedback and subsecond adapter targets remain unqualified/unmet.

Evidence: `checks/commit-owner-*` under the retained candidate evidence directory; exact-head comparison `checks/commit-owner-heartbeat-head.*` in the attached `console-commit-baseline` worktree. No new observer-free timing was run solely for this ownership prerequisite.

### Original native trace-timeout interpretation

The original candidate native probe remains failed and incomplete. Its retained Send 2 stages show UI action at 503948.2643377 and controller completion at 503968.1553042: **19.8909665 seconds before UI sync and trace waiting**. The original 15-second deadline starts before the action; thus the budget was already at least 4.891 seconds overdue before the trace check. The check asserts immediately if any trace work remains after that deadline, explaining why this run failed there while the baseline reached the later send-duration assertion. This is a lower bound derived from the original recorded stages and unchanged test ordering, not evidence that candidate trace work subsequently retired. Preserve the candidate's incomplete result; no trace hang or successful cleanup is inferred from it. Preparation still exceeded the original budget.


## Shared screen-free configuration capture

TASK-34563.9 implements one Chat-owned configuration producer and thin mounted/runtime adapters under [ADR-225](../../backlog/decisions/225-console-send-preparation-and-io-ownership.md) and the [reviewed capture plan](../superpowers/plans/2026-10-07-console-configuration-capture.md). Common Library, review-admission, character, prompt, skill, MCP, capability and payload capture now has one owner. Existing RAG/tool-policy/presentation differences remain explicit adapter inputs. Default asynchronous runtime capture no longer imports the Console session adapter; supported custom synchronous providers retain call shape, affinity, errors and live lookup. Six relocated helper bodies retain identical ASTs and direct-call compatibility imports. Arbitrary rebinding of relocated aliases at their former module is outside the documented ADR-220 relocation contract.

The meaningful unchanged-source RED caught the actual unconditional UI import. Final verification passed **31 targeted cases**: eight new controls, 22 existing capture/context/wiring controls and one real mounted Send journey. These establish detached complete values, owning-session selection, custom callbacks, held-worker cancellation/source drift and collection of a detached view. One additional mounted Send-button test fails during fixture configuration load before Send; unchanged 8fff47b36c reproduces that same `raw_source_selection_changed` failure. Its assertions and deadlines remain unchanged. The passing neighboring mounted test uses its existing isolated-profile marker and ordinary-send policy with exchange capture disabled. Nonempty dictionary/world-info DB capture is not directly qualified by these tests.

All candidate and comparison runs retired their contained Job normally, released the native identity, drained pipe/monitor tasks, kept source current and removed private profiles, with zero diagnostic overflow/races and no forced cleanup. Product/test hashes remained frozen. Scoped lint introduces no diagnostics and existing formatter transformations are unchanged; new files pass both. Independent final review found no material capture-boundary issue. No full suite or new timing run was used for this prerequisite.

This removes 375 lines from the duplicate adapters into a 372-line shared producer, a net reduction of three production lines. It does not remove required I/O or prove lower elapsed latency. The controller still exceeds its pre-existing size ratchet and whole-app cold/module-census qualification remains open. App-owned initial hook review and atomic received-intent promotion are still needed before enabling immediate Preparing feedback. Actual100ms feedback and subsecond adapter targets remain unqualified/unmet.

Evidence: candidate `checks/capture-final-*` and `checks/capture-static-final.json`; unchanged comparison `checks/capture-mounted-baseline.*` in the retained `console-commit-baseline` worktree.


## Resident initial hook review

TASK-34563.10 implements the runtime review prerequisite under [ADR-225](../../backlog/decisions/225-console-send-preparation-and-io-ownership.md) and the [reviewed plan](../superpowers/plans/2026-10-07-console-initial-hook-review.md). The stock waiting-for-Send review now lives in the existing resident interrupt host. A disposable modal claims an exact review token and attachment generation; unmount releases presentation, while explicit Cancel, Settings, Stop, Close and disposal settle the resident request. A cancelled waiter cannot dispatch, and completing consent after waiter loss requires a new Send. Manual review and supported injected three-argument callbacks retain their routes.

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

All four follow-up runs, including both REDs and the 16-pass/one-failure intermediate batch, retired their contained Jobs, native identities, pipes/monitors and private profiles normally. Diagnostic overflow and lookup races were zero in these four runs; this does not revise earlier recorded races or establish general production cleanup. Source fingerprints match. Evidence: checks/hook-review-eager-factory-{red,green}.*, checks/hook-review-remount-token-red.*, checks/hook-review-followup-final.*, and followup-{static,audit}.json in the retained evidence directories. No timing or full sweep was run. Immediate rendered feedback and subsecond adapter entry remain unqualified/unmet; the received-intent admission work is next under ADR-225.


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
and the observed early-receipt fallback under ADR-225.

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
Existing ADR-225 and ADR-126 govern the change. Work continues under task
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
run-log source qualification records. ADR-225 and ADR-126 continue to apply.

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
not completion of the Console responsiveness work. Existing ADR-225 and
ADR-126 remain the governing decisions.


### Related-member preparation: verification and unresolved timing

Task 34563.20 uses the existing related-path admission for one installed MCP
or hook operation. Primary-only borrowed authority, separate recovered canonical
custody, every member check and all effect/reconciliation ordering remain intact.
Custom or changed acquisition callbacks retain the original positional route;
a late callback change refuses only after the issued lease has an owner for
cleanup. Independent source review found no correctness blocker.

The original three count controls failed only their final acquisition-count
assertions, after real output, complete member order and physical retirement
passed: permission 3, catalog 2 and hook 4 acquisitions. Candidate counts are
one each. The first integrated run had 48 passes and one unsupported fixture
expectation: a malformed changed keyword default did not reject in its unbound
namespace. The corrected control directly verifies that three ordinary calls
consume the changed default, without replacing it through batch keywords. All
ten new controls then passed. The 39 existing integration controls passed on
unchanged product, and the restored canonical permission update passed separately.
Those checks cover consent reconciliation, source drift, enrollment, pause,
provider preparation/cancellation, and uncertain native retirement.

The source-frozen 78-target `related-source-spans-1` run completed three saved
replies with complete linked traces and zero checkpoints, but adapter latency
was **12.199708 / 8.840663 / 9.037641 seconds**, versus the previous
**9.943307 / 5.684903 / 5.117334 seconds**. This is not a speed success.
Several unchanged bodies also slowed, while hook raw entry became cheaper;
the samples do not isolate causality. Send heartbeat maxima were .505/.295/.209
seconds, typing .566 seconds and idle 3.130 seconds. Both latency goals remain
unmet. The driver completed in 90.937 seconds (pytest 84.54), source hashes and
bindings remained current, and monitoring retired with no event/detail overflow
or unfinished entries. The bounded ancestor walk recorded 122 limit misses.

A sequential original-source/candidate comparison therefore added passive
original Windows `_Native.open_handle` entry counts on the measured read thread.
These include the surrounding configuration work for a hook read, but are not
a process-wide syscall census. Both exact product files were preserved by hash,
staged to a627620e43 for the original run, and restored byte-for-byte only after
native retirement. Ordinary source counts were permission **1581 -> 2199**,
catalog **1348 -> 1262**, hook **1599 -> 1341**. With actual enrolled profiles
and confirmed selector/primary evidence, counts were permission **1785 -> 1541**,
catalog **1508 -> 1386**, hook **2289 -> 2071**. The existing one-second settling
policy remained unchanged and random hook temporaries were never pre-admitted.
Original six cases reached only the final count assertion; all thirteen candidate
count/compatibility cases passed. With the 39 unchanged controls and restored
canonical case, the verified union now contains 53 distinct cases.

A further exact-source permission-only pair records actual evidence state and
reuse outcomes. Both reads entered with confirmed unbound selector evidence and
returned the original reused lease. Original native entries were **1581**, with
**168** inside three acquisitions (56 each) and **1413** outside; candidate was
**1409**, with **56** inside its one acquisition and **1353** outside. Original
failed only the final one-acquisition assertion; candidate passed. Every earlier
output/member/retirement assertion passed, detail overflow was zero, and both
runners retired normally with no PID overflow or lookup race. The latest test
source hash is c9fac7c395d2feec19c95409545c2299d05daaf1d4a02848acdaa8087db7d4ed.

The earlier 2199-entry permission observation did not reproduce and lacked reuse
metadata, so its cause remains unproven. For bound profiles, a member without
confirmed evidence can still send a grouped acquisition through full derivation;
for an unbound profile, confirmed selector evidence alone permits reuse. No
settling policy changed. These counts justify retaining the small related-member
consolidation, but establish neither uniformly lower cold-read cost nor faster
whole-Send latency. Most remaining native entries are outside the directly counted member
acquisition. Source tracing finds additional canonical acquisitions beneath
source selection before the raw operation is active; the next observer separates
those from actual file access and nested validation. No additional cache
or weakened permission boundary follows from these measurements.

All listed runners reached an empty contained Job at normal parent exit, released
native identity and retired pipes/monitoring without force; private profiles were
removed. The first integration, whole-Send and final native comparison each
recorded one PID lookup race, with zero history overflow; the final ten controls,
restored canonical case and original native comparison recorded neither. These
are bounded diagnostic cleanup receipts, not universal production-cleanup proof.
Scoped syntax/new-test formatting and lint pass; raw-participant lint stays clean
and storage-admission retains the same eight pre-existing diagnostics. Existing
whole-file format debt and prior platform gaps remain qualified. Task 34563.20
and the larger responsiveness work remain In Progress under ADR-225/ADR-126.


### Original source-read ancestry after related-member batching

The bounded observer now distinguishes every original acquisition from only the
direct member batch, and attributes native opens to their exact raw-check or
acquisition caller. The first run reached all behavior and retirement checks but
failed the permission diagnostic: redundant caller dimensions exhausted 64 buckets
(35 opens aggregated), while walking past the measured read into pytest incorrectly
marked every ancestry as truncated. The observer was corrected to stop at that
exact measured-read frame and to merge redundant helper dimensions when the owning
check/acquisition is already identified. Neither limit nor product code changed.

The corrected `related-read-ancestry-2` passed both original installed permission
and hook reads in 22.625 driver seconds (14.51 pytest). Counts/exits reconcile,
there were zero bucket/active/detail overflows and zero depth misses, and original
function bindings remained installed. Permission: 1409 native opens, seven actual
acquisitions (six canonical observations plus one direct member batch). The six
canonical acquisitions account for 336 opens. Seven full raw checks each account
for 73 opens; two initial nested checks and two final nested checks each account
for 146. Between each initial/final nested MCP pair, the stock source path is read
from the existing operation and compared lexically; there is no intermediate
native effect or selector callback. The two initial checks took .350909 seconds
under this observer; the retained final pair took .353530 seconds. These are
instrumented overlapping timings, not expected whole-Send savings.

Hook: 1341 native opens and three actual acquisitions, including surrounding config
admission. Runtime-operation discovery invokes 50 hook checks (400 native opens)
and 27 config checks (162 opens). These checks repeat parent identity validation;
the hook source body itself remains small. Source review is evaluating whether
read-only parent preparation can share a finite proof while retaining actual leaf
and publication gates. No parent-walk optimization or lock durability change has
been implemented. Six canonical acquisitions can separately share one finite
pending source owner only if all fresh generation observations and refusal points
remain; that larger change is also not implemented from this diagnostic.

Both runs reached an empty contained Job at normal parent exit, released native
identity and pipes, and removed their private profiles without forced termination
or PID history overflow/race. Observer source after correction:
a7c9c963f946f00fc5dbc339dc169d7578c31efa48ec05f2bee63d710e62723d.
Task20 product hashes remained unchanged. The next smallest measured contraction
is duplicate nested installed MCP validation, under existing ADR-225/ADR-126.


### Nested installed MCP validation contraction (task 34563.21)

The stock same-source installed MCP nesting now inspects issued actor, source and
participant metadata, then retains one original full source/path check immediately
before its body. Explicit selected-read values, non-MCP routes and custom sources
retain the original branch. No permission result, file witness or lifetime is
cached. Existing ADR-225 and ADR-126 govern this contraction.

The sequential original count control reached all payload/member/retirement
assertions and failed only the final expected nesting count: five scope checks
and seven full checks. Candidate scope checks are three and full checks five;
actual native opens fell from 1409 to 1263, with seven acquisitions unchanged.
The bounded ancestry observer recorded zero overflow and depth misses. These are
operation counts on a seeded permission read, not a whole-Send speed claim.

The integrated run passed 51 cases and failed one new test fixture: it expected
__fspath__ conversion, while the unchanged lexical_path contract uses str(value).
The fixture now observes __str__ ordering and explicitly rejects unexpected
__fspath__ use; production was unchanged. The four final controls passed both
explicit-path cases, the count test and restored canonical permission update.
Together the runs verify 53 distinct cases: late source/selector/custody/participant
drift, foreign actors and unissued tokens, custom/non-MCP compatibility, accepted
pause, corruption, cancellation and actual native retirement.

Integrated timing was 210.563 driver / 203.80 pytest seconds; final controls took
45.703 / 39.22 seconds. Both runners reached an empty contained Job at normal
parent exit, released identity, retired pipes/monitoring and removed private
profiles without force. PID history overflow was zero; the integrated run had one
lookup race and final controls none. These receipts qualify diagnostic cleanup,
not universal ordinary-app native cleanup.

Product SHA256: 78aca0d9d602054f762949b0e3b719d111ff11a6403a66278e80eb50cde5519d.
Final integration test SHA256:
349b587e120dfb7fbeb8ea6bbe1db5626b725438b2d479a245b4749a7dbc43ce.
The product and three affected test files pass scoped Ruff checks; test formatting
and syntax pass. Existing whole-product format debt and platform qualification
remain. No full suite was run. Task 34563.21 remains In Progress under the broader
DoD; the last whole-Send measurement still misses both responsiveness targets.

Next diagnosis measures only full runtime checks made inside the original
read-only parent walk, separating that subset from leaf allocation, append/atomic
bookkeeping and other trusted-directory helpers before any further contraction.


### Parent-walk subset before consolidation

The bounded parent-walk attribution run passed its original hook read and all
payload/member/physical retirement assertions. Twenty original walks entered and
exited. Only 42 runtime source checks were inside those walks: 18 config checks
accounted for 108 native opens and 24 hook checks for 192. Their instrumented
inclusive times were .113958 and .196600 seconds; all corresponding native
ancestry was outside participant proof. Remaining runtime checks occur at other
boundaries and are not candidates for removal by a parent-walk contraction.

This sample had 2154 total native opens and four acquisitions, including three
config acquisitions; the earlier 1341 sample had three acquisitions. The direct
hook batch still used one acquisition with 56 native opens and confirmed unbound
selector evidence. Do not infer a whole-read regression or an additive savings
claim from these differently initialized samples. The specific parent-walk subset
is the evidence for the next finite-operation change.

Run parent-walk-attribution-1: 1 pass, 14.046 driver / 7.64 pytest seconds. Bucket,
active/detail overflow and ancestry depth misses were zero, and original bindings
were unchanged. Normal contained Job exit proved an empty tree; native identity
and pipes retired, profile removed, no forced termination or PID history overflow
or lookup race. Observer SHA256:
0e9e7900949f59eb81d69fe4de25e782c325314f709503085a52eb8db9e015c3.
Product remained task21 checkpoint256a0f66b2. No performance acceptance follows.


Task 34563.22 causal baseline (`parent-preparation-red`) reached all original
directory-allocation, exact leaf payload, fresh leaf gate, descriptor-close and
lease-retirement assertions, then failed only its final full-check count:
six checks per stock parent walk instead of two. Product hashes remained
78aca0d9d602054f762949b0e3b719d111ff11a6403a66278e80eb50cde5519d
(raw) and 83fcedb357de00d7613355cc0ecc21c52a7a85618bab39c14f0ea41dba9733b8
(private paths). The test source was
a5ecabe8a075ed56e4a6867057485843cfcbb014a0b3fbfdbb4ed78fed57f431.
Elapsed 7.531 driver / 3.21 pytest seconds; normal contained retirement, private
profile removal and zero PID overflow/races. Product implementation follows this
causal failure, not a count-only stub or altered native callback.


### Finite parent preparation review and adversarial controls (task 34563.22)

The first integrated draft passed 38 controls and failed one old observer in
85.672 driver / 80.24 pytest seconds. All 13 new preparation controls passed,
including source/actor/lease drift, changed callbacks, expired closure use and
uncertain close retention. The old retirement observer watched only
private_paths._native_close; finite cleanup reaches the same original
raw._close_descriptor directly. Its relocated observer follows actual registered
FD closure and checks EBADF/removal before reuse, while retaining the zero-full-
check oracle across both original native and finite close callers.

Stock walk validation fell from six to two full checks; five original directory
allocations, one fresh actual leaf check and seven positive FD closes remained.
This is operation-count evidence, not whole-Send acceptance. Native containment
retired normally with empty Job, released identity/pipes and removed profile,
without force or PID history overflow/race.

Review identified a missing relation between the actual returned parent FD and
the already retained source pin after a transient rename/restore. Final source
validation alone does not prove those two descriptors identify the same parent.
The qualification will require an existing exact parent pin and compare actual
returned-FD identity with that same retained pin after final source validation;
unpinned parents keep original per-allocation checks.

The first rename regression stopped before mutation because pathlib.stat and the
installed Windows stat/fstat interface project different device identities. After
using the same installed native interface and protected-DACL creation, the native
rename was refused in this Windows config-scope fixture. Neither run proves the
rename/restore race. They took 8.063 and 8.156 driver seconds; both retired their
native Job/identity/pipes and private profiles normally, no force or overflow,
and respectively zero/one PID lookup races. The exact rename case is retained for
POSIX, and a portable real descriptor-substitution case now targets the returned
FD relation directly without changing source/pin/ownership metadata or callbacks.


### Final task34563.22 controls and matched no-span timing

The portable original-walker descriptor substitution and real unpinned-parent
controls both failed their intended oracles before the identity correction.
The first passed final source proof yet delivered one actual substitute FD;
the second used finite preparation without an exact parent pin. All owned
resources retired. That functional run took10.875 driver/5.63pytest seconds and
was coordinated for possible overlap with the integration chat's final functional
checks; it is not timing evidence. Source hashes stayed frozen, containment
retired normally and PID overflow/races were zero.

After pin qualification and returned-FD comparison, integrated controls passed
41 with one POSIX skip (92.468/86.92 seconds; one PID diagnostic lookup race).
Final review then exposed native identity reads after the last custody fence.
The causal control observed exactly two original fstats, closed one actual lease,
and proved the descriptor was incorrectly delivered before its final refusal
assertion failed. It took8.547/3.36 seconds, with positive native retirement and
zero PID overflow/races. The final implementation runs the existing source/custody
check after those fstats; no extra parent scan or new authority is introduced.

Final integrated selection: **42 pass, one POSIX-only skip**,101.297 driver/
96.05pytest seconds. The original stock walk retains two full checks, five
component allocations, one fresh actual leaf gate and seven positive FD closes.
An unpinned child uses seven original full checks. Both real descriptor
substitution and late actual lease revocation refuse with zero delivered handles.
Hook-read attribution remains1125native opens, three total acquisitions/one direct
member admission, nine balanced walks and zero bounded-observer overflow/depth
misses. The final contained Job was empty at normal parent exit; identity, pipes
and private profile retired, no forced termination or PID overflow/lookup race.
The exact rename/restore test remains unverified here and is skipped on Windows;
the portable FD test does not claim to reproduce that filesystem race.

Final product SHA256:
- raw_participants.py:0e6122758ed48527501e47d4d7b606d839b77755045da3aeed738e9c86227911
- private_paths.py:79944dfa7ca51586eb97e5d17c4ea89d964bb631c0a0f71d02b96e6301e1dd95
- new parent controls:a38598b3dc4221dd04766745ba1dfab3eafe1f3c1ec2ccb46f7f17e86fbe04f9

The existing `run-native-pair.py` ran baseline task21 (`256a0f66b2`) then final
candidate without detailed spans/native sampling, preserving its passive stage
and heartbeat observer. Exact baseline/candidate copies were retained; only the
two product files were staged, and final candidate bytes were restored after
positive baseline retirement. All7783Python source hashes were unchanged within
each run, and exactly those two files differed between samples. Deadlines, guards,
three saved replies, two nonstreaming plus one streaming reply, three complete
linked traces and zero dispatch checkpoints were unchanged.

| Sample | Send1 to adapter | Send2 to adapter | Send3 to adapter | Driver elapsed |
| --- | ---: | ---: | ---: | ---: |
| Baseline |11.483s|7.107s|10.582s|87.328s|
| Candidate |11.053s|6.469s|10.622s|86.688s|

Two Sends improved modestly and the third was effectively flat/slower. This one
pair supports no uniform speed claim and is far outside the one-second target.
The candidate third Send took the legacy ui_submit route at3.166s, whereas the
baseline third reached awaiting_review at.009s; retain that route difference
when interpreting the comparison. Candidate pre-controller intervals were
2.290/1.958/3.313s; post-durable-commit to trace reservation remained
6.324/2.853/4.493s. Those are stage intervals, not additive profiler timings.

Candidate Send heartbeat maxima were.345/.142/.277s and typing.403s, versus
baseline.373/.191/.201s and typing.905s. The100ms responsiveness target remains
unmet; heartbeat is not proof of terminal paint. Both timing runs retired normally
with empty Job, released identity/pipes and removed profile, no force or PID
overflow; baseline had one PID lookup race and candidate zero. These remain
bounded diagnostic cleanup receipts, not universal ordinary-app cleanup proof.

Scoped product/test lint is unchanged: raw and changed tests pass; private_paths
retains exactly the same pre-existing E721 at132. Changed functions and tests are
formatted. No full sweep was run; whole-file format debt and cross-platform
qualification remain. Existing ADR-225/ADR-126 apply. Task22 stays In Progress
under the broader DoD, and the next latency diagnostic adds bounded original
context/version caller purposes and requested-ID counts to the existing opt-in
observer, while preserving live revalidation across awaits and durable commit.


### Next context-read purpose attribution (source prepared; run pending)

Read-only review found no unconditional duplicate version query within one
ordinary synchronous compaction preflight: its captured snapshot tuple is already
reused downstream. Preaccept assessment, conditional transformed-request
assessment, postcommit dispatch and presentation workers have different freshness
boundaries. The existing version API already deduplicates/chunks IDs; empty input
performs no SELECT. The observed Linux5/4/5 helper starts therefore need caller
attribution before another consolidation is selected.

The existing opt-in SendSpanObserver now adds only the original
ChatPersistenceService.get_message_versions event target. Exact original-code
ancestry anchors distinguish early/changed-request assessment, dispatch,
presentation, manual/micro operations and preview/revalidation; they add no event
targets. Bounded records link version calls to their parent snapshot and retain
only requested exact-list/tuple lengths, caller names/lines and event indices,
not IDs, content, arguments or frames. Default60targets,4096events and existing
native deadlines remain. Detail limits are512records/128active/64ancestors with
explicit overflow, unmatched and unfinished counters. Counts are requested IDs,
not SQL rows/operations, and span times overlap.

Observer SHA25622cd31a3d9f35ae733ca35183ad3fe6e402f199ccff2a91422f660804b8dfc5f.
Ruff, formatting, AST/default-target comparison and root source review pass.
Runtime verification is pending. The next contained diagnostic should use the
integrated CLEAN snapshot after task22 and the independent live-config-lock
repair, avoiding duplicate diagnosis of the earlier UI path. The third-Send
receipt branch is already covered by the same observer. No production callbacks,
persistence gates or permissions were changed by this extension.

## Hook path-selection audit while integrated pricing qualification runs

The warm stock hook snapshot already contains `HookConfigSnapshot.profile_data_dir`,
derived from the saved raw mapping under the config writer lock. `_current` calls
`default_hook_permissions_path` only to compare that selection, invoking guarded
`get_user_data_dir` again. The subsequent stock raw hook scope compares
`selected_read` against current canonical selection; original participant, lease,
source and effect checks remain fresh. This identifies a possible naming-versus-
establishment boundary, not permission reuse.

Read-only analysis of retained original-body spans gives the following seconds
per Send. Each row contains five mutually disjoint `_current` contexts. Columns
are nested/inclusive and must not be added.

| Sample | Send | Hook context lifetime | Proven pre-yield minimum | Default-path lookup | Nested user-directory body | Permission read body |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| domain-capture-spans-1 | 1 | 2.990 | 2.970 | .208 | .195 | .123 |
| domain-capture-spans-1 | 2 | 1.971 | 1.952 | .347 | .335 | .126 |
| domain-capture-spans-1 | 3 | 1.810 | 1.794 | .163 | .152 | .105 |
| related-source-spans-1 | 1 | 3.132 | 3.089 | .321 | .304 | .177 |
| related-source-spans-1 | 2 | 3.074 | 3.046 | .591 | .565 | .213 |
| related-source-spans-1 | 3 | 2.970 | 2.937 | .278 | .259 | .217 |

The observer did not capture `_current`'s first yield. Its original `_read_state`
return precedes the successful yield, establishing the pre-yield lower bound;
the remaining caller/cleanup tail is 17-44 ms per Send. The default-path lookup
accounts for about 7-19% of the observed hook-context lifetime. The larger
unattributed pre-yield interval contains config/raw admission and lock preparation;
it is not evidence that JSON reading itself consumes those seconds.

Artifacts are the existing `.spans.json` and `.probe.json` pairs under
`.superpowers/sdd/2026-10-06-console-shared-tool-preparation/native-pairs/`.
Both have current original bindings, stable run sources, zero event/detail
overflow, and complete selected pairs. Raw ancestry walks have 164/122 depth
misses respectively, so child attribution is not exhaustive. Both retired the
contained process normally; the related-source run records one PID-lookup race.
Both precede task22: hook/config source is unchanged, but raw/private-path code
has changed. These establish historical priority only, not current savings.

No product change follows this audit. Cold/default-root selection owns serialized
fallback establishment and configured-base verification that cannot be removed
as a simple path substitution. Changed getters and snapshot providers must retain
the existing callback route and refusal behavior. The next source-frozen integrated
caller diagnostic remains the selection gate for further implementation; the
whole preparation cost takes priority over this bounded naming duplication.

## Integrated caller diagnostic after task22 and display-read fixes

The expanded related-source CI selection at CLEAN06ff60ba216a1551924d526563fe11cda411437e
contains75 cases including the task22 parent controls. Independently inspected XML:
Linux74 pass/1 skip52.793s; macOS74 pass/1 skip58.714s; Windows62 pass/13 skips115.284s;
zero failures/errors. The skip sets remain platform-specific. These are targeted
correctness checks, not a whole-suite or latency acceptance claim.

The integrated diagnostic used CLEAN HEAD49ca07d799e24a704d685ed48257943a5473c9d0
plus its explicitly frozen retained work. All7822 Python source hashes and HEAD
were unchanged. The first attempt, integrated-context-caller-detail-1, stopped
before native launch because CLEAN has no .venv: the original Windows launcher
rejects a missing executable before its process-creation API. Its empty log,
no-PID custody receipt and2.0s ValueError source receipt are retained; the private
profile was retained by the failure policy. It supplies no timing evidence.

The existing runner now accepts an explicit interpreter and uses the integration
owner's established -I source bootstrap: prepend CLEAN/core and CLEAN, then assert
Tests runner and application module origins under CLEAN. All containment/hash
function ASTs and native/test deadlines are unchanged. The corrected run uses the
already-tested source-current-venv-312 environment; no .pth or production source
was changed. Its receipt also records the exact command, interpreter, launcher
hash and private profile path.

integrated-context-caller-detail-2 completed91.047s driver/83.863s pytest, one pass.
Three user and three assistant messages were saved, two nonstreaming plus one
streaming reply completed, all three traces are complete and linked, and no
checkpoints remain. Normal empty Job at parent exit, tree/identity/pipes retired,
no forced retirement or PID-history overflow, one PID-lookup race; private profile
removed. This is diagnostic process retirement, not universal native cleanup.

| Stage or inclusive domain (seconds) | Send1 | Send2 | Send3 |
| --- | ---: | ---: | ---: |
| Action to adapter | 11.065324 | 9.555754 | 10.168618 |
| Action to controller entry | 2.750636 | 3.427218 | 3.379840 |
| Configuration capture | 2.312481 | 2.531542 | 2.617389 |
| Nested MCP maximum preparation | 1.921856 | 1.980092 | 2.160735 |
| Durable commit complete | 4.270445 | 5.017099 | 5.179785 |
| Commit complete to trace reservation | 6.351184 | 4.118808 | 4.737720 |
| Provider composition | 1.475132 | 1.629 | 1.656 |
| Fresh compose permission read | .711 | .722 | .619 |
| Six recovery-guard entries combined | .593 | .714 | .687 |

Domains overlap and are not additive savings. All three receipt returns are
accepted at the original successful source line; no third-Send legacy fallback
occurred. Awaiting-review stage arrives at.121416/.014436/.014201s, not a measured
paint time. Send heartbeat maxima.522995/.258465/.269512s and typing maximum.380146s
still exceed100ms. Both speed targets remain unmet.

The79-target observer remained source-current and retired normally. Event/detail
and context overflow, context unmatched/unfinished and raw unmatched/unfinished
counts are zero. Raw ancestry has90 depth misses, limiting exhaustive child
attribution. Generic exception-unwound function starts remain unfinished by
observer design; do not convert them into completed spans.

Exact context records reconcile28 snapshot/version pairs. The five live
acceptance/dispatch get_message_versions calls total2.662ms across all Sends:
Send1 dispatch2 requested IDs/.346ms; Send2 preaccept2/.512ms then dispatch4/.717ms;
Send3 preaccept4/.284ms then dispatch6/.803ms. They straddle commit and are not an
unchanged accepted state. No changed-request preaccept reread was observed.
Presentation contributes3/6/7 calls before adapter entry, with.303105/.541830/.824583s
worker elapsed respectively;16 calls total. Requested counts are2 throughout
Send1; four2-ID then two4-ID calls in Send2; three4-ID then four6-ID calls in Send3.
There are five later presentation calls during submission completion and two
outside those windows. Counts do not identify actual ID sets or prove unchanged
versions. The existing display reader already serializes warming and uses owner/
revision checks plus a1s TTL. This probe cannot infer invalidation causes or
contention, so these worker spans are not attributed as Send delay.

Decision: preserve live acceptance/dispatch freshness and focus TASK-34563.23 on
finite pending MCP canonical admission. The existing raw _Acquisition begins
before source selection, but repeated installed binding observations reacquire
canonical storage until active raw custody exists. The retained-lease observation
path still reads fresh generation witnesses. Consolidate only that admission
within each individual finite preparation, keeping related-member admission,
recovered destinations, original effect gates and actual cleanup independent.
No permission payload or lease is shared across capture, commit or composition.
The task plan and focused causal/lifetime controls precede product edits.

TASK-34563.23 causal baseline: pending-mcp-red-1 fails only the final expected
2-acquisition assertion with actual7 (six canonical observations and one member
admission). All real output/member-order/source gates,11 original selected_path
calls,12 fresh witness bodies, seven distinct issued leases closed exactly once,
and actual FD retirement pass first. Native opens1263. Selected six source hashes
are unchanged;11.875s driver/7.26s pytest. Contained native tree is empty at parent
exit, identity/pipes/profile retired, no force or overflow, one PID-lookup race.
This instrumented RED establishes the duplicated acquisition, not wall-time savings.

After RED, root-owned observer accounting permits exactly the independently
retained canonical observation lease alongside the unchanged ordered direct-member
leases. The restored-custody test now identifies the actual member lease by object
identity instead of list position, validates both correct execution selections,
and requires crossed canonical/recovered selections to refuse. These assertions
support early transfer without relaxing source, membership or cleanup checks.


TASK-34563.23 integrated baseline before the candidate
-----------------------------------------------------

`pending-mcp-integrated-baseline-1` runs the original wall-clock fixture on frozen
CLEAN `b678c396b466e8f8d1697ddfe2d30d55e51be26a`, without the optional function-span
observer. All Python source hashes and HEAD remain unchanged during this run.
The existing isolated CLEAN interpreter uses the previously verified explicit
source bootstrap; the timing runner's containment helper bodies and deadlines
are unchanged. No phase-span plugin is loaded.

The run passes in 89.281 seconds driver time. All three user turns and assistant
replies persist, all three traces complete and link, and no dispatch checkpoint
remains. Action-to-adapter times are **12.620 / 10.626 / 11.735 seconds**. Maximum
Send heartbeat delays are **1.089 / 0.426 / 0.486 seconds**; typing reaches
**0.503 seconds**. Neither the one-second Send target nor the 100ms input target
is met. Stage-return times are not a paint measurement.

The native Job is empty at normal parent exit; tree, identity, pipe tasks and
private profile retire normally, with zero forced cleanup, identity overflow or
PID lookup races. This is an integrated baseline, not evidence for a task23
improvement. Any later UI/source changes must be disclosed when comparing its
sample to this baseline.

## Canonical admission and actual native object census

TASK-34563.23 preserves all 11 original source selections and 12 fresh witness
observations while reducing stock preparation from seven real storage acquisitions
to two: one canonical observer and one independent member admission. Final
`pending-mcp-green-2` passes 17 causal/lifetime/late-callback controls in 79.500s
driver / 73.52s pytest, with unchanged source hashes and normal empty native Job,
identity, pipe and private-profile retirement (zero force, overflow or PID races).
The preceding compatibility run supplies 25 passing existing source and restored
custody controls. Its one accounting failure and two earlier lifetime-accounting
failures are corrected and included in the final passing selection; product code
has not changed between these runs.

Lease counts alone concealed substantial duplicated native work. The passive
original-body census records actual object identity from the original returned
native info; it does not issue additional reads. `pending-mcp-native-census-1`
passes with **1,013 open attempts, 962 successful opens across 23 real objects**.
Six ancestor directories each receive **103 opens (618 combined)**. Bootstrap is
opened 94 times; admission and registry.lock 54 each; unbound-owner 44; registry.json
42. Request/object/ancestry census overflow is zero. A second unchanged-source
sample in `pending-mcp-green-2` again reports 1,013 attempts. An earlier 1,436-open
sample includes a member evidence-reuse miss (479 member opens versus 56 in the
later successful reuse samples); it cannot establish a deterministic regression.

These are native open calls, not a count of distinct files. They establish the
next structural target: `_control_observation` currently restarts initial root,
admission and marker ancestry five times, then separately validates the completed
read. TASK-34563.24 will share one initial verified parent across those readers,
retain each actual record/lock read, and preserve the independent final named
validation. No witness or permission result is cached. Task23 remains In Progress
until sequential integrated candidate timing is recorded; neither speed target
has been met.

## Shared initial control traversal: task24

The original-body regression `control-parent-red-1` fails only its final initial
walk assertion: five initial drive walks instead of one; the two independent
completion walks, actual shared lock, three record reads, generation payload and
all physical native HANDLE closes pass first (7.312s driver / 2.73s pytest).

The implementation shares one verified root-parent walk, opens the root and
admission child relative to owned descriptors, and retires every initial pin
before yielding records. It uses the existing private native allocation/close
path. The original standalone reader ABI remains intact; stock reader identity
uses the existing activation callback anchors, and late drift refuses. The root
uses existing-child walk semantics so a private root below a trusted sticky POSIX
parent keeps its original behavior. Final named metadata/ancestor validation is
unchanged. No fresh witness, record parsing, registry lock or actual effect gate
is removed.

`control-parent-green-1` passes 25 checks with two POSIX skips in 28.96s pytest,
including the original Windows unsafe-ancestor DACL mutation. Source hashes remain
unchanged and the owned native Job, identity and pipes retire normally, with zero
force/overflow/races. Its driver then fails to remove the test-created temporary
ACL subtree. A separate scoped cleanup restores only existing-owner DACL access
and removes the retained 26 temporary objects; the original failure and subsequent
cleanup receipt remain recorded separately. This was test-profile cleanup, not
an unretired native process or a product success claim.

After custom-reader and module-identity review corrections,
`control-parent-module-green-2` passes four original-body controls in 23.406s driver /
18.73s pytest, with unchanged sources and normal native tree/identity/pipes/profile
retirement (zero force/overflow/races). Whole permission-read native attempts are
**689 versus 1,013**, a reduction of 324 (32.0%). Successful opens fall from 962 to 638
on the same count of 23 actual objects. Six ancestor objects fall from 103 opens
each to 55; bootstrap falls from 94 to 58. The original 11 source selections,
12 fresh witnesses, two real storage acquisitions, independent member admission
and completion pair remain. The member evidence reuse succeeds with 56 opens.
The initial control observation itself has 44 attempts, 42 actual handles positively
retired, and exactly one initial drive walk plus two completion walks. All census
overflow/depth-limit counts are zero.

These counts establish less actual filesystem work; they do not establish a 32%
wall-time improvement. Task24 remains open for its final lifetime checks and an
integrated combined-source Send sample. The one-second Send and 100ms input targets
remain unmet.

Final combined targeted verification, `control-parent-integrated-focused-1`:
**54 passed, five POSIX-only skips, one deselected** in 157.906s driver /
151.78s pytest. The deselected original Windows DACL mutation already passed in
`control-parent-green-1`; its separately recovered test-profile cleanup is noted
above. All selected source hashes remain unchanged. The real restored-custody
child and parent interpreters, owned native Job, identity, pipe tasks and private
profile retire normally, with zero force, overflow or PID races. The final census
again records 689 native attempts with the original witness/selection counts.
Scoped formatting and lint checks pass; bootstrap's 13 existing E721/E731 findings
are identical to its saved baseline and excluded from that scoped check. Final
independent source review found no remaining blocker. POSIX-specific CI and the
combined application timing sample remain outstanding.

## Combined application timing after task23 and task24

`control-parent-integrated-send-1` uses the original minimally instrumented
three-Send fixture on clean combined HEAD
`d97510c202918a51e9567c1c1669f49765b7c486`. No optional function-span observer is
loaded. All 7,828 Python source hashes and HEAD remain unchanged. The run passes
in 83.234s driver / 76.72s pytest, with three saved user turns and three replies,
three complete linked traces, and zero remaining dispatch checkpoints. The owned
native Job is empty at normal parent exit; tree, identity, pipe tasks and private
profile retire normally, with zero force, overflow or PID races.

| Measured interval (seconds) | Send1 | Send2 | Send3 |
| --- | ---: | ---: | ---: |
| Action to provider adapter | 9.141634 | 8.710774 | 9.299418 |
| Action to controller entry | 1.450203 | 2.081731 | 2.296478 |
| Durable commit body | .359961 | .208315 | .209540 |
| Action to durable commit completion | 2.791011 | 3.231549 | 3.627809 |
| Commit completion to trace reservation entry | 5.935990 | 4.903601 | 3.767843 |
| Trace reservation body | .186524 | .324053 | 1.454944 |
| Maximum Send heartbeat delay | .714115 | .285889 | .403179 |

Typing heartbeat maximum is .284907s; startup reaches 1.863500s and initial idle
3.803818s. First Send's action-dispatched stage returns at 1.449650s; subsequent
awaiting-review stages return at .011871s and .008459s. These stage returns are
not paint measurements. Neither the one-second Send nor 100ms input target is met.
The earlier minimally instrumented baseline on `b678c396` was
12.620 / 10.626 / 11.735s. Intervening display/Context source changes prevent
attributing the combined timing difference solely to these filesystem changes.

Remaining source observations are mapped, not silently removed: six pre-active
selection/participant checks, three outer/nested guarded method entries, the
reader operation check and the actual file-open check account for 11 selections.
The twelfth witness is the separate readable/recovery approval check, including
restored mapping, workspace and owner approval. The caller-side participant field
lookup before the source-lock gate contains a potential pure duplicate, but fresh
post-lock validation, recovery approval and actual file-effect gates remain
required. The measured postcommit interval now takes priority for current phase
attribution; operation counts alone do not identify all of its elapsed delay.


## Current function attribution on combined task23/task24 source

`control-parent-integrated-spans-1` runs the existing original-function span
observer on unchanged combined HEAD `d97510c202918a51e9567c1c1669f49765b7c486`.
All 7,829 Python hashes remain unchanged; the extra file relative to the preceding
sample is the peer's Collections regression, with no intervening product edit.
It passes in 64.141s driver time, with three saved turns/replies, three complete
linked traces and zero checkpoints. Native Job/tree/identity/pipes/profile retire
normally, zero force/overflow/PID races. The observer records 60 original code
objects, current bindings/sources, zero event overflow and zero unfinished admission
entries; monitoring retires. The optional deeper preparation-detail switch is off.

| Inclusive measured interval (seconds) | Send1 | Send2 | Send3 |
| --- | ---: | ---: | ---: |
| Action to provider adapter | 7.190675 | 7.861561 | 6.448245 |
| Configuration capture | .946 | 2.076 | 1.286 |
| Commit body | .249 | .146 | .195 |
| Commit completion to trace reservation entry | 4.024365 | 3.318833 | 3.127000 |
| Postcommit prompt history | .269 | .285 | .319 |
| Postcommit hook admission | .234 | .289 | .596 |
| Tool-provider composition | 1.408 | 1.035 | .891 |
| Agent run-log writer binding | .292 | .367 | .096 |

These are inclusive spans: nested recovery, controller, bridge and gateway
intervals overlap and must not be added. Adapter elapsed includes all work and
scheduling, while the table isolates selected original function lifetimes. The
same product's earlier minimally instrumented sample was 8.711-9.299s, so neither
this lower sample nor the optional instrumentation establishes an isolated saving.
Typing heartbeat still reaches .377s and Send maxima .406/.395/.332s. Both original
responsiveness targets remain unmet.

The largest individually identified preparation body after commit is still
stock tool-provider composition. Together with the precommit configuration capture
and the native object census, this justifies investigating nested MCP guarded
helper validation under one synchronous source owner. Separate post-lock source
validation, restored-owner approval, actual destination/open/write gates and
positive native retirement remain required. No check is removed based on timing
alone; direct/custom calls and corrupt/missing-file behavior need causal controls.

## Task25: registration consumes its already-observed source

The former registration path repeated three complete source/witness observations
just to record or retrieve participant metadata. Installed MCP preparation now
uses one pure registry helper after its original initial source selection. Exact
source/type/owner/path identity remains checked. The full pre-lock, post-lock,
outer/nested method, readable/recovery and actual file-effect gates remain.
The separately discovered pending-preparation source-retarget error category is
restored without changing its authority or native callback order.

| Original permission-read census | After task24 | After task25 |
| --- | ---: | ---: |
| Native open attempts | 689 | 575 |
| Successful opens | 638 | 530 |
| Actual filesystem objects | 23 | 23 |
| Each of six common ancestor directories | 55 | 46 |
| Source selections / fresh witnesses | 11 / 12 | 8 / 9 |
| Full raw checks / independent acquisitions | 5 / 2 | 5 / 2 |

The final count run also verifies real payload, each retained selection's fresh
witness, recovery approval, member order, exact lease lifetime and physical handle
retirement. All native census overflow/depth limits are zero. Compared with the
1,013-attempt pre-task24 checkpoint, these two changes remove 438 attempts (43.2%).
This is repeated traversal of 23 objects, not hundreds of unique payload files.
It does not establish a proportional wall-time improvement.

`mcp-registration-red-1` on unchanged saved product fails only the final expected
8/9 count assertion (actual11/12); all earlier behavior/custody assertions pass.
`mcp-registration-green-1` initially passes64 checks and fails11 existing lifetime
cases. Six representative failures reproduce against exact saved a759 source in
`mcp-registration-saved-controls-1`, covering all four causes: retarget refusal
category, a nonpersisting permission seed, a legitimate history cache hit that
never armed the intended close injection, and socket-only Windows pipe polling.
The product category is repaired; the fixture repairs retain original actual
native operations, positive target/custody proof and all original deadlines.

Final affected verification comprises22 passes across five sequential batches:

| Evidence label suffix (prefix mcp-registration-) | Tests | Driver seconds |
| --- | ---: | ---: |
| final-boundaries-1 | 14 | 49.828 |
| final-permission-1 | 2 | 28.125 |
| final-history-read-1 | 2 | 28.234 |
| final-history-append-1 | 2 | 23.672 |
| final-history-atomic-1 | 2 | 22.360 |

All11 earlier failed cases pass in this final scope. Every final batch preserves
selected source hashes and retires its native Job/tree/identity, pipe tasks and
private profile normally, with zero force, diagnostic overflow or PID races.
The initial broad green run overflowed its bounded PID diagnostic inventory194
times; normal native Job emptiness still held, but that inventory is incomplete.
Final child-heavy cases were split into pairs to preserve complete diagnostics
without raising any limits. Scoped lint/format/diff checks pass; the older lifetime
module retains its same13 pre-existing Ruff findings. Independent source review
found no remaining issue.

TASK-34563.25 stays In Progress pending supported-host CI and combined-source Send
verification. Latest pre-task25 Send samples and the unmet one-second/100ms targets
remain as recorded above. No latency acceptance follows from this count reduction.

### First task25 integrated timing attempt

`mcp-registration-integrated-send-1` ran on saved combined
`13b0523fceea5192ce0e7bfd4d89f989eba72e75` with the optional phase/detail observers
off. It failed after34.641s, before any provider call; this is not a usable Send
latency sample. All7,832 Python hashes and HEAD remained unchanged. Native Job,
tree, identity, pipes and private profile retired normally; zero force, overflow
or PID races.

The first failure is the diagnostic's immediate custody assertion after the
visible action returned `awaiting_review` in77ms. That supported hook path starts
an asynchronous continuation before turn acceptance. While assertion teardown
was underway, the continuation refused and a config-route worker hit
`raw_resources_not_retired` for an existing uncertain selected source; Textual's
WorkerFailed then masked the earlier assertion. The retained child-log tail does
not identify the original transition that made that config state uncertain.
Do not attribute it to registration or Actor startup without further evidence.

Task25 plan9 scopes an observer correction: follow only the current exact
screen/hooks/session/generation and accepted turn, using the already-established
15s Send deadline. A refusal or changed identity still fails; no retry, review
approval or deadline extension. The integration owner separately owns shutdown
corrections. Overall latency and cross-platform qualification remain open.

## Task25 integrated Send result and remaining preparation cost

The corrected observer follows only the exact pending Send within the original
15-second deadline. Sequential runs on saved combined
`d5247a2d051d37426c8864a482869403c56442f1` both complete three saved user turns,
three replies and three linked complete traces, with zero remaining dispatch
checkpoints. The first two provider calls are non-streaming; the third streams.
All 7,833 Python hashes and HEAD remain unchanged in both runs. The native Job is
empty at parent exit; tree, identity, pipe tasks and private profile retire
normally, with zero force, identity overflow or lookup races.

The minimally instrumented `mcp-registration-integrated-send-2` passes in
68.063 seconds driver / 62.695 seconds pytest. Optional phase and detail observers
are off. Its measured intervals are:

| Seconds | Send 1 | Send 2 | Send 3 |
| --- | ---: | ---: | ---: |
| Action to provider adapter | 7.635265 | 7.887032 | 7.808551 |
| Action to controller entry | 1.351106 | 1.772445 | 2.441423 |
| Durable commit body | .234482 | .284052 | .261044 |
| Action to durable commit completion | 2.605608 | 4.003077 | 4.164699 |
| Commit completion to trace reservation entry | 4.580382 | 3.446441 | 3.199156 |
| Trace reservation body | .202893 | .193265 | .129342 |
| Maximum Send heartbeat delay | .478189 | .283349 | .316309 |

Typing heartbeat reaches .349622 seconds; startup reaches 2.061333 seconds.
Neither the one-second Send target nor the 100ms input target is met. Changes
integrated between this and the prior sample prevent attributing a particular
wall-time saving to task25 alone.

The existing preparation-detail observer then runs on that same frozen source as
`mcp-registration-integrated-detail-1`. It passes in 67.375 seconds driver with
adapter samples 7.530910 / 7.882063 / 8.602772 seconds. Original bindings and
sources remain current and monitoring retires. Event/caller capacity overflow,
unmatched starts and unfinished selected/admission/generator entries are zero.
The bounded raw ancestry walk reaches its limit 92 times; hook raw counts are
partial observations, not a complete native census. No limit was increased.

| Inclusive seconds in action-to-completed-submission window | Send 1 | Send 2 | Send 3 |
| --- | ---: | ---: | ---: |
| Configuration capture | 1.046527 | 1.579835 | 1.254978 |
| Initial MCP maximum | .563845 | 1.110835 | .942009 |
| Three guarded MCP reads | .956469 | 1.015194 | 1.022746 |
| Tool-provider composition | 1.344138 | 1.077705 | .932056 |
| Hook admission, two calls | .483491 | .651377 | .595428 |
| Message-version reads, 4 / 5 / 9 calls | .396714 | .322155 | .798993 |
| Run-log writer binding | .346349 | .286001 | .211673 |

These parent/child spans overlap and cannot be added. Background calls inside
the measured window are included, so the aggregates alone are not critical-path
proof. Each Send contains five hook contexts and 450 / 508 / 450 attributed raw
checks, with .421419 / .579274 / .541762 seconds inclusive raw time; the ancestry
limit above applies. The first received-intent call returns None at wiring.py
line 437, the combined raw-mapping/standard-configuration qualification gate.
The observer does not distinguish that gate's operands. Later received-intent
calls return an accepted ID at line 561; the fallback is not forced or bypassed.

The evidence still supports simplifying nested MCP load preparation under its
existing synchronous owner. A generic nested-guard bypass remains unsuitable:
permission loading can return defaults without opening a file, and corrupt-load
recovery can rename a file. Final source validation and each actual effect gate
must therefore remain. Supported-host qualification is still tracked by the
integration owner; task25 remains In Progress until that evidence is complete.

## Task26: one owner for the stock permission payload read

The named Console payload reader now uses its existing checked source scope
instead of opening two additional nested load scopes. The original payload body,
mutation lock, readable/recovery check, actual file/backup gates, final native
validation and source receipt remain. Custom skipped members retain their ordinary
route. Definition-time metadata covers only those omitted calls, including the
raw scope/check and their actual package/module join; custom descriptors are
rejected before binding. No read cache or lease crosses an await.

The unchanged checked-load baseline (`owned-permission-load-red-1`) records 621
native open attempts, 574 successful opens and 23 objects. On the final product,
`owned-permission-isolated-count-1` records 529 attempts, 486 successful opens and
the same 23 objects: 92 fewer attempts (14.8%). The existing source scope count
falls from three to one; reader, file and final checks remain one each. Fresh
selection/witness counts fall from 9/10 to 7/8. The two independent acquisitions,
full payload, member order and physical descriptor/lease retirement remain.
The final isolated run passes in 8.547s driver / 4.750s pytest with unchanged
source, normal Job/tree/identity/pipes/profile retirement and zero diagnostic
capacity or ancestry misses. This is an operation-count result, not Send timing.

The first 19-case integrated batch also exposed why count comparisons must retain
the admission evidence. Its count recorded 952 attempts with the same four raw
checks and 7/8 observations. Every additional attempt was inside the direct
related-member acquisition (56 to 479): its original reuse check observed an
actual temporary-profile parent permission change (mode 0700 to 0744 and changed
ACL), declined reuse and performed ordinary admission. The final quiet count
has unchanged posture and accepts reuse again. The modifying actor is not
established. Preserve both records and the original refusal; do not treat that
423-attempt fallback as a product improvement or a reason to relax identity.
Final local task26 verification covers 109 distinct passing controls (26 new,
83 existing). Shared preparation/provider composition and corrected custom-guard
controls pass 61/61; standalone snapshot, source semantics, cancellation and
actual Console wiring pass 25/25. The original new batch supplies the remaining
count/normalization/default/corrupt/custom/source-drift controls. Final effect
controls include the real inactive restored-source path and exact native read-FD
close uncertainty, with physical custody retained on uncertainty. New test-only
expectation corrections preserve original error categories and distinguish a
generator's first entry from its resume; no product or budget was changed to
obtain those passes. Every batch has unchanged-source and normal containment
retirement receipts, with zero forced cleanup/identity overflow/races.

Root reviewed both implementation lanes before integration and repaired the
independent reviews' direct compatibility findings. Scoped static checks pass;
all original raw module statements and `_checked_read` remain AST-identical.
Task26 AC1–3 are checked; AC4 and the original performance targets stay open
until combined application timing and supported-host verification complete.

### Remaining native work after the owned permission load

The final isolated `owned-permission-isolated-count-1` XML reconciles all 529
native open attempts across 23 objects. One attempt opens the actual permission
payload. The repeated infrastructure work is:

| Mutually exclusive owner | Native open attempts |
| --- | ---: |
| Four retained source/effect/publication checks | 184 |
| Two storage acquisitions | 112 |
| Other setup, recovery control and directory pinning | 232 |
| Actual payload leaf | 1 |
| Total | 529 |

Each retained check has 38 opens in its fresh source proof and eight for parent
association. Each acquisition has 56 opens. The largest buckets outside those
checks/acquisitions are `_control_observation` (96), private descriptor allocation
(40) and the final `pinned_directory` tree pass (26). These are operation counts,
not distinct files or independent elapsed spans. The existing count observer has
zero ancestry or capacity misses in this sample.

Source review found an unselected setup option: move initial installed MCP
selection/preparation under the existing source mutex to avoid a separate
pre-lock source observation. Its approximate ceiling is only one 38-open witness
per read, and moving that observation changes lock order relative to recovery
admission. It needs a separate deadlock/cancellation argument and is not selected
for implementation. Keep the four existing effect/publication checks. The current
priority is the larger whole-Send configuration/catalog and agent-startup cost.

An independent source-only review identifies a concrete hypothesis for the local
provider span: `_default_specs` synchronously reads three tool-exposure config
gates, with an additional internal-prompt config lookup when Ask User is enabled.
Enabled deep search reads nine settings to use one description timeout. Building
specs for multiple admitted roots repeats that configuration work. Native root
resolution/ancestor checks are a separate constructor cost. These are confirmed
call chains, not yet a measured partition of the remaining composition duration.

Those exposure gates are absent from the current turn snapshot and are observed
at provider construction, not re-read at invocation. Any selected consolidation
must therefore capture fresh catalog inputs at the existing composition owner,
including the current environment/prompt precedence, and share only immutable
construction data among stock builders. Invocation-time root, kill-switch and
permission checks retain their existing lifetimes. Reusing an older turn snapshot
would change the current freshness contract and is not the proposed approach.

The optional observer extension at `98cb24aad5` adds the four original targets
needed to locate that remainder while retaining all default targets and bounds.
Its static checks pass; runtime qualification and combined observer-off timing
remain pending after the independent initial-draft repair. No new product change
or proportional speed claim is made from this source-only review.


## Task26 integrated timing after draft repairs

Both sequential runs use frozen integration HEAD
`031b8d28dc4c984182f3c8f5e74077f037ea0d9d`, including task26
(`7bc2d4aa6d41fa4da50ee161c6ecf13a95ccf8a8`), the separate startup/switch draft
repair (`dbf27ba534b998c82808f69ef08ca59b19a88754`) and the saved optional
observer (`6b629e982a688eddd6044ae1f531e367cf99d528`). All three parallel lanes
and the integration owner were shell-idle during measurement. The source is
combined; comparison with d524 does not isolate task26's wall-time contribution.

`owned-permission-integrated-send-1` is the existing quiet diagnostic without the
optional spans or heavy native census. It completes in 60.735 s driver / 56.018 s
pytest. All three user/assistant messages persist; nonstreaming/nonstreaming/
streaming replies yield three complete linked traces and zero checkpoints.

| Quiet measurement (seconds) | Send1 | Send2 | Send3 |
| --- | ---: | ---: | ---: |
| Action to adapter | 7.049159 | 6.762781 | 6.703718 |
| Action to controller entry | 1.180988 | 1.700554 | 1.967575 |
| Durable commit body | 0.221166 | 0.244692 | 0.236082 |
| Action to durable commit complete | 2.281652 | 3.431674 | 3.207896 |
| Durable commit complete to trace entry | 4.379221 | 3.012768 | 3.229847 |
| Trace reservation body | 0.172533 | 0.142002 | 0.126962 |

Send heartbeat maxima are .414586/.251354/.326992s; typing .259959s, startup
1.620373s, idle .794014s and shutdown .030924s. Both product targets remain
unmet. First Send takes the ordinary `ui_submit` route; the later two use early
received intent. These stages are not rendered-frame evidence. The old heavy
40,000-open whole-Send budget has not been requalified by this lighter diagnostic.

`owned-permission-integrated-detail-1` then completes in 59.500 s driver / 54.703 s
pytest on the same frozen source, again with all three replies/linked traces and
zero checkpoints. Adapter values are 6.636999/6.321080/5.923517s; they remain
separate instrumented samples, not replacements for the quiet values above.
The four additional original-body spans now locate the composition remainder:

| Inclusive original span (seconds) | Send1 | Send2 | Send3 |
| --- | ---: | ---: | ---: |
| Whole provider composition | .768296 | .614665 | .643922 |
| Shared tool preparation | .740071 | .611466 | .640525 |
| Local external catalog | .540950 | .366745 | .360997 |
| Compose permission payload | .179042 | .187315 | .221321 |
| Local provider function | .026118 | .001266 | .001510 |
| Turn configuration capture | .742217 | 1.279160 | .845061 |
| Nested MCP maximum capture | .435419 | .620263 | .422285 |
| Five hook-current contexts | 1.282689 | 1.484787 | 1.579353 |
| Awaited prompt history | .311771 | .225704 | .232971 |
| Run-log binding | .278829 | .257641 | .188324 |

These rows overlap and must not be added. The local-provider function is tiny on
this sampled route, so its speculative config/root changes are not selected.
The observer does not establish that every enabled multi-root/deep-search builder
ran; those optional shapes remain unmeasured. Catalog preparation and hook-current
entry are the larger domains. Catalog already decodes one JSON payload; nested
source guards and governance/worker admission are its remaining work. Initial
maximum and postcommit catalog observations retain separate freshness boundaries.

The 83-target observer keeps its original bindings/source current, zero global
events, unchanged 4096 event / 64 ancestry limits and normal monitoring retirement.
There are zero event/detail/context overflows, unmatched returns, unfinished
admission/generator/hook/raw/context entries or context ancestry misses. Raw
hook ancestry still records 92 depth misses, so raw child attribution is partial.
First receipt returns `None` at the combined gate on wiring line 437; later ones
return accepted at line 563. The current probe does not distinguish that gate's
operands; no failed operand or enabled-builder behavior is inferred.

Both runs retain unchanged HEAD and all 7841 Python source hashes, and reach an
empty contained Job at normal parent exit. Native identity, pipe/identity tasks
and private profiles retire with zero forced cleanup, identity overflow or lookup
races. Original deadlines/guards stay unchanged. These are scoped diagnostic
retirement receipts, not universal production cleanup proof. Supported-host
qualification of the new task26 controls is requested from the integration owner;
its final acceptance criterion stays open. The larger speed/stability task remains
In Progress. Follow-up review options are maintained in the linked optimization
review list; further work follows the measured hook/catalog boundaries.

### Current hook-context partition

The same frozen detail run partitions each of the five hook-current contexts by
original start/yield/read/return events. Unlike the inclusive table above, these
rows do not overlap:

| Boundary (seconds across five contexts) | Send 1 | Send 2 | Send 3 |
| --- | ---: | ---: | ---: |
| Config snapshot entry to first yield | .527 | .633 | .863 |
| Default hook path derivation | .184 | .342 | .189 |
| Hook raw scope entry to first yield | .136 | .135 | .156 |
| Hook scope yield to state-read entry | .355 | .296 | .293 |
| Actual hook state read | .063 | .061 | .064 |
| Remaining caller, interstitial and exit work | .018 | .018 | .015 |
| Total hook-current lifetime | 1.283 | 1.485 | 1.579 |

Configuration entry accounts for 41-55% and includes admission, interprocess
locking, config reading and projection. The pre-state-read interval contains
hook lock creation/opening and locking. Send 2's first default path takes 210 ms;
Send 3's first config entry takes 399 ms, including 263 ms in raw entry. Elapsed
time alone does not establish contention. The 92 raw ancestry misses still
limit deeper attribution, and an exception-unwound create has no completed
return span; this does not establish zero cost.

Source review finds the same lock pair in config and hook owners: attempt empty
private-file creation, catch existing-file refusal, then independently prepare
and open an append stream. Both establish the application-owned parent. Simply
removing creation would lose the successful cold-create fsync. The next candidate
is one finite private lock-stream operation, preserving actual exclusive-create
outcome, cold durability, current parent/leaf checks, ordinary portalocker order
and existing stream/native retirement. Obtain causal original counts before
product changes. Keep the five fresh hook observations separate. OPT-09 records
this candidate; OPT-03's directory-selection contraction remains deferred.


### Task27 original lock-preparation control

`lock-stream-parent-red-1` ran the four original-code cold/warm config/hook
cases before product changes. All four failed only the final duplicate-parent
assertion: two establishments instead of one. Earlier assertions confirmed real
payloads, original private-file callbacks, held portalocker streams, registered
native descriptors and complete physical descriptor/lease retirement.

| Original case | Parent establishments | Whole observed read native attempts | Descriptor closes |
| --- | ---: | ---: | ---: |
| Cold config | 2 | 1623 | 48 |
| Warm config | 2 | 1622 | 47 |
| Cold hook | 2 | 2117 | 68 |
| Warm hook | 2 | 1125 | 42 |

Each warm case includes one exception-unwound create attempt. Counts are native
attempts over the whole observed read, including surrounding preparation; they
are not unique files or lock-leaf opens. Cold means the lock file is absent,
not a cold process. These are causal test counts, not quiet latency measurements.
The driver took 17.765 s and pytest 14.066 s. Saved HEAD `34bd9b0dca` and the
source manifest stayed unchanged. The contained Job exited normally and retired
its native identity, pipes/tasks and private profile, with zero forced cleanup,
overflow or lookup races. This establishes task27's RED before implementation.


### Source review of pre-controller preparation on frozen031b

The quiet first Send spends 1.143 s between UI dispatch and legacy submission,
with controller entry 1.181 s after action. Later receipts return awaiting review
in 8.8/7.0 ms, while their controller entries occur 1.701/1.968 s after action.
The detailed source trace confirms only the combined initial refusal and later
accepted receipts; it does not identify a failed predicate operand.

In the first detailed fallback, hook-current (.302 s) and MCP read_sources
(.414 s) execute in workers and are awaited. Configuration capture (.103 s) then
runs on the main thread, after two separate user-directory calls (.031/.030 s).
A third .029 s directory call is nested within capture and is not additive.
This proves some synchronous native work but does not explain the whole interval.

Later detail shows hook -> MCP maximum -> configuration workers in sequence.
Send 2: hook .021-.499 s, MCP .506-1.105 s, configuration 1.126-1.546 s,
versions 1.617-1.716 s, capture return 1.779 s, controller entry 2.090 s.
Send 3: hook .017-.602 s, MCP .609-1.012 s, configuration 1.032-1.321 s,
versions 1.307-1.383 s (partly overlapping configuration), capture return
1.447 s, controller entry 1.556 s. Unselected main-thread gaps remain.
Async intervals combine actual work and waits; whole-Send heartbeat maxima
cannot attribute pre-entry blocking or distinguish scheduling from lock wait.

There is no duplicate MCP maximum: read_sources is the worker within
capture_console_definition_maximum, whose result is explicitly passed into the
configuration producer, skipping its fallback maximum capture. Hook review
precedes preparation intentionally: later reads can migrate catalog schemas or
audit policy changes. Starting them before ready consent changes effect order.
After ready review, MCP and non-MCP preparation could be independently retained
and joined before original snapshot assembly/revalidation, but shared locks and
failure/custom-callback contracts need evidence before implementation. OPT-30
and OPT-31 retain these options. The separate attachment completion repair must
be integrated and measured before attributing current first-Send behavior.


### Task27 draft integration and posture correction

The combined draft run `lock-stream-draft-edges-1` completed in 31.484 s driver /
27.396 s pytest with 10 passes and six failures. All four original duplicate-
parent controls and six actual locking/integration cases passed. Four new helper
controls and two posture cases were refused before their intended operation:
the fixtures incorrectly declared an application-owned parent for an explicit
custom config path, whose real policy returns None. The original append API
failed the same fixture. These failures were not accepted as causal race evidence.

The posture fixture was corrected to a real HookPermissions owner with the
configured profile parent, inside the original held config scope. No guard or
assertion was weakened. `lock-stream-posture-red-2` then failed both cases only
at the final unsafe-acceptance assertion: actual EEXIST followed by owned-parent
privacy drift, and actual cold-create fsync followed by non-owned ancestor write
permission drift. Original identities, payloads, actual mutation and stream/FD/
lease retirement were confirmed first. Windows used real DACL changes rather
than unsupported chmod modes; POSIX controls use the corresponding mode changes.

The corrected run took 8.968 s driver / 5.451 s pytest. Both runs retained HEAD
and all selected source hashes, normal empty Job/identity/pipes/profile retirement,
and zero forced cleanup, overflow or lookup races. The shared correction retains
a fresh full ancestor traversal after both proven exclusive outcomes, checks the
current owned-parent private posture, and keeps the actual created FD through
fsync and stream transfer. No cold traversal saving is claimed. Final integrated
qualification and current Send timing remain pending.


### Task27 corrected integration and regression evidence

The corrected private_paths source (SHA256
50e0e7322e2626d00abd229da32642d10f328968a19b472a519551932bd38c83)
passed all 12 integrated controls in `lock-stream-integrated-green-1`:
four original count controls, six real config/hook locking and body-failure
controls, and both real parent/ancestor posture races. Driver/pytest elapsed
was 34.469/29.714 s. Parent establishment is now once in all four cases.

| Case | Original to current whole-read native attempts | Current descriptor closes |
| --- | ---: | ---: |
| Cold config | 1623 to 1484 | 38 |
| Warm config | 1622 to 1495 | 38 |
| Cold hook | 2117 to 1980 | 59 |
| Warm hook | 1125 to 1008 | 34 |

These are whole observed read attempts, not unique files or latency savings.
Cold successful creation still fsyncs the actual FD later transferred into the
stream; warm acquisition adds no fsync. Both paths retain fresh full ancestor
validation and exact current owned-parent privacy. The old public append
signature/docstring and 54 unrelated original definitions are unchanged.

`lock-stream-native-regression-1` was not accepted: the new allocator test
attempted to patch the read-only dynamic Windows supports_dir_fd property;
failed teardown left its open callback installed for two following cases.
The independent existing MCP retained-history test also failed importing asyncio
in its child because its copied environment omitted Windows interpreter inputs.
The allocator fixture now patches only mutable support sets when necessary;
the roundtrip allowlist correction was reviewed but is excluded from this change
because its full startup/backup/restore acceptance is outside task27. Product,
assertions, native cleanup and deadlines remain unchanged.

The clean `lock-stream-native-regression-2` rerun passed 21 cases with six
POSIX-only skips (31.390/27.694 s), including all ten new helper cases, existing
private artifact controls and original raw native retirement. Together, all
22 new controls pass. Both accepted runs retain saved HEAD34bd9b0dca and source
hashes, normal empty Job/tree/identity/pipes/profile retirement, and zero forced
cleanup, identity overflow or lookup races.

The later original config/hook combined batch
`lock-stream-config-hook-regression-1` ran 102 cases in 273.016/268.516 s:
29 config lifetimes and six config lock-order cases passed; hook permissions
had 61 passes and six failures. Its cumulative identity diagnostics overflowed
424 times and recorded one lookup race. The Job still retired normally and
source hashes remained current, but that combined batch is not clean native
qualification. Focused smaller controls will retain the original diagnostics.

The six hook failures are platform mismatches in original tests: three actual
symlink creations fail with Windows privilege error1314; two stdlib pathlib
mode operations assume POSIX permissions instead of the real DACL; the v2
launch-serialization test waits for loop.subprocess_exec, which the intentionally
unsupported Windows command executor refuses before launch (ADR-163). These
are not counted as passes or silently skipped. Real new DACL race controls are
green; original POSIX cases still require supported-host execution. No product
change or deadline extension is justified by those failures.

Scoped new-test lint/format checks pass; product Ruff findings remain baseline
1/56/3 for private_paths/config/hook_permissions, with no new findings. Independent
source review found no remaining blocker in the corrected lock-stream delta.
Current integrated Send timing and supported-host qualification remain separate.


Final focused original controls: `lock-stream-original-controls-1` passed nine
cases (six native config lock-order cases, real legacy hook launch/revoke,
config-lease/foreign-file refusal and recovery admission) in25.047/21.299s.
`lock-stream-history-controls-1` passed actual installed history append/migration/
rotation across pause and foreign-task/path helper refusal, in7.250/3.917s.
Both kept HEAD and source hashes, with normal native retirement and zero forced
cleanup, identity overflow or lookup races. These focused controls qualify the
changed call paths without the unrelated full-app backup/restore roundtrip.
Its Windows environment fixture issue remains documented for separate review;
its unverified fixture edit is excluded. Original POSIX-only controls remain a
supported-host qualification item. No assertions or deadlines were relaxed.


### Integrated task27 quiet and detailed Send samples, 2026-10-08

Integration owner saved task27 in7824bc6850b017760b55b6bfc60295fa34861085;
all22new controls passed again in46.82spytest/53.531sdriver with unchanged
source/HEAD and normal native retirement, zero forced cleanup/overflow/races.
The quiet and detail runs below use frozen cleanb4a284513b5837998017c12e146aea58b0356a7d,
the exact earlier source-current Python3.12 interpreter and unchanged native
launchers. Peer tests, source edits and host analysis were paused during timing.

Quiet `lock-stream-integrated-send-1` PASS74.938sdriver/68.730spytest:

| Timing in seconds | Send1 | Send2 | Send3 |
| --- | ---: | ---: | ---: |
| Action to adapter | 8.530106 | 8.031268 | 7.938409 |
| Action to controller | 2.007627 | 2.359209 | 2.097941 |
| Durable commit body | .338777 | .220950 | .300818 |
| Action to durable commit complete | 3.304732 | 4.067844 | 3.822725 |
| Commit complete to trace reservation entry | 4.795620 | 3.550926 | 3.618096 |
| Trace reservation body | .184684 | .201938 | .125797 |

All three now use early awaiting-review receipts, returned at.131398/.009174/
.008970s; the first no longer takes the previous legacy fallback. This is not a
render/paint timestamp. Heartbeat maxima: Send.629569/.399890/.276254s,
typing.481068s, startup1.969293s, idle.307386s and shutdown.038336s.
The sample is slower than the older031b point sample. Count reduction does not
establish latency improvement, and combined source changes plus point-sample
variation do not isolate task27 as its cause. Main acceptance remains unmet.

The diagnostic completed3user/3assistant records, F/F/T provider streaming,
three complete traces and three response links with zero dispatch checkpoints.
All7847Python hashes and savedHEAD remained unchanged; native Job/tree/identity/
pipes/profile retired normally with zero forced cleanup/overflow/lookup races.
The original40k native-open acceptance budget and rendered feedback were not
requalified by this quiet diagnostic.

Detailed `lock-stream-integrated-detail-1` PASS74.047sdriver, adapter10.506469/
7.696659/8.025910s. These are separately instrumented observations. Source/HEAD
and the same transcript/trace/retirement outcomes were preserved; PID diagnostics
recorded one lookup race, with zero forced cleanup/overflow. This limitation is
retained. The83original targets remained current; monitor retired with zero
global events, overflow, unmatched starts or unfinished entries.92raw-ancestry
misses still limit deeper attribution. Inclusive, overlapping elapsed times:

| Original context | Send1 | Send2 | Send3 |
| --- | ---: | ---: | ---: |
| Provider composition | 1.654856 | 1.089550 | .985552 |
| Shared tool preparation | 1.614051 | 1.086097 | .981463 |
| Local external catalog | 1.247642 | .414013 | .684662 |
| Both permission payload reads | .897138 | .620473 | .367427 |
| Local-provider builder | .037420 | .001276 | .001776 |
| Configuration capture | 1.703063 | 1.243197 | .971674 |
| Nested MCP maximum | 1.378437 | .603273 | .514787 |
| Non-MCP capture worker | .285242 | .605918 | .448519 |
| Five hook-current contexts | 1.596547 | 2.133842 | 3.082386 |
| Actual hook state reads | .082659 | .076663 | .076575 |
| Hook default path resolution | .219344 | .429573 | .428778 |
| Prompt-history persistence | .384663 | .334634 | .319255 |
| Scoped run-log binding | .507633 | .299105 | .106503 |

These nested times must not be added. Most hook contexts record70full raw
checks, whose total elapsed varies roughly.06-.54s; one records128checks.
Neither contention nor a scheduling/native-I/O cause follows from those totals.
No new cache, skipped permission boundary or weak durability is selected.

The quiet record intentionally has no stacks or detailed main-call census.
The existing detail locates the first synchronous receipt's92.124ms inside
receive_console_visible_intent, in92.321ms dispatch; action returns104.084ms.
Later receipt callbacks take3.036/4.022ms. All return accepted at original line563,
with no selected descendant events inside the first receipt. The next smallest
existing-observer extension is original ConsoleRuntime.accept_received_intent;
phase heartbeat maxima cannot identify its internals or explain typing stalls.

Source review preserves OPT33: secure_private_directory has a separate ancestor
walker, not task22's prepared parent walk. Per-component runtime discovery may
repeat full native parent-pin validation. This is separate layered checking,
not general raw-check recursion. Measure existing versus creation paths before
considering finite sharing; retain actual mkdir/chmod/postcondition gates and
custom/actor/native lifetime behavior. The initial pin-miss hypothesis is
superseded by this direct separate-walker finding. OPT34 retains the unselected
existing-first warm lock alternative. The review list now has34stable entries.


Task27 supported-host qualification: CI run37739739905 on integratedb4a reports
mcp-registration107/107PASS on each Windows/Linux/macOS host, including all22new
lock-stream controls alongside the previous85cases. Standard JUnit supplies the
host evidence. Task34563.27 is Done within its bounded scope; main task34563 and
latency qualification34563.4 remain open. Separate stock-skill and Context shutdown
DB-operation/path-lease failures remain with the integration owner. No main Send,
paint/heartbeat or whole-app lifetime acceptance is inferred from this task's green
qualification.


### Isolated parent-metadata census and default-off control, 2026-10-08

The optional observer attaches only to the existing warm-hook read in
`test_raw_lock_stream_preparation`. It observes six original synchronous bodies,
with 21 fixed scalar aggregates and 128 active slots. Local START/RETURN events
are supplemented by code-filtered global PY_UNWIND because Python 3.12 cannot
register UNWIND locally. It retains no observed arguments, paths, results, frames
or native handles. Global events are therefore nonzero during this read, then
zero after retirement. The receipt checks selected defining bodies/source files;
it does not establish every installed instance-bound dispatch join. Review
narrowed the final receipt wording to state that limit explicitly.

`raw-parent-metadata-1`: 1 PASS, 7.110s driver / 3.586s pytest.
`raw-parent-metadata-control-1`: the exact same isolated node with the diagnostic
disabled, 1 PASS, 7.484s driver / 3.442s pytest. Both retain original payload,
lock, descriptor and lease assertions; audited product/test source is unchanged
during each run, and normal native retirement proves tree/Job/identity/pipes/
profile removal with zero forced cleanup, identity overflow or lookup races.
The later receipt-label clarification changes no observed target or callback.

| Original counter | Instrumented | Diagnostic off |
| --- | ---: | ---: |
| Parent establishment | 1 | 1 |
| Descriptor closes | 34 | 34 |
| Whole-read native open attempts | 1,375 | 1,008 |

The control matches the previous multi-node task27 warm-hook count. The extra
367 attempts in the instrumented sample remain an observation-associated/state
gap, not an established ACL, admission-confirmation or direct callback cause.
Callbacks contain no native operations; source hashes are read outside counting.
All ten shared product/fixture hashes match the earlier task27 batch. Existing
admission evidence has time-sensitive confirmation, but no receipt proves it
caused this difference. Do not use this census as an observer-free workload count
or infer performance improvement from these two elapsed samples.

Within the instrumented sample alone, 64 parent checks total .225993s: named
stat .207486s, retained fstat .015998s. Named calls account for 442 native open
attempts; retained calls account for zero. Both final `_stat_handle` groups total
.030495s, with .017863s in nested security queries. These inclusive times overlap
and must not be added. Thread CPU readings are coarse, not zero-cost evidence.
The other 933 opens reconcile the total exactly (868 returns, 65 unwinds). All
330 global unwind callbacks are accounted for: 65 selected, 265 unselected.
No overflow, unmatched/unfinished entries or exit-order mismatch occurred;
monitor ownership and registrations retired.

OPT25 stays deferred: even this instrumented final-metadata ceiling is small,
and it would not remove named path walks. OPT33 still requires its own eligible
component/effect partition. Source review also identified conditional repeated
default-root filesystem selection (OPT35), outside the explicit-directory test.
The review list preserves these limits rather than turning them into promises.
No further full Send run or product optimization follows from this diagnostic.


Source follow-up selects OPT36 for bounded task34563.28 qualification. The stock
config posture helper loops over `_posture`, which invokes the existing native
snapshot on a singleton; the complete requested tuple can use that same helper
once. This consolidates overlapping work without a new cache or authority
record. Preserve OSError fallback, empty/ordered/duplicate/missing results and
original memo invalidation/creation brackets. The bound companion path remains
unchanged. Existing b4a detail has 19 default-hook getter spans: their nested
get_user_data_dir bodies take roughly .033-.246s while nested companion checks
take .0006-.0055s. These nested elapsed values expose a remainder but do not
record the return branch; they do not prove memo membership or expected savings.
Use a direct original-body native count/retirement control first.


Task28 causal baseline `user-dir-posture-red-1`: one intended failure only at
`126 == 2 * 10` native-open assertion, after exact scalar-equivalent ordered
stamps, duplicate positions, original callbacks and physical invalidity of all
126 returned HANDLEs passed. Eleven requested entries span ten distinct native
tree nodes but rebuild eleven trees (63 accumulated nodes). Recorded native
starts: 126 opens/NTFS checks, 315 identity queries, 63 final metadata/security/
token-owner queries. Driver5.250s/pytest2.057s, unchanged source/observer hashes,
normal native tree/Job/identity/pipes/profile retirement and zero forced cleanup,
overflow or lookup races. The reviewed implementation now uses the existing
shared snapshot; GREEN and supported-host qualification are still pending.


Task28 native GREEN/controls: `user-dir-posture-green-1` confirms 11-to-1 tree
observations and 126-to-20 actual opens with identical requested stamp order,
duplicates and exact descriptor bytes; every actual HANDLE is positively closed.
The original tree primitive schedule also passes. That first 8-case batch has
7 passes and one new fixture error before the changed-state read: WindowsOS
fchmod accepts only private modes, so fchmod0755 is not a valid mutation tool.
The fixture now uses the existing native DACL replacement, retaining the same
inode, actual public-read exposure, original resolver count and private FD restore.
Product code and assertions were not relaxed.

`user-dir-posture-controls-1` passes the corrected memo control plus seven
original controls: exact ACL changes, replacement after child observation,
unavailable/changed related evidence, actual uncertain-close custody and bound
profile directory creation/verification. Driver15.109s/pytest11.347s. Across the
two bundles, all seven new and eight existing distinct controls pass. First
GREEN driver13.359s/pytest9.531s. Both preserve audited source hashes and normal
native tree/Job/identity/pipes/profile retirement, zero forced cleanup/overflow/
lookup races. No deadline or callback is substituted to obtain passing results.
The original uncertain-close test deliberately retains its exact failed handle,
then reclaims only that test-owned handle; ordinary OSError-to-None fallback
remains the pre-existing policy, not a claim of successful resource retirement.

Product change is five additions/two removals in `_user_data_dir_stamps`; config
SHA256 d7b3aae6f709aa18b30ad1113f4c681400e4b858158c68d62494fb2fa274e846.
Every other config AST statement is unchanged; changed-helper lint is clean
with the same 56 pre-existing file diagnostics. New tests pass Ruff/format and
parse checks; independent source/plan/test reviews are clear. Task34563.28 stays
In Progress for supported-host/integrated qualification. Original POSIX PERF07
controls still need their supported hosts; no full suite was requested or run.
The review list now has 36 stable IDs and retains OPT36 as implemented with
these remaining limits. Overall one-second Send and 100ms feedback acceptance
remain unmet.


Task28 whole-operation follow-up `user-dir-posture-whole-hook-1` runs the same
isolated warm-hook control with optional diagnostics disabled on saved435309614e.
It passes1case,7.234sdriver/3.593spytest, source hashes unchanged and positive
native retirement with zero forced cleanup/overflow/races. Whole-read attempts
are952 versus1,008 in the prior default-off control:56fewer, about5.6%, while
parent establishment1 and descriptor closes34 remain identical. This keeps
the helper126-to20 reduction separate from the smaller whole-domain result.
The two single pytest durations (3.442s before,3.593s after) establish no latency
improvement. Most operation work remains; integrated Send is still unqualified.

OPT37 records a distinct unselected scheduling lead. Source review finds the
legacy hook controller already defers unissued indicator refresh during Send,
coalesces an existing refresh flight and replays after settlement. Two running
worker labels therefore do not establish redundant full reads. Early receipt
returns through runtime custody before that legacy dispatch flag is set. Any
new alignment must use exact existing runtime/session ownership, preserve fresh
Send authority and started native readers, and be bounded to initial snapshot
work rather than an arbitrary approval/provider wait. Visit misses can reconcile
disabled/removed grants; this is not a disposable-cache authority shortcut.
No change is selected without actual early-route overlap evidence.


### Existing b4a hook interval partition (2026-10-08)

Reanalysis of the saved `lock-stream-integrated-detail-1.spans.json` and its
original probe stage times; no new application run or observer. Root and the
baseline lane independently reproduced these totals. Each Send contains five
nonoverlapping original `HookPermissions._current` intervals before provider
entry. All 15 contain exactly one same-thread config entry/first-yield pair,
default-path lookup, subsequent hook raw-scope entry/first-yield pair and state
read. The following intervals do not overlap:

| Interval, seconds summed across five contexts | Send 1 | Send 2 | Send 3 |
| --- | ---: | ---: | ---: |
| Config snapshot entry to first yield | .813687 | 1.118458 | 1.776939 |
| Default hook-path lookup | .219344 | .429573 | .428778 |
| Hook raw admission entry to first yield | .184013 | .180155 | .245701 |
| Hook admission complete to state-read entry | .271391 | .303288 | .532488 |
| Actual state read | .082659 | .076663 | .076575 |
| State-read return to hook-context return | .024868 | .025144 | .021286 |
| Other connecting gaps | .000584 | .000561 | .000619 |
| **Whole hook-context lifetime** | **1.596547** | **2.133842** | **3.082386** |

The hook context's own yield was not observed. Its entire post-read tail includes
snapshot construction, caller-held time and cleanup; caller-held time is therefore
bounded above by 1.56%, 1.18% and .69% of the context totals. It does not explain the
large lifetime. Fresh config snapshot preparation is the largest measured region.
The admission-to-read gap contains private lock-stream establishment and lock
acquisition; the existing trace does not separate them or prove contention.

Separately, the attributed original full raw checks number 350/408/350 and take
.402730/.550966/.979359 inclusive seconds inside those contexts. The immediate
hook-runtime discovery branch accounts for 125 checks each; config/runtime discovery
for 55/91/55. Those counts do not identify the secure-directory walk's eligible
subset. The admitted wrappers' second checks take only .029266/.036912/.045058
seconds per Send. Do not prioritize that small duplication as the main delay.
These nested raw durations must not be added to, or subtracted as nonoverlapping
parts of, the table above. The 92 raw-ancestry misses limit caller attribution,
not the directly paired interval table.

Source review keeps OPT33 deferred: an establishment walk would need the exact
selected-directory pin, full actual-FD-to-pin/final-custody validation before its
close, and original full checks at mkdir/chmod effects. The existing parent-walk
wrapper cannot be inserted unchanged. OPT38 records the separate scalar posture
fan-out in hook visit stamps; its cost is unmeasured. Both remain in the user's
optimization review list. This historical b4a analysis precedes task28 and the
integration lane's later cleanup fixes; it is neither a current latency result nor
a savings estimate. Main Send and input/render acceptance remain open.


Task28 supported-host/integrated qualification is complete on integrated commit
`82347279ee1827311e21460e79a2cf55d08a3aae`, CI run `37749712963`.
Root independently read the downloaded `user-directory-posture.xml` reports:

| Host | Passed | Platform skips | Failures / errors |
| --- | ---: | ---: | ---: |
| Windows 2022 | 7 | 12 original POSIX-only memo cases | 0 / 0 |
| Ubuntu 24.04 | 16 | 3 Windows metadata/DACL cases | 0 / 0 |
| macOS 15 | 16 | 3 Windows metadata/DACL cases | 0 / 0 |

All seven new cases ran on Windows; the four portable new cases and all twelve
original PERF07 cases ran on both POSIX hosts. Artifacts are under the integration
lane's `deepseek-uat/ci-823472-context/context-and-skill-lifetime-{host}` directory.
This is the selected directory qualification step, not a claim the wider workflow
passed: separate Linux skill/context lifetime failures remained in that run.

Root also independently read `integrated-directory-count-green-1.xml`, source and
custody receipts: 8 passed, 0 skipped/failed/errors, 18.289s JUnit/24.187s driver,
HEAD and sources unchanged, normal native retirement, no force/overflow/lookup
races. This confirms integration with the count-shutdown change without importing
those unrelated changes into task28. Scoped static/review evidence remains above;
product bytes were not changed after it. Task34563.28 can close under existing
ADR-225/126. Main Send-under-one-second and input/render-under-100ms goals remain
open; no new whole-Send latency claim follows from these targeted checks.


### Config-entry partition inside the saved b4a intervals (2026-10-08)

Root and the baseline lane independently partitioned the same 15 original config
entry-to-first-yield intervals. Each contains exactly three nonoverlapping
same-thread raw-scope entry intervals; their full yielded lifetimes would overlap
and are not summed here. This uses existing saved events only, without another
native run or observer change.

| Nonoverlapping portion, summed seconds per Send | Send 1 | Send 2 | Send 3 |
| --- | ---: | ---: | ---: |
| Three raw-scope entry intervals | .354701 | .556913 | .931842 |
| Second scope yield to third scope start | .289124 | .457623 | .684550 |
| Before first raw scope | .107086 | .041933 | .066793 |
| First scope yield to second scope start | .004655 | .006963 | .008296 |
| Third scope yield to config snapshot yield | .058121 | .055026 | .085457 |
| **Total config preparation** | **.813687** | **1.118458** | **1.776939** |

The first outer raw admission accounts for .335618/.534387/.850807s of the first
row. Both small nested admissions together take only .019083/.022526/.081035s;
removing nested guards would not address most of this region and remains
unselected. The largest remaining gap contains config lock-stream preparation,
lock acquisition/checks and selector revalidation before the guarded config read.
The final gap includes actual config reading and snapshot construction. Individual
lock wait, parser and lock-stream timings are absent; no contention or pure-I/O
attribution follows. No `get_user_data_dir` or `verified_user_data_directory`
event occurs on the same thread inside these config intervals. The separately
measured default hook-path getter runs afterward.

The next source review therefore targets the original outer admission and
lock-preparation boundaries. These are historical measured regions, not projected
savings. Subsequent directory and lifetime changes require a new integrated quiet
sample before current performance claims. All native runs remain sequential.


### Immediate Send feedback and submitted-text visibility (2026-10-09)

TASK-34601.1 pauses further latency reduction. The existing status chip now
shows `Sending…` immediately upon local receipt, then `Waiting for reply…`,
and `Streaming reply…` once resident reply output exists. Tool setup/activity
and user decisions retain their specific labels. Collapsing status retains the
active feedback; completion and cancellation clear it. These labels do not assert
remote receipt. Existing durable acceptance, save-failure draft retention, and
temporary-chat behavior are unchanged (ADR-225).

The current reply check reads resident active transcript nodes and pending chunks
without copying history, materializing/persisting the stream, or introducing a
second state owner. The screen uses its existing received claim through promotion
to avoid an empty status interval before the run starts.

Pre-rebase qualification at be0701398eeb plus the feedback diff:

- 42 unique targeted feedback cases passed across `send-feedback-ui-2`, `-3`
  and `-4`. The first run of this set exposed outdated label expectations and
  incorrect new-test setup; corrected cases were rerun. The new UI test file
  uses the existing bootstrap-profile marker so it also runs independently.
- The original held-read Enter/button checks supplied `Sending…` frames in
  38.9/75.6 ms; subsequent typing supplied frames in 21.4/23.4 ms. Mounted tests
  cover the preparation-to-provider wait, expanded/collapsed status, approvals,
  cancellation, navigation, repeat input, and save-failure draft retention.
- Source-only independent review found no actionable issue. Touched tests pass
  Ruff; product diagnostics are pre-existing (2 agent, 43 screen, 169 store,
  0 status-chip). No new diagnostics were introduced; F811's embedded source
  line changes with the shorter screen. Existing screen-size debt is not waived
  or increased.

One sequential ordinary-profile probe observed natural transcript cells for all
three submitted messages, retaining original preparation and file-backed saving.
The external observer uses the exact message ID/visible transcript region, so
text remaining in the composer cannot qualify.

| Seconds from original Send action | First Send | Next Send | Third Send |
| --- | ---: | ---: | ---: |
| Preparing feedback frame | 0.0365 | 0.0261 | 0.0541 |
| Submitted user text frame | 1.9599 | 1.4955 | 1.9106 |

Receipt: `send-display-feedback-1/{run,frames,probe}.json`, under the task's
`claude-watch-final-gate-review` artifact directory. Parent/child tests passed,
all three turns settled and linked responses, sources/HEAD were unchanged, and
the process guard detected no concurrent native run. Observer SHA256:
`adcc61b94cffd0037b1caf508b0d53f6630d601bbc8700cb88c2e9605cf356c8`
(archived as `send-display-feedback-observer.py`).

Limits: one cold and two subsequent samples, headless natural compositor frames,
not physical terminal flush or a latency distribution. This instrumented run is
not a new quiet whole-Send benchmark and does not replace the earlier 2.718 s
warm adapter-entry result. Provider responses are stubbed. The first stub reply
was observed at 6.4067 s; later reply frames were not observed, so the combined
`message_frames_complete` flag is false. All three user-text frames are positive
observations; no real-provider first-token or later-response display claim follows.
Post-rebase qualification must be recorded separately. Further optimization
candidates remain in `console-optimization-review-list.md`.
Post-rebase integration preserves the landed captured-press acknowledgement:
`Sending…` and its pending USER row paint before received-custody preparation,
then `Preparing` reflects the real held configuration reader. The same press
snapshot/body/revision reaches the received owner; later draft edits survive.
The exact dispatching acknowledgement may pass its own disabled-composer
presentation, while unrelated acknowledgements and every source, setup,
recovery and authority gate retain their original refusal.

`post-rebase-smoke-3` on 1eddc7a4a8 passed all receipt, captured-draft and authority
cases. Its two failures were the earlier stage-specific `Preparing <=100ms`
assertions: natural `Sending` frames were 87.21/92.11ms, pending USER frames
86.88/196.47ms, and later held-reader `Preparing` frames 316.38/356.43ms
(Enter/button respectively). The Enter pending row met the landed <=100ms
contract in this sample; the button result has no such claim. Typing mutation
was 2.55/3.16ms and its supplied frame 11.78/8.52ms. The corrected test keeps the
100ms immediate Sending and input gates, the original 0.5s held-reader budget,
natural Preparing while held, and exact pending-row linkage; it records these
endpoints separately. These remain headless supplied-frame observations, not
physical terminal flush or real-provider first-token measurements.
