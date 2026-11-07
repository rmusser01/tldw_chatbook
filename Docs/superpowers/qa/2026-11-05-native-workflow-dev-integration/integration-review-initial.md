# Independent current-dev integration review — initial gate

What this change does: It adds native, explicitly approved phased workflows in one shared worktree, with private results, finite accounting, physical activity ownership and inert restart recovery. This integration preserves current dev's extracted Console/review owners and shipped review-boundary schema, and defers UI refresh coroutine creation until a Textual worker actually starts.

**SPEC: CHANGES REQUIRED. QUALITY: CHANGES REQUIRED / HOLD.**

**Critical: 0. Important: 1. Minor: 0.** These counts are confirmed source findings. The unresolved ordinary-turn regression evidence is a separate gate hold, not an invented second code defect.

The frozen source is **not ready to pass the integration review gate or be marked ready on the strength of this review**. Fix finding 1 and review its bounded regression evidence. Resolve or explicitly establish the scope of the adverse ordinary-turn cohort before claiming the affected integration checks are green. The existing draft PR can remain available for discussion.

## Review identity and method

- Base: `3c39215749a96d5c1b8c05972fdd2aeb7588b002`.
- Code head: `5dc21d4b4b10bcaee7908765159d94568d8e5dd6`.
- Read the integration brief first, then TASK-38943, the current integration plan, ADR-285's governing contracts/current-dev amendment and the original feature specification's structural, authority, scope, evidence, accounting and recovery requirements.
- Used `integration-review-code.diff`, `integration-review-focus.diff`, the 85-path focus map and the governance material as the review map. Read connected source in the integration export. A read-only byte comparison found **all 85 focus paths and all 157 changed production paths identical to the frozen Git head**. This verifies which source was read; it is not a test result or a claim to have inspected every line.
- Separately reviewed the frozen two-workflow Windows checkout supplement, its ADR/ruling additions and `ci-longpaths-validation.json`. It is outside the code head above.
- Assumed the documented single executing Console runtime per profile, with finite workflow sizes and parallel native child threads. External editors/processes remain possible and are not constrained by worktree admission.
- No test reruns, product imports, profile changes, Git/index/ref changes, PR/CI mutations or subagents. The only written artifact is this report. Receipt results below are observed existing evidence, not reviewer-executed tests.

## Must fix

### 1. Important — completion replaces the original input evidence with a later input version

**Reference:** [workflow_run_service.py:1264](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.superpowers/pr-prep/native-phased-workflows-2026-11-05/integration/tldw_chatbook/Agents/workflow_run_service.py:1264), within `checkpoint_task` (1264–1268). Connected producers/consumers: `prepare_task_inputs` at 977–1029, `_task_input_result` at 1085–1095, `_fresh_task_result_value` at 1714–1726; [console_workflow_wave.py:724](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.superpowers/pr-prep/native-phased-workflows-2026-11-05/integration/tldw_chatbook/Chat/console_workflow_wave.py:724).

**What this is:** The checkpoint is the durable evidence used to accept a task result, approve phase completion and admit dependent work. Preparation already retains the original aggregate task-input manifest as `tracker.inputs`.

**Problem:** `checkpoint_task` obtains a new manifest of the task inputs after the task returns, then stores it without comparing it to the input version retained before execution. A scoped task can read a declared **read-only** `input.txt`, produce an output based on those bytes, and return while an external editor changes `input.txt` during the provider call. With no required verifier, checkpoint publication saves the new input digest and marks the native successful result completed. `_fresh_task_result_value` then compares the file with that newly saved digest and accepts the stale handoff as fresh. The task did not operate on that input version.

The connected call search found input validation in preparation/launch (`agent_service.py:4826`, `agent_service.py:5205`, `console_workflow_preparation.py:2606`), not between actual task execution and this checkpoint. `_WorkflowWave._settle` proceeds from original owner drain and native result readback to `checkpoint_task`. Physical drain proves the work ended; it does not prove the result used the later input bytes.

**User impact:** A user may approve or continue a workflow whose saved result was derived from obsolete source, even though the retained checkpoint and freshness review say it matches the current inputs. This contradicts the approved rule that observed external drift invalidates affected evidence and that freshness is rechecked before verified completion/dependent work. This is an evidence/approval defect, not a claim that external editors can be prevented.

**Minimal fix direction:** Preserve and compare the pre-execution provenance of inputs the task is not authorized to change at the original completion/checkpoint owner. A changed or unavailable input must not become a fresh successful handoff merely by recapturing its new bytes. Keep intentional task-owned input/output edits valid under explicit ownership semantics; comparing the whole aggregate manifest unconditionally would incorrectly reject normal edits to a file that is also a declared output. Reuse the existing bounded manifest and required-result owners rather than introducing a new executor or authority path.

**Required regression:** Hold an actual task after it has received/read a declared read-only input, change that input before original native return, then let the task finish. Assert that checkpoint/phase review and dependent admission cannot certify the result as fresh. Keep the unchanged-input positive and an intentional input/output edit control. The existing [test_workflow_run_service.py:797](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.superpowers/pr-prep/native-phased-workflows-2026-11-05/integration/Tests/Agents/test_workflow_run_service.py:797) changes the input **after** checkpoint publication, so it does not cover this interval.

This is a source-proven counterexample; I did not execute a new reproducer under the review's no-rerun constraint. The root has forwarded it to the sole writer for reproduction and a scoped fix.

## Connected coverage and conclusions

The following are conclusions about the code paths inspected, not blanket certification of the entire feature:

- **Catalog and recovery:** Read the exact-catalog bridge and row validator, constructor call before general DDL, canonical stage allocation, staging validator's existing transaction/authorizer changes, successor catalog generation, startup recovery selection and inert service `resume`/recovery projection. Shipped review v25 remains, workflow stages are 26–30, old local 25–29 require complete frozen catalogs and a matching stamp, and the bridge adds only the published review artifacts before canonical successors. Unknown catalogs refuse. Existing-owner FULL transactions and rollback remain. I did not find a second bridge or a restore-time scheduler. I sampled the frozen legacy declarations and their integration tests; I did not manually rederive every physical SQL variant.
- **Scope and physical activity:** Read the complete worktree activity owner, task/planning scope guards and connected provider guard sites. Admission uses complete footprints, including aliases/nested roots and runtime-wide opaque effects, before capacity. Child membership retains the claim, and settlement and positive physical drain are separate prerequisites. Task tools intersect captured owners/schema/binding and current permission checks. Local write observations/absent-file preconditions stay in the actual invocation path; virtual CLI remains the installed read-only command set. I did not exhaustively read every filesystem implementation.
- **Current extracted receivers:** Read the changes and connected call paths in actual request execution, agent reply, streaming request, project dispatch, trace maintenance and review delivery, plus source-record consumers. The definition-time records target those modules, not abandoned monolithic implementations. Streaming retains the current Stop/watchdog boundary; unknown usage after Stop is not promoted to successful accounting by worker return.
- **Ordinary review:** Read restrict-only capture, late message/boundary checks, final catalog binding, registry authentication and the actual review composer. The possible-review maximum is set before activity admission; ordinary unrelated input does not receive blanket opaque admission from this classifier. Native acceptance, current registry authentication, permissions and persona/Goal filtering still determine actual tools. Provider construction occurs through original ordinary preparation. Read the current review-child runner and finish code: original source/database scopes return before child finish; notification cleanup ends before self-owned final readback; the original exact registered child owner is selected under the tree lock and activated outside it. The adverse ordinary cohort still limits broader regression confidence.
- **Workflow coordination and accounting:** Read task-input preparation, launch validation callers, original checkpoint/settlement, dependency freshness, selected planning preflight/idempotency/zero-limit factoring, the main execution loop, quiescence/active intervals, release notifications and recovery protections. Required results and checkpoints use the private AgentRuns store; optional logs do not stand in for required handoffs. Existing phase authorization, controls and ledger checks are distinct from stored IDs. The freshness defect above is in this path.
- **Coroutine fix:** Inspected changed refresh scheduling sites, especially the extracted `config_sync.publish_console_presentation` owner. They pass the bound async callable to `run_worker`, leaving coroutine construction to actual worker entry. This matches the retained three-case warning-as-error regression and preserves direct awaited calls.
- **UI/governance/CI:** Reviewed the changed Console attachment/command/refresh boundaries and control evidence rather than independently exercising the UI. The Windows supplement adds only the existing conditional `git config --global core.longpaths true` bootstrap before checkout in the two affected jobs. Its frozen validation says both YAML documents are valid and original job settings are preserved. This is a checkout repair; fresh Windows CI and Windows native workflow qualification are not implied. ADR-285 amends the integration boundaries; a second ADR is not required for implementing those approved amendments. I found no separate plan defect in the sampled requirements.

I did **not** inspect every line of the 120,110-line code package, every historical QA artifact, or every one of the 85 focus files in full. In particular, hooks/process/SSH custody, all UI widget bodies and all native configuration source-record combinations received connected sampling rather than a complete independent requalification. The report does not inherit historical ten-task approvals as a new whole-integration pass.

## Evidence read and remaining gate holds

| Existing receipt | Actual conclusion |
| --- | --- |
| `final-native.log` and `final-native-receipt.json` | 40 passed in 446.58s. Receipt records candidate origins, exit 0 and identical 9,108-file before/after source digests. Covers actual native planning, selected phase-wave paths, archive/restore/rebackup and held review/ordinary-child ownership. |
| `ui-final-attempt2.log` | 20 passed in 445.82s: 17 workflow cases plus the exact three deferred-refresh cases. Brief records warning-as-error flags. This is the stated mounted fixture scope, not a whole-app/live-provider claim. |
| `review-current-attempt12.log` | 22 passed, including actual held native reviewer and adverse original child-scope return. |
| `native-correction-current-attempt2.log` | 1 passed in 116.38s; root confirms bounded fixture timeout correction. The earlier adverse attempt remains separate. |
| `schema-final-attempt3.log` | **59 passed, 1 failed** on the populated legacy fixture's insertion order (`run chain scope mismatch`). It is not a wholly green receipt. |
| `review-native-attempt8.log` | Later 28 passed, including the two populated bridge cases, query-plan/constraint case and 25 current continuation cases described in the writer report. Separate source epoch; not silently substituted for every prior case. |
| `db-recovery-attempt2.log` | 54 passed, including complete private result backup/restore/rebackup. |
| `receiver-schema-guards-attempt1.log` | 14 passed, 50 deselected, with actual extracted module origins printed from the candidate. |
| `remote-bundle-attempt2.log` | 16 passed. The report explicitly supersedes the earlier stale generated-bundle receipt. |
| Six `final-check_*.log` static/census logs | All pass. `static-source-before.json` and `static-source-after-css.json` bind the same 3,228-input aggregate `c3faa7fdac61c8679a11c908e56f6dbb5402403a41e658d629f2835e57a71e96`. |
| `css-sync-final.json` | Existing exit-0 receipt reports all 15 generated sheets reproduce. |
| `preservation-latest.json` | Existing receipt reports original dev/HEAD, all 292 protected files and staged digest `c5fcb00eab4ea09c49d1797567cbd4cc9f0eb9bb8d99cc2355f764638aaa7248` preserved. I did not mutate those surfaces. |

**Open evidence hold:** `final-unit.log` ran no tests because it named the wrong DB test path. The older ordinary-tree selection in `final-unit2.log` has early failures and a hanging teardown; the root's latest update says the owned process is being interrupted for a traceback and its cause is not established. The later selections in that command were not reached. Neither the successful native subset nor source inspection converts this into a baseline pass. Obtain the actual diagnosis and a bounded relevant receipt, or document a demonstrated pre-existing/fixture-only limitation with the affected production path still covered, before closing the quality gate.

**Still pending:** finding 1's actual regression/fix evidence, the writer's final consolidated identities/report, any new delta's scoped review, and fresh CI after the Windows checkout supplement is pushed. Known trailing whitespace in the hooks command executor is a disclosed cleanup item, not counted as a behavioral Minor here.

No full suite, fresh live provider, performance/scale benchmark, Windows native workflow run, multi-process profile scheduling or whole-system qualification was performed or inferred.

Verdict: fix finding 1 and close the adverse-evidence hold before marking this integration ready.

Not checked: tests were not rerun, and coverage was risk-focused rather than line-complete. Risk: the frozen head can certify a handoff against input bytes it did not use; unresolved ordinary-turn failures also prevent a green quality claim.
