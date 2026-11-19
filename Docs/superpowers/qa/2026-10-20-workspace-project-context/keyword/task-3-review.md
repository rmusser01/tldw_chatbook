# K3 task review

## Spec compliance

**Issues found.** The implementation covers the requested K3 lifecycle, runtime composition, schema/membership changes, native lifetime ownership, and binding-wide v2 Clear refinement, but two lifecycle interleavings violate the revocation and exact-job management requirements (I1 and I2 below).

Reviewed source: e65c6a033e0307cf643d3e0c05d4306d874ebaf6 -> 7f5255e2706ac1d6a83803bd4574f78d293cf6a7. This is the K3 task gate, not a merge or whole-branch review. The brief and accepted clarifications take precedence over implementer rationale. Root-owned unstaged metadata was excluded from this gate.

## Strengths

- Native retirement is represented separately from coroutine completion: service.py:580 joins the actual writer, requires DB owner-thread closure and scanner retirement, and only then releases the native capability and retained admission. Tests/Workspaces/test_project_index_service.py:294 exercises a held real native connection and repeated cancellation.
- lifetime.py:115 supplies creation-time handle_list/pass_fds inheritance; closing the POSIX parent reference does not explicitly unlock the shared open description. Tests/Workspaces/test_project_index_recovery.py covers parent death, Windows suspended child creation before Python/handshake, eventual native child retirement, and hardlink refusal. These tests support the chosen crash-restart model rather than relying on elapsed time or PIDs alone.
- runtime.py:78 acquires real storage/activation admission before freezing selector/root/installation/generation identity, and uses per-operation/execution pins rather than an idle permanent lease. isolated_restore.py:33 delegates verification to the existing owner. app_lifecycle.py:698 closes the owner without requiring a Console.
- service.py:225 implements an explicit binding-wide Clear: revoke first, acquire stable lifetime capabilities, validate namespace ownership, preserve control/lease identities, and reclaim in place. Workspace_Project_Index_DB.py:876 and :925 reject legacy/unproven layouts and provide incremental reclamation. Tests/Workspaces/test_project_index_service.py:846 measures an admitted near-capacity generated fixture with old authority/root data and successful rebuilding.
- Workspace_Project_Index_DB.py:603 checks Building state and the exact durable claim before publication. Schema v2 adds durable authority witnesses and per-refresh membership; legacy claims are quarantined rather than acquiring proof from a new lock. Cross-process mutation tests at Tests/Workspaces/test_project_index_service.py:358 explicitly reject held publication after detach/exclude/retarget/revoke.
- Scope stays within K3: the package contains no tool/UI feature, Library selection changes, Media/Notes creation, watcher, or automatic Resume. The listed notification seams for access/exclusion edits already funnel through save_runtime_binding; their omission as separate changed hunks is therefore not a missing implementation.

## Issues

### Critical

None identified.

### Important

**I1 - New scanner units remain admitted after another session revokes or pauses the job.** `tldw_chatbook/Workspaces/project_index/service.py:499`.

The admit(directory) callback tests only run.stop and the registry authority revision. Pause/Disable/Clear in another service/process changes the index control scope/job state, but does not change that registry revision and cannot set this service's local stop Event. A concrete interleaving is: a page commits; another session completes Pause or Disable; the scanner requests its next unit. The callback still returns authority, so scanner.py:302-326 sends a fresh Directory request and reads its next bounded batch. Only the subsequent emit hits the durable claim/state check and rejects publication.

This is a new unit admitted after revocation, not a unit already admitted before revocation and not a stale-publication defect (publication is correctly fenced). It violates the revocation contract at the processing-admission boundary. Under the existing registry-then-store fence, the callback should also require the exact scope, active job token/revision, enabled Building state, and live authority before authorizing a new unit. Release locks before worker I/O. Add a focused two-service/process barrier immediately before the next admission; Pause and Disable/Clear must return no authority and cause no subsequent Directory/read request. The existing held-emit test cannot establish this property because its barrier is after the read.

**I2 - A delayed management operation can stop a newer accepted job for the same binding.** `tldw_chatbook/Workspaces/project_index/service.py:199`, called from `:209` and `:320`.

_stop_local selects every run with the same key at the time it executes; it does not retain the job/scope that the management operation validated. _revoke releases its registry fence before calling it, and neither _revoke nor pause serializes that interval with _start. For example, Disable revokes a retired/idle scope and stalls before _stop_local; another local caller observes Off, successfully Enables and Refreshes the new scope; the earlier Disable then sets the new run's stop Event and joins it. The earlier scope CAS was valid, but its later cancellation effect targets a different, newer job. The same selection pattern exists after Pause once its old worker has naturally retired.

Capture the precise affected local run identity/identities as part of the validated transition, then signal/join those identities outside the fence. Do not broaden an old operation into a later key-wide stop. Add a focused barrier between durable revocation and local stop, start a new enabled job, release the delayed operation, and assert that the replacement job's stop remains clear and it completes. The existing Clear/Enable race at Tests/Workspaces/test_project_index_service.py:931 stalls _clear_binding after _stop_local, so it does not cover this earlier interval.

### Minor

**M1 - Completed run records accumulate for the lifetime of the app.** `tldw_chatbook/Workspaces/project_index/service.py:440` and `:580`.

Every explicit Refresh/Resume inserts a _Run in _runs, and retirement never removes it. Subsequent starts and stops scan all historical runs; error records can also retain exception tracebacks and their frames. Retire/prune fully released run objects after any waiting callers have captured their references, or retain only a bounded history if useful. Keep durable job state in the store. This is a boundedness/maintenance improvement, not a blocker for the task's current correctness gate.

**M2 - Reported validation output is not pristine (inherited noise).** `task-3-report.md`, Final commands and source coverage / Limits.

The report discloses one inherited test warning and 16 baseline E721 diagnostics in existing modules. These are not newly introduced K3 findings, and the report explicitly compares the lint baseline. Preserve them as inherited noise in the execution record; do not describe the raw whole output as warning-free. No K3 code fix is requested for unrelated baseline diagnostics.

## Cannot-verify integration gates and limits

- K4 owns requesting-run permission/exclusion-floor ranking and final fresh-content disclosure, including the 10-result/16-KiB cap and smaller run budgets. This K3-only diff does not establish those end-to-end guarantees. K5 owns the user-facing explicit per-local-binding controls. Neither is a K3 missing-feature finding.
- Native Windows evidence is reported; POSIX flock/pass_fds branches were source-inspected only. The three declared Windows skips remain limits (newline filenames, POSIX mode-bit denial without a qualified Windows ACL-denial counterpart, and ancestor rename blocked by retained no-delete-share handles). Do not label this cross-platform execution qualification.
- The reported sampled capacity highwater is observational, not exhaustive proof of all physical peaks. The diff has conservative DB/journal admission and an in-place cleanup path; I did not independently run a full-capacity native quota experiment. Future vector/temp creators must join the same accounting contract.
- Existing storage/activation/backup-owner registration and K1/K2 scanner safety were consumed as prior-task integration contracts. This review did not expand into a second review of those systems. The ownership inventory change describes the new nested namespaces and retained lifetime files.
- The report claims initial missing-service/native RED, but its first summary does not include the exact initial command/assertion transcript. Later specific RED corrections and final selective receipts are described. Controller should retain the original initial-RED artifact or clarification in the execution record rather than treating prose as a regenerated test run.

## Exact checks and evidence discipline

1. Read task-3-brief.md first, then task-3-report.md, then the task-reviewer template. Read the supplied commit/stat/-U10 package in bounded passes. Recovered the small initial output-truncated test span (runtime identity through retirement setup) from the same package; no git diff was rederived. A reference-only parser of that package supplied exact new-file line anchors.
2. Named risk: cross-thread retained pin release and mandatory nested admission. Inspected unchanged activation.execution_scope (activation.py:183-221), StorageLease lifetime/close (storage_admission.py:540-605), acquire_storage's existing admission seam, and participants._core_operation (participants.py:432-439). The retained scope does not carry a thread-local context reset that must run on its creator; no defect found in transferring this pin's eventual closure. Existing independent repository provenance remains active.
3. Named risk: missing access/exclusion notification seams. Checked registry_service.py:set_folder_binding_access and _save_binding_exclusions plus their add/remove callers; both delegate to the changed save_runtime_binding notification. No missing direct notification found.
4. Named risk: whether I1 is a real pre-read boundary or only publication. The scanner diff cuts off _worker and does not include the unchanged scan loop. Inspected only scan's admission-to-Directory/emit sequence at scanner.py:287-371. It confirms a new worker Directory unit follows the service callback before emit; this is the evidence for I1.
5. I2 is established by the changed functions' exact selection and ordering; no repository crawl or hypothetical alternate integration was needed.
6. No tests or suites were rerun. No dynamic reproduction is claimed for I1/I2. Their regression-test recipes above are recommendations for the implementer. The final reported covering outcomes are retained as implementer evidence: storage/recovery 30 passed; final service selection 14 passed plus one observation-deadline failure corrected by the isolated capacity rerun; failed writer dispatch separately passed; earlier scanner/worker selection had three explicit platform skips. No broad rerun was used to replace those receipts.
7. Checkout, source, index, HEAD and branch remained read-only. Only this requested review artifact was written, with explicit UTF-8 via escalated shell. No subagents, network operations, source fixes, commits or metadata closure were performed.

## Assessment

**Task quality: Needs fixes.** The source has strong native ownership, cleanup, and publication fencing, with meaningful targeted tests. I1 and I2 leave processing admission and local cancellation less precisely scoped than durable publication; fix those two lifecycle gaps before approving K3. Minor findings and stated platform/evidence limits do not independently block the task gate.