# Finite Console activation preparation

Status: implemented and scoped integration verified; broader platform/performance acceptance remains open.
Task: [TASK-34563.7](../../../backlog/tasks/task-34563.7%20-%20Share-finite-activation-preparation.md).
Spec: [accepted Send preparation architecture](../specs/2026-10-06-console-send-preparation-architecture-design.md).

ADR required: no new ADR; existing ADR-225 applies.
ADR path: backlog/decisions/225-console-send-preparation-and-io-ownership.md.
Reason: directly implements finite domain-owned preparation through the existing validated control-observation API. It changes neither the authoritative owners nor public storage/activation contracts.

## Measured problem and chosen boundary

Original Console measurements find six RecoveryAdmissionGuard execution entries per Send, each .184-.568 seconds. They span real worker/task and effect boundaries and remain independent. Within one execution_scope, source-scope, startup and each owner permission currently rediscover validated control records and the registry.

Reuse bootstrap._control_observation(root) for exactly one synchronous execution_scope admission before its yield. It already reads validated records and registry and rechecks named identities, stamps and ancestors at exit. Exit that observation before yielding an allowed decision; any completion failure refuses the body. Retain acquire_storage/StorageLease.execution_context(path), exact per-target gates and every real ActivationStore requirement/approval read. No control-observation handles/data survive the allowed yield or transfer into another admission, thread or task. The actual StorageLease retains its existing guarded-body lifetime. No cache, new scheduler, lease batching or context bag.

## API and compatibility

Extract the existing activation_permission record projection into a private _activation_permission_from_records helper in activation.py. Preserve its exact candidate selection based on the config selector; generation_witnesses uses different overlap rules and is not an interchangeable oracle. Public activation_permission retains its signature and original standalone read route. The stock execution_scope branch composes source/startup/owner projections through one existing observation. Run identifier validation and the existing startup projection separately for every owner in original order: its pending-record branch can perform live binding/root checks, so only records/registry data is shared, never the startup boolean.

Only unchanged directly consumed/bypassed stock callbacks and readers use the shared branch. Supported custom public permission/source/startup/read callbacks retain the original live route, argument shapes, short-circuit behavior and errors, without eager shared reads or catch-and-retry. Record originals in their defining modules where needed; avoid import-time capture of an already customized callback and avoid a generic provenance framework. Review defaults as well as function/body identity where sharing bypasses their ordinary meaning. Preserve the existing empty-owner behavior.

Do not move all admission work out of the worker or remove nested guards. The existing source, selector, pause, per-owner and physical-native controls are constraints. Sharing projection inputs grants no execution permission.

## Lanes and ownership

- Shared preparation: only tldw_chatbook/Backup_Recovery/activation.py and bootstrap.py; finite implementation after root records RED.
- Controller/provider integration: read-only caller/fallback review and existing integration-control selection. Consumers already enter execution_scope; no controller/provider product rewrite is necessary for this slice. No competing source edits.
- Baseline verification: only new Tests/Backup_Recovery/test_activation_preparation.py; original count, freshness, custom-route and real refusal controls. No native/test execution.
- Root: sole integration owner, this plan/task/report, sequential RED/GREEN and native runs. Existing tests stay unchanged unless a concrete fixture defect or intentional assertion update is reviewed.

## Work and verification

1. Add original-body count controls using real retained storage admission and local passive code observation. One/two-owner stock execution_scope must observe one control-record and one registry preparation; preserve each real owner verdict. Establish RED on unchanged product first.
2. Cover real pending/generation/registry changes, denied owner approval, later-entry freshness and exact retained path/source. Mutation during a shared observation must refuse before the caller body; a second owner on the same path isolates freshness from new lease acquisition. Preserve custom callback/read signatures and refusal semantics. Reuse existing native observation/refusal fixtures rather than creating another harness.
3. Implement the finite projection extraction and narrow stock branch. Root and integration lane review source semantics, completion failure handling, empty-owner behavior and original fallback.
4. Run new controls plus existing activation-binding and generation-observation controls. Then run selected stock nested-agent, paired-owner, fresh-intake, retained-selection, model-worker lifetime and bridge cleanup controls sequentially with original deadlines. Because execution_scope is shared, include the relevant MCP/RAG independent-source/lifetime routes without running their full suites.
5. Once source and tests are ready, root runs final integrated targeted checks, scoped lint/format and source review. Any final native app sample runs sequentially in a coordinated window. Preserve every failed/incomplete receipt and diagnostic limit. Count reduction is not a subsecond/100ms claim; those targets remain independently measured.
6. Record actual outcomes, source hashes, retirement and remaining gaps in the phase-attribution report and task notes, then commit locally. No push/merge/full-suite authorization is inferred.

## Review state

Both source and integration reviews agree that this single-admission boundary is viable and that acquire_storage(related_paths=...) cannot replace the independent leases: its borrowable execution selection remains the primary path. Root retains the precise final compatibility and drift checks before implementation. The separate native-commit/early-receipt draft is outside this slice.

Original RED: one owner performs 3 control-record / 2 registry reads (138 native opens), two owners 5 / 4 (260 opens); both real bodies succeed and retain the original lease. Both fail the new 1/1 requirement. Baseline original controls: 87 passed, one existing Windows raw-chmod privacy expectation failed, two POSIX cases skipped. A separate passed DACL control left its test ancestor private to the contained identity, so outer temporary cleanup failed; preserved as a baseline limitation. No source/deadline change or waiver.


## Integrated result

Root integrated the two source files after the original count RED, mechanical extraction checks and independent source review. Final checks after native observation completion revalidate selector, exact retained lease selection and direct callback inputs. Shared eligibility is bounded to 1..64 ordinary owner strings; larger/custom/empty inputs preserve the old route. Final integrated scope: 129 passing cases, two original POSIX skips and two explicit existing Windows exclusions. The three MCP/RAG fixture failures reproduced unchanged and passed after test-only Windows TOML path encoding corrections. No guard, deadline or approval requirement changed.

The final observer-free app sample completed three linked replies/traces with no pending checkpoints in 10.304/9.710/8.964 seconds to adapter entry. This proves neither a matched latency gain nor the subsecond/100ms targets. Source hashes and all final contained native retirement receipts are current and normal. Pre-existing lint/format findings, the original DACL fixture residue and platform gaps are recorded in the [phase-attribution report](../../Development/2026-10-06-console-send-phase-attribution.md). No push, merge or full-suite run.


## Guard-level continuation (2026-10-07)

The original 60-target `agent-preparation-spans-current-1` observer at
`7d150a5cf5` records six independent guard admissions per Send, with first-yield
costs totaling 1.10, 1.06 and 1.09 seconds. The observer is diagnostic and does
not establish a whole-Send speed improvement. Each admission still invokes
separate control observations for its related source paths. Extend this task
under the same ADR-225; AC4 records the additional outcome.

Preserve acquisition and refusal order. A private finite activation preparation
scope may lazily share the existing checked control observation per bootstrap
root while the recovery guard walks its original sources. Each source retains
its actual lease selection, source-scope decision and every startup/owner
approval. Complete all control observations, then recheck every participating
lease and selector before the guard publishes execution state or yields to its
body. No observation survives that boundary, a nested admission, another actor
or a later call. Direct execution_scope and custom callbacks retain their
ordinary route; unsupported batch inputs cannot grant authority.

Ownership remains disjoint: shared-preparation source lane owns activation.py
and admission_runtime.py; baseline lane owns only the new
Tests/Backup_Recovery/test_activation_guard_preparation.py; integration lane
reviews call order, compatibility and tests; root owns final integration,
sequential native runs and this plan/report/task. Root records original 2/2
control-read RED before source edits, then runs the new controls together with
original activation, guard and relevant agent/MCP/RAG entry controls. A final
whole-Send run reports raw timing and retirement separately from read counts.

### Guard continuation outcome

Root took product ownership after the shared lane's read-only analysis, with no
concurrent source edits. Baseline owned the 17 new guard cases; integration
owned the narrow original observer translation and final review. Final root
integration passed all 65 selected cases after both lanes were frozen. The
mixed fallback regression reproduced a final-lease-check omission before its
correction. All observations retire before all successful source witnesses are
rechecked; the normal helper ends before guard yield. Independent reviews and
scoped lint/format/whitespace checks pass. The phase-attribution report retains
actual 2/2 to 1/1 read counts, 216 to 145 native opens, failed-run history and the
bounded PID-history limitation. ADR-225 remains the governing decision.
