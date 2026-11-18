# K5 idle metric alignment review

Reviewed source: `3820fe11572696f62d27291971f5c8d1dba0faec`. This is a bounded read-only audit of the existing final idle receipt and timing contract, not a new source review or test run. I1/I2 dispositions remain unchanged. Final K5 spec/quality gates remain OPEN: the measured performance requirements are not satisfied.

## Finding and exact boundaries

There is no fixture-only expensive status/scope/exclusion lookup in the corrected admitted clock. `test_k5_idle_latency.py:115–125` starts immediately before `service._fence(key)` and ends after its exit. It includes current registry authority/root/chain acquisition and validation, `authority.guard()`, `store.load_status(key)`, exact scope/state checks, the requesting-floor/current-exclusion union and canonical decorated `WorkspaceProjectIndexDB.lexical_candidates`, including its own internal guard, activation/storage admission, transaction, current scope/policy, exclusion-before-ranking SQL and metadata materialization.

These operations are present in production `Workspaces/project_index/service.py:300–321`. In particular, `load_status` is genuinely called by production, despite fetching counts beyond the service's immediate scope/state needs; its query/admission cost cannot be dismissed as an assertion added by the probe. `Workspaces/registry_service.py:2684` also really obtains current binding authority, captures root identity/chain and reads sensitive/exact exclusions. These costs are not an optional test wrapper.

Two small differences are real:

- Production invokes the initial guard before opening the store and entering the fence; the probe invokes it inside the fence. It calls the same guard once, plus the lexical method's own guard. The phase receipt places the initial probe guard around hundredths of a millisecond. This ordering discrepancy does not explain the failed budgets.
- The fixture additionally asserts `status.indexed == 20000` and requires Ready specifically, where production allows Ready or Partial. These are scalar checks of the status already loaded, not additional DB reads. Candidate count assertions occur after the clock. The second perf-counter pair used for lexical instrumentation adds negligible but included instrumentation overhead.

The candidate store/read pin and outer registry scope are opened once at probe line107 and closed after all three workloads. Their open/retirement costs are excluded by the explicit cached-candidate contract. This is not whole service/search latency. The requesting run guard is test-seeded (`Tests/Agents/test_project_index_tool.py:31`, default guard true), although the search authority itself is captured from the real provider; it is not a measurement of every possible real Console run-guard cost.

A faithful future ordering correction is to start the admitted clock before the initial guard, then enter the real fence, do the real current scope/state load and floor merge, call the decorated lexical method, exit the fence, and stop. Keep store setup outside this cached metric as already declared. Move only fixture scalar corpus/count assertions outside. Do not drop the guard/status/fence, replace them with retained values, bypass decorators, or use SQL-only timing. No rerun is justified solely to remove the tiny assertion/order difference, and no pass should be predicted from it.

## Original target versus later ruling

The original spec, `Docs/superpowers/specs/2026-10-20-workspace-aware-project-context-design.md:426–428`, says cached lexical candidate lookup p95 <=500 ms, with live validation and semantic/provider latency separately reported. The historical `k4_lexical_query_profile.py:35–39` timed canonical decorated `lexical_candidates` plus `Rig.fence`; `Rig.fence` (`Tests/Workspaces/test_project_index_service.py:45`) is an in-memory lock. Its status lookup and manually seeded scope were outside the clock. Thus the original K4 <=500 ms receipt did not promise that an actual registry fence, sensitive-exclusion acquisition and service-level status load together fit inside500 ms. Corpus shape, path+text source and authority setup also differ, so these receipts do not establish a like-for-like product regression.

The later `k5-final-idle-correction-brief.md` deliberately broadened the gate to admitted candidate p95 <=500 ms on both retained caches and explicitly made inner lexical diagnostic only. That is a stronger requirement, not simply a faithful repetition of the K4 timer. It is binding until root explicitly amends it. A return to the original narrower target must be recorded as a requirement deviation/amendment, never a correction that makes the six failed admitted results disappear.

## Recorded outcome

`final-idle-latency.json`, qualified by `final-idle-source.json`, contains50 measured samples after5 warmups for every cell on the same retained20k nested and wide caches. Milliseconds, rounded:

| Corpus/workload | Admitted p50 | Admitted p95 | Canonical lexical p95 |
| --- | ---: | ---: | ---: |
| Nested/common | 758.93 | 1173.76 | 577.31 |
| Nested/rare | 528.70 | 754.74 | 296.30 |
| Nested/miss | 479.46 | 546.32 | 200.81 |
| Wide/common | 592.31 | 697.77 | 302.62 |
| Wide/rare | 477.49 | 1139.39 | 379.45 |
| Wide/miss | 443.99 | 521.91 | 180.91 |

All six admitted targets fail. Even under the original narrower canonical lexical interpretation, nested/common still fails500 ms. The other five lexical results do not close the amended gate. Do not subtract independently computed p95 values to infer phase p95s.

The separate20 real public marker calls per cache measured native Capture p95 of280.03 ms nested/349.06 ms wide and whole-tool p95 of2296.65/2469.38 ms. Whole-tool clocks run immediately around `invoke`; production ownership starts with an empty registry and retires at return, with no test outer ownership scope. They also include the capture instrumentation's post-Capture `psutil.pid_exists` retirement assertion and bookkeeping, so label them instrumented end-to-end measurements, not pure uninstrumented timings. Capture's own clock ends before that test proof, while including production worker/pipes retirement. These are separate costs and have no specified500 ms whole-tool budget.

## Recommendation and narrow remedy classification

Use two durable, explicit metric names: (1) canonical cached lexical lookup, measured around the complete decorated public DB method with a current, properly fenced authority; (2) admitted cached candidates, measured around guard plus actual registry fence/status/floor plus that lexical lookup. Report live Capture and whole public invocation separately. Measuring the lexical component must not remove any production admission or fence from execution; it only defines a nested reporting boundary.

My recommendation is to retain the original500 ms canonical lexical target and require an explicit root ruling on the later admitted500 ms extension before optimizing broader authority infrastructure. If root accepts admitted latency as reported rather than <=500 ms for this delivery, record that as an explicit missed-target deviation with the six raw results, user-visible whole-tool cost, and rationale. It cannot be silently interpreted into a pass. The nested/common canonical lexical miss separately needs either a narrowly demonstrated source remedy or an explicit accepted target exception; changing the contract boundary alone is insufficient.

`idle-phase-diagnosis.json` and `idle-diagnosis-{nested,wide}.txt` support separating the costs: current fence/sensitive-exclusion work and activated status load consume substantial time, while common-query eligibility/ranking and canonical activation also contribute. The profiled run has only five measured phase samples, profiler overhead and overlapping cumulative stacks; its durations are diagnostic, not replacement release p95s or additive exclusive costs. Source `Workspace_Project_Index_DB.py:967–998` invokes eligibility filtering before top-k across matching rows. A focused investigation of the nested/common canonical query is a defensible narrow remedy scope, preserving path+text matches, exact scope, policy and mandatory/requesting exclusions before ranking, activation/admission and deterministic order. The existing profile does not prove a particular SQL rewrite will meet the target.

If root instead retains admitted <=500 ms, the evidence requires reducing real production cost, not removing status/fence from the fixture. That is potentially broader ownership/authority work and needs an explicit scoped design/ruling before code, especially any proposal to retain authority/exclusion values across calls. A benchmark-only adjustment is not a remedy for that requirement.

## Limits / declined to judge

No source edits, test execution, rebuild, guard bypass or additional profiling occurred in this audit. OS background was uncontrolled, as recorded; light coordination was allowed. I have not certified any proposed optimization, changed an acceptance budget, inferred a new numeric admitted budget, or accepted a performance exception. Root must rule on the metric contract/deviation and any narrow follow-up; final release gate stays open.
## Root's proposed explicit ruling — reviewer disposition

Root subsequently proposed retaining canonical lexical <=500 ms as the gate, explicitly accepting the unmet stronger admitted <=500 ms extension as a recorded deviation, and authorizing only measured per-row exclusion pre-normalization to address nested/common. This classification is defensible. With that explicit amendment recorded before code, a source-qualified six-cell canonical pass, scoped correctness/security review of the exact remedy and the already required evidence can close this task's gate while admitted latency remains an acknowledged limitation. It must not be labeled a passing admitted metric. The final whole-branch reviewer remains free to assess the accepted limitation independently.

I identify no source/security reason why this latency deviation itself must remain blocking. A per-call normalization of the immutable currently admitted exclusion tuple can be a bounded remedy candidate, provided it preserves `_path_allowed`'s exact separator replacement, strip and casefold semantics; subtree including empty/root, file, name and direct_children behavior; the current plus requesting-run floor before ranking/LIMIT; and all existing runtime fences/guards/admission. No cross-call authority or exclusion cache, raw SQL bypass, or security-boundary removal follows from this disposition. These are review conditions, not advance approval of uninspected code or planned measurements. The gate is still open now.