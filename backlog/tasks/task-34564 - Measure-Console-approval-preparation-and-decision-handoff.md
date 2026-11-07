---
id: TASK-34564
title: Measure Console approval preparation and decision handoff
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-06 06:41'
updated_date: '2026-11-07 06:37'
labels: []
dependencies: []
references:
  - backlog/decisions/355-lightweight-rag-state-path-resolution.md
documentation:
  - >-
    Docs/superpowers/specs/2026-10-05-console-approval-ux-and-responsiveness-design.md
  - backlog/decisions/235-console-approval-interaction-and-feedback.md
  - Docs/superpowers/plans/2026-10-05-console-approval-ux-and-responsiveness.md
priority: high
type: enhancement
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Identify which stages cause pauses before permission cards and after answers so performance changes address measured costs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Native and browser baselines distinguish preparation, input queueing, paint, settlement, tool work and provider continuation using recorded clocks and conditions.
- [ ] #2 Timing records contain no request or conversation content and reject unsupported cross-clock or headless-as-native claims.
- [ ] #3 Warm timing distributions report at least 40 samples with median, nearest-rank p95 and maximum; first-use cases are separate.
- [ ] #4 Private-profile startup and recovery guards remain intact; a refused launch is reported as unqualified rather than a passing baseline.
- [x] #5 Protected RAG state-directory resolution avoids optional RAG/ML runtime imports while preserving all current path-selection, dynamic configuration and permission-protection semantics.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
### Task 1 Establish the baseline and measurement contract

**Backlog:** TASK-34564. **ADR:** ADR-235; existing testing and live-verification lessons govern evidence.

**Files:** Create `Tests/Benchmarks/console_approval_latency.py`, `Tests/Benchmarks/test_console_approval_latency_measurement.py`; record receipts under `Docs/superpowers/qa/2026-10-05-console-approval-ux/`. Reuse the isolation pattern in `Tests/private_profile.py`, production compositor observations in `Tests/Benchmarks/console_character_switcher_latency.py`, native approval journey in `Docs/superpowers/qa/2026-09-19-approval-action-ownership/native_check.py`, and the existing Windows terminal qualification tooling. Do not alter production behavior for the baseline.

**Interfaces:** Produce `ApprovalTimingRecorder.new_correlation() -> str`, `record(stage: str, *, correlation: str, clock: str, timestamp_ns: int) -> None`, and `summarize_approval_timings(samples: Sequence[Mapping[str, object]]) -> dict[str, object]`. The recorder accepts only defined stages/clocks and correlations it minted, not arbitrary event bodies. Summary inputs contain transport, clock_qualified, first_use, feedback_ms and actionable_ms; missing durations are None. Outputs name sample_count, feedback_p95_ms, actionable_p95_ms, the corresponding median/max fields, feedback_budget_ms=100.0, actionable_budget_ms=200.0, missing_boundaries and qualified. The CLI receipt also contains revision and recorded conditions. Consumers keep identical boundaries before and after changes.

The percentile regression's exact numerical assertions are:

```python
samples = [
    {"transport": "native", "clock_qualified": True, "first_use": False,
     "feedback_ms": float(value), "actionable_ms": 100.0}
    for value in range(1, 41)
]
report = summarize_approval_timings(samples)
assert report["sample_count"] == 40
assert report["feedback_p95_ms"] == 38.0
assert report["feedback_budget_ms"] == 100.0
assert report["actionable_budget_ms"] == 200.0
```

- [ ] **Step 1: Establish safe execution.** Reproduce the isolated startup refusal using the shared private-profile launcher and capture its typed cause without user data. Check the selected config/bootstrap roots and interpreter origins before application imports. Once a supported private launch succeeds, run `python -m pytest Tests/UI/test_approval_action_ownership.py -q` from that interpreter as a baseline control. A guard refusal or collection failure stops this prerequisite; it is not an approval test result.
- [ ] **Step 2: Write failing measurement tests.** Add `test_nearest_rank_p95_uses_complete_sample_count`, `test_missing_paint_cannot_qualify`, `test_uncalibrated_clock_pair_is_unverified`, `test_disallowed_content_is_rejected`, and `test_compositor_only_receipt_cannot_claim_native_transport`. Assert the exact 100/200 ms targets and 40-sample minimum, first-use separation, and missing-event failure. Run the new test file and verify failures concern the missing recorder behavior, not bootstrap.
- [ ] **Step 3: Implement the recorder and transparent observers.** Observe complete request, local checks, input before handler queueing, committing handler, actual compositor/native/browser paint, resolver entry, host outcome, worker release, dispatch, result and next model output. Unavailable original grant/backend-start observations remain explicitly missing; do not rename an early dispatch marker to confirmed start. Use bounded local records and sampled UI stacks. Patch the actual dispatched callable or a named injection seam; class reassignment of an @on handler does not replace Textual's captured dispatch function.
- [ ] **Step 4: Run qualified baseline captures.** Add CLI `python -m Tests.Benchmarks.console_approval_latency --transport native|browser --scenario single|batch|raw-deny|large --samples 40 --output <private-evidence-directory>`. Launch the real app and served UI through their existing launchers with a private offline profile and harmless owned fixtures. Native capture uses a real terminal driver; browser capture includes its actual presented frames and clock calibration. Collect cold and warm samples separately. Keep compositor-only characterization explicitly separate from native/browser receipts.
- [ ] **Step 5: Verify and record attribution.** Run the new measurement tests, inspect the timeline and sampled stacks, and state the measured component responsible for each gap. Retain failed attempts as unqualified. If an observed defect needs a repair beyond this plan's exact tasks, update the plan and that task's AC with its reproduction, files and fix before implementing it; do not guess or hide it inside qualification.
- [ ] **Step 6: Close only after qualification.** Update the task with receipt paths, actual conditions, limitations, ADR check and Implementation Notes; check only supported ACs and set Done through the CLI when its DoD holds. Commit only the probe, tests and evidence. Do not claim speed improvement at this stage.

ADR required: yes
ADR path: backlog/decisions/235-console-approval-interaction-and-feedback.md
Reason: Existing ADR-235 governs the approved Console approval interaction and observational measurement; this task changes no authority.

Execution-order ruling:
Ruling: Allow functional UX Tasks 2–5 after the supported private control and reviewed recorder, deferring unavailable native/browser presented-frame baseline qualification to the final gate — the binding spec requires evidence before choosing performance fixes and truthful final qualification, while the plan assumed an unavailable capture capability — cost if wrong: transport measurements may require later UX rework and final completion remains blocked; retain the unchanged source baseline and make no speed claim or speculative optimization.

Current-dev PR3036 integration: follow Docs/superpowers/plans/2026-10-11-pr3036-current-dev-integration.md. ADR required: yes. ADR path: backlog/decisions/235-console-approval-interaction-and-feedback.md. Reason: preserve approved interaction/noncontrolling observations and shared credential masking under existing human-decision authority. Source history/checked qualification is retained above; current acceptance is reopened for actual sequential-base invocation/privacy/UI/native/browser qualification. No cap/clock/guard relaxation or headless-as-native claim.

2026-10-13 current planning preparation only: actual Rules mergea7bde4e5; queue-armed font6ad961cf (not merged),47 source-bound tests; docs-only approval preparation78e5aff9. Follow appended current preparation in Docs/superpowers/plans/2026-10-11-pr3036-current-dev-integration.md. Preserve152paths/134owners/six task histories/ADR235/current open AC. Actual post3023 merged source binding and pre-edit supported launch/original-control/input-clock contract remain prerequisites before functional release. Preserve config-free Rules Stop/chip/custody/provider seams, existing shared CLI/URL redaction, unchanged caps/census190/P0 selection20/fourprocess/sixlanes. BrowserCDP does not qualify native physical input/paint. ADR required:no new for metadata; existing decisions and required behavior amendments retained. No newID/Done/current acceptance/private-code exception/paid/key/native/publication waiver.
2026-10-15 actual-base integration: user deferred #3023; continue on dev1c72cfd48f2bd903093ce6b032684de74c511919 without private #3023 changes. Existing approval decision remaps to ADR235 after a live ref/worktree census (original source221/private229 provenance retained); no new architecture or task ID. Retain this task plan/AC/history and full original approval/privacy/currentness/geometry/timing requirements. First establish the supported private launch, passing original control and reviewed content-free measurement contract before functional edits; Root owns publication and final external qualification.

2026-11-06 owner explicitly authorized Playwright/CDP and QA-only native capture. Guarded capability probes established actual browser Screenshot source/sequence to presented PipelineReporter/FramePresented identity, plus bounded native owned-ROI BitBlt/GetDIBits capture with stale/partial/hidden rejection and calibrated QPC brackets. These synthetic probes remain product_samples=0. Product qualification will use actual private TldwCli and the existing console_provider_gateway_factory to script planning input, actual submit_draft/ConsoleRuntime/AgentService/tool-discovery/review/controller paths and named private Workspace binding; real harmless fs_read/private dry-run writes, raw denial before execution. No fake readiness, resolver injection, backend/Running facts or external model/network. This complete-workload seam is required because a request-only controller fixture lacks real retained tool-activity feedback. Observers must not access the new lazy feedback store or Details/draft helpers before first use. Keep first-use and >=40warm per shape/transport separate, exact-dev5ca baseline in isolated read-only worktree, production source closures and conservative clock/capture bounds. Existing ADR235; QA-only input/observation seams change no authority or lifetime. No product samples until final source freeze and local tests stop.

Measured preparation fix2026-11-07, before implementation: original guarded f3cff direct-container lookup with config+sensitive_paths already resident took1.5782705s CPU1.546875s and added570 dependency names, returning4 private paths with no network/profile violations. It does not explain the fullApp60s binding wait. Implement ADR355: share canonical lightweight RAG state-directory resolution, preserve existing public getter APIs, exact env/current/legacy precedence, coercion/blank/expanduser/relative-path/error behavior and dynamic user-data/config reads. Keep sensitive-path memoization/failure behavior and every authority/storage/recovery guard unchanged. First add fresh-process optional-import blocker and path-parity/currentness regressions, observe RED on original code, then minimally extract/delegate. Run targeted sensitive-path/file-tool/RAG-config compatibility checks, canonical worker rebuild/identity/stdlib closure, diagnostic/CI and boot checks affected by the new import edge. Record fresh original-helper before/after diagnostic and re-run real current-source native/browser qualification; no p95 or60s-cause claim from the isolated probe. ADR required: yes. ADR path: backlog/decisions/355-lightweight-rag-state-path-resolution.md. Reason: shared security-relevant path accessor and dependency-loading boundary, preserving existing data authority.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented the bounded content-free timing recorder and warm nearest-rank summary contract. The contract retains the 100/200 ms budgets, 40-warm-sample minimum, separate first-use cases, and explicit refusal of missing-paint, uncalibrated-clock, mixed-transport and compositor-as-native claims.

The supported private launcher installs the unchanged real-profile guard before redirecting the Windows home. Existing approval controls: 25 passed. Recorder tests: 16 passed. Each targeted run reports one pre-existing Pydantic warning. The failed selection trace and reproducible launcher are retained under Docs/superpowers/qa/2026-10-05-console-approval-ux/; the incident is documented in backlog/docs/lessons-testing-evidence.md.

Partial delivery only: no native/browser timing capture, transparent production-stage observations, calibrated presentation timestamps or overall budget qualification has been completed. TASK-34564 remains In Progress, and its unverified acceptance outcomes stay open. The unchanged product baseline is b0ac6672231e29fb7efa2e9236d2f123c2e0a163. The controller's execution-order correction permits independently specified functional UX work after review, retaining final qualification and prohibiting speculative optimization or speed claims.

Existing ADR-235 (backlog/decisions/235-console-approval-interaction-and-feedback.md) applies. Owned changes are the measurement probe/tests, reproducible QA evidence, testing lesson, execution-order plan and this task record. The delivered foundations passed independent review and a focused refusal-gate fix/re-review at a49161f80d; full qualification remains open.

The launcher now stops unsuccessfully before pytest import when startup admission is refused. A real owned invalid-recovery fixture reproduced RED (1 failed) and GREEN (1 passed); the 16 recorder and 25 approval controls still pass. Refusal-regression receipts and Tests/Benchmarks/test_console_approval_private_control.py cover that boundary.
Current1c72 integration: supported private control/recorder release accepted by Root. Three overwritten original JSON/XML receipts remain a documented retention gap; fresh unchanged25-case control independently verified in E/pr3036-control-recovery-r1, never reconstructed original timing. Current recorder contract assertions pass; actual native/browser samples0/0 and100/200ms/40warm qualification remain open. See owned Task1 report and E/pr3036-current-preedit-r1; ADR235, In Progress.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Renumbering provenance

Originally TASK-34411 in the reviewed approval checkout. Renumbered to TASK-34564 during PR integration onto current dev because older unrelated TASK-34411/34412 already landed. The six approval records moved together to preserve dependency order; original verification hashes/commit references retain their historical context.

2026-11-06 resync: recorder16 and refused-startup1 contracts pass. No qualified native/browser presented-frame timing samples or40-warm distributions exist. Guarded alternative capability probes prepared but unexecuted pending user method approval; user expressly requires timing before auto-merge. Actual Windows execution is separately proven and is not a timing substitute. Existing ADR235/236 applies; no new authority. Evidence Docs/superpowers/qa/2026-11-06-console-approval-resync/README.md. Status remains In Progress.

The owner approved alternative capture methods. Synthetic capabilities are demonstrated: native owned-ROI capture takes about 7-11 ms with stale/partial/hidden controls; installed Edge 156 provides readable trace images uniquely joined to actual presentation events and calibrated clock intervals. These remain zero product timing samples. The request-only fixture was insufficient for retained feedback, and a gateway-wrapper proposal skipped original trace admission; both were rejected before product launch. The replacement uses the untouched stock gateway with an owned loopback scripted HTTP endpoint, real submit/AgentService/tool activity, real private binding consent and no external model. Main and isolated worker protection/source closures are required. Final code is locally frozen for exact-revision product qualification, first-use and 40 warm samples per shape/transport remain open.

ADR355 path-loading follow-up (2026-11-07): extracted one lightweight RAG state-path owner and delegated the two legacy getters plus the protected-path import edge, preserving live getter seams, path precedence and protection semantics. Scope: four production files and two focused test files. Meaningful original-helper optional-runtime-blocker RED and raw-error canary RED became 16 critical passes; affected compatibility produced 72 passes. Eight existing failures reproduce on exact dev8b (seven raw-selection fixture failures and one Windows HOME expectation); one further test stops at WinError1314 symlink privilege, with two optional skips. Only AC5 is supported here. AC1-4, task status, after-probe and actual native/browser timing qualification remain open; no full-app 60-second-cause or latency improvement claim. CI and canonical artifacts are handled separately. Decision: backlog/decisions/355-lightweight-rag-state-path-resolution.md. Evidence and limitations: Docs/superpowers/qa/2026-11-06-console-approval-resync/rag-state-paths-followup.md.
<!-- SECTION:NOTES:END -->
