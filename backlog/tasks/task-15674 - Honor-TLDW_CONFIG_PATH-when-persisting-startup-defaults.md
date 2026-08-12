---
id: TASK-15674
title: Honor TLDW_CONFIG_PATH when persisting startup defaults
status: Done
assignee: []
created_date: '2026-08-12 06:35'
updated_date: '2026-08-12 16:57'
labels:
  - config
  - privacy
  - regression
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The original generated-video UAT observed a byte fingerprint change in the unrelated default config: built-in default keys had appeared while existing values remained unchanged. Restoring the validated snapshot was the correct precaution, but concurrent activity meant the writer was not identified. A controlled current-development startup-to-approved-quit reproduction did not attribute that drift to this app lifecycle. The remaining task scope is regression-only: lock the effective-config persistence contract and correct the historical causal claim. No production fix is required.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Running the real app from mounted startup through approved-quit persistence with `TLDW_CONFIG_PATH` pointing to a scratch profile leaves a distinct decoy default config byte-for-byte unchanged.
- [x] #2 Defaults needed by the isolated run are written only to the effective profile path if persistence is required.
- [x] #3 A regression test uses distinct profile and decoy default configs and proves no cross-profile write.
- [x] #4 Existing no-override startup persistence behavior remains covered.
- [x] #5 No config values or credentials are emitted in diagnostics.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a real TldwCli startup-to-approved-quit subprocess regression with separate effective and decoy default profiles, deterministic network disablement, and privacy-safe persistence evidence.
2. Mutation-prove the regression against the production effective-config lookup while preserving current production code unchanged.
3. Correct the UAT, TASK-3401.14 notes, TASK-15674 wording, and live-verification lesson to distinguish observed drift from proven causality.
4. Run only touched-file and named config controls, then Ruff, temporary py_compile, privacy checks, and git diff --check.

ADR required: no
ADR path: N/A
Reason: regression-only characterization of the existing effective-config boundary; no new storage, security, runtime, or cross-module decision.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Regression-only closeout; no production app or config code changed.
- Commits `ce5cf39d1` and `e6909deb5` add and privacy-harden `Tests/ProductionApp/test_config_profile_isolation.py`. The regression drives the real `TldwCli` from mounted startup through approved quit, wraps the app-module shutdown persistence seam, records an effective-path boolean, and proves the distinct decoy remains byte-identical.
- The child receives only a minimal process environment plus disposable home/XDG/temp/data locations; model-catalog networking is disabled before import. Diagnostics are limited to booleans, counts, constant messages, and sanitized phase/error labels.
- Mutation RED: 1 failed after effective-path selection was temporarily forced away from the override; restored named GREEN: 1 passed. The mutation command excluded the ProductionApp parent conftest so the child lifecycle, rather than parent collection setup, observed the mutation. Fresh final focused verification: 4 passed in 5.28s with no warnings.
- Ruff passed on the new test; `py_compile` passed with output confined to a temporary directory; branch and cached diff checks passed. Scope and privacy review found only the intended design, plan, regression, UAT, TASK-3401.14, lesson, and TASK-15674 files, with no credentials, private source paths, media, build/cache artifacts, or production mutation residue.
- Commits `15fe69703` and `5c1e15709` correct the UAT and TASK-3401.14 evidence and update the live-verification lesson: fingerprint drift proves mutation, not writer identity. Design and execution are documented in the effective-config isolation spec and plan.
- Added/modified paths: `Tests/ProductionApp/test_config_profile_isolation.py`; the effective-config design and plan; the ComfyUI H3 UAT; `backlog/docs/lessons-live-verification.md`; TASK-3401.14; and TASK-15674.
- ADR required: no. ADR path: N/A. Reason: regression-only characterization of an existing effective-config boundary; no storage, security, runtime, or cross-module decision changed.
- Deviation: the mutation-only collection boundary noted above. Per user instruction, no broad or full suite was run. The existing incident lesson was updated; no new unresolved lesson emerged.

- Post-review privacy correction: the execution plan replaced three developer-local absolute interpreter paths with portable `python` commands. This documentation-only correction resolves the path finding and does not change production or test behavior. Pytest was not rerun; the immediately preceding focused evidence remains 4 passed in 5.28s with no warnings. The final branch privacy and scope audits were repeated after this correction.
<!-- SECTION:NOTES:END -->
