---
id: TASK-15674
title: Honor TLDW_CONFIG_PATH when persisting startup defaults
status: In Progress
assignee: []
created_date: '2026-08-12 06:35'
updated_date: '2026-08-12 16:46'
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
- [ ] #1 Running the real app from mounted startup through approved-quit persistence with `TLDW_CONFIG_PATH` pointing to a scratch profile leaves a distinct decoy default config byte-for-byte unchanged.
- [ ] #2 Defaults needed by the isolated run are written only to the effective profile path if persistence is required.
- [ ] #3 A regression test uses distinct profile and decoy default configs and proves no cross-profile write.
- [ ] #4 Existing no-override startup persistence behavior remains covered.
- [ ] #5 No config values or credentials are emitted in diagnostics.
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
- Draft closeout: commits `ce5cf39d1` and `e6909deb5` add and privacy-harden `Tests/ProductionApp/test_config_profile_isolation.py`; no production app or config code changed.
- The regression drives the real `TldwCli` from mounted startup through approved-quit persistence with disposable home/XDG directories, an effective scratch config, a separate decoy default config, a scratch data directory, and model-catalog networking disabled. Persistence ran, selected the exact effective path, and left the decoy byte-identical; effective-profile bytes are not required to change because persistence may be idempotent.
- Mutation verification temporarily made production effective-path lookup ignore the override. The named test failed with `effective_path_selected=false` while mounted and persistence evidence remained true; production source was then restored. The focused final controls passed: 4 passed.
- Evidence was corrected only in `Docs/superpowers/qa/2026-08-09-comfyui-h3-console-generation-uat.md`, `backlog/tasks/task-3401.14 - UAT-end-to-end-ComfyUI-H3-generation-through-Console.md`, `backlog/docs/lessons-live-verification.md`, and this TASK-15674 file. The corrections preserve the observed delta—built-in default keys appeared while existing values remained unchanged—and the exact snapshot restore without assigning an unproven writer. The unexplained mutation remains valid investigation evidence, but the controlled lifecycle does not establish a confirmed cross-profile writer. Prompts, host identity, credentials, media/source identity, config values, real user paths, and raw logs remain absent.
- ADR required: no. ADR path: N/A. This is regression-only characterization of an existing boundary.
- Final task-wide static analysis, acceptance-criteria checks, hygiene, and Done closeout remain pending; status stays In Progress and all acceptance criteria remain unchecked.
<!-- SECTION:NOTES:END -->
