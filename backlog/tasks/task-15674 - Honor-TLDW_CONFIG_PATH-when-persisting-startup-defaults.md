---
id: TASK-15674
title: Honor TLDW_CONFIG_PATH when persisting startup defaults
status: Done
assignee: []
created_date: '2026-08-12 06:35'
updated_date: '2026-08-12 17:05'
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
- The lifecycle regression drives the real `TldwCli` from mounted startup through approved quit, wraps the app-module shutdown persistence symbol, records effective-path selection as a boolean, and proves the distinct decoy remains byte-identical. It delegates to the current production config persistence function rather than copying the seam.
- The child receives only a minimal process environment plus disposable home/XDG/temp/data locations; model-catalog networking is disabled before import. Diagnostics are limited to booleans, counts, constant messages, and sanitized phase/error labels.
- Mutation evidence before final integration: forcing effective-path selection away from the override produced 1 failed RED; restoring production produced 1 passed GREEN. The mutation-only command excluded the ProductionApp parent conftest so the child lifecycle, rather than parent collection setup, observed the mutation. Rebase did not change the effective-path seam or test behavior, so mutation was not repeated.
- Final-review correction: fetched and rebased the isolated branch without conflicts onto current `origin/dev`; it was 0 commits behind after integration. The exact post-rebase focused selection passed 4 tests in 6.54 seconds with one `RequestsDependencyWarning` about installed HTTP dependency compatibility. No broad suite was run.
- Ruff passed on the new test. `py_compile` passed with output confined to a temporary directory. The shell lacked a portable `python` command, so these gates used the repository virtual environment without recording its developer-local path in tracked evidence.
- Final diff, scope, privacy, credential-value, and artifact audits passed. The branch contains exactly the effective-config design and plan, lifecycle regression, corrected ComfyUI H3 UAT, live-verification lesson, TASK-3401.14, and TASK-15674; `tldw_chatbook/config.py` and `tldw_chatbook/app.py` have no branch-introduced diff.
- Documentation corrections preserve the observed default-config fingerprint drift and precautionary restore while removing the unsupported writer attribution. The lesson now records that fingerprint drift proves mutation, not actor identity. A post-review correction also replaced three developer-local interpreter paths in the plan with portable commands.
- ADR required: no. ADR path: N/A. Reason: regression-only characterization of an existing effective-config boundary; no storage, security, runtime, or cross-module decision changed.
- No new unresolved lesson emerged. All commit references were intentionally omitted after rebase so closeout evidence does not retain stale rewritten hashes.
<!-- SECTION:NOTES:END -->
