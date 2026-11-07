---
id: TASK-37204
title: Qualify tool recovery and command-hook contracts on Windows and Linux
status: Done
assignee:
  - '@codex'
created_date: '2026-10-28 17:57'
updated_date: '2026-10-28 18:14'
labels: []
dependencies: []
documentation:
  - Docs/superpowers/plans/2026-10-28-tool-recovery-platform-qualification.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #3284 needs trustworthy cross-platform recovery evidence. Command-launch tests currently expect POSIX behavior on native Windows despite the accepted unsupported-platform contract, and Linux recovery controls have not yet been run.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The reported Windows hook batch has no unexpected failures or teardown errors; only cases requiring unsupported native command execution are explicitly excluded.
- [x] #2 Windows still proves pre-launch refusal and zero leaked custody without hiding platform-neutral hook and recovery tests.
- [x] #3 A Linux runner executes the actual native/fenced recovery and real command-hook controls on the PR head with retained results.
- [x] #4 Any observed Linux regressions are repaired and PR #3284 records accurate platform-specific evidence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: backlog/decisions/163-expanded-console-hook-runtime.md; backlog/decisions/252-tool-availability-refusals-before-hook-preparation.md. Reason: test qualification and CI wiring under existing platform/authority contracts, with no new command backend or runtime policy. 1. Reproduce the Windows batch and trace every failure to its exact boundary. 2. Correct only unsupported command-launch assumptions and strengthen native refusal cleanup controls. 3. Add a bounded Linux PR job using existing CI dependency conventions, run real recovery/command cases and fix observed regressions. 4. Verify targeted tests, lint and task hygiene, document results, review and push to PR #3284.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Corrected precise POSIX command-execution expectations under ADR-163 and strengthened repeated native Windows refusal without root/environment/launch/custody access. Native Windows RED: 20 failures and one teardown error, comprising the original 19 plus an incorrect host-restoration assertion in the refusal control. GREEN: 72 passed, 20 explicit POSIX skips, one existing Pydantic warning, no failures or errors in 141.21s. The extra invalid-replacement skip corrects a false positive. Platform-neutral recovery and predispatch controls remain active. Added bounded Linux/Python 3.12 CI for the actual execution, pipeline, native/fenced recovery, skill and replacement controls, with JUnit retention; Linux execution pending. E9/F lint, formatting, workflow parsing and whitespace checks pass. Independent review found no actionable issue. No new ADR or production runtime change: existing ADR-163 and ADR-252 apply.

First Linux job 40741825139 exited 4 during pytest argument parsing because the minimal dependency set has no pytest-cov but the command included --no-cov. No tests executed; no JUnit artifact was produced. Removed the unnecessary plugin flag while retaining empty addopts. Linux product behavior is not yet qualified.

Linux qualification completed on implementation head 5157f909ac: run https://github.com/rmusser01/tldw_chatbook/actions/runs/40742429112, Ubuntu 24.04 / Python 3.12.15, 142 passed in 59.68s. Downloaded JUnit independently reports 142 tests, zero failures/errors/skips (59.672s). Full real command-execution and pipeline suites passed alongside native/fenced reload, replacement and skill gates. No Linux runtime defect was observed after the CLI setup flag repair. ADR-163 and the original hook spec now record the measured Linux qualification, preserving Windows refusal and process-group limits. Updated architecture, plan, lesson and PR description with exact evidence. Final static/task/ADR/format/whitespace checks pass; independent review found no actionable issue. No full suite or exact Fedora/live-model reproduction was performed.
<!-- SECTION:NOTES:END -->
