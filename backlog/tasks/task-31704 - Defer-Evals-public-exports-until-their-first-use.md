---
id: TASK-31704
title: Defer Evals public exports until their first use
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 18:49'
updated_date: '2026-09-05 18:57'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep character probe imports independent of the unrelated evaluation runner and network stack.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Importing the character probe package and every module preserves the existing forbidden measurement-stack guard.
- [x] #2 All three existing public Evals exports resolve to the same canonical classes on first and repeated access, and unsupported names raise AttributeError.
- [x] #3 Affected evaluation behavior and isolated import-closure regressions pass without exemptions or public API changes.
- [x] #4 Importing evaluation normalizers does not load ServerEvaluationsService or httpx; its existing public identity, star import, and unknown-name behavior remain correct.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve fresh-process RED and both traced eager edges: Evals initializer to runner to Chat to Library to SkillsInterop to TLS/httpx; then character_probe.storage normalizer constant to Evaluations_Interop initializer to ServerEvaluationsService to TLS/httpx. 2. Pin isolated public export identity, repeated lookup, star-import, discoverability and unknown-name behavior; defer the exact three Evals exports and only ServerEvaluationsService in the interop initializer. Keep the normalizer constant at its existing owner. 3. Run character probe tests, affected orchestrator/task loader behavior, complete Evaluations_Interop suite, isolated closure guards, scoped checks, parent review, and commit. ADR required: no. ADR path: backlog/decisions/097-boot-budget-ratchets.md. Reason: direct first-use import deferral under the existing ratchet policy, preserving public interfaces and existing guard strictness.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Deferred the exact three public Evals class exports and only ServerEvaluationsService in the interop initializer. The import tracer identified two independent routes to TLS/httpx; removing only the first did not satisfy the unchanged character-probe all-module guard. The normalizer constant retains its canonical owner. Four isolated RED/GREEN regressions pin deferred residency, no-httpx normalization, canonical first and repeated identities, discoverability, star imports, and unsupported-name errors. Complete character-probe and interop suites plus affected orchestrator, runner, and task-loader files passed: 352 passed, six existing feature skips in 22.44s (/private/tmp/tldw-31704-evals-final.xml). Scoped Ruff/format and diff checks passed. ADR required: no new ADR; direct application of backlog/decisions/097-boot-budget-ratchets.md. No guard exemptions, budget changes, or dependency additions.

Parent reviewed the final scoped diff with no actionable findings.
<!-- SECTION:NOTES:END -->
