---
id: TASK-24403
title: Fast PR lane preserves required gate and full coverage cadence
status: Done
assignee:
  - '@codex'
created_date: '2026-08-29'
updated_date: '2026-08-29 22:04'
labels:
  - ci
  - infrastructure
  - github-actions
dependencies:
  - task-22250
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the account-saturating full pull-request test fan-out with one bounded,
required fast lane while preserving comprehensive coverage on main and manual
runs and preparing the reviewed dedicated nightly workflow for default-branch
activation. The new lane must become enforceable without introducing a required
status context that strands existing pull requests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pull requests targeting `dev` run one serial fast-test job and do not create the heavyweight `Tests` workflow fan-out
- [x] #2 The fast lane runs all CI-contract, smoke, operation-lease, and minimum-Textual MCP targets from the approved design using non-overlapping pytest paths
- [x] #3 The existing required context `Derived artifacts reproduce from their sources` fails when the pull-request fast lane does not succeed and retains its install-free artifact checks
- [x] #4 Pushes to `main` and manual dispatch retain their documented comprehensive test coverage, and the dedicated nightly workflow source retains the five-environment full-tree run against one resolved `dev` commit
- [x] #5 The fast lane installs only the application and explicit essential pytest dependencies, not `requirements-test.txt` or optional ML/document/browser stacks
- [x] #6 Existing open pull requests are not stranded by a newly required status context, and branch protection requires no context migration
- [x] #7 Workflow-contract tests, clean Python 3.11 minimal-dependency execution, YAML parsing, Ruff, diff checks, and a live pull-request run verify the new contract
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record the long-lived coverage cadence and stable required-context decision in ADR-103 and the approved design specification.
2. Add RED workflow-contract tests for the fast-lane target set, dependency boundary, required-gate aggregation, heavy-workflow event ownership, and non-overlapping pytest selection.
3. Add the serial Python 3.11 fast lane to the existing required workflow and make the stable derived-artifacts job explicitly aggregate its result.
4. Remove pull-request and schedule admission, the embedded nightly job, and obsolete PR-only summary permissions from the heavyweight `Tests` workflow while preserving main and manual coverage; add the dedicated schedule/manual nightly workflow source with one immutable `dev` SHA shared by every matrix leg.
5. Verify collection and execution in a clean minimal environment, run the focused contract suite, parse changed YAML, run Ruff and diff checks, and mutation-test the aggregator failure path.
6. Rebase on latest `dev`, open the PR, verify its live required gate and routine runner fan-out, address all review feedback, then merge the exact reviewed head; TASK-19600 separately promotes only the reviewed nightly workflow to default-branch `main` and proves the real schedule.

ADR required: yes

ADR path: `backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md`

Reason: this changes the repository's long-lived required CI contract, dependency
boundary, and coverage cadence across pull requests, main, nightly, and manual runs.

Design: `Docs/superpowers/specs/2026-08-29-fast-pr-lane-design.md`
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Every acceptance criterion is checked
- [x] #2 Automated workflow-contract tests cover the new policy and its fail-closed paths
- [x] #3 Targeted static analysis and verification pass
- [x] #4 ADR-103, the design, and operational CI documentation agree
- [x] #5 The PR's live required gate completes and reports a truthful verdict
- [x] #6 Review feedback is resolved and implementation notes are recorded
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added one serial, minimal Python 3.11 PR lane to `derived-artifacts.yml`; the
  existing protected job now truthfully aggregates that prerequisite while all
  artifact checkers remain install-free and independently diagnostic.
- Restricted the heavyweight `Tests` workflow to `main` pushes and manual
  dispatch, and added the reviewed dedicated nightly source with one resolved
  `dev` SHA shared and recorded by all five matrix legs. TASK-19600 owns the
  atomic promotion of that workflow to default-branch `main` and live schedule
  proof.
- Added fail-closed workflow contracts for exact targets, dependency boundaries,
  event ownership, prerequisite result handling, selection-suppressing pytest
  flags, and every `continue-on-error` bypass. Independent review also drove the
  immutable nightly SHA resolver and the all-required-steps bypass mutation.
- Verification found and reproduced a stale-widget `Pilot.click()` race in one
  selected MCP viewport test. The test now uses its focused keyboard binding;
  the incident and pressure evidence are retained in
  `backlog/docs/lessons-testing-evidence.md`.
- Final local evidence: 671 exact-lane tests passed in a clean minimal Python
  3.11 environment, 42 focused workflow contracts passed, Ruff/YAML/diff and all
  six artifact guards passed, and the MCP fix passed 12 isolated plus six
  concurrent ordered-prefix pressure runs. PR #2208 then proved the live shape:
  one 7m09s fast job, no heavyweight `Tests` fan-out, followed by the unchanged
  required artifact context passing in 4m29s. Independent review approved the
  full diff. Qodo's two docstring-compliance threads were resolved by documenting
  all twelve newly added or renamed workflow-contract tests.
<!-- SECTION:NOTES:END -->
