---
id: TASK-24403
title: Fast PR lane preserves required gate and full coverage cadence
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-29'
updated_date: '2026-08-29'
labels:
  - ci
  - infrastructure
  - github-actions
priority: high
dependencies:
  - task-22250
---

## Description

Replace the account-saturating full pull-request test fan-out with one bounded,
required fast lane while preserving comprehensive coverage on main and manual
runs and preparing the reviewed dedicated nightly workflow for default-branch
activation. The new lane must become enforceable without introducing a required
status context that strands existing pull requests.

## Acceptance Criteria

- [ ] Pull requests targeting `dev` run one serial fast-test job and do not create the heavyweight `Tests` workflow fan-out
- [ ] The fast lane runs all CI-contract, smoke, operation-lease, and minimum-Textual MCP targets from the approved design using non-overlapping pytest paths
- [ ] The existing required context `Derived artifacts reproduce from their sources` fails when the pull-request fast lane does not succeed and retains its install-free artifact checks
- [ ] Pushes to `main` and manual dispatch retain their documented comprehensive test coverage, and the dedicated nightly workflow source retains the five-environment full-tree run against `dev`
- [ ] The fast lane installs only the application and explicit essential pytest dependencies, not `requirements-test.txt` or optional ML/document/browser stacks
- [ ] Existing open pull requests are not stranded by a newly required status context, and branch protection requires no context migration
- [ ] Workflow-contract tests, clean Python 3.11 minimal-dependency execution, YAML parsing, Ruff, diff checks, and a live pull-request run verify the new contract

## Implementation Plan

1. Record the long-lived coverage cadence and stable required-context decision in ADR-103 and the approved design specification.
2. Add RED workflow-contract tests for the fast-lane target set, dependency boundary, required-gate aggregation, heavy-workflow event ownership, and non-overlapping pytest selection.
3. Add the serial Python 3.11 fast lane to the existing required workflow and make the stable derived-artifacts job explicitly aggregate its result.
4. Remove pull-request and schedule admission, the embedded nightly job, and obsolete PR-only summary permissions from the heavyweight `Tests` workflow while preserving main and manual coverage; add the dedicated schedule/manual nightly workflow source.
5. Verify collection and execution in a clean minimal environment, run the focused contract suite, parse changed YAML, run Ruff and diff checks, and mutation-test the aggregator failure path.
6. Rebase on latest `dev`, open the PR, verify its live required gate and routine runner fan-out, address all review feedback, then merge the exact reviewed head; the dependent activation task separately promotes only the reviewed nightly workflow to default-branch `main` and proves the real schedule.

ADR required: yes

ADR path: `backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md`

Reason: this changes the repository's long-lived required CI contract, dependency
boundary, and coverage cadence across pull requests, main, nightly, and manual runs.

Design: `Docs/superpowers/specs/2026-08-29-fast-pr-lane-design.md`

## Definition of Done

- [ ] Every acceptance criterion is checked
- [ ] Automated workflow-contract tests cover the new policy and its fail-closed paths
- [ ] Targeted static analysis and verification pass
- [ ] ADR-103, the design, and operational CI documentation agree
- [ ] The PR's live required gate completes and reports a truthful verdict
- [ ] Review feedback is resolved and implementation notes are recorded
