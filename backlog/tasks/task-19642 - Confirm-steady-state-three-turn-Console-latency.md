---
id: TASK-19642
title: Confirm steady-state three-turn Console latency
status: To Do
assignee:
  - '@codex'
created_date: '2026-08-21 21:21'
labels:
  - console
  - performance
  - diagnostics
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run a separately pre-registered steady-state confirmation of the real-provider three-turn Console comparison after balanced burn-in, preserving the original inconclusive TASK-19641 evidence unchanged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Five complete balanced burn-in blocks run after one warmup per arm, every burn-in conversation satisfies the full product, privacy, cleanup, and ownership contract, and burn-in is excluded from all measured summaries by a predeclared rule.
- [ ] #2 Thirty fresh measured three-turn samples per arm use the same pinned control and candidate, endpoint-reported model alias, retained server contract, fixtures, request parameters, isolation, and 10% non-regression gates as TASK-19641; the report discloses that the original evidence did not retain a model-weight digest.
- [ ] #3 All ninety measured conversations complete the exact 1/3/1 provider-round, `load_tools`, confined `fs_write`, terminal-follow-up path with zero prompt loss and clean final ownership.
- [ ] #4 Before filtering, the complete 108-conversation terminal-row identity/order sequence exactly matches the predeclared schedule with global sample-ID uniqueness; retained artifacts collectively preserve phase provenance, summaries retain only excluded burn-in counts and contract status, and no artifact makes a performance claim from burn-in samples.
- [ ] #5 Independent recomputation, privacy scans, focused tests, and static checks exactly validate the retained evidence and verdict.
- [ ] #6 The original TASK-19641 evidence remains byte-identical and the confirmatory evidence is stored separately.
- [ ] #7 The first complete protocol-valid attempt is definitive regardless of verdict; correctable derived-artifact defects are fixed and re-reviewed without reacquisition, only uncorrectable acquisition or raw-evidence failures may be retried, and all attempt states remain linked and retained.
- [ ] #8 The original harness revision and runner digest are explicit, its digest-verified original statistics module produces the measured summary, and publication atomically promotes only artifacts whose canonical digest and attempt ID exactly match an approving independent-review receipt.
<!-- AC:END -->
