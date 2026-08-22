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
- [ ] #2 Thirty fresh measured three-turn samples per arm use the same pinned control, candidate, model, fixtures, request parameters, isolation, and 10% non-regression gates as TASK-19641.
- [ ] #3 All ninety measured conversations complete the exact 1/3/1 provider-round, `load_tools`, confined `fs_write`, terminal-follow-up path with zero prompt loss and clean final ownership.
- [ ] #4 The retained artifacts collectively preserve phase provenance: raw evidence and the manifest retain exact burn-in and measured identities, summaries retain only excluded burn-in counts and contract status, and no artifact makes a performance claim from burn-in samples.
- [ ] #5 Independent recomputation, privacy scans, focused tests, and static checks exactly validate the retained evidence and verdict.
- [ ] #6 The original TASK-19641 evidence remains byte-identical and the confirmatory evidence is stored separately.
<!-- AC:END -->
