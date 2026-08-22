---
id: TASK-20010
title: Confirm steady-state three-turn Console latency
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-21 21:21'
updated_date: '2026-08-22 16:09'
labels:
  - console
  - performance
  - diagnostics
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run a separately pre-registered steady-state confirmation of the real-provider three-turn Console comparison after balanced burn-in, preserving the original inconclusive benchmark evidence unchanged. Latest-dev integration will canonically renumber that benchmark TASK-20009 while leaving its retained artifacts' internal TASK-19641 label byte-identical.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Five complete balanced burn-in blocks run after one warmup per arm, every burn-in conversation satisfies the full product, privacy, cleanup, and ownership contract, and burn-in is excluded from all measured summaries by a predeclared rule.
- [ ] #2 Thirty fresh measured three-turn samples per arm use the same pinned control and candidate, endpoint-reported model alias, retained server contract, fixtures, request parameters, isolation, and 10% non-regression gates as the original benchmark (canonical TASK-20009 after latest-dev integration); the report discloses that the original evidence did not retain a model-weight digest.
- [ ] #3 All ninety measured conversations complete the exact 1/3/1 provider-round, `load_tools`, confined `fs_write`, terminal-follow-up path with zero prompt loss and clean final ownership.
- [ ] #4 Before filtering, the complete 108-conversation terminal-row identity/order sequence exactly matches the predeclared schedule with global sample-ID uniqueness; retained artifacts collectively preserve phase provenance, summaries retain only excluded burn-in counts and contract status, and no artifact makes a performance claim from burn-in samples.
- [ ] #5 Independent recomputation, privacy scans, focused tests, and static checks exactly validate the retained evidence and verdict.
- [ ] #6 The original benchmark evidence remains byte-identical, including its internal pre-integration TASK-19641 label, and the confirmatory evidence is stored separately under canonical TASK-20010.
- [ ] #7 The first complete protocol-valid attempt is definitive regardless of verdict; correctable derived-artifact defects are fixed and re-reviewed without reacquisition, only uncorrectable acquisition or raw-evidence failures may be retried, and all attempt states remain linked and retained.
- [ ] #8 The original harness revision and runner digest are explicit, its digest-verified original statistics module produces the measured summary, and publication atomically promotes only artifacts whose canonical digest and attempt ID exactly match an approving independent-review receipt.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the pinned candidate object, rebase onto the exact refreshed `origin/dev`, preserve dev's colliding TASK-19641/ADR-077, renumber the branch-owned benchmark task/Change Review ADR without modifying immutable evidence or protocol fixtures, and pin the exact post-integration implementation baseline.
2. Extend the existing benchmark schedule with predeclared balanced burn-in and validate the complete terminal-row order and identity before statistical filtering.
3. Pin and digest-check the original harness/evidence, reuse its validators and statistics directly, and fail closed on protocol, revision, workspace, or listener drift.
4. Add an append-only attempt ledger and atomic acquisition lock that make the first complete protocol-valid attempt definitive and preserve retry lineage.
5. Wire confirmatory acquisition through the existing parent/child runner without changing production Console code.
6. Bind independent review to the exact canonical artifact digest and publish approved evidence through verified sibling-copy plus atomic rename.
7. Verify the harness, run a disposable live smoke against port 9099, acquire and independently review the official 30-block confirmation, then publish the retained evidence.
8. Run the full test/lint/format gates and final evidence checks, record the measured verdict without altering the original evidence, and complete Backlog hygiene only after every gate passes.

Detailed executable plan: `Docs/superpowers/plans/2026-08-22-confirmatory-steady-state-console-latency.md`

ADR required: no

ADR path: `backlog/decisions/079-change-review-consent-and-asynchronous-finalization.md` after latest-dev integration (existing governing ADR, renumbered from branch-local ADR-077)

Reason: this task changes benchmark-only tooling and retained evidence, while the renumbered ADR-079 already governs the Change Review behavior being measured.
<!-- SECTION:PLAN:END -->
