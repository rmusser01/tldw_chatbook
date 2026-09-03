---
id: TASK-26840
title: Console approval card - reuse the single-row render tree
status: Done
assignee: []
created_date: '2026-09-01 14:57'
updated_date: '2026-09-01 20:53'
labels:
  - console
  - agents
  - approvals
  - performance
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reduce the visible delay and intermittent lag when an ordinary one-row Console permission prompt appears, without changing permission decisions, security boundaries, or batch behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Repeated updates for the same ordinary one-row approval shape reuse the mounted non-committing row/details subtree instead of replacing it
- [x] #2 A changed approval round updates all visible content and creates fresh decision controls so stale actions cannot reach it
- [x] #3 Batch and raw-shell approval shapes retain their existing behavior
- [x] #4 Targeted tests demonstrate the reuse contract and preserve first-open answerability
- [x] #5 A before-and-after mounted timing probe records the effect on the one-row path
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm the current one-row construction and approval-event invariants against the mounted card and its precomposed sibling patterns.
2. RED: add mounted regressions proving a same-shape one-row update preserves the non-committing row/details identity while refreshing content, a queued old Select message cannot reach the new round, and no-pause generations leave only the latest controls; keep shape-change coverage.
3. GREEN: minimally update the existing mounted non-committing widgets in place, create fresh generation-bound decision controls, and retain the current rebuild path for first mount, batches, and structurally different rows.
4. Run focused approval-card, first-open, and controller tests; run scoped Ruff and diff checks.
5. Repeat the production-styled timing probe, self-review the diff, record evidence, check all acceptance criteria, and close the task if the Definition of Done is satisfied.

ADR required: no
ADR path: N/A
Reason: this is a rendering optimization inside the existing approval-card owner; permission policy, persistence, service contracts, security boundaries, and UI structure remain unchanged.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the ordinary one-row steady-state fast path while leaving the first-mount, batch, raw-shell, and optional-detail shape-change paths unchanged. A mounted row now retains only its non-committing header/details widgets; every changed round gets a fresh Horizontal, Select, and fast buttons. All already-pruning controls subtrees are retired before the latest generation mounts, so queued old Select events, expanded overlays, stale button presses, and no-pause three-generation updates cannot affect the new permission round.

Evidence: RED/GREEN regressions cover mounted detail identity, changed visible content/options/effects/context, fresh decision identity, queued old Select messages, expanded-state reset, no-pause generations, and the submitted new round. Tests: Tests/UI/test_console_mcp_approval.py 93 passed; test_chat_approval_card.py plus test_approval_context_lines.py 28 passed; first-ever launch answerability 1 passed. The navigation first-open paint deadline fails identically on this tree and exact untouched base e966302e3 (title/tool paint, controls miss the sampled frame), so it is a paired pre-existing flake rather than a regression. The full suite was not run, per repository policy requiring opt-in.

Timing, same 60-update mounted probe: exact base sync median/p95/max 1.185/1.319/1.560 ms and settled 8.492/14.305/58.973 ms; final tree sync 1.233/1.312/1.428 ms and settled 7.563/8.795/46.149 ms. First prompt was unchanged at 12.645 vs 12.543 ms; warm settled median improved 8.456 to 7.525 ms. This is about 11% lower median and 39% lower p95 visible settle latency.

Verification: scoped Ruff passed, git diff --check passed, and independent correctness/security review found no remaining Critical or Important issue after two stale-interaction fixes. ADR required: no; rendering optimization only, with no permission policy, persistence, service contract, security-boundary, or long-lived UI-structure decision. Added the hidden-precomposition first-open lesson to backlog/docs/lessons-testing-evidence.md.

Pre-PR UAT: a Textual-web/CDP run drove the production `ChatApprovalCard` with two native calculator calls from a live local llama.cpp provider. Round 1 recorded `approve_once`; round 2 rendered the changed expression while reporting `row_reused=True`, `fresh_controls=True`, and a 9.37 ms synchronous `set_batch`; Deny then resolved only the second live call. Screenshots and the honest boundary between this passing scoped flow and unrelated full-Console project-inspector/provider-preflight blockers are recorded in `Docs/superpowers/qa/console-uat-parallelization/task-26840-approval-reuse-cdp-2026-09-01.md`. The isolated run left the real config and data fingerprints unchanged.

PR review follow-up after rebasing onto dev `64f7a54ca`: Qodo's duplicated-control-literal finding was valid, so the shared labels, CSS classes, and tooltips now have one named module source used by both construction paths. Cubic's private-`SelectOverlay` concern was also avoidable: the stale-event regression now uses public `Select.value`, whose change posts `Select.Changed` asynchronously. Mutation verification removed the existing membership guard and reproduced the intended failure (`ValueError` for the stale Select), proving the public path still exercises the race. Final focused verification: 123 passed; scoped Ruff, duplicate-task guard across 2,922 files, and `git diff --check` passed. The existing three dependency/thread warnings are unchanged.

Modified files: chat_approval_card.py, test_chat_approval_card.py, test_console_mcp_approval.py, lessons-testing-evidence.md, this task record, the UAT evidence note, and two UAT PNGs.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

This task was originally created as TASK-26836 at 2026-09-01 14:57. On the
final pre-PR rebase, current `dev` already contained the older TASK-26836
created at 2026-09-01 14:51. Per the TASK-19601 older-arrival owner rule, this
younger task was renumbered to the then-free TASK-26840. Its task file, lesson
reference, UAT evidence note, and evidence filenames were updated together;
the `/private/tmp/tldw-approval-uat-26836.XHn8sn` strings in the evidence note
remain unchanged because they are the exact historical paths used for the run.

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Targeted automated tests pass
- [x] #2 Scoped static analysis and diff hygiene pass
- [x] #3 Implementation Notes include approach, evidence, ADR decision, and modified files
- [x] #4 Acceptance criteria are checked and task status is Done
<!-- DOD:END -->
