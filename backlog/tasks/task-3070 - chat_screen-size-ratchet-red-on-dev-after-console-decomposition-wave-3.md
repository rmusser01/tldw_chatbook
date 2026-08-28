---
id: TASK-3070
title: chat_screen size ratchet red on dev after console decomposition wave 3
status: Done
assignee:
  - '@codex'
created_date: '2026-08-07 18:20'
updated_date: '2026-08-27 23:07'
labels:
  - console
  - architecture-gate
  - tech-debt
dependencies: []
references:
  - Docs/superpowers/specs/2026-08-13-console-decomposition-wave6-design.md
  - >-
    Docs/superpowers/specs/2026-08-23-console-decomposition-wave6-closeout-amendment.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Console's one-way architecture ratchets are materially red on verified current
`origin/dev` after TASK-3070.2. The final Wave-6 delivery base was 19,863 lines and
630 methods; the current amendment base is 19,884 lines and 632 methods. The completed
extractions removed 4,958 lines and 130 methods, while concurrent Console work accounts
for 2,670 lines and 50 methods. The remaining 2,157-line and 39-method deficits are not
transitional noise. Resolve them through the approved
closeout amendment and lower the ratchets to the exact earned counts; never raise
either budget to accept growth.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 On the final rebased dev base, both the `chat_screen.py` line-count and `ChatScreen` method-count ratchets pass without increasing either budget.
- [x] #2 The source-inspected Wave 6 move/delegate/stay inventory is enforced by AST ownership tests; exact screen/region-owned stays remain in place, and every retained delegate is framework-required, has a real caller, and its complete definition span (decorators excluded) is bounded to five physical source lines.
- [x] #3 Image/H3, video, conversation-browser, retrieval/RAG, skill, character, fleet/wake, first-chat, auto-speak, realtime, and review/selection ownership moves to the reviewed controllers without changing DOM or ADR-068 review-note ownership, persistence ordering, worker groups, session/tap/sink and cancellation identity, transcript/usage/fallback ordering, remount/shutdown behavior, privacy, or user-visible outcomes.
- [x] #4 Every extracted family has isolated controller coverage using plain fakes without mounting Textual, including the Workspace browser extension, and the mounted product suites continue to cover screen/region integration.
- [x] #5 The coordinated child tasks TASK-3070.1 through TASK-3070.14 are completed independently with their focused verification evidence before this parent is closed.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: `backlog/decisions/068-console-text-selection-and-annotations.md` (also governed by `Docs/superpowers/specs/2026-08-02-screen-decomposition-design.md` and `DESIGN.md` section 7)
Reason: This is the next implementation wave of the existing, approved Console decomposition architecture and preserves ADR-068's screen-owned review-note workflow; it does not introduce a new runtime, storage, security, or cross-module policy.

1. Complete TASK-3070.1: record the rebased line/method baseline and lock the reviewed source-inspected ownership inventory, residue budget, and AST rules before production movement.
2. Complete TASK-3070.2 through TASK-3070.10 one PR at a time after its predecessor merges: characterize the real product boundary, add isolated no-mount controller tests, extract one family with named late-bound dependencies, and prove its screen delegates and invariants.
3. Extend the existing Workspace controller as the single conversation-browser state owner while preserving collapse/config and activation/resume behavior; keep Textual decorators, direct DOM, modal/picker presentation, and OS-player launch on the screen or existing region widgets.
4. After each extraction, run its isolated controller tests, focused product/mounted suites, AST ownership/dependency checks, required mutations, static checks, and privacy/diff gates before beginning the next child.
5. Complete TASK-3070.11: lock the post-Wave-6 delivery evidence and the approved
   closeout amendment without changing production behavior.
6. Complete TASK-3070.12 and TASK-3070.13 serially: extract realtime orchestration,
   then review/selection workflows, behind explicit no-sibling controller boundaries.
7. Complete TASK-3070.14: rebase onto final `origin/dev`, repeat the baseline
   comparison, lower both ratchets to the exact earned counts, update canonical
   decomposition progress and every task's notes, run the approved focused and
   required CI DoD gates, and only then close this parent.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-08-10, supervisor-fleet PR 2b: still red. Verified failing at 48a54ed9c (pre-fleet dev) and at 762596846 (dev after fleet PR 2a), so this predates all fleet work. Recorded here because PR 2b adds +121 net lines to chat_screen.py (Task 4's fleet-section wiring and Task 5's coalescer/cancel handler/cost-ticker wiring), all DOM-adjacent and placed alongside the existing precedents they mirror — i.e. this PR grows an already-violated budget rather than breaching a green one. Also observed at the same commits: Tests/UI/test_console_control_bar_coalescing.py fails 2/3 on dev independently of any fleet change.

2026-08-11, supervisor-fleet PR 3a-1 (Task 7) — **the figure in this task's Description is
now three orders of magnitude out of date, and that is the finding.** Measured in this
worktree, not inherited:

| what | lines | budget | over |
|---|---|---|---|
| when this task was filed | 18,930 | 18,909 | **+21** |
| this PR's own merge base (`ecfc9ab95`) | 20,045 | 17,727 | **+2,318** |
| this PR's HEAD (`d87bef16d`) | 20,063 | 17,727 | **+2,336** |
| current `origin/dev` tip | 22,047 | 17,727 | **+4,320** |

Two things moved at once: the budget was RATCHETED DOWN (18,909 → 17,727) by later
decomposition waves, and the file kept growing. PR 3a-1 contributed **18** of the
overage, in one commit (`cced002ab`, the F3 cost-chip wiring in
`_build_console_cost_state`); PR 2b contributed +119 before it. Neither breached a green
budget — both grew an already-violated one.

This task's own stated purpose was to stop the red becoming "pre-existing noise that
hides something real". A 21-line figure sitting in the Description while the real number
is 2,336 (4,320 on dev) does exactly that: anyone triaging reads "21 lines, transitional"
and moves on. The budget is deliberately NOT raised here — `Tests/Architecture/
test_screen_size_ratchet.py` documents it as a one-way ratchet, and raising it is the
decomposition owner's call, not a passing PR's.

2026-08-27 closeout: TASK-3070.1 through TASK-3070.14 and the related TASK-21201 are complete with checked acceptance criteria, plans, implementation notes, and focused evidence. On final dev c4e52794e2, `chat_screen.py` is 17,037 physical lines with 565 direct `ChatScreen` definitions and 535 unique method names, lowering the immutable historical 17,727/593 ceiling by 690 lines and 28 definitions. Source-inspected inventory and AST boundary tests enforce the approved move/delegate/stay rules, and the reviewed controller families retain isolated plain-fake coverage plus focused mounted integration evidence.

The latest base passed the 61-test Console architecture set and 323 Research Workspace/import-closure tests; earlier current-branch evidence includes 97 citation-persistence tests and 51 character-pagination/provider-continuation tests. Required Derived Artifacts run 33123813806 passed on head 5858b7eb4d. Tests run 33123813819 passed the workflow-shape and all three native artifact-lease legs before a repository-wide external cancellation stopped non-required shards, with no code failure. Targeted Ruff, format, compile, privacy, diagnostic inventory, backlog-ID, ancestry, and diff gates passed; no local full suite was run. Qodo reported zero bugs, zero rule violations, zero requirement gaps, and no inline threads.

Final post-closeout repair evidence: required Derived Artifacts run 33125171664 passed on head 20d67f35d1. The subsequent non-required shard failures were inspected node by node and compared with an exact c4e52794e2 dev worktree. Three branch-specific regressions were minimally repaired (Watchlists mounted-state test contract, deferred exchange-export import contract, and Media resize focus preservation); all other locally reproducible failures either matched dev or passed in isolation / required unavailable optional host tooling. After the final rebase onto dev 7e84a7bef4, 96 focused regression, exchange-export, closeout/size/review boundary, diagnostic-inventory, and backlog-ID tests pass together. The live Console ratchet remains exactly 17,037 physical lines and 565 direct definitions. No local full suite was run.

ADR required: no. Existing ADR-068 and the approved Console decomposition designs govern the completed ownership boundaries. No new lesson was warranted because the work applies the existing final-rebase ratchet lesson.
<!-- SECTION:NOTES:END -->
