---
id: TASK-3070
title: chat_screen size ratchet red on dev after console decomposition wave 3
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-07 18:20'
updated_date: '2026-08-13 18:05'
labels:
  - console
  - architecture-gate
  - tech-debt
dependencies: []
references:
  - Docs/superpowers/specs/2026-08-13-console-decomposition-wave6-design.md
  - Docs/superpowers/specs/2026-08-23-console-decomposition-wave6-closeout-amendment.md
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
- [ ] #1 On the final rebased dev base, both the `chat_screen.py` line-count and `ChatScreen` method-count ratchets pass without increasing either budget.
- [ ] #2 The source-inspected Wave 6 move/delegate/stay inventory is enforced by AST ownership tests; exact screen/region-owned stays remain in place, and every retained delegate is framework-required, has a real caller, and its complete definition span (decorators excluded) is bounded to five physical source lines.
- [ ] #3 Image/H3, video, conversation-browser, retrieval/RAG, skill, character, fleet/wake, first-chat, auto-speak, realtime, and review/selection ownership moves to the reviewed controllers without changing DOM or ADR-068 review-note ownership, persistence ordering, worker groups, session/tap/sink and cancellation identity, transcript/usage/fallback ordering, remount/shutdown behavior, privacy, or user-visible outcomes.
- [ ] #4 Every extracted family has isolated controller coverage using plain fakes without mounting Textual, including the Workspace browser extension, and the mounted product suites continue to cover screen/region integration.
- [ ] #5 The coordinated child tasks TASK-3070.1 through TASK-3070.14 are completed independently with their focused verification evidence before this parent is closed.
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
<!-- SECTION:NOTES:END -->
