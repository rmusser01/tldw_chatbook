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
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Console's one-way architecture ratchets are materially red on verified current
`origin/dev` after TASK-3070.2: `tldw_chatbook/UI/Screens/chat_screen.py` is 22,172
lines against a 17,727-line ceiling, and `ChatScreen` has 712 direct methods against a
593-method ceiling. This task was originally filed at a 21-line overage after wave 3; later waves
lowered the allowed ceiling while unrelated feature work kept accumulating in the
screen. The result is now 4,445 lines and 119 methods over budget, not transitional
noise. Resolve it through the amended Wave 6 controller extractions and lower the
ratchets to the exact earned counts; never raise either budget to accept the growth.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 On the final rebased dev base, both the `chat_screen.py` line-count and `ChatScreen` method-count ratchets pass without increasing either budget.
- [ ] #2 The source-inspected Wave 6 move/delegate/stay inventory is enforced by AST ownership tests, and every retained screen delegate is framework-required, has a real caller, and its complete definition span (decorators excluded) is bounded to five physical source lines.
- [ ] #3 Image/H3, video, conversation-browser, retrieval/RAG, skill, character, fleet/wake, first-chat, and auto-speak ownership moves to the reviewed controllers without changing DOM ownership, persistence ordering, worker groups, cancellation identity, remount/shutdown behavior, privacy, or user-visible outcomes.
- [ ] #4 Every extracted family has isolated controller coverage using plain fakes without mounting Textual, including the Workspace browser extension, and the mounted product suites continue to cover screen/region integration.
- [ ] #5 The coordinated child tasks TASK-3070.1 through TASK-3070.11 are completed independently with their focused verification evidence before this parent is closed.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A (governed by `Docs/superpowers/specs/2026-08-02-screen-decomposition-design.md` and `DESIGN.md` section 7)
Reason: This is the next implementation wave of the existing, approved Console decomposition architecture; it does not introduce a new runtime, storage, security, or cross-module policy.

1. Complete TASK-3070.1: record the rebased line/method baseline and lock the reviewed source-inspected ownership inventory, residue budget, and AST rules before production movement.
2. Complete TASK-3070.2 through TASK-3070.10 one PR at a time after its predecessor merges: characterize the real product boundary, add isolated no-mount controller tests, extract one family with named late-bound dependencies, and prove its screen delegates and invariants.
3. Extend the existing Workspace controller as the single conversation-browser state owner while preserving collapse/config and activation/resume behavior; keep Textual decorators, direct DOM, modal/picker presentation, and OS-player launch on the screen or existing region widgets.
4. After each extraction, run its isolated controller tests, focused product/mounted suites, AST ownership/dependency checks, required mutations, static checks, and privacy/diff gates before beginning the next child.
5. Complete TASK-3070.11: rebase onto final `origin/dev`, repeat the baseline comparison, lower both ratchets to the exact earned counts, update canonical decomposition progress and every task's notes, run the final repository DoD, and only then close this parent.
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
