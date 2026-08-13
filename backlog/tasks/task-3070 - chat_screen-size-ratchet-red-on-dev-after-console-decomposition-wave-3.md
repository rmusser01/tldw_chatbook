---
id: TASK-3070
title: chat_screen size ratchet red on dev after console decomposition wave 3
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-07 18:20'
updated_date: '2026-08-11 04:55'
labels:
  - console
  - architecture-gate
  - tech-debt
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/Architecture/test_screen_size_ratchet.py::test_screen_does_not_grow_past_its_budget[tldw_chatbook/UI/Screens/chat_screen.py]`
fails on dev: 18,930 lines against a budget of 18,909. Introduced by PR #1408 (console
decomposition wave 3) itself — confirmed byte-identical on a clean dev-tip worktree at
`15407a641` during the TASK-3035/3045 architecture-gate refresh (PR #1416), so it is not
an artifact of any other branch. The decomposition stream is actively shrinking this
screen; the 21-line overage is presumably transitional. Filed so the red gate is owned
rather than becoming the next "pre-existing noise" that hides something real (see
lessons-testing-evidence.md's TASK-2610 entry for how that ends). Resolve by shrinking
the screen below budget in the next wave — not by raising the budget, unless the
decomposition stream's owner explicitly decides the budget is wrong.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The chat_screen size-ratchet test passes on dev
- [ ] #2 The resolution shrinks the screen (or an explicit, documented owner decision adjusts the budget)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A (governed by `Docs/superpowers/specs/2026-08-02-screen-decomposition-design.md` and `DESIGN.md` section 7)
Reason: This is the next implementation wave of the existing, approved Console decomposition architecture; it does not introduce a new runtime, storage, security, or cross-module policy.

1. Record the current post-rebase measurement and map the image/H3, video, conversation-browser, and retrieval/RAG ownership clusters method by method.
2. Write and review a wave-6 design that preserves Textual entry points on `ChatScreen`, keeps DOM work in regions/screen handlers, and puts non-DOM state/behaviour in explicit controllers wired through `UI/Console_Modules/wiring.py`.
3. Characterize each cluster through its real product boundary before moving code, then extract one controller at a time with exact dependency signatures and separate commits.
4. Run the controller, UI, Chat, Video Generation, RAG, architecture, worker-group, and import/reference regression suites after each extraction; mutation-check the screen delegations and controller ownership seams.
5. Rebase onto final `origin/dev`, measure again, lower the ratchet to the exact earned line/method counts, update the decomposition spec/task notes, and run repository-level verification before marking Done.
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
