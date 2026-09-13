---
id: TASK-32463
title: 'Screen-size ratchet: both budgeted screens are over — decide who pays it back'
status: To Do
assignee: []
created_date: '2026-09-12 00:14'
labels:
  - tests
  - architecture
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/Architecture/test_screen_size_ratchet.py` is red on `origin/dev` for both screens it budgets, and has been for long enough that the numbers are no longer near-misses:

- `tldw_chatbook/UI/Screens/chat_screen.py` — 24,389 lines against a 16,966 budget (+7,423)
- `tldw_chatbook/UI/Screens/library_screen.py` — 34,856 lines against a 33,204 budget (+1,652)
- plus `test_task_22507_4_does_not_worsen_chat_screen_base`, red for the same reason

(Measured at `de11cac918`; the same three were red at the branch's base `ff2dc03145` with 23,946 and 34,800, so both screens are still growing.)

The file's own ledger is explicit that the budget may only go DOWN and that raising it "defeats the entire mechanism and re-opens the hole this test was written to close". Meanwhile a permanently red ratchet teaches every wave to ignore it, which is the same hole by another route: three reds are now indistinguishable from four.

This is a decision task, not a refactor task. Someone has to say who pays the debt back and by how much — which extractions, into which modules (`UI/Console_Modules/`, `UI/Library_Modules/`), on what horizon — and, if the answer is "not soon", what the ratchet should do in the meantime so that a NEW overshoot is still visible against the accepted one.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A decision is recorded naming who pays back each screen's overshoot and over what horizon
- [ ] #2 If the debt is not paid immediately, the ratchet distinguishes the accepted overshoot from a new one, so a further increase still fails
- [ ] #3 `Tests/Architecture/test_screen_size_ratchet.py` reflects the decision and is green on dev, with no budget raised in contradiction of its own ledger rule
<!-- AC:END -->
