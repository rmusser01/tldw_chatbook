---
id: TASK-18909
title: Reduce Console screen-switch mount cost (1.55s warm on fast hardware)
status: To Do
assignee: []
created_date: '2026-08-19 16:31'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Measured 2026-08-19 at dev f6ae7d23e (TASK-18908 spike): switching to the Console/ChatScreen costs ~1.55s WARM on an M-series Mac with a scratch config, vs Home 0.76s / Settings 0.89s / Library 1.39s. Screens rebuild fully per switch by design (caching was root-caused to a freeze in July); the cost is construction+compose+CSS-apply of a 21k-line screen plus its Console module chain (transcript+bridge+controller+store grew +9.6k lines in the Aug 15-18 window). On constrained Windows hardware at 3-5x this is 5-8s — the residual of the reported incident. Profile a warm switch (Py-spy or cProfile around handle_screen_navigation) and land the top deferrable items (likely first-visit import already handled by preimport thread; suspect compose-time work that can move to post-first-paint workers).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Profile of a warm Console switch exists with named top costs,Top deferrable compose-time work moved off the first paint (or documented why each cannot be),Warm Console switch re-measured and reported in the task,Latency guardrail budgets updated if the improvement moves baseline
<!-- AC:END -->
