---
id: TASK-31816
title: >-
  ui_ready module census has zero headroom: pin 972 breaches at 974 under load
  on trees that do not contain the change
status: To Do
assignee: []
created_date: '2026-09-06 04:43'
labels:
  - performance
  - tech-debt
  - architecture
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/Performance/test_ui_ready_module_census.py asserts at most 972 tldw_chatbook modules are resident at _ui_ready, and the warm-boot measurement IS 972 -- zero headroom. Under machine load it measures 974 and fails, and it fails identically on trees that do not contain the change under test: wave-6 task 2 measured the same 974 with the same assertion text on its branch and on an ISOLATED baseline worktree at the parent commit, and wave-6 task 3 measured 8 passed / 2 failed over 10 isolated single-node runs on a quiet machine. It is therefore not load-gated in the simple sense and not attributable to any Library decomposition PR -- the Library work only ever REMOVES modules from the first-paint window (the born-lazy controller import), and none of the 25 modules the failure names is a Library module; Console, DB, LLM_Calls, Scheduling, Navigation and Widgets own every one. A guard pinned exactly at its own measurement flips on ordinary run-to-run wobble, which trains readers to dismiss it -- the failure mode a ratchet exists to prevent. The budget belongs to whoever owns first-paint residency, not to a passing subsystem extraction. dev's own 06acf148f pre-import paydown is the precedent for the paydown direction.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The census is deterministic at HEAD: 10 consecutive isolated single-node runs on an otherwise-quiet machine all pass, with the observed module count recorded for each
- [ ] #2 An explicit decision is recorded in the guard's own module docstring between (a) re-pinning with a STATED headroom and the run-to-run wobble that headroom absorbs, and (b) paying residency down below the current pin -- with the reasoning, not just the outcome
- [ ] #3 If the paydown direction is taken, the modules removed from the first-paint window are named individually and the removal follows dev's own 06acf148f pre-import paydown precedent
- [ ] #4 The guard's own docstring states how a future breach should be triaged (raise the pin vs. pay down) and how to tell a real residency regression from wobble, so the next caller does not re-derive it from scratch
- [ ] #5 backlog/docs/library-decomposition-recipe.md section 7's documented-pre-existing-failures list has the ui_ready census entry removed, with the commit that fixed it named
<!-- AC:END -->
