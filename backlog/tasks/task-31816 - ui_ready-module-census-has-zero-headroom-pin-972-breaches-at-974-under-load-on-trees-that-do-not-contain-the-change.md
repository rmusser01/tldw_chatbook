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
Tests/Performance/test_ui_ready_module_census.py asserts at most 972 tldw_chatbook modules are resident at _ui_ready, and the warm-boot measurement IS 972 -- zero headroom. Under machine load it measures 974 and fails, and it fails identically on trees that do not contain the change under test: wave-6 task 2 measured the same 974 with the same assertion text on its branch and on an ISOLATED baseline worktree at the parent commit, and wave-6 task 3 measured 8 passed / 2 failed over 10 isolated single-node runs on a quiet machine. It is therefore not load-gated in the simple sense and not attributable to any Library decomposition PR -- the Library work only ever REMOVES modules from the first-paint window (the born-lazy controller import), and none of the 25 modules the failure names is a Library module; Console, DB, LLM_Calls, Scheduling, Navigation and Widgets own every one. A guard pinned exactly at its own measurement flips on ordinary run-to-run wobble, which trains readers to dismiss it -- the failure mode a ratchet exists to prevent. The budget belongs to whoever owns first-paint residency, not to a passing subsystem extraction. dev's own 06acf148f pre-import paydown is the precedent for the paydown direction. ADR-097 (backlog/decisions/097-boot-budget-ratchets.md, "The four boot budgets are ratchets: they never rise") already settles the SHAPE of the fix and forbids the obvious one: the legitimate responses, in that ADR's own order of preference, are (1) defer the new cost off the guarded path, (2) shed at least as much existing cost from that path in the same PR, or (3) an explicit owner exception recorded as a row in its append-only exception ledger in the same commit. The ADR states plainly that "Raising a constant is not an option the failure message offers", so "raise the pin and give it some headroom" is off the table. The guard's own docstring also documents the tolerated run-to-run wobble as +/-1 module while the observed breach is +2, so this is a genuine ADR-097 breach owned by whatever work added those residents, not a guard that merely needs slack.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The census is deterministic at HEAD: 10 consecutive isolated single-node runs on an otherwise-quiet machine all pass, with the observed module count recorded for each
- [ ] #2 The resolution follows one of ADR-097's three legitimate responses and which one is recorded with its reasoning: defer the new cost off the first-paint window, shed at least as much existing cost from it in the same PR, or take an explicit owner exception with a row appended to that ADR's append-only exception ledger in the same commit (the 970->972 tls_trust row, owner commit 6fac5dbf95, is the worked precedent). Raising the constant without a ledger row is a defect by that ADR's own terms
- [ ] #3 If the paydown direction is taken, the modules removed from the first-paint window are named individually and the removal follows dev's own 06acf148f pre-import paydown precedent
- [ ] #4 The guard's own docstring states how a future breach should be triaged under ADR-097 and how to tell a real residency regression from the documented +/-1 wobble, so the next caller does not re-derive it from scratch
- [ ] #5 The two residents responsible for the +2 excess over the pin are named individually -- the breach message already diffs the census against boot_budget_snapshots/ui_ready_modules.txt, so this is a read, not an investigation -- and the work that introduced them is identified, so the paydown lands with its actual owner rather than with whichever PR next trips the guard
- [ ] #6 backlog/docs/library-decomposition-recipe.md section 7's documented-pre-existing-failures list has the ui_ready census entry removed, with the commit that fixed it named
<!-- AC:END -->
