---
id: TASK-40323
title: "Subscription- and skill-set Console tests red on the round-13 base: first-run wizard x4, command popup x9 and eight more skill nodes, all on no lane"
status: To Do
created_date: 2026-11-12 19:36
labels:
- follow-up
- tests
- console
- dev-red
priority: low
references:
- Tests/Wizards/test_first_run_subscription_readiness.py
- Tests/UI/test_console_command_popup.py
- Tests/Performance/test_console_skill_trust_stock_owned.py
dependencies:
- TASK-38948
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (send stream, TASK-38948's covering runs at `-n 4`, load 23-46). The notes record these at file level only; the nodes are not named.
- Subscription set: 4 in the first-run wizard file (most likely `Tests/Wizards/test_first_run_subscription_readiness.py`, 9 nodes) fail the same way on base ae9f9e8d21. (The 8 Personas nodes belong to TASK-40322.)
- Skill set: 17 fail on base too: `Tests/UI/test_console_command_popup.py` x9 (of 15 functions), visit dispatch x2, controller wiring x1, held pointer `[skill]`, managed received acceptance, and the stock-owned `[send_enter]` / `[send_button]` cases.

**Why.** None of these files is in the UI census or a PR job (`test.yml` runs Tests/UI only on main pushes), so they can stay red indefinitely, and nobody has established whether they are red on CI or only under the local RecoveryRequired trap.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each failing node is named in the notes with its failure, and red or green is established on CI as well as locally
- [ ] #2 Each red is fixed or re-pinned with the commit that changed what it pinned; a product defect gets a test that is red before its fix
- [ ] #3 Every node of these files that is green runs on a PR lane
<!-- AC:END -->
