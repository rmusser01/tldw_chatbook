---
id: TASK-40324
title: "test_stock_console_skill_setup_has_original_owned_lifetime[app_closing] hangs to its 240 s bound now and then"
status: To Do
created_date: 2026-11-12 19:36
labels:
- follow-up
- tests
- console
- flaky
priority: low
references:
- Tests/Performance/test_console_skill_trust_stock_owned.py
dependencies:
- TASK-38948
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (send stream, TASK-38948's covering runs). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** A shutdown-time hang in the stock skill-trust setup costs 4 minutes when it hits and masks real failures in the same run; nothing on CI would notice if it became a real hang on close. TASK-38948 measured it on base and fix alike: base 2 of 16 four-way-parallel runs and 0 of 19 serial; fix 4 of 59. An asyncio task dump showed the script's outer `wait_for` pending with the loop idle.

**Evidence.** `Tests/Performance/test_console_skill_trust_stock_owned.py::test_stock_console_skill_setup_has_original_owned_lifetime[app_closing]`: outer bound `asyncio.wait_for(exercise(), timeout=240)` (`:719`), subprocess `_run(..., timeout=240)` (`:751`), the app_closing branch at `:401-414` and `:471-472`. Last changed by 3e618d7119; on no lane. The same file's `[send_enter]` and `[send_button]` cases are red on the base (TASK-40323 owns them).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The notes say what the pending wait is when the case hangs (a product hang on app close, or a test wait), with the evidence
- [ ] #2 The case passes 20 of 20 runs at `-n 4` under load, or a product hang on close is fixed with a test that was red
- [ ] #3 When it does hang, it fails within 60 s and names the pending wait
<!-- AC:END -->
