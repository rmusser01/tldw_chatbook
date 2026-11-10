---
id: TASK-39167
title: "Two Settings and Console UI tests race a push or a value change with one pilot.pause(): the create-endpoint return and the first edit after a deep link"
status: To Do
created_date: 2026-11-08 06:54
labels:
- follow-up
- tests
- settings
- console
- flaky
priority: low
references:
- Tests/UI/test_console_endpoint_discovery.py
- Tests/UI/test_settings_deep_link_first_edit.py
dependencies:
- TASK-38961
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 12 of the 2026-10-23 burn-down (TASK-38961's notes, round-12 branch fu/r12-ctests; and the settings stream). Checked by reading on dev b590a15afd and on the round-12 stack 9091820bf2, 2026-11-08; line numbers are dev 3a6498ae14's (re-read 2026-11-09, both test files unchanged since b590a15afd).

**The create-endpoint return.** `test_console_endpoint_discovery.py::test_create_endpoint_returns_to_responsive_settings_with_settled_evidence[outcome0]` failed 2 of 10 whole-file runs at load 45-161 (TASK-38961's notes).
- The test (`:125`) pushes the modal (`await app.push_screen(modal)`, `:139`), waits one `pilot.pause()`, presses New (`#console-settings-endpoint-new`, `:141`), waits one more `pilot.pause()`, then asserts that `app.screen` is the template modal and queries `#endpoint-template-name` (`:144-145`), later calling `_pick_in_pick_mode` (`:153`). A press or query that lands before the pushed screen has composed is the race TASK-38961's notes name.
- The node is not laned. Other nodes of the file are: 4 UI census entries on dev (7 on the round-12 stack), and 9 node ids on the workflow lane (`.github/workflows/derived-artifacts.yml:681-689` on dev 3a6498ae14).

**The first edit after a deep link.** `test_settings_deep_link_first_edit.py::test_the_first_edit_after_a_deep_link_keeps_the_linked_provider` failed once at load ~100 with `'api_key' == 'claude_subscription'` (settings stream).
- It sets `auth.value = "claude_subscription"` and waits one `pilot.pause()` (`:94-95`) before it asserts the staged value.
- The file is a whole-file UI census entry.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each test waits on the state it then reads (the composed modal's field; the staged auth value), with a bound that fails naming it
- [ ] #2 Each passes 20 of 20 runs at `-n 4`, with load recorded, at least ten of them at a one-minute load of 40 or more
<!-- AC:END -->
