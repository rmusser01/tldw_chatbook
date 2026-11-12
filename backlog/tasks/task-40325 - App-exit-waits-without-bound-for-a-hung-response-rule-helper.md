---
id: TASK-40325
title: "App exit waits without bound for a hung response-rule helper"
status: To Do
created_date: 2026-11-12 19:36
labels:
- follow-up
- console
- response-rules
- shutdown
priority: medium
references:
- tldw_chatbook/Chat/console_runtime.py
- tldw_chatbook/Chat/response_rules/resources.py
- tldw_chatbook/Chat/response_rules/evaluator.py
dependencies:
- TASK-39166
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (load stream, TASK-39166). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** If a response-rule helper's HTTP call hangs, quitting the app hangs with it.
- `ConsoleRuntime.dispose()` waits on `_rule_helper_pool.wait_for_cleanup()` with no bound (`Chat/console_runtime.py:6695-6708`); that wait is `while self.unsettled_count: await asyncio.sleep(0.01)` (`Chat/response_rules/resources.py:81-84`).
- A lease settles only in the worker thread's `finally` (`run_sync`, `:214-235`). The evaluator's `asyncio.timeout(min(30, remaining))` (`Chat/response_rules/evaluator.py:239-244`) cancels only the awaiter, because the worker is shielded.
- For most providers the helper's 30 s bound never reaches the HTTP call: `project_chat_handler_kwargs` drops `request_timeout` for every provider whose `PROVIDER_PARAM_MAP` lacks it (`Chat/Chat_Functions.py:1526-1554`). TASK-34356 (In Progress) owns that half (its AC#2); this task owns the unbounded exit wait.
- The only test (`Tests/Chat/test_response_rules_resources.py:231-249`) stubs `chat_api_call_fn` and checks kwargs only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 App exit with a hung rule helper finishes within a stated bound and records the abandoned helper; a test with a helper that never returns is red before the fix
- [ ] #2 A helper that finishes normally still settles its lease before exit
<!-- AC:END -->
