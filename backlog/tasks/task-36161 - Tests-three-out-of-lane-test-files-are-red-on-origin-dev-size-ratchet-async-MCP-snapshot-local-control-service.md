---
id: TASK-36161
title: >-
  Tests: three out-of-lane test files are red on origin/dev (size ratchet, async
  MCP snapshot, local control service)
status: To Do
assignee: []
created_date: '2026-10-22 18:16'
updated_date: '2026-10-22 19:12'
labels:
  - tests
  - ci
  - dev-red
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while rebasing TASK-33620.15.5's branch onto origin/dev 7700c63a68 (2026-10-22). Three test files that no PR-required lane runs fail on a detached origin/dev worktree, with the same test ids on the branch, so they are dev's own reds, seen only after merge by nightly-deep's full `pytest ./Tests/` run (test.yml runs on main push or dispatch only) (see lessons-testing-evidence: 'A regression test no PR lane runs went red within days'). Tests/Architecture/test_module_size_ratchet.py: 8 rows over budget (mcp_workbench.py 6954/6760, change_review_screen.py 4978/4967, llm_screen.py 5185/5180, console_transcript.py 8652/8353, app_command_providers.py 1133/1123, app_ingest_queue.py 4999/4985, app_lifecycle.py 2338/2134, tldw_api/client.py 16795/16687); CI's derived-artifacts job runs only the app.py, app_startup.py, app_served.py and three Settings rows. Tests/Chat/test_console_async_mcp_snapshot.py: 26 failures, mostly AttributeError on private names the code no longer has (ConsolePromptQueueUIController._launch_chain_async, ConsoleChatController._compose_local_provider_async) plus off-loop assertions. Tests/MCP/test_local_control_service.py: 3 failures, AttributeError: 'FakeLocalStore' object has no attribute 'path'. Reproduce: run each file alone from a detached origin/dev worktree (`.venv/bin/python -m pytest <file> -q`); the reds are deterministic (line counts, missing attributes), not timing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each named test id passes on origin/dev, or the test is retired with a recorded reason
- [ ] #2 Size-ratchet rows are re-pinned at measured values with dated owner-decision comments, or the files shrink back under them
- [ ] #3 Each of the three files is on a PR-required lane, or the reason it is not is recorded
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-10-22, on origin/dev e2cffbed60: a fourth file, same shape. Tests/MCP/test_local_server_tools.py::test_deny_state_handler_returns_tool_result fails on a detached origin/dev worktree and on TASK-33620.15.5's rebased branch with the same id (ToolResult.approval_decision 'denied' != None: the handler now records the denial the test's expected value omits). derived-artifacts.yml runs three other Tests/MCP files but not this one, so only nightly-deep sees it. The other three files' reds are unchanged on e2cffbed60 (size ratchet: the same 8 rows).
<!-- SECTION:NOTES:END -->
