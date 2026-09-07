---
id: TASK-14877
title: 'Dev baseline: 9 failing tests in Tests/Chat mask real regressions'
status: To Do
assignee: []
created_date: '2026-08-10 22:46'
updated_date: '2026-08-11 02:33'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
origin/dev at 59cf35d6e fails 9 tests in Tests/Chat with zero related changes in flight. Verified twice on a pristine detached checkout of origin/dev: Tests/Chat + Tests/MCP gives 9 failed / 4331 passed / 62 skipped, and the same 9 names fail when their files are run in isolation. Groups: (a) 5x Tests/Chat/test_console_agent_swap.py::test_mcp_tool_call_* (executes_end_to_end_when_state_allows, ask_state_routes_through_review_hook_and_approves, session_approval_suppresses_card_on_next_turn, ask_state_times_out_denies, gates_subagent_call_same_as_primary); (b) 1x Tests/Chat/test_console_ephemeral.py::test_promotion_restores_ephemeral_flag_if_persist_returns_none_unexpectedly; (c) 3x Tests/Chat/test_tool_output_disclosure.py (full_tool_output_is_reachable_from_the_mounted_transcript, pressing_o_expands_the_selected_marker, two_calls_in_one_turn_expand_independently). Cost already incurred: supervisor-fleet PR 2a (#1477) rebased onto dev and its battery went from 0 failures to 9, which had to be individually traced against a pristine dev checkout to prove none belonged to the branch — roughly 30 minutes of comparison runs that a green baseline would have made unnecessary. The MCP group is the most dangerous: it covers the tool-call permission path, so a genuine permission regression landing there would be indistinguishable from this noise.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Root cause identified for each of the three groups (they may have distinct causes)
- [ ] #2 All 9 tests pass on a clean origin/dev checkout, or any that encode obsolete behavior are deliberately rewritten with the change documented
- [ ] #3 Tests/Chat is green on dev across two consecutive runs
- [ ] #4 Tests/UI/test_css_class_coverage_contract.py::test_registry_entries_are_still_composed also fails on dev (flags console_transcript.py) — verified reproducing on merge-base 762596846, unrelated to any fleet work; include it in the sweep
<!-- AC:END -->
