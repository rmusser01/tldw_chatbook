---
id: TASK-42650
title: "Six canvas and Mermaid message-action tests fail on dev under the scratch-profile wrapper"
status: To Do
assignee: []
created_date: 2026-11-27 18:00
updated_date: 2026-11-27 18:38
labels:
- follow-up
- tests
- console
priority: medium
references:
- Tests/Chat/test_console_message_actions.py
- tldw_chatbook/Chat/console_message_actions.py
- tldw_chatbook/Canvas/config_policy.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 27 (2026-11-27) while rerunning, on the stack's final top, the test files that dev's #3733 touched. Six tests in `Tests/Chat/test_console_message_actions.py` fail on dev 9c4c8f9768 exactly as on the stack, run whole or each alone, under the scratch-profile wrapper:

- `test_mermaid_fences_keep_html_ordinals_and_use_distinct_replay_identity`
- `test_oversized_mermaid_keeps_discovery_safe_and_refuses_open`
- `test_mermaid_compile_repair_does_not_advertise_v1_diagram_execution`
- `test_canvas_html_actions_use_parsed_fences_and_stable_block_identity`
- `test_canvas_candidate_defers_compatibility_validation_until_open`
- `test_canvas_action_discovery_and_dispatch_do_not_compile_candidates`

They build `ConsoleMessageActionService()` without a `canvas_enabled_reader`, so the service uses `Canvas/config_policy.py::get_canvas_execution_enabled`. Inside these tests that reader raises `RecoveryRequired('raw_source_selection_changed')` (traced 2026-11-27 with the reader wrapped after conftest loads), and `ConsoleMessageActionService._read_canvas_enabled` turns any exception into "disabled". It is not the config default: the shipped default and the code default both turn Canvas on. Each test then sees Canvas as disabled: dispatch returns `blocked` with "Canvas is disabled. Restart Chatbook after re-enabling it.", discovery lists no canvas actions, and two get `None` where they expect a block reference (`AttributeError: 'NoneType' object has no attribute 'language'`).

No PR lane runs the file whole (the fast PR contract runs four of its nodes). The workflows that do run it whole (`nightly-deep.yml`, and `test.yml`'s core tests) have not run in the last month (last runs 2026-09-27 and 2026-10-03), so whether the six are red on CI or only under the local wrapper is unknown.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Why the default Canvas reader raises `RecoveryRequired('raw_source_selection_changed')` in these tests is found, and whether CI's environment hits it too is checked
- [ ] #2 The six tests pass on dev under the scratch-profile wrapper, without enabling Canvas in any shared or real config and without hiding a reader failure the product should report
- [ ] #3 The whole file runs in a PR lane, so a red there shows on PRs
<!-- AC:END -->
