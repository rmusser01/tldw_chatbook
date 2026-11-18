---
id: TASK-41506
title: "test_screen_navigation.py uses 34 fixed 3-second waits and its route nodes are on no PR lane"
status: To Do
assignee: []
created_date: 2026-11-17 05:27
labels:
- follow-up
- tests
- navigation
priority: medium
references:
- Tests/UI/test_screen_navigation.py
dependencies:
- TASK-41127
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41127 (libtests, round 15). The file holds 34 fixed 3 s waits plus `_wait_for_initial_screen`; at load 85 its 12-node slice took 96.1 s serial, and its route nodes were left off lane.

2026-11-18, from the TASK-36952.3 media series' wide A/B (chain over dev 5e3c8b2bd4, 2026-11-17, before TASK-41127's wall-clock waits): five real-app route nodes failed on base and head alike against the fixed 3 s Library wait: `test_media_route_round_trips_to_the_library_media_row`, `test_search_route_round_trips_to_the_library_rag_row`, `test_boot_with_prompts_default_tab_lands_on_library_with_prompts_row_selected`, `test_prompt_receipt_owner_vetoes_real_app_navigation_until_settlement` and `test_research_workspace_runs_round_trip_restores_independent_context`. Over six interleaved serial rounds at load 70-160, the search route failed 2 of 6 on head and 1 of 6 on base, and the media route 2 of 6 on head and 3 of 6 on base. TASK-36952.3.7's reruns also saw `test_search_all_palette_command_lands_on_library_with_honest_toast` stay on ChatScreen with base code, landing 0 of 4 times even with the wait raised to 30 s, so that node has a navigation race beyond the fixed wait.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Fixed waits are replaced by condition waits with a bounded deadline
- [ ] #2 The route nodes run on a PR lane, with their measured serial time recorded
- [ ] #3 The file's wall-clock time at load under 40 falls measurably, with before and after in the notes
<!-- AC:END -->
