---
id: TASK-39072
title: >-
  Console: two UI tests race the loop after their own action (the fleet
  pause-copy toggle and the workspace switcher open)
status: To Do
assignee: []
created_date: '2026-11-06 17:16'
labels:
  - console
  - tests
  - flaky
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two Console UI tests fail on dev intermittently because they act before work they started has landed. (1) Tests/UI/test_console_automatic_work_pause.py::test_pause_copy_is_painted_with_no_live_children_and_survives_view_clear calls _sync_console_agent_section, closes the fleet section with set_open(False), scrolls, then clicks the fleet toggle expecting it to open. A coalesced fleet sync (_run_coalesced_console_agent_fleet_sync -> _sync_console_agent_section -> _sync_console_fleet_surface_flags) often lands between the set_open(False) and the click and re-opens the section, so the click closes it and the notice has no region (the helper's own assertion message then raises AttributeError on the app's missing region). Instrumented 2026-11-06 on dev 2684e5453f and on TASK-36571.1's branch: the same set_open sequence on both trees, failing 3 of 4 probe runs on dev and 2 of 4 on the branch; whole-file interleaved runs failed 2-3 of 7 params in most runs on both. (2) Tests/UI/test_console_workspace_lifecycle.py's switcher tests press Alt+W and pause 0.2 s before querying the modal's Rename/Archive buttons; the switcher is pushed by a worker after two off-loop registry reads (about 50-60 ms each), and when the test runs after another in the same process the modal is often not yet mounted (is_mounted False, no buttons; the buttons appeared 650-940 ms after the press in the failing runs). Whole-file interleaved runs failed in 3 of 8 runs on each tree. Both run in the CI Tests/UI shards.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The pause-copy test waits until the fleet section's sync has settled before closing it, and fails if the click does not open the section showing the pause notice
- [ ] #2 The workspace lifecycle tests wait for the switcher's rows (or its mount) instead of a fixed pause, and still fail when the Rename or Archive control is missing
- [ ] #3 Each file passes 10 of 10 whole-file runs on dev with two other pytest processes running
<!-- AC:END -->
