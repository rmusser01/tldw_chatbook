---
id: TASK-41040
title: "Console worktree-recovery preview tests are red on dev: the previewed tools no longer match the live request"
status: To Do
created_date: 2026-11-15 02:03
labels:
- follow-up
- architecture-review-2026-10-25
- tests
- console
- workspaces
priority: medium
references:
- Tests/UI/test_console_worktree_recovery.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On dev ecad404bd4 (2026-11-15) (bootstrap profile, `-n 4`), `Tests/UI/test_console_worktree_recovery.py` fails about 50 nodes:
- **`test_controller_previews_match_live_worktree_schemas` (48 params).** Each fails one of two ways:
  - The first-request plan count is 2 where the test expects 3.
  - The previewed tool schemas differ from the live request's. For example, `calculator` appears where `read_agent_messages` is expected.
- **`test_controller_bridge_turn_child_write_visible_confirmation[True|False]`.** Both params fail on a missing file, and one also errors at teardown with a FleetChat scripting fault.

Architecture-review runs on 2026-11-13 also saw these nodes red on dev.

If the product is wrong, the tool list the Console previews for a recovered worktree turn is not the list it sends. The file runs on no PR lane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The task notes say whether the preview, the live request or the test is wrong, and name the commit
- [ ] #2 Every node of the file passes on dev at `-n 4`; a fixed product defect is shown red first
<!-- AC:END -->
