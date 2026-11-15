---
id: TASK-41061
title: "Four native MCP event tests are red on dev"
status: To Do
created_date: 2026-11-15 02:03
labels:
- follow-up
- architecture-review-2026-10-25
- tests
- console
- mcp
priority: low
references:
- Tests/Chat/test_native_v2_mcp_events.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On dev ecad404bd4 (2026-11-15) (bootstrap profile, `-n 4`), four nodes of `Tests/Chat/test_native_v2_mcp_events.py` fail:
- `test_unqualified_ordinary_mcp_keeps_real_existing_dispatch`: the submit is not accepted.
- `test_selected_deny_never_reserves_a_native_mcp_worker`: execution owners are still reserved after a selected deny.
- `test_revocation_during_actual_nested_capture_discards_without_host_leak`: raises `MixinSourceChangedError: mixin_session_changed`.
- `test_real_mcp_lifecycle_publish_and_operation_finish_return_are_joined`: times out.

The architecture-review fleet runs on dev beab5c1e4e (2026-11-13) saw the same four red. If the product is wrong, an ordinary MCP dispatch is refused, or a denied MCP tool still holds a worker slot. The file runs on no PR lane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The task notes say for each node whether the product or the test is wrong, and name the commit
- [ ] #2 The four nodes pass on dev; a fixed product defect is shown red first
<!-- AC:END -->
