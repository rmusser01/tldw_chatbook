---
id: TASK-41061
title: "Native MCP event tests are red on dev, and the file does not finish at -n 4"
status: In Progress
assignee:
- '@claude'
created_date: 2026-11-15 02:03
updated_date: '2026-11-26 05:18'
labels:
- follow-up
- architecture-review-2026-10-25
- tests
- console
- mcp
priority: high
references:
- Tests/Chat/test_native_v2_mcp_events.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On dev ecad404bd4 (2026-11-15) (bootstrap profile, `-n 4`), five nodes of `Tests/Chat/test_native_v2_mcp_events.py` fail, and the file does not finish at `-n 4`:
- `test_unqualified_ordinary_mcp_keeps_real_existing_dispatch`: the submit is not accepted.
- `test_selected_deny_never_reserves_a_native_mcp_worker`: execution owners are still reserved after a selected deny.
- `test_revocation_during_actual_nested_capture_discards_without_host_leak`: raises `MixinSourceChangedError: mixin_session_changed`.
- `test_real_mcp_lifecycle_publish_and_operation_finish_return_are_joined`: times out.
- `test_accepted_context_uses_positive_original_drain_without_reviving_dispatch[negative_drain]` fails about 12 s in, run alone, and the pytest process then never exits (TASK-41028 has the same shape). At `-n 4` the file reports 42 of 43 nodes and stalls past the 120 s timeout; on 2026-11-15 the node that never reported was `test_passive_accepted_mcp_context_still_refuses_actual_saved_grant_revocation[mcp]`, which passes alone. These two findings come from the review of this filing (plain runs, no trap).

The architecture-review fleet runs on dev beab5c1e4e (2026-11-13) saw the first four red. If the product is wrong, an ordinary MCP dispatch is refused, or a denied MCP tool still holds a worker slot. A file that cannot exit is a hang pytest-timeout cannot interrupt, so it is filed high, like TASK-41028. The file runs on no PR lane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The task notes say for each node whether the product or the test is wrong, and name the commit
- [ ] #2 A pytest run of Tests/Chat/test_native_v2_mcp_events.py alone, serially and at `-n 4`, runs to completion and the pytest process exits on its own within the run's timeout
- [ ] #3 Every node of Tests/Chat/test_native_v2_mcp_events.py passes on dev at `-n 4` (a node may first be deleted or retargeted, with its reason in the task notes); a fixed product defect is shown red first
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-run the file at base 6e41919880 and at the #3480 merge (e10ce63fe8) to name each red node's cause and the commit that introduced it (scratch probes only, no product edits).
2. Decide per node whether the product or the test is wrong; fix a product defect with its test red first, repair a drifted test keeping its assertions, and record any deleted or retargeted node with its reason.
3. Find what keeps the process alive after its summary (a non-daemon thread or an un-awaited task) and make the file exit on its own, serially and at `-n 4`.
4. Lane the file in the recovered-suites step, timed with the step's flags.
<!-- SECTION:PLAN:END -->
