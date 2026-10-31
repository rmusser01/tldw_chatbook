---
id: TASK-36511
title: Console attach holds the UI thread 180-270 ms in synchronous reconcile work
status: To Do
assignee: []
created_date: 2026-10-23 22:44
labels:
- follow-up
- performance
- console
priority: medium
references:
- Tests/Performance/_compact_original_config_gate.py
- Tests/Performance/test_compact_setup_ownership.py
- Tests/Performance/test_compact_model_original_checked_read.py
dependencies:
- TASK-36390
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-36390 (2026-10-23). The compact-setup gates have 14 nodes in `test_compact_setup_ownership.py` and `test_compact_model_original_checked_read.py`. While a compact read is held, they require the shared loop to answer a probe within 100 ms.

Once TASK-36390 fixed the gate's install step, 3-5 of the 14 nodes failed per run at load average 9-18. The probe delays were bimodal: 0 ms, or 179-271 ms. Stacks (`TLDW_LOCAL_COMPACT_STACKS=1`) put the main thread in synchronous Console attach work at the probe:
- `_reconcile_console_after_attach`
- `_prepare_allocation_reconcile`
- `_sync_native_console_chat_ui` -> `_build_console_provider_selection`

Owner ruling (2026-10-23): keep the 100 ms bound and treat this attach work as a real UI-thread stall. The compact gates stay off a PR lane until they pass.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No synchronous segment of a Console attach blocks the UI thread for more than 100 ms on the compact gates' fixtures
- [ ] #2 All 14 compact-setup gate nodes pass 20 of 20 runs under parallel load
- [ ] #3 The compact-setup gate files run on a PR lane
<!-- AC:END -->
