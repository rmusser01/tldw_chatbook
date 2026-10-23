---
id: TASK-36388
title: Remove the Console provider context-window suppress queue and set_custom_value, which nothing uses
status: To Do
created_date: 2026-10-22 23:58
labels:
- follow-up
- cleanup
priority: low
references:
- tldw_chatbook/UI/Screens/settings_screen.py
- tldw_chatbook/Widgets/model_search_picker.py
- Tests/UI/test_console_provider_apply_defaults_flow.py
dependencies:
- TASK-35712
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the review of TASK-35712 (PR #3218, 2026-10-22). TASK-35712 deletes the dead `_set_provider_context_window_input_value` (settings_screen.py around :16136 on dev), which is the only thing that appends to `_provider_context_window_suppress_queue`. Once it lands, the queue has no writer; it was already effectively dead before. `set_custom_value` (`Widgets/model_search_picker.py:477`) has no callers either. Removing the queue touches a snapshot dataclass and `Tests/UI/test_console_provider_apply_defaults_flow.py` (around :2504 and :2647).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The suppress queue and set_custom_value are removed, along with any reads that can no longer see a value
- [ ] #2 Tests that referenced them are updated, and the covering tests pass
- [ ] #3 No module size ratchet row grows
<!-- AC:END -->
