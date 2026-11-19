---
id: TASK-36388
title: Remove the Console provider context-window suppress queue and set_custom_value, which nothing uses
status: Done
assignee:
- '@claude'
created_date: 2026-10-22 23:58
updated_date: 2026-10-23 16:43
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
- [x] #1 The suppress queue and set_custom_value are removed, along with any reads that can no longer see a value
- [x] #2 Tests that referenced them are updated, and the covering tests pass
- [x] #3 No module size ratchet row grows
<!-- AC:END -->

## Implementation Plan

1. Confirm no writer: grep the source and Tests for `_provider_context_window_suppress_queue`, `context_window_suppress_queue`, `set_custom_value`, including getattr/setattr strings and on_/watch_/action_ names.
2. Remove the queue: its `__init__` field, the `_VllmDefaultPresentationSnapshot.context_window_suppress_queue` field with its capture and restore, the clear in the widget-echo reset, and the dead queue check in `handle_model_context_window_changed`.
3. Remove `ModelSearchPicker.set_custom_value`.
4. Drop the queue from `test_vllm_default_late_ack_failure_restores_complete_provider_presentation`'s suppression tuple (it still checks the three live queues); run the covering tests (the picker's and the Settings context-window tests).
5. Lower settings_screen.py's ratchet row to its new exact size; ratchets and preflight.

## Implementation Notes

**What was removed.**
- Nothing wrote to `_provider_context_window_suppress_queue`. A grep of tldw_chatbook/ and Tests/ for the name, `context_window_suppress`, and getattr/setattr strings found only its own field, snapshot, restore, clear and read. It goes, together with:
  - `_VllmDefaultPresentationSnapshot.context_window_suppress_queue`, with its capture and restore;
  - its `.clear()` in the widget-echo reset;
  - the queue check at the top of `handle_model_context_window_changed`, which could never see a value.

  The three live queues (endpoint, env var, API key) are unchanged.
- `ModelSearchPicker.set_custom_value` had no callers: no direct call, getattr, string, or on_/watch_/action_ name. It goes. `normalize_custom_model_id` is still used elsewhere in the picker.

**Behaviour.** No change: the queue was always empty, so its check was always false.

**Tests.** `test_vllm_default_late_ack_failure_restores_complete_provider_presentation` drops the queue from its before/after suppression tuple and still checks the three live queues. It is a plain test that hits the local RecoveryRequired trap, so it was run under a temporary `@private_profile_test` marker, then reverted: 1 passed.

**Covering run, same command at HEAD and at dev 52b3207ff1.** The run covered:
- Tests/Widgets/test_model_search_picker*.py;
- the Settings context-window, Advanced, deep-link, qwencloud, scope-inspector, search-index and server-window files;
- Tests/Architecture/test_module_size_ratchet.py.

No failure at HEAD is absent at dev. What fails in both: the RecoveryRequired plain tests, the search-index pins, and eight other modules' ratchet rows that dev has outgrown. Four tests failed only in the first HEAD run, which shared the CPUs with a second pytest. Rerun alone they passed, and the three files rerun with `-n 4` show only the two context-window reds that dev has too.

**Ratchet.** settings_screen.py is 35,660 -> 35,647; its row is lowered. model_search_picker.py has no ratchet row (1,173 -> 1,164).

**Files.**
- `tldw_chatbook/UI/Screens/settings_screen.py`
- `tldw_chatbook/Widgets/model_search_picker.py`
- `Tests/UI/test_console_provider_apply_defaults_flow.py`
- `Tests/Architecture/test_module_size_ratchet.py`
