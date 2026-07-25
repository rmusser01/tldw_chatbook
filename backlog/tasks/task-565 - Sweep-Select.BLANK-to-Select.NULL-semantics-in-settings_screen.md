---
id: TASK-565
title: Sweep Select.BLANK to Select.NULL semantics in settings_screen
status: Done
assignee:
  - '@claude'
created_date: '2026-07-25 07:57'
updated_date: '2026-07-25 15:58'
labels:
  - settings
  - rag
  - tech-debt
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Final 541 review confirmed Select.BLANK does not exist on Textual 8.2.7 (silently resolves to Widget.BLANK == False). Four sites in settings_screen.py compare/compose with it; the compose fallback (~:8405, SP3-era) would raise InvalidSelectValueError at mount if a hand-corrupted config pointer names a nonexistent profile. Others are dead comparisons degrading UX copy.

PARTIAL DELIVERY in PR #863 (Qodo review): the compose fallback (was ~:8405) and the profile-Select change handler (was ~:10999) were fixed there, both with regression tests. Remaining sites for this task: `_library_rag_selected_profile_id` (~:8106, blank selection + Set active yields an adapter-level error instead of a friendly notice) and `_select_value_text` (~:5540, provider category).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All Select.BLANK usages in settings_screen.py replaced with Select.NULL-correct logic
- [x] #2 Corrupt active-profile pointer no longer crashes Settings mount (regression test) — delivered in PR #863
- [x] #3 Blank selection + Set active yields a friendly notice instead of an adapter error
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add regression tests for `_library_rag_selected_profile_id` and `_select_value_text` blank-select handling (Select.NULL, not the nonexistent Select.BLANK).
2. Fix both sites in settings_screen.py to compare against `Select.NULL`.
3. Confirm the existing "Choose a profile first." friendly-notice path in `_trigger_library_rag_profile_set_active` is now reachable.
4. Run the RAG profile region test file green.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed the two remaining Select.BLANK comparisons (settings_screen.py:5540 `_select_value_text`, :8106 `_library_rag_selected_profile_id`) -- Select.BLANK does not exist on Textual 8.2.7 and silently resolves to Widget.BLANK (False), so the comparison never matched the real blank sentinel Select.NULL. Both now compare against Select.NULL.

Effect: `_library_rag_selected_profile_id()` now correctly returns None when the profile picker sits on its blank row, which lets the pre-existing "Choose a profile first." warning notice in `_trigger_library_rag_profile_set_active` fire instead of the stringified sentinel ("Select.NULL") being passed to `activate_profile()` as a bogus id. `_select_value_text` now renders "" for a blank provider Select instead of the literal text "Select.NULL", with real values unaffected.

AC #2 (mount-crash fix) was already delivered in PR #863 -- left as-is, no changes needed there.

Tests added to Tests/UI/test_settings_rag_profile_region.py: test_select_value_text_treats_select_null_as_blank, test_select_value_text_still_renders_real_values_unchanged (regression guard for no behavior change on real values), test_library_rag_selected_profile_id_returns_none_for_select_null (full pilot mount, real Select widget), test_set_active_with_blank_selection_shows_friendly_notice_not_adapter_error (AC3, asserts the friendly notice fires and _dispatch_rag_set_active is never called). All 4 verified RED before the fix, GREEN after; full file 111 passed.
<!-- SECTION:NOTES:END -->
