---
id: TASK-42259
title: "Late presses and toggles that Console modals resolve by list index or from current state can act on the wrong row"
status: To Do
assignee: []
created_date: 2026-11-22 00:17
updated_date: 2026-11-22 00:17
labels:
- follow-up
- console
- ui
- bug
priority: medium
references:
- tldw_chatbook/Widgets/Console/console_reaction_picker_modal.py
- tldw_chatbook/Widgets/Console/console_prompt_mixins_view.py
- tldw_chatbook/Widgets/Settings_Widgets/personal_context_review_modal.py
- tldw_chatbook/Widgets/Console/console_rewind_modal.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the recompose stream (round 19). Handlers that resolve a late user action by position or from current state, which the message keeper never forwards to (so dev already has the exposure), read at e5626ea043:
- `ConsoleReactionPickerModal._row_pressed` (`console_reaction_picker_modal.py:661`) maps the pressed button id to `self._row_ids.index(button_id)`. Row ids are index-based (`console-reaction-picker-row-{index}`, line 453) and are rebuilt on every re-filter (line 440), so a press that arrives after a re-filter selects whichever option now holds that index.
- `PersonalContextReviewModal.handle_checkbox_changed` (`personal_context_review_modal.py:610`) and `ConsolePromptMixinsView._toggle` (`console_prompt_mixins_view.py:246`) read the row from the index in the Checkbox id (`personal-context-review-select-{index}`) against the current list.
- Static-id actions act on whatever is current, for example `ConsoleRewindModal` Restore and Summarize on `self._selected`, and Delete and Save buttons.
No pilot reproduction was run; this is from reading the handlers.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At least one handler is reproduced with a pilot (a press or toggle queued across a re-filter or rebuild) and shown to act on the wrong row, or shown not to
- [ ] #2 Each handler listed resolves a late action to the row the user pressed (by a stable row identity), or refuses it, and a test per handler is red before the fix
- [ ] #3 Static-id actions whose target can change before the late press lands are listed in the task notes with the fix or the reason none is needed
<!-- AC:END -->
