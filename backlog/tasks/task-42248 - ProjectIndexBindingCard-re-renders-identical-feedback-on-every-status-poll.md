---
id: TASK-42248
title: "ProjectIndexBindingCard re-renders identical feedback on every status poll"
status: To Do
assignee: []
created_date: 2026-11-22 00:17
updated_date: 2026-11-22 00:17
labels:
- follow-up
- ui
- settings
- performance
priority: low
references:
- tldw_chatbook/Widgets/Settings_Widgets/project_index_binding_card.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the testsb stream (round 19). `ProjectIndexBindingCard._feedback` (`tldw_chatbook/Widgets/Settings_Widgets/project_index_binding_card.py:193`) calls `self.query_one(".project-index-feedback", Static).update(text)` with no check that the text changed, so each poll that produces the same feedback text re-renders the widget.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Feedback text equal to what the card already shows does not update the widget
- [ ] #2 A test counts `Static.update` calls across repeated polls with identical feedback and is red before the fix
- [ ] #3 Feedback that changes still renders at once
<!-- AC:END -->
