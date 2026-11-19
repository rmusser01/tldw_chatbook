---
id: TASK-41913
title: "The Console thinking disclosures tests are red locally and on no lane"
status: To Do
assignee: []
created_date: 2026-11-18 20:31
updated_date: 2026-11-18 20:31
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/UI/test_console_thinking_disclosures.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41509 (round 16). The tests named as replacement coverage for the deleted Settings thinking nodes are on no lane: `test_console_thinking_disclosures.py` (red locally on `RecoveryRequired: raw_source_selection_changed`), `test_model_thinking_visibility_defaults_on_and_fails_safe` and `test_show_model_thinking_is_canonical_immediate_and_rolls_back`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The three pass locally through the scratch wrapper
- [ ] #2 They run on a PR lane with their measured time
<!-- AC:END -->
