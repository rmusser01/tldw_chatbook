---
id: TASK-38692
title: "Retire the unloaded legacy sheets _new_ingest.tcss and _chatbooks_improved.tcss and the tests that pin them"
status: To Do
created_date: 2026-11-03 11:40
labels:
- follow-up
- css
- tests
priority: low
references:
- tldw_chatbook/css/features/_new_ingest.tcss
- tldw_chatbook/css/features/_chatbooks_improved.tcss
- Tests/UI/test_non_obscuring_focus_contract.py
- Tests/UI/test_design_token_governance.py
dependencies:
- TASK-36523
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the round-5 wave of the 2026-10-23 burn-down (TASK-36523, PR #3370, 2026-11-03). Checked on dev 9bbd5c2515.

TASK-36523 deleted the unloaded `css/main.tcss`. It kept two more unloaded sheets, `features/_new_ingest.tcss` and `features/_chatbooks_improved.tcss`, because tests pin them as legacy files. Neither sheet is in `build_css.py`'s manifest, and no loader names them. Outside the sheets themselves, they are referenced only by:
- `Tests/UI/test_non_obscuring_focus_contract.py`: `NEW_INGEST` (:91), `CHATBOOKS_IMPROVED` (:106), `SOURCE_ONLY_CSS_MODULES` (:124), and the contract tests that read them (:1416, :1449, :1620, :1902);
- `Tests/UI/test_design_token_governance.py`: `_LEGACY_SHEETS` (:55-79);
- a docstring in `UI/Chatbooks_Window_Improved.py` (:332), which says the chatbooks sheet "is NOT in the build manifest".

Draft PR #3342 edits `test_non_obscuring_focus_contract.py`, so this work should follow it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both sheets are deleted, and the app's loaded stylesheets are unchanged (the generated bundles reproduce byte for byte)
- [ ] #2 No test or docstring names either sheet, and the remaining focus-contract and token-governance tests pass
<!-- AC:END -->
