---
id: TASK-35649
title: Six class-level DEFAULT_CSS declarations are outside the allowlist on dev
status: To Do
created_date: 2026-10-20 04:30
dependencies: []
labels:
- follow-up
- css
- tests
priority: medium
references:
- Tests/UI/test_widget_css_consolidation.py
- tldw_chatbook/UI/Evals/skill_eval_detail.py
- tldw_chatbook/UI/Evals/skill_eval_panel.py
- tldw_chatbook/UI/LLM_Management/llamacpp_setup_view.py
- tldw_chatbook/UI/Wizards/first_run_voice_pickers.py
- tldw_chatbook/Widgets/Library/library_file_notes_workspace.py
- tldw_chatbook/Widgets/session_summary_dialog.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the review of Phase 9's rebase onto dev 88fe3108e5 (2026-10-20). `Tests/UI/test_widget_css_consolidation.py::test_class_level_css_stays_within_the_allowlist` fails on plain dev with six class-level `DEFAULT_CSS` declarations it does not allow:
- `UI/Evals/skill_eval_detail.py::SkillEvalDetail`
- `UI/Evals/skill_eval_panel.py::SkillEvalPanel`
- `UI/LLM_Management/llamacpp_setup_view.py::LlamaCppSetupView`
- `UI/Wizards/first_run_voice_pickers.py::VoiceOptionPicker`
- `Widgets/Library/library_file_notes_workspace.py::FileNotesHistoryDialog`
- `Widgets/session_summary_dialog.py::SessionSummaryDialog`

The guard keeps widget CSS in the generated sheets, where the boot CSS budget and the parse cache can see it. The test runs in no PR lane, so dev's CI never reported it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 test_class_level_css_stays_within_the_allowlist passes on dev
- [ ] #2 Each of the six widgets' rules lives in the generated sheets (or its allowlist entry has a recorded reason), and preflight's stylesheet checks pass
- [ ] #3 Each widget renders the same before and after
- [ ] #4 The allowlist test runs in a PR lane, so the next unlisted declaration fails a PR
<!-- AC:END -->
