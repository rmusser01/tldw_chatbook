---
id: TASK-41714
title: "Loguru calls pass %s format strings in about ten modules, so their values never reach the log"
status: To Do
assignee: []
created_date: 2026-11-18 06:57
updated_date: 2026-11-18 06:57
labels:
- follow-up
- logging
priority: low
references:
- tldw_chatbook/app_navigation.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41500 and its review (round 16), widened by the filing review. Loguru does not substitute `%s`, so these calls print a literal `%s` and drop their values, for example: `app_navigation.py` (such as "Reusing installed screen instance ... route: %s"), `app.py` ~1252/4337/4369, `LLM_Calls/LLM_API_Calls.py` ~341 and five more, `Chat/console_runtime.py` ~6922, `UI/Screens/chat_screen.py` ~7607, `Library/local_media_chunk_tool_service.py` ~990, `UI/Settings_Modules/manual_sync.py` ~62, `Chat/prompt_template_manager.py` ~86, `UI/Wizards/first_run_appearance_step.py` ~179, `UI/Screens/scheduling/conflicts_tab.py` ~309 and `Event_Handlers/LLM_Management_Events/llm_management_events.py` ~128. `Tests/test_stdlib_logging_format_style.py` guards the opposite case.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every loguru call in the package that passes arguments uses `{}` placeholders or f-strings and logs its values
- [ ] #2 A repo-wide test fails on a loguru call that passes `%s` with arguments, red before the change
<!-- AC:END -->
