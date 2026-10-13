---
id: TASK-34845
title: summarize_chunk passes rotated arguments into analyze (FIXME)
status: To Do
assignee: []
created_date: '2026-10-12 12:10'
labels: [bug, providers]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-34812 review (same species): summarize_chunk (Summarization_General_Lib.py ~2595, FIXME-marked, currently only called by tests) rotates the first three arguments into analyze() — text lands on api_name — and swaps temp/system_message relative to analyze's (api_name, input_data, custom_prompt_arg, api_key, system_message, temp, ...) order. Verify every live call path (or confirm it is dead), fix the rotation, and remove the FIXME.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 summarize_chunk's call into analyze passes every argument in the correct position, verified against analyze's signature.
- [ ] #2 Callers audited; a test covers the corrected path (or the function is deleted with the decision recorded if truly dead).
<!-- AC:END -->
