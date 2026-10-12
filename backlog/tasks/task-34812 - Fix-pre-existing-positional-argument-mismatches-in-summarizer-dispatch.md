---
id: TASK-34812
title: Fix pre-existing positional-argument mismatches in summarizer dispatch (kobold, tabbyapi, ollama)
status: To Do
assignee: []
created_date: '2026-10-12 10:05'
labels: [bug, providers]
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-33245 dispatch-table review (verified mechanically against the pre-refactor elif chain, which carried the same mismatches): three local summarizer routes pass arguments positionally onto signatures whose parameter order does not match, so values land on the wrong parameters. The TASK-33245 refactor preserved them bug-for-bug deliberately; the new `_SUMMARIZER_DISPATCH` table is now the single, testable place to correct them.

- **kobold**: dispatch passes `(text, api_key, cp, temp, sm, st)` onto `summarize_with_kobold(input_data, api_key, custom_prompt_input, system_message=None, temp=None, kobold_api_ip=..., streaming=False)` — so `system_message=temp`, `temp=system_message`, the streaming bool lands on `kobold_api_ip`, and `streaming` stays False.
- **tabbyapi**: passes `(text, cp, temp, sm, st)` onto `summarize_with_tabbyapi(input_data, custom_prompt_input, system_message=None, api_key=None, temp=None, ...)` — so `system_message=temp`, `api_key=system_message`, `temp=streaming`.
- **ollama**: the 7-positional call's last value binds `model=streaming`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All three routes pass every argument by its correct parameter (keyword or correctly-ordered positional), verified against each function's signature.
- [ ] #2 Regression tests pin the corrected routing for all three providers through `_dispatch_to_api`/`analyze`.
- [ ] #3 Targeted local-summarizer suites pass (kobold/tabby config tests included).
<!-- AC:END -->
