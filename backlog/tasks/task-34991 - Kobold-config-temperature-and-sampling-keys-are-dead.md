---
id: TASK-34991
title: Kobold config temperature and sampling keys are dead
status: To Do
assignee: []
created_date: '2026-10-14 13:00'
labels: [bug, providers]
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from the TASK-34844 review: config.py builds kobold_api.temperature (default 0.7) and sibling sampling keys (top_p, max_tokens) that no code consumes — summarize_with_kobold goes straight to the 0.7 literal when temp is None even though kobold_legacy is already resolved in scope. The llama sibling consults its config section as the None default (Local_Summarization_Lib.py:352-361); kobold should do the same, or the dead keys should be removed from the config defaults so users stop configuring a no-op.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 temp=None consults the kobold config section (llama-style) or the dead config keys are removed, with the choice recorded.
- [ ] #2 A test pins the chosen behavior.
<!-- AC:END -->
