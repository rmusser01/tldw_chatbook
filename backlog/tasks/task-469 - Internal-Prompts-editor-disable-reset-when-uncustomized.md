---
id: TASK-469
title: 'Internal Prompts editor: disable Reset button when the prompt is not customized'
status: To Do
assignee: []
created_date: '2026-07-22 22:10'
labels:
  - internal-prompts
  - polish
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The editor modal's Reset button is always enabled. Reset on an uncustomized prompt is a harmless no-op (delete_settings_from_cli_config no-ops), but the spec intended Reset to be disabled/no-op when there is nothing to reset. Disable the button when override_state.customized is False for the prompt.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Reset is disabled (or clearly inert) when the prompt has no override / is not customized
- [ ] #2 Reset is enabled when the prompt is customized
- [ ] #3 A test covers both states
<!-- AC:END -->
