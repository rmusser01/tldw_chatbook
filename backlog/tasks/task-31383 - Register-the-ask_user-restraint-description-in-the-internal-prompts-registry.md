---
id: TASK-31383
title: Register the ask_user restraint description in the internal-prompts registry
status: To Do
assignee: []
created_date: '2026-09-04 19:28'
labels:
  - console
  - agents
  - prompts
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The design spec (2026-08-19-console-user-interaction-design.md, section 5.1) asked for the ask_user tool description -- most of whose words say when NOT to ask -- to be registered in the internal-prompts registry alongside the other agent prompts rather than hardcoded at the call site, so it is inspectable and (once the registry's Settings surface lands) editable like every other prompt the app puts in front of a model. M2 (PR #2379) shipped it as the ASK_USER_DESCRIPTION constant in Agents/ask_user_questions.py because the PRD's acceptance criteria did not require the registration. With the gate defaulting ON, every user gets whatever this text says, so it deserves the same treatment as the registry's other 29 prompts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The ask_user tool description is served from the internal-prompts registry with a stable key and the constant becomes its default
- [ ] #2 The LocalToolSpec built by _default_specs reads the registry copy at spec-build time
- [ ] #3 Existing ask_user tests pass unchanged and one test pins that a registry override reaches the spec
<!-- AC:END -->
