---
id: TASK-31420
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

## Renumbering provenance

Filed as task-31383 on 2026-09-04 (PR #2383, branch
chore/console-interaction-follow-up-tasks). Renumbered to task-31420 the
same day: while the PR waited, dev landed its own task-31383 ("Make failed
Console replies offer Retry instead of Continue"), and under the 2026-08-21
owner rule (TASK-19601) the older arrival keeps the id. 31420 is the first
id above the highest task id on any remote branch or local worktree at the
time (31419). No dependency, doc, or code referenced the old id.
