---
id: TASK-31420
title: Register the ask_user restraint description in the internal-prompts registry
status: Done
assignee:
  - '@Robert'
created_date: '2026-09-04 19:28'
updated_date: '2026-09-05 00:05'
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
- [x] #1 The ask_user tool description is served from the internal-prompts registry with a stable key and the constant becomes its default
- [x] #2 The LocalToolSpec built by _default_specs reads the registry copy at spec-build time
- [x] #3 Existing ask_user tests pass unchanged and one test pins that a registry override reaches the spec
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add PromptSpec agents.ask_user_tool_description to Internal_Prompts/agents_prompts.py with the ASK_USER_DESCRIPTION literal as its default (no placeholders; contract note on restraint + default-ON gate).
2. local_tool_provider._default_specs reads the description through get_internal_prompt at spec-build time (lazy import; the constant stays the catalog default).
3. Tests: golden parity between the catalog default and ASK_USER_DESCRIPTION; an override via the resolver's config seam reaches the built spec; existing ask_user suites unchanged.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
agents.ask_user_tool_description is a PromptSpec in Internal_Prompts/agents_prompts.py whose default is byte-identical to ASK_USER_DESCRIPTION (pinned by test_ask_user_tool_description_matches_source_constant). _default_specs builds the ask_user LocalToolSpec from get_internal_prompt('agents.ask_user_tool_description') at spec-build time (lazy import, so Internal_Prompts stays off the boot path -- census unchanged at 966), falling back to the constant if the resolver returns empty. An [internal_prompts.agents] override reaches the built spec (test_a_registry_override_reaches_the_built_spec). No placeholders; the contract note explains the default-ON gate and the busy sentence.
<!-- SECTION:NOTES:END -->
