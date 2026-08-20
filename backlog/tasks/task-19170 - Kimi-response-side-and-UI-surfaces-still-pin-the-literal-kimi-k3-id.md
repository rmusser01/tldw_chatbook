---
id: TASK-19170
title: Kimi response-side and UI surfaces still pin the literal kimi-k3 id
status: To Do
assignee: []
created_date: '2026-08-20 20:34'
labels:
  - llm
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-18803 moved the Moonshot/Z.ai REQUEST builder gates onto family predicates in model_capabilities.py, but response-side and UI surfaces still branch on exact-id literals: provider_continuation.py _KIMI_K3_MODEL (reasoning-content checkpoint replay gates on model == kimi-k3, so a kimi-k2.6 turn that returns reasoning_content -- the wire accepts reasoning_effort across the kimi series, probe-verified in TASK-18803 -- does not get k3-style preserved-thinking replay), console_chat_controller.py keep_all (moonshot + kimi-k3 literal), moonshot.py _apply_continuations / checkpoint replay (resolution.model == _DEFAULT_MODEL), and settings_screen.py _model_profile_reasoning_effort_options (the curated Kimi/GLM effort option lists are shown only for the exact ids kimi-k3 / glm-5.2; other family members fall back to the generic OpenAI-flavored list whose values the builders reject client-side). These are staleness-shaped like the 18803 findings but are response-handling/UX questions, not request 400s, and need their own probe evidence (does kimi-k2.6 return reasoning_content? must it be replayed?) before predicating.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each literal site above is either converted to a family predicate consultation or documented with probe evidence for why the exact id is correct
- [ ] #2 Kimi/GLM family members added by a new release get a sensible reasoning-effort option list in Settings without a code edit
<!-- AC:END -->
