---
id: TASK-19170
title: Kimi response-side and UI surfaces still pin the literal kimi-k3 id
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-20 20:34'
updated_date: '2026-08-20 20:49'
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Wire probes (standalone curl, real Moonshot key): does kimi-k2.6/kimi-latest/kimi-k3 return reasoning_content with reasoning_effort set; multi-turn follow-up WITH vs WITHOUT prior reasoning_content replayed (reject/degrade/dont-care); tool-call loop leg if decisive. GLM: no key -- Settings-list question decided client-side; replay question recorded unprobeable.\n2. Commit probe verdicts BEFORE fixes.\n3. Fixes per evidence: new/extended family predicate(s) in model_capabilities.py for the response-side preserved-thinking question; convert provider_continuation.py parse gates, moonshot.py candidate/replay, console_chat_controller keep_all, plus the same-rule companion validators (agent_runtime, ChaChaNotes_DB, Character_Chat_Lib, chatbook_importer) in lockstep; Settings derives effort options from the family predicates.\n4. Red-first pins per changed site, kimi-k3/glm-5.2 controls, mutations with Edit restores; live kimi-k2.6 multi-turn continuation through the production seam + dev control.\n5. Hygiene: report, notes, PR against dev.
<!-- SECTION:PLAN:END -->
