---
id: TASK-19170
title: Kimi response-side and UI surfaces still pin the literal kimi-k3 id
status: Done
assignee:
  - '@claude'
created_date: '2026-08-20 20:34'
updated_date: '2026-08-20 21:31'
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
- [x] #1 Each literal site above is either converted to a family predicate consultation or documented with probe evidence for why the exact id is correct
- [x] #2 Kimi/GLM family members added by a new release get a sensible reasoning-effort option list in Settings without a code edit
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Wire probes (standalone curl, real Moonshot key): does kimi-k2.6/kimi-latest/kimi-k3 return reasoning_content with reasoning_effort set; multi-turn follow-up WITH vs WITHOUT prior reasoning_content replayed (reject/degrade/dont-care); tool-call loop leg if decisive. GLM: no key -- Settings-list question decided client-side; replay question recorded unprobeable.\n2. Commit probe verdicts BEFORE fixes.\n3. Fixes per evidence: new/extended family predicate(s) in model_capabilities.py for the response-side preserved-thinking question; convert provider_continuation.py parse gates, moonshot.py candidate/replay, console_chat_controller keep_all, plus the same-rule companion validators (agent_runtime, ChaChaNotes_DB, Character_Chat_Lib, chatbook_importer) in lockstep; Settings derives effort options from the family predicates.\n4. Red-first pins per changed site, kimi-k3/glm-5.2 controls, mutations with Edit restores; live kimi-k2.6 multi-turn continuation through the production seam + dev control.\n5. Hygiene: report, notes, PR against dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
COMPLETE (2026-08-20, branch fix/task-19170-ui-model-pins; full report: Docs/superpowers/plans/2026-08-20-task-19170-report.md).

STEP 1 (committed 2234cf50e BEFORE any fix): wire probes with the real Moonshot key -- every versioned kimi id (k2.5, k2.6, k2.7-code, k3) returns reasoning_content with AND without reasoning_effort; kimi-latest returns none; multi-turn and tool-loop follow-ups answer 200 both WITH and WITHOUT the prior reasoning_content replayed (accepted, never required; k2.6 even ignores it token-wise on plain turns). GLM wire unprobeable (no Z.ai key) -- recorded; the Settings question was decidable client-side.

FIX (AC #1): new RESPONSE-side predicate moonshot_model_returns_reasoning_content (versioned kimi family; deliberately narrower than the request-side predicate -- kimi-latest excluded). Converted: provider_continuation no-calls-round parse gate; moonshot.py candidate branches (via a non-blank _preserved_reasoning guard) and _apply_continuations (now SHAPE-branched, model literal eliminated); console keep_all; plus the same-rule companions (agent_runtime final-update + first-create, ChaChaNotes owner rule, chat-history import, chatbook import -- the latter two gained no-calls shape guards protecting pre-19170 durable data). DOCUMENTED-CORRECT, kept: the parse invariant 'complete k3 checkpoints must end with the final reasoning round' stays k3-pinned -- widening it would invalidate stored pre-19170 k2.x complete checkpoints (durable-data evidence, pinned by a dedicated test). GLM replay: unprobeable, zai.py response side untouched (it has no glm-5.2 response literal).

AC #2: Settings derives the curated Kimi/GLM effort lists from the 18803 REQUEST-side family predicates, so kimi-k2.6/kimi-k3-turbo/kimi-latest and glm-5.3/glm-5.2-air get the values their builders accept with no code edit; guidance copy family-aware; constants renamed KIMI_/GLM_REASONING_EFFORT_SELECT_OPTIONS; user guide aligned. Also fixed a pre-existing dev red in test_settings_kimi_zai (pin predating 18803's wire-verified 'medium').

EVIDENCE: red-first 30 failed/418 passed with sites untouched -> 448 passed; cluster gate 1884 passed/2 failed -- both fails are the TASK-18801 manifest reds, digest-IDENTICAL on a detached clean-base worktree. Twelve mutations (literal restores, predicate widening/drops, shape-guard drops) kill 9/22/1/1/2/1/1/7/1/1/1/1 pins, all Edit-restored with empty git diff. LIVE through chat_api_call: kimi-k2.6 two-turn -- complete checkpoint preserving 385 reasoning chars, replayed verbatim in the turn-2 wire payload, '255'->'510'; kimi-k3 control identical live; clean-base control: checkpoint None, nothing replayed (the gap, live).
<!-- SECTION:NOTES:END -->
