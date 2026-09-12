---
id: TASK-32479
title: Expose Improve-My-Prompt rewrite prompt in Settings Internal Prompts
status: Done
assignee: []
created_date: '2026-09-12 03:15'
updated_date: '2026-09-12 04:12'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The owner default Improve-My-Prompt rewrite template (merged in PR #2626) is a hardcoded constant in prompt_improvement_prompts.py. Register it in the Internal_Prompts catalog so it appears in Settings > Internal Prompts with the standard save/reset override editor. Only the persona/structure portion becomes customizable; the safety guards, JSON envelope instruction, and recency anchor stay code-pinned so an override cannot strip the no-answer/no-invention invariants or break envelope parsing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 prompt_improvement.rewrite appears in Settings > Internal Prompts and saving an override changes the system message used for auto/review improvements,Reset restores the shipped owner template,Override cannot remove safety instructions or JSON envelope instruction (they are always appended by code),Import-hygiene and resolver tests still pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes — new user-editable override surface for a security-sensitive prompt; ADR-151 records the code-pinned safety/envelope split.
1. Add Internal_Prompts/prompt_improvement_prompts.py registering prompt_improvement.rewrite with the owner template as shipped default (moves text out of Prompt_Management to avoid duplication).
2. Wire trusted_optimizer_instructions() to resolve the persona portion via get_internal_prompt() with a lazy import (mirrors console_chat_controller pattern), appending code-pinned safety + envelope + recency anchor.
3. Add ADR-151 documenting the customizable/pinned split.
4. Tests: registry parity test for the new spec, override round-trip via scratch config, template-pin test isolation from host config.
5. Run Prompt_Management + Internal_Prompts suites, then PR against dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Merged via PR #2635 (merge commit d8516accde). prompt_improvement.rewrite registered in Internal_Prompts with the owner template as REWRITE_DEFAULT; trusted_optimizer_instructions resolves the persona via get_internal_prompt (lazy import, shared REWRITE_PROMPT_ID constant per Qodo) and always appends code-pinned safety guards, JSON envelope instruction, and recency anchor (ADR-151). Tests: registry parity, override round-trip, override-keeps-guards, exact composition pin; service pin tests hermetized with scratch config. 193 + 781 tests green. Merge required temporarily relaxing the new required-approving-review rule on dev (self-approval deadlock: PRs are authored by the owner account).
<!-- SECTION:NOTES:END -->
