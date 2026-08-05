---
id: TASK-472
title: 'Onboard prompt_selector UI analysis-prompt templates to the Internal Prompts registry'
status: To Do
assignee: []
created_date: '2026-07-22 22:10'
labels:
  - internal-prompts
  - enhancement
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred from the program (picker UX, different feature). Widgets/prompt_selector.py holds ~30 hardcoded system+user analysis-prompt templates keyed by media type. Decide whether these belong in the Internal Prompts registry or remain a separate picker concern; if onboarded, they need a UX that fits a keyed template set rather than the single-prompt editor.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A decision is recorded on whether prompt_selector templates join the registry or stay separate
- [ ] #2 If onboarded: templates are registry-backed with parity tests and a suitable editing UX
- [ ] #3 If not: the rationale is documented and the templates left as-is
<!-- AC:END -->
