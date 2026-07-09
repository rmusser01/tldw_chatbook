---
id: TASK-146
title: Expose Console API key recovery action
status: Done
assignee: []
created_date: '2026-06-30 04:38'
updated_date: '2026-06-30 04:45'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the Console setup-blocked state provide an explicit recovery action for missing provider API credentials so users can reach Settings and unblock sending.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Console setup-blocked state shows a visible Configure API+API Key action when the selected provider is missing an API key
- [x] #2 The recovery action opens Settings with the provider/model context and API key field intent
- [x] #3 Provider/model selection and endpoint recovery actions keep their existing behavior
- [x] #4 Textual UI tests cover the visible action label sizing and navigation context
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This is a visible affordance/copy change inside the existing Console recovery action and Settings navigation boundary; it does not change storage, contracts, provider boundaries, or long-lived application structure.

1. Update targeted tests to expect the Configure API+API Key recovery action for missing API keys.
2. Run the updated focused tests before implementation to confirm the old behavior fails.
3. Change the missing-API-key Console recovery action label/tooltip while preserving Settings navigation context.
4. Adjust layout styling only if the longer action label is clipped or undersized.
5. Run focused Console/Textual tests plus diff checks, then record implementation notes and mark acceptance criteria complete.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Updated the Console missing-API-key recovery action to use the explicit `Configure API+API Key` label and a Settings-focused tooltip while preserving the existing `provider-recovery` action path and `field="api_key"` navigation context. Kept provider/model selection and endpoint recovery actions unchanged. Aligned the empty transcript recovery action with the same label so the setup blocker exposes one consistent recovery affordance. Added/updated Textual UI coverage for the callout label, sizing, inspector copy, empty-state action, and Settings context.

Verification:
- Focused red run before implementation: 4 expected failures on the old `Add API Key` contract.
- Focused post-implementation run: 4 passed.
- Broader Console UI run: 422 passed.
- `git diff --check` passed.
<!-- SECTION:NOTES:END -->
