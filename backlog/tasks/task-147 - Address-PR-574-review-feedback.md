---
id: TASK-147
title: Address PR 574 review feedback
status: Done
assignee: []
created_date: '2026-06-30 15:12'
updated_date: '2026-06-30 15:21'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address PR review findings and deterministic UI failures for the Console API key recovery action without changing the approved user-facing recovery flow.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Review feedback for repeated Configure API+API Key literals is addressed
- [x] #2 Review feedback for label-coupled API-key styling is addressed
- [x] #3 Gemini empty-tooltip recovery action concern is covered and fixed
- [x] #4 Deterministic PR UI failures are reproduced and resolved
- [x] #5 Relevant UI tests and diff checks pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: These changes refine tests and helper behavior within the existing Console recovery/UI contract; they do not alter storage, provider boundaries, service contracts, or long-lived architecture.

1. Replace repeated test literals with the existing Console recovery label constant.
2. Add a regression test for provider_action_label with empty tooltip and update the helper to return that pair directly.
3. Decouple API-key warning styling from label text by passing an is_api_key_recovery boolean from semantic field context.
4. Preserve staged handoff suggested prompts by writing them into the active Console session draft before UI sync reloads the composer.
5. Update stale clean-run recovery assertions to target the Workbench recovery callout instead of hidden legacy selectors.
6. Run failing tests, touched Console suites, and diff checks; then update notes and push.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Addressed PR #574 review feedback by replacing repeated test-side recovery label literals with `CONSOLE_PROVIDER_CONFIGURE_API_KEY_LABEL`, returning provider recovery labels even when tooltips are empty, and decoupling API-key warning styling from display copy via an `is_api_key_recovery` boolean. Fixed deterministic UI failures by persisting staged handoff suggested prompts into the active Console session draft before the sync pass reloads the composer, and by updating the clean-run setup-state test to assert the visible Workbench recovery callout instead of hidden legacy selectors.

Verification:
- Reproduced the PR UI failure slice locally: 3 deterministic failures.
- Targeted review/CI slice: 12 passed.
- Expanded touched/failed UI files: 695 passed.
- `git diff --check` passed.
<!-- SECTION:NOTES:END -->
