---
id: TASK-78
title: OpenAI-Compatible Model Discovery
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-06-04 16:58'
labels:
  - providers
  - settings
  - console
  - model-discovery
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Allow users to manually discover models from configured OpenAI-compatible provider endpoints, use them immediately, and explicitly persist selected model IDs to the existing provider model list.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Users can discover models from an eligible configured OpenAI-compatible provider.
- [ ] #2 Discovered models are available in Settings and Console for the current app session.
- [ ] #3 Users can explicitly save selected discovered model IDs to the existing top-level providers list.
- [ ] #4 Unsupported endpoints and ambiguous provider config keys show safe recovery messages.
- [ ] #5 Focused automated tests cover discovery parsing, cache, merge, persistence, and UI states.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/002-openai-compatible-model-discovery.md
Reason: Defines provider/config/catalog boundaries and persistence policy.

1. Add discovery contracts and provider key resolution tests.
2. Add OpenAI-compatible endpoint discovery parsing and safe error handling.
3. Add runtime cache merge and persistence helpers.
4. Wire the local provider catalog and scope service.
5. Add Settings discover and save workflow.
6. Add Console merged model consumption and warnings.
7. Run focused tests and manual UI QA.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added Settings Providers & Models discovery workflow: discover, save selected, clear runtime cache, safe status/recovery copy, and discovered-model selection list.
- Added mounted Settings regressions for eligible controls, explicit selected-model persistence, and ambiguous provider-key recovery without endpoint leakage.
- Captured actual textual-web/CDP Settings evidence for idle controls and safe discovery recovery; user approval is pending.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
