---
id: TASK-78
title: OpenAI-Compatible Model Discovery
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-05 14:01'
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
- [x] #1 Users can discover models from an eligible configured OpenAI-compatible provider.
- [x] #2 Discovered models are available in Settings and Console for the current app session.
- [x] #3 Users can explicitly save selected discovered model IDs to the existing top-level providers list.
- [x] #4 Unsupported endpoints and ambiguous provider config keys show safe recovery messages.
- [x] #5 Focused automated tests cover discovery parsing, cache, merge, persistence, and UI states.
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
- Added Console merged-model consumption so runtime-discovered models appear in Console settings alongside saved models while saved model order remains first.
- Added unknown-capability warning metadata for runtime-discovered Console selections without blocking send when provider readiness is otherwise valid.
- Added focused Console regressions for saved-first ordering, runtime-discovered model selection, and visible `Capabilities unknown` summary copy.
- Captured actual textual-web/CDP Console evidence for `gpt-5` as a runtime-discovered model with `Credential: ready`; user approval is pending.
- Completed focused closeout verification: provider catalog discovery tests passed, Settings + Console UI tests passed, combined provider/UI sweep passed, and `git diff --check` was clean.
- Checked optional local endpoint QA at `127.0.0.1:9099`; no local OpenAI-compatible endpoint was reachable, so that manual send/discovery check remains documented as unavailable rather than blocking.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented OpenAI-compatible model discovery as a local manual workflow. Added provider list key resolution, endpoint discovery, runtime cache, saved/discovered merge, explicit Settings persistence, Console consumption, focused tests, ADR 002, and CDP screenshot evidence. Deferred native provider-specific discovery, server keyring credentials, and tldw_server catalog sync per the PRD.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
