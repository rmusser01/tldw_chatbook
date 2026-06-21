---
id: TASK-130
title: Console provider and model configuration UAT
status: To Do
assignee: []
created_date: '2026-06-21 00:36'
labels:
  - console
  - providers
  - settings
  - uat
dependencies:
  - TASK-128
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify and harden provider, model, endpoint, and model-setting configuration from Settings through Console runtime use for providers already supported by chat_api_call().
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 User can select or manually enter a provider supported by chat_api_call() without selected text being clipped or obscured.
- [ ] #2 User can select or manually enter a model and see the effective provider/model reflected consistently in Settings, Console rail, and send readiness.
- [ ] #3 Endpoint and credential-source fields are visible, editable where allowed, and preserved according to the existing Settings ownership contract.
- [ ] #4 Model settings, including sampling and provider-specific controls, are visible and editable without hiding focused input text.
- [ ] #5 Providers without native streaming support can complete as a single assistant message instead of being treated as broken.
- [ ] #6 Focused regression coverage verifies provider selection, model entry, endpoint preservation, model setting persistence, and non-streaming fallback.
- [ ] #7 Rendered CDP/Textual-web evidence is captured before approval and PR completion.
<!-- AC:END -->

## Parallel Ownership

Owns Settings-to-Console provider/model configuration, readiness signals, model settings, endpoint/credential field behavior, and streaming fallback verification. Avoid chat tab lifecycle, message actions, and workspace conversation ownership except through stable public seams.

ADR required: no, unless implementation introduces new provider configuration ownership or runtime boundary rules.
ADR path: N/A unless implementation planning identifies a contract change.
Reason: Existing provider and Settings contracts should be exercised and hardened, not redesigned.
