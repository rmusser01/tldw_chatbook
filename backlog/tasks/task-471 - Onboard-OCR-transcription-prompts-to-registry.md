---
id: TASK-471
title: 'Onboard OCR / transcription prompts to the Internal Prompts registry'
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
Deferred from the program (flagged model-specific/brittle). The docext VLM extraction prompt, image-analysis prompt, and Qwen2Audio transcription chat-template prompts in Local_Ingestion are hardcoded. Evaluate whether they can be safely registry-backed (some are model-specific chat templates) and onboard the ones that can.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Onboardable OCR/image/transcription prompts are registered with byte-identical defaults
- [ ] #2 Model-specific chat-template prompts that cannot be safely edited are documented as intentionally excluded
- [ ] #3 Call sites resolve via the registry where onboarded
<!-- AC:END -->
