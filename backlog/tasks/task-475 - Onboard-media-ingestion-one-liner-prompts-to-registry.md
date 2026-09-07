---
id: TASK-475
title: 'Onboard media-ingestion one-liner prompts to the Internal Prompts registry'
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
Deferred from the program. Local_Ingestion Document_Processing / Image_Processing / audio_processing hold one-liner default prompts ('Please provide a comprehensive summary of this {type}.', image-analysis, chunk-summarize/combine). Onboard the stable ones to the registry.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The onboardable media-ingestion one-liners are registered with byte-identical defaults (parity tests)
- [ ] #2 Call sites resolve via the registry; existing custom_prompt overrides still win
- [ ] #3 A transport-boundary or call-site test proves an override is honored
<!-- AC:END -->
