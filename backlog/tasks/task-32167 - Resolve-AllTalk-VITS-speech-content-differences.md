---
id: TASK-32167
title: Resolve AllTalk VITS speech-content differences
status: To Do
assignee: []
created_date: '2026-09-09 08:19'
labels: []
dependencies:
  - TASK-32152
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Real macOS CPU playback with the pinned AllTalk V2 VITS model completed for p225 and p226, but full transcripts from two recognizers retained lexical differences. Determine which differences are spoken-content defects versus recognition uncertainty before changing recommended voices or claiming complete content qualification. The current evidence is preserved under Docs/QA/tts-macos-burndown-2026-09-09/providers.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Complete retained p225 and p226 clips receive independent listening or suitably independent recognition review with raw disagreements preserved.
- [ ] #2 Any confirmed provider or application defect is fixed and reproduced with the same text and exact model and speaker provenance; recognition-only differences remain explicitly classified.
- [ ] #3 Playback and cleanup evidence remain separate from content quality and any voice recommendation names the qualified speaker and model without silently remapping saved choices.
<!-- AC:END -->
