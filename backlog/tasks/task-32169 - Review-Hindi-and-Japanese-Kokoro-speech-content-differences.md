---
id: TASK-32169
title: Review Hindi and Japanese Kokoro speech-content differences
status: To Do
assignee: []
created_date: '2026-09-09 08:27'
labels: []
dependencies:
  - TASK-32149
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Real PyTorch and ONNX language playback now completes with model-aligned frontends. Full Whisper-small and Whisper-medium transcripts retain Hindi lexical differences and Japanese homophone or pronunciation uncertainty. Review the saved whole clips with suitable native-language listening before claiming content or pronunciation quality. Preserve the existing two-recognizer results and exact model and voice provenance under Docs/QA/tts-macos-burndown-2026-09-09/kokoro-languages.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Native-language review covers both complete Hindi hf_alpha and Japanese jf_alpha clips from PyTorch and ONNX and distinguishes recognition spellings from audible lexical or phoneme errors.
- [ ] #2 Any confirmed application frontend defect is reproduced and fixed without weakening content comparisons or silently changing the selected model and voice.
- [ ] #3 Model or pronunciation limitations remain explicitly documented with actionable voice or upstream follow-ups and raw source and playback evidence stays available.
<!-- AC:END -->
