---
id: TASK-32113
title: Qualify additional audio.cpp models and verify quantization claims on macOS
status: Done
assignee:
  - '@codex'
created_date: '2026-09-09 00:41'
updated_date: '2026-09-09 02:04'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extend native macOS qualification beyond Supertonic F16 with accessible PocketTTS reference speech and an honestly identified Supertonic quantization experiment.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PocketTTS runs through managed application speech admission and real CPU/Metal playback using an authorized synthetic reference, or a precise runtime blocker is recorded.
- [x] #2 Supertonic artifact hashes and tensor types distinguish actual file quantization from filenames and runtime quantization.
- [x] #3 Any claimed successful new configuration has full speech content validation, source/model provenance, and joined owned-process cleanup.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: qualification of existing ADR-023/ADR-050 native runtime contracts and supported upstream configuration choices.

1. Pin model/binary provenance and prepare PocketTTS configurations using a nonpersonal synthetic reference.
2. Inspect purported Supertonic Q8 artifact hashes and GGUF tensor types; distinguish runtime quantization explicitly.
3. Run each meaningful configuration serially through managed app speech, real output and independent complete-file content checks.
4. Record exact successful configurations, upstream/runtime blockers, and joined owned-child cleanup in the QA report.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Qualified PocketTTS English Q8 using an authorized synthetic reference and Supertonic's selected-weight runtime Q8 setting on CPU and Metal: eight full Console replies, sixteen independently decoded source/device captures, exact source/device PCM matches, twelve managed child exits zero and complete cleanup. PocketTTS ASR matched 8/8 captures; Supertonic matched 5/8 strictly, with three complete/completes differences preserved as limitations.

Inspected the published Supertonic Q8 GGUF: it is byte-identical to the original file and contains 698 F32 plus 72 I64 tensors, with no Q8 tensors. Results distinguish the requested in-memory quantization setting from actual downloaded-file quantization; runtime tensor memory was not inspected. The advanced managed configuration path was exercised, not Guided cloning. Exact assets, revisions, transcripts, cleanup and residual gaps are in Docs/QA/tts-runtime-recovery-2026-09-09/native-validation.md and evidence-summary.json. ADR required: no; this validates existing ADR-023/050 contracts without changing implementation.
<!-- SECTION:NOTES:END -->
