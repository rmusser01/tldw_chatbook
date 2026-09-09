---
id: TASK-32151
title: Qualify registered Higgs Audio V2 on macOS CPU
status: Done
assignee:
  - '@codex'
created_date: '2026-09-09 05:50'
updated_date: '2026-09-09 08:40'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Mac has sufficient memory but lacks the registered Higgs V2 runtime and assets. Establish real provider evidence without silently substituting audio.cpp Higgs V3.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pinned V2 generation runtime, model and compatible tokenizer install and initialize in an isolated macOS ARM CPU environment, with exact provenance or a reproduced prerequisite blocker.
- [x] #2 Real production Lab and Console requests produce complete playable speech; recorded output and independent content verification distinguish transport from pronunciation.
- [x] #3 Cancellation, successor requests and close join native work without releasing the model early; any required production fix has a targeted regression.
- [x] #4 Configured Higgs dtype reaches the pinned V2 constructor through torch_dtype, while legacy dtype constructors remain supported; a failure-first regression proves both call contracts.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A; existing ADR-023 provider/runtime ownership and ADR-039/040 Global/Studio/Lab behavior apply.
Reason: The reproduced constructor-keyword correction preserves the registered provider boundary and runtime ownership; no new provider contract or runtime support policy is introduced.

1. Pin public primary-source runtime and asset requirements, preflight resources, and provision only task-owned environments, caches or containers.
2. Reproduce the pinned Higgs V2 torch_dtype mismatch with a focused failing production-adapter test, prefer that current constructor keyword while retaining the legacy dtype signature, and run targeted compatibility regressions.
3. Run a bounded real initialization and synthesis attempt while coordinating the exclusive inference/audio slot with other TTS validation.
4. Exercise production application routing, complete playback where the platform exposes a device, cancellation and successor behavior. Preserve exact logs, sources, model hashes and resource joins.
5. Retain explicit prerequisite evidence for unmet criteria and do not mark Done while required hardware/provider validation is missing.

Pinned constructor evidence: boson-ai/higgs-audio@05a145bb490501b534563bf51bf2f7aa2326b271; /private/tmp/tts-macos-burndown/higgs/import-probe.json.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Corrected the pinned Higgs V2 constructor contract by preferring torch_dtype while retaining legacy dtype signatures. Qualified the registered V2 CPU float32 runtime and exact model/tokenizer/Hubert assets through real Lab and Console paths. Failure-first adapter tests and independent provider review pass. Existing ADR-023/039/040 apply; the registered provider boundary is unchanged.

Both original bundled-PortAudio attempts retain complete generated speech but failed native sink cleanup. The isolated candidate library, identified by actual loaded-image SHA256, passed Lab playback, Console drain, active-generation cancellation, successor and clean process exit; all three successful clips matched full text in ASR. Stop joined 512 decoder forwards over 96.400 seconds, so responsive per-step cancellation is explicitly separate work. The default library was not changed. Runtime adoption and the closed serve-engine stopping-criterion contract have concrete follow-ups. Evidence and original failures: Docs/QA/tts-macos-burndown-2026-09-09/providers/higgs/{README.md,candidate-README.md,STOPPING-CRITERIA-CONTRACT.md}.
<!-- SECTION:NOTES:END -->
