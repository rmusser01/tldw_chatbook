---
id: TASK-32159
title: Honor configured Kokoro engine and voices in the Speech Lab
status: Done
assignee:
  - '@codex'
created_date: '2026-09-09 06:09'
updated_date: '2026-09-09 08:40'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live qualification found the fresh Speech Lab ONNX switch overrides the configured PyTorch engine. Official non-English voice choices also need to remain selectable through normal catalog projection.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A fresh Speech Lab uses the configured Kokoro ONNX or PyTorch engine, and the generated request matches the displayed switch; an explicit session toggle still overrides it.
- [x] #2 Official available Kokoro language voices can be selected without being silently replaced by an English default during catalog projection.
- [x] #3 Mounted targeted regressions and real speech validation cover inherited engine and non-English voice selection; saved global configuration is not changed by Lab use.
- [x] #4 The explicit Kokoro language selector includes Hindi, Italian and Brazilian Portuguese as well as the existing supported languages, and emits the selected language with its voice.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A; existing ADR-039 Global/Studio TTS ownership and ADR-140 official Kokoro runtime apply.
Reason: Repair inherited initial controls and the static official voice catalog within existing settings and provider boundaries.

1. Add a mounted failing regression for fresh Kokoro engine inheritance and verify exact official voice IDs against pinned runtime assets.
2. Seed the session engine switch from its configured setting while preserving explicit user toggles; extend the existing static voice choices for official supported language voices as necessary.
3. Exercise catalog projection and actual generated request, including a non-English configured voice; ensure no settings write or generation occurs during mounting.
4. Rerun targeted UI tests and serialized real CPU/MPS/ONNX playback using the corrected harness that applies the same axis seeds as the production Lab owner.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The fresh Speech Lab engine switch now inherits the configured Kokoro engine while preserving explicit session toggles. The static catalog contains the 54 official voices, and the language selector exposes all nine supported language choices including Hindi, Italian and Brazilian Portuguese. Catalog projection preserves the requested voice and generated engine/language selection; global settings remain unchanged. Existing ADR-039/140 apply; no new settings ownership boundary was introduced.

Mounted UI regressions, independent review and real CPU/MPS/ONNX English plus non-English playback pass. The initial harness voice mismatch was separately traced to missing production axis-default seeding and retained as a harness failure, not a product voice regression. The final installed wheel also passed playback and Stop/recovery; source/voice identity and unchanged user config are recorded in Docs/QA/tts-macos-burndown-2026-09-09/{kokoro-english,kokoro-languages,integration}/README.md.
<!-- SECTION:NOTES:END -->
