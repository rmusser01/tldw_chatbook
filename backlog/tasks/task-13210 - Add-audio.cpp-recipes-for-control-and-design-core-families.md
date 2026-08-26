---
id: TASK-13210
title: Add audio.cpp recipes for control and design core families
status: To Do
assignee: []
created_date: '2026-08-09 17:39'
labels:
  - tts
  - audio-cpp
  - recipes
  - compatibility
dependencies:
  - TASK-13209
references:
  - backlog/decisions/050-audio-cpp-generated-model-setup-ownership.md
  - backlog/decisions/051-private-tts-clone-reference-assets.md
documentation:
  - Docs/superpowers/specs/2026-08-09-audio-cpp-guided-model-setup-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add exact release-0.5.1 recipes and evidence for fish_audio, higgs_audio_tts, omnivoice, qwen3_tts, and voxcpm2.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All 21 declared release-0.5.1 package entries across fish_audio (2), higgs_audio_tts (2), omnivoice (4), qwen3_tts (9), and voxcpm2 (4) have exact immutable recipe identities and no unreviewed package gap.
- [ ] #2 Each recipe declares its exact tts/clone capabilities, required and optional files, safe model fields, voice/reference combination and precedence, design/control defaults permitted by the typed contract, backend/platform tuples, compatibility state, and pinned recipe revision.
- [ ] #3 Design and control support remains a recipe-bounded typed surface rather than arbitrary options; unsupported upstream controls are omitted truthfully and cannot be smuggled through profile or request mappings.
- [ ] #4 Exact positive fixtures and adversarial missing/conflicting/ambiguous/version/path fixtures prove deterministic recognition, and generated-config/catalog tests cross-check every recipe's exact task and model identity.
- [ ] #5 Every tuple labeled Verified has provisioned real-process text/clone/control evidence as declared plus definitive cleanup; accelerated labels require device-specific evidence and no fallback failure may inherit a Verified label from another backend.
- [ ] #6 The support/accounting UI and documentation name this exact family/package/platform/backend subset without regressing earlier recipe batches, clone privacy, or manual audio.cpp sources.
<!-- AC:END -->
