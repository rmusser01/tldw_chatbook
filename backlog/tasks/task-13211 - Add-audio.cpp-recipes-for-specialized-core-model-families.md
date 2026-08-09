---
id: TASK-13211
title: Add audio.cpp recipes for specialized core model families
status: To Do
assignee: []
created_date: '2026-08-09 17:39'
labels:
  - tts
  - audio-cpp
  - recipes
  - compatibility
dependencies:
  - TASK-13210
references:
  - backlog/decisions/050-audio-cpp-generated-model-setup-ownership.md
  - backlog/decisions/051-private-tts-clone-reference-assets.md
documentation:
  - Docs/superpowers/specs/2026-08-09-audio-cpp-guided-model-setup-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add exact release-0.5.1 recipes and evidence for confucius4_tts, vevo2, index_tts2, irodori_tts, and moss_tts_local.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All 16 declared release-0.5.1 package entries across confucius4_tts (1), vevo2 (3), index_tts2 (4), irodori_tts (6), and moss_tts_local (2) have exact immutable recipe identities and no unreviewed package gap.
- [ ] #2 Clone-only confucius4_tts is discoverable and usable through typed clone capability without a fabricated tts entry, while vevo2 non-TTS tasks and every unrelated upstream task remain excluded from native TTS admission.
- [ ] #3 Each recipe declares exact required/optional files, safe model fields, voice/reference/design/control semantics, backend/platform tuples, compatibility state, lazy-load behavior, and pinned recipe revision without family-name inference.
- [ ] #4 Exact positive fixtures and adversarial missing/conflicting/ambiguous/version/path fixtures prove deterministic recognition, and generated-config/catalog tests cross-check every accepted model/task identity.
- [ ] #5 Every tuple labeled Verified has provisioned real-process text or clone evidence as declared plus definitive cleanup; unsupported or unevidenced specialized combinations remain explicit blockers rather than silent fallbacks.
- [ ] #6 The support/accounting UI and documentation name this exact subset, count moss_tts_local once despite community attribution, and preserve all earlier recipe/manual-source behavior.
<!-- AC:END -->
