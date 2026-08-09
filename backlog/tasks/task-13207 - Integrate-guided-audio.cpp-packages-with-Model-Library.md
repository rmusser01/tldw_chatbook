---
id: TASK-13207
title: Integrate guided audio.cpp packages with Model Library
status: To Do
assignee: []
created_date: '2026-08-09 17:39'
labels:
  - tts
  - audio-cpp
  - model-library
  - settings
dependencies:
  - TASK-13202
  - TASK-13203
references:
  - backlog/decisions/050-audio-cpp-generated-model-setup-ownership.md
  - backlog/decisions/051-private-tts-clone-reference-assets.md
documentation:
  - Docs/superpowers/specs/2026-08-09-audio-cpp-guided-model-setup-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Connect reviewed audio.cpp model artifacts to the existing Model Library and provide dependency-aware removal without installing the server binary.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Model Library exposes only reviewed audio.cpp artifacts mapped to an exact recipe/package identity and displays family, variant, speech tasks, size/checksum/license/source, required companion files, and exact evidenced compatibility without implying it installs audiocpp_server.
- [ ] #2 Installation occurs only after explicit user action through the existing shared artifact-store owner, verifies the declared artifact, and returns the exact installed package root/identity to the preserved Guided Settings draft without launching audio.cpp or changing global/Studio defaults.
- [ ] #3 Returning from Model Library preserves all unrelated Settings draft fields and shows the installed package in the same exact review/validation flow as a scanned local package before Save.
- [ ] #4 A removal preview accounts for global guided configurations, TTS profiles and clone references, character assignments, active/staged runtime generations, and shared artifact owners before any bytes are removed.
- [ ] #5 Removal requires an explicit resolution for every blocking dependency, never silently retargets a configuration/profile/character, never deletes a reference asset as a side effect, and does not disrupt an immutable live child snapshot.
- [ ] #6 Interrupted install/remove, checksum failure, missing files, shared ownership, and source disappearance produce truthful recoverable state with no partial authority transfer or orphaned registry entry.
- [ ] #7 Hermetic store/UI/dependency tests and a clean-profile UAT cover install → exact-root return → guided Save → sample generation plus blocked and approved removal, with no network or large artifact requirement in normal CI.
<!-- AC:END -->
