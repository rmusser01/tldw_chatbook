---
id: TASK-13207
title: Integrate guided audio.cpp packages with Model Library
status: In Progress
assignee: []
created_date: '2026-08-09 17:39'
updated_date: '2026-08-13 15:58'
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
  - >-
    Docs/superpowers/specs/2026-08-13-audio-cpp-model-library-integration-design.md
  - Docs/superpowers/plans/2026-08-13-audio-cpp-model-library-integration.md
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
- [ ] #8 The pinned 21-family, 67-package inventory has no open recipe gap and keeps recipe support separate from artifact availability: every variant is reviewed as downloadable, local-only, or explicitly unsupported, with no family-specific installer path or silent accounting gap.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR
ADR path: backlog/decisions/050-audio-cpp-generated-model-setup-ownership.md; backlog/decisions/051-private-tts-clone-reference-assets.md
Reason: the approved ADR-050 amendment and ADR-051 already govern artifact/runtime and private clone-reference ownership.

1. Add the bounded static-manifest schema/refresh foundation, then complete the 21-family/67-package recipe accounting before populating and joining audited artifacts.
2. Persist optional managed artifact identity and implement typed Settings↔Model Library handoff with activate=False provisioning.
3. Preserve one complete versioned Speech/TTS panel snapshot, including Realtime drafts, and merge only an exact non-stale result.
4. Add service-owned inactive-root leasing and retain shared artifact authority across Save, staged generation, live child, and cleanup retry.
5. Add one ordered removal authority plus non-mutating contention probe.
6. Serialize profile/reference/assignment/bundle/settings mutations with exact shared root leases, then preview and revalidate all consumers before removal.
7. Harden mounted Model Library/Settings UI truth and accessibility.
8. Run hermetic, static, privacy, concurrency, UAT, review, documentation, and task-hygiene gates.

Executable TDD plan: Docs/superpowers/plans/2026-08-13-audio-cpp-model-library-integration.md
<!-- SECTION:PLAN:END -->
