---
id: TASK-13207
title: Integrate guided audio.cpp packages with Model Library
status: Done
assignee: []
created_date: '2026-08-09 17:39'
updated_date: '2026-08-14 19:37'
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
- [x] #1 Model Library exposes only reviewed audio.cpp artifacts mapped to an exact recipe/package identity and displays family, variant, speech tasks, size/checksum/license/source, required companion files, and exact evidenced compatibility without implying it installs audiocpp_server.
- [x] #2 Installation occurs only after explicit user action through the existing shared artifact-store owner, verifies the declared artifact, and returns the exact installed package root/identity to the preserved Guided Settings draft without launching audio.cpp or changing global/Studio defaults.
- [x] #3 Returning from Model Library preserves all unrelated Settings draft fields and shows the installed package in the same exact review/validation flow as a scanned local package before Save.
- [x] #4 A removal preview accounts for global guided configurations, TTS profiles and clone references, character assignments, active/staged runtime generations, and shared artifact owners before any bytes are removed.
- [x] #5 Removal requires an explicit resolution for every blocking dependency, never silently retargets a configuration/profile/character, never deletes a reference asset as a side effect, and does not disrupt an immutable live child snapshot.
- [x] #6 Interrupted install/remove, checksum failure, missing files, shared ownership, and source disappearance produce truthful recoverable state with no partial authority transfer or orphaned registry entry.
- [x] #7 Hermetic store/UI/dependency tests and a clean-profile UAT cover install → exact-root return → guided Save → sample generation plus blocked and approved removal, with no network or large artifact requirement in normal CI.
- [x] #8 The pinned 21-family, 67-package inventory has no open recipe gap and keeps recipe support separate from artifact availability: every variant is reviewed as downloadable, local-only, or explicitly unsupported, with no family-specific installer path or silent accounting gap.
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Implemented the reviewed 21-family/67-package inventory join, curated inactive
  provisioning, exact detached Settings handoff, shared inactive-root leases,
  staged/live runtime ownership, dependency preview, and acknowledged removal
  through the public artifact service.
- ADR check: no new ADR. The implementation follows ADR-050 for artifact/runtime
  ownership and ADR-051 for private clone-reference assets.
- Added hermetic registry, acquisition, Settings/Model Library, runtime,
  dependency, removal, interruption, privacy, accessibility, and concurrency
  coverage. All five required mutation checks failed their named test and passed
  after restoration.
- Final exact unrestricted 17-file matrix: 1,288 passed with only 5 existing
  dependency/deprecation warnings. Changed-file Ruff and format checks, scoped
  mypy, deterministic CSS sync, privacy scan, backlog parse, and diff checks are
  clean.
- Live UAT is recorded in
  `Docs/superpowers/qa/audio-cpp-model-library-2026-08-13/live-uat.md`. The
  isolated path-free Supertonic flow provisioned the exact pinned artifact
  inactive, returned and saved the exact managed identity/root through the real
  Settings flow, reloaded it in a fresh app, exposed only
  `supertonic-3-f16`, and generated a valid 319,904-byte PCM16 mono 44.1 kHz
  WAV. Human playback confirmed intelligible speech.
- The earlier Inflect HTTP 500 remains a prerequisite-specific diagnostic
  result, not a compatibility claim or release blocker; the path-free
  Supertonic flow supplies the required Guided end-to-end evidence.
- No new general lesson was added: the recompose/mount synchronization incident
  is already covered by the repository testing lessons.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the reviewed audio.cpp Model Library integration with exact inactive
provisioning, Settings handoff, runtime/shared lease ownership, dependency-aware
acknowledged removal, and evidence-backed lifecycle UI. Final verification:
1,288 release-matrix tests passed; Ruff, formatting, scoped mypy, CSS sync,
privacy, backlog, and diff gates passed. Clean isolated Supertonic UAT completed
install, exact-root return, real Save, fresh-app reload, catalog, and intelligible
WAV generation.
<!-- SECTION:FINAL_SUMMARY:END -->
