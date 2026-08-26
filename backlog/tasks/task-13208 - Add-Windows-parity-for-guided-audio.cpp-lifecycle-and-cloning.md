---
id: TASK-13208
title: Add Windows parity for guided audio.cpp lifecycle and cloning
status: To Do
assignee: []
created_date: '2026-08-09 17:39'
labels:
  - tts
  - audio-cpp
  - windows
  - lifecycle
  - privacy
dependencies:
  - TASK-13204
  - TASK-13207
references:
  - backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md
  - backlog/decisions/029-local-private-data-boundary.md
  - backlog/decisions/050-audio-cpp-generated-model-setup-ownership.md
  - backlog/decisions/051-private-tts-clone-reference-assets.md
documentation:
  - Docs/superpowers/specs/2026-08-09-audio-cpp-guided-model-setup-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Provide Windows process, path, ACL, scanner, backend-selection, clone-materialization, and definitive-shutdown parity for guided setup.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Guided binary detection, file/folder selection, canonical package identity, auto-port allocation, generated configuration, and bounded scanning work with Windows path, drive, Unicode, long-path, symlink, and reparse-point semantics without applying POSIX assumptions.
- [ ] #2 Chatbook uses Windows-native no-shell process creation, waits on and terminates only the exact handle it owns, closes that handle definitively, and makes no ownership claim for arbitrary descendants or daemonizing server builds.
- [ ] #3 Restart, crash, cancellation, app close, and close-during-start/stop races honor one bounded shutdown budget while retained joining proves no owned process handle, task, client, generated artifact, or endpoint remains.
- [ ] #4 Generated artifacts and clone-reference materializations use an explicitly implemented owner-private Windows ACL posture, surface that actual posture truthfully, and clean recognized exact-owned paths without following reparse points or deleting unknown directories.
- [ ] #5 Backend Auto and explicit CPU/accelerated choices are recipe- and device-aware on Windows, use the same allowlisted definitive-cleanup fallback rule, and label only provisioned tuple evidence as Verified.
- [ ] #6 Settings and Speech Lab preserve the same saved/applied/process truth, keyboard/focus behavior, sample/clone flows, stable errors, and privacy guarantees on Windows as on POSIX.
- [ ] #7 Windows-specific unit/integration tests plus pinned Windows CPU real-process UAT prove generated JSON acceptance, health/catalog, text and clone WAV synthesis, Model Library/local package paths, exact shutdown, no orphaned child/handle, and audible playback in a disposable profile.
<!-- AC:END -->
