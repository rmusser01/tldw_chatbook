---
id: TASK-13208
title: Add Windows parity for guided audio.cpp lifecycle and cloning
status: In Progress
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
  - Docs/superpowers/specs/2026-08-14-audio-cpp-windows-lifecycle-parity-design.md
  - Docs/superpowers/plans/2026-08-14-task-13208-audio-cpp-windows-lifecycle-parity.md
  - Docs/superpowers/qa/audio-cpp-windows-2026-08-14/README.md
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

## Implementation Plan

ADR required: yes

ADR path: `backlog/decisions/029-local-private-data-boundary.md` amendment

Reason: Windows owner-private DACL verification changes the security posture
for generated audio.cpp launch artifacts and clone-reference materializations;
the existing runtime and ownership ADRs otherwise remain authoritative.

Detailed executable plan:
`Docs/superpowers/plans/2026-08-14-task-13208-audio-cpp-windows-lifecycle-parity.md`

1. Add one TTS-scoped stdlib Win32 filesystem capability for absolute paths,
   no-reparse handles, identity, protected DACLs, locks, and exact cleanup.
2. Use that capability in the existing bounded package scanner without
   changing recipe matching or result ownership.
3. Add a Windows storage branch to the existing generated launch artifact.
4. Add a Windows record branch to the existing clone-reference materializer.
5. Admit reviewed `.exe` binaries and Windows x86/x64 backend evidence without
   executing during detection or Save.
6. Retain spawn across cancellation and close the exact subprocess transport
   only after the existing supervisor settles every generation resource.
7. Enable the existing Settings/Speech Lab flow with truthful DACL and
   saved/applied/process copy.
8. Add a hermetic Windows 3.12 CI gate and one parameterized PowerShell UAT.
9. Close the task only after hosted Windows evidence, clean shutdown, and human
   audible confirmation pass; otherwise keep it In Progress.

## Implementation Notes

- Added one stdlib Win32 artifact capability and reused the existing scanner,
  generated-config, clone-materialization, supervisor, managed-store, Settings,
  and Speech Lab ownership boundaries. No dependency, secondary supervisor,
  Job Object, descendant ownership claim, or Windows-only UX was added.
- Windows 10+ x86/x64 on Python 3.12+ is capability-gated. ARM remains
  unsupported pending separately provisioned runtime/backend evidence and UAT.
- Amended ADR-029 only for verified protected-DACL posture on generated
  audio.cpp configs and operation-scoped clone references; all raw native
  failures remain behind bounded path-free outcomes.
- Added a required hermetic `windows-latest` x86/x64 job and a generic
  parameterized PowerShell UAT. The UAT launches a user-provided server in
  place, copies only the clone model package into its disposable managed store,
  requires structural text/clone WAV evidence, and cannot report final pass
  without explicit audible-and-intelligible confirmation.
- Rebased host-side focused verification reached 1,036 passing tests plus two expected
  native-Windows skips. Four sandbox-denied loopback real-child tests passed
  when granted loopback access. One suite-load mount-readiness miss passed
  immediately in isolation. Two unrelated mounted UI baseline failures
  reproduced outside this change; changed-file Ruff, format, scoped mypy, CI
  shape, harness, and diff gates pass.
- Release gate: hosted Windows CI and provisioned objective/audible UAT evidence
  are not yet available for both x86 and x64. TASK-13208 therefore remains In
  Progress and all acceptance criteria remain unchecked.
