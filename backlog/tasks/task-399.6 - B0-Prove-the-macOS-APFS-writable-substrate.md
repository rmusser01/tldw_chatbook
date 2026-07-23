---
id: TASK-399.6
title: B0 Prove the macOS APFS writable substrate
status: To Do
assignee: []
created_date: '2026-07-23 14:23'
labels:
  - notes
  - filesystem
  - macos
dependencies:
  - TASK-399.2
documentation:
  - >-
    Docs/superpowers/specs/2026-07-22-file-backed-notes-authority-recovery-design.md
  - backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md
parent_task_id: TASK-399
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Establish a finite, release-blocking executable and packaged go/no-go contract for safe native file mutation before B1 implementation begins, then requalify the exact final B1 native adapter before any writable action is exposed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A packaged-app probe verifies the actual root volume and reports every required primitive separately.
- [ ] #2 The initial writable matrix contains exactly the current macOS major release and its immediately preceding major release at B0 check-in, names both tested releases/builds explicitly, and infers no writable range from the application's broader minimum macOS version.
- [ ] #3 On each exact allowed release/build, the probe demonstrates pinned no-follow traversal, atomic no-replace/displaced-target exchange, file/directory durability barriers, and required full-fsync behavior on local APFS.
- [ ] #4 Nested mounts, cross-device paths, network/cloud volumes, symlink substitution, hardlinks, unsafe names, and unsupported primitives fail closed with specific reasons.
- [ ] #5 Files carrying ACLs, extended attributes, flags, unusual ownership, or other unround-trippable metadata remain read-only.
- [ ] #6 Packaged Linux and Windows probes, and every macOS release/build outside the exact two-entry pilot, retain the full read-only experience and expose no writable action.
- [ ] #7 backlog/docs/file-backed-notes-apfs-capability-matrix.md records runner hardware, the exact allowed macOS/APFS releases and builds, probe results, the named power-cut/reboot and crash/fsync methods plus observed durability results, native mutation-adapter ABI/artifact SHA-256, probe version/artifact hash, tested application build/commit provenance, and explicit go/no-go.
- [ ] #8 The same result is checked in as the versioned package resource `tldw_chatbook/Notes/file_notes_apfs_capabilities.v1.json`; its canonical schema covers the manifest version, exact allowed macOS major/build entries, APFS capability result, native mutation-adapter ABI/artifact SHA-256, probe version/artifact hash, non-gating application build/commit provenance, explicit go/no-go, and `payload_sha256`, computed over the UTF-8 canonical JSON bytes with that checksum member omitted.
- [ ] #9 Wheel and packaged-app tests prove the manifest is included and readable through the installed-package resource API; schema failure, checksum failure, missing data, mutation-adapter ABI/artifact mismatch, probe-artifact mismatch, unsupported OS/build, or any no-go entry produces read-only eligibility.
- [ ] #10 Runtime eligibility requires both an allowed installed manifest entry and a successful fresh probe of the actual root and primitives. The final B1 release candidate reruns the two-release qualification against its exact packaged adapter/probe before controls can ship; any later adapter change or rolling support-set update requires new probe, power-cut evidence, manifest update, and release review rather than automatic inference.
- [ ] #11 No create, save, rename, move, restore, or delete control is implemented or exposed by this task, and write controls remain hidden outside the approved matrix.
<!-- AC:END -->
