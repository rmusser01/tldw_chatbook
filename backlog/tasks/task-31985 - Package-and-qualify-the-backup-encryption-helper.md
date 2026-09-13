---
id: TASK-31985
title: Package and qualify the backup encryption helper
status: Done
assignee: []
created_date: '2026-09-07 23:48'
updated_date: '2026-09-08 02:34'
labels:
  - backup-recovery
dependencies:
  - task-31978
  - task-31984
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: Qualified wheels work without Go, runtime downloads, or PATH helper substitution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Qualified wheels work without Go, runtime downloads, or PATH helper substitution.
- [x] #2 Source/editable installation and unsupported-platform behavior are explicit and tested.
- [x] #3 Native platform evidence, pinned dependencies, integrity/version checks, and upgrade interoperability accompany each advertised tuple.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a packaging-specific behavioral regression and establish RED before production changes.
2. Implement package-resource helper capability validation using the package-owned delivery manifest, digest, executable identity and permissions, protocol version, and native platform tuple; never search PATH, download, or build at runtime.
3. Add the pinned Go helper build command and platform-tagged wheel command so candidate native tuples are explicit and pure-Python wheels cannot contain a native helper.
4. Extend package data, source-distribution inventory, build script, and distribution validation for the helper source, pinned modules, licenses, delivery manifest, and native wheel contents.
5. Add native qualification metadata and CI workflow that record actual OS, filesystem, Python, pipe, trust, integrity, and interoperability evidence while leaving untested tuples unavailable.
6. Prove the native wheel offline in an isolated environment without Go or PATH substitution; separately test source and documented editable workflows plus missing, malformed, upgrade-interoperability, and unsupported cases.
7. Run the focused packaging tests and scoped Python/Go/distribution guards, self-review the diff, and record exact evidence and limitations.

ADR required: yes
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: direct implementation of the approved storage, credential, archive, and recovery contract; reuse ADR-126 with ADR-029/030/036/059/060.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented ADR-126 package-owned helper delivery. Qualified native wheels carry one reproducibly built helper, its digest and Python qualification cells, and complete age/hpke/x-crypto/Go license and patent notices. Pure wheels, source and editable installs remain explicitly unavailable; runtime never searches PATH, downloads, or compiles. The release builder keeps generated trees outside the checkout.

Qualified evidence is limited to darwin/arm64, macOS 26.5.2 (25F84), APFS, Python 3.12.11 and Go 1.26.2. The helper declares minimum macOS12.0 and has a verified Go ad-hoc signature; Developer ID/notarization and other candidate cells remain unqualified. Native runtime ran in a fresh environment without Go, under inherited macOS network denial with a real EPERM network probe, and verified roundtrip, tamper/missing/malformed resources and both-direction source/package interoperability. Separate offline fresh environments actually installed the sdist and editable checkout; the editable installation executed the documented explicit contributor build.

Final targeted commands: python -m pytest Tests/Packaging/test_backup_helper_distribution.py -q (16 passed,21.67s); python -m pytest Tests/Backup_Recovery/test_crypto.py -q (37 passed,11.64s); python -m pytest Tests/Packaging/test_installed_distribution.py::test_built_artifacts_match_distribution_contract -q (1 passed,8.19s). Used Python3.12.11, GOTOOLCHAIN=local and private offline Go/uv caches. Existing RequestsDependencyWarning reproduced on baseline. Scoped Ruff fatal checks and formatting, Go test/vet/module verification, linked-module/license-byte checks, JSON, shell syntax, codesign, digest and git diff --check passed. No full suite or remote workflow ran.

Independent spec/quality review identified runtime Python qualification, complete linked notices and actual installation evidence gaps. Fix commit 3a75df0be addresses all three; scoped independent re-review reports all addressed and no new breakage. Implementation commits:2758ac733 and3a75df0be.

Files: Packaging/backup_age sources/build/qualification/notices, wheel and sdist inventories/build/checker, Backup_Recovery/crypto.py and helper_manifest.json, focused packaging/crypto tests and fixtures, manual qualification workflow. Incident-backed testing and Backlog lessons record generated build-tree contamination and exact stash recovery after external worktree cleanup. No unrelated main-checkout changes were included.
<!-- SECTION:NOTES:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-01-encryption.md#task-2)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)
