---
id: TASK-31985
title: Package and qualify the backup encryption helper
status: In Progress
assignee: []
created_date: '2026-09-07 23:48'
updated_date: '2026-09-08 00:54'
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
- [ ] #1 Qualified wheels work without Go, runtime downloads, or PATH helper substitution.
- [ ] #2 Source/editable installation and unsupported-platform behavior are explicit and tested.
- [ ] #3 Native platform evidence, pinned dependencies, integrity/version checks, and upgrade interoperability accompany each advertised tuple.
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

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-01-encryption.md#task-2)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

## Implementation Notes

Implemented ADR-126's package-owned helper delivery gate. Source, editable, sdist,
and `py3-none-any` artifacts retain an all-unavailable delivery manifest. An explicit
qualified native-wheel build compiles only the selected pinned Go target offline,
checks its reproducible digest, tags the wheel for that platform, and carries the
binary, age license, notices, and one authoritative manifest entry. Runtime resolution
continues to use the fixed package path and now distinguishes an installed digest
mismatch while mapping other capability-probe failures to `helper_unavailable`; it
never searches PATH, downloads, or builds.

Actual native qualification is limited to darwin/arm64, macOS 26.5.2 (25F84) on APFS,
Python 3.12.11, Go 1.26.2, with a macOS 12.0 minimum load command. Python 3.11/3.13
and every other candidate tuple remain explicitly unavailable. The installed-wheel
probe used a fresh environment with Go absent and an inherited macOS process sandbox
network denial; its real network probe returned `EPERM` before round-trip, integrity,
missing/malformed-resource, and both-direction interoperability checks.

Focused evidence: `Tests/Packaging/test_backup_helper_distribution.py` 13 passed;
`Tests/Backup_Recovery/test_crypto.py` 37 passed; the existing distribution-contract
guard passed; Go test/vet/module verification, Python syntax/format checks, shell
syntax, JSON validation, `codesign --verify`, helper digest, and `git diff --check`
passed. Test runs emitted one pre-existing RequestsDependencyWarning from the shared
development environment. No remote workflow or untested matrix cell was executed.

Packaging-generated `build/lib` first caused a whole-root architecture guard to see a
duplicate runtime-policy owner, so build/test copies now stay outside the checkout.
An external broad cleanup later stashed this active task and removed its worktree;
work stopped, recovered exact stash `3de040179723f7ae95dbbf2d63cf73bfff753771`, and
resumed in a verified self-contained clone. Both incidents are recorded in the testing
and backlog-hygiene lessons.

Modified delivery/runtime files: `pyproject.toml`, `MANIFEST.in`,
`Packaging/build_dist.sh`, `Packaging/check_manifest.py`,
`Packaging/backup_age/*`, `tldw_chatbook/Backup_Recovery/crypto.py`, and
`tldw_chatbook/Backup_Recovery/helper_manifest.json`. Focused packaging and real-helper
fixtures plus the manual native qualification workflow carry the release evidence.

Review fixes carry the qualified Python minor versions into the installed manifest and
reject unqualified interpreters before helper invocation; the py3 wheel therefore
keeps the application installable on Python >=3.11 while encryption stays unavailable
outside its tested cell. Native wheels now include byte-identical upstream hpke,
x/crypto, and Go license texts plus Go's PATENTS grant; `go version -m` confirmed those
two modules and age are the only linked non-standard dependencies. Separate fresh
offline environments installed the sdist and editable checkout. The sdist stayed
unavailable without `_age`; the editable environment executed the documented explicit
contributor build into a private path and remained unavailable afterward. Fix-round
evidence: packaging 16 passed, crypto 37 passed, and the distribution guard passed.
