---
id: TASK-31984
title: Qualify bounded age helper protocol
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-07 23:47'
updated_date: '2026-09-08 00:41'
labels:
  - backup-recovery
dependencies:
  - task-31978
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: Streaming encrypted round trips interoperate with official age and reject incomplete authentication.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Streaming encrypted round trips interoperate with official age and reject incomplete authentication.
- [ ] #2 Header, KDF, memory, password transport, cancellation, and child cleanup limits are demonstrated with synthetic data.
- [ ] #3 Unqualified or absent helpers report unavailable before password collection or maintenance; no plaintext fallback occurs.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Establish importable behavioral RED for encrypted round trips and pin verified age v1.3.2.
2. Build a private real-helper fixture and protocol v1 with bounded password/header admission, scrypt work factor 18, fixed errors, and streaming authenticated EOF.
3. Implement package-owned capability verification and serialized, cancellable concurrent pipe transport with private unpublished output.
4. Qualify adversarial envelopes, cancellation/crash/backpressure, secret transport, memory bounds, and official age interoperability.
5. Run focused Python/Go tests, scoped lint/format and diff checks; document source qualification and actual evidence; commit only task-owned changes. Leave ACs unchecked and status In Progress pending independent review.
ADR required: yes
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: direct implementation of the approved encryption contract; reuse ADR-126.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the source-qualified age v1.3.2 helper and package-owned Python transport under ADR-126 (backlog/decisions/126-complete-local-backup-and-recovery.md). Protocol v1 uses pipe-only passwords, bounded single-scrypt header admission, work factor 18, independent streamed 2 TiB limits, authenticated EOF, fixed errors, serialized jobs, cancellation/reaping, private no-replacement publication, and strict digest/platform/info capability checks.

Qualification: Go 1.26.2 with GOTOOLCHAIN=local and Python 3.12.11 on macOS arm64. Tests/Backup_Recovery/test_crypto.py: 36 passed, 1 existing RequestsDependencyWarning, 11.72s. Tests/Utils/test_config_encryption.py plus test_sensitive_config_keys.py: 53 passed, same existing warning, 6.75s. go test -count=1 ./... passed (0.539s); go vet, gofmt -l, Ruff fatal checks, Ruff format --check, git diff --check, and go mod verify passed. Four Go mutations failed as intended. Both directions interoperate with a separately built official age CLI; real 128 MiB streaming/RSS, cancellation/crash cleanup, hostile metadata/envelopes, FIFO-resource rejection, and bounded stderr/info cases pass.

Files: Packaging/backup_age source, module locks and qualification README; Backup_Recovery crypto package; focused Tests/Backup_Recovery fixture/tests; strict-integer-version testing lesson. Production remains unavailable without packaged resources. Task 2 owns native delivery and other platform/Python qualification. No user data, config, credentials, full test sweep, or remote publication was used. All criteria remain unchecked and status stays In Progress pending independent review.
<!-- SECTION:NOTES:END -->


## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-01-encryption.md#task-1)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.
