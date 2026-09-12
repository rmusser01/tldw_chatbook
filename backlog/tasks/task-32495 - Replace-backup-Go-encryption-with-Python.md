---
id: TASK-32495
title: Replace backup Go encryption with Python
status: In Progress
created_date: 2026-09-12 15:09
labels:
- backup-recovery
references:
- https://github.com/rmusser01/tldw_chatbook/pull/2642
documentation:
- Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md
- backlog/decisions/126-complete-local-backup-and-recovery.md
updated_date: 2026-09-12 15:29
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User explicitly rejected Go as an oversight in the original backup plan. Replace the backup encryption implementation and remove Go build/package/runtime requirements, preserving the existing backup scope and archive/recovery contract. This user correction supersedes the plan's Go-helper instruction.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Backup encryption and decryption use Python and existing project crypto dependencies, with no Go toolchain or Go executable required for source installs, packaging or runtime.
- [ ] #2 Preserve the existing passphrase-encrypted .tldw-backup.zip.age format, streaming budgets, authentication, cancellation/cleanup and credential/rollback behavior; verify compatibility against independent existing vectors/artifacts.
- [ ] #3 Remove obsolete Go packaging/test requirements and update affected documentation, qualification checks and PR2642 without adding unrelated features; run targeted regression and Linux verification with exact remaining native-platform limitations documented.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stage1 — Read the existing crypto/process/package contracts and select a Python replacement preserving the format. In Progress.
Stage2 — Implement and verify streaming passphrase encryption using established primitives from the existing pycryptodomex dependency; retain a cancellable Python worker if needed by the existing process contract. Not Started.
Stage3 — Remove Go build/distribution assumptions and adapt affected tests and capability checks, retaining fail-closed platform checks. Not Started.
Stage4 — Run targeted interoperability, malformed-input, resource, cancellation, packaging and Linux regression checks; independent review and PR update. Not Started.
No changes to native Linux publication or new crypto formats without resolving them as separate concrete requirements.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
User: 'go should haven ever been used'. Plan authoring choice is superseded; do not insist on Go due to original plan. Python cryptography candidates are under read-only assessment. Project already depends on pycryptodomex. Existing pure-Python age packages and Rust-backed pyrage are not assumed suitable; stream limits and byte-password compatibility must be verified. Existing crypto public transform signature and .age format should remain stable. Linux baseline is tracked separately byTASK32494 and shows unrelated Darwin-native publication/barrier gaps.
Design/plan recorded in Docs/superpowers/specs/2026-09-12-python-backup-encryption-design.md and matching plans path, implementing the user's explicit language correction while preserving .age files and existing API. Existing pycryptodomex provides primitives; no additional crypto runtime dependency or Rust replacement. Bounded isolated Python worker retains cancellable KDF and pipe secret handling. Task1 worker implementation/review dispatched under SDD; root owns independent legacy test-vector provenance, packaging plan and records. Existing Linux test ciphertext is synthetic and predates the correction; preserve it for compatibility, with no additional Go execution.
Fixed independent interoperability fixtures committed749f116b2: official age1.3.2 ciphertext (SHAfb98c008...), previous worker empty/raw4096-byte password ciphertext (SHA3f67cd98...), prior worker streaming ciphertext (SHA747e0916...). Root verified remote/local hashes; README records fully synthetic inputs and original37pass test receipt. No newGo execution. Worker/test-only implementation active; independent read-only packaging map delegated while root prepares unchanged transport/capability integration. Design/plan commitc19068652.
Task1 worker frozen for independent review: 60 tests passed (15.95s), including three fixed pre-existing age archives and exact ciphertext reproduction with recorded entropy. Ruff/format/compile and production Bandit passed; test-only Bandit findings retained in report. Worker protocol2/python/age-v1. Parent transport and Go packaging removal remain pending. Evidence /private/tmp/task32495-python-worker-report.md; reviewer dispatched.
Task1 review found closed stdout could emit interpreter-finalization diagnostics. Narrow fix uses unbuffered FileIO at protocol boundary. Real info/encrypt/decrypt closed-pipe regressions RED3 then GREEN3; final worker module63passed16.39s. Production Bandit0, Ruff/format/compile pass;26 LOW test-only Bandit findings disclosed. Fixed worker SHA451a3dc158c3c1e8ee599f384a23046eb9d446ca94e8c5e04b8ffe9c6e068c05. Scoped re-review pending; no Go executed. Packaging dependency map completed /private/tmp/task32495-python-packaging-map.md.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
