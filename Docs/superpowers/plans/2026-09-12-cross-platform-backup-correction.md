# Cross-platform backup correction plan

TASK-32496. Spec: ../specs/2026-09-12-cross-platform-backup-correction.md.
ADR required: existing ADR amendment.
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md.
Reason: correct platform operation contracts without changing recovery behavior.

Use subagent-driven development for the Windows native boundary and independent
review. Root owns Linux implementation, actual product testing, task tracking and
PR publication. Keep file ownership disjoint until reviewed integration.

## Stage 1 — Reproduce and map
Status: Complete.
Record real Linux product failure and native Windows runner failure. Map calls to
private paths, locking, identity, flush and exclusive publication.

## Stage 2 — Platform operations
Status: Complete. Native APFS, ext4 and local NTFS operations are verified, including private ownership, nested directory publication, metadata, locks and persistence barriers. Actual installed backup, restore, replacement and retained-copy rollback have passed on all three platforms.
Windows implementer owns new platform adapter modules and Windows primitive tests.
Root owns Linux native operations, integration imports and release contracts.
Preserve containment, permissions, exclusive rename and failure propagation.

## Stage 3 — Actual product verification
Status: Complete. Revision e733 passes all six actual Windows product-flow groups, including later rollback; macOS and the supplied Linux host also pass. Final revision a5a536565 passes all 117 Windows support tests and 42 native cases, including canonical TTS reference seed, capture, restore and fresh read. All 31 final artifact hashes verify. Final Linux checks pass 19/19; local macOS regressions pass. Exact receipts and pre-existing broader failures are recorded in the verification document.
Use installed package, private profile and synthetic data on supplied Linux host,
GitHub Actions Windows runner and local macOS. Create and verify backups, restore
and open them, then exercise replacement and retained-copy rollback. Fix failures.

## Stage 4 — Review and PR evidence
Status: Complete. Native adapter, integration, SQLite snapshot and final TTS/file-inventory reviews are accepted. Scoped security/static checks introduce no findings. PR2642 records verified product and final support results. Its existing dev conflicts and requester-written merge-summary requirement remain explicit merge blockers.
Independent review of changed boundaries and evidence; targeted static/Bandit
checks; update existing PR2642 with exact results and platform support.
