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
Status: In Progress. Linux and macOS operations verified; 42 Windows native cases pass, including nested directory publication and metadata. Readonly directory handling and installed validation scratch paths corrected; actual Windows product verification remains required.
Windows implementer owns new platform adapter modules and Windows primitive tests.
Root owns Linux native operations, integration imports and release contracts.
Preserve containment, permissions, exclusive rename and failure propagation.

## Stage 3 — Actual product verification
Status: In Progress. Production3b4a passes every actual installed macOS/Linux product case. Windows3b4a passes three F9 modes, installed two-profile, support53 and native294/294 but replacement/rollback fail in shared seed admission. CPU diagnostic135fd passes native42 and Linux72; Windows seed observer expires with main-thread CPU dominance and sparse completion-port polling stacks. A bounded five-second main-thread profile is next. All current artifact hashes and installed receipts verify. No production or test deadline changes.
Use installed package, private profile and synthetic data on supplied Linux host,
GitHub Actions Windows runner and local macOS. Create and verify backups, restore
and open them, then exercise replacement and retained-copy rollback. Fix failures.

## Stage 4 — Review and PR evidence
Status: In Progress. Native adapter, integration and SQLite snapshot reviews addressed; PR updated with actual incomplete Windows status.
Independent review of changed boundaries and evidence; targeted static/Bandit
checks; update existing PR2642 with exact results and platform support.
