# Complete local backup and restore Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the approved complete local recovery workflow through independently testable implementation slices.

**Architecture:** Implement the approved Backup_Recovery service through existing storage owners, with fixed startup admission and private immutable staging. Keep archive parsing separate from normal application startup, and expose only qualified operations through thin user views.

**Tech Stack:** Python ≥3.11, Textual 8.x, SQLite/FTS5, Pydantic, stdlib ZIP64/file primitives, and a bundled helper built from official age Go library.

**Spec:** [Approved complete local backup and restore](../specs/2026-09-07-complete-local-backup-restore-design.md), revision 4.

## Global Constraints

- Python ≥3.11; retain the repository's current Textual 8.x dependency contract.
- “All identified Chatbook-owned local profiles and durable data are included by default, including configured database locations outside the default directory.”
- “External folders and model files are explicit options. Server-owned data has a separate recovery boundary and is not captured by this feature.”
- “Managed credentials are excluded by default. Including exportable credentials requires a password-encrypted archive.”
- “Partial archives support extraction and isolated recovery of validated dependency groups; they cannot replace current installation data in v1.”
- “Use .tldw-backup.zip for plaintext and .tldw-backup.zip.age for encrypted output.”
- “v1 reader defaults: 100,000 members; 16 MiB manifest; 1 TiB expanded payload; 256 GiB per member; 1,024 UTF-8 bytes per path.”
- “Default encrypted/plain input-container and decrypted-container budgets are each 2 TiB, independently enforced as actual bytes stream”; outer header 64 KiB; one KDF with 256 MiB working-memory budget.
- “No rebuild starts automatically, including a local rebuild.” Restored execution and reconnection require durable local owner review.
- “Run targeted checks only; a full suite needs separate user authorization.” Runtime tests listed here are required evidence, not results already obtained.
- “Do not expose Complete backup or destructive replacement before its full contract is qualified.” Keep per-operation/platform capability failures visible.
- ADR required: yes. ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md. Reason: direct implementation of the approved storage, credential, archive, and recovery contract; reuse ADR-126 with ADR-029/030/036/059/060.

---

## Approved baseline and delivery strategy

The user approved specification revision 4 on 2026-09-07. ADR-126 is Accepted.
This is a planning deliverable: no production code, helper binary, new schema, or
runtime capability has been implemented or qualified by creating these documents.

The feature spans independent package-delivery, persistence, archive, recovery,
and presentation concerns. Use the six component plans below, with one scoped PR
per Backlog task. Complete helper delivery and maintenance/bootstrap qualification
before dependent archive/replacement execution. A successful primitive test does
not authorize exposing the whole feature. If a cohort proves too large for a
reviewable PR, split that existing task before implementation and update dependency
links; never silently defer owners while retaining a Complete label.

## Component plans

| Plan | Deliverable | Tasks |
| --- | --- | --- |
| [Encryption helper delivery](2026-09-07-backup-recovery-01-encryption.md) | An independently qualified, packaged age helper with bounded secret transport. | 1–2 |
| [Storage inventory and maintenance admission](2026-09-07-backup-recovery-02-inventory-admission.md) | A complete declared owner census and enforced startup/capture admission boundary. | 3–10 |
| [Coherent capture and qualified archives](2026-09-07-backup-recovery-03-capture-archives.md) | Verified local recovery archives and bounded inspection with explicit coverage and credentials. | 11–16 |
| [Restore publication and recovery](2026-09-07-backup-recovery-04-restore-recovery.md) | Both restore destinations, persistent activation safety, exact rollback, and interruption recovery. | 17–23 |
| [User-facing backup and recovery](2026-09-07-backup-recovery-05-user-workflows.md) | Usable F9, first-run, and damaged-installation recovery flows over the same services. | 24–25 |
| [Recovery release qualification](2026-09-07-backup-recovery-06-release-evidence.md) | Demonstrated complete/replace capabilities with native and product evidence. | 26 |

## Backlog and execution order

| Step | Backlog task | Dependencies |
| --- | --- | --- |
| 1 | [TASK-31984: Qualify bounded age helper protocol](../../../backlog/tasks/task-31984%20-%20Qualify-bounded-age-helper-protocol.md) | Approved design |
| 2 | [TASK-31985: Package and qualify the backup encryption helper](../../../backlog/tasks/task-31985%20-%20Package-and-qualify-the-backup-encryption-helper.md) | 1 |
| 3 | [TASK-31986: Declare recovery inventory and side-effect-free profile discovery](../../../backlog/tasks/task-31986%20-%20Declare-recovery-inventory-and-side-effect-free-profile-discovery.md) | Approved design |
| 4 | [TASK-31987: Implement stable maintenance admission and native storage qualification](../../../backlog/tasks/task-31987%20-%20Implement-stable-maintenance-admission-and-native-storage-qualification.md) | 3 |
| 5 | [TASK-31988: Fence all supported startup routes before storage bootstrap](../../../backlog/tasks/task-31988%20-%20Fence-all-supported-startup-routes-before-storage-bootstrap.md) | 3, 4 |
| 6 | [TASK-31989: Add recovery adapters for core conversation and library stores](../../../backlog/tasks/task-31989%20-%20Add-recovery-adapters-for-core-conversation-and-library-stores.md) | 3, 4 |
| 7 | [TASK-31990: Add recovery adapters for local research writing study and evaluation data](../../../backlog/tasks/task-31990%20-%20Add-recovery-adapters-for-local-research-writing-study-and-evaluation-data.md) | 3, 4 |
| 8 | [TASK-31991: Add recovery adapters for workspace operational and device-local state](../../../backlog/tasks/task-31991%20-%20Add-recovery-adapters-for-workspace-operational-and-device-local-state.md) | 3, 4 |
| 9 | [TASK-31992: Add recovery inventory for configuration durable files and optional content](../../../backlog/tasks/task-31992%20-%20Add-recovery-inventory-for-configuration-durable-files-and-optional-content.md) | 3, 4 |
| 10 | [TASK-31993: Integrate maintenance participants across persistence owners](../../../backlog/tasks/task-31993%20-%20Integrate-maintenance-participants-across-persistence-owners.md) | 5, 6, 7, 8, 9 |
| 11 | [TASK-31994: Persist recovered media references and deletion tombstones](../../../backlog/tasks/task-31994%20-%20Persist-recovered-media-references-and-deletion-tombstones.md) | 3, 4, 9, 10 |
| 12 | [TASK-31995: Implement bounded immutable archive inspection](../../../backlog/tasks/task-31995%20-%20Implement-bounded-immutable-archive-inspection.md) | 1, 2, 3, 4 |
| 13 | [TASK-31996: Validate and migrate imported SQLite under restricted owner policies](../../../backlog/tasks/task-31996%20-%20Validate-and-migrate-imported-SQLite-under-restricted-owner-policies.md) | 6, 7, 8, 11, 12 |
| 14 | [TASK-31997: Implement staged credential exclusion and encrypted credential recovery](../../../backlog/tasks/task-31997%20-%20Implement-staged-credential-exclusion-and-encrypted-credential-recovery.md) | 6, 7, 8, 9, 13 |
| 15 | [TASK-31998: Capture a coherent final inventory with optional external content](../../../backlog/tasks/task-31998%20-%20Capture-a-coherent-final-inventory-with-optional-external-content.md) | 10, 11, 12, 13, 14 |
| 16 | [TASK-31999: Write verify and publish recovery archives without overwrite](../../../backlog/tasks/task-31999%20-%20Write-verify-and-publish-recovery-archives-without-overwrite.md) | 2, 12, 15 |
| 17 | [TASK-32000: Plan and stage both restore destinations with explicit dependency mapping](../../../backlog/tasks/task-32000%20-%20Plan-and-stage-both-restore-destinations-with-explicit-dependency-mapping.md) | 12, 13, 14, 16 |
| 18 | [TASK-32001: Implement durable publication journal and crash reconciliation](../../../backlog/tasks/task-32001%20-%20Implement-durable-publication-journal-and-crash-reconciliation.md) | 4, 5, 17 |
| 19 | [TASK-32002: Gate restored projections and invalidate stale retrieval state](../../../backlog/tasks/task-32002%20-%20Gate-restored-projections-and-invalidate-stale-retrieval-state.md) | 6, 7, 8, 17, 18 |
| 20 | [TASK-32003: Persist per-generation recovery activation requirements](../../../backlog/tasks/task-32003%20-%20Persist-per-generation-recovery-activation-requirements.md) | 5, 8, 9, 18 |
| 21 | [TASK-32004: Restore and reopen an isolated profile](../../../backlog/tasks/task-32004%20-%20Restore-and-reopen-an-isolated-profile.md) | 17, 18, 19, 20 |
| 22 | [TASK-32005: Replace selected local data with verified encrypted rollback](../../../backlog/tasks/task-32005%20-%20Replace-selected-local-data-with-verified-encrypted-rollback.md) | 10, 14, 16, 17, 18, 19, 20, 21 |
| 23 | [TASK-32006: Expose retained recovery copies and safe later rollback](../../../backlog/tasks/task-32006%20-%20Expose-retained-recovery-copies-and-safe-later-rollback.md) | 18, 21, 22 |
| 24 | [TASK-32007: Expose backup and restore in canonical F9 Settings](../../../backlog/tasks/task-32007%20-%20Expose-backup-and-restore-in-canonical-F9-Settings.md) | 16, 21, 22, 23 |
| 25 | [TASK-32008: Expose startup-independent recovery and first-run restore](../../../backlog/tasks/task-32008%20-%20Expose-startup-independent-recovery-and-first-run-restore.md) | 5, 12, 21, 22, 23, 24 |
| 26 | [TASK-32009: Qualify complete backup and replacement release capabilities](../../../backlog/tasks/task-32009%20-%20Qualify-complete-backup-and-replacement-release-capabilities.md) | 2, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25 |

Dependencies in this table are step numbers; Backlog frontmatter uses actual task IDs.
All dependencies point to already-created earlier tasks. No new user-owned Codex
tasks or subagents were created by this planning work.

## Cross-plan contracts

- Task 1 owns `crypto.transform` and `helper_capability`; task 2 supplies verified
  packaged resources. This layer imports no app/config/database services.
- Task 3 owns `models.StorageItem`, `Inventory`, `SchemaPolicy`, `OwnerAdapter`,
  `inventory.discover`, `classify_entries`, and `owner_registry.register/registered`.
  Cohorts in tasks 6–9 and 11/19 register concrete implementations. Census rows are
  the exhaustive checklist, not just the examples in the plan file lists.
- Task 4 owns `admission.Admission`, `native_files.publish_new`, and
  `qualification.qualified_for`; task 5 owns fixed `control_records.register_pending`
  and `bootstrap.startup_permission`; task 10 binds actual lifecycle participants.
- Task 12 owns `limits.ArchiveLimits`, `archive_models.SealedArchive`, and
  `archive_reader.acquire`. Task 13 owns `sqlite_validation.validate_candidate`,
  using registered `private_sqlite.open_recovery_validation` and installed policies.
- Task 14 owns staged credential policies; callers explicitly select exclude/include/
  rollback and encrypted state. Its scope changes are journal evidence, not imported
  authority. Task 15 owns `capture.CaptureResult`, `capture`, and `compare_scope`;
  task 16 writes/verifies/publishes through that same reader.
- Task 17 owns `restore_plan.RestorePlan`, `plan_restore`, and `staging.stage_restore`;
  task 18 owns `journal.Journal` and `publication.publish_candidate`. The journal
  state machine is reused by isolated restore, replacement, and later rollback.
- Task 19 owns projection quarantine/readiness; task 20 owns `ActivationStore`.
  They are independent gates: owner activation never makes an invalid index valid.
- Tasks 21/22/23 own `ProfileCatalog`, isolated restore/fresh launch, replacement,
  retained copies/later rollback, and one `RecoveryService`. UI tasks consume these
  app-owned services. Service status snapshots contain operation ID, phase, counts,
  byte totals, sanitized issue codes, actual output ID/path, and the separate
  archive-verified/restoration-validated/opened/needs-setup states; never credentials.
- Task 26 derives operation capability from demonstrated platform/owner/protocol
  support, with separately qualified archive publication versus installation
  replacement. It cannot convert missing evidence into a successful status.

All proposed new paths are listed under their owner task; modifications name existing
files verified during planning. Adapter extension points stay beside owning domains.
Do not introduce a general service registry, generic migration engine, alternate
settings surface, or a second independent backup format.

## Spec-to-task coverage

| Approved contract | Implementation steps |
| --- | --- |
| Full selected-profile baseline, custom paths, explicit omissions, server boundary | 3, 6–10, 15, 26 |
| Stable admission, aliases, old/new roots, legacy-writer limit and drain ordering | 4, 5, 10, 15, 22, 26 |
| Fixed bootstrap discovery, custom control roots, damaged-installation recovery | 5, 17, 18, 20, 21, 25 |
| Secret transport, helper packaging, format/resource limits, encrypted rollback | 1, 2, 12, 14, 16, 22 |
| Immutable archive input, malicious ZIP/SQLite, restricted migrations | 12, 13, 17, 26 |
| Staged sanitization, SQLite remnants, readable credential values/scope collisions | 6–9, 13, 14, 22 |
| Directory topology, empty roots, metadata omissions, no-overwrite output | 4, 9, 12, 16, 17, 18 |
| Final inventory under maintenance and scope/budget re-preview | 3, 4, 10, 15, 26 |
| Recovered-media stable references, explicit deletion tombstones, later backups | 9, 11, 15–17, 21, 26 |
| Restore/retire/preserve, shared dependencies, exact rollback and later changes | 17, 18, 22, 23 |
| Projection invalidation, provenance/reconciliation, no automatic rebuild | 19, 20–22, 26 |
| Persistent activation, owner review, quarantined sync/queues/permissions | 8, 18, 20–23, 25, 26 |
| New isolated namespace/config, fresh process, credential identity, reopening | 14, 17, 18, 20, 21, 25 |
| Cancellation, per-volume space, failure states, retained evidence | 4, 12, 15–18, 22–26 |
| F9 Settings, command palette, first run, minimal launcher, honest status | 23–26 |
| Native qualification, real DB/process crash tests, installed wheels, live TUI | 1, 2, 4–26 as scoped in each task |

## Review and validation of this plan

Check links, existing modification targets, new-file ownership, task dependencies,
interface consistency, resource constants, and each accepted review correction.
Code fences are test/interface/invariant instructions; this planning change does not
create the proposed Python or Go files. No runtime tests have been run for it.
Use the task-specific gates during execution and record unsupported native tuples
honestly. A full test sweep requires a separate user request.

## Execution handoff

Use the selected execution mode with review after each task. The first executable
slice is task 1, followed by helper packaging and the inventory/admission work.
Subagent-driven execution uses a fresh task worker and review gates; inline execution
uses executing-plans with checkpoints. No execution starts as part of this approval
record or planning artifact.

## Allocation record

At filing, 428 locally available branch/remote refs and
72 worktrees had maximum task ID 31983.
The CLI's first allocation was TASK-31979; it was renumbered before any
links were published. Implementation tasks are TASK-31984 through
TASK-32009. Recheck remote/worktree allocations before merge; no
claim is made about unseen remote changes.
