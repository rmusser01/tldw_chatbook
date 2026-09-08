---
id: TASK-31988
title: Fence all supported startup routes before storage bootstrap
status: In Progress
assignee:
  - codex
created_date: '2026-09-07 23:50'
labels:
  - backup-recovery
dependencies:
  - task-31978
  - task-31986
  - task-31987
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: Every supported launch path checks fixed bootstrap admission before affected storage or runtime composition.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every supported launch path checks fixed bootstrap admission before affected storage or runtime composition.
- [ ] #2 Damaged config and inaccessible custom recovery roots cannot bypass a pending operation.
- [ ] #3 Provably disjoint profiles remain usable; ambiguous scope is blocked without deleting evidence.
<!-- AC:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-02-inventory-admission.md#task-5)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.

## Implementation Plan

ADR required: yes
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: direct implementation of approved startup and private-storage admission boundaries.

1. Add the specified custom-root pending-operation regression and establish behavioral RED.
2. Persist bounded private versioned fixed bootstrap associations, with verified local selectors, shared admission authority, and replacement/control separation; interrupted writes retain evidence.
3. Fence all supported console, module, direct-app, web, MCP, and headless persistence routes before config/bootstrap side effects, including spawned imports.
4. Integrate registered private SQLite and owned-file admission with lifetime retirement and classified exemptions. Preserve no-fence startup compatibility on unqualified native hosts without qualifying backup.
5. Verify damaged/replaced config, inaccessible custom roots, corrupt/unknown records, catalog loss, interrupted registration, alias authority, disjoint scopes, and import-time sentinels.
6. Run focused bootstrap/entrypoint, DB inventory/interop, owner inventory, and production composition guards; scoped lint, format, and diff checks. Self-review and update owner documentation and implementation evidence.
7. Commit scoped changes with the approved subject; retain In Progress and unchecked ACs for controller independent review.

## Implementation Notes

Implemented fixed recovery startup admission and process/connection/file lifetime
participation under [ADR-126](../decisions/126-complete-local-backup-and-recovery.md).
Private version-1 pending/profile records live outside replaceable storage, select
one fixed admission authority, preserve interrupted/corrupt evidence, and admit
other profiles only with intact disjoint mappings. Initial enrollment refuses live
unbound clients with close/restart guidance. Ordinary config edits do not widen live
scope; no-conflict native-unqualified startup remains available without qualifying
backup. No recovery fences are cleared by this task.

The SQLite seam keeps both existing raw connection sites and target/private-parent
validation. A dedicated lease thread counts actual connections and file writers;
worker-thread close and successful abandoned-connection retirement release holds,
while close failures remain conservative. Foreign browser-cookie clone reads and
memory/verified immutable descriptor sources retain explicit exemptions; read-only
owned stores participate. Shared private file boundaries participate as well.

Route census found early package and direct-worker imports beyond the initial file
list. Controller approved guards for config, Web_Server package, MCP.server, RAG_Search
package, TTS package, and the Chatterbox direct worker. Actual subprocess tests cover
normal/module/direct/spawn launches, missing or broken config, and no default-state
creation. All 22 package `__main__` routes have explicit AST classifications.

Final targeted evidence (shared original Python 3.12 interpreter, isolated clone and
fixtures; no full sweep):

- `python -m pytest Tests/Backup_Recovery/test_bootstrap.py Tests/Architecture/test_recovery_entrypoints.py -q`: 57 passed, 4 existing request/source-parse warnings (11.53s); all required startup/process scenarios executed without skips.
- `python -m pytest Tests/DB/test_private_sqlite.py Tests/DB/test_private_sqlite_inventory.py Tests/DB/test_private_sqlite_interop_owners.py -q`: 299 passed, 1 existing Windows functional-posture skip, 1 existing RequestsDependencyWarning (50.86s).
- `python -m pytest Tests/Architecture/test_backup_owner_inventory.py -q`: 11 passed, existing request/source-parse warnings (12.70s).
- `python -m pytest Tests/ProductionApp/test_service_composition_lifecycle.py -q`: 4 passed, 1 existing request warning (11.22s); both actual app composition/scheduler scenarios passed.
- Scoped `ruff check --select E9,F63,F7,F82`, new-module `ruff format --check`, and `git diff --check`: clean; exact commands in the report.

Self-review preserved the old SQLite census function through a decorator. The path
selection sentinel now forbids resolving the actual DB target while permitting
independent physical verification of the fixed authority marker; trusted-parent
checks are unchanged. Existing close-failure tests caught a destructor retry and
now pass with no retry of explicit custom-close attempts. A real config-read atime
trap was documented in `lessons-testing-evidence.md`. Owner census and
[producer/user startup guidance](../docs/backup-recovery-startup.md) were updated.

Remaining boundaries are intentional later-task work: maintenance-capability capture
seams, binding refresh, bootstrap/owner draining, journal-authorized reconciliation,
activation gates, and full raw-owner qualification. Complete backup/replacement is
not exposed. Status remains In Progress and ACs remain unchecked for independent
controller review, as explicitly directed.

Independent review fix round 1 closes three important gaps: owner authorization now
uses directional physical containment, successful custom SQLite closes are followed
by native close before lease retirement, and a verified disjoint profile retains
ordinary native-unqualified access while affected/uncertain scopes still refuse.
Behavioral regressions reproduced all three failures before their fixes. An actual
Chatterbox pathname probe without PYTHONPATH also exposed a source-launch import
failure; the worker now selects its adjacent package before the startup guard.
Private source and editable-registration probes verify both pending refusal and
allowed runtime arrival without default-state writes or model effects.

Fix verification: startup/entrypoint files 65 passed; targeted private SQLite file
255 passed, one existing Windows posture skip. Scoped lint, focused format, and diff
checks pass. Comparing the exact producer census against the review base found no
changes in the three modified production files. Existing route/AST checks passed.
No native or wheel-install qualification is claimed. Review minors M1/M2 remain
deferred by the controller; existing downstream boundaries and unchecked/In Progress
status remain unchanged. ADR-126 still directly governs these fixes.
