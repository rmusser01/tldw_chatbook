---
id: TASK-31988
title: Fence all supported startup routes before storage bootstrap
status: Done
assignee:
  - codex
created_date: '2026-09-07 23:50'
updated_date: '2026-09-08 05:35'
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
- [x] #1 Every supported launch path checks fixed bootstrap admission before affected storage or runtime composition.
- [x] #2 Damaged config and inaccessible custom recovery roots cannot bypass a pending operation.
- [x] #3 Provably disjoint profiles remain usable; ambiguous scope is blocked without deleting evidence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
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
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented ADR-126 fixed recovery startup checks and observable process, connection and file admission. Private version-1 pending/profile records use one fixed authority, preserve interrupted or unknown evidence, and allow unrelated profiles only with verified disjoint mappings. Startup enrollment precedes config/runtime imports; existing unbound activity requires close/restart before enrollment. Config changes cannot silently widen a live scope, and unavailable backup primitives do not block positively verified unrelated ordinary startup.

All identified supported console/module/direct/web/MCP/indexing/TTS routes are guarded, including multiprocessing spawn and actual Chatterbox pathname invocation. The exact 22-route AST census is maintained. SQLite and shared private-file boundaries participate in counted lifetime leases; directional containment refuses undeclared ancestor changes, and custom SQLite close must complete native close before releasing admission. Explicit close failures remain fenced and are not retried by GC. Foreign read-only exemptions remain explicitly classified.

Independent review found three Important defects in containment, custom close and disjoint native-unavailable behavior. Commit cc7883119 fixes all three; actual source/editable Chatterbox probes also identified and fixed a package-import regression. Scoped re-review reports all findings addressed and no new Critical/Important breakage. Existing warning noise and redundant locator-loss test variants remain recorded for final branch review.

Final covering evidence: python -m pytest Tests/Backup_Recovery/test_bootstrap.py Tests/Architecture/test_recovery_entrypoints.py -q (65 passed, 17.38s); targeted Tests/DB/test_private_sqlite.py (255 passed, one existing Windows posture skip, 21.39s). Actual pathname source/editable x pending/allowed cases passed 4 without PYTHONPATH. Earlier named DB inventory/interop run passed 299 with the same skip; owner census passed 11; production composition passed 4 including actual app/scheduler cases. Fix producer comparison was unchanged in all three production files. Scoped Ruff, focused format and git diff --check passed. All runs used private fixtures and the shared Python 3.12.11 interpreter read-only; no full sweep or other-platform qualification is claimed.

Files: Backup_Recovery/{bootstrap,control_records,storage_admission}.py; DB/private_sqlite.py; Utils/private_paths.py; supported route guards; startup/entrypoint tests and DB target-selection guard; exact owner inventory; startup guidance and atime lesson. Commits 503641906 and cc7883119. Owner/user contract: backlog/docs/backup-recovery-startup.md. ADR required: yes; reused backlog/decisions/126-complete-local-backup-and-recovery.md, no new ADR.

Capture capabilities, binding refresh and safe startup/owner draining, journal reconciliation, activation and product workflows remain subsequent implementation slices; Complete backup/replacement is not exposed here. Wheel/model execution and unsupported native platforms were not qualified by these tests. No user data, shared environment mutation, push or merge.
<!-- SECTION:NOTES:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-02-inventory-admission.md#task-5)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.
