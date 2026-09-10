---
id: TASK-32004
title: Restore and reopen an isolated profile
status: In Progress
assignee: []
created_date: 2026-09-08 00:00
labels:
- backup-recovery
dependencies:
- task-31978
- task-32000
- task-32001
- task-32002
- task-32003
updated_date: 2026-09-10 21:13
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: Isolated recovery creates and reopens a separate profile without altering original local data.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Isolated recovery creates and reopens a separate profile without altering original local data.
- [ ] #2 Damaged current configuration and databases do not prevent archive-only recovery.
- [ ] #3 Fresh launch respects relocated paths, credential/device isolation, durable activation, and projection readiness.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute original component04 Task21 only: (1) implement inert private ProfileCatalog register/resolve with opaque IDs and checked explicit config/data locators, test source-preserving restart and corrupt/linked/changed mapping refusals; (2) compose stage/publication/installed validation/activation/catalog under native maintenance after dependency contracts are ready, with fresh local identities; (3) fresh-process launch verifies catalog/admission/activation and filters inherited selectors; (4) focused isolated archive-only/reopen fixtures, scoped guards/Ruff/Bandit, review and docs. No task completion or launch exposure from the catalog primitive alone.
<!-- SECTION:PLAN:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-04-restore-recovery.md#task-21)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Starting independent ProfileCatalog primitive from original Task21 while publication/activation integration dependencies finish. Existing task found; no duplicate. Catalog is a convenience locator registry, never restoration/activation authority, and cannot authorize a launch on its own. Constructor and lookup create no files; private immutable per-ID records, repeated exact registration may be verified durably; changed mappings refuse. No edits to cli/isolated executor yet; ADR-126.
ProfileCatalog primitive implemented and independently reviewed, no launch/executor exposed. Private per-opaque-ID strict records persist explicit config/data locators only; constructor/resolve never write; exact registration retries revalidate and reflush file/catalog/control/ancestor associations, changed ID mappings refuse. Locators checked without config parsing; linked/damaged/public records refuse, targets remain unchanged. Initial18 behavioral reds ->18green; review identified absentcontroldir and missingparentretrybarrier, both proven2reds thenfixed. Final23passed .72s (/private/tmp/chatbook-catalog-reviewed.log), fulltouchedRuff/formatclean/Bandit0. Review /private/tmp/chatbook-profile-catalog-review.md approved scopedfixes. Exactnewproducerrow ProfileCatalog.register/open1 generic_boundary. This conveniencecatalog is not restoration/activation authority; catalogrebuild and isolatedexecutor/freshlaunch remain required originalTask21work. ADR126.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->