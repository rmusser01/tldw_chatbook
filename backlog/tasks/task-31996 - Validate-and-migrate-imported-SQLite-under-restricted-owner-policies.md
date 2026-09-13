---
id: TASK-31996
title: Validate and migrate imported SQLite under restricted owner policies
status: Done
assignee: []
created_date: 2026-09-07 23:55
labels:
- backup-recovery
dependencies:
- task-31978
- task-31989
- task-31990
- task-31991
- task-31994
- task-31995
updated_date: 2026-09-10 15:35
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: Imported schema is qualified before migration and cannot activate unexpected SQL or application side effects.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Imported schema is qualified before migration and cannot activate unexpected SQL or application side effects.
- [x] #2 Valid supported SQLite/FTS stores migrate and validate under restricted connections.
- [x] #3 Unsupported capabilities, schemas, and resource failures remain explicit without unrestricted fallbacks.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Original component03 Task13: 1–2 behavioral hostile-trigger regression;3 registered read-only/restricted staged writable connection with trusted_schema OFF, disabled extensions, authorizer and cancellation/SQL/memory limits;4 exact installed SchemaPolicy catalog/version qualification before execution;5 only installed migrations in disposable staging under same restricted policy, followed by integrity/foreign-key/domain/asset validation;6 valid mutated owner fixtures, sentinels, budgets, missing primitives, legit FTS, versions, migrations/rollback;7 named focused SQLite/inventory guards;8–9 scoped lint/Bandit, doc updates, spec/code review and task-owned commit. Reuse existing SchemaPolicy and recovery owner conventions; no ordinary repository instantiation/unrestricted fallback. Task11 recovered owner pending review; qualify it after final contract.
<!-- SECTION:PLAN:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-03-capture-archives.md#task-13)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Begin independent validation implementation using original plan and existing installed owner policies. Task11 catalog implementation available pending review; final dependency qualification remains required. Source investigation /private/tmp/chatbook-backup-task13-source-map.md is evidence only, original spec governs.
Root independent source review approves restricted validator scope: exact frozen catalog precedes version/domain; installed-only migrations remain same restricted connection with rollback; extension/trust/authorizer/progress limits fail closed. 47new and373guard evidence reviewed. Staging ownership/publication and dependency file mapping remain executor responsibilities. Commit waits Task11 shared recovered/privateSQLite delta finalization; no new validation code requested.
Root independent source review approved. Final combined recovered+validator83passed14.10s includes discovered chat.attachments alias. Original validator47pass, privateSQLite guards373pass1existingWindows-onlyskip; focusedadapters6pass; newmodule/testRuffclean; currentproductionRuff45==45baseline andBandit7==7nonew. See /private/tmp/chatbook-sqlite-validation-report.md. Exact current Task11+13 index prepared together while pre-existing TTS/admission/docsWIP preserved. Staging placement/filedependency mappings remain originalexecutors responsibilities.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Validates exact installed SQLite/FTS schema and domain data before supported migrations under extension-disabled/trusted-schema-off, authorizer and resource/cancellation limits. Known migration executes on same restricted connection with rollback on failure. No unrestricted fallback or imported application code.
<!-- SECTION:FINAL_SUMMARY:END -->
