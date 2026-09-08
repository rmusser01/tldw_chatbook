---
id: TASK-31992
title: Add recovery inventory for configuration durable files and optional content
status: In Progress
assignee:
  - codex
created_date: '2026-09-07 23:52'
labels:
  - backup-recovery
dependencies:
  - task-31978
  - task-31986
  - task-31987
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: All app-owned durable file categories and directory topology are classified with explicit dependency and metadata rules.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All app-owned durable file categories and directory topology are classified with explicit dependency and metadata rules.
- [ ] #2 External folders/models/temporary media/diagnostics remain opt-in without weakening custom app-owned storage coverage.
- [ ] #3 Aliases, unknown durable entries, missing required files, and unsupported metadata produce truthful coverage results.
<!-- AC:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-02-inventory-admission.md#task-9)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.

## Implementation Plan

1. Preserve the existing tree inventory and establish a new behavioral metadata RED; retain empty-directory evidence.
2. Add immutable versioned preview metadata and locally supplied optional selections, keeping explicit topology and aggregate owner overlap checks.
3. Trace registered producers and canonical selectors for current/history config, templates/prompts, local definitions, persona assets, TTS stores/references, generated assets and model stores; map unresolved rows explicitly and keep completeness blocked.
4. Add import-light installed owner factories, known managed-secret locations and pure relocation policy without runtime bootstrap or secret decryption.
5. Classify external/model/diagnostic/temporary selections and output/control exclusions; refuse links, aliases, mounts, unsupported metadata and missing selected/required sources.
6. Run targeted file inventory cases and affected architecture/import/admission guards, scoped static checks and self-review.
7. Update owner documentation and exact implementation evidence; commit task-owned files, leaving status In Progress and ACs unchecked for independent controller review.

ADR required: yes
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: direct implementation of the approved recovery inventory and metadata contract, reusing ADR-029/030/036/059/060.

## Implementation Notes

Implemented versioned directory/file metadata and pure selection-aware recovery
inventory; config/history/definition owners, TTS schema4/BLOB capture, persona
manual/builtin references, Skills and installed model closure adapters. Config
remapping uses exact installed section.key names and never edits arbitrary prose.
Checked SHA256 streaming retains native retirement/admission; ordinary raw writers
remain explicitly participant_pending for coordinated task10 drain. Existing
outputs, unknown owners, unsafe pending state and unsupported metadata still block
Complete; no archive/activation/credential qualification is claimed.

Validation: final focused inventory51 passed; combined core/inventory/SQLite and
owner census181 passed; shared-reader/admission guards62 passed; runtime/path/Skills/
TTS seams78 passed; new SQLite behavioral copy/policy23 passed. Diagnostic inventory
guard remains a proven clean-BASE failure in unchanged earlier-slice files, retained
as task26 branch verification debt. Scoped27-file fatal lint,13-file format check,
exact lazy-export AST comparison and git diff --check passed. Full command history,
intermediate failures and owner handoffs are in the task9 execution report.

ADR: backlog/decisions/126-complete-local-backup-and-recovery.md, reused with the
approved ownership/private-path/sync ADRs. Design references preserved. Five-digit
Backlog direct-file fallback used per lessons-backlog-hygiene; no CLI rewrite or
malformed task-task file. Status stays In Progress and ACs unchecked for controller
review. No unrelated work, full suite, network/download, push or merge.
