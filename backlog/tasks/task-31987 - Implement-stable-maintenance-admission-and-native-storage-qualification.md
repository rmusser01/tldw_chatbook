---
id: TASK-31987
title: Implement stable maintenance admission and native storage qualification
status: In Progress
assignee:
  - codex
created_date: '2026-09-07 23:49'
labels:
  - backup-recovery
dependencies:
  - task-31978
  - task-31986
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: Participating processes cannot mutate admitted namespaces during maintenance, including aliases and inode replacement.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Participating processes cannot mutate admitted namespaces during maintenance, including aliases and inode replacement.
- [ ] #2 Deadlock, timeout, cancellation, and crashed-holder cases preserve data and durable recovery evidence.
- [ ] #3 Native publication cannot overwrite an existing artifact, and unsupported storage returns an explicit unavailable capability.
<!-- AC:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-02-inventory-admission.md#task-4)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.

## Implementation Plan

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved native storage and admission contract; reuse ADR-126.

1. Establish importable behavioral RED publication regression.
2. Implement stable namespace registration, verified aliases, reserved remapping, and native cross-process admission outside managed roots.
3. Close mutation admission before draining established participants; order acquisition and preserve safe retirement on timeout/cancellation.
4. Implement private creation and qualified durable native no-replace publication with pinned parents; refuse unsupported operations/filesystems.
5. Exercise independent child processes for aliases, inode replacement, remapping, retirement, death, stale evidence, contention and native publication failures.
6. Run named focused/native tests, exact persistence census, affected legacy advisory-lock guard, scoped lint/format and diff checks.
7. Self-review, record exact evidence and interfaces, and commit scoped changes. Leave In Progress and AC unchecked for controller review.

## Implementation Notes

Implemented the ADR-126 cooperative maintenance/native publication foundation. Stable per-namespace OS locks survive target inode replacement; shared/hardlink/symlink/nested roots receive ordered common admission. Maintenance closes gates and drains established connections without holding registry authority, then revalidates/pins scope. Remapping durably reserves old/new aliases before draining and retains fail-closed evidence on interruption. Legacy InstanceLockStatus remains advisory.

Native Darwin renameatx_np(RENAME_EXCL) publishes owned regular files and empty/populated directories with pinned parents, no-follow tree validation, fsync/F_FULLFSYNC and parent flushes. Existing names are never replaced; no hardlink fallback exists. Installed versioned evidence matches only demonstrated Darwin25.5.0/arm64/Python3.12.11/APFS flags76583040. Overall isolated restore/replacement remain unavailable. The record ships in wheel/sdist and is explicitly required by distribution checks.

Source census includes exact open/write/native rename-callable and os.replace symbols. Owner documentation: [Native admission contract](../docs/backup-recovery-native-admission.md). ADR required: yes; reused [ADR-126](../decisions/126-complete-local-backup-and-recovery.md). Supplemental registration/remap/incompatible/per-call-cancellation APIs and narrowly scoped packaging changes follow controller rulings. No full sweep was run.

Evidence used the read-only interpreter `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python` in the isolated clone. Behavioral RED: initial publication test DID NOT RAISE FileExistsError; nested-root child escaped maintenance; linked directory payload published; native producer census missed actual callable aliases. Each was corrected and verified GREEN. Final commands/results:

- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Backup_Recovery/test_admission.py Tests/Backup_Recovery/test_native_files.py -q`: 36 passed, 1 existing warning, 7.91s; no native skips.
- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Architecture/test_backup_owner_inventory.py -q`: 10 passed, 4 existing warnings, 9.90s.
- `UV_CACHE_DIR=/tmp/task31985-uv-cache /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Utils/test_instance_lock.py -q`: 10 passed, 1 existing warning, 0.81s; matches the prior 10-test baseline.
- `UV_CACHE_DIR=/tmp/task31985-uv-cache /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Packaging/test_installed_distribution.py::test_built_artifacts_match_distribution_contract -q`: 1 passed, 1 existing warning, 14.65s; build/source output stayed outside checkout.
- Scoped Ruff E9/F63/F7/F82 over all changed Python passed; format check over the six focused/native/architecture files passed; `git diff --check` passed.

Self-review verified pinned-root qualification, descriptor/lock lifetimes, no registry hold during drain, strict private registry metadata and actual source/destination evidence. All scopes must use their shared authoritative control root (bootstrap association is task5); independent custom roots are not automatically federated. Pending remap reconciliation is reserved for the local journal/control owner (task18), and historical aliases intentionally remain conservative. Process-crash/OS-flush tests do not simulate physical power loss. Owner integration, startup fences and higher-level restore gates are later slices. Status remains In Progress with unchecked criteria for controller independent review.

### Independent review fix round 1

Addressed both Important findings: full native directory barriers now follow metadata publication, and the installed evidence is strictly validated against supported protocol 2 before capability admission. Actual APFS directory F_FULLFSYNC succeeds; file/directory publication ordering tests delegate to real syscalls. A bounded private version-1 `registry.pending.json` before/after intent is fully persisted before local registry replacement and remains until the new registry and parent metadata pass their full barriers. This prevents final-remap barrier failure from silently clearing the only pending evidence. Independent processes verify refusal for initial/intent/reservation/final mapping failures; cleanup tests permit only retained-intent refusal or a previously durably committed mapping. No physical power-loss claim or ad-hoc reconciliation API is added. Native intent-removal producers now have exact census rows. Status and acceptance checkboxes remain unchanged for controller re-review.

Fix-round validation: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Backup_Recovery/test_native_files.py Tests/Backup_Recovery/test_admission.py -q --tb=short` — 65 passed, 1 baseline warning in 12.46s; `... -m pytest Tests/Architecture/test_backup_owner_inventory.py -q` — 11 passed, 4 baseline warnings in 11.60s. Scoped Ruff E9/F63/F7/F82, six-file format check and diff check passed. Explicit initial-only registry creation prevents the new intent writer from reconstructing an existing disappeared registry. The crash helper now captures the strengthened constructor refusal as well as the existing remap fence. No unrelated suites repeated.
