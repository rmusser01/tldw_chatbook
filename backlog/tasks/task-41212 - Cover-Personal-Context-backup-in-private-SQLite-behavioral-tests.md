---
id: TASK-41212
title: Cover Personal Context backup in private SQLite behavioral tests
status: Done
assignee:
  - '@codex'
created_date: '2026-11-15 10:22'
updated_date: '2026-11-16 04:42'
labels:
  - tests
  - ci
  - backup
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_chatbook/pull/3524'
  - Tests/DB/test_private_sqlite.py
  - >-
    backlog/tasks/task-40134 -
    A-backup-captures-Personal-Context-s-repository-database.md
  - >-
    backlog/tasks/task-40722 -
    Restore-plugin-backup-coverage-in-private-SQLite-CI-inventories.md
priority: medium
type: bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After rebasing PR #3524 onto dev cc6006fcf5, the plugin and Notes CI repairs are present, but the backup-owner completeness check rejects recovery.personal_context.repository. TASK-40134 enabled that owner for centralized backup without adding it to the existing behavioral copy-backup matrix. Include the owner in the real SQLite copy test so coverage tracks the current policy.

Reproduced on Windows with Python 3.12.10, Textual 8.2.8 and pytest 8.3.5 under an isolated private profile. Run `python -m pytest Tests/DB/test_private_sqlite.py::test_every_backup_enabled_owner_has_a_behavioral_operation -q --tb=short --no-cov`. The expected-owner set has the extra entry `recovery.personal_context.repository`. The current static capture inventory and the original Notes delete/publication regression pass.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every enabled centralized backup owner is represented in the behavioral matrix, including Personal Context repository capture.
- [x] #2 The Personal Context owner passes the existing real transactional copy test and read-only verification.
- [x] #3 The original three PR #3524 blocker tests pass without weakening the owner registry, census, or Notes behavior assertions.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: Test-only coverage for an existing TASK-40134 backup policy, using the existing transactional-copy fixture.

1. Reproduce the current owner completeness failure and confirm the Personal Context capture calls copy_private_sqlite with the enabled recovery owner.
2. Add that owner once to COPY_BACKUP_OWNER_IDS, reusing the existing real-copy test.
3. Run the failing completeness check, the new owner parameter, and relevant private SQLite tests. Retain the passing static inventory and Notes repro results from the same source base.
4. Obtain independent review, update PR validation, and merge through the configured queue after required CI passes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The initial backup-owner completeness failure was reproduced and the Personal Context owner was exercised by the existing real SQLite copy/read-only matrix (26 checks passed). Latest dev 713fa22f92 (#3495) now includes that same owner entry, so the rebase adopts its coverage and this PR retains only the investigation record, with no duplicate DB test change. Existing plugin inventory and Notes Trash-publication repairs are also inherited from dev.

ADR required: no; existing private-SQLite backup policy and behavioral matrix are unchanged. The matrix, static inventory and targeted Notes receipts are retained with the PR investigation evidence.
<!-- SECTION:NOTES:END -->
