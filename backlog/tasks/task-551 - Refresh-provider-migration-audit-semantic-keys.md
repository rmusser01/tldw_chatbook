---
id: TASK-551
title: Refresh provider migration audit semantic keys
status: Done
assignee: []
created_date: '2026-07-24 21:26'
updated_date: '2026-07-24 21:27'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the provider-migration static guard after intentional multiline formatting changed the normalized source lines without adding, removing, or reclassifying legacy builder seams.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The migration audit semantic snippets exactly match current legacy builder lines
- [x] #2 Audited reason categories and per-file match counts remain unchanged
- [x] #3 No production client construction or runtime policy code changes are made
- [x] #4 The provider migration audit and full RuntimePolicy suite pass
- [x] #5 Task notes record the verified drift, ADR decision, and validation
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record the four-file RED drift and compare audited snippets with actual normalized source lines.
2. Update only the audit document semantic snippets and informational line hints to this branch's current source.
3. Run the focused provider-migration guard and full RuntimePolicy suite.
4. Review the diff and record verification.

ADR required: no
ADR path: N/A
Reason: This is a documentation-oracle correction for unchanged, already-classified compatibility seams; it changes no storage, service contract, provider boundary, or runtime architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Summary: Refreshed four provider-migration audit paths whose normalized semantic keys had drifted after multiline formatting, without changing any production construction seam.

- Replaced the two study-handler one-line call keys with their current first-line forms.
- Replaced seven assignment-prefixed `app.py` keys with the continuation-line forms actually matched by the guard, preserving all 34 audited app matches.
- Replaced the former one-line annotated bootstrap signature with its current first-line form, preserving all six bootstrap matches.
- Refreshed the affected `app.py`, study-handler, and bootstrap line hints to the current branch.

RED evidence:
- `_audit_drift()` reported exactly four paths: both study handlers, `app.py`, and `runtime_policy/bootstrap.py`.
- The mismatch was limited to old versus current normalized line forms; actual and audited call counts were unchanged.

Verification:
- `Tests/RuntimePolicy/test_server_client_provider_migration_audit.py`: 10 passed.
- Full `Tests/RuntimePolicy`: 251 passed; one pre-existing `requests` dependency-version warning.
- `git diff --check`: passed.
- Diff review confirmed no Python or other production file was changed.

ADR required: no
ADR path: N/A
Reason: This corrects documentation keys for existing compatibility seams and changes no storage, service contract, provider boundary, or runtime architecture.

Files modified:
- `Docs/Development/server-client-provider-migration-audit.md`
- `backlog/tasks/task-551 - Refresh-provider-migration-audit-semantic-keys.md`
<!-- SECTION:NOTES:END -->
