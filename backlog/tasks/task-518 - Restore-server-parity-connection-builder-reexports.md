---
id: TASK-518
title: Restore server parity connection-builder reexports
status: Done
assignee: []
created_date: '2026-07-24 18:37'
updated_date: '2026-07-24 18:37'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the documented server-parity compatibility surface after automated unused-import cleanup removed two intentional connection-contract reexports.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 server_parity_contracts reexports active-server status and server-switch invalidation builders
- [x] #2 All five connection builders remain object-identical to server_connection_contracts exports
- [x] #3 Intentional reexports are explicit and Ruff-clean
- [x] #4 The server connection/parity contract tests pass
- [x] #5 Task documentation records the merge-base failure, regression commit, ADR decision, verification, and implementation notes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the exact AttributeError on feature branch and merge base and trace it to the automated F401 removal.
2. Restore the two intentional reexports explicitly so Ruff recognizes the public compatibility surface.
3. Run server connection and parity contract tests.
4. Run Ruff format/check and git diff --check; independently review before completion.

ADR required: no
ADR path: N/A
Reason: This restores an existing tested compatibility export removed by mechanical lint cleanup; it does not introduce a new interface or architectural decision.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Summary: Restored the two intentional server-connection builder reexports removed by mechanical unused-import cleanup.

Approach and base comparison:
- On both merge base ba6b45cdf4dd548796e072f5933cdcf44c8c0344 and the feature branch, test_server_parity_contracts_reexports_connection_builders failed with AttributeError because server_parity_contracts no longer exposed build_active_server_status_contract.
- Commit f56eb4121 mechanically removed build_active_server_status_contract and build_server_switch_invalidation_contract as F401 findings while retaining the other three connection-builder reexports.
- Restored only those two imports as explicit same-name aliases, making their compatibility reexport intent visible to Ruff. The builders remain the original objects from server_connection_contracts; no wrapper or runtime behavior was added.

Verification:
- Exact reexport regression: 1 passed.
- Tests/UX_Interop/test_server_connection_contracts.py plus Tests/UX_Interop/test_server_parity_contracts.py: 18 passed.
- Ruff format check: file already formatted.
- Ruff check: all checks passed.
- git diff --check: clean for owned files.
- Self-review: the production diff contains exactly two explicit imports; no unrelated behavior changed.

ADR required: no
ADR path: N/A
Reason: This restores an existing tested compatibility surface and introduces no new interface or architectural decision.

Files modified:
- tldw_chatbook/UX_Interop/server_parity_contracts.py
- backlog/tasks/task-518 - Restore-server-parity-connection-builder-reexports.md
<!-- SECTION:NOTES:END -->
