---
id: TASK-639
title: Reconcile persistent diagnostic inventory after verified repairs
status: Done
assignee: []
created_date: '2026-07-25 19:54'
updated_date: '2026-07-25 19:56'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the production diagnostic governance sentinel after reviewed branch repairs moved existing calls and added two constant Anthropic validation warnings without changing persistent sink semantics.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The inventory records the two new TASK-492 Anthropic validation warnings
- [x] #2 All other owner changes are limited to reviewed call movement or formatting
- [x] #3 The persistent sink topology retains the same five files and unchanged sink-call digests
- [x] #4 The architecture diagnostic inventory tests pass
- [x] #5 No production diagnostic or sink behavior changes as part of reconciliation
- [x] #6 Task notes record RED evidence, ADR decision, verification, and self-review
- [x] #7 The canonical inventory generator is Ruff-formatted and lint-clean
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the full-suite inventory sentinel failure and compute the exact owner and sink delta without writing.
2. Review every changed owner against production diffs and verify privacy-safe diagnostic content and stable sink semantics.
3. Regenerate the committed inventory with the canonical script.
4. Run focused architecture and privacy-sentinel tests, Ruff/format checks, and git diff --check; self-review the generated delta.

ADR required: no
ADR path: N/A
Reason: This refreshes a generated governance snapshot after already-reviewed implementation changes; it introduces no new diagnostic policy, persistent sink, or architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Summary: Reconciled the generated production diagnostic inventory after explicitly reviewing every owner and sink delta, and formatted the canonical generator without behavior changes.

RED evidence and review:
- A fresh full-suite run failed Tests/Architecture/test_persistent_diagnostic_inventory.py because the committed snapshot no longer matched the branch. The run was interrupted at 1 failed, 255 passed, 1 skipped to diagnose the first failure immediately.
- Owner files remain 422. TASK-494 remains exactly 7,143 calls. Existing call movement or formatting changed digests in Subscriptions_DB.py, eval_runner.py, mcp_workbench.py, home_screen.py, library_screen.py, conflicts_tab.py, schedules_workbench.py, watchlists_collections_screen.py, and app.py without changing their call counts.
- TASK-492 rises from 1,008 to 1,010 solely because the reviewed Anthropic tool normalization repair added two constant validation warnings. Neither warning includes tool payloads, identifiers, exception text, or user content.
- Persistent sink files remain five. app.py sink line numbers moved by 18 lines, while both sink call digests remain byte-for-byte identical.
- The generator itself had a pre-existing Ruff formatting delta; AC was updated before applying the single expression-wrap change.
- No production diagnostics or sinks were modified by this reconciliation.

Verification:
- Canonical inventory check: 422 owners, 1,010 TASK-492 calls, 7,143 TASK-494 calls, 5 sink files.
- Architecture inventory plus persistent diagnostic boundary/sentinel matrices: 29 passed.
- JSON summary/topology assertions: passed.
- Ruff format check: 2 files already formatted.
- Ruff check: all checks passed.
- git diff --check: passed.
- Self-review: the generated delta contains ten reviewed owner updates, two line-only sink offsets with stable digests, and the expected +2 constant warnings; no policy or privacy boundary changed.

ADR required: no
ADR path: N/A
Reason: This refreshes a generated governance snapshot and formatter-clean tooling after existing reviewed implementation changes; it makes no architectural decision.

Files modified:
- Docs/security/production-diagnostic-inventory.json
- scripts/check_persistent_diagnostic_inventory.py
- backlog/tasks/task-639 - Reconcile-persistent-diagnostic-inventory-after-verified-repairs.md
<!-- SECTION:NOTES:END -->
