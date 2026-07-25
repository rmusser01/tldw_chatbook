---
id: TASK-557
title: Reconcile Library ingest state and mounted test contracts
status: Done
assignee: []
created_date: '2026-07-25 17:53'
updated_date: '2026-07-25 18:04'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore deterministic Library ingest behavior after persisted type options were added by keeping render derivation side-effect free, treating generic chunk controls as built-in, and aligning mounted tests with scoped config and recompose contracts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Building Library ingest display state does not mutate the screen-owned form echo
- [x] #2 Generic chunk-size and overlap controls follow the sibling Chunk toggle instead of optional-package detection
- [x] #3 Search-history and rail-preference precedence tests reject only calls to their own config sections
- [x] #4 The different-canvas ingest completion test tolerates the rail recompose while preserving selection
- [x] #5 The five deterministic RED cases and focused ingest/config suites pass
- [x] #6 Task notes record RED evidence and ADR applicability
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the five deterministic RED failures and trace each to the current form/config/recompose contract.
2. Derive the render-time generic options from a copied form so compose remains side-effect free.
3. Distinguish sibling-field dependencies from optional-feature dependencies for generic chunk controls.
4. Narrow config precedence sentinels to the sections under test and poll across the expected rail recompose.
5. Run the deterministic cases, focused Library ingest/config suites, the full Library shell module, Ruff, formatter, and diff checks.
6. Self-review and document the separate note-conflict flake.

ADR required: no
ADR path: N/A
Reason: These are routine correctness and test-determinism repairs inside the existing Library ingest form and mounted UI contracts; no storage, ownership, service, or cross-module boundary changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reconciled the five deterministic Library-shell failures exposed by the UI sweep. Render derivation now copies generic type options instead of mutating the screen-owned form. The ingest canvas distinguishes sibling-field dependencies from optional installed-feature dependencies, so chunk size/overlap follow the Chunk toggle. Mount-time no-op option events are suppressed before they can repopulate reset state. Config precedence sentinels now reject only the search or rail section under test, and the live media-count assertion tolerates the expected rail recompose. RED evidence: five focused cases failed for over-broad get_cli_setting interception, mutated reset type_options, permanently disabled chunk inputs, and transiently unmounted media row; a new mount-event test also failed with three spurious OptionValueChanged messages. Verification: seven focused regressions pass; ingest capabilities/state/canvas suites pass 133/133; the full Library shell passes 257/257; Ruff, formatter, and diff checks pass. ADR required: no; these repairs preserve the existing Library form and mounted UI boundaries. Modified: tldw_chatbook/Library/ingest_capabilities.py, tldw_chatbook/UI/Screens/library_screen.py, tldw_chatbook/Widgets/Library/library_ingest_canvas.py, Tests/UI/test_library_ingest_canvas.py, Tests/UI/test_library_shell.py, and this task file.
<!-- SECTION:NOTES:END -->
