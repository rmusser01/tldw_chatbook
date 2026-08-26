# Feature Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver TASK-203 as PR 6 of 6: add visible-row Prompt multi-select with keyboard support, explicit-ID Chatbook export, and confirmed partial-outcome bulk soft delete.

**Architecture:** Reuse `Library.row_selection.RowSelection` and TASK-198's browse fingerprint. Selection is local and limited to visible rows; every scope change clears it. TASK-197 owns explicit-ID export, TASK-202's modal owns confirmation presentation, and `PromptScopeService.bulk_delete_prompts` owns mutation routing/outcomes. Refresh then reconcile successes and visible failures.

**Tech Stack:** Python 3.11+, Textual keyboard/UI state, Prompt scope service, Chatbook export scope, pytest, Backlog.md CLI.

---

## Merge Gate and ADR

- Begin only after TASK-197 is merged into latest `origin/dev`; create a fresh worktree/branch.
- ADR required: no.
- ADR path: N/A.
- Reason: selection/UI behavior uses ADR-011/ADR-040 and the TASK-197 Chatbook Prompt-record ADR; no new ownership or portable format boundary is introduced.
- Bulk tagging remains out of scope because TASK-203 acceptance covers selection, export, and delete only.

## File Responsibility Map

- Modify `tldw_chatbook/Library/row_selection.py`: admit normalized Prompt identities and explicit visible-scope behavior without weakening other kinds.
- Modify `tldw_chatbook/Library/library_prompts_state.py`: selected row state, select mode, fingerprints, bulk outcomes and refresh reconciliation.
- Modify `tldw_chatbook/Widgets/Library/library_prompts_canvas.py`: markers, select toolbar, bulk action controls, row keyboard semantics.
- Modify `tldw_chatbook/UI/Screens/library_screen.py`: scope-clear ownership, key actions, confirmation fingerprint, export launch, delete worker and reconciliation.
- Modify `tldw_chatbook/Prompt_Management/prompt_scope_service.py`: normalized local-only `bulk_delete_prompts` with per-ID outcomes.
- Reuse `tldw_chatbook/Widgets/Library/prompt_delete_confirmation_modal.py` and `tldw_chatbook/Library/library_export_scope.py`; change only if integration requires a typed bulk input already anticipated by TASK-202/TASK-197.
- Add/modify `Tests/Library/test_row_selection.py`, `Tests/Library/test_library_prompts_state.py`, a focused bulk Prompt test file if useful, `Tests/Prompt_Management/test_prompt_scope_service.py`, `Tests/UI/test_library_prompts_canvas.py`, and `Tests/UI/test_library_shell.py`.
- Modify `Docs/User_Guide/library/prompts.md` and TASK-203.

## Task 1: Refresh the Final Merge-gated Baseline

- [ ] Fetch `origin/dev`, prove TASK-197 is merged, and create `codex/task-203-prompt-multiselect` from it.
- [ ] Mark TASK-203 In Progress and expand its criteria with visible-row scope, keyboard controls, explicit-ID export retention, confirmed partial delete, page clamping, and failure reconciliation.
- [ ] Add implementation plan with ADR check and commit Backlog state before code.

## Task 2: Extend Selection and Pure Prompt State

- [ ] Add red `Tests/Library/test_row_selection.py` cases for normalized local Prompt keys, duplicate toggle, Select visible, Clear, export scope, and kind isolation.
- [ ] Add red `Tests/Library/test_library_prompts_state.py` cases for entering/exiting select mode, selected markers/count, exact visible scope, and clearing on query/collection/page/sort fingerprint changes.
- [ ] Add reducers for delete outcome reconciliation: successes clear; failures retain only if still visible after refresh; hidden/reordered failures are reported then cleared; select mode/empty result remains truthful.
- [ ] Keep selection keys as backend + stable source ID, never row index or cached artifact type.
- [ ] Run the pure red tests, implement the smallest changes, and rerun green.

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Library/test_row_selection.py Tests/Library/test_library_prompts_state.py -q
```

## Task 3: Add the Local Bulk-delete Service Contract

- [ ] Add service tests for zero/duplicate/invalid/foreign IDs, Prompt and Recipe rows, concurrent type change, already deleted, mixed successes/failures, policy denial, and no body content in errors/logs.
- [ ] Define a normalized `PromptBulkOutcome` containing successful identities and bounded failed identity/category entries; do not throw away partial success.
- [ ] Implement `PromptScopeService.bulk_delete_prompts(mode="local", prompt_identifiers=...)`. Re-fetch/classify each current record before soft delete; do not trust cached UI artifact type.
- [ ] Route through the live local adapter and existing soft-delete semantics. Server/mixed bulk calls remain unavailable from Library.
- [ ] Decide transaction granularity explicitly in tests: per-ID settlement is required to return useful partial outcomes; a failure for one ID must not misreport other settled IDs.

## Task 4: Render Visible-row Select Mode and Keyboard Behavior

- [ ] Add failing canvas pilots: every visible row gets a marker; Enter/Space toggles the focused row instead of opening it; Escape clears/exits; Clear removes selection but stays in mode; Select visible selects only current page rows.
- [ ] Assert the action strip copy shows selected count and visible-vs-total scope, with stable controls for Select visible, Clear, Export selected, and Delete selected.
- [ ] Add normal row activation regression proving editor opening still works outside select mode.
- [ ] Implement state-driven markers/actions in `library_prompts_canvas.py` and screen-level key routing in `library_screen.py`, respecting focused text controls/modals.
- [ ] On query/collection/page/sort change, clear selection before requesting the new browse scope. No hidden selection survives.
- [ ] Inspect 80x24 and 100x30 layouts for action clipping and focus order.

## Task 5: Integrate Explicit-ID Chatbook Export

- [ ] Add UI tests that Export selected captures selected identities plus current browse fingerprint and opens TASK-197's export canvas as `ExportScope(kind="prompts", ids=...)`.
- [ ] Assert only selected IDs are resolved even when other rows match. Concurrently missing rows use TASK-197's skip/no-empty behavior.
- [ ] Keep selection after export success, user cancellation, and export failure so retry/another action remains available.
- [ ] Invalidate an open export launch if the captured scope/selection cannot be represented safely; do not silently export current visible rows instead.

## Task 6: Integrate Fingerprinted Bulk-delete Confirmation

- [ ] Add tests for Prompt-only, Recipe-only, mixed counts, bounded name preview, dirty editor irrelevance, Cancel, and literal markup-looking names using `PromptDeleteConfirmationModal`.
- [ ] Capture a deterministic selection + browse-scope fingerprint when opening confirmation. Any selection/scope change before Confirm invalidates the decision and performs no deletion.
- [ ] Call `bulk_delete_prompts` only after valid confirmation. Disable/guard duplicate submission while the worker is active.
- [ ] Refresh browse rows, active collection catalog/count, and rail count after settlement; then reconcile selection from the refreshed visible IDs.
- [ ] Clamp the page to the new last valid page. If no rows remain, focus the Prompt list toolbar and render the honest empty state.
- [ ] Report bounded partial results: success count, visible retry failures retained, and off-view/reordered failures reported and cleared.

## Task 7: Regression, Documentation, and Visual QA

- [ ] Update `Docs/User_Guide/library/prompts.md` with visible-page selection scope, keyboard behavior, selection-clearing transitions, export retention, confirmation, partial delete, and absence of bulk tagging.
- [ ] Run focused/affected suites.

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Library/test_row_selection.py Tests/Library/test_library_prompts_state.py Tests/Library/test_library_export_scope.py Tests/Prompt_Management/test_prompt_scope_service.py Tests/UI/test_library_prompts_canvas.py Tests/UI/test_library_shell.py Tests/UI/test_prompt_delete_confirmation_modal.py -q
git diff --check
```

- [ ] Visually inspect normal/select/empty/partial failure/last-page/narrow-terminal states. Verify focus, markers, count copy, action grouping, and no hidden retained selection.
- [ ] Run the full suite, self-review for stale destructive intent, incorrect ID scope, accidental export clearing, page underflow, and off-view selection; request independent review.

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest
```

## Task 8: Close the Series

- [ ] Complete TASK-203 criteria and implementation notes with tests/docs/ADR check; mark Done only after DoD.
- [ ] Open one ready PR against `dev`, resolve every review/CI issue, merge, and confirm the merge on `origin/dev`.
- [ ] Audit TASK-202/196/198/199/197/203 and absorbed TASK-2700/2701 statuses, ADR links, notes, and merged PR references. Report any unfinished external follow-up without expanding this six-PR scope.
