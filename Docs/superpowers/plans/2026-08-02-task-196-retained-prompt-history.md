# Feature Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver TASK-196 as PR 2 of 6: expose bounded, index-backed retained Prompt history through the app-wired scope and safely restore an eligible snapshot as a new current version.

**Architecture:** Treat local `sync_log` create/update payloads as retained history, but query them with a Prompt-specific bounded DB API and composite index. Normalize history through `PromptScopeService`, model disclosure/paging state in the Library layer, and reuse the ordinary conditional update transaction for restore. Compatibility-invalid snapshots are preview-only.

**Tech Stack:** Python 3.11+, SQLite schema migration, Textual workers/state, pytest with real SQLite, Backlog.md CLI.

---

## Merge Gate and ADR

- Begin only after the TASK-202 PR is merged and visible on latest `origin/dev`; create a fresh branch/worktree from that commit.
- ADR required: yes.
- ADR path: allocate the first number unreserved across latest `dev` and visible in-flight repository refs as `backlog/decisions/NNN-local-prompt-retained-version-history.md`.
- Reason: this makes retained sync payloads a user-visible source and fixes restore, keyword, pruning, and concurrency semantics.
- Commit the ADR and linked Backlog plan before implementation code.

## File Responsibility Map

- Modify `tldw_chatbook/DB/Prompts_DB.py`: v3→v4 migration, composite history index, bounded query, atomic keyword snapshots, conditional restore support through ordinary update.
- Modify `tldw_chatbook/Prompt_Management/local_prompt_service.py`: remove whole-log scanning behavior or convert it to the same bounded DB contract so no public adapter retains the unsafe path.
- Modify `tldw_chatbook/Prompt_Management/prompt_scope_service.py`: implement app-wired local history and conditional restore routing.
- Modify `tldw_chatbook/Prompt_Management/prompt_normalizers.py`: one normalized retained-version envelope and compatibility reason.
- Modify `tldw_chatbook/Library/library_prompts_state.py`: immutable `PromptHistoryState`, rows, previews, paging, eligibility, outcome helpers.
- Modify `tldw_chatbook/Widgets/Library/library_prompts_canvas.py`: lazy disclosure, version rows, read-only preview, load-older, restore action.
- Modify `tldw_chatbook/UI/Screens/library_screen.py`: token-guarded history workers, confirmation, conflict/no-change/success outcomes.
- Add/modify DB, service, state, and UI tests under `Tests/Prompts_DB/`, `Tests/Prompt_Management/`, `Tests/Library/`, and `Tests/UI/`.
- Modify `Docs/User_Guide/library/prompts.md`, TASK-196, and the new ADR.

## Task 1: Refresh the Baseline and Record the Decision

- [ ] Fetch latest `origin/dev`, confirm TASK-202 is merged, and create `codex/task-196-retained-prompt-history` in a fresh ignored worktree.
- [ ] Inspect `backlog/decisions/` on that baseline plus visible in-flight repository refs, allocate the first unreserved number, and write the ADR with: retained-not-complete terminology; create/update-only rows; page-size+1 predecessor; pruning gaps; additive keywords; conditional restore; no-change; compatibility-only snapshots; transaction rollback.
- [ ] Mark TASK-196 In Progress, replace its comma-collapsed criterion with the approved measurable criteria, and add a plan containing the exact ADR path/reason.
- [ ] Commit ADR + Backlog state before code.

## Task 2: Add the Bounded Indexed History Query

- [ ] Add failing migration tests in a focused file such as `Tests/Prompts_DB/test_prompts_db_retained_history.py` for fresh v4 schema and v3→v4 upgrade.
- [ ] Add a failing `EXPLAIN QUERY PLAN` regression with many unrelated `sync_log` rows. The plan must use the new composite index for entity/type/operation/order rather than scan the table.
- [ ] Add failing paging tests for create/update-only rows, descending change IDs, strict limit validation, `before_change_id`, at most `page_size + 1` decoded snapshots, and an exact index-only retained count that is refreshed in the page read transaction.
- [ ] Run the red DB tests.

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Prompts_DB/test_prompts_db_retained_history.py -q
```

- [ ] Bump `_CURRENT_SCHEMA_VERSION` and add a v3→v4 migration that creates the composite history index without modifying retained rows.
- [ ] Implement a DB method such as `get_prompt_history_entries(entity_uuid, limit, before_change_id=None)` with fixed operations (`create`, `update`), parameterized values, descending `change_id`, bounded JSON decoding, and no `get_sync_log_entries(0)` fallback.
- [ ] Make malformed payload decoding produce an explicit record error/preview state rather than dropping the row or failing the whole page.
- [ ] Run DB tests green.

## Task 3: Capture Effective Keywords Atomically

- [ ] Add red tests proving new create/update snapshots contain ordered effective keywords only after keyword membership settles, and that row/links/snapshot roll back together on keyword failure.
- [ ] Characterize older snapshots with no `keywords` key; they must normalize with `keywords_captured=False`.
- [ ] Move/add snapshot payload construction inside the existing Prompt transaction after keyword links are final. Do not version collections, usage, deletion, or derive keyword ownership elsewhere.
- [ ] Verify nested/immediate transaction behavior and preserve older consumers' additive-field compatibility.

## Task 4: Normalize Retained Versions and Restore Eligibility

- [ ] Add red normalizer tests for legacy text, supported structured-v2 Prompt/Recipe, foreign structured-v1, malformed JSON, definition/compiled mismatch, artifact-type/kind mismatch, unknown format/schema, and future artifact type.
- [ ] Extend `prompt_normalizers.py` with a normalized version model/mapping containing version, change ID, operation, timestamp, fields, keyword capture state, compatibility state/reason, and restore eligibility.
- [ ] Compare each visible row only with its immediate retained predecessor. Report `Created` for v1 and `Earlier baseline unavailable` across a pruning/version gap.
- [ ] Preserve exact preview text without attempting v1 conversion or reparsing foreign definitions.

## Task 5: Route the App-wired Local Service and Implement Conditional Restore

- [ ] Add failing `Tests/Prompt_Management/test_prompt_scope_service.py` cases proving `mode="local"` routes list/restore to the live in-module `LocalPromptService`; keep server routing intact for existing callers.
- [ ] Add focused local adapter tests proving the whole-log scanning helper is gone and every page invokes the bounded DB method once.
- [ ] Implement paginated `list_prompt_versions` on the live local adapter and normalize it through `PromptScopeService`.
- [ ] Extend restore to accept retained `change_id`, selected version, and `expected_version`. Re-resolve the exact retained row and current state in one immediate conditional transaction; reject pruned/replaced snapshots, stale versions, and deleted current records before validating the target and calling ordinary update fields.
- [ ] For older snapshots without keywords, retain current keywords and disclose that choice. For modern snapshots, replace keywords in the same transaction.
- [ ] Detect byte-identical current content/metadata/keywords before update and return `no_change` without a new sync row.
- [ ] Add rollback tests for pruning-after-preview, deleted current records, duplicate name, and keyword validation failure; current row and history count must remain unchanged.
- [ ] Run service and DB tests green.

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Prompts_DB/test_prompts_db_retained_history.py Tests/Prompt_Management/test_local_prompt_service.py Tests/Prompt_Management/test_prompt_scope_service.py -q
```

## Task 6: Add Pure History UI State

- [ ] Add red tests in `Tests/Library/test_library_prompts_state.py` for closed/loading/loaded/error state, append-only older pages, selected preview, scope token, count, dirty disablement, compatibility disablement, no-change, conflict, and restore success copy.
- [ ] Add immutable history row/page/state types and reducers/builders. Keep Textual imports out of this state logic.
- [ ] Model the captured current version and selected source version explicitly; never infer either from row position.
- [ ] Ensure stale page/preview results cannot merge when prompt identity or request token changes.

## Task 7: Build the Lazy Disclosure and Guarded Workers

- [ ] Add failing UI pilots for collapsed-by-default `Retained history (…)` changing to the exact `Retained history (N)`, first page load only on open, Load older versions, read-only lane previews, literal text rendering, and Retry after error.
- [ ] Add dirty-working-copy and compatibility tests: viewing remains available; Restore is disabled with exact reason.
- [ ] Add confirmation tests for normal and Prompt↔Recipe restore. Confirm copy must state that a new current version is created.
- [ ] Implement history state ownership in `LibraryScreen`, using worker groups plus prompt identity/request tokens. A late result cannot update a different editor.
- [ ] Render the disclosure and preview in `LibraryPromptsListCanvas`. Do not mount editable block controls for history previews.
- [ ] On success refresh current detail/history and announce `Restored vX as current vY.`; on stale expected version enter the existing conflict state; keep the selected row after validation/name/keyword failures.

## Task 8: Documentation, Verification, and Merge Gate

- [ ] Document retained-history limits, lazy loading, preview-only records, conditional restore, new-version behavior, and old-keyword retention in `Docs/User_Guide/library/prompts.md`.
- [ ] Run affected suites.

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Prompts_DB Tests/Prompt_Management/test_local_prompt_service.py Tests/Prompt_Management/test_prompt_scope_service.py Tests/Library/test_library_prompts_state.py Tests/UI/test_library_prompts_canvas.py Tests/UI/test_library_shell.py -q
git diff --check
```

- [ ] Run the full suite, perform self-review, request independent review, and record exact verification.

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest
```

- [ ] Complete all TASK-196 criteria, add notes linking the ADR and naming modified files/tests, then mark Done only when all checks pass.
- [ ] Open one ready PR against `dev`, resolve every review/CI issue, merge, and verify the merge on `origin/dev`. Do not start TASK-198 implementation before that merge is confirmed.
