# Feature Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver TASK-198 as PR 3 of 6: replace sampled in-memory Prompt filtering with exact local pagination/search/sort and expose complete named-collection creation, rename, browse, and membership management.

**Architecture:** Add a dedicated `PromptScopeService.browse_prompts` contract backed by whitelisted SQLite queries and exact counts. Keep the Library surface local-only. Store browse scope/results as immutable UI state with request fingerprints; keep Prompt Save and collection membership as separate transactions and outcomes.

**Tech Stack:** Python 3.11+, SQLite, Textual workers and controls, pytest, Backlog.md CLI.

---

## Merge Gate and ADR

- Begin only after TASK-196 is merged into latest `origin/dev`; create a fresh branch/worktree.
- ADR required: no.
- ADR path: N/A.
- Reason: this surfaces existing local collection ownership through `PromptScopeService`; case-fold validation is service-level and adds no schema migration.

## File Responsibility Map

- Modify `tldw_chatbook/DB/Prompts_DB.py`: exact paginated Prompt browse query/count and transaction helpers used by collections.
- Modify `tldw_chatbook/Prompt_Management/prompt_scope_service.py`: browse DTO/service contract, complete local collection catalog, serialized case-fold validation, membership replacement.
- Modify `tldw_chatbook/Library/library_prompts_state.py`: `PromptBrowseScope`, `PromptBrowseResult`, paging/collection state and reducers.
- Modify `tldw_chatbook/Widgets/Library/library_prompts_canvas.py`: search, collection selector/chooser, paging, management and membership controls.
- Modify `tldw_chatbook/UI/Screens/library_screen.py`: dedicated Prompt browse state/workers, debounce and stale tokens, collection/membership outcomes.
- Add/modify tests in `Tests/Prompts_DB/`, `Tests/Prompt_Management/test_prompt_scope_service.py`, `Tests/Library/test_library_prompts_state.py`, `Tests/UI/test_library_prompts_canvas.py`, and `Tests/UI/test_library_shell.py`.
- Modify `Docs/User_Guide/library/prompts.md` and TASK-198.

## Task 1: Refresh the Merge-gated Baseline

- [ ] Fetch `origin/dev`, prove the TASK-196 merge is present, and create `codex/task-198-prompt-collections` from it.
- [ ] Mark TASK-198 In Progress and replace its current comma-collapsed criterion with the approved exact/local-only browse, complete collection catalog, multi-membership, and separate-outcome criteria.
- [ ] Add the implementation plan with ADR required/path/reason and commit Backlog state before code.

## Task 2: Define the Browse Contract in Pure Tests

- [ ] Add immutable `PromptBrowseScope`/`PromptBrowseResult` tests in `Tests/Library/test_library_prompts_state.py` for query normalization, local backend, collection ID, sort whitelist, page/page size, exact total/pages, scope fingerprint, loading/error/empty/no-match states, and page clamping.
- [ ] Add service contract tests for `browse_prompts(mode="local", query, collection_id, sort_by, sort_order, page, page_size)` and reject unsupported/mixed mode in this Library path.
- [ ] Keep existing bounded `search_prompts` behavior unchanged for Console `/prompt` and picker call sites.
- [ ] Run the focused red tests.

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Library/test_library_prompts_state.py Tests/Prompt_Management/test_prompt_scope_service.py -k "browse_prompt" -q
```

## Task 3: Implement Exact SQLite Browse and Counting

- [ ] Add DB tests with more than 100 active Prompt/Recipe rows for exact total and stable pagination.
- [ ] Cover search + collection + each allowed sort; ensure the collection join is applied before search/count/page and deleted rows never appear.
- [ ] Add injection-shaped sort inputs and prove they are rejected before SQL construction. Use a constant map from public sort values to fixed SQL fragments.
- [ ] Implement one local DB browse method returning rows plus exact total under the same normalized scope. Use parameterized values and deterministic ID tie-breakers.
- [ ] Assert query page size is bounded and page metadata remains correct after concurrent deletion.
- [ ] Run DB tests green.

## Task 4: Complete and Harden Collection Services

- [ ] Add tests with more than 200 collections for exact total, bounded catalog pages, search, and Load more behavior.
- [ ] Seed pre-existing case-fold collisions (`Sales`/`sales`) and assert chooser labels become `Sales · #id` while all mutations remain ID-based.
- [ ] Add concurrent create/rename tests proving case-insensitive duplicate checks occur in the same serialized local write transaction; rename excludes its own ID. Do not add a NOCASE unique migration.
- [ ] Add membership tests: one Prompt in multiple collections, one-collection-centric updates, inactive/foreign Prompt IDs, inactive collection IDs, and full rollback on any invalid member.
- [ ] Implement exact collection catalog results and local transaction methods on the live adapter in `prompt_scope_service.py`; keep existing server collection methods routed for existing callers but do not expose them in Library.
- [ ] Implement `replace_prompt_collection_membership`/equivalent scope methods with explicit outcomes separate from Prompt content Save.

## Task 5: Replace the Sampled Library Prompt State

- [ ] Add UI-state tests proving Library Prompt rows come from `PromptBrowseResult`, not `_local_source_records["prompts"]`; preserve the rail's lightweight total count seam.
- [ ] Remove `_library_prompts_filter`/sampled page derivation in favor of one `PromptBrowseScope` and result. Query, collection, sort, or page change must generate a new fingerprint.
- [ ] Run local DB work in an exclusive thread worker. Debounce search and use a monotonic request token so late success/error cannot overwrite a newer scope.
- [ ] Distinguish loading, empty library, empty collection, no matches, and service error with Retry. Do not turn failure into zero results.
- [ ] Restore focus to a valid row or toolbar after page/result changes.

## Task 6: Build Complete Collection and Membership UI

- [ ] Add failing canvas tests for `All prompts`, New collection, search, sort, page controls, exact `shown/total` copy, collection search, Load more, case-collision labels, and literal markup-looking names.
- [ ] Add collection manager tests for create/rename validation and explicit success/failure outcomes; do not add collection Delete.
- [ ] Add editor membership tests showing current memberships, shared manager, Apply membership, dirty content independence, and multi-membership.
- [ ] Implement toolbar/chooser/manager controls in `library_prompts_canvas.py` with stable IDs and keyboard order. Large catalogs must remain scrollable.
- [ ] Wire intents in `library_screen.py` through `PromptScopeService` only. Successful membership Apply refreshes membership/browse/count without claiming Prompt Save.
- [ ] Verify no Library source selector or server/All-sources claim appears.

## Task 7: Documentation and Focused Verification

- [ ] Update `Docs/User_Guide/library/prompts.md` with local-only collection behavior, complete pagination/search, collision labels, multi-membership, and separate Save/Apply outcomes.
- [ ] Run affected tests.

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Prompts_DB Tests/Prompt_Management/test_prompt_scope_service.py Tests/Library/test_library_prompts_state.py Tests/UI/test_library_prompts_canvas.py Tests/UI/test_library_shell.py -q
git diff --check
```

- [ ] Render/inspect empty, no-match, error, >100 Prompt, >200 collection, collision, membership, and narrow-terminal states. Check clipping, focus, stale result suppression, and truthful counts.
- [ ] Run the full suite, self-review, and request independent review.

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest
```

- [ ] Complete TASK-198 acceptance boxes and implementation notes, including ADR check, and mark Done only after verification.
- [ ] Open one ready PR against `dev`, resolve CI/review, merge, and verify the merge on `origin/dev`. Do not start TASK-199 implementation before that confirmation.
