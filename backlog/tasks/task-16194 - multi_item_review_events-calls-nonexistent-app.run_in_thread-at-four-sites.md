---
id: TASK-16194
title: 'multi_item_review_events calls nonexistent app.run_in_thread at four sites'
status: Done
assignee: []
created_date: '2026-08-14 03:05'
labels:
  - bug
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Event_Handlers/multi_item_review_events.py:172,256,262,296` call `app.run_in_thread(...)`, which does not exist in Textual 8.2.8 and is defined nowhere in the repo — every one of those code paths dies in AttributeError when reached. TASK-15471 found and fixed the identical bug in `collections_tag_events.py` (rename/merge/delete were all dead on dev) and verified this residue is real and untouched. Fix with the same pattern: `asyncio.to_thread` + the memory-db guard, and add tests that actually drive the four paths (the collections fix's test file is the reference — its born-red evidence was `AttributeError: '_FakeApp' object has no attribute 'run_in_thread'`). Surfaced during TASK-15471 (per-click I/O off-loop, PR #1625 merged `172ada448`) and its concurrency review; evidence in the session review record.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All four sites execute their threaded work successfully (no AttributeError)
- [x] #2 Tests drive each repaired path, born-red against the current dev behavior
- [x] #3 A grep confirms no other run_in_thread call sites remain in the repo
<!-- AC:END -->

## Implementation Plan

1. Read TASK-15471's fix in `collections_tag_events.py` (commit `b643690a7`) and its test file `Tests/Event_Handlers/test_collections_tag_events.py` as the reference pattern: a local `_media_db_off_loop(app, func, *args)` helper using `asyncio.to_thread`, guarded by `getattr(app.media_db, "is_memory_db", False)` so a per-connection `:memory:` DB stays on the loop thread.
2. Read all four call sites in `multi_item_review_events.py` and classify what each threaded callable touches:
   - line 172 (`generate_single_analysis`): `app.llm_api_client.chat_with_model` — an LLM call, not `media_db`. No memory-db guard needed; plain `asyncio.to_thread`.
   - lines 256 + 262 (`save_analysis_to_db`): `app.media_db.execute_query(...)` followed by a separate `app.media_db.commit` — the second call is *also* broken (`MediaDatabase` has no `commit` method), so even patching only `run_in_thread` would leave the UPDATE uncommitted. Fold into one `execute_query(..., commit=True)` call (the DB's own supported commit path, same kwarg `Prompts_DB.execute_query_with_retry` uses).
   - line 296 (`load_existing_analyses`): `app.media_db.execute_query(...)` — plain DB read, needs the guard.
3. Add a local `_media_db_off_loop` helper (mirroring the reference, extended with `**kwargs` since `execute_query`'s `commit` is keyword-only) and rewrite the three `media_db`-touching sites through it; rewrite the LLM site with a bare `asyncio.to_thread`.
4. Write `Tests/Event_Handlers/test_multi_item_review_events.py` with duck-typed fakes (`_FakeMediaDB`, `_FakeLLMClient`, `_FakeApp` with no `run_in_thread`), one test per repaired path plus a memory-db-guard regression test.
5. Prove born-red: temporarily revert the source file to its pre-fix content, run the new tests, confirm they fail on `AttributeError: '_FakeApp' object has no attribute 'run_in_thread'` (or the same error caught and surfaced as a string/False/`{}` by the function's own `except`), then restore the fix.
6. Run the new tests green, run the full `Tests/Event_Handlers/` suite for regressions, grep the repo for any remaining `run_in_thread` call sites, and run ruff check/format on the touched files.

## Implementation Notes

Mechanical replication of TASK-15471's fix, applied to `Event_Handlers/multi_item_review_events.py`. Added a local `_media_db_off_loop(app, func, /, *args, **kwargs)` helper (same shape as `collections_tag_events._media_db_off_loop`, extended with `**kwargs` because `execute_query`'s `commit` param is keyword-only) and rewired all three `media_db`-touching call sites through it; the one non-DB site (the LLM call) got a bare `asyncio.to_thread` since it never touches `media_db` and the memory-db guard has nothing to check there.

Investigating "what each threaded callable touches" (per the task's instruction) surfaced a second, independent bug at the `save_analysis_to_db` site: the pre-existing code called `app.media_db.execute_query(...)` (no `commit=True`) and then separately awaited `app.media_db.commit` — but `MediaDatabase` (`DB/Client_Media_DB_v2.py`) has no `commit` method at all. So that site was doubly broken: `run_in_thread` doesn't exist, and even a naive patch to `asyncio.to_thread` would have hit a *second* `AttributeError` on `.commit`, or — had that second call silently been a no-op — left the `UPDATE` uncommitted forever (`execute_query`'s own `commit` kwarg defaults to `False`, and this DB does not use `isolation_level=None`/autocommit). Fixed by folding the two calls into one `execute_query(..., commit=True)` — the DB's own supported commit path, the same kwarg `Prompts_DB.execute_query_with_retry` already uses elsewhere in the repo.

**Per-site table:**

| Site (pre-fix line) | What it touches | Fix |
|---|---|---|
| `generate_single_analysis`, L172 | `app.llm_api_client.chat_with_model` — LLM call, not DB | Bare `asyncio.to_thread(...)`, no memory-db guard (nothing DB-related to guard) |
| `save_analysis_to_db`, L256 | `app.media_db.execute_query(query, params)` | `_media_db_off_loop(app, app.media_db.execute_query, query, params, commit=True)` |
| `save_analysis_to_db`, L262 | `app.media_db.commit` — **method does not exist on `MediaDatabase`** | Removed; folded into the L256 call via `commit=True` |
| `load_existing_analyses`, L296 | `app.media_db.execute_query(query, media_ids)` | `_media_db_off_loop(app, app.media_db.execute_query, query, media_ids)` |

**Born-red evidence** (`Tests/Event_Handlers/test_multi_item_review_events.py`, run against the pre-fix file content restored via Edit/Write, then reverted the same way — no git checkout/reset used):
- `test_generate_single_analysis_calls_llm_without_run_in_thread` — `AssertionError: assert 'The generated analysis body.' in "Error generating analysis: '_FakeApp' object has no attribute 'run_in_thread'"` (caught by the function's own `except`, surfaced as a string).
- `test_save_analysis_to_db_persists_in_one_committed_call` — `assert False is True` (caught, surfaced as `False`); stderr: `Error saving analysis to DB: '_FakeApp' object has no attribute 'run_in_thread'`.
- `test_load_existing_analyses_returns_rows_without_run_in_thread` — `assert {} == {1: 'existing analysis', 2: None}` (caught, surfaced as `{}`); stderr: `Error loading existing analyses: '_FakeApp' object has no attribute 'run_in_thread'`.
- `test_save_analysis_to_db_off_loop_guard_still_works_for_memory_db` — same as the second case (this test additionally pins the `is_memory_db=True` synchronous-call branch of the new helper).

All four went green after restoring the fix. Full `Tests/Event_Handlers/` suite: 58 passed, 1 skipped (pre-existing, unrelated skip), no regressions.

**AC#3 grep** — `grep -rn "\.run_in_thread(" --include="*.py" .` returns zero matches (exit 1) repo-wide; `grep -rn "def run_in_thread"` also returns zero (confirms it never existed as a defined method). The only remaining textual occurrences of `run_in_thread` are in comments/docstrings/test names documenting the historical bug (this file, `collections_tag_events.py`, both test files).

**Files modified:**
- `tldw_chatbook/Event_Handlers/multi_item_review_events.py` — added `_media_db_off_loop`; fixed all four sites.
- `Tests/Event_Handlers/test_multi_item_review_events.py` — new, 4 tests.

**Lint:** `ruff check` and `ruff format --check` clean on both touched files.
