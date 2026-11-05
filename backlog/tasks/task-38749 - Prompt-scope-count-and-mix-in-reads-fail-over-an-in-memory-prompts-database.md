---
id: TASK-38749
title: "Prompt scope count and mix-in reads fail over an in-memory prompts database"
status: Done
assignee:
- '@claude'
created_date: 2026-11-04 06:34
updated_date: '2026-11-05 20:56'
labels:
- follow-up
- prompts
- bug
priority: low
references:
- tldw_chatbook/Prompt_Management/prompt_scope_service.py
- tldw_chatbook/DB/Prompts_DB.py
- Tests/Prompt_Management/test_prompt_scope_service.py
dependencies:
- TASK-38686
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the review of PR #3402 (TASK-38686). Checked on dev bb80905eb3, 2026-11-04.

`PromptScopeService.count_prompts` (`Prompt_Management/prompt_scope_service.py:1773`) and `read_mixin_records` (:1952) run their local read in a worker thread, through `asyncio.to_thread(run_finite_local_worker, ...)` (:1808 and :1977). `PromptsDatabase` keeps one connection per thread (`threading.local`). For `:memory:`, a new connection is a new, empty database.

A probe on this base, run through the scratch-profile wrapper, built a `:memory:` `PromptsDatabase`, added a prompt inline, and called both reads through a `PromptScopeService` over it:
- `count_prompts` raised `DatabaseError: Failed to list prompts: no such table: Prompts`;
- `read_mixin_records` raised `OperationalError: no such table: Prompts`.

#3402 hit the same problem in `save_prompt` once its write moved off the loop, and now runs a `:memory:` save inline (`prompt_db.is_memory_db`). These two reads have no such check. The service already expects memory-backed mix-in reads: it keeps separate mix-in authority keys for memory databases (`_mixin_memory_authorities`, :1941).

Priority is low because production prompt databases are file-backed. The failure reaches tests, and any in-memory fallback.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Over a `:memory:` prompts database, `count_prompts` returns the real count and `read_mixin_records` returns the stored mix-ins
- [x] #2 Over a file-backed database, both reads still run off the event loop
- [x] #3 A test over a `:memory:` database covers both reads and is red before the change
<!-- AC:END -->

## Implementation Plan

1. Add one private `PromptScopeService` helper that runs a finite local Prompts call: inline over a `:memory:` database (a worker thread's own connection would be a fresh, empty database), otherwise in a worker thread through `run_finite_local_worker` (or the caller's native runner).
2. Route `count_prompts`, `read_mixin_records` and `save_prompt`'s local write through it, so the memory check lives in one place.
3. Add a `:memory:` test covering both reads next to TASK-38686's memory save test; show it red at base, then green. Keep the existing off-loop tests green (file-backed reads still use `asyncio.to_thread`).

## Implementation Notes

- `PromptScopeService._run_local` is now the one place that decides where a finite local Prompts call runs: inline over a `:memory:` database (a worker thread's own connection to it is a fresh, empty database), otherwise in a worker thread via `run_finite_local_worker` (or the Console's native preparation runner, for `read_mixin_records`). `count_prompts`, `read_mixin_records` and `save_prompt`'s local write all route through it, so TASK-38686's inline-memory save check moved into the helper instead of being copied twice more.
- Tests (`Tests/Prompt_Management/test_prompt_scope_service.py`): `test_local_reads_over_an_in_memory_database_see_its_rows` (AC#1, AC#3) and `test_file_backed_local_reads_stay_off_the_event_loop` (AC#2: both local service calls record a thread ident other than the loop's).
- Red at base: the memory test failed on `count_prompts` (`DatabaseError: ... no such table: Prompts`); with the count lines removed temporarily it failed on `read_mixin_records` (`OperationalError: no such table: Prompts`). Green after the change; the file-backed test passes at base and after (the invariant dev already keeps).
- Covering runs (`-n 4`, scratch wrapper): scope service, mix-in, local service, Console mix-in admission/forks/journey/retry, prompt draft save, Library prompts seam: 442 passed. Console/Library prompt UI files: 230 passed, 8 failed; the same 8 fail with the base `prompt_scope_service.py` (pre-existing, not this change).
- Files: `tldw_chatbook/Prompt_Management/prompt_scope_service.py`, `Tests/Prompt_Management/test_prompt_scope_service.py`. The test file runs on no PR lane (as with TASK-38686's memory save test beside it).
