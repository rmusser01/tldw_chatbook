---
id: TASK-38749
title: "Prompt scope count and mix-in reads fail over an in-memory prompts database"
status: To Do
assignee: []
created_date: 2026-11-04 06:34
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
- [ ] #1 Over a `:memory:` prompts database, `count_prompts` returns the real count and `read_mixin_records` returns the stored mix-ins
- [ ] #2 Over a file-backed database, both reads still run off the event loop
- [ ] #3 A test over a `:memory:` database covers both reads and is red before the change
<!-- AC:END -->
