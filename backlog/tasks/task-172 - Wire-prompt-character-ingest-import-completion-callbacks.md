---
id: TASK-172
title: Wire prompt/character ingest-import completion callbacks
status: Done
assignee: []
created_date: '2026-07-11 23:53'
labels:
  - follow-up
  - ingest
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
T167 found the notes-import success/failure callbacks were never dispatched (no worker handler matched the group), and fixed it for notes only. The sibling prompt (prompt_ingest_events.py) and character (character_ingest_events.py) import handlers have the same latent gap — their on_import_success/failure callbacks are defined but never invoked. Wire them the same way notes was fixed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Prompt import completion invokes its success/failure callback,Character import completion invokes its success/failure callback
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Applied the notes/T167 dispatch-wrapper pattern to both sibling import handlers.
Each import worker was passed directly to `app.run_worker(..., group="file_operations")`,
but the worker-state registry has no handler for that group, so the
`on_import_success`/`on_import_failure` callbacks were dead code (status-area summary,
toast, and list/sidebar refresh never fired).

Fix: wrap each worker in a thin `async` dispatch coroutine that awaits the worker,
calls the failure callback + re-raises on exception, and calls the success callback on
success — then hand that wrapper to `run_worker` in place of the raw worker (keeping the
same name/group/description). Safe because these workers are plain coroutines on the main
event loop (no `thread=True`). Prompt callbacks take `(results, worker_name)`; character
callbacks take a single arg.

- Task 1 (prompt): `prompt_ingest_events.py` — added `_run_prompt_import_worker_and_dispatch`;
  removed the dead `app.prompt_import_*_handler` assignments; deleted the dead
  `prompt_import_success_handler`/`prompt_import_failure_handler` class attrs in `app.py`.
- Task 2 (character): `character_ingest_events.py` — added `_run_char_import_worker_and_dispatch`;
  deleted the dead `character_import_success_handler`/`character_import_failure_handler` class
  attrs in `app.py` (these were declared-only, never set or read — character uses local closures).

Tests are harness-free (`Tests/Event_Handlers/test_prompt_ingest_events.py`,
`Tests/Event_Handlers/test_character_ingest_events.py`): a `_make_mock_app` whose
`run_worker` side-effect captures the callable, then the test awaits the async trigger,
awaits the captured worker, and asserts the dispatched effect (summary written / error toast /
re-raise). Notes untouched; no new UI or copy changes.

Modified/added files: `tldw_chatbook/Event_Handlers/prompt_ingest_events.py`,
`tldw_chatbook/Event_Handlers/character_ingest_events.py`, `tldw_chatbook/app.py`,
`Tests/Event_Handlers/test_prompt_ingest_events.py`,
`Tests/Event_Handlers/test_character_ingest_events.py`.
<!-- SECTION:NOTES:END -->
