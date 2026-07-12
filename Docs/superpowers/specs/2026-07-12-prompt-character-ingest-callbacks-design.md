# Wire prompt & character ingest-import completion callbacks (task 172)

**Status:** Design approved (brainstorm), pending spec review.
**Backlog:** task-172 — "Wire prompt/character ingest-import completion callbacks".
**Builds on:** T167, which fixed the identical gap for notes.

## Problem

The prompt and character "Import Selected Files Now" handlers define completion callbacks but never invoke them. Each runs its import worker via `app.run_worker(import_worker, name=..., group="file_operations", …)` with no `on_success`/`on_failure`, relying on the app's `on_worker_state_changed` to dispatch. But that path delegates to `worker_handler_registry.handle_event()`, which has no handler for the `file_operations` group (or these worker names) — it returns `handled=False` and only logs a `"No handler found"` warning. So the post-import **status-area summary, the completion toast, and the list/sidebar refreshes never fire** for prompts or characters.

T167 already fixed this exact gap for notes; this task applies the same fix to the two siblings.

- **Prompt** (`prompt_ingest_events.py`): `process_prompt_import_success(results, worker_name)` / `process_prompt_import_failure(error, worker_name)` are stashed on `app.prompt_import_success_handler` / `_failure_handler` (declared `= None` at `app.py:2367-2368`, assigned at `prompt_ingest_events.py:343-344`) and **never read**.
- **Character** (`character_ingest_events.py`): `on_import_success_char(results)` / `on_import_failure_char(error)` are local closures **never called**.

## Goal / Acceptance

- **AC1** — Prompt import completion invokes its success callback on success and its failure callback on a catastrophic worker failure.
- **AC2** — Character import completion invokes its success callback on success and its failure callback on a catastrophic worker failure.

## Chosen approach

Mirror T167's notes fix exactly: wrap each import worker in a small dispatch coroutine that runs the worker and directly invokes the success/failure callback, then hand that wrapper to `app.run_worker`. The workers are plain `async` coroutines (no `thread=True`), so they run on the main event loop — calling the UI-touching callbacks directly from the wrapper is safe, the same reasoning documented in the notes fix (`note_ingest_events.py:484-503`).

Considered and **declined**: extracting a shared `run_import_worker_with_dispatch()` helper across notes/prompt/character. T167 chose the inline pattern; the task is explicitly "wire them the same way notes was fixed"; and extraction would pull the already-shipped notes fix into this diff and add indirection for ~8-line wrappers that differ (prompt passes `worker_name`, notes/character do not). Kept inline. (A future consolidation task could revisit all three.)

## Components

### 1. Prompt dispatch wrapper (`prompt_ingest_events.py`)

Insert after the two callback definitions and before `app.run_worker`:

```python
async def _run_prompt_import_worker_and_dispatch():
    try:
        results = await import_worker_target()
    except Exception as e:
        process_prompt_import_failure(e, "prompt_import_worker")
        raise
    process_prompt_import_success(results, "prompt_import_worker")
    return results
```

Change `app.run_worker(import_worker_target, name="prompt_import_worker", …)` to pass `_run_prompt_import_worker_and_dispatch`. The wrapper passes the literal `"prompt_import_worker"`, trivially satisfying each callback's `if worker_name != "prompt_import_worker": return` guard.

Also **remove the now-dead handler plumbing** (approved): the `app.prompt_import_success_handler = …` / `app.prompt_import_failure_handler = …` assignments (`prompt_ingest_events.py:343-344`) and the `prompt_import_success_handler: Optional[Callable] = None` / `prompt_import_failure_handler: Optional[Callable] = None` class attributes (`app.py:2367-2368`). They only existed to feed the broken dispatch this fix replaces and are written-but-never-read afterward.

### 2. Character dispatch wrapper (`character_ingest_events.py`)

Insert after the two callback definitions and before `app.run_worker`:

```python
async def _run_char_import_worker_and_dispatch():
    try:
        results = await import_worker_char()
    except Exception as e:
        on_import_failure_char(e)
        raise
    on_import_success_char(results)
    return results
```

Change `app.run_worker(import_worker_char, name="character_import_worker", …)` to pass `_run_char_import_worker_and_dispatch`. Character's callbacks take a single argument (like notes).

## Data flow

```
"Import Now" button → handle_ingest_<x>_import_now_button_pressed(app, event)
  → app.run_worker(_run_<x>_import_worker_and_dispatch, group="file_operations")
      wrapper: await import_worker_<x>()
        success → on_import_success_<x>(results[, name])  → status-area summary + toast + list/sidebar refresh
        exception → on_import_failure_<x>(error[, name]) → status-area error + error toast → re-raise
  → Textual marks worker SUCCESS/ERROR → on_worker_state_changed → registry unhandled → one "No handler found" warning log (harmless, pre-existing, same as notes)
```

## Error handling

Semantics are unchanged from today except that the callbacks now fire:
- A **catastrophic** worker exception (prompt: `import_prompts_from_files` raising, re-raised by `import_worker_target`; character: a failure before/at the file loop) routes to `on_import_failure_<x>` and is then **re-raised** so Textual still records the worker error. The re-raise yields only the pre-existing harmless `"No handler found for worker … (Group: file_operations)"` warning log — **no second user-facing toast** (verified: `on_worker_state_changed` → `worker_handler_registry.handle_event` returns `handled=False` for this group; no ERROR-state notification for these worker names).
- **Per-file** failures are unchanged: the character worker records each as a `failure`/`conflict` result and the prompt worker records `status` per file; both surface through the *success* callback's summary. Character's `on_import_failure_char` therefore only fires on a worker-level exception, not per-file errors.

## Testing

One harness-free unit test module (`Tests/Event_Handlers/test_ingest_import_dispatch.py`), following the project's fake-`app` pattern (no Textual harness). Each test builds a fake `app` (a `SimpleNamespace`/`MagicMock`) exposing exactly what the trigger + callbacks touch, calls the async trigger, captures the coroutine handed to the fake `run_worker`, and awaits it:

- **Prompt success:** fake app with `selected_prompt_files_for_import=[Path("p.json")]`, `query_one`→stub `TextArea` (`.text`, `.load_text`, `.refresh`), `notify`/`call_later`/`run_worker` as mocks; patch `prompts_db_initialized`→`True` and `import_prompts_from_files`→a results list. After awaiting the captured coroutine, assert the stub TextArea's `load_text` was called with the summary (i.e. `process_prompt_import_success` fired).
- **Prompt failure:** same, but `import_prompts_from_files` raises. Assert awaiting the captured coroutine **re-raises**, and `app.notify` was called with an error severity (i.e. `process_prompt_import_failure` fired).
- **Character success:** fake app with `selected_character_files_for_import=[Path("c.png")]`, `notes_service` (mock whose `_get_db(...)` returns a fake db), `notes_user_id`, `query_one`→stub TextArea (`.clear`, `.load_text`, `.text`), `notify`/`call_later`/`run_worker` mocks, settable `_chat_character_filter_populated`; patch `ccl.import_and_save_character_from_file`→a char id (and `ccl.load_character_card_from_file`→a card dict or let its `try/except pass` swallow). Assert the stub TextArea's `load_text` got the summary.
- **Character failure:** set `selected_character_files_for_import` to a truthy-but-non-iterable object (`__bool__`→`True`, `__iter__`→raises) so the trigger's `if not …` guard passes but `import_worker_char` raises. Assert awaiting the captured coroutine **re-raises** and `on_import_failure_char`'s error toast fired.

**RED/GREEN:** today `run_worker` receives the raw `import_worker_<x>`, which returns/raises without ever calling the callback — so "the success callback fired" assertions FAIL before the fix and PASS after (the wrapper invokes them). The tests capture and await the coroutine passed to `run_worker`, so they exercise the exact object the fix changes.

## Scope / non-goals

- **No behavior change** beyond making the two dead callbacks fire and removing the dead prompt handler attrs. No new UI, no message copy changes, no new refresh behavior beyond what the existing callbacks already do.
- **Notes is untouched** (already fixed by T167); no shared-helper refactor.
- The pre-existing `"No handler found for worker"` warning log for `file_operations` workers is out of scope (it predates this task and also affects notes).
