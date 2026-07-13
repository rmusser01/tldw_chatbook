# Wire prompt & character ingest-import completion callbacks — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax. Spec: `Docs/superpowers/specs/2026-07-12-prompt-character-ingest-callbacks-design.md`. Branch `claude/followups-ingest-callbacks` off dev `993653d8`. Line numbers are exact at the branch point; grep symbols if they drift.

**Goal:** Make the prompt and character "Import Now" completion callbacks actually fire (status-area summary + toast + list/sidebar refresh), by wrapping each import worker in a dispatch coroutine — exactly as T167 already did for notes.

**Architecture:** Each import worker runs via `app.run_worker(..., group="file_operations")`, but the worker-state registry has no handler for that group, so the callbacks are dead. Replace the raw worker passed to `run_worker` with a thin `async` wrapper that runs the worker and directly invokes `on_import_success`/`on_import_failure` (safe: these are plain coroutines on the main event loop). Mirrors `note_ingest_events.py:484-510`.

**Tech Stack:** Python ≥3.11, Textual, pytest + pytest-asyncio, `unittest.mock`.

## Global Constraints

- **Mirror the notes fix (`note_ingest_events.py:484-510`) exactly** — a `try: results = await import_worker(); except Exception as e: on_failure(e); raise` then `on_success(results); return results` wrapper, handed to `run_worker` in place of the raw worker. Keep each existing `run_worker` `name`/`group`/`description`.
- **No behavior change** beyond making the two dead callbacks fire and removing the dead prompt handler attrs. No new UI, no copy changes, notes untouched, no shared-helper refactor.
- **Prompt callbacks take `(results, worker_name)` / `(error, worker_name)`** — the wrapper passes the literal `"prompt_import_worker"` (satisfies their `if worker_name != "prompt_import_worker": return` guard). **Character callbacks take a single arg** (`results` / `error`).
- **Tests mirror `Tests/Event_Handlers/test_note_ingest_events.py`**: a `_make_mock_app` helper whose `run_worker` side-effect **captures** the callable into `app._captured_worker["callable"]`; the test `await`s the async trigger, then `await worker_callable()`, then asserts the dispatched effect. `@pytest.mark.asyncio`, event = `Button.Pressed(Mock(spec=Button))`.
- **Test command** (venv, isolated HOME):
  ```
  HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
    /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <files> \
    -q -p no:cacheprovider -o addopts="" --timeout=120
  ```
- Every commit ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Prompt import dispatch wrapper

**Files:**
- Create: `Tests/Event_Handlers/test_prompt_ingest_events.py`
- Modify: `tldw_chatbook/Event_Handlers/prompt_ingest_events.py` (add wrapper, swap the `run_worker` first arg, remove the dead `app.prompt_import_*_handler` assignments at ~:340-344)
- Modify: `tldw_chatbook/app.py` (remove the dead class attrs at :2367-2368)

**Interfaces:**
- Trigger under test: `handle_ingest_prompts_import_now_button_pressed(app, event)` (async), `prompt_ingest_events.py:220`.
- Module-level patch points in `prompt_ingest_events.py`: `prompts_db_initialized` (imported alias, :19), `import_prompts_from_files` (imported, :20).
- Widget the callbacks touch: `#prompt-import-status-area` (a `TextArea`); success calls `.load_text(summary)`, the trigger only assigns `.text` (so `load_text` is a clean success signal). Failure reads `.text` then `.load_text(...)` and calls `app.notify(..., severity="error", ...)`.

- [ ] **Step 1: Write the failing tests**

Create `Tests/Event_Handlers/test_prompt_ingest_events.py`:
```python
# Tests/Event_Handlers/test_prompt_ingest_events.py
"""Task 172: the prompt "Import Now" completion callbacks were dead code
(nothing dispatched the file_operations worker group). These assert that a
successful import invokes the success callback and a catastrophic worker
failure invokes the failure callback and re-raises."""
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from textual.css.query import QueryError
from textual.widgets import Button, TextArea

from tldw_chatbook.Event_Handlers.prompt_ingest_events import (
    handle_ingest_prompts_import_now_button_pressed,
)


def _make_mock_app() -> Mock:
    app = Mock()
    app.selected_prompt_files_for_import = [Path("p.json")]
    app.notify = Mock()
    app.call_later = Mock()

    status_area = Mock(spec=TextArea, text="")
    widgets = {"#prompt-import-status-area": status_area}

    def query_one_side_effect(selector, widget_type=None):
        try:
            return widgets[selector]
        except KeyError:
            raise QueryError(f"{selector} not found")

    app.query_one = Mock(side_effect=query_one_side_effect)

    captured = {}
    def run_worker_side_effect(worker_callable, **kwargs):
        captured["callable"] = worker_callable
        return Mock()
    app.run_worker = Mock(side_effect=run_worker_side_effect)
    app._captured_worker = captured
    app._status_area = status_area
    return app


@pytest.mark.asyncio
async def test_successful_prompt_import_invokes_success_callback():
    app = _make_mock_app()
    results = [{"status": "success", "file_path": "p.json", "prompt_name": "P", "message": "ok"}]
    with patch("tldw_chatbook.Event_Handlers.prompt_ingest_events.prompts_db_initialized", return_value=True), \
         patch("tldw_chatbook.Event_Handlers.prompt_ingest_events.import_prompts_from_files", return_value=results):
        await handle_ingest_prompts_import_now_button_pressed(app, Button.Pressed(Mock(spec=Button)))
        await app._captured_worker["callable"]()

    # The success callback is the ONLY code path that calls load_text on the
    # status area (the trigger only assigns `.text`).
    app._status_area.load_text.assert_called_once()
    assert "Summary:" in app._status_area.load_text.call_args.args[0]


@pytest.mark.asyncio
async def test_failed_prompt_import_invokes_failure_callback_and_reraises():
    app = _make_mock_app()
    with patch("tldw_chatbook.Event_Handlers.prompt_ingest_events.prompts_db_initialized", return_value=True), \
         patch("tldw_chatbook.Event_Handlers.prompt_ingest_events.import_prompts_from_files",
               side_effect=RuntimeError("boom")):
        await handle_ingest_prompts_import_now_button_pressed(app, Button.Pressed(Mock(spec=Button)))
        with pytest.raises(RuntimeError):
            await app._captured_worker["callable"]()

    # failure callback surfaced an error-severity toast
    assert any(c.kwargs.get("severity") == "error" for c in app.notify.call_args_list)
```

- [ ] **Step 2: Run to verify they fail**

Run:
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Event_Handlers/test_prompt_ingest_events.py -q -p no:cacheprovider -o addopts="" --timeout=120
```
Expected: FAIL. Today `run_worker` receives the raw `import_worker_target`; awaiting it returns/raises without ever calling the callbacks — so `load_text` is never called (success test) and no error-severity notify occurs (failure test).

- [ ] **Step 3: Add the dispatch wrapper + swap the run_worker arg**

In `prompt_ingest_events.py`, replace the dead-handler block and the `run_worker` call (currently `:340-353`):
```python
    # Store these handlers on the app instance temporarily or pass them via a different mechanism
    # For simplicity here, we'll assume app.py's on_worker_state_changed will call them.
    # A more robust way is to make these methods of a class or use a dispatch dictionary in app.py.
    app.prompt_import_success_handler = process_prompt_import_success
    app.prompt_import_failure_handler = process_prompt_import_failure

    # Run the worker
    app.run_worker(
        import_worker_target,  # The async callable
        name="prompt_import_worker",  # Crucial for identifying the worker later
        group="file_operations",
        description="Importing selected prompt files."
        # No on_success or on_failure here
    )
```
with:
```python
    async def _run_prompt_import_worker_and_dispatch():
        # Task 172: the file_operations worker group has no worker-state
        # handler, so process_prompt_import_success/_failure were never
        # invoked. This worker is a plain coroutine (no thread=True), so it
        # runs on the main event loop -- dispatching the callbacks directly
        # here (as T167 did for notes) is safe.
        try:
            results = await import_worker_target()
        except Exception as e:
            process_prompt_import_failure(e, "prompt_import_worker")
            raise
        process_prompt_import_success(results, "prompt_import_worker")
        return results

    app.run_worker(
        _run_prompt_import_worker_and_dispatch,
        name="prompt_import_worker",  # Crucial for identifying the worker later
        group="file_operations",
        description="Importing selected prompt files."
    )
```

- [ ] **Step 4: Remove the now-dead class attrs in `app.py`**

Delete `app.py:2367-2368`:
```python
    prompt_import_success_handler: Optional[Callable] = None
    prompt_import_failure_handler: Optional[Callable] = None
```
(These are written only at the block just removed and read nowhere. Confirm no other reader: `grep -rn "prompt_import_success_handler\|prompt_import_failure_handler" tldw_chatbook/` should return no matches after this task.)

- [ ] **Step 5: Run to verify they pass + app import smoke**

Run:
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Event_Handlers/test_prompt_ingest_events.py -q -p no:cacheprovider -o addopts="" --timeout=120
HOME=/private/tmp/tldw-chatbook-test-home PYTHONPATH=$(pwd) \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c "import tldw_chatbook.app; print('import ok')"
```
Expected: both tests PASS + `import ok` (proves the `app.py` attr removal didn't break the module).

- [ ] **Step 6: Commit**

```bash
git add Tests/Event_Handlers/test_prompt_ingest_events.py tldw_chatbook/Event_Handlers/prompt_ingest_events.py tldw_chatbook/app.py
git commit -m "fix(ingest): dispatch prompt import completion callbacks; drop dead handler attrs (172)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Character import dispatch wrapper

**Files:**
- Create: `Tests/Event_Handlers/test_character_ingest_events.py`
- Modify: `tldw_chatbook/Event_Handlers/character_ingest_events.py` (add wrapper, swap the `run_worker` first arg at ~:368)
- Modify: `backlog/tasks/task-172 - Wire-prompt-character-ingest-import-completion-callbacks.md`

**Interfaces:**
- Trigger under test: `handle_ingest_characters_import_now_button_pressed(app, event)` (async), `character_ingest_events.py:232`.
- Module-level patch points in `character_ingest_events.py`: `ccl.import_and_save_character_from_file`, `ccl.load_character_card_from_file` (`ccl` = `Character_Chat_Lib`, imported :16).
- The trigger needs `app.notes_service._get_db(app.notes_user_id)`, and the success callback sets `app._chat_character_filter_populated` + `app.call_later(...)`. Widget: `#ingest-character-import-status-area` (a `TextArea`); the trigger calls `.clear()` + `.load_text("Starting…")`, so success is identified by the LAST `load_text` call carrying the summary. Character's callbacks take one arg.

- [ ] **Step 1: Write the failing tests**

Create `Tests/Event_Handlers/test_character_ingest_events.py`:
```python
# Tests/Event_Handlers/test_character_ingest_events.py
"""Task 172: the character "Import Now" completion callbacks were dead code
(nothing dispatched the file_operations worker group). These assert that a
successful import invokes the success callback and a catastrophic worker
failure invokes the failure callback and re-raises."""
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from textual.css.query import QueryError
from textual.widgets import Button, TextArea

from tldw_chatbook.Event_Handlers.character_ingest_events import (
    handle_ingest_characters_import_now_button_pressed,
)


class _BoomList:
    """Truthy (passes the trigger's `if not …` guard) but raises on iteration,
    so `import_worker_char` fails at the loop -- the only way to exercise the
    catastrophic-failure path, since the worker swallows per-file errors."""
    def __bool__(self):
        return True
    def __iter__(self):
        raise RuntimeError("boom")


def _make_mock_app(*, selected) -> Mock:
    app = Mock()
    app.selected_character_files_for_import = selected
    app.notes_user_id = "user-1"
    app.notes_service = Mock()
    app.notes_service._get_db = Mock(return_value=Mock())
    app.notify = Mock()
    app.call_later = Mock()
    app._chat_character_filter_populated = False

    status_area = Mock(spec=TextArea, text="")
    widgets = {"#ingest-character-import-status-area": status_area}

    def query_one_side_effect(selector, widget_type=None):
        try:
            return widgets[selector]
        except KeyError:
            raise QueryError(f"{selector} not found")

    app.query_one = Mock(side_effect=query_one_side_effect)

    captured = {}
    def run_worker_side_effect(worker_callable, **kwargs):
        captured["callable"] = worker_callable
        return Mock()
    app.run_worker = Mock(side_effect=run_worker_side_effect)
    app._captured_worker = captured
    app._status_area = status_area
    return app


@pytest.mark.asyncio
async def test_successful_character_import_invokes_success_callback():
    app = _make_mock_app(selected=[Path("c.png")])
    with patch("tldw_chatbook.Event_Handlers.character_ingest_events.ccl.import_and_save_character_from_file",
               return_value=123), \
         patch("tldw_chatbook.Event_Handlers.character_ingest_events.ccl.load_character_card_from_file",
               return_value={"name": "TestChar"}):
        await handle_ingest_characters_import_now_button_pressed(app, Button.Pressed(Mock(spec=Button)))
        await app._captured_worker["callable"]()

    # success callback writes the summary (the LAST load_text call) and sets the flag
    assert "Summary:" in app._status_area.load_text.call_args.args[0]
    assert app._chat_character_filter_populated is True


@pytest.mark.asyncio
async def test_failed_character_import_invokes_failure_callback_and_reraises():
    app = _make_mock_app(selected=_BoomList())
    await handle_ingest_characters_import_now_button_pressed(app, Button.Pressed(Mock(spec=Button)))
    with pytest.raises(RuntimeError):
        await app._captured_worker["callable"]()

    assert any(c.kwargs.get("severity") == "error" for c in app.notify.call_args_list)
```

- [ ] **Step 2: Run to verify they fail**

Run:
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Event_Handlers/test_character_ingest_events.py -q -p no:cacheprovider -o addopts="" --timeout=120
```
Expected: FAIL. Today `run_worker` gets the raw `import_worker_char`; awaiting it never calls `on_import_success_char`/`on_import_failure_char`, so the summary is never written / the flag stays `False` / no error toast.

- [ ] **Step 3: Add the dispatch wrapper + swap the run_worker arg**

In `character_ingest_events.py`, replace the `run_worker` call (currently `:368-373`):
```python
    app.run_worker(
        import_worker_char,
        name="character_import_worker",
        group="file_operations",
        description="Importing selected character files."
    )
```
with the wrapper defined first, then the swapped call:
```python
    async def _run_char_import_worker_and_dispatch():
        # Task 172: the file_operations worker group has no worker-state
        # handler, so on_import_success_char/on_import_failure_char were never
        # invoked. This worker is a plain coroutine (no thread=True), so it
        # runs on the main event loop -- dispatching the callbacks directly
        # here (as T167 did for notes) is safe.
        try:
            results = await import_worker_char()
        except Exception as e:
            on_import_failure_char(e)
            raise
        on_import_success_char(results)
        return results

    app.run_worker(
        _run_char_import_worker_and_dispatch,
        name="character_import_worker",
        group="file_operations",
        description="Importing selected character files."
    )
```

- [ ] **Step 4: Run to verify they pass**

Run the Task-2 Step-1 command. Expected: both tests PASS.

- [ ] **Step 5: Mark backlog task 172 Done**

```bash
perl -0pi -e 's/- \[ \] #1/- [x] #1/' "backlog/tasks/task-172 - Wire-prompt-character-ingest-import-completion-callbacks.md"
perl -0pi -e 's/^status: .*/status: Done/mi' "backlog/tasks/task-172 - Wire-prompt-character-ingest-import-completion-callbacks.md"
```
Then add a short `## Implementation Notes` section to that file: the notes/T167 wrapper pattern applied to prompt + character; dead prompt handler attrs removed; harness-free capture-the-worker tests. Confirm the status/checkbox actually changed (`grep -n "status:\|\[x\]" "backlog/tasks/task-172 "*.md`); if the frontmatter key differs, edit it to `Done` by hand.

- [ ] **Step 6: Commit**

```bash
git add Tests/Event_Handlers/test_character_ingest_events.py tldw_chatbook/Event_Handlers/character_ingest_events.py "backlog/tasks/task-172 - Wire-prompt-character-ingest-import-completion-callbacks.md"
git commit -m "fix(ingest): dispatch character import completion callbacks; task 172 done (172)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Final gate (after Task 2)

```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Event_Handlers/test_prompt_ingest_events.py Tests/Event_Handlers/test_character_ingest_events.py \
  Tests/Event_Handlers/test_note_ingest_events.py -q -p no:cacheprovider -o addopts="" --timeout=180
```
Plus `python -c "import tldw_chatbook.app"`. Then the whole-branch review and finishing-a-development-branch. (Notes test included to confirm the shared pattern still holds and nothing regressed.)
