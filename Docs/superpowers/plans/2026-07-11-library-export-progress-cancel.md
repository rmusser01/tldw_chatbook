# Library chatbook export — progress + cancel — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax. Spec: `Docs/superpowers/specs/2026-07-11-library-export-progress-cancel-design.md`. Branch `claude/followups-export-progress` off origin/dev `bc8636b2`. Anchors are exact at the branch point; grep symbols, line numbers drift.

**Goal:** Give the Library chatbook export a live per-phase item-count status line and a working Cancel control, by adding progress + cancel hooks to `ChatbookCreator` and wiring them into the Library export canvas.

**Architecture:** `ChatbookCreator.create_chatbook` gains optional `progress_callback`/`cancel_check` (default `None` → byte-for-byte current behavior), carried as instance state so the `_collect_*`/`_discover_relationships`/`_create_zip_archive` helpers can emit progress and check cancel without new signatures. The zip is written to a sibling `.partial` and `os.replace`d onto the destination only on success (atomic finalize). `LocalChatbookService.export_chatbook` forwards the hooks and surfaces a `cancelled` flag. The Library screen's threaded export worker passes a throttled progress callback (marshalled to the UI thread, `run_id`-staleness-guarded) and a `threading.Event`-backed cancel check; a Cancel button sets the event.

**Tech Stack:** Python ≥3.11, Textual, pytest. No new third-party deps. No new widget/progress-bar (text status line only).

## Global Constraints

- **No behavior change when hooks are absent:** `create_chatbook(...)` with `progress_callback=None, cancel_check=None` must behave exactly as today (existing tests still pass unchanged).
- **Atomic finalize:** zip to `output_path.with_name(output_path.name + ".partial")` (same directory as the destination → `os.replace` is atomic across volumes); the destination file is never created/clobbered until success.
- **Cleanup:** the temp `work_dir` AND the `.partial` file are removed on cancel, on failure, and on success.
- **Cancel ≠ run_id bump:** the Cancel button sets the `threading.Event` but does NOT bump `_library_export_run_id`; navigate-away/reset sets the event AND bumps `run_id` (existing behavior).
- **Every UI-thread apply handler** (`_apply_library_export_progress`, `_apply_library_export_cancelled`) checks `run_id != self._library_export_run_id` before mutating state/DOM — mirrors `_apply_library_export_failure`.
- **Marshalling** off the worker thread wraps `self.app.call_from_thread(...)` in `try/except Exception` (Textual `NoApp` is not a `RuntimeError`) — mirrors `_marshal_library_export_failure`.
- **Progress callback must never raise into export logic:** the creator's `_emit_progress` swallows callback exceptions (debug-log); the UI marshaller swallows too.
- **Throttle:** emit when the phase changed, OR `current >= total`, OR `≥ 0.1s` since the last emit — always flush on phase-change and the final tick.
- **Cancel granularity:** effective at the next item/file checkpoint; a single in-progress file copy / `zf.write` is not mid-interrupted (documented, acceptable).
- **Only the Library export canvas** gets UI wiring; the server-mode export path is untouched.
- **Staging:** explicit paths only (never `git add -A`). Every commit ends with the trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- **Test command** (venv, isolated HOME):
  ```
  HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
    /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <files> \
    -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
  ```

---

### Task 1: Creator progress + cancel hooks + atomic finalize

**Files:**
- Modify: `tldw_chatbook/Chatbooks/chatbook_creator.py`
- Test: `Tests/Chatbooks/test_chatbook_creator.py`

**Interfaces:**
- Produces:
  - `ExportProgress` — `@dataclass(frozen=True)` with `phase: str`, `current: int`, `total: int`.
  - `ChatbookExportCancelled(Exception)`.
  - `ChatbookCreator.create_chatbook(..., progress_callback: Optional[Callable[[ExportProgress], None]] = None, cancel_check: Optional[Callable[[], bool]] = None)` — returns the same `(success, message, dependency_info)` tuple; on cancel returns `(False, "Export cancelled", {"cancelled": True, "missing_dependencies": [], "auto_included": []})`.

**Phase names** (used verbatim by later tasks): `conversations`, `notes`, `characters`, `media`, `prompts`, `relationships`, `packaging`.

- [ ] **Step 1: Write the failing tests**

Add to `Tests/Chatbooks/test_chatbook_creator.py` (reuse the existing `chatbook_creator`, `temp_db_paths`, `tmp_path` fixtures and the `@patch` DB mocks that `test_create_chatbook_minimal` uses — copy its decorator stack and empty-selection shape):

```python
def test_create_chatbook_reports_packaging_progress(self, mock_prompts_db, mock_chacha_db, chatbook_creator, tmp_path):
    """progress_callback fires ExportProgress events; packaging counts are monotonic to total."""
    from tldw_chatbook.Chatbooks.chatbook_creator import ExportProgress
    output_path = tmp_path / "cb.zip"
    events = []
    success, _msg, _dep = chatbook_creator.create_chatbook(
        name="P", description="", content_selections={
            ContentType.CONVERSATION: [], ContentType.NOTE: [], ContentType.CHARACTER: [],
        }, output_path=output_path, progress_callback=events.append,
    )
    assert success is True
    packaging = [e for e in events if e.phase == "packaging"]
    assert packaging, "expected at least one packaging progress event"
    assert all(isinstance(e, ExportProgress) for e in packaging)
    currents = [e.current for e in packaging]
    assert currents == sorted(currents) and packaging[-1].current == packaging[-1].total

def test_create_chatbook_cancel_during_packaging_leaves_no_output(self, mock_prompts_db, mock_chacha_db, chatbook_creator, tmp_path):
    """cancel_check True during packaging → cancelled result, no destination file, temp cleaned."""
    output_path = tmp_path / "cb.zip"
    calls = {"n": 0}
    def cancel_after_first_package():
        # allow collection to pass; trip on the first packaging checkpoint
        calls["n"] += 1
        return calls["n"] > 1
    success, message, dep = chatbook_creator.create_chatbook(
        name="C", description="", content_selections={
            ContentType.CONVERSATION: [], ContentType.NOTE: [], ContentType.CHARACTER: [],
        }, output_path=output_path, cancel_check=cancel_after_first_package,
    )
    assert success is False
    assert dep.get("cancelled") is True
    assert message == "Export cancelled"
    assert not output_path.exists()
    assert not output_path.with_name(output_path.name + ".partial").exists()

def test_create_chatbook_success_leaves_no_partial(self, mock_prompts_db, mock_chacha_db, chatbook_creator, tmp_path):
    """Atomic finalize: a successful export yields a valid zip and no .partial sibling."""
    import zipfile
    output_path = tmp_path / "cb.zip"
    success, _msg, _dep = chatbook_creator.create_chatbook(
        name="S", description="", content_selections={ContentType.CONVERSATION: []},
        output_path=output_path,
    )
    assert success is True
    assert output_path.exists() and zipfile.is_zipfile(output_path)
    assert not output_path.with_name(output_path.name + ".partial").exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run (RED):
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  "Tests/Chatbooks/test_chatbook_creator.py::TestChatbookCreator::test_create_chatbook_reports_packaging_progress" \
  "Tests/Chatbooks/test_chatbook_creator.py::TestChatbookCreator::test_create_chatbook_cancel_during_packaging_leaves_no_output" \
  "Tests/Chatbooks/test_chatbook_creator.py::TestChatbookCreator::test_create_chatbook_success_leaves_no_partial" \
  -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: FAIL — `create_chatbook()` got an unexpected keyword argument `progress_callback` / cannot import `ExportProgress`. (Confirm the test class is `TestChatbookCreator`; if the surrounding class differs, use its actual name.)

- [ ] **Step 3: Add the types + instance-state init**

At the top of `chatbook_creator.py`, ensure `import os` and `from typing import Callable` are present (add if missing). After the imports / before `class ChatbookCreator`, add:

```python
@dataclass(frozen=True)
class ExportProgress:
    """A single progress tick emitted during chatbook creation."""
    phase: str
    current: int
    total: int


class ChatbookExportCancelled(Exception):
    """Raised internally when cancel_check() returns True at a checkpoint."""
```
(`from dataclasses import dataclass` — add if not already imported.)

In `ChatbookCreator.__init__` (around line 69), initialize the hook slots so the helpers are always safe:
```python
        self._progress_callback: Optional[Callable[[ExportProgress], None]] = None
        self._cancel_check: Optional[Callable[[], bool]] = None
```

Add two private helpers to the class (place them just below `__init__`):
```python
    def _emit_progress(self, phase: str, current: int, total: int) -> None:
        cb = self._progress_callback
        if cb is None:
            return
        try:
            cb(ExportProgress(phase=phase, current=current, total=total))
        except Exception:
            logger.opt(exception=True).debug("ChatbookCreator: progress_callback raised; ignored")

    def _check_cancel(self) -> None:
        if self._cancel_check is not None and self._cancel_check():
            raise ChatbookExportCancelled()

    def _cleanup_paths(self, *paths: Optional[Path]) -> None:
        for p in paths:
            if not p:
                continue
            try:
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                elif p.exists():
                    p.unlink()
            except Exception:
                logger.opt(exception=True).debug(f"ChatbookCreator: cleanup failed for {p}")
```

- [ ] **Step 4: Accept the params, store them, and rework the finalize/except/finally**

Change the `create_chatbook` signature (around line 97) to add, after `auto_include_dependencies: bool = True`:
```python
        progress_callback: Optional[Callable[["ExportProgress"], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
```
Immediately inside the method body (before the `try:` at line ~137) add:
```python
        self._progress_callback = progress_callback
        self._cancel_check = cancel_check
        work_dir: Optional[Path] = None
        partial_path: Optional[Path] = None
```
Inside the `try`, where `work_dir` is currently assigned (line ~146) drop the `Path(...)` onto the pre-declared name (`work_dir = Path(tempfile.mkdtemp(...))`).

Replace the packaging block (lines ~245-252, the `if output_path.suffix == '.zip': ... else: ...` duplication) with:
```python
            # Package into archive (atomic finalize: write .partial, then os.replace)
            if output_path.suffix != '.zip':
                output_path = output_path.with_suffix('.zip')
            partial_path = output_path.with_name(output_path.name + ".partial")
            logger.info(f"ChatbookCreator.create_chatbook: Creating ZIP archive at {output_path}")
            self._create_zip_archive(work_dir, output_path, partial_path)
```

Replace the trailing `except Exception as e:` block (lines ~278-284) so cancel is handled first, both paths clean up, and a `finally` clears the hooks:
```python
        except ChatbookExportCancelled:
            logger.info("ChatbookCreator.create_chatbook: cancelled by request")
            self._cleanup_paths(work_dir, partial_path)
            return False, "Export cancelled", {
                "cancelled": True,
                "missing_dependencies": list(self.missing_dependencies),
                "auto_included": list(self.auto_included_characters),
            }
        except Exception as e:
            logger.opt(exception=True).error("ChatbookCreator.create_chatbook: Error creating chatbook")
            self._cleanup_paths(work_dir, partial_path)
            dependency_info = {
                "missing_dependencies": list(self.missing_dependencies),
                "auto_included": list(self.auto_included_characters),
            }
            return False, f"Error creating chatbook: {str(e)}", dependency_info
        finally:
            self._progress_callback = None
            self._cancel_check = None
```
(The success path already `shutil.rmtree(work_dir)`s before returning — leave it; after a successful `os.replace` there is no `.partial` to clean.)

- [ ] **Step 5: Rework `_create_zip_archive` for the partial path + per-file progress/cancel**

Replace `_create_zip_archive` (line ~1090) with:
```python
    def _create_zip_archive(self, work_dir: Path, output_path: Path, partial_path: Path) -> None:
        """Zip work_dir into a sibling .partial, then atomically replace output_path."""
        files = [p for p in work_dir.rglob('*') if p.is_file()]
        total = len(files)
        with zipfile.ZipFile(partial_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for idx, file_path in enumerate(files):
                self._check_cancel()
                arcname = file_path.relative_to(work_dir)
                zf.write(file_path, arcname)
                self._emit_progress("packaging", idx + 1, total)
        os.replace(partial_path, output_path)
```

- [ ] **Step 6: Add check/emit to each collection phase and to `_discover_relationships`**

For each `_collect_*` method, convert its top-level `for` over the id list to `enumerate`, and insert `self._check_cancel()` + `self._emit_progress(...)` as the FIRST statements of the loop body, BEFORE any per-item `try:` (so a cancel propagates rather than being swallowed by the item-level `except`). Use this exact shape (shown for conversations):

```python
        total = len(conversation_ids)
        for idx, conv_id in enumerate(conversation_ids):
            self._check_cancel()
            self._emit_progress("conversations", idx + 1, total)
            # ...existing loop body (logger.debug + try/except)...
```

Apply the same to every phase, using its id-list variable and phase name:

| Method (approx line) | id-list variable | phase |
|---|---|---|
| `_collect_conversations` (286) | `conversation_ids` | `conversations` |
| `_collect_notes` (641) | `note_ids` | `notes` |
| `_collect_characters` (716) | `character_ids` | `characters` |
| `_collect_media` (774) | `media_ids` | `media` |
| `_collect_prompts` (878) | `prompt_ids` | `prompts` |

Then in `_discover_relationships` (line ~1013), as the first statements of the method body add:
```python
        self._check_cancel()
        self._emit_progress("relationships", 1, 1)
```

- [ ] **Step 7: Run the tests to verify they pass**

Run the three Step-1 tests (GREEN) plus the whole existing creator suite to prove no-behavior-change:
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Chatbooks/test_chatbook_creator.py \
  -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: PASS (new tests green; all pre-existing creator tests still green).

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/Chatbooks/chatbook_creator.py Tests/Chatbooks/test_chatbook_creator.py
git commit -m "feat(export): ChatbookCreator progress + cancel hooks with atomic finalize (157)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Service passthrough of hooks + `cancelled` flag

**Files:**
- Modify: `tldw_chatbook/Chatbooks/local_chatbook_service.py:248-273` (`export_chatbook`)
- Test: `Tests/Chatbooks/test_local_chatbook_service_export.py` (create)

**Interfaces:**
- Consumes: `ChatbookCreator.create_chatbook(..., progress_callback=, cancel_check=)` (Task 1).
- Produces: `LocalChatbookService.export_chatbook(self, request_data, *, progress_callback=None, cancel_check=None)` → dict now includes `"cancelled": bool`.

- [ ] **Step 1: Write the failing test**

Create `Tests/Chatbooks/test_local_chatbook_service_export.py`:
```python
import asyncio
from unittest.mock import patch, MagicMock

from tldw_chatbook.Chatbooks.local_chatbook_service import LocalChatbookService


def _service(tmp_path):
    db_paths = {"ChaChaNotes": str(tmp_path / "c.db"), "Media": str(tmp_path / "m.db"),
                "Prompts": str(tmp_path / "p.db")}
    return LocalChatbookService(db_paths)


def test_export_forwards_hooks_and_maps_cancelled(tmp_path):
    svc = _service(tmp_path)
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return False, "Export cancelled", {"cancelled": True, "missing_dependencies": [], "auto_included": []}

    cb = lambda evt: None
    cc = lambda: True
    with patch("tldw_chatbook.Chatbooks.local_chatbook_service.ChatbookCreator") as CC:
        CC.return_value.create_chatbook.side_effect = fake_create
        result = asyncio.run(svc.export_chatbook(
            {"name": "N", "content_selections": {}, "output_path": str(tmp_path / "o.zip")},
            progress_callback=cb, cancel_check=cc,
        ))
    assert captured["progress_callback"] is cb
    assert captured["cancel_check"] is cc
    assert result["cancelled"] is True
    assert result["success"] is False


def test_export_success_reports_not_cancelled(tmp_path):
    svc = _service(tmp_path)
    with patch("tldw_chatbook.Chatbooks.local_chatbook_service.ChatbookCreator") as CC:
        CC.return_value.create_chatbook.return_value = (True, "ok", {"missing_dependencies": [], "auto_included": []})
        result = asyncio.run(svc.export_chatbook(
            {"name": "N", "content_selections": {}, "output_path": str(tmp_path / "o.zip")}))
    assert result["cancelled"] is False
    assert result["success"] is True
```

- [ ] **Step 2: Run to verify it fails**

Run (RED):
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Chatbooks/test_local_chatbook_service_export.py \
  -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: FAIL — `export_chatbook()` got an unexpected keyword argument `progress_callback`, and/or `KeyError: 'cancelled'`.

- [ ] **Step 3: Implement the passthrough**

In `local_chatbook_service.py`, change the `export_chatbook` signature to:
```python
    async def export_chatbook(self, request_data: Any, *, progress_callback=None, cancel_check=None) -> dict[str, Any]:
```
Add `progress_callback=progress_callback, cancel_check=cancel_check,` to the `creator.create_chatbook(...)` call. After the call, derive and return `cancelled`:
```python
        cancelled = bool(dependency_info.get("cancelled", False)) if isinstance(dependency_info, dict) else False
        return {
            "success": success,
            "message": message,
            "path": str(output_path),
            "dependency_info": dependency_info,
            "name": payload.get("name") or Path(output_path).stem,
            "cancelled": cancelled,
        }
```

- [ ] **Step 4: Run to verify it passes**

Run the Step-1 tests (GREEN). Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chatbooks/local_chatbook_service.py Tests/Chatbooks/test_local_chatbook_service_export.py
git commit -m "feat(export): local chatbook service forwards progress/cancel hooks + cancelled flag (157)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Progress throttle + status-line formatting helpers (pure)

**Files:**
- Create: `tldw_chatbook/Library/export_progress.py`
- Test: `Tests/Library/test_export_progress.py` (create; `mkdir -p Tests/Library` if absent)

**Interfaces:**
- Produces:
  - `ExportProgressThrottle(min_interval: float = 0.1)` with `should_emit(phase: str, current: int, total: int, now: float) -> bool`.
  - `format_export_progress_line(phase: str, current: int, total: int) -> str`.
  - `EXPORT_PHASE_LABELS: dict[str, str]`.

- [ ] **Step 1: Write the failing tests**

Create `Tests/Library/test_export_progress.py`:
```python
from tldw_chatbook.Library.export_progress import (
    ExportProgressThrottle, format_export_progress_line,
)


def test_throttle_emits_on_first_call_and_phase_change():
    t = ExportProgressThrottle(min_interval=1.0)
    assert t.should_emit("media", 1, 10, now=100.0) is True     # first ever
    assert t.should_emit("media", 2, 10, now=100.1) is False    # within interval, same phase
    assert t.should_emit("packaging", 1, 5, now=100.2) is True  # phase change flushes


def test_throttle_emits_on_final_tick_and_interval():
    t = ExportProgressThrottle(min_interval=1.0)
    t.should_emit("media", 1, 10, now=0.0)
    assert t.should_emit("media", 10, 10, now=0.05) is True     # current >= total flushes
    t2 = ExportProgressThrottle(min_interval=0.1)
    t2.should_emit("media", 1, 10, now=0.0)
    assert t2.should_emit("media", 2, 10, now=0.2) is True      # interval elapsed


def test_format_progress_line():
    assert format_export_progress_line("media", 42, 318) == "Collecting media…  42/318"
    assert format_export_progress_line("packaging", 210, 540) == "Packaging archive…  210/540 files"
    assert format_export_progress_line("relationships", 1, 1) == "Resolving links…  1/1"
```

- [ ] **Step 2: Run to verify it fails**

Run (RED):
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Library/test_export_progress.py \
  -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: FAIL — `ModuleNotFoundError: tldw_chatbook.Library.export_progress`.

- [ ] **Step 3: Implement the module**

Create `tldw_chatbook/Library/export_progress.py`:
```python
"""Pure helpers for rendering chatbook-export progress in the Library canvas.

Kept dependency-free (no Textual) so it is trivially unit-testable; the screen
supplies ``time.monotonic()`` as ``now`` so the throttle has no hidden clock.
"""
from __future__ import annotations

from typing import Optional

EXPORT_PHASE_LABELS: dict[str, str] = {
    "conversations": "Collecting conversations",
    "notes": "Collecting notes",
    "characters": "Collecting characters",
    "media": "Collecting media",
    "prompts": "Collecting prompts",
    "relationships": "Resolving links",
    "packaging": "Packaging archive",
}


def format_export_progress_line(phase: str, current: int, total: int) -> str:
    label = EXPORT_PHASE_LABELS.get(phase, "Exporting")
    unit = " files" if phase == "packaging" else ""
    return f"{label}…  {current}/{total}{unit}"


class ExportProgressThrottle:
    """Decides whether a progress tick should be pushed to the UI thread.

    Emits when the phase changes, when the phase's final item is reached
    (``current >= total``), or when ``min_interval`` seconds have elapsed since
    the last emit — so a fast inner loop never floods the UI, yet the line
    never freezes mid-count.
    """

    def __init__(self, min_interval: float = 0.1) -> None:
        self._min_interval = min_interval
        self._last_phase: Optional[str] = None
        self._last_emit: Optional[float] = None

    def should_emit(self, phase: str, current: int, total: int, now: float) -> bool:
        if (
            phase != self._last_phase
            or current >= total
            or self._last_emit is None
            or (now - self._last_emit) >= self._min_interval
        ):
            self._last_phase = phase
            self._last_emit = now
            return True
        return False
```

- [ ] **Step 4: Run to verify it passes**

Run the Step-1 tests (GREEN). Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Library/export_progress.py Tests/Library/test_export_progress.py
git commit -m "feat(export): pure progress-throttle + status-line formatter for Library export (157)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Library UI — worker plumbing + live progress line

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` (`__init__` state; `handle_library_export_submit`; `_start_library_export_worker`/`_run_library_export_worker`; `_run_library_export_via_service`; add `_apply_library_export_progress` + `_refresh_library_export_status_line`)
- Test: `Tests/UI/test_library_export_progress_apply.py` (create; `mkdir -p Tests/UI` if absent)

**Interfaces:**
- Consumes: `LocalChatbookService.export_chatbook(..., progress_callback=, cancel_check=)` (Task 2); `ExportProgressThrottle`, `format_export_progress_line` (Task 3); `ExportProgress` (Task 1).
- Produces: `self._library_export_cancel_event: threading.Event | None`; `_apply_library_export_progress(run_id, phase, current, total)`; `_refresh_library_export_status_line()`. `_run_library_export_via_service` now takes `*, progress_callback=None, cancel_check=None` and its worker builds them.

- [ ] **Step 1: Write the failing tests**

Create `Tests/UI/test_library_export_progress_apply.py` (uses a `SimpleNamespace` fake `self`, so no app/screen instantiation):
```python
from types import SimpleNamespace
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen


def _fake(run_id=7, running=True):
    calls = []
    fake = SimpleNamespace(
        _library_export_run_id=run_id,
        _library_export_running=running,
        _library_export_status="",
        _refresh_library_export_status_line=lambda: calls.append("refresh"),
    )
    return fake, calls


def test_progress_apply_ignores_stale_run():
    fake, calls = _fake(run_id=7)
    LibraryScreen._apply_library_export_progress(fake, 3, "media", 5, 10)  # 3 != 7
    assert fake._library_export_status == ""
    assert calls == []


def test_progress_apply_ignores_when_not_running():
    fake, calls = _fake(run_id=7, running=False)
    LibraryScreen._apply_library_export_progress(fake, 7, "media", 5, 10)
    assert fake._library_export_status == ""
    assert calls == []


def test_progress_apply_updates_current_run():
    fake, calls = _fake(run_id=7)
    LibraryScreen._apply_library_export_progress(fake, 7, "media", 5, 10)
    assert fake._library_export_status == "Collecting media…  5/10"
    assert calls == ["refresh"]
```

- [ ] **Step 2: Run to verify it fails**

Run (RED):
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/UI/test_library_export_progress_apply.py \
  -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: FAIL — `AttributeError: type object 'LibraryScreen' has no attribute '_apply_library_export_progress'`.

- [ ] **Step 3: Add imports + the cancel-event field**

Near the top of `library_screen.py`, ensure `import threading` and `import time` are present (add if missing). Add the import:
```python
from tldw_chatbook.Library.export_progress import (
    ExportProgressThrottle, format_export_progress_line,
)
```
In `__init__` (alongside the other `_library_export_*` fields, ~line 636) add:
```python
        self._library_export_cancel_event: threading.Event | None = None
```

- [ ] **Step 4: Create the cancel event at submit**

In `handle_library_export_submit` (line ~3418), right where it sets `_library_export_running = True`, add before the worker dispatch:
```python
        self._library_export_cancel_event = threading.Event()
```
(The event is created but nothing sets it yet; navigate-away and the Cancel button are wired in Task 5.)

- [ ] **Step 5: Add the apply + refresh methods**

Add to `LibraryScreen` (place next to `_apply_library_export_failure`):
```python
    def _apply_library_export_progress(self, run_id: int, phase: str, current: int, total: int) -> None:
        """UI-thread progress tick: update the status line in place if this run is current."""
        if run_id != self._library_export_run_id or not self._library_export_running:
            return
        self._library_export_status = format_export_progress_line(phase, current, total)
        self._refresh_library_export_status_line()

    def _refresh_library_export_status_line(self) -> None:
        """Update only the #library-export-status-line widget (no recompose)."""
        if not self.is_mounted or self._library_selected_row_id != LIBRARY_ROW_INGEST_EXPORT:
            return
        try:
            widget = self.query_one("#library-export-status-line", Static)
            widget.update(self._library_export_status)
            widget.display = bool(self._library_export_status)
        except (NoMatches, QueryError):
            pass
```
(Confirm `Static`, `NoMatches`, `QueryError`, and `LIBRARY_ROW_INGEST_EXPORT` are already imported in this file — they are used by `_update_library_export_canvas_after_run`; reuse the same imports.)

- [ ] **Step 6: Build the progress callback + cancel_check in the worker and thread them through**

Change `_run_library_export_via_service` (line ~3524) signature to accept the hooks and forward them at the `export_chatbook` call:
```python
    @staticmethod
    def _run_library_export_via_service(
        service: Any,
        payload: dict[str, Any],
        *,
        name: str,
        description: str,
        progress_callback=None,
        cancel_check=None,
    ) -> dict[str, Any]:
```
At the single `asyncio.run(service.export_chatbook(payload))` call (line ~3554) pass the hooks:
```python
        export_result = asyncio.run(service.export_chatbook(
            payload, progress_callback=progress_callback, cancel_check=cancel_check,
        ))
```

In `_run_library_export_worker` (line ~3600), before the `outcome = self._run_library_export_via_service(...)` call (line ~3642), build the throttled callback and the cancel check, capturing `run_id` and the event:
```python
        cancel_event = self._library_export_cancel_event
        throttle = ExportProgressThrottle()

        def _progress_cb(evt) -> None:
            try:
                if not throttle.should_emit(evt.phase, evt.current, evt.total, time.monotonic()):
                    return
                self.app.call_from_thread(
                    self._apply_library_export_progress, run_id, evt.phase, evt.current, evt.total,
                )
            except Exception:
                # NoApp/shutdown mid-marshal must not crash the worker.
                pass

        outcome = self._run_library_export_via_service(
            service, payload, name=name, description=description,
            progress_callback=_progress_cb,
            cancel_check=(cancel_event.is_set if cancel_event is not None else None),
        )
```
(Replace the existing `outcome = self._run_library_export_via_service(service, payload, name=name, description=description)` line.)

- [ ] **Step 7: Run to verify it passes**

Run the Step-1 apply tests (GREEN), plus a smoke import of the screen:
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/UI/test_library_export_progress_apply.py \
  -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
HOME=/private/tmp/tldw-chatbook-test-home PYTHONPATH=$(pwd) \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c "import tldw_chatbook.UI.Screens.library_screen; print('import ok')"
```
Expected: PASS + `import ok`.

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/UI/Screens/library_screen.py Tests/UI/test_library_export_progress_apply.py
git commit -m "feat(export): live per-phase progress line in the Library export canvas (157)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Library UI — Cancel button + cancelled outcome

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_export_canvas.py` (add the Cancel `Button`)
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` (`handle_library_export_cancel`; propagate `cancelled` through `_run_library_export_via_service` + `_run_library_export_worker`; `_marshal_library_export_cancelled` + `_apply_library_export_cancelled`; set the event in `_reset_library_export_transient_state`; hide the Cancel button in `_update_library_export_canvas_after_run`)
- Modify: `backlog/tasks/task-157 - Chatbook-export-progress-reporting-and-cancel.md` (mark Done, tick ACs)
- Test: `Tests/UI/test_library_export_cancel.py` (create), plus a widget-level `app.run_test()` smoke.

**Interfaces:**
- Consumes: everything from Tasks 1–4; the `cancelled` key added to the service outcome; `self._library_export_cancel_event` (Task 4).
- Produces: `handle_library_export_cancel`; `_apply_library_export_cancelled(run_id)`; the canvas `#library-export-cancel` button.

- [ ] **Step 1: Write the failing tests**

Create `Tests/UI/test_library_export_cancel.py`:
```python
import threading
from types import SimpleNamespace

import pytest

from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library.library_export_canvas import LibraryExportCanvas
from tldw_chatbook.Library.library_export_state import build_library_export_form_state
from tldw_chatbook.Library.library_export_scope import ExportScope


def test_cancel_apply_ignores_stale_run():
    calls = []
    fake = SimpleNamespace(
        _library_export_run_id=9, _library_export_running=True,
        _library_export_status="Packaging archive…  1/5", _library_export_error="x",
        _update_library_export_canvas_after_run=lambda: calls.append("update"),
    )
    LibraryScreen._apply_library_export_cancelled(fake, 4)  # 4 != 9
    assert fake._library_export_running is True
    assert calls == []


def test_cancel_apply_current_run_sets_cancelled_status():
    calls = []
    fake = SimpleNamespace(
        _library_export_run_id=9, _library_export_running=True,
        _library_export_status="Packaging archive…  1/5", _library_export_error="x",
        _update_library_export_canvas_after_run=lambda: calls.append("update"),
    )
    LibraryScreen._apply_library_export_cancelled(fake, 9)
    assert fake._library_export_running is False
    assert fake._library_export_status == "Export cancelled."
    assert fake._library_export_error == ""
    assert calls == ["update"]


def test_cancel_handler_sets_event():
    fake = SimpleNamespace(
        _library_export_cancel_event=threading.Event(),
        _library_export_running=True, _library_export_status="",
        _refresh_library_export_status_line=lambda: None,
    )
    LibraryScreen.handle_library_export_cancel(fake, None)
    assert fake._library_export_cancel_event.is_set()
    assert fake._library_export_status == "Cancelling…"


@pytest.mark.asyncio
async def test_cancel_button_visible_only_while_running():
    from textual.app import App

    def _state(running):
        return build_library_export_form_state(
            scope=ExportScope(kind="everything"), counts={"total": 3}, name="n",
            description="", media_quality="thumbnail", destination="/tmp/x.zip",
            running=running, status_line="Exporting…" if running else "",
        )

    class Host(App):
        def compose(self):
            yield LibraryExportCanvas(_state(True), id="library-export-canvas")

    app = Host()
    async with app.run_test() as pilot:
        cancel = pilot.app.query_one("#library-export-cancel")
        assert cancel.display is True
```
(If the `counts=` shape differs from `{"total": 3}`, match whatever `build_library_export_form_state` expects — check `Tests/Library/` for an existing constructor call to copy.)

- [ ] **Step 2: Run to verify it fails**

Run (RED):
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/UI/test_library_export_cancel.py \
  -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: FAIL — `LibraryScreen` has no `_apply_library_export_cancelled`/`handle_library_export_cancel`; `#library-export-cancel` not found.

- [ ] **Step 3: Add the Cancel button to the canvas widget**

In `tldw_chatbook/Widgets/Library/library_export_canvas.py`, in `compose()` immediately after the `#library-export-submit` `Button` (the last yielded widget, ~line 137-143), add:
```python
        cancel_button = Button(
            "Cancel",
            id="library-export-cancel",
            classes="library-canvas-action",
            compact=True,
        )
        cancel_button.display = bool(state.running)
        yield cancel_button
```

- [ ] **Step 4: Add the Cancel handler**

In `library_screen.py`, next to `handle_library_export_submit`, add:
```python
    @on(Button.Pressed, "#library-export-cancel")
    def handle_library_export_cancel(self, event: "Button.Pressed") -> None:
        """Request cancellation of the in-flight export.

        Sets the worker's cancel Event (idempotent) and flips the status line to
        "Cancelling…". Deliberately does NOT bump _library_export_run_id: the run
        is still the current, visible one until the worker reports back with the
        cancelled outcome (see _apply_library_export_cancelled).
        """
        if not self._library_export_running:
            return
        event_obj = self._library_export_cancel_event
        if event_obj is not None:
            event_obj.set()
        self._library_export_status = "Cancelling…"
        self._refresh_library_export_status_line()
```

- [ ] **Step 5: Propagate `cancelled` through the worker outcome**

In `_run_library_export_via_service` (library_screen.py), add `"cancelled": bool(export_result.get("cancelled", False)),` to the **not-success** return dict (the `if not export_result.get("success"):` branch, ~line 3540-3546) and `"cancelled": False,` to the success return dict (~line 3592-3598) and to the exception-return dict (~line 3529-3535). Then in `_run_library_export_worker` (line ~3645), branch on it:
```python
        if outcome.get("cancelled"):
            self._marshal_library_export_cancelled(run_id)
        elif outcome["success"]:
            self._marshal_library_export_success(
                run_id, outcome["path"], outcome["dependency_info"],
                bool(outcome["registry_recorded"]), outcome["message"],
            )
        else:
            self._marshal_library_export_failure(run_id, outcome["message"])
```

- [ ] **Step 6: Add the marshal + apply for cancelled**

Add to `LibraryScreen` (next to `_marshal_library_export_failure` / `_apply_library_export_failure`):
```python
    def _marshal_library_export_cancelled(self, run_id: int) -> None:
        """Marshal a cancelled run onto the UI thread (called from the worker)."""
        try:
            self.app.call_from_thread(self._apply_library_export_cancelled, run_id)
        except Exception:
            pass

    def _apply_library_export_cancelled(self, run_id: int) -> None:
        """UI-thread completion for a cancelled run: clear running, show cancelled, return to form."""
        if run_id != self._library_export_run_id:
            return
        self._library_export_running = False
        self._library_export_status = "Export cancelled."
        self._library_export_error = ""
        self._update_library_export_canvas_after_run()
```

- [ ] **Step 7: Signal cancel on navigate-away + hide the button on completion**

In `_reset_library_export_transient_state` (line ~3139), before bumping `_library_export_run_id`, set the outgoing event so the worker stops:
```python
        if self._library_export_cancel_event is not None:
            self._library_export_cancel_event.set()
```
In `_update_library_export_canvas_after_run` (line ~3827), inside the existing `try:` that updates the widgets, add a line hiding the cancel button when not running:
```python
            self.query_one("#library-export-cancel", Button).display = bool(state.running)
```

- [ ] **Step 8: Run to verify it passes**

Run the Step-1 tests (GREEN):
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/UI/test_library_export_cancel.py Tests/UI/test_library_export_progress_apply.py \
  -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: PASS.

- [ ] **Step 9: Mark the backlog task Done**

Tick the ACs and set status in `backlog/tasks/task-157 - Chatbook-export-progress-reporting-and-cancel.md`:
```bash
perl -0pi -e 's/- \[ \] (#\d)/- [x] $1/g' "backlog/tasks/task-157 - Chatbook-export-progress-reporting-and-cancel.md"
perl -0pi -e 's/^status: .*/status: Done/m' "backlog/tasks/task-157 - Chatbook-export-progress-reporting-and-cancel.md"
```
Add a short `## Implementation Notes` section summarizing the approach (creator hooks + atomic finalize, service passthrough, throttle helper, Library canvas progress line + Cancel button).

- [ ] **Step 10: Commit**

```bash
git add tldw_chatbook/Widgets/Library/library_export_canvas.py tldw_chatbook/UI/Screens/library_screen.py \
  Tests/UI/test_library_export_cancel.py "backlog/tasks/task-157 - Chatbook-export-progress-reporting-and-cancel.md"
git commit -m "feat(export): Cancel control + cancelled outcome for Library export (157)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Final gate (after Task 5)

Run the full touched-surface suite:
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Chatbooks/ Tests/Library/test_export_progress.py Tests/UI/test_library_export_progress_apply.py Tests/UI/test_library_export_cancel.py \
  -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Then the whole-branch review (opus), then finishing-a-development-branch. Served-TUI visual QA of the running progress line + Cancel press is worthwhile but optional (a live export with enough items to see counts move); the modal FileSave picker resists scripting, so a manual capture is acceptable.
