# Console Clipboard Paste + Drag-Drop Attach (TASK-216) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dropping a file onto the terminal (path-paste) auto-attaches it to the Console composer, and Alt+V grabs an image from the OS clipboard into the same pending-attachment pipeline.

**Architecture:** A pure helper module (`Chat/console_paste_attach.py`) owns path extraction from pasted text, attachability gating, and the `ImageGrab` wrapper. `attachment_core` gains a bytes entry so clipboard images run the identical validate/resize pipeline with no temp files. The screen intercepts path-like pastes in `on_paste` after its existing guards, and an Alt+V binding grabs the clipboard off-loop in a dedicated worker group.

**Tech Stack:** Python ≥3.11, Textual 8.2.7, PIL (ImageGrab — verified working on this platform), pytest. No new dependencies.

**Spec:** `Docs/superpowers/specs/2026-07-13-console-paste-dnd-attach-design.md` — read it first; decisions there are settled.

## Global Constraints

- Run all tests with `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <paths> -q --no-header` from the worktree root (`/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/console-paste-dnd-216`).
- Modify ONLY files each task names. NEVER touch files outside the worktree.
- Existing tests are read-only; test-file additions append-only at true EOF — verify `git diff HEAD -- <testfile> | grep -c '^-[^-]'` is 0 after appending.
- Exact values (verbatim from spec): worker group `"console-clipboard-grab"` (`exclusive=True`); binding `alt+v` → action `paste_clipboard_image`, footer label `"Paste image"`, `show=True`; clipboard display name `clipboard-YYYYmmdd-HHMMSS.png`; toasts: `"No image on the clipboard."` / `"Clipboard images aren't readable on this platform — use Attach or drop a file."`; multi-drop toast `"Attached first of N dropped files."`
- Path-paste interception runs AFTER `on_paste`'s existing setup-modal + `_should_capture_console_input` guards and must `event.stop()` + dismiss guidance on the attach route (no draft-sync call — draft unchanged).
- The attach route reuses `_process_console_attachment(path)` verbatim (worker group `console-attachment`, from PR #621) — no parallel attach path.
- SPEC AMENDMENT (approved deviation to record in the spec during Task 4): the "command-palette entry" becomes the footer-visible Alt+V binding — the app's ^p palette is a custom app-level modal, not Textual's CommandPalette; the `show=True` footer hint is the same discoverability mechanism alt+m "Model" uses.
- Legacy chat untouched; legacy image regression gate must stay green unedited.
- CI is intentionally cancelled remotely — verify locally.
- End every commit message with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Pure module — path extraction, attachability, clipboard grab

**Files:**
- Create: `tldw_chatbook/Chat/console_paste_attach.py`
- Test: `Tests/Chat/test_console_paste_attach.py` (new)

**Interfaces:**
- Consumes: `tldw_chatbook.Utils.path_validation.is_safe_path`, `tldw_chatbook.Chat.attachment_core.ATTACHMENT_FILTER_SPECS`, `PIL.ImageGrab` (guarded).
- Produces (used by Tasks 3–4):
  - `@dataclass(frozen=True) DroppedPaste(path: str, total_dropped: int)`
  - `extract_dropped_path(pasted_text: str) -> DroppedPaste | None` — pure string logic, no filesystem access.
  - `looks_attachable(path: str, allowed_root: str | None = None) -> bool` — existence + `is_safe_path` (root defaults to home like `load_processed_file`) + extension in the `ATTACHMENT_FILTER_SPECS` pattern union.
  - `@dataclass(frozen=True) ClipboardGrab(kind: Literal["image", "paths", "empty", "unavailable"], png_bytes: bytes | None = None, paths: tuple[str, ...] = ())`
  - `grab_clipboard_image() -> ClipboardGrab` — sync/blocking; callers run off-loop.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Chat/test_console_paste_attach.py
from io import BytesIO

from PIL import Image as PILImage

from tldw_chatbook.Chat import console_paste_attach as cpa
from tldw_chatbook.Chat.console_paste_attach import (
    ClipboardGrab,
    DroppedPaste,
    extract_dropped_path,
    grab_clipboard_image,
    looks_attachable,
)


# --- extract_dropped_path matrix ---

def test_extracts_plain_absolute_path():
    result = extract_dropped_path("/Users/me/Pictures/photo.png")
    assert result == DroppedPaste(path="/Users/me/Pictures/photo.png", total_dropped=1)


def test_extracts_path_with_trailing_newline_and_spaces():
    result = extract_dropped_path("/tmp/a.png \n")
    assert result is not None and result.path == "/tmp/a.png"


def test_extracts_single_quoted_path():
    result = extract_dropped_path("'/Users/me/My Files/photo.png'")
    assert result is not None and result.path == "/Users/me/My Files/photo.png"


def test_extracts_double_quoted_path():
    result = extract_dropped_path('"/Users/me/My Files/photo.png"')
    assert result is not None and result.path == "/Users/me/My Files/photo.png"


def test_extracts_backslash_escaped_spaces():
    result = extract_dropped_path("/Users/me/My\\ Files/photo.png")
    assert result is not None and result.path == "/Users/me/My Files/photo.png"


def test_extracts_file_uri_with_percent_encoding():
    result = extract_dropped_path("file:///Users/me/My%20Files/photo.png")
    assert result is not None and result.path == "/Users/me/My Files/photo.png"


def test_multi_drop_returns_first_with_count():
    result = extract_dropped_path("/tmp/a.png\n/tmp/b.png\n/tmp/c.md\n")
    assert result == DroppedPaste(path="/tmp/a.png", total_dropped=3)


def test_prose_containing_a_path_is_not_a_drop():
    assert extract_dropped_path("what does /etc/hosts do?") is None


def test_multiline_prose_is_not_a_drop():
    assert extract_dropped_path("line one\nnot /a/path at all\n") is None


def test_relative_and_tilde_paths_are_not_drops():
    assert extract_dropped_path("notes.md") is None
    assert extract_dropped_path("~/notes.md") is None


def test_empty_and_whitespace_are_not_drops():
    assert extract_dropped_path("") is None
    assert extract_dropped_path("   \n") is None


# --- looks_attachable ---

def test_looks_attachable_true_for_supported_existing_in_root(tmp_path):
    target = tmp_path / "photo.png"
    target.write_bytes(b"x")
    assert looks_attachable(str(target), allowed_root=str(tmp_path)) is True


def test_looks_attachable_false_for_missing_out_of_root_unsupported(tmp_path):
    missing = tmp_path / "nope.png"
    assert looks_attachable(str(missing), allowed_root=str(tmp_path)) is False

    outside = tmp_path / "esc.png"
    outside.write_bytes(b"x")
    assert looks_attachable(str(outside), allowed_root=str(tmp_path / "inner")) is False

    unsupported = tmp_path / "binary.exe"
    unsupported.write_bytes(b"x")
    assert looks_attachable(str(unsupported), allowed_root=str(tmp_path)) is False


# --- grab_clipboard_image kind mapping (ImageGrab monkeypatched) ---

def _png_of(size=(8, 8)):
    return PILImage.new("RGB", size, (5, 5, 200))


def test_grab_maps_image_to_png_bytes(monkeypatch):
    monkeypatch.setattr(cpa, "_grabclipboard", lambda: _png_of())
    grab = grab_clipboard_image()
    assert grab.kind == "image"
    assert grab.png_bytes is not None
    assert PILImage.open(BytesIO(grab.png_bytes)).size == (8, 8)


def test_grab_maps_path_list_to_paths(monkeypatch):
    monkeypatch.setattr(cpa, "_grabclipboard", lambda: ["/tmp/a.png", "/tmp/b.md"])
    grab = grab_clipboard_image()
    assert grab.kind == "paths"
    assert grab.paths == ("/tmp/a.png", "/tmp/b.md")


def test_grab_maps_none_to_empty(monkeypatch):
    monkeypatch.setattr(cpa, "_grabclipboard", lambda: None)
    assert grab_clipboard_image().kind == "empty"


def test_grab_maps_errors_to_unavailable(monkeypatch):
    def _boom():
        raise OSError("no clipboard backend")

    monkeypatch.setattr(cpa, "_grabclipboard", _boom)
    assert grab_clipboard_image().kind == "unavailable"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_paste_attach.py -q --no-header`
Expected: collection error — `ModuleNotFoundError: No module named 'tldw_chatbook.Chat.console_paste_attach'`

- [ ] **Step 3: Write the module**

```python
# tldw_chatbook/Chat/console_paste_attach.py
"""Detect drag-drop path pastes and grab clipboard images for Console attach.

Terminal reality this module encodes: dropping a file onto a terminal pastes
its PATH as text, and clipboard IMAGES produce no terminal event at all —
they are read explicitly via PIL.ImageGrab (macOS/Windows; unavailable on
most Linux setups).

Pure module: no Textual imports; ``grab_clipboard_image`` is sync/blocking
and callers must run it off the event loop.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from fnmatch import fnmatch
from io import BytesIO
from typing import Literal
from urllib.parse import unquote, urlparse

from loguru import logger

from tldw_chatbook.Chat.attachment_core import ATTACHMENT_FILTER_SPECS
from tldw_chatbook.Utils.path_validation import is_safe_path

_SUPPORTED_PATTERNS: tuple[str, ...] = tuple(
    pattern
    for _label, patterns in ATTACHMENT_FILTER_SPECS
    for pattern in patterns.split(";")
)


@dataclass(frozen=True)
class DroppedPaste:
    """A path-paste candidate extracted from pasted text."""

    path: str
    total_dropped: int


@dataclass(frozen=True)
class ClipboardGrab:
    """Result of reading the OS clipboard for attachable content."""

    kind: Literal["image", "paths", "empty", "unavailable"]
    png_bytes: bytes | None = None
    paths: tuple[str, ...] = ()


def _decode_token(token: str) -> str | None:
    """Return the path a single pasted token denotes, or None.

    Args:
        token: One whitespace-trimmed line of pasted text.

    Returns:
        An absolute path string when the token is path-like (quoted,
        backslash-escaped, ``file://`` URI, or plain absolute), else None.
    """
    if not token:
        return None
    if token.startswith("file://"):
        parsed = urlparse(token)
        candidate = unquote(parsed.path)
        return candidate if candidate.startswith("/") else None
    if len(token) >= 2 and token[0] == token[-1] and token[0] in ("'", '"'):
        candidate = token[1:-1]
        return candidate if candidate.startswith("/") else None
    if "\\ " in token:
        candidate = token.replace("\\ ", " ")
        return candidate if candidate.startswith("/") else None
    if token.startswith("/"):
        # A bare absolute path must not contain unescaped spaces — pasted
        # prose like "what does /etc/hosts do?" never reaches here because
        # the caller splits per line and rejects lines with spaces.
        return token if " " not in token else None
    return None


def extract_dropped_path(pasted_text: str) -> DroppedPaste | None:
    """Return the first dropped path when a paste is purely path-like.

    Args:
        pasted_text: Raw text from the terminal paste event.

    Returns:
        The first path plus the total number of dropped paths, or None when
        any line of the paste is not path-like (prose stays text).
    """
    lines = [line.strip() for line in pasted_text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return None
    decoded: list[str] = []
    for line in lines:
        candidate = _decode_token(line)
        if candidate is None:
            return None
        decoded.append(candidate)
    return DroppedPaste(path=decoded[0], total_dropped=len(decoded))


def looks_attachable(path: str, allowed_root: str | None = None) -> bool:
    """Return whether a dropped path is safe and supported for auto-attach.

    Args:
        path: Absolute path extracted from a paste.
        allowed_root: Directory the file must live under; defaults to the
            user's home directory (matching ``attachment_core``).

    Returns:
        True when the file exists, is inside the allowed root, and matches
        the shared picker filter patterns. (Known inherited wart: the specs
        advertise tiff/svg the image pipeline rejects — TASK-222.)
    """
    root = allowed_root or os.path.expanduser("~")
    if not os.path.isfile(path):
        return False
    if not is_safe_path(path, root):
        return False
    name = os.path.basename(path)
    return any(fnmatch(name, pattern) for pattern in _SUPPORTED_PATTERNS)


def _grabclipboard():
    """Read the OS clipboard via PIL.ImageGrab (module seam for tests)."""
    from PIL import ImageGrab

    return ImageGrab.grabclipboard()


def grab_clipboard_image() -> ClipboardGrab:
    """Read the OS clipboard, classifying what it holds.

    Returns:
        ``image`` with PNG bytes for a clipboard image; ``paths`` for
        copied files (e.g. Finder); ``empty`` when the clipboard holds no
        image; ``unavailable`` when the platform/backend cannot be read.
    """
    try:
        grabbed = _grabclipboard()
    except Exception:
        logger.opt(exception=True).info("Clipboard image grab unavailable.")
        return ClipboardGrab(kind="unavailable")
    if grabbed is None:
        return ClipboardGrab(kind="empty")
    if isinstance(grabbed, list):
        return ClipboardGrab(
            kind="paths", paths=tuple(str(item) for item in grabbed)
        )
    buffer = BytesIO()
    grabbed.save(buffer, format="PNG")
    return ClipboardGrab(kind="image", png_bytes=buffer.getvalue())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_paste_attach.py -q --no-header`
Expected: 17 passed.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_paste_attach.py Tests/Chat/test_console_paste_attach.py
git commit -m "feat(console): path-drop extraction and clipboard grab helpers

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: `attachment_core.process_attachment_bytes`

**Files:**
- Modify: `tldw_chatbook/Chat/attachment_core.py` (append after `process_attachment_path`)
- Test: `Tests/Chat/test_attachment_core.py` (append-only)

**Interfaces:**
- Consumes: `ChatImageHandler` (`MAX_IMAGE_SIZE`, `_process_image_data(image_data, extension, mime_type)` — async staticmethod, verified in `Event_Handlers/Chat_Events/chat_image_events.py`).
- Produces (used by Task 4): `async process_attachment_bytes(data: bytes, *, display_name: str, mime_type: str = "image/png") -> PendingAttachment` — raises `ValueError` for oversized/corrupt bytes; result has `insert_mode="attachment"`, `file_type="image"`, `file_path=""`.

- [ ] **Step 1: Write the failing tests** (append at true EOF)

```python
def test_process_attachment_bytes_builds_image_pending():
    import asyncio
    from io import BytesIO

    from PIL import Image as PILImage

    from tldw_chatbook.Chat.attachment_core import process_attachment_bytes

    buffer = BytesIO()
    PILImage.new("RGB", (32, 32), (200, 10, 10)).save(buffer, format="PNG")
    data = buffer.getvalue()

    attachment = asyncio.run(
        process_attachment_bytes(data, display_name="clipboard-20260713-120000.png")
    )
    assert attachment.insert_mode == "attachment"
    assert attachment.file_type == "image"
    assert attachment.file_path == ""
    assert attachment.mime_type == "image/png"
    assert attachment.display_name == "clipboard-20260713-120000.png"
    assert attachment.data and attachment.processed_size == len(attachment.data)


def test_process_attachment_bytes_rejects_corrupt_and_oversized(monkeypatch):
    import asyncio

    import pytest as _pytest

    from tldw_chatbook.Chat import attachment_core
    from tldw_chatbook.Chat.attachment_core import process_attachment_bytes

    with _pytest.raises(ValueError, match="not a valid image"):
        asyncio.run(process_attachment_bytes(b"junk", display_name="x.png"))

    monkeypatch.setattr(attachment_core, "MAX_IMAGE_BYTES", 4)
    with _pytest.raises(ValueError, match="too large"):
        asyncio.run(
            process_attachment_bytes(b"12345678", display_name="big.png")
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_attachment_core.py -q --no-header -k "attachment_bytes"`
Expected: FAIL with `ImportError: cannot import name 'process_attachment_bytes'`.

- [ ] **Step 3: Implement** (append to `attachment_core.py`; also add module constant near `MAX_ATTACHMENT_BYTES`)

```python
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # matches ChatImageHandler.MAX_IMAGE_SIZE
```

```python
async def process_attachment_bytes(
    data: bytes,
    *,
    display_name: str,
    mime_type: str = "image/png",
) -> PendingAttachment:
    """Build an image PendingAttachment from raw bytes (clipboard path).

    Runs the same validate/resize pipeline as file-based image attachments
    (10 MB cap, PIL validation, resize via ChatImageHandler) with no temp
    files.

    Args:
        data: Raw image bytes (e.g. PNG-encoded clipboard grab).
        display_name: User-facing name (e.g. ``clipboard-20260713-120000.png``).
        mime_type: MIME type of ``data``.

    Returns:
        An attachment-mode image PendingAttachment with ``file_path=""``.

    Raises:
        ValueError: If the bytes exceed the image cap or are not a valid image.
    """
    from io import BytesIO

    from PIL import Image as PILImage

    from tldw_chatbook.Event_Handlers.Chat_Events.chat_image_events import (
        ChatImageHandler,
    )

    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"Image too large ({len(data) / 1024 / 1024:.1f}MB). "
            f"Maximum size: {MAX_IMAGE_BYTES / 1024 / 1024:.0f}MB"
        )
    try:
        PILImage.open(BytesIO(data)).verify()
    except Exception as exc:
        raise ValueError("Clipboard data is not a valid image.") from exc
    extension = ".png" if "png" in mime_type else ".jpg"
    processed = await ChatImageHandler._process_image_data(data, extension, mime_type)
    return PendingAttachment(
        file_path="",
        display_name=display_name,
        file_type="image",
        insert_mode="attachment",
        data=processed,
        mime_type=mime_type,
        text_content=None,
        original_size=len(data),
        processed_size=len(processed),
    )
```

- [ ] **Step 4: Run tests**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_attachment_core.py -q --no-header`
Expected: all pass (pre-existing 14 + 2 new). Verify append-only on the test file.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/attachment_core.py Tests/Chat/test_attachment_core.py
git commit -m "feat(chat): bytes entry into the attachment pipeline for clipboard images

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: `on_paste` interception

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (`on_paste` at line ~8443 — locate by name)
- Test: `Tests/UI/test_console_native_chat_flow.py` (append-only)

**Interfaces:**
- Consumes: `extract_dropped_path`, `looks_attachable` (Task 1); existing `_process_console_attachment(file_path)` (verified at ~7414, worker group `console-attachment`).
- Produces: path-pastes that pass gating route to attach; everything else unchanged.

- [ ] **Step 1: Write the failing tests** (append at true EOF; follow the file's ConsoleHarness idiom from neighboring attachment tests; monkeypatch `chat_screen`-imported names)

```python
async def test_path_paste_routes_to_attach_instead_of_draft(tmp_path, monkeypatch):
    from PIL import Image as PILImage

    from textual.events import Paste

    image_path = tmp_path / "dropped.png"
    PILImage.new("RGB", (8, 8), (9, 9, 9)).save(image_path, format="PNG")

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause()

        # Widen the attach root to tmp_path for both gating and processing.
        # chat_screen imports these helpers BY NAME, so patch the consuming
        # module's bindings, not the source module's.
        import tldw_chatbook.Chat.attachment_core as attachment_core
        import tldw_chatbook.UI.Screens.chat_screen as chat_screen_module
        from tldw_chatbook.Chat.console_paste_attach import (
            looks_attachable as original_attachable,
        )

        original_load = attachment_core.load_processed_file

        async def _rooted(file_path, *, allowed_root=None):
            return await original_load(file_path, allowed_root=str(tmp_path))

        monkeypatch.setattr(attachment_core, "load_processed_file", _rooted)
        monkeypatch.setattr(
            chat_screen_module,
            "looks_attachable",
            lambda path, allowed_root=None: original_attachable(
                path, allowed_root=str(tmp_path)
            ),
        )

        console.on_paste(Paste(text=str(image_path)))
        for _ in range(80):
            store = console._ensure_console_chat_store()
            session_id = store.active_session_id
            if session_id and store.pending_attachment(session_id) is not None:
                break
            await pilot.pause(0.05)

        store = console._ensure_console_chat_store()
        pending = store.pending_attachment(store.active_session_id)
        assert pending is not None and pending.file_type == "image"
        assert composer.draft_text() == ""  # path did NOT land as draft text


async def test_prose_paste_still_lands_in_draft():
    from textual.events import Paste

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause()

        console.on_paste(Paste(text="what does /etc/hosts do?"))
        await pilot.pause()
        assert composer.draft_text() == "what does /etc/hosts do?"
```

(The test patches `chat_screen_module.looks_attachable` because Step 3 imports the helpers by name — keep test and import style in lockstep if you deviate, and disclose.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_native_chat_flow.py -q --no-header -k "path_paste or prose_paste"`
Expected: first test FAILS (path lands as draft text / no pending staged); second passes already (baseline behavior) — that is acceptable RED evidence for the pair.

- [ ] **Step 3: Implement**

In `chat_screen.py`, add to the module imports: `from tldw_chatbook.Chat.console_paste_attach import extract_dropped_path, looks_attachable`.

In `on_paste`, after the `_should_capture_console_input` guard and BEFORE `composer.insert_pasted_text(event.text)`:

```python
        dropped = extract_dropped_path(event.text)
        if dropped is not None and looks_attachable(dropped.path):
            event.stop()
            self._dismiss_console_guidance()
            if dropped.total_dropped > 1:
                self.app_instance.notify(
                    f"Attached first of {dropped.total_dropped} dropped files."
                )
            self.run_worker(
                self._process_console_attachment(dropped.path),
                exclusive=True,
                group="console-attachment",
            )
            return
```

- [ ] **Step 4: Run tests**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_native_chat_flow.py -q --no-header` (full file)
Expected: all pass (pre-existing + 2 new). Verify append-only.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_native_chat_flow.py
git commit -m "feat(console): dropped file paths auto-attach from paste

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Alt+V clipboard grab

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (BINDINGS list at ~403-443; new action + worker near `_process_console_attachment`)
- Modify: `Docs/superpowers/specs/2026-07-13-console-paste-dnd-attach-design.md` (spec amendment per Global Constraints)
- Test: `Tests/UI/test_console_native_chat_flow.py` (append-only)

**Interfaces:**
- Consumes: `grab_clipboard_image`, `ClipboardGrab` (Task 1); `process_attachment_bytes` (Task 2); `extract_dropped_path`/`looks_attachable` route from Task 3; composer `set_pending_attachment_label`; store pending methods.
- Produces: `action_paste_clipboard_image()` (binding target), `async _paste_console_clipboard_image()` (worker body, group `"console-clipboard-grab"`).

- [ ] **Step 1: Write the failing tests** (append at true EOF)

```python
async def test_alt_v_grabs_clipboard_image_into_pending(monkeypatch):
    from io import BytesIO

    from PIL import Image as PILImage

    import tldw_chatbook.UI.Screens.chat_screen as chat_screen_module

    buffer = BytesIO()
    PILImage.new("RGB", (16, 16), (10, 200, 10)).save(buffer, format="PNG")
    png = buffer.getvalue()

    from tldw_chatbook.Chat.console_paste_attach import ClipboardGrab

    monkeypatch.setattr(
        chat_screen_module,
        "grab_clipboard_image",
        lambda: ClipboardGrab(kind="image", png_bytes=png),
    )

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        console.query_one("#console-native-composer", ConsoleComposerBar).focus()
        await pilot.pause()

        await pilot.press("alt+v")
        for _ in range(80):
            store = console._ensure_console_chat_store()
            sid = store.active_session_id
            if sid and store.pending_attachment(sid) is not None:
                break
            await pilot.pause(0.05)

        store = console._ensure_console_chat_store()
        pending = store.pending_attachment(store.active_session_id)
        assert pending is not None
        assert pending.file_type == "image"
        assert pending.display_name.startswith("clipboard-")


async def test_alt_v_unavailable_platform_toasts(monkeypatch):
    import tldw_chatbook.UI.Screens.chat_screen as chat_screen_module

    from tldw_chatbook.Chat.console_paste_attach import ClipboardGrab

    monkeypatch.setattr(
        chat_screen_module,
        "grab_clipboard_image",
        lambda: ClipboardGrab(kind="unavailable"),
    )

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    notifications: list[str] = []
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        monkeypatch.setattr(
            console.app_instance,
            "notify",
            lambda message, **kwargs: notifications.append(str(message)),
        )
        await pilot.press("alt+v")
        for _ in range(40):
            if notifications:
                break
            await pilot.pause(0.05)
        assert any("aren't readable on this platform" in n for n in notifications)
        store = console._ensure_console_chat_store()
        sid = store.active_session_id
        assert sid is None or store.pending_attachment(sid) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_native_chat_flow.py -q --no-header -k "alt_v"`
Expected: FAIL (`grab_clipboard_image` not importable from chat_screen / binding missing).

- [ ] **Step 3: Implement**

1. Extend the chat_screen import from Task 3's line: `from tldw_chatbook.Chat.console_paste_attach import extract_dropped_path, grab_clipboard_image, looks_attachable` and add `from tldw_chatbook.Chat.attachment_core import process_attachment_bytes` (check whether chat_screen already imports from attachment_core at module level — Task 3 review may have placed imports; extend, don't duplicate).
2. BINDINGS (after the `alt+m` entry): `Binding("alt+v", "paste_clipboard_image", "Paste image", show=True),`
3. Action + worker (near `_process_console_attachment`):

```python
    def action_paste_clipboard_image(self) -> None:
        """Grab an image from the OS clipboard into the pending attachment."""
        self.run_worker(
            self._paste_console_clipboard_image(),
            exclusive=True,
            group="console-clipboard-grab",
        )

    async def _paste_console_clipboard_image(self) -> None:
        """Read the clipboard off-loop and stage its image (or route paths)."""
        from datetime import datetime as _datetime

        grab = await asyncio.to_thread(grab_clipboard_image)
        if grab.kind == "unavailable":
            self.app_instance.notify(
                "Clipboard images aren't readable on this platform — "
                "use Attach or drop a file.",
                severity="warning",
            )
            return
        if grab.kind == "empty":
            self.app_instance.notify("No image on the clipboard.")
            return
        if grab.kind == "paths":
            candidate = grab.paths[0] if grab.paths else ""
            if candidate and looks_attachable(candidate):
                if len(grab.paths) > 1:
                    self.app_instance.notify(
                        f"Attached first of {len(grab.paths)} dropped files."
                    )
                await self._process_console_attachment(candidate)
            else:
                self.app_instance.notify("No image on the clipboard.")
            return
        try:
            display_name = (
                f"clipboard-{_datetime.now().strftime('%Y%m%d-%H%M%S')}.png"
            )
            attachment = await asyncio.to_thread(
                lambda: asyncio.run(
                    process_attachment_bytes(
                        grab.png_bytes or b"", display_name=display_name
                    )
                )
            )
        except Exception as exc:
            logger.opt(exception=True).warning("Clipboard image processing failed.")
            self.app_instance.notify(
                f"Could not attach clipboard image: {escape_markup(str(exc))}",
                severity="error",
            )
            return
        store = self._ensure_console_chat_store()
        session = store.ensure_session(
            workspace_id=store.workspace_context.active_workspace_id
        )
        store.set_pending_attachment(session.id, attachment)
        composer = self._console_composer_or_none()
        if composer is not None:
            composer.set_pending_attachment_label(attachment.label)
        self.app_instance.notify(
            f"{escape_markup(attachment.display_name)} attached"
        )
        self._sync_console_control_bar()
```

(Verify `escape_markup` is the file's existing escape helper name — match whatever the Phase-1 toasts use.)
4. Spec amendment: in the spec's Decisions table, change the Image-trigger row's "…+ a command-palette entry" to "…footer-visible binding (show=True) — the app's ^p palette is a custom app-level modal, not Textual's CommandPalette; the footer hint is the same discoverability mechanism alt+m uses (approved deviation)."

- [ ] **Step 4: Run tests**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_native_chat_flow.py -q --no-header` (full file), then `Tests/Chat/ -q --no-header`.
Expected: all pass. Append-only verified.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_native_chat_flow.py "Docs/superpowers/specs/2026-07-13-console-paste-dnd-attach-design.md"
git commit -m "feat(console): Alt+V grabs clipboard images into the attach pipeline

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Full verification + visual QA gate

**Files:** none expected (fix-forward only).

- [ ] **Step 1: Full affected surface**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Chat/ \
  Tests/UI/test_console_native_chat_flow.py \
  Tests/UI/test_console_native_transcript.py \
  Tests/UI/test_console_internals_decomposition.py \
  Tests/UI/test_chat_image_attachment.py \
  Tests/Event_Handlers/Chat_Events/test_chat_image_events.py \
  Tests/Event_Handlers/Chat_Events/test_chat_image_properties.py \
  Tests/unit/test_chat_image_unit.py \
  Tests/DB/test_chat_image_db_compatibility.py \
  Tests/Widgets/test_chat_message_enhanced.py \
  -q --no-header
```
Expected: 0 real failures (cursor-blink flakes pass isolated if they appear).

- [ ] **Step 2: Live QA captures** — textual-serve rig per the console-inline-images README recipe. Into `Docs/superpowers/qa/console-paste-dnd-2026-07/`:
  1. `drop-path-attached.png` — a real bracketed paste of an image path into the focused composer → 📎 indicator staged, draft empty.
  2. `drop-md-inlined.png` — pasted .md path → collapsed 📄 segment.
  3. `prose-with-path-stays-text.png` — "what does /etc/hosts do?" pasted → lands as draft text.
  4. `multi-drop-first-of-n.png` — two newline-separated paths pasted → first attached, "Attached first of 2 dropped files." toast.
  5. `alt-v-attached.png` — Alt+V with a seeded clipboard grab → clipboard-*.png staged. Under headless serve the OS clipboard is unreachable: monkeypatch/seed the grab in the serve process (e.g. sitecustomize or env-guarded stub) and DISCLOSE it in the README — the mounted tests cover the real ImageGrab seam; the capture proves the UX.
  6. `alt-v-unavailable-toast.png` — grab stubbed unavailable → the honest platform toast.
  Write README (rig, commit, capture stories, the clipboard-stub disclosure), commit evidence.
- [ ] **Step 3: Visual approval gate** — present captures; NO merge/PR without user approval.
- [ ] **Step 4: Wrap-up** — finishing-a-development-branch: TASK-216 backlog file ACs checked + Implementation Notes + Done (riding the branch), PR to dev on approval.

---

## Deferred (do not implement)

Multi-attachment staging (TASK-217); Linux clipboard backends (xclip/wl-paste); filter/caps config unification (TASK-222); legacy chat paste behavior.
