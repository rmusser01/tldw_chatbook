# Console Chat Native Attachments & Image Support — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the native Console chat a working attach flow (all legacy file types), image send to vision providers, DB persistence/resume, and a placeholder image chip + Save Image action in the transcript — by extracting the legacy attachment logic into a shared UI-agnostic core.

**Architecture:** New `tldw_chatbook/Chat/attachment_core.py` holds validation, file processing (via existing `file_handler_registry`/`ChatImageHandler`), and vision gating. The legacy `ChatAttachmentHandler` becomes a thin adapter over the core (zero behavior change). The Console gets per-session pending-attachment state in `ConsoleChatStore`, multimodal provider payloads in `ConsoleChatController`, native picker wiring + send gating in `ChatScreen`, a labeled collapsed-segment API + attachment indicator in `ConsoleComposerBar`, and a chip + Save Image action in the transcript/action service.

**Tech Stack:** Python ≥3.11, Textual, pytest, PIL/Pillow. No new dependencies.

**Spec:** `Docs/superpowers/specs/2026-07-12-console-chat-attachments-phase1-design.md` — read it before starting. All decisions there are settled; do not relitigate.

## Global Constraints

- Run all tests with `.venv/bin/python -m pytest <paths> -q --no-header` from the repo root (venv-only pytest; system python lacks deps).
- Execute in an isolated git worktree off `origin/dev` (other sessions mutate this checkout) — use superpowers:using-git-worktrees before Task 1.
- **Legacy regression gate — do NOT edit these files** (needing to is a design smell; stop and escalate): `Tests/UI/test_chat_image_attachment.py`, `Tests/Event_Handlers/Chat_Events/test_chat_image_events.py`, `Tests/Event_Handlers/Chat_Events/test_chat_image_properties.py`, `Tests/unit/test_chat_image_unit.py`, `Tests/DB/test_chat_image_db_compatibility.py`, `Tests/Widgets/test_chat_message_enhanced.py`. Baseline verified green 2026-07-12 (81 passed, 1 skipped).
- Legacy chat behavior is unchanged: keep warn-and-drop send semantics, keep `pending_image`/`pending_attachment` reactives, keep 100 MB / 10 MB caps.
- Constants (exact values): `MAX_ATTACHMENT_BYTES = 100 * 1024 * 1024`, `DEFAULT_MAX_HISTORY_IMAGES = 10`.
- Never serialize raw image bytes into screen-state payloads (`_serialize_console_message` gets `image_mime_type`/`attachment_label` metadata only).
- CI checks are intentionally cancelled remotely — verify locally, don't wait on CI.
- End every commit message with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Shared attachment core module

**Files:**
- Create: `tldw_chatbook/Chat/attachment_core.py`
- Test: `Tests/Chat/test_attachment_core.py` (new)

**Interfaces:**
- Consumes: `tldw_chatbook.Utils.file_handlers.file_handler_registry.process_file(path) -> ProcessedFile` (async; fields `content`, `attachment_data`, `attachment_mime_type`, `display_name`, `insert_mode`, `file_type`), `tldw_chatbook.Utils.path_validation.is_safe_path`, `tldw_chatbook.model_capabilities.is_vision_capable(provider, model)` and `get_model_capabilities()` (registry instance with `.get_model_capabilities(provider, model) -> dict`).
- Produces (used by Tasks 2–9):
  - `@dataclass PendingAttachment(file_path: str, display_name: str, file_type: str, insert_mode: Literal["inline","attachment"], data: bytes | None, mime_type: str | None, text_content: str | None, original_size: int, processed_size: int)` with property `label -> str` (e.g. `"photo.png · 240 KB"`).
  - `async load_processed_file(file_path: str, *, allowed_root: str | None = None) -> ProcessedFile`
  - `async process_attachment_path(file_path: str, *, allowed_root: str | None = None) -> PendingAttachment`
  - `vision_block_reason(provider: str, model: str | None) -> str | None`
  - `max_history_images(provider: str, model: str | None) -> int`
  - `image_content_parts(text: str, image_data: bytes, mime_type: str) -> list[dict[str, Any]]`
  - Constants: `MAX_ATTACHMENT_BYTES`, `DEFAULT_MAX_HISTORY_IMAGES`, `ATTACHMENT_FILTER_SPECS: tuple[tuple[str, str], ...]`

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Chat/test_attachment_core.py
import asyncio

import pytest
from PIL import Image as PILImage

from tldw_chatbook.Chat import attachment_core
from tldw_chatbook.Chat.attachment_core import (
    DEFAULT_MAX_HISTORY_IMAGES,
    PendingAttachment,
    image_content_parts,
    max_history_images,
    process_attachment_path,
    vision_block_reason,
)


def _write_png(path, size=(4, 4)):
    PILImage.new("RGB", size, color=(200, 10, 10)).save(path, format="PNG")


def test_process_attachment_path_rejects_paths_outside_allowed_root(tmp_path):
    outside = tmp_path / "evil.txt"
    outside.write_text("nope")
    with pytest.raises(ValueError, match="outside allowed directories"):
        asyncio.run(
            process_attachment_path(str(outside), allowed_root=str(tmp_path / "inner"))
        )


def test_process_attachment_path_rejects_oversized_files(tmp_path, monkeypatch):
    big = tmp_path / "big.txt"
    big.write_text("x" * 64)
    monkeypatch.setattr(attachment_core, "MAX_ATTACHMENT_BYTES", 16)
    with pytest.raises(ValueError, match="File too large"):
        asyncio.run(process_attachment_path(str(big), allowed_root=str(tmp_path)))


def test_process_attachment_path_inlines_text_files(tmp_path):
    note = tmp_path / "notes.md"
    note.write_text("# hello\nworld")
    attachment = asyncio.run(
        process_attachment_path(str(note), allowed_root=str(tmp_path))
    )
    assert attachment.insert_mode == "inline"
    assert attachment.file_type == "text"
    assert "world" in (attachment.text_content or "")
    assert attachment.data is None
    assert attachment.display_name == "notes.md"
    assert attachment.label.startswith("notes.md · ")


def test_process_attachment_path_attaches_images(tmp_path):
    image = tmp_path / "photo.png"
    _write_png(image)
    attachment = asyncio.run(
        process_attachment_path(str(image), allowed_root=str(tmp_path))
    )
    assert attachment.insert_mode == "attachment"
    assert attachment.file_type == "image"
    assert isinstance(attachment.data, bytes) and attachment.data
    assert attachment.mime_type == "image/png"
    assert attachment.processed_size == len(attachment.data)


def test_vision_block_reason_none_for_vision_model():
    assert vision_block_reason("OpenAI", "gpt-4o") is None


def test_vision_block_reason_names_model_and_override():
    reason = vision_block_reason("llama_cpp", "text-model-7b")
    assert reason is not None
    assert "text-model-7b" in reason
    assert "can't accept images" in reason
    assert "[model_capabilities.models]" in reason


def test_max_history_images_uses_capability_value_and_default():
    assert max_history_images("Anthropic", "claude-3-opus") == 5  # direct mapping
    assert max_history_images("FakeProv", "fake-model") == DEFAULT_MAX_HISTORY_IMAGES
    assert max_history_images("OpenAI", None) == DEFAULT_MAX_HISTORY_IMAGES


def test_image_content_parts_builds_data_url():
    parts = image_content_parts("look", b"\x89PNG", "image/png")
    assert parts[0] == {"type": "text", "text": "look"}
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    only_image = image_content_parts("", b"\x89PNG", "image/png")
    assert [p["type"] for p in only_image] == ["image_url"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest Tests/Chat/test_attachment_core.py -q --no-header`
Expected: FAIL / errors with `ModuleNotFoundError: No module named 'tldw_chatbook.Chat.attachment_core'`

- [ ] **Step 3: Write the module**

```python
# tldw_chatbook/Chat/attachment_core.py
"""Shared, UI-agnostic attachment processing for legacy chat and native Console.

Extracted from ChatAttachmentHandler so both the legacy chat window and the
native Console consume one validation/processing/vision-gating pipeline.
No Textual imports allowed in this module.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from loguru import logger

from tldw_chatbook.Utils.file_handlers import ProcessedFile, file_handler_registry
from tldw_chatbook.Utils.path_validation import is_safe_path
from tldw_chatbook.model_capabilities import (
    get_model_capabilities as _get_capabilities_registry,
    is_vision_capable,
)

MAX_ATTACHMENT_BYTES = 100 * 1024 * 1024  # matches the legacy handler's 100MB cap
DEFAULT_MAX_HISTORY_IMAGES = 10  # used when model capabilities omit max_images

# (label, semicolon-separated glob patterns) — single source for both UIs' pickers.
ATTACHMENT_FILTER_SPECS: tuple[tuple[str, str], ...] = (
    ("All Supported Files", "*.png;*.jpg;*.jpeg;*.gif;*.webp;*.bmp;*.tiff;*.tif;*.svg;*.txt;*.md;*.log;*.py;*.js;*.ts;*.java;*.cpp;*.c;*.h;*.cs;*.rb;*.go;*.rs;*.json;*.yaml;*.yml;*.csv;*.tsv;*.pdf;*.doc;*.docx;*.rtf;*.odt;*.epub;*.mobi;*.azw;*.azw3;*.fb2"),
    ("Image Files", "*.png;*.jpg;*.jpeg;*.gif;*.webp;*.bmp;*.tiff;*.tif;*.svg"),
    ("Document Files", "*.pdf;*.doc;*.docx;*.rtf;*.odt"),
    ("E-book Files", "*.epub;*.mobi;*.azw;*.azw3;*.fb2"),
    ("Text Files", "*.txt;*.md;*.log;*.text;*.rst"),
    ("Code Files", "*.py;*.js;*.ts;*.java;*.cpp;*.c;*.h;*.cs;*.rb;*.go;*.rs;*.swift;*.kt;*.php;*.r;*.m;*.lua;*.sh;*.bash;*.ps1;*.sql;*.html;*.css;*.xml"),
    ("Data Files", "*.json;*.yaml;*.yml;*.csv;*.tsv"),
)


def _format_size(size: int) -> str:
    if size >= 1024 * 1024:
        return f"{size / 1024 / 1024:.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.0f} KB"
    return f"{size} B"


@dataclass
class PendingAttachment:
    """One processed, not-yet-sent attachment staged on a chat session."""

    file_path: str
    display_name: str
    file_type: str
    insert_mode: Literal["inline", "attachment"]
    data: bytes | None = None
    mime_type: str | None = None
    text_content: str | None = None
    original_size: int = 0
    processed_size: int = 0

    @property
    def label(self) -> str:
        """Return the user-facing chip/indicator label."""
        size = self.processed_size or self.original_size
        return f"{self.display_name} · {_format_size(size)}"


async def load_processed_file(
    file_path: str,
    *,
    allowed_root: str | None = None,
) -> ProcessedFile:
    """Validate and process a file attachment (moved intact from ChatAttachmentHandler)."""
    root = allowed_root or os.path.expanduser("~")
    logger.info(f"Processing file attachment: {file_path}")
    if not is_safe_path(file_path, root):
        raise ValueError("File path is outside allowed directories")
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    file_size = os.path.getsize(file_path)
    if file_size > MAX_ATTACHMENT_BYTES:
        raise ValueError(
            f"File too large: {file_size / 1024 / 1024:.1f}MB "
            f"(max {MAX_ATTACHMENT_BYTES / 1024 / 1024:.0f}MB)"
        )
    return await file_handler_registry.process_file(file_path)


async def process_attachment_path(
    file_path: str,
    *,
    allowed_root: str | None = None,
) -> PendingAttachment:
    """Validate, process, and normalize a file into a PendingAttachment."""
    processed = await load_processed_file(file_path, allowed_root=allowed_root)
    data = processed.attachment_data
    return PendingAttachment(
        file_path=str(file_path),
        display_name=processed.display_name or Path(file_path).name,
        file_type=processed.file_type,
        insert_mode=processed.insert_mode,
        data=data,
        mime_type=processed.attachment_mime_type,
        text_content=processed.content,
        original_size=os.path.getsize(file_path),
        processed_size=len(data) if data is not None else len(processed.content or ""),
    )


def vision_block_reason(provider: str, model: str | None) -> str | None:
    """Return user-facing blocked-send copy when the model can't accept images."""
    if model and is_vision_capable(provider, model):
        return None
    model_label = model or "the selected model"
    return (
        f"Console send blocked: {model_label} can't accept images. "
        "Remove the attachment, switch to a vision model, or mark this model as "
        "vision-capable under [model_capabilities.models] in config.toml."
    )


def max_history_images(provider: str, model: str | None) -> int:
    """Return how many recent session images to resend for this model."""
    if not model:
        return DEFAULT_MAX_HISTORY_IMAGES
    capabilities = _get_capabilities_registry().get_model_capabilities(provider, model)
    value = capabilities.get("max_images")
    if isinstance(value, int) and value > 0:
        return value
    return DEFAULT_MAX_HISTORY_IMAGES


def image_content_parts(
    text: str,
    image_data: bytes,
    mime_type: str,
) -> list[dict[str, Any]]:
    """Build OpenAI-style multimodal content parts with a base64 data URL."""
    encoded = base64.b64encode(image_data).decode("ascii")
    parts: list[dict[str, Any]] = []
    if text:
        parts.append({"type": "text", "text": text})
    parts.append(
        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}}
    )
    return parts
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest Tests/Chat/test_attachment_core.py -q --no-header`
Expected: all PASS. Note: `test_vision_block_reason_none_for_vision_model` relies on the `gpt-4o` direct capability mapping; if it fails, check `model_capabilities.py` DEFAULT_MODEL_CAPABILITIES still maps `gpt-4o` → vision.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/attachment_core.py Tests/Chat/test_attachment_core.py
git commit -m "feat(chat): add shared UI-agnostic attachment core"
```

---

### Task 2: Legacy handler consumes the core (behavior-identical refactor)

**Files:**
- Modify: `tldw_chatbook/UI/Chat_Modules/chat_attachment_handler.py` (`_load_processed_file` at lines 151–171, `handle_attach_image_button` filter block at lines 79–88)

**Interfaces:**
- Consumes: `attachment_core.load_processed_file`, `attachment_core.ATTACHMENT_FILTER_SPECS` (Task 1).
- Produces: nothing new — external behavior of `ChatAttachmentHandler` is unchanged.

- [ ] **Step 1: Replace `_load_processed_file` body with a core delegation**

Replace lines 151–171 (`async def _load_processed_file` through `return await file_handler_registry.process_file(file_path)`) with:

```python
    async def _load_processed_file(self, file_path: str) -> Any:
        """Validate and process a file attachment via the shared core."""
        from ...Chat.attachment_core import load_processed_file

        return await load_processed_file(file_path)
```

Also delete the now-unused `import os` **only if** nothing else in the file uses `os` (grep first — `_process_file_worker` and others may not; if unsure, leave it).

- [ ] **Step 2: Replace the inline filter table with the shared specs**

In `handle_attach_image_button`, replace the `file_filters = Filters(...)` block (lines 79–88) with:

```python
        from ...Chat.attachment_core import ATTACHMENT_FILTER_SPECS

        file_filters = Filters(
            *[(label, create_filter(patterns)) for label, patterns in ATTACHMENT_FILTER_SPECS],
            ("All Files", lambda path: True),
        )
```

The specs are copied verbatim from this file, so picker behavior is identical.

- [ ] **Step 3: Run the legacy regression gate**

Run: `.venv/bin/python -m pytest Tests/UI/test_chat_image_attachment.py Tests/Event_Handlers/Chat_Events/test_chat_image_events.py Tests/Event_Handlers/Chat_Events/test_chat_image_properties.py Tests/unit/test_chat_image_unit.py Tests/DB/test_chat_image_db_compatibility.py Tests/Widgets/test_chat_message_enhanced.py -q --no-header`
Expected: 81 passed, 1 skipped (same as baseline). If any test fails, the refactor changed behavior — fix the handler, never the tests.

- [ ] **Step 4: Commit**

```bash
git add tldw_chatbook/UI/Chat_Modules/chat_attachment_handler.py
git commit -m "refactor(chat): legacy attachment handler delegates to shared core"
```

---

### Task 3: Console models + store — image fields, pending attachment, image-aware persistence

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_models.py` (`ConsoleChatMessage` at lines 177–188)
- Modify: `tldw_chatbook/Chat/console_chat_store.py` (`ConsoleChatSession` at 72–82, `append_message` at 313–333, `_persist_new_message_or_defer` at 505–512, `_persist_new_message` at 514–531, `_persist_existing_message` at 533–554)
- Test: `Tests/Chat/test_console_chat_store.py` (append new tests; do not modify existing ones)

**Interfaces:**
- Consumes: `attachment_core.PendingAttachment` (Task 1).
- Produces (used by Tasks 4, 7–9):
  - `ConsoleChatMessage` gains fields `image_data: bytes | None = None`, `image_mime_type: str | None = None`, `attachment_label: str | None = None`.
  - `ConsoleChatSession` gains field `pending_attachment: PendingAttachment | None = None`.
  - Store methods: `pending_attachment(session_id) -> PendingAttachment | None`, `set_pending_attachment(session_id, attachment) -> ConsoleChatSession`, `clear_pending_attachment(session_id) -> ConsoleChatSession`.
  - `append_message(session_id, *, role, content, persist=False, image_data=None, image_mime_type=None, attachment_label=None) -> ConsoleChatMessage`.

- [ ] **Step 1: Write the failing tests** (append to `Tests/Chat/test_console_chat_store.py`)

```python
from tldw_chatbook.Chat.attachment_core import PendingAttachment


def _image_attachment(name="photo.png"):
    return PendingAttachment(
        file_path=f"/tmp/{name}",
        display_name=name,
        file_type="image",
        insert_mode="attachment",
        data=b"\x89PNG-bytes",
        mime_type="image/png",
        original_size=11,
        processed_size=11,
    )


class RecordingPersistence:
    def __init__(self):
        self.created = []
        self.updated = []
        self._counter = 0

    def create_conversation(self, **kwargs):
        return "conv-1"

    def create_message(self, **kwargs):
        self.created.append(kwargs)
        self._counter += 1
        return f"msg-{self._counter}"

    def update_message_content(self, **kwargs):
        self.updated.append(kwargs)
        return True


def test_pending_attachment_is_per_session():
    store = ConsoleChatStore()
    first = store.create_session(title="A")
    second = store.create_session(title="B")

    store.set_pending_attachment(first.id, _image_attachment())

    assert store.pending_attachment(first.id) is not None
    assert store.pending_attachment(second.id) is None

    store.clear_pending_attachment(first.id)
    assert store.pending_attachment(first.id) is None


def test_append_message_persists_image_fields():
    persistence = RecordingPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session()

    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="what is this?",
        image_data=b"\x89PNG-bytes",
        image_mime_type="image/png",
        attachment_label="photo.png · 11 B",
        persist=True,
    )

    assert message.image_data == b"\x89PNG-bytes"
    assert message.attachment_label == "photo.png · 11 B"
    assert persistence.created[-1]["image_data"] == b"\x89PNG-bytes"
    assert persistence.created[-1]["image_mime_type"] == "image/png"


def test_image_only_user_message_persists_immediately():
    persistence = RecordingPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session()

    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="",
        image_data=b"\x89PNG-bytes",
        image_mime_type="image/png",
        persist=True,
    )

    assert len(persistence.created) == 1
    assert persistence.created[0]["content"] == ""
    assert persistence.created[0]["image_data"] == b"\x89PNG-bytes"


def test_editing_message_content_does_not_wipe_persisted_image():
    persistence = RecordingPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session()
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="original",
        image_data=b"\x89PNG-bytes",
        image_mime_type="image/png",
        persist=True,
    )

    store.update_message_content(message.id, "edited")

    assert persistence.updated[-1]["image_data"] == b"\x89PNG-bytes"
    assert persistence.updated[-1]["image_mime_type"] == "image/png"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest Tests/Chat/test_console_chat_store.py -q --no-header`
Expected: the 4 new tests FAIL (`AttributeError`/`TypeError` for missing fields/methods); every pre-existing test still passes.

- [ ] **Step 3: Implement**

In `console_chat_models.py`, extend `ConsoleChatMessage` (after `feedback`):

```python
@dataclass
class ConsoleChatMessage:
    """A native Console transcript message."""

    role: ConsoleMessageRole
    content: str
    id: str = field(default_factory=lambda: str(uuid4()))
    turn_id: str | None = None
    status: ConsoleMessageStatus = "complete"
    persisted_message_id: str | None = None
    variants: "ConsoleVariantSet | None" = None
    feedback: ConsoleMessageFeedback | None = None
    image_data: bytes | None = None
    image_mime_type: str | None = None
    attachment_label: str | None = None
```

In `console_chat_store.py`:

1. Add import: `from tldw_chatbook.Chat.attachment_core import PendingAttachment` (no cycle: attachment_core imports only Utils + model_capabilities).
2. Extend `ConsoleChatSession` with `pending_attachment: PendingAttachment | None = None` (after `draft`).
3. Add accessor methods after `set_session_draft` (line 268):

```python
    def pending_attachment(self, session_id: str) -> PendingAttachment | None:
        """Return the staged, not-yet-sent attachment for a session."""
        return self._session_or_raise(session_id).pending_attachment

    def set_pending_attachment(
        self,
        session_id: str,
        attachment: PendingAttachment,
    ) -> ConsoleChatSession:
        """Stage an attachment on a session, replacing any previous one."""
        session = self._session_or_raise(session_id)
        session.pending_attachment = attachment
        return session

    def clear_pending_attachment(self, session_id: str) -> ConsoleChatSession:
        """Remove the staged attachment from a session."""
        session = self._session_or_raise(session_id)
        session.pending_attachment = None
        return session
```

4. Extend `append_message`:

```python
    def append_message(
        self,
        session_id: str,
        *,
        role: ConsoleMessageRole,
        content: str,
        persist: bool = False,
        image_data: bytes | None = None,
        image_mime_type: str | None = None,
        attachment_label: str | None = None,
    ) -> ConsoleChatMessage:
        """Append a message to a session and optionally persist it."""
        self._session_or_raise(session_id)
        message = ConsoleChatMessage(
            role=role,
            content=content,
            status=self._initial_status(role=role, content=content),
            image_data=image_data,
            image_mime_type=image_mime_type,
            attachment_label=attachment_label,
        )
        self._messages_by_session[session_id].append(message)
        self._sessions[session_id].updated_at = _utc_now_iso()
        self._message_session_index[message.id] = session_id
        if persist:
            self._persist_new_message_or_defer(session_id=session_id, message=message)
        return self._snapshot(message)
```

5. In `_persist_new_message_or_defer`, change the defer condition so image-only messages persist immediately:

```python
        if not message.content and message.image_data is None:
            self._pending_persistence_message_ids.add(message.id)
            self.persist_session_if_needed(session_id)
            return
```

6. In `_persist_new_message`, replace `image_data=None, image_mime_type=None` with:

```python
            image_data=message.image_data,
            image_mime_type=message.image_mime_type,
```

7. In `_persist_existing_message`, replace `image_data=None, image_mime_type=None` with the same two lines. (This is the fix for edit-wipes-image: `ChaChaNotes_DB.update_message(image_data=None)` NULLs both columns.)

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest Tests/Chat/test_console_chat_store.py -q --no-header`
Expected: all PASS (new and pre-existing).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_models.py tldw_chatbook/Chat/console_chat_store.py Tests/Chat/test_console_chat_store.py
git commit -m "feat(console): per-session pending attachments and image-aware persistence"
```

---

### Task 4: Controller — multimodal payloads, image-only send, vision block, history cap

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` (`submit_draft` 98–137, `_validated_draft` 469–483, `_provider_messages_for_session` 616–633, `_provider_messages_through_message` 635–651, `_ensure_user_continuation_instruction` 457–467, `ConsoleProviderGatewayProtocol.stream_chat` type hint at 32)
- Test: `Tests/Chat/test_console_chat_controller.py` (append)

**Interfaces:**
- Consumes: store methods + message fields from Task 3; `attachment_core.image_content_parts`, `max_history_images`, `vision_block_reason`; `model_capabilities.is_vision_capable`.
- Produces: provider message payloads are now `list[dict[str, Any]]` where `content` is either `str` or the parts list from `image_content_parts`. `submit_draft` accepts an empty draft when a pending image attachment exists, blocks (with `vision_block_reason` copy) when the model isn't vision-capable, and clears the pending attachment after appending the user message. Monkeypatch seams for tests: module attributes `is_vision_capable`, `max_history_images` on `console_chat_controller`.

- [ ] **Step 1: Write the failing tests** (append to `Tests/Chat/test_console_chat_controller.py`; reuse the existing `RecordingStreamingGateway` and `BlockedGateway` classes defined at the top of that file)

```python
from tldw_chatbook.Chat import console_chat_controller as controller_module
from tldw_chatbook.Chat.attachment_core import PendingAttachment


def _pending_image(name="photo.png", data=b"\x89PNG-bytes"):
    return PendingAttachment(
        file_path=f"/tmp/{name}",
        display_name=name,
        file_type="image",
        insert_mode="attachment",
        data=data,
        mime_type="image/png",
        original_size=len(data),
        processed_size=len(data),
    )


def test_submit_draft_sends_image_parts_when_vision_capable(monkeypatch):
    monkeypatch.setattr(controller_module, "is_vision_capable", lambda p, m: True)
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway, model="vision-model")
    session = store.ensure_session()
    store.set_pending_attachment(session.id, _pending_image())

    result = asyncio.run(controller.submit_draft("what is this?"))

    assert result.accepted
    user_payload = gateway.messages_seen[-1]
    assert user_payload["role"] == "user"
    assert isinstance(user_payload["content"], list)
    assert user_payload["content"][0] == {"type": "text", "text": "what is this?"}
    assert user_payload["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert store.pending_attachment(session.id) is None  # consumed on send


def test_submit_draft_blocks_pending_image_on_non_vision_model(monkeypatch):
    monkeypatch.setattr(controller_module, "is_vision_capable", lambda p, m: False)
    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store, provider_gateway=RecordingStreamingGateway(), model="text-model"
    )
    session = store.ensure_session()
    store.set_pending_attachment(session.id, _pending_image())

    result = asyncio.run(controller.submit_draft("look at this"))

    assert not result.accepted
    assert "can't accept images" in result.visible_copy
    assert store.pending_attachment(session.id) is not None  # kept for model switch


def test_image_only_draft_is_sendable(monkeypatch):
    monkeypatch.setattr(controller_module, "is_vision_capable", lambda p, m: True)
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway, model="vision-model")
    session = store.ensure_session()
    store.set_pending_attachment(session.id, _pending_image())

    result = asyncio.run(controller.submit_draft(""))

    assert result.accepted
    user_payload = gateway.messages_seen[-1]
    assert [part["type"] for part in user_payload["content"]] == ["image_url"]


def test_history_images_capped_to_most_recent(monkeypatch):
    monkeypatch.setattr(controller_module, "is_vision_capable", lambda p, m: True)
    monkeypatch.setattr(controller_module, "max_history_images", lambda p, m: 1)
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway, model="vision-model")
    session = store.ensure_session()
    store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="first",
        image_data=b"img-1", image_mime_type="image/png",
    )
    store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="second",
        image_data=b"img-2", image_mime_type="image/png",
    )

    asyncio.run(controller.submit_draft("and now?"))

    contents = [m["content"] for m in gateway.messages_seen if m["role"] == "user"]
    assert contents[0] == "first"           # over budget → text only
    assert isinstance(contents[1], list)    # most recent image kept
    assert contents[2] == "and now?"


def test_non_vision_history_stays_plain_strings(monkeypatch):
    monkeypatch.setattr(controller_module, "is_vision_capable", lambda p, m: False)
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway, model="text-model")
    session = store.ensure_session()
    store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="had an image",
        image_data=b"img-1", image_mime_type="image/png",
    )

    asyncio.run(controller.submit_draft("plain follow-up"))

    for message in gateway.messages_seen:
        assert isinstance(message["content"], str)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest Tests/Chat/test_console_chat_controller.py -q --no-header`
Expected: the 5 new tests FAIL (missing import seams / TypeErrors); pre-existing tests pass.

- [ ] **Step 3: Implement**

1. Add imports near the top of `console_chat_controller.py`:

```python
from tldw_chatbook.Chat.attachment_core import (
    image_content_parts,
    max_history_images,
    vision_block_reason,
)
from tldw_chatbook.model_capabilities import is_vision_capable
```

2. Widen type hints: `ConsoleProviderGatewayProtocol.stream_chat(..., messages: list[dict[str, Any]])`; `_ensure_user_continuation_instruction(provider_messages: list[dict[str, Any]])` (logic unchanged); return types of both `_provider_messages_*` builders → `list[dict[str, Any]]`.

3. `_validated_draft` gains `allow_empty`:

```python
    @staticmethod
    def _validated_draft(draft: str, *, allow_empty: bool = False) -> tuple[str, str | None]:
        raw_draft = str(draft or "")
        if not raw_draft.strip():
            if allow_empty:
                return "", None
            return "", "Type a message before sending."
        if not validate_text_input(
            raw_draft,
            max_length=MAX_CONSOLE_DRAFT_LENGTH,
            allow_html=False,
        ):
            return "", "Message blocked: remove unsafe markup or shorten your message."
        clean_draft = sanitize_string(raw_draft, max_length=MAX_CONSOLE_DRAFT_LENGTH)
        if not clean_draft.strip():
            if allow_empty:
                return "", None
            return "", "Type a message before sending."
        return clean_draft, None
```

4. In `submit_draft`, after `session = self.store.ensure_session(...)` replace the draft-validation and user-append block:

```python
        pending = self.store.pending_attachment(session.id)
        pending_image = (
            pending
            if pending is not None
            and pending.insert_mode == "attachment"
            and pending.data is not None
            else None
        )
        clean_draft, validation_error = self._validated_draft(
            draft, allow_empty=pending_image is not None
        )
        if validation_error is not None:
            return self._block(session.id, validation_error)
        if pending_image is not None:
            block_reason = vision_block_reason(
                self.provider, self.model or self.configured_model
            )
            if block_reason is not None:
                return self._block(session.id, block_reason)
        if self.store.workspace_context.has_policy_blocks:
            return self._block(session.id, self.store.workspace_context.recovery_copy)
```

and the user-message append:

```python
        self._maybe_auto_title_session(session, clean_draft)
        self.store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content=clean_draft,
            image_data=pending_image.data if pending_image is not None else None,
            image_mime_type=pending_image.mime_type if pending_image is not None else None,
            attachment_label=pending_image.label if pending_image is not None else None,
            persist=self.store.persistence is not None,
        )
        if pending_image is not None:
            self.store.clear_pending_attachment(session.id)
```

5. Replace both payload builders with a shared helper (preserving each one's exact skip semantics — the through-message variant does NOT skip failed messages and uses variant-selected content):

```python
    def _provider_messages_for_session(
        self,
        session_id: str,
        *,
        before_message_id: str | None = None,
    ) -> list[dict[str, Any]]:
        collected: list[ConsoleChatMessage] = []
        for message in self.store.messages_for_session(session_id):
            if message.id == before_message_id:
                break
            collected.append(message)
        return self._provider_message_payloads(collected, skip_failed=True)

    def _provider_messages_through_message(
        self,
        session_id: str,
        message_id: str,
    ) -> list[dict[str, Any]]:
        collected: list[ConsoleChatMessage] = []
        for message in self.store.messages_for_session(session_id):
            collected.append(message)
            if message.id == message_id:
                break
        return self._provider_message_payloads(
            collected, skip_failed=False, use_variant_content=True
        )

    def _provider_message_payloads(
        self,
        session_messages: list[ConsoleChatMessage],
        *,
        skip_failed: bool,
        use_variant_content: bool = False,
    ) -> list[dict[str, Any]]:
        model = self.model or self.configured_model
        vision = bool(model) and is_vision_capable(self.provider, model or "")
        image_budget = max_history_images(self.provider, model)
        image_ids = [
            message.id
            for message in session_messages
            if message.role is ConsoleMessageRole.USER and message.image_data is not None
        ]
        allowed_image_ids = set(image_ids[-image_budget:]) if vision else set()
        payloads: list[dict[str, Any]] = []
        for message in session_messages:
            if message.role not in {ConsoleMessageRole.USER, ConsoleMessageRole.ASSISTANT}:
                continue
            if skip_failed and message.status == "failed":
                continue
            text = (
                message.variants.current.content
                if use_variant_content and message.variants is not None
                else message.content
            )
            if (
                message.id in allowed_image_ids
                and message.image_data is not None
                and message.image_mime_type
            ):
                payloads.append(
                    {
                        "role": message.role.value,
                        "content": image_content_parts(
                            text, message.image_data, message.image_mime_type
                        ),
                    }
                )
                continue
            if not text:
                continue
            payloads.append({"role": message.role.value, "content": text})
        return payloads
```

Import `ConsoleChatMessage` into the module's existing `console_chat_store` import line if not already importable (`from tldw_chatbook.Chat.console_chat_models import ... ConsoleChatMessage` — check the existing import list first).

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_chat_store.py -q --no-header`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_controller.py Tests/Chat/test_console_chat_controller.py
git commit -m "feat(console): multimodal provider payloads with vision gating and image cap"
```

---

### Task 5: Composer — labeled collapsed segments, attach indicator, clear button

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_composer_bar.py` (`_DraftSegment` at 33–38, `_segment_display_text` at 148–155, new `insert_file_segment` after `insert_pasted_text` at 699–728, new `set_pending_attachment_label`, `compose` at 1109–1213)
- Test: `Tests/UI/test_console_native_chat_flow.py` (append)

**Interfaces:**
- Consumes: nothing new.
- Produces (used by Tasks 6–7):
  - `_DraftSegment` gains `label: str | None = None`; collapsed segments with a label display the label instead of `"Pasted Text: N Characters"`.
  - `ConsoleComposerBar.insert_file_segment(text: str, label: str) -> None` — appends an always-collapsed labeled segment; `draft_text()` still returns full text.
  - `ConsoleComposerBar.set_pending_attachment_label(label: str | None) -> None` — shows/hides `#console-attachment-indicator` (Static, `📎 {label}`), toggles `#console-clear-attachment` button (✕) visibility, sets attach button label `Attach`↔`📎✓`.
  - New widget IDs: `#console-attachment-indicator`, `#console-clear-attachment` (the screen handles the button press in Task 6).

- [ ] **Step 1: Write the failing tests** (append to `Tests/UI/test_console_native_chat_flow.py`; unmounted construction is safe — all composer query_one calls are NoMatches-guarded)

```python
def test_insert_file_segment_collapses_with_custom_label():
    composer = ConsoleComposerBar()
    composer.insert_file_segment("file body text", "📄 notes.md · 4 KB")

    assert composer.draft_text() == "file body text"
    assert composer._display_draft_text() == "📄 notes.md · 4 KB"


def test_insert_file_segment_appends_after_typed_draft():
    composer = ConsoleComposerBar()
    composer.insert_text("see attached: ")
    composer.insert_file_segment("file body", "📄 a.md · 9 B")

    assert composer.draft_text() == "see attached: file body"
    assert composer._display_draft_text() == "see attached: 📄 a.md · 9 B"


def test_paste_collapse_label_still_defaults_to_character_count():
    composer = ConsoleComposerBar(paste_collapse_threshold=5)
    composer.insert_pasted_text("0123456789")

    assert composer._display_draft_text() == "Pasted Text: 10 Characters"


async def test_attachment_indicator_visibility_follows_label():
    app = _build_test_app(screen_name="chat")
    async with app.run_test(size=(160, 48)) as pilot:
        await _wait_for_selector(pilot, "#console-native-composer")
        composer = app.screen.query_one("#console-native-composer", ConsoleComposerBar)

        composer.set_pending_attachment_label("photo.png · 240 KB")
        await pilot.pause()
        indicator = app.screen.query_one("#console-attachment-indicator", Static)
        clear_button = app.screen.query_one("#console-clear-attachment", Button)
        assert "photo.png" in str(indicator.renderable)
        assert indicator.styles.display.value != "none"
        assert clear_button.styles.display.value != "none"

        composer.set_pending_attachment_label(None)
        await pilot.pause()
        assert indicator.styles.display.value == "none"
        assert clear_button.styles.display.value == "none"
```

(`_build_test_app` and `_wait_for_selector` are already imported at the top of this test file. If `_configure_native_ready_console` is needed to dismiss the setup modal for composer queries, call it on `app` before `run_test` — follow the neighboring composer tests in this file.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest Tests/UI/test_console_native_chat_flow.py -q --no-header -k "file_segment or attachment_indicator or collapse_label"`
Expected: FAIL with `AttributeError: ... insert_file_segment` / `NoMatches` for the indicator.

- [ ] **Step 3: Implement**

1. `_DraftSegment` gains a label:

```python
@dataclass
class _DraftSegment:
    """Private composer segment with canonical payload and display state."""

    text: str
    collapse_state: _CollapseState = "literal"
    label: str | None = None
```

2. `_segment_display_text` honors it:

```python
    @staticmethod
    def _segment_display_text(segment: _DraftSegment) -> str:
        """Return display text for a single draft segment."""
        if segment.collapse_state == "collapsed":
            if segment.label:
                return segment.label
            return f"Pasted Text: {len(segment.text)} Characters"
        if segment.collapse_state == "confirm":
            return "Unfurl?"
        return segment.text
```

3. Add `insert_file_segment` directly after `insert_pasted_text`:

```python
    def insert_file_segment(self, text: str, label: str) -> None:
        """Append inlined file content as a labeled, display-collapsed segment.

        Args:
            text: Full file text that becomes part of the canonical draft.
            label: Display-only token shown in place of the text (e.g.
                ``"📄 notes.md · 4 KB"``).
        """
        if not text:
            self._sync_interaction_classes()
            self._sync_current_action_state()
            return
        if not self._segments_initialized:
            existing = self.draft_text()
            self._segments = [_DraftSegment(existing)] if existing else []
            self._segments_initialized = True
        if self._draft_selection_all:
            self._segments = []
            self._draft_selection_all = False
        self._reset_pending_unfurl_state()
        self._segments.append(
            _DraftSegment(text, collapse_state="collapsed", label=label)
        )
        self._sync_hidden_input()
        self._refresh_visible_draft()
        self._sync_interaction_classes()
        self._sync_current_action_state()
```

4. Add attachment-indicator state. In `__init__`, after `self._can_save_chatbook = False`, add `self._pending_attachment_label: str | None = None`. Add the method:

```python
    def set_pending_attachment_label(self, label: str | None) -> None:
        """Show or clear the composer's pending-attachment indicator."""
        normalized = label.strip() if label else None
        self._pending_attachment_label = normalized
        try:
            indicator = self.query_one("#console-attachment-indicator", Static)
            clear_button = self.query_one("#console-clear-attachment", Button)
            attach_button = self.query_one("#console-attach-context", Button)
            actions = self.query_one("#console-composer-actions", Horizontal)
        except NoMatches:
            return
        if normalized:
            indicator.update(escape(f"📎 {normalized}"))
            indicator.styles.display = "block"
            indicator.styles.width = "auto"
            indicator.styles.max_width = 28
            clear_button.styles.display = "block"
            actions.styles.width = 42
            actions.styles.min_width = 42
            actions.styles.max_width = 42
            attach_button.label = "📎✓"
            attach_button.tooltip = f"Attached: {normalized}. Press to replace."
        else:
            indicator.update("")
            indicator.styles.display = "none"
            indicator.styles.width = 0
            clear_button.styles.display = "none"
            actions.styles.width = 37
            actions.styles.min_width = 37
            actions.styles.max_width = 37
            attach_button.label = "Attach"
            attach_button.tooltip = (
                "Attach files or context through the active Console session."
            )
```

(`escape` is `rich.markup.escape`, already imported in this module for the status line — verify the import name at the top of the file; add `from rich.markup import escape` if absent.)

5. In `compose()`, add the indicator Static after the `recovery` Static (hidden by default):

```python
        attachment_indicator = Static(
            "",
            id="console-attachment-indicator",
            classes="console-attachment-indicator",
        )
        attachment_indicator.styles.display = "none"
        attachment_indicator.styles.width = 0
        attachment_indicator.styles.min_width = 0
        attachment_indicator.styles.height = 1
        yield attachment_indicator
```

and the clear button inside the `actions` container, right after the Attach button:

```python
            clear_attachment = self._bounded_button(
                "✕",
                width=4,
                id="console-clear-attachment",
                classes="destination-action-button console-clear-attachment-button",
                tooltip="Remove the pending attachment.",
            )
            clear_attachment.styles.display = "none"
            yield clear_attachment
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest Tests/UI/test_console_native_chat_flow.py -q --no-header`
Expected: all PASS (including the 120 pre-existing tests in this file — the paste-collapse suite proves labels didn't regress default behavior).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Console/console_composer_bar.py Tests/UI/test_console_native_chat_flow.py
git commit -m "feat(console): labeled file segments and attachment indicator in composer"
```

---

### Task 6: Screen — native attach flow (picker → worker → route)

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (`_handle_console_attach_context` at 6235–6248; new `_process_console_attachment`, `handle_console_clear_attachment`)
- Test: `Tests/UI/test_console_native_chat_flow.py` (append)

**Interfaces:**
- Consumes: `attachment_core.process_attachment_path`, `ATTACHMENT_FILTER_SPECS` (Task 1); store pending methods (Task 3); `composer.insert_file_segment` / `set_pending_attachment_label` (Task 5); existing `enhanced_file_picker.FileOpen`/`Filters`; existing `self._console_composer_or_none()`, `self._sync_console_control_bar()`.
- Produces: `_process_console_attachment(file_path: str) -> None` (async, worker-run) — Task 7 relies on pending state it stages; `@on(Button.Pressed, "#console-clear-attachment") handle_console_clear_attachment`.

- [ ] **Step 1: Write the failing test** (append to `Tests/UI/test_console_native_chat_flow.py`)

```python
async def test_console_attachment_worker_stages_image_and_inlines_text(tmp_path):
    from PIL import Image as PILImage

    image_path = tmp_path / "photo.png"
    PILImage.new("RGB", (4, 4), color=(0, 100, 0)).save(image_path, format="PNG")
    text_path = tmp_path / "notes.md"
    text_path.write_text("# heading\nbody")

    app = _build_test_app(screen_name="chat")
    _configure_native_ready_console(app)
    async with app.run_test(size=(160, 48)) as pilot:
        await _wait_for_selector(pilot, "#console-native-composer")
        screen = app.screen
        import tldw_chatbook.Chat.attachment_core as attachment_core
        # Test files live in tmp_path, outside $HOME — widen the safety root.
        original = attachment_core.load_processed_file

        async def _rooted(file_path, *, allowed_root=None):
            return await original(file_path, allowed_root=str(tmp_path))

        attachment_core.load_processed_file = _rooted
        try:
            await screen._process_console_attachment(str(image_path))
            await pilot.pause()
            store = screen._ensure_console_chat_store()
            session_id = store.active_session_id
            pending = store.pending_attachment(session_id)
            assert pending is not None and pending.file_type == "image"
            composer = screen.query_one("#console-native-composer", ConsoleComposerBar)
            assert composer._pending_attachment_label is not None

            await screen._process_console_attachment(str(text_path))
            await pilot.pause()
            assert "body" in composer.draft_text()
            assert "notes.md" in composer._display_draft_text()
        finally:
            attachment_core.load_processed_file = original
```

Note: `_process_console_attachment` must call `process_attachment_path`, which calls `load_processed_file` **via the module namespace** (`attachment_core.load_processed_file`, not a direct local reference) so this monkeypatch seam works — implement it that way.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest Tests/UI/test_console_native_chat_flow.py -q --no-header -k "attachment_worker"`
Expected: FAIL with `AttributeError: ... _process_console_attachment`.

- [ ] **Step 3: Implement**

In `attachment_core.py`, make the internal call monkeypatch-friendly: inside `process_attachment_path`, call the module-level name via globals (plain `await load_processed_file(...)` already resolves through module globals in Python — no change needed; just do NOT alias it to a local at import sites).

Replace `_handle_console_attach_context` (chat_screen.py:6235–6248) entirely:

```python
    async def _handle_console_attach_context(self, event: Button.Pressed) -> None:
        """Open the native Console file picker and stage the selected attachment."""
        event.stop()
        from fnmatch import fnmatch

        from tldw_chatbook.Chat.attachment_core import ATTACHMENT_FILTER_SPECS
        from tldw_chatbook.Widgets.enhanced_file_picker import FileOpen, Filters

        def create_filter(patterns: str):
            pattern_list = patterns.split(";")

            def filter_func(path: Path) -> bool:
                return any(fnmatch(path.name, pattern) for pattern in pattern_list)

            return filter_func

        file_filters = Filters(
            *[(label, create_filter(patterns)) for label, patterns in ATTACHMENT_FILTER_SPECS],
            ("All Files", lambda path: True),
        )

        def on_file_selected(file_path: Optional[Path]) -> None:
            if file_path:
                self.run_worker(
                    self._process_console_attachment(str(file_path)),
                    exclusive=True,
                )

        await self.app.push_screen(
            FileOpen(
                location=".",
                title="Select File to Attach",
                filters=file_filters,
                context="chat_images",
            ),
            callback=on_file_selected,
        )
```

Add the worker + clear handler after it:

```python
    async def _process_console_attachment(self, file_path: str) -> None:
        """Process a picked file and route it into the native Console composer."""
        from tldw_chatbook.Chat.attachment_core import process_attachment_path

        try:
            attachment = await process_attachment_path(file_path)
        except Exception as exc:
            logger.error(f"Console attachment processing failed for {file_path}: {exc}")
            self.app_instance.notify(
                str(exc) or "Failed to process attachment.", severity="error"
            )
            return
        composer = self._console_composer_or_none()
        if attachment.insert_mode == "inline":
            if composer is None or not attachment.text_content:
                self.app_instance.notify(
                    "Nothing to insert from this file.", severity="warning"
                )
                return
            composer.insert_file_segment(
                attachment.text_content, f"📄 {attachment.label}"
            )
            self.app_instance.notify(f"{attachment.display_name} content inserted")
        else:
            store = self._ensure_console_chat_store()
            session = store.ensure_session(
                workspace_id=store.workspace_context.active_workspace_id
            )
            store.set_pending_attachment(session.id, attachment)
            if composer is not None:
                composer.set_pending_attachment_label(attachment.label)
            self.app_instance.notify(f"{attachment.display_name} attached")
        self._sync_console_control_bar()

    @on(Button.Pressed, "#console-clear-attachment")
    def handle_console_clear_attachment(self, event: Button.Pressed) -> None:
        """Remove the pending native Console attachment."""
        event.stop()
        store = self._ensure_console_chat_store()
        if store.active_session_id is not None:
            store.clear_pending_attachment(store.active_session_id)
        composer = self._console_composer_or_none()
        if composer is not None:
            composer.set_pending_attachment_label(None)
        self.app_instance.notify("Attachment cleared")
```

Also update the two `@on` handler docstrings at 6225–6233 ("Route the Console attach affordance through the native file picker."). `Path` and `Optional` are already imported in chat_screen.py (verify; add if missing).

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest Tests/UI/test_console_native_chat_flow.py -q --no-header`
Expected: all PASS. The old "Console attachment is unavailable" dead-bridge path is gone; if any existing test asserted that copy, it is asserting dead code — flag it rather than silently updating (check `git grep -l "Console attachment is unavailable"` — expected: no test hits).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_native_chat_flow.py
git commit -m "feat(console): native attach flow replaces dead legacy bridge"
```

---

### Task 7: Screen — send gating and composer sync for attachments

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (`_console_send_blocked_reason` at 6161–6174, `_send_console_message_from_visible_action` at 6181–6211, `_sync_console_composer_action_state` at 6742–6769)
- Test: `Tests/UI/test_console_native_chat_flow.py` (append)

**Interfaces:**
- Consumes: `attachment_core.vision_block_reason`; store pending methods (Task 3); `_active_console_settings_readiness()` (returns `(effective_settings, readiness)`; `effective_settings.provider` / `.model`).
- Produces: `_console_pending_image_attachment() -> PendingAttachment | None`, `_console_attachment_blocked_reason() -> str`. Send is blocked with visible copy while a pending image sits on a non-vision model; the block clears on model change (readiness recomputed every `_sync_native_console_chat_ui`). Image-only sends (empty draft + pending image) are allowed through to the controller. Composer indicator follows the active session on every sync.

- [ ] **Step 1: Write the failing tests** (append to `Tests/UI/test_console_native_chat_flow.py`; `_pending_image`-style helper from Task 4's controller tests can be duplicated locally here — tests in different files must be self-contained)

```python
def _staged_image_attachment():
    from tldw_chatbook.Chat.attachment_core import PendingAttachment

    return PendingAttachment(
        file_path="/tmp/photo.png",
        display_name="photo.png",
        file_type="image",
        insert_mode="attachment",
        data=b"\x89PNG-bytes",
        mime_type="image/png",
        original_size=11,
        processed_size=11,
    )


async def test_pending_image_on_non_vision_model_blocks_send(monkeypatch):
    import tldw_chatbook.Chat.attachment_core as attachment_core

    monkeypatch.setattr(attachment_core, "is_vision_capable", lambda p, m: False)
    app = _build_test_app(screen_name="chat")
    _configure_native_ready_console(app)
    async with app.run_test(size=(160, 48)) as pilot:
        await _wait_for_selector(pilot, "#console-native-composer")
        screen = app.screen
        store = screen._ensure_console_chat_store()
        session = store.ensure_session()
        store.set_pending_attachment(session.id, _staged_image_attachment())

        reason = screen._console_send_blocked_reason()
        assert "can't accept images" in reason


async def test_pending_image_on_vision_model_does_not_block(monkeypatch):
    import tldw_chatbook.Chat.attachment_core as attachment_core

    monkeypatch.setattr(attachment_core, "is_vision_capable", lambda p, m: True)
    app = _build_test_app(screen_name="chat")
    _configure_native_ready_console(app)
    async with app.run_test(size=(160, 48)) as pilot:
        await _wait_for_selector(pilot, "#console-native-composer")
        screen = app.screen
        store = screen._ensure_console_chat_store()
        session = store.ensure_session()
        store.set_pending_attachment(session.id, _staged_image_attachment())

        assert screen._console_attachment_blocked_reason() == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest Tests/UI/test_console_native_chat_flow.py -q --no-header -k "non_vision_model_blocks or does_not_block"`
Expected: FAIL with `AttributeError: ... _console_attachment_blocked_reason` (first test may fail on missing helper too).

- [ ] **Step 3: Implement**

Add two helpers next to `_console_send_blocked_reason`:

```python
    def _console_pending_image_attachment(self):
        """Return the active session's staged image attachment, if any."""
        store = self._console_chat_store
        if store is None or store.active_session_id is None:
            return None
        try:
            pending = store.pending_attachment(store.active_session_id)
        except KeyError:
            return None
        if (
            pending is None
            or pending.insert_mode != "attachment"
            or pending.file_type != "image"
            or pending.data is None
        ):
            return None
        return pending

    def _console_attachment_blocked_reason(self) -> str:
        """Return blocked-send copy when a staged image can't reach the model."""
        from tldw_chatbook.Chat.attachment_core import vision_block_reason

        if self._console_pending_image_attachment() is None:
            return ""
        effective_settings, _readiness = self._active_console_settings_readiness()
        return (
            vision_block_reason(effective_settings.provider, effective_settings.model)
            or ""
        )
```

(Import note: `vision_block_reason` is imported inside the method so tests monkeypatching `attachment_core.is_vision_capable` take effect — `vision_block_reason` reads `is_vision_capable` from its own module globals.)

In `_console_send_blocked_reason`, before `return ""`:

```python
        attachment_reason = self._console_attachment_blocked_reason()
        if attachment_reason:
            return attachment_reason
        return ""
```

In `_send_console_message_from_visible_action`, change the empty-draft early return:

```python
        if not draft.strip() and self._console_pending_image_attachment() is None:
            self._focus_console_composer_if_needed(force=True)
            return
```

In `_sync_console_composer_action_state`, extend the blocked computation and keep the composer indicator in sync with the active session (covers session switches):

```python
        setup_blocked_reason = self._console_setup_blocked_reason()
        attachment_blocked_reason = self._console_attachment_blocked_reason()
        send_blocked = (
            send_blocked
            or bool(setup_blocked_reason)
            or bool(attachment_blocked_reason)
        )

        pending = self._console_pending_image_attachment()
        composer.set_pending_attachment_label(pending.label if pending else None)

        composer.sync_action_state(
            has_draft=bool(composer.draft_text().strip()) or pending is not None,
            run_active=run_active,
            can_save_chatbook=can_save_chatbook,
            send_blocked=send_blocked,
            setup_blocked_reason=setup_blocked_reason or attachment_blocked_reason,
        )
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest Tests/UI/test_console_native_chat_flow.py -q --no-header`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_native_chat_flow.py
git commit -m "feat(console): block send with visible reason for non-vision image attachments"
```

---

### Task 8: Screen — resume hydration + screen-state serialization metadata

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (`_console_messages_from_conversation_tree` at 1956–1979, `_serialize_console_message` at 5333–5349, `_restore_console_message` at 5351–5383)
- Test: `Tests/UI/test_console_native_chat_flow.py` (append)

**Interfaces:**
- Consumes: `ConsoleChatMessage` image fields (Task 3). Conversation-tree message rows carry `image_data`/`image_mime_type` (surfaced by `chat_conversation_service.py:171-172`).
- Produces: resume keeps image messages (including image-only rows); screen-state snapshots carry `image_mime_type` + `attachment_label` (never bytes), so transcript chips survive screen switches while bytes rehydrate from the DB.

- [ ] **Step 1: Write the failing tests** (append to `Tests/UI/test_console_native_chat_flow.py`)

```python
async def test_resume_hydrates_image_messages_including_image_only_rows():
    app = _build_test_app(screen_name="chat")
    _configure_native_ready_console(app)
    async with app.run_test(size=(160, 48)) as pilot:
        await _wait_for_selector(pilot, "#console-native-composer")
        screen = app.screen
        tree = {
            "conversation": {"title": "Saved", "workspace_id": None},
            "root_threads": [
                {
                    "message": {
                        "id": "m-1",
                        "sender": "user",
                        "content": "",
                        "image_data": b"\x89PNG-bytes",
                        "image_mime_type": "image/png",
                    },
                    "children": [
                        {
                            "message": {
                                "id": "m-2",
                                "sender": "assistant",
                                "content": "a red square",
                            },
                            "children": [],
                        }
                    ],
                }
            ],
        }
        messages = screen._console_messages_from_conversation_tree(tree)

        assert len(messages) == 2
        assert messages[0].image_data == b"\x89PNG-bytes"
        assert messages[0].image_mime_type == "image/png"
        assert messages[0].content == ""
        assert messages[1].content == "a red square"


def test_console_message_serialization_carries_image_metadata_not_bytes():
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="look",
        image_data=b"\x89PNG-bytes",
        image_mime_type="image/png",
        attachment_label="photo.png · 11 B",
    )
    payload = ChatScreen._serialize_console_message(message)

    assert payload["image_mime_type"] == "image/png"
    assert payload["attachment_label"] == "photo.png · 11 B"
    assert "image_data" not in payload

    restored = ChatScreen._restore_console_message(payload)
    assert restored is not None
    assert restored.image_mime_type == "image/png"
    assert restored.attachment_label == "photo.png · 11 B"
    assert restored.image_data is None
```

**Important:** before writing the resume test, check the real shape `_iter_console_tree_messages` expects (read it in chat_screen.py near line 1930) — the fake `tree["root_threads"]` above assumes `{"message": {...}, "children": [...]}` nodes; adjust the fixture to the actual node shape (an existing resume test in this file shows it — search for `root_threads`). Row keys `content`/`sender`/`id` are per `_console_messages_from_conversation_tree` and `_console_message_role_from_persisted` (lines 1951–1954).

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest Tests/UI/test_console_native_chat_flow.py -q --no-header -k "resume_hydrates or serialization_carries"`
Expected: resume test FAILS (image-only row dropped → `len(messages) == 1`); serialization test FAILS (`KeyError: 'image_mime_type'`).

- [ ] **Step 3: Implement**

Rewrite the row loop in `_console_messages_from_conversation_tree`:

```python
        for row in self._iter_console_tree_messages(tree.get("root_threads")):
            content = str(row.get("content") or "")
            raw_image = row.get("image_data")
            image_data = bytes(raw_image) if isinstance(raw_image, (bytes, bytearray)) else None
            raw_mime = row.get("image_mime_type")
            image_mime_type = str(raw_mime) if raw_mime else None
            if not content and image_data is None:
                continue
            persisted_message_id = row.get("id")
            messages.append(
                ConsoleChatMessage(
                    role=self._console_message_role_from_persisted(row),
                    content=content,
                    status="complete",
                    persisted_message_id=(
                        str(persisted_message_id)
                        if persisted_message_id is not None
                        else None
                    ),
                    image_data=image_data,
                    image_mime_type=image_mime_type,
                )
            )
```

In `_serialize_console_message`, add two keys (bytes stay out — the snapshot must remain JSON-safe):

```python
            "image_mime_type": message.image_mime_type,
            "attachment_label": message.attachment_label,
```

In `_restore_console_message`, add two constructor arguments:

```python
            image_mime_type=(
                str(payload["image_mime_type"])
                if payload.get("image_mime_type")
                else None
            ),
            attachment_label=(
                str(payload["attachment_label"])
                if payload.get("attachment_label")
                else None
            ),
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest Tests/UI/test_console_native_chat_flow.py -q --no-header`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_native_chat_flow.py
git commit -m "fix(console): hydrate image messages on resume and carry image metadata in screen state"
```

---

### Task 9: Transcript chip + Save Image action

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py` (`_message_render_text` at 76–96)
- Modify: `tldw_chatbook/Chat/console_message_actions.py` (`available_actions` at 84–103, `dispatch` at 137–216)
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (`_parse_console_message_action_button_id` at 6546–6564, action-dispatch handler around 6373–6425; new `_save_console_message_image`)
- Test: `Tests/UI/test_console_native_transcript.py` (append transcript/service tests), `Tests/UI/test_console_native_chat_flow.py` (append save handler test)

**Interfaces:**
- Consumes: `ConsoleChatMessage` image fields (Task 3); `get_cli_setting("chat.images", "save_location", "~/Downloads")` from `tldw_chatbook.config`.
- Produces: transcript rows for image messages render a `🖼` chip line; `ConsoleMessageActionService.available_actions` includes `("save-image", "Save Image")` only for messages with image data/metadata; `dispatch("save-image", ...)` returns `status="completed"`, `visible_copy="Saving image to disk."`; screen handler `_save_console_message_image(message_id)` writes bytes (store first, DB fallback via `persisted_message_id`) to the configured save location.

- [ ] **Step 1: Write the failing tests**

Append to `Tests/UI/test_console_native_transcript.py` (this file already imports `ConsoleChatMessage`, `ConsoleMessageRole`, and transcript internals — follow its imports; add `from tldw_chatbook.Chat.console_message_actions import ConsoleMessageActionService` and `from tldw_chatbook.Widgets.Console.console_transcript import _message_render_text` if not present):

```python
def test_image_message_row_renders_chip_line():
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="what is this?",
        image_data=b"\x89PNG-bytes",
        image_mime_type="image/png",
        attachment_label="photo.png · 11 B",
    )
    rendered = _message_render_text(message, selected=False)
    assert "🖼 photo.png · 11 B" in rendered.plain


def test_image_only_message_row_renders_chip_without_body():
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="",
        image_data=b"\x89PNG-bytes",
        image_mime_type="image/png",
    )
    rendered = _message_render_text(message, selected=False)
    assert "🖼" in rendered.plain


def test_save_image_action_only_offered_for_image_messages():
    service = ConsoleMessageActionService()
    plain = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="text")
    with_image = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="pic",
        image_data=b"\x89PNG-bytes",
        image_mime_type="image/png",
    )
    plain_ids = [action.action_id for action in service.available_actions(plain)]
    image_ids = [action.action_id for action in service.available_actions(with_image)]
    assert "save-image" not in plain_ids
    assert "save-image" in image_ids

    result = service.dispatch("save-image", with_image)
    assert result.status == "completed"
    assert result.visible_copy == "Saving image to disk."
```

Append to `Tests/UI/test_console_native_chat_flow.py`:

```python
async def test_save_console_message_image_writes_file(tmp_path, monkeypatch):
    app = _build_test_app(screen_name="chat")
    _configure_native_ready_console(app)
    async with app.run_test(size=(160, 48)) as pilot:
        await _wait_for_selector(pilot, "#console-native-composer")
        screen = app.screen
        monkeypatch.setattr(
            "tldw_chatbook.UI.Screens.chat_screen.get_cli_setting",
            lambda section, key, default=None: str(tmp_path)
            if (section, key) == ("chat.images", "save_location")
            else default,
        )
        store = screen._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="pic",
            image_data=b"\x89PNG-bytes",
            image_mime_type="image/png",
        )

        await screen._save_console_message_image(message.id)

        saved = list(tmp_path.glob("console_image_*.png"))
        assert len(saved) == 1
        assert saved[0].read_bytes() == b"\x89PNG-bytes"
```

(Verify `get_cli_setting` is imported at module level in chat_screen.py — `git grep -n "get_cli_setting" tldw_chatbook/UI/Screens/chat_screen.py`. If it's imported from `tldw_chatbook.config`, the monkeypatch target above is correct; if imported differently, patch the name chat_screen actually uses.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest Tests/UI/test_console_native_transcript.py Tests/UI/test_console_native_chat_flow.py -q --no-header -k "chip or save_image or save_console"`
Expected: FAIL (no chip in rendered text; `save-image` absent; `_save_console_message_image` missing).

- [ ] **Step 3: Implement**

`console_transcript.py` — add a chip helper above `_message_render_text` and use it:

```python
def _message_image_chip(message: ConsoleChatMessage) -> str | None:
    """Return the placeholder chip line for a message carrying an image."""
    if message.image_data is None and not message.image_mime_type:
        return None
    label = message.attachment_label or message.image_mime_type or "image"
    return f"🖼 {label}"
```

In `_message_render_text`, after `body = _message_body(message)`:

```python
    chip = _message_image_chip(message)
    if chip:
        body = f"{body}\n{chip}" if body else chip
```

(The chip's `\n` naturally routes rendering through the existing two-line branch; `_message_row_signature` already embeds the rendered Content, so row diffing picks the chip up with no signature changes.)

`console_message_actions.py` — add below `_FAILED_RETRY_ACTIONS`:

```python
    _SAVE_IMAGE_ACTIONS: tuple[tuple[str, str], ...] = (("save-image", "Save Image"),)

    @staticmethod
    def _has_image(message: ConsoleChatMessage) -> bool:
        return message.image_data is not None or bool(message.image_mime_type)
```

In `available_actions`, after the `if message.variants is not None:` block (before the `failed` early return):

```python
        if self._has_image(message):
            completed_actions = completed_actions + list(self._SAVE_IMAGE_ACTIONS)
```

In `dispatch`, before the final `return ConsoleActionResult(... status="wip" ...)`:

```python
        if action_id == "save-image":
            return ConsoleActionResult(
                action_id=action_id,
                status="completed",
                visible_copy="Saving image to disk.",
                target_message_id=message.id,
            )
```

`chat_screen.py` — add the prefix to `_parse_console_message_action_button_id` (order within the tuple doesn't matter; no prefix collides with `save-as-`):

```python
            ("console-message-action-save-image-", "save-image"),
```

In the action-dispatch handler, after the `feedback` branch (around line 6402):

```python
        if action_id == "save-image" and result.status == "completed":
            self.run_worker(self._save_console_message_image(message_id), exclusive=True)
            return True
```

Add the save worker near `_save_console_message_as_note`:

```python
    async def _save_console_message_image(self, message_id: str) -> None:
        """Write a Console message's image to the configured save location."""
        import mimetypes as _mimetypes
        from datetime import datetime as _datetime

        store = self._ensure_console_chat_store()
        try:
            message = store.get_message(message_id)
        except KeyError:
            self.app_instance.notify(
                "Console message no longer exists.", severity="warning"
            )
            return
        image_data = message.image_data
        mime_type = message.image_mime_type
        if image_data is None and message.persisted_message_id is not None:
            db = getattr(self.app_instance, "chachanotes_db", None)
            row = (
                db.get_message_by_id(message.persisted_message_id)
                if db is not None
                else None
            )
            if row:
                image_data = row.get("image_data")
                mime_type = row.get("image_mime_type") or mime_type
        if not image_data:
            self.app_instance.notify(
                "No image data available for this message.", severity="warning"
            )
            return
        save_location = Path(
            os.path.expanduser(
                get_cli_setting("chat.images", "save_location", "~/Downloads")
            )
        )
        save_location.mkdir(parents=True, exist_ok=True)
        extension = _mimetypes.guess_extension(mime_type or "image/png") or ".png"
        target = (
            save_location
            / f"console_image_{_datetime.now().strftime('%Y%m%d_%H%M%S')}{extension}"
        )
        target.write_bytes(bytes(image_data))
        self.app_instance.notify(f"Image saved to {target}")
```

(`os` and `Path` are imported in chat_screen.py; verify `get_cli_setting` import per Step 1's note.)

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest Tests/UI/test_console_native_transcript.py Tests/UI/test_console_native_chat_flow.py Tests/Chat/ -q --no-header`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Console/console_transcript.py tldw_chatbook/Chat/console_message_actions.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_native_transcript.py Tests/UI/test_console_native_chat_flow.py
git commit -m "feat(console): transcript image chip and Save Image message action"
```

---

### Task 10: Full verification + visual review gate

**Files:**
- No production changes expected (fix-forward only if verification finds issues).

- [ ] **Step 1: Run the complete affected test surface**

Run:
```bash
.venv/bin/python -m pytest \
  Tests/Chat/ \
  Tests/UI/test_console_native_chat_flow.py \
  Tests/UI/test_console_native_transcript.py \
  Tests/UI/test_console_workbench_contract.py \
  Tests/UI/test_chat_image_attachment.py \
  Tests/Event_Handlers/Chat_Events/test_chat_image_events.py \
  Tests/Event_Handlers/Chat_Events/test_chat_image_properties.py \
  Tests/unit/test_chat_image_unit.py \
  Tests/DB/test_chat_image_db_compatibility.py \
  Tests/Widgets/test_chat_message_enhanced.py \
  -q --no-header
```
Expected: 0 failures beyond the documented pre-existing UI baseline (compare any failure against `origin/dev` with `git stash` if unsure whether it's pre-existing — inherited failures must be byte-identical at the parent commit).

- [ ] **Step 2: End-to-end verify in the running app**

Use the `verify` skill / textual-serve capture recipe (see `Docs/superpowers/qa/` recipes and the Console capture notes in project memory): launch the app, open Console, and walk: attach an image (indicator appears) → switch to a non-vision model (send blocks with reason) → switch back (block clears) → send (chip renders in transcript) → select the message (Save Image appears) → Save Image (file lands in `~/Downloads` or configured path) → attach a `.md` file (collapsed `📄` segment; send includes file text). Capture screenshots of: composer with indicator, blocked state, transcript chip + action row.

- [ ] **Step 3: Visual review gate**

Present the three screenshots against the Console style anchor for user approval **before merging** (standing Console rule: every screen change needs explicit visual approval).

- [ ] **Step 4: Final commit / branch wrap-up**

Follow superpowers:finishing-a-development-branch — do not merge or open a PR without the visual gate from Step 3.

---

## Deferred follow-ups (file as backlog tasks after merge, do not implement)

1. Inline pixel/TGP image rendering in the Console transcript with Toggle View (port from `ChatMessageEnhanced._render_pixelated/_render_regular`).
2. Clipboard image paste and drag-drop attach.
3. Multiple attachments per message.
4. Pending attachment survival across screen navigation.
5. Sync v2 attachment upload (task-57) — until then image messages sync text-only.
6. Chatbook export carrying images (task-19 adjacent).
