# Console Inline Image Rendering (TASK-215) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render Console transcript images inline — pixel mode (rich-pixels) and graphics mode (textual-image auto) — with a per-message Toggle View cycling pixels → graphics → hidden, honoring the previously-dead `[chat.images]` config keys.

**Architecture:** A new pure module (`Chat/console_image_view.py`) owns mode resolution, per-message view state, and a bounded off-loop render cache. The transcript gains an additive keyed `image` row kind fed by a prebuilt spec map, so the streaming reconcile loop never rebuilds image rows. The screen owns state + cache, preps images in a dedicated worker group, and serializes view overrides (strings only) in the existing screen-state allowlist.

**Tech Stack:** Python ≥3.11, Textual 8.2.7, rich-pixels, textual-image (both already core deps), PIL, pytest.

**Spec:** `Docs/superpowers/specs/2026-07-13-console-inline-image-rendering-design.md` — read it first; all decisions there are settled.

## Global Constraints

- Run all tests with `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <paths> -q --no-header` from the worktree root (`/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/console-inline-images-215`).
- Modify ONLY files each task names. NEVER touch files outside the worktree (no `~/.config/`, no main checkout).
- Existing tests are read-only; test-file additions must be append-only at true end-of-file — after appending verify `git diff HEAD -- <testfile> | grep -c '^-[^-]'` is 0. If a pre-existing test fails, your change is wrong.
- Exact values (verbatim from spec): mode literals `"pixels" | "graphics" | "hidden"`; decode cap **1024 px** longest side; LRU bound **16** entries; pixels renderable cap **80 cols × 40 lines**; graphics row style caps `max-width: 80` cells / `max-height: 40` rows; worker group `"console-image-prep"`; action id `"toggle-image-view"`, label `"View"`; serialization key `"image_view_modes"`.
- Never serialize raw image bytes or renderables into screen-state payloads (strings only in `image_view_modes`).
- Legacy chat rendering untouched; legacy image regression gate (`Tests/UI/test_chat_image_attachment.py`, `Tests/Event_Handlers/Chat_Events/test_chat_image_events.py` + `_properties`, `Tests/unit/test_chat_image_unit.py`, `Tests/DB/test_chat_image_db_compatibility.py`, `Tests/Widgets/test_chat_message_enhanced.py`) must stay green unedited.
- CI is intentionally cancelled remotely — verify locally.
- End every commit message with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Pure module — modes, resolution, view state, render cache

**Files:**
- Create: `tldw_chatbook/Chat/console_image_view.py`
- Modify: `tldw_chatbook/Utils/terminal_utils.py` (one additive line: expose `terminal_type` in the returned dict)
- Test: `Tests/Chat/test_console_image_view.py` (new)

**Interfaces:**
- Consumes: `terminal_utils.detect_terminal_capabilities()` (returns dict with `sixel`/`tgp`/`unicode`/`recommended_mode`; this task adds `terminal_type`), `terminal_utils.get_image_render_mode(config_mode) -> Literal["pixels","regular"]` (existing, currently consumer-less — verified at terminal_utils.py:119).
- Produces (used by Tasks 2–4):
  - `ConsoleImageViewMode = Literal["pixels", "graphics", "hidden"]`
  - `resolve_default_mode(app_config: Mapping[str, Any]) -> Literal["pixels", "graphics"]`
  - `next_view_mode(current: ConsoleImageViewMode) -> ConsoleImageViewMode` (pixels→graphics→hidden→pixels)
  - `class ConsoleImageViewState`: `mode_for(message_id, default)`, `set_mode(message_id, mode, default)` (dropping entries equal to default), `serialize() -> dict[str, str]`, `restore(payload)`, `prune(live_message_ids)`
  - `class ConsoleImageRenderCache`: `prepare(message_id, image_data) -> bool` (sync, CPU-bound — callers run it off-loop), `get_pil(message_id) -> PIL.Image | None`, `get_pixels(message_id) -> Pixels | None` (lazy-built, cached), `is_failed(message_id) -> bool` (negative cache), `pending_ids(messages) -> list[tuple[str, bytes]]` (image messages with bytes, not cached, not failed), `evict_session(message_ids)`, `clear()`
  - `@dataclass(frozen=True) ConsoleImageRowSpec: message_id: str; mode: Literal["pixels","graphics"]; pixels: Pixels | None; pil: "PIL.Image.Image | None"`
  - Constants: `IMAGE_DECODE_MAX_DIMENSION = 1024`, `IMAGE_CACHE_MAX_ENTRIES = 16`, `PIXELS_MAX_COLS = 80`, `PIXELS_MAX_LINES = 40`

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Chat/test_console_image_view.py
from io import BytesIO

from PIL import Image as PILImage

from tldw_chatbook.Chat.console_image_view import (
    IMAGE_CACHE_MAX_ENTRIES,
    IMAGE_DECODE_MAX_DIMENSION,
    ConsoleImageRenderCache,
    ConsoleImageViewState,
    next_view_mode,
    resolve_default_mode,
)


def _png_bytes(size=(64, 64), color=(200, 10, 10)) -> bytes:
    buffer = BytesIO()
    PILImage.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_next_view_mode_cycles_three_states():
    assert next_view_mode("pixels") == "graphics"
    assert next_view_mode("graphics") == "hidden"
    assert next_view_mode("hidden") == "pixels"


def test_resolve_default_mode_explicit_config_wins(monkeypatch):
    import tldw_chatbook.Chat.console_image_view as civ

    monkeypatch.setattr(
        civ, "detect_terminal_capabilities", lambda: {"terminal_type": "kitty"}
    )
    assert resolve_default_mode({"chat": {"images": {"default_render_mode": "pixels"}}}) == "pixels"
    assert resolve_default_mode({"chat": {"images": {"default_render_mode": "regular"}}}) == "graphics"


def test_resolve_default_mode_auto_uses_terminal_override(monkeypatch):
    import tldw_chatbook.Chat.console_image_view as civ

    monkeypatch.setattr(
        civ, "detect_terminal_capabilities", lambda: {"terminal_type": "kitty"}
    )
    config = {
        "chat": {
            "images": {
                "default_render_mode": "auto",
                "terminal_overrides": {"kitty": "regular", "default": "pixels"},
            }
        }
    }
    assert resolve_default_mode(config) == "graphics"


def test_resolve_default_mode_auto_falls_back_to_default_override(monkeypatch):
    import tldw_chatbook.Chat.console_image_view as civ

    monkeypatch.setattr(
        civ, "detect_terminal_capabilities", lambda: {"terminal_type": "xterm"}
    )
    config = {
        "chat": {
            "images": {
                "default_render_mode": "auto",
                "terminal_overrides": {"kitty": "regular", "default": "pixels"},
            }
        }
    }
    assert resolve_default_mode(config) == "pixels"


def test_resolve_default_mode_auto_without_overrides_uses_capability_mode(monkeypatch):
    import tldw_chatbook.Chat.console_image_view as civ

    monkeypatch.setattr(
        civ, "detect_terminal_capabilities", lambda: {"terminal_type": "unknown"}
    )
    monkeypatch.setattr(civ, "get_image_render_mode", lambda mode: "regular")
    assert resolve_default_mode({"chat": {"images": {"default_render_mode": "auto"}}}) == "graphics"


def test_resolve_default_mode_garbage_falls_back_to_pixels(monkeypatch):
    import tldw_chatbook.Chat.console_image_view as civ

    monkeypatch.setattr(
        civ, "detect_terminal_capabilities", lambda: {"terminal_type": "unknown"}
    )
    monkeypatch.setattr(civ, "get_image_render_mode", lambda mode: "regular")
    assert resolve_default_mode({"chat": {"images": {"default_render_mode": "nonsense"}}}) in {
        "pixels",
        "graphics",
    }
    assert resolve_default_mode({}) in {"pixels", "graphics"}


def test_view_state_defaults_overrides_and_prune():
    state = ConsoleImageViewState()
    assert state.mode_for("m-1", default="pixels") == "pixels"

    state.set_mode("m-1", "hidden", default="pixels")
    assert state.mode_for("m-1", default="pixels") == "hidden"
    assert state.serialize() == {"m-1": "hidden"}

    # Setting back to the default drops the entry.
    state.set_mode("m-1", "pixels", default="pixels")
    assert state.serialize() == {}

    state.set_mode("m-1", "graphics", default="pixels")
    state.set_mode("m-2", "hidden", default="pixels")
    state.prune({"m-2"})
    assert state.serialize() == {"m-2": "hidden"}


def test_view_state_restore_ignores_invalid_entries():
    state = ConsoleImageViewState()
    state.restore({"m-1": "graphics", "m-2": "bogus", 3: "hidden"})
    assert state.serialize() == {"m-1": "graphics"}


def test_cache_prepares_downscales_and_serves_both_renderables():
    cache = ConsoleImageRenderCache()
    big = _png_bytes(size=(2048, 512))

    assert cache.prepare("m-1", big) is True
    pil = cache.get_pil("m-1")
    assert pil is not None
    assert max(pil.width, pil.height) <= IMAGE_DECODE_MAX_DIMENSION
    assert cache.get_pixels("m-1") is not None
    assert cache.get_pixels("m-1") is cache.get_pixels("m-1")  # lazy build cached


def test_cache_negative_caches_corrupt_bytes():
    cache = ConsoleImageRenderCache()
    assert cache.prepare("m-bad", b"not an image") is False
    assert cache.is_failed("m-bad") is True
    assert cache.get_pil("m-bad") is None


def test_cache_lru_bound_evicts_oldest():
    cache = ConsoleImageRenderCache()
    payload = _png_bytes(size=(8, 8))
    for index in range(IMAGE_CACHE_MAX_ENTRIES + 1):
        cache.prepare(f"m-{index}", payload)
    assert cache.get_pil("m-0") is None  # evicted
    assert cache.get_pil(f"m-{IMAGE_CACHE_MAX_ENTRIES}") is not None


def test_cache_pending_ids_and_session_eviction():
    class _Message:
        def __init__(self, message_id, image_data):
            self.id = message_id
            self.image_data = image_data

    cache = ConsoleImageRenderCache()
    payload = _png_bytes()
    cache.prepare("m-done", payload)
    cache.prepare("m-bad", b"junk")
    messages = [
        _Message("m-done", payload),
        _Message("m-bad", b"junk"),
        _Message("m-new", payload),
        _Message("m-none", None),
    ]
    pending = cache.pending_ids(messages)
    assert [message_id for message_id, _ in pending] == ["m-new"]

    cache.evict_session({"m-done"})
    assert cache.get_pil("m-done") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_image_view.py -q --no-header`
Expected: collection error — `ModuleNotFoundError: No module named 'tldw_chatbook.Chat.console_image_view'`

- [ ] **Step 3: Add `terminal_type` to `detect_terminal_capabilities`'s return**

In `tldw_chatbook/Utils/terminal_utils.py`, the function already computes `terminal_type` for metrics (around line 92-109). Immediately BEFORE `return capabilities` (line ~117), add:

```python
    capabilities['terminal_type'] = terminal_type
```

Also extend the docstring's Returns list with `- terminal_type: Detected terminal name ('kitty', 'wezterm', 'iterm2', ... or 'unknown')`.

- [ ] **Step 4: Write the module**

```python
# tldw_chatbook/Chat/console_image_view.py
"""Console inline-image view modes, per-message state, and render cache.

Pure module: no Textual imports (rich_pixels renders via Rich, PIL is
imaging). The graphics-mode widget itself is created in the transcript.

This module is the FIRST real consumer of ``[chat.images].default_render_mode``,
``[chat.images.terminal_overrides]``, and ``terminal_utils`` detection —
legacy chat defines those keys but never reads them.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Iterable, Literal, Mapping

from loguru import logger
from PIL import Image as PILImage
from rich_pixels import Pixels

from tldw_chatbook.Utils.terminal_utils import (
    detect_terminal_capabilities,
    get_image_render_mode,
)

ConsoleImageViewMode = Literal["pixels", "graphics", "hidden"]

IMAGE_DECODE_MAX_DIMENSION = 1024
IMAGE_CACHE_MAX_ENTRIES = 16
PIXELS_MAX_COLS = 80
PIXELS_MAX_LINES = 40

_RENDER_MODES: tuple[ConsoleImageViewMode, ...] = ("pixels", "graphics", "hidden")
_LEGACY_TO_MODE = {"pixels": "pixels", "regular": "graphics"}


def next_view_mode(current: ConsoleImageViewMode) -> ConsoleImageViewMode:
    """Return the next mode in the pixels -> graphics -> hidden cycle.

    Args:
        current: The current view mode.

    Returns:
        The next mode; unknown input restarts the cycle at "pixels".
    """
    try:
        index = _RENDER_MODES.index(current)
    except ValueError:
        return "pixels"
    return _RENDER_MODES[(index + 1) % len(_RENDER_MODES)]


def _chat_images_config(app_config: Mapping[str, Any]) -> Mapping[str, Any]:
    chat = app_config.get("chat") if isinstance(app_config, Mapping) else None
    images = chat.get("images") if isinstance(chat, Mapping) else None
    return images if isinstance(images, Mapping) else {}


def resolve_default_mode(app_config: Mapping[str, Any]) -> Literal["pixels", "graphics"]:
    """Resolve the session-default inline render mode from config + terminal.

    Resolution order (spec-defined; no prior consumer existed to mirror):
    explicit ``default_render_mode`` of ``pixels``/``regular`` wins; ``auto``
    consults ``terminal_overrides[<terminal_type>]``, then
    ``terminal_overrides["default"]``, then ``get_image_render_mode("auto")``;
    anything unrecognized falls back to "pixels".

    Args:
        app_config: The application config mapping (``[chat.images]`` section
            is read; missing sections are tolerated).

    Returns:
        "pixels" or "graphics" (the config value "regular" maps to "graphics").
    """
    images = _chat_images_config(app_config)
    configured = str(images.get("default_render_mode", "auto")).strip().lower()
    if configured in _LEGACY_TO_MODE:
        return _LEGACY_TO_MODE[configured]  # type: ignore[return-value]

    overrides = images.get("terminal_overrides")
    overrides = overrides if isinstance(overrides, Mapping) else {}
    try:
        terminal_type = str(
            detect_terminal_capabilities().get("terminal_type", "unknown")
        )
    except Exception:
        logger.opt(exception=True).warning("Terminal capability detection failed.")
        terminal_type = "unknown"
    for key in (terminal_type, "default"):
        override = str(overrides.get(key, "")).strip().lower()
        if override in _LEGACY_TO_MODE:
            return _LEGACY_TO_MODE[override]  # type: ignore[return-value]

    try:
        resolved = get_image_render_mode("auto")
    except Exception:
        logger.opt(exception=True).warning("Image render mode resolution failed.")
        resolved = "pixels"
    return _LEGACY_TO_MODE.get(resolved, "pixels")  # type: ignore[return-value]


class ConsoleImageViewState:
    """Per-message inline-image view overrides (non-default entries only)."""

    def __init__(self) -> None:
        self._overrides: dict[str, ConsoleImageViewMode] = {}

    def mode_for(
        self,
        message_id: str,
        *,
        default: Literal["pixels", "graphics"],
    ) -> ConsoleImageViewMode:
        """Return the effective mode for a message.

        Args:
            message_id: Native Console message ID.
            default: The session-default render mode.

        Returns:
            The stored override, or the default when none is stored.
        """
        return self._overrides.get(message_id, default)

    def set_mode(
        self,
        message_id: str,
        mode: ConsoleImageViewMode,
        *,
        default: Literal["pixels", "graphics"],
    ) -> None:
        """Store a mode override, dropping entries equal to the default.

        Args:
            message_id: Native Console message ID.
            mode: The mode chosen for this message.
            default: The session-default render mode.
        """
        if mode == default:
            self._overrides.pop(message_id, None)
        else:
            self._overrides[message_id] = mode

    def serialize(self) -> dict[str, str]:
        """Return a JSON-safe snapshot of the overrides."""
        return dict(self._overrides)

    def restore(self, payload: Any) -> None:
        """Replace overrides from a saved snapshot, ignoring invalid entries.

        Args:
            payload: The previously serialized mapping (tolerates garbage).
        """
        self._overrides.clear()
        if not isinstance(payload, Mapping):
            return
        for key, value in payload.items():
            if isinstance(key, str) and value in _RENDER_MODES:
                self._overrides[key] = value

    def prune(self, live_message_ids: Iterable[str]) -> None:
        """Drop overrides for messages that no longer exist.

        Args:
            live_message_ids: IDs of messages currently in any session.
        """
        live = set(live_message_ids)
        for message_id in [m for m in self._overrides if m not in live]:
            del self._overrides[message_id]


@dataclass(frozen=True)
class ConsoleImageRowSpec:
    """Prebuilt payload for one transcript image row."""

    message_id: str
    mode: Literal["pixels", "graphics"]
    pixels: Pixels | None = None
    pil: "PILImage.Image | None" = None


class ConsoleImageRenderCache:
    """Bounded cache of decoded transcript images (LRU + negative cache).

    ``prepare`` is synchronous CPU work (PIL decode + LANCZOS downscale) —
    callers must run it off the event loop (``asyncio.to_thread``).
    """

    def __init__(self, *, max_entries: int = IMAGE_CACHE_MAX_ENTRIES) -> None:
        self._max_entries = max_entries
        self._images: OrderedDict[str, PILImage.Image] = OrderedDict()
        self._pixels: dict[str, Pixels] = {}
        self._failed: set[str] = set()

    def prepare(self, message_id: str, image_data: bytes) -> bool:
        """Decode, downscale, and cache an image; negative-cache failures.

        Args:
            message_id: Native Console message ID (cache key).
            image_data: Raw image bytes.

        Returns:
            True when the image is cached, False when decoding failed.
        """
        try:
            pil = PILImage.open(BytesIO(image_data))
            pil.load()
            if max(pil.width, pil.height) > IMAGE_DECODE_MAX_DIMENSION:
                pil.thumbnail(
                    (IMAGE_DECODE_MAX_DIMENSION, IMAGE_DECODE_MAX_DIMENSION),
                    PILImage.Resampling.LANCZOS,
                )
        except Exception:
            logger.opt(exception=True).warning(
                f"Console image prep failed for message {message_id}."
            )
            self._failed.add(message_id)
            return False
        self._failed.discard(message_id)
        self._images[message_id] = pil
        self._images.move_to_end(message_id)
        self._pixels.pop(message_id, None)
        while len(self._images) > self._max_entries:
            evicted_id, _ = self._images.popitem(last=False)
            self._pixels.pop(evicted_id, None)
        return True

    def get_pil(self, message_id: str) -> PILImage.Image | None:
        """Return the cached decoded image, refreshing its LRU position."""
        pil = self._images.get(message_id)
        if pil is not None:
            self._images.move_to_end(message_id)
        return pil

    def get_pixels(self, message_id: str) -> Pixels | None:
        """Return (lazily building) the pixels renderable for a cached image."""
        cached = self._pixels.get(message_id)
        if cached is not None:
            return cached
        pil = self.get_pil(message_id)
        if pil is None:
            return None
        # Half-block rendering: one text line shows two pixel rows.
        scaled = pil.copy()
        scaled.thumbnail(
            (PIXELS_MAX_COLS, PIXELS_MAX_LINES * 2), PILImage.Resampling.LANCZOS
        )
        pixels = Pixels.from_image(scaled)
        self._pixels[message_id] = pixels
        return pixels

    def is_failed(self, message_id: str) -> bool:
        """Return whether decoding previously failed for this message."""
        return message_id in self._failed

    def pending_ids(self, messages: Iterable[Any]) -> list[tuple[str, bytes]]:
        """Return (message_id, bytes) pairs needing preparation.

        Args:
            messages: Objects with ``id`` and ``image_data`` attributes.

        Returns:
            Pairs for messages that carry bytes but are neither cached nor
            negative-cached.
        """
        pending: list[tuple[str, bytes]] = []
        for message in messages:
            image_data = getattr(message, "image_data", None)
            message_id = getattr(message, "id", None)
            if (
                isinstance(message_id, str)
                and image_data
                and message_id not in self._images
                and message_id not in self._failed
            ):
                pending.append((message_id, image_data))
        return pending

    def evict_session(self, message_ids: Iterable[str]) -> None:
        """Drop cache entries (and failure marks) for a closed session."""
        for message_id in message_ids:
            self._images.pop(message_id, None)
            self._pixels.pop(message_id, None)
            self._failed.discard(message_id)

    def clear(self) -> None:
        """Drop all cache state (used on full store restore)."""
        self._images.clear()
        self._pixels.clear()
        self._failed.clear()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_image_view.py -q --no-header`
Expected: 12 passed.

Also run the terminal_utils neighbors: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/ -q --no-header -k "terminal_utils"`
Expected: whatever exists passes (check first with `--collect-only`; if no tests exist for terminal_utils, note it in the report).

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Chat/console_image_view.py tldw_chatbook/Utils/terminal_utils.py Tests/Chat/test_console_image_view.py
git commit -m "feat(console): image view modes, resolution, and bounded render cache

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Transcript — image row kind fed by spec map

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py` (`_TranscriptRow` ~line 153, `_transcript_rows` ~566, `_build_row_widget` ~679, `_update_row_widget` ~707, `ConsoleTranscript.__init__`/state)
- Test: `Tests/UI/test_console_native_transcript.py` (append-only)

**Interfaces:**
- Consumes: `ConsoleImageRowSpec` (Task 1: `message_id`, `mode` in {"pixels","graphics"}, `pixels` renderable, `pil` image).
- Produces: `ConsoleTranscript.set_image_specs(specs: Mapping[str, ConsoleImageRowSpec]) -> None`; image rows keyed `image:{message_id}` with signature `("image", message_id, mode)`; row order message → image → actions → action-help.

- [ ] **Step 1: Write the failing tests** (append at true EOF; this file's non-mounted tests construct `ConsoleTranscript()` directly — follow the file's existing import conventions; add `from tldw_chatbook.Chat.console_image_view import ConsoleImageRowSpec` and the PIL/Pixels imports near the new tests if the file's top-block is not extended)

```python
def _image_row_spec(message_id: str, mode: str = "pixels"):
    from io import BytesIO

    from PIL import Image as PILImage
    from rich_pixels import Pixels

    from tldw_chatbook.Chat.console_image_view import ConsoleImageRowSpec

    pil = PILImage.new("RGB", (16, 16), (10, 120, 40))
    return ConsoleImageRowSpec(
        message_id=message_id,
        mode=mode,
        pixels=Pixels.from_image(pil) if mode == "pixels" else None,
        pil=pil if mode == "graphics" else None,
    )


def test_transcript_emits_image_row_when_spec_present():
    transcript = ConsoleTranscript()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="look",
        image_data=b"\x89PNG-bytes",
        image_mime_type="image/png",
    )
    transcript.set_messages([message])
    transcript.set_image_specs({message.id: _image_row_spec(message.id)})

    rows = transcript._transcript_rows()
    kinds = [row.kind for row in rows]
    assert "image" in kinds
    image_row = next(row for row in rows if row.kind == "image")
    assert image_row.key == f"image:{message.id}"
    assert image_row.signature == ("image", message.id, "pixels")
    # Order: message row immediately precedes its image row.
    message_index = kinds.index("message")
    assert kinds[message_index + 1] == "image"


def test_transcript_omits_image_row_without_spec_or_when_hidden():
    transcript = ConsoleTranscript()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="look",
        image_data=b"\x89PNG-bytes",
        image_mime_type="image/png",
    )
    transcript.set_messages([message])
    # No specs set at all -> no image rows (unmounted-test posture).
    assert all(row.kind != "image" for row in transcript._transcript_rows())
    # Hidden mode is expressed by the screen simply omitting the spec.
    transcript.set_image_specs({})
    assert all(row.kind != "image" for row in transcript._transcript_rows())


def test_image_row_signature_stable_across_streaming_ticks():
    transcript = ConsoleTranscript()
    user = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="look",
        image_data=b"\x89PNG-bytes",
        image_mime_type="image/png",
    )
    assistant = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT, content="", status="streaming"
    )
    transcript.set_messages([user, assistant])
    transcript.set_image_specs({user.id: _image_row_spec(user.id)})

    first = next(r for r in transcript._transcript_rows() if r.kind == "image")
    assistant.content = "more streamed text"
    transcript.set_messages([user, assistant])
    second = next(r for r in transcript._transcript_rows() if r.kind == "image")
    assert first.signature == second.signature


def test_image_row_widget_builds_for_both_modes():
    transcript = ConsoleTranscript()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="look",
        image_data=b"\x89PNG-bytes",
        image_mime_type="image/png",
    )
    transcript.set_messages([message])

    transcript.set_image_specs({message.id: _image_row_spec(message.id, "pixels")})
    pixels_row = next(r for r in transcript._transcript_rows() if r.kind == "image")
    pixels_widget = transcript._build_row_widget(pixels_row, track=False)
    assert pixels_widget.id == f"console-image-{message.id}"

    transcript.set_image_specs({message.id: _image_row_spec(message.id, "graphics")})
    graphics_row = next(r for r in transcript._transcript_rows() if r.kind == "image")
    graphics_widget = transcript._build_row_widget(graphics_row, track=False)
    assert graphics_widget.id == f"console-image-{message.id}"
    assert graphics_widget.styles.max_width.value == 80
    assert graphics_widget.styles.max_height.value == 40
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_native_transcript.py -q --no-header -k "image_row or emits_image"`
Expected: FAIL with `AttributeError: ... set_image_specs`.

- [ ] **Step 3: Implement**

1. `_TranscriptRow` (~line 153): extend the `kind` Literal with `"image"` and add field `image_spec: "ConsoleImageRowSpec | None" = None`. Import `ConsoleImageRowSpec` from `tldw_chatbook.Chat.console_image_view` at the top (absolute import, matching file style).
2. `ConsoleTranscript` state: in `__init__` (or class body beside `_messages`), add `self._image_specs: dict[str, ConsoleImageRowSpec] = {}`. Add:

```python
    def set_image_specs(self, specs: Mapping[str, ConsoleImageRowSpec]) -> None:
        """Replace the prebuilt inline-image row payloads keyed by message ID.

        Args:
            specs: Mapping of message ID to its prepared image-row payload.
                Messages absent from the mapping render no image row (covers
                hidden mode, unprepared cache, and metadata-only messages).
        """
        self._image_specs = dict(specs)
```

(Add `Mapping` to the file's typing imports if absent.)
3. `_transcript_rows` (~566): immediately after appending the message row and BEFORE the `if selected:` actions block, add:

```python
            image_spec = self._image_specs.get(message.id)
            if image_spec is not None:
                rows.append(
                    _TranscriptRow(
                        key=f"image:{message.id}",
                        kind="image",
                        signature=("image", message.id, image_spec.mode),
                        message=message,
                        image_spec=image_spec,
                    )
                )
```

4. `_build_row_widget` (~679): add a branch before the final fallback:

```python
        if row.kind == "image" and row.image_spec is not None:
            return self._image_row_widget(row.image_spec)
```

and the builder + guarded import:

```python
    def _image_row_widget(self, spec: ConsoleImageRowSpec) -> Widget:
        """Build the mounted widget for one inline-image row."""
        widget: Widget | None = None
        if spec.mode == "graphics" and spec.pil is not None:
            try:
                from textual_image.widget import Image as _GraphicsImage

                widget = _GraphicsImage(spec.pil, id=f"console-image-{spec.message_id}")
            except Exception:
                logger.opt(exception=True).warning(
                    "textual-image unavailable; falling back to pixels row."
                )
                widget = None
        if widget is None:
            pixels = spec.pixels
            if pixels is None and spec.pil is not None:
                pixels = Pixels.from_image(spec.pil)
            widget = Static(
                pixels if pixels is not None else "",
                id=f"console-image-{spec.message_id}",
            )
        widget.add_class("console-transcript-image")
        widget.styles.max_width = 80
        widget.styles.max_height = 40
        return widget
```

(Imports: `from rich_pixels import Pixels` and `logger` — check what the module already imports; it uses no loguru today, so add `from loguru import logger` per repo convention.)
5. `_update_row_widget` (~707): image rows rebuild on signature change — find the method's dispatch and ensure kind `"image"` returns a NEWLY built widget (the reconciler's replace path mounts it):

```python
        if row.kind == "image" and row.image_spec is not None:
            return self._image_row_widget(row.image_spec)
```

(Match how existing kinds are handled in `_update_row_widget` — read the method first; if other kinds update in place via `.update(...)`, image rows still return a fresh widget because pixels↔graphics swaps widget CLASS.)

- [ ] **Step 4: Run tests**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_native_transcript.py -q --no-header`
Expected: all pass (pre-existing 33+ and the 4 new). Verify append-only: `git diff HEAD -- Tests/UI/test_console_native_transcript.py | grep -c '^-[^-]'` → 0.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Console/console_transcript.py Tests/UI/test_console_native_transcript.py
git commit -m "feat(console): transcript image row kind fed by prebuilt spec map

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Toggle View message action

**Files:**
- Modify: `tldw_chatbook/Chat/console_message_actions.py` (beside `_SAVE_IMAGE_ACTIONS` ~line 70, `available_actions` ~101, `dispatch` ~230)
- Test: `Tests/UI/test_console_native_transcript.py` (append-only)

**Interfaces:**
- Consumes: existing `_has_image(message)` (line ~73).
- Produces: action `("toggle-image-view", "View")` offered for image messages (same gating as save-image, appearing BEFORE it); `dispatch("toggle-image-view", message)` → `ConsoleActionResult(status="completed", visible_copy="Toggled image view.", target_message_id=message.id)`. Task 4 wires the button prefix `console-message-action-toggle-image-view-`.

- [ ] **Step 1: Write the failing test** (append at true EOF)

```python
def test_toggle_image_view_action_offered_and_dispatched_for_image_messages():
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
    assert "toggle-image-view" not in plain_ids
    assert "toggle-image-view" in image_ids
    assert image_ids.index("toggle-image-view") < image_ids.index("save-image")

    result = service.dispatch("toggle-image-view", with_image)
    assert result.status == "completed"
    assert result.visible_copy == "Toggled image view."
    assert result.target_message_id == with_image.id
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_native_transcript.py -q --no-header -k "toggle_image_view"`
Expected: FAIL (`"toggle-image-view" in image_ids` assertion).

- [ ] **Step 3: Implement**

In `console_message_actions.py`:
1. Beside `_SAVE_IMAGE_ACTIONS` (line ~70):

```python
    _IMAGE_VIEW_ACTIONS: tuple[tuple[str, str], ...] = (("toggle-image-view", "View"),)
```

2. In `available_actions` (~101), change the image-actions extension to offer View before Save Image:

```python
        if self._has_image(message):
            completed_actions = (
                completed_actions
                + list(self._IMAGE_VIEW_ACTIONS)
                + list(self._SAVE_IMAGE_ACTIONS)
            )
```

3. In `dispatch`, next to the `save-image` branch (~230):

```python
        if action_id == "toggle-image-view":
            return ConsoleActionResult(
                action_id=action_id,
                status="completed",
                visible_copy="Toggled image view.",
                target_message_id=message.id,
            )
```

4. In `console_transcript.py`'s `_ACTION_TOOLTIPS` dict, add: `"toggle-image-view": "Cycle image view: pixels, graphics, hidden."` (matching neighboring phrasing).

- [ ] **Step 4: Run tests**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_native_transcript.py -q --no-header`
Expected: all pass. Append-only check on the test file → 0 deletions.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_message_actions.py tldw_chatbook/Widgets/Console/console_transcript.py Tests/UI/test_console_native_transcript.py
git commit -m "feat(console): Toggle View message action for inline images

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Screen wiring — state, prep worker, spec map, serialization, handler

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (anchors, current line numbers: `transcript.set_messages` 6326 inside `_sync_native_console_chat_ui`-adjacent sync helper; `_serialize_native_console_state` 5683; `_restore_native_console_state` 5721; `_parse_console_message_action_button_id` 7323; action dispatch handler ~6798; session-close call sites ~8870/8882)
- Test: `Tests/UI/test_console_native_chat_flow.py` (append-only)

**Interfaces:**
- Consumes: everything from Tasks 1–3.
- Produces: `_ensure_console_image_view() -> tuple[ConsoleImageViewState, ConsoleImageRenderCache]` (lazy attrs `_console_image_view_state`, `_console_image_cache`, `_console_image_default_mode` resolved once per screen life); `_build_console_image_specs(messages) -> dict[str, ConsoleImageRowSpec]`; `_kick_console_image_prep(pending)` worker (`group="console-image-prep"`, `exclusive=True`); serialization key `"image_view_modes"`; button prefix + handler branch for `toggle-image-view`.

- [ ] **Step 1: Write the failing tests** (append at true EOF of `Tests/UI/test_console_native_chat_flow.py`; use the file's ConsoleHarness + `_configure_native_ready_console` idiom — see the Phase-1 attachment tests near the end of the file)

```python
async def test_image_message_gets_inline_row_after_prep_and_toggle_cycles():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        from io import BytesIO

        from PIL import Image as PILImage

        buffer = BytesIO()
        PILImage.new("RGB", (32, 32), (200, 10, 10)).save(buffer, format="PNG")
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="look at this",
            image_data=buffer.getvalue(),
            image_mime_type="image/png",
        )
        await console._sync_native_console_chat_ui()
        # Prep runs in a worker; wait for the image row to appear.
        for _ in range(80):
            if console.query(f"#console-image-{message.id}"):
                break
            await pilot.pause(0.05)
        assert console.query(f"#console-image-{message.id}"), "image row never appeared"

        # Toggle: pixels -> graphics (widget swaps, still present)
        console._handle_console_toggle_image_view(message.id)
        await console._sync_native_console_chat_ui()
        await pilot.pause()
        assert console.query(f"#console-image-{message.id}")

        # Toggle: graphics -> hidden (row disappears)
        console._handle_console_toggle_image_view(message.id)
        await console._sync_native_console_chat_ui()
        await pilot.pause()
        assert not console.query(f"#console-image-{message.id}")

        # Toggle: hidden -> pixels (row returns)
        console._handle_console_toggle_image_view(message.id)
        await console._sync_native_console_chat_ui()
        await pilot.pause()
        assert console.query(f"#console-image-{message.id}")


def test_image_view_modes_ride_screen_state_allowlist_and_prune_stale():
    app = _build_test_app()
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    session = store.ensure_session()
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="pic",
        image_data=b"\x89PNG-bytes",
        image_mime_type="image/png",
    )
    state, _cache = screen._ensure_console_image_view()
    state.restore({message.id: "hidden", "stale-id": "graphics"})

    payload = screen._serialize_native_console_state()
    assert payload is not None
    # Live override survives; the stale one is pruned at serialize time.
    assert payload["image_view_modes"] == {message.id: "hidden"}

    fresh = ChatScreen(app)
    fresh._restore_native_console_state(payload)
    fresh_state, _ = fresh._ensure_console_image_view()
    assert fresh_state.serialize() == {message.id: "hidden"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_native_chat_flow.py -q --no-header -k "inline_row or image_view_modes"`
Expected: FAIL with `AttributeError: ... _ensure_console_image_view`.

- [ ] **Step 3: Implement in `chat_screen.py`**

1. Imports (module top, with the other `tldw_chatbook.Chat` imports):

```python
from tldw_chatbook.Chat.console_image_view import (
    ConsoleImageRenderCache,
    ConsoleImageRowSpec,
    ConsoleImageViewState,
    next_view_mode,
    resolve_default_mode,
)
```

2. Lazy owner (near `_ensure_console_chat_store`):

```python
    def _ensure_console_image_view(self) -> tuple[ConsoleImageViewState, ConsoleImageRenderCache]:
        """Return (view state, render cache) for inline images, creating lazily."""
        if getattr(self, "_console_image_view_state", None) is None:
            self._console_image_view_state = ConsoleImageViewState()
            self._console_image_cache = ConsoleImageRenderCache()
            self._console_image_default_mode = resolve_default_mode(
                getattr(self.app_instance, "app_config", {}) or {}
            )
        return self._console_image_view_state, self._console_image_cache
```

3. Spec-map builder + prep worker:

```python
    def _build_console_image_specs(self, messages) -> dict[str, ConsoleImageRowSpec]:
        """Build image-row payloads for prepared, non-hidden image messages."""
        state, cache = self._ensure_console_image_view()
        default_mode = self._console_image_default_mode
        specs: dict[str, ConsoleImageRowSpec] = {}
        for message in messages:
            if getattr(message, "image_data", None) is None:
                continue
            mode = state.mode_for(message.id, default=default_mode)
            if mode == "hidden":
                continue
            pil = cache.get_pil(message.id)
            if pil is None:
                continue
            specs[message.id] = ConsoleImageRowSpec(
                message_id=message.id,
                mode=mode,
                pixels=cache.get_pixels(message.id) if mode == "pixels" else None,
                pil=pil if mode == "graphics" else None,
            )
        return specs

    async def _prep_console_images(self, pending: list[tuple[str, bytes]]) -> None:
        """Prepare pending transcript images off-loop, then resync once."""
        _state, cache = self._ensure_console_image_view()

        def _prepare_all() -> None:
            for message_id, image_data in pending:
                cache.prepare(message_id, image_data)

        await asyncio.to_thread(_prepare_all)
        await self._sync_native_console_chat_ui()
```

4. Hook into the transcript sync: at the `transcript.set_messages(messages)` site (line ~6326), immediately after it add:

```python
            transcript.set_image_specs(self._build_console_image_specs(messages))
            _state, cache = self._ensure_console_image_view()
            pending_images = cache.pending_ids(messages)
            if pending_images:
                self.run_worker(
                    self._prep_console_images(pending_images),
                    exclusive=True,
                    group="console-image-prep",
                )
```

5. Toggle handler + dispatch wiring:

```python
    def _handle_console_toggle_image_view(self, message_id: str) -> None:
        """Cycle one message's inline-image view mode."""
        state, _cache = self._ensure_console_image_view()
        current = state.mode_for(message_id, default=self._console_image_default_mode)
        state.set_mode(
            message_id,
            next_view_mode(current),
            default=self._console_image_default_mode,
        )
```

In `_parse_console_message_action_button_id` (~7323) add the prefix tuple entry (order within the tuple is unimportant; no prefix collision):

```python
            ("console-message-action-toggle-image-view-", "toggle-image-view"),
```

In the action-dispatch handler (~6798 region, beside the `save-image` branch):

```python
        if action_id == "toggle-image-view" and result.status == "completed":
            self._handle_console_toggle_image_view(message_id)
            await self._sync_native_console_chat_ui()
            return True
```

6. Serialization: in `_serialize_native_console_state` (~5683), add to the returned dict (prune first against all live message ids across sessions):

```python
        image_state, _ = self._ensure_console_image_view()
        live_ids = {
            message.id
            for session in store.sessions()
            for message in store.messages_for_session(session.id)
        }
        image_state.prune(live_ids)
        # inside the returned dict literal:
            "image_view_modes": image_state.serialize(),
```

In `_restore_native_console_state` (~5721), after sessions/messages restore succeeds:

```python
        image_state, cache = self._ensure_console_image_view()
        image_state.restore(payload.get("image_view_modes"))
        cache.clear()
```

7. Session-close eviction: at both `close_session(session_id)` call sites (~8870, 8882), capture the closing session's message ids BEFORE closing and evict after:

```python
                    closing_ids = [
                        m.id for m in store.messages_for_session(session_id)
                    ]
                    self._ensure_console_chat_controller().close_session(session_id)
                    _state, cache = self._ensure_console_image_view()
                    cache.evict_session(closing_ids)
```

(Adapt to each call site's local structure — read both first; if they share a helper, put the eviction in the helper once.)

- [ ] **Step 4: Run tests**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_native_chat_flow.py -q --no-header` (full file, ~3 min)
Then: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/ Tests/UI/test_console_native_transcript.py -q --no-header`
Expected: all pass. Append-only check on the flow test file → 0 deletions.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_native_chat_flow.py
git commit -m "feat(console): wire inline image rendering — prep worker, toggle handler, screen state

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
  Tests/UI/test_console_workbench_contract.py \
  Tests/UI/test_console_internals_decomposition.py \
  Tests/UI/test_chat_image_attachment.py \
  Tests/Event_Handlers/Chat_Events/test_chat_image_events.py \
  Tests/Event_Handlers/Chat_Events/test_chat_image_properties.py \
  Tests/unit/test_chat_image_unit.py \
  Tests/DB/test_chat_image_db_compatibility.py \
  Tests/Widgets/test_chat_message_enhanced.py \
  -q --no-header
```

Expected: 0 real failures (the two cursor-blink tests in internals_decomposition are documented load flakes — verify isolated if they fail).

- [ ] **Step 2: Live QA captures** — textual-serve rig per `Docs/superpowers/qa/console-attachments-2026-07/README.md` (isolated HOME, ready-seeded llama_cpp config, real send with the vision-override recipe or restore a seeded image conversation). Required captures into `Docs/superpowers/qa/console-inline-images-2026-07/`:
  1. `inline-pixels-default.png` — image message rendered inline in pixels mode (serve default).
  2. `toggle-graphics.png` — after Toggle View: graphics mode (halfcell under serve — record as expected; TGP evidence optional via local kitty).
  3. `toggle-hidden-chip-only.png` — second toggle: row collapsed, chip remains.
  4. `image-with-actions-row.png` — selected image message showing View + Save Image in the action row (note the actions-below-image consequence for the gate).
  5. `resume-inline-rehydrated.png` — relaunch + resume: image preps and renders again from DB bytes.
  Write a README (rig, commit, captures, defects), commit evidence.

- [ ] **Step 3: Visual approval gate** — present captures to the user; do NOT merge or open a PR without explicit approval (standing Console rule).

- [ ] **Step 4: Wrap-up** — superpowers:finishing-a-development-branch (PR to dev on approval; update TASK-215 backlog file: In Progress → Done with Implementation Notes, riding the PR).

---

## Deferred (do not implement)

Clipboard paste/drag-drop (TASK-216), multi-attachment (TASK-217), config-driven filter/caps unification (TASK-222), `[image omitted]` payload placeholder (TASK-224), legacy chat rendering changes.
