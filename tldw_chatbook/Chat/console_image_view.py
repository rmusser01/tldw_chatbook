"""Console inline-image view modes, per-message state, and render cache.

Pure module: no Textual imports (rich_pixels renders via Rich, PIL is
imaging). The graphics-mode widget itself is created in the transcript.

This module is the FIRST real consumer of ``[chat.images].default_render_mode``,
``[chat.images.terminal_overrides]``, and ``terminal_utils`` detection —
legacy chat defines those keys but never reads them.
"""

from __future__ import annotations

import threading
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
    if not isinstance(app_config, Mapping):
        return {}
    # Same both-shapes handling as config.resolve_tldw_api_config: the live app_config nests raw TOML under COMPREHENSIVE_CONFIG_RAW.
    raw = app_config.get("COMPREHENSIVE_CONFIG_RAW")
    source = raw if isinstance(raw, Mapping) else app_config
    chat = source.get("chat")
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
    if configured not in ("auto", ""):
        # Anything unrecognized (not "pixels"/"regular", not "auto", not
        # missing/empty) pins to "pixels" immediately rather than falling
        # through the terminal-auto path.
        return "pixels"

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

    Thread model: a reentrant lock guards all cache state; ``prepare`` runs
    in a worker thread while the event loop reads concurrently. Heavy PIL
    work happens outside the lock.
    """

    def __init__(self, *, max_entries: int = IMAGE_CACHE_MAX_ENTRIES) -> None:
        self._max_entries = max_entries
        self._images: OrderedDict[str, PILImage.Image] = OrderedDict()
        self._pixels: dict[str, Pixels] = {}
        self._failed: set[str] = set()
        self._lock = threading.RLock()

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
            with self._lock:
                self._failed.add(message_id)
            return False
        with self._lock:
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
        with self._lock:
            pil = self._images.get(message_id)
            if pil is not None:
                self._images.move_to_end(message_id)
            return pil

    def get_pixels(self, message_id: str) -> Pixels | None:
        """Return (lazily building) the pixels renderable for a cached image."""
        with self._lock:
            cached = self._pixels.get(message_id)
            if cached is not None:
                return cached
            pil = self._images.get(message_id)
            if pil is None:
                return None
            self._images.move_to_end(message_id)
            pil = pil.copy()
        # Half-block rendering: one text line shows two pixel rows. Heavy
        # PIL/Pixels work happens outside the lock.
        pil.thumbnail(
            (PIXELS_MAX_COLS, PIXELS_MAX_LINES * 2), PILImage.Resampling.LANCZOS
        )
        pixels = Pixels.from_image(pil)
        with self._lock:
            # A racing evict may have dropped this entry while we were
            # thumbnailing; only store if it's still live, else drop the
            # now-orphaned pixels on the floor.
            if message_id in self._images:
                self._pixels[message_id] = pixels
        return pixels

    def is_failed(self, message_id: str) -> bool:
        """Return whether decoding previously failed for this message."""
        with self._lock:
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
        with self._lock:
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
        with self._lock:
            for message_id in message_ids:
                self._images.pop(message_id, None)
                self._pixels.pop(message_id, None)
                self._failed.discard(message_id)

    def clear(self) -> None:
        """Drop all cache state (used on full store restore)."""
        with self._lock:
            self._images.clear()
            self._pixels.clear()
            self._failed.clear()
