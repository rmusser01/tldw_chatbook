"""Bounded, off-thread preparation and elapsed-time character presentation."""

from __future__ import annotations

import math
import threading
import warnings
from bisect import bisect_right
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from typing import Any

MAX_PREPARATION_BYTES = 64 * 1024 * 1024
_DECODE_LOCK = threading.Lock()
_BUDGET_LOCK = threading.Lock()
_preparation_bytes = 0


def normalize_expression_mode(value: object) -> str:
    """Return a supported local preference, defaulting to Dynamic."""
    value = str(value).strip().lower()
    return value if value in {"dynamic", "static"} else "dynamic"


def _bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, int)):
        text = str(value).lower()
        if text in {"true", "1", "yes", "on"}:
            return True
        if text in {"false", "0", "no", "off"}:
            return False
    return default


def expression_motion_enabled(
    config: Mapping[str, Any], *, react: bool, manual: bool
) -> bool:
    """Apply global motion preferences without changing reaction precedence."""
    appearance = config.get("appearance", {})
    if not isinstance(appearance, Mapping):
        appearance = {}
    return (
        (react or manual)
        and normalize_expression_mode(appearance.get("character_expression_mode"))
        == "dynamic"
        and _bool(appearance.get("animations_enabled"), True)
        and not _bool(appearance.get("reduce_motion"), False)
    )


def preparation_bytes() -> int:
    """Return accounted active and in-flight RGBA bytes (not process RSS)."""
    with _BUDGET_LOCK:
        return _preparation_bytes


def _adjust_budget(amount: int) -> None:
    global _preparation_bytes
    with _BUDGET_LOCK:
        if amount > 0 and _preparation_bytes + amount > MAX_PREPARATION_BYTES:
            raise ValueError("Character animation preparation budget exceeded")
        _preparation_bytes += amount


@dataclass
class PreparedExpression:
    """Owned downscaled composited frames; caller must close after final use."""

    frames: tuple[Any, ...]
    durations_ms: tuple[int, ...]
    plays: int
    fallback_reason: str = ""
    _owned_bytes: int = 0

    def frame_at(self, elapsed_ms: float) -> tuple[int, bool]:
        """Select by elapsed visible time; zero plays means infinite repetition."""
        if len(self.frames) < 2:
            return 0, True
        total = sum(self.durations_ms)
        elapsed_ms = max(0, elapsed_ms)
        if self.plays and elapsed_ms >= total * self.plays:
            return len(self.frames) - 1, True
        position = elapsed_ms % total
        boundaries = []
        end = 0
        for delay in self.durations_ms:
            end += delay
            boundaries.append(end)
        return bisect_right(boundaries, position), False

    def close(self) -> None:
        """Release owned buffers once; no renderer may still be using them."""
        for frame in self.frames:
            frame.close()
        self.frames = ()
        if self._owned_bytes:
            _adjust_budget(-self._owned_bytes)
            self._owned_bytes = 0


def expression_image_size(data: bytes) -> tuple[int, int]:
    """Read dimensions without loading pixel buffers; validation precedes this call."""
    from PIL import Image

    with Image.open(BytesIO(data)) as image:
        return image.size


def prepare_expression(
    data: bytes, size: tuple[int, int], *, animate: bool
) -> PreparedExpression:
    """Decode validated image bytes within native limits and the shared RGBA budget.

    Args:
        data: Immutable bytes admitted by the Visual Identity resolver.
        size: Maximum prepared pixel dimensions, independent of source dimensions.
        animate: Decode the visible animation, or only encoded frame zero.

    Returns:
        Owned composited frames, normalized total plays and optional fallback reason.

    Raises:
        ValueError: Invalid image or a limit that prevents even safe frame-zero decode.
    """
    from PIL import Image

    from tldw_chatbook.Character_Chat.visual_identity import (
        MAX_EXPRESSION_ASSET_BYTES,
        MAX_EXPRESSION_ASSET_DECODED_PIXELS,
        MAX_EXPRESSION_FRAME_COUNT,
        MAX_EXPRESSION_IMAGE_DIMENSION,
    )

    if not data or len(data) > MAX_EXPRESSION_ASSET_BYTES or min(size) < 1:
        raise ValueError("Character expression image limits exceeded")
    frames: list[Any] = []
    reserved = 0
    # Threads are not cancelled while decoding. Serializing here also bounds
    # obsolete jobs whose widgets were removed while a codec was running.
    with _DECODE_LOCK, warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        try:
            with Image.open(BytesIO(data)) as image:
                width, height = image.size
                canvas = width * height * 4
                if max(width, height) > MAX_EXPRESSION_IMAGE_DIMENSION:
                    raise ValueError("Character expression image limits exceeded")
                # Account decoder canvas, conversion and resize scratch before
                # n_frames/seek/load, including GIF plugins that scan to count.
                _adjust_budget(canvas * 3)
                reserved = canvas * 3
                count = int(getattr(image, "n_frames", 1))
                if (
                    count > MAX_EXPRESSION_FRAME_COUNT
                    or width * height * count > MAX_EXPRESSION_ASSET_DECODED_PIXELS
                ):
                    raise ValueError("Character expression image limits exceeded")
                start = 1 if animate and image.info.get("default_image") else 0
                selected_count = count - start if animate else 1
                ratio = min(1.0, size[0] / width, size[1] / height)
                target = max(1, int(width * ratio)), max(1, int(height * ratio))
                frame_bytes = target[0] * target[1] * 4
                reason = ""
                try:
                    _adjust_budget(frame_bytes * selected_count)
                except ValueError:
                    start, selected_count = 0, 1
                    reason = "Animation exceeds the preparation budget; showing a still frame."
                    _adjust_budget(frame_bytes)
                reserved += frame_bytes * selected_count
                raw_loop = image.info.get("loop")
                if raw_loop is None:
                    plays = 1
                elif not isinstance(raw_loop, int) or raw_loop < 0:
                    raise ValueError("Invalid animation loop count")
                else:
                    plays = (
                        raw_loop + 1 if image.format == "GIF" and raw_loop else raw_loop
                    )
                delays = []
                for index in range(start, start + selected_count):
                    image.seek(index)
                    image.load()
                    # Pillow seek/load provides disposal-composited frames.
                    rgba = image.convert("RGBA")
                    try:
                        frame = rgba.resize(target, Image.Resampling.LANCZOS)
                    finally:
                        rgba.close()
                    frames.append(frame)
                    duration = image.info.get("duration", 0)
                    valid_delay = (
                        isinstance(duration, (int, float))
                        and math.isfinite(duration)
                        and duration > 0
                    )
                    delays.append(max(1, round(duration)) if valid_delay else 0)
                if len(frames) > 1 and not all(delays):
                    for frame in frames:
                        frame.close()
                    frames.clear()
                    image.seek(0)
                    image.load()
                    rgba = image.convert("RGBA")
                    try:
                        frames.append(rgba.resize(target, Image.Resampling.LANCZOS))
                    finally:
                        rgba.close()
                    delays = [0]
                    reason = "Invalid animation timing; showing a still frame."
                owned = frame_bytes * len(frames)
                result = PreparedExpression(
                    tuple(frames), tuple(delays), plays, reason, owned
                )
                _adjust_budget(-(reserved - owned))
                reserved = 0
                return result
        except Exception as exc:
            for frame in frames:
                frame.close()
            if isinstance(exc, (MemoryError, ValueError)):
                raise ValueError(str(exc)) from exc  # noqa: TRY004 - one public decode failure contract
            raise ValueError("Character expression could not be decoded") from exc
        finally:
            if reserved:
                _adjust_budget(-reserved)
