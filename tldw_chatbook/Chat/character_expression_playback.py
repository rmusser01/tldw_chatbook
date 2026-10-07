"""Bounded, off-thread preparation and elapsed-time character presentation."""

from __future__ import annotations

import hashlib
import math
import threading
import warnings
from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from tldw_chatbook.Utils.Utils import coerce_bool_flag

MAX_PREPARATION_BYTES = 64 * 1024 * 1024
_DECODE_LOCK = threading.Lock()
_BUDGET_LOCK = threading.Lock()
_preparation_bytes = 0

# Shared decode retention (task-16/F13): the Visual Identity resolver's
# deliberate corruption-check decode seek/loads every frame anyway, so it
# deposits the composited RGBA frames here -- bounded by a dedicated cap that
# is deliberately independent of MAX_PREPARATION_BYTES so retained decodes can
# never crowd out (or fail because of) live animation preparations. Total
# process RGBA memory therefore stays bounded at 64 MiB live + 16 MiB shared.
_RETAINED_DECODE_ENTRY_LIMIT = 8
_RETAINED_DECODE_BYTES_LIMIT = 16 * 1024 * 1024
_retained_decodes: OrderedDict[tuple[str, ...], _SharedDecodedFrames] = OrderedDict()
_retained_decodes_bytes = 0
_RETAINED_DECODES_LOCK = threading.Lock()


@dataclass(slots=True)
class _SharedDecodedFrames:
    """Composited full-size frames and timing captured during one decode.

    Frames are shared read-only inputs for ``_prepare_from_shared`` and are
    never explicitly closed: eviction and reset only drop the store's
    reference, so an in-flight resize on another thread keeps its own
    reference alive and the buffers are released by refcounting once every
    consumer is done. (The retained frames are in-memory ``convert`` results
    with no file handle to release.)
    """

    frames: tuple[Any, ...]
    durations_raw: tuple[Any, ...]
    loop_raw: Any
    default_image: bool
    image_format: str
    size: tuple[int, int]
    owned_bytes: int


class SharedFrameRetention:
    """Collect composited RGBA frames during one decode for shared reuse.

    The resolver's inspection decode is the only producer. Frames that do not
    fit the shared caps are closed by the store and retention stops for the
    rest of that decode; nothing here ever fails the decode itself.
    """

    __slots__ = (
        "_bytes",
        "_default_image",
        "_finished",
        "_frames",
        "_image_format",
        "_key",
        "_loop_raw",
        "_size",
    )

    def __init__(
        self,
        key: tuple[str, ...],
        *,
        loop_raw: Any,
        default_image: bool,
        image_format: str,
        size: tuple[int, int],
    ) -> None:
        self._key = key
        self._loop_raw = loop_raw
        self._default_image = default_image
        self._image_format = image_format
        self._size = size
        self._frames: list[Any] = []
        self._bytes = 0
        self._finished = False

    def accept(self, frame: Any) -> bool:
        """Offer one composited frame; False means stop offering entirely."""

        if self._finished or self._bytes >= _RETAINED_DECODE_BYTES_LIMIT:
            frame.close()
            return False
        self._frames.append(frame)
        self._bytes += frame.width * frame.height * 4
        return True

    def seal(self, durations_raw: tuple[Any, ...]) -> None:
        """Publish the collected frames with their raw per-frame durations."""

        global _retained_decodes_bytes
        if self._finished:
            return
        self._finished = True
        if not self._frames or len(durations_raw) != len(self._frames):
            self._release_collected()
            return
        with _RETAINED_DECODES_LOCK:
            if self._key in _retained_decodes:
                self._release_collected()
                return
            while _retained_decodes and (
                len(_retained_decodes) + 1 > _RETAINED_DECODE_ENTRY_LIMIT
                or _retained_decodes_bytes + self._bytes > _RETAINED_DECODE_BYTES_LIMIT
            ):
                # Eviction drops the store reference only (frames are
                # GC-managed; see _SharedDecodedFrames).
                _, evicted = _retained_decodes.popitem(last=False)
                _retained_decodes_bytes -= evicted.owned_bytes
            if (
                len(_retained_decodes) + 1 > _RETAINED_DECODE_ENTRY_LIMIT
                or _retained_decodes_bytes + self._bytes > _RETAINED_DECODE_BYTES_LIMIT
            ):
                self._release_collected()
                return
            _retained_decodes[self._key] = _SharedDecodedFrames(
                frames=tuple(self._frames),
                durations_raw=durations_raw,
                loop_raw=self._loop_raw,
                default_image=self._default_image,
                image_format=self._image_format,
                size=self._size,
                owned_bytes=self._bytes,
            )
            _retained_decodes_bytes += self._bytes

    def abort(self) -> None:
        """Discard collected frames (decode failed or caps were exceeded)."""

        if self._finished:
            return
        self._finished = True
        self._release_collected()

    def _release_collected(self) -> None:
        self._frames = []


def _reset_shared_decode_state() -> None:
    """Forget every retained shared decode (test isolation only)."""

    global _retained_decodes_bytes
    with _RETAINED_DECODES_LOCK:
        _retained_decodes.clear()
        _retained_decodes_bytes = 0


def _retained_shared_decode(key: tuple[str, ...]) -> _SharedDecodedFrames | None:
    with _RETAINED_DECODES_LOCK:
        entry = _retained_decodes.get(key)
        if entry is not None:
            _retained_decodes.move_to_end(key)
        return entry


def _decode_frame(image: Any, index: int) -> None:
    """Seek to and fully decode one frame (decode-work seam for tests)."""

    image.seek(index)
    image.load()


def normalize_expression_mode(value: object) -> str:
    """Return a supported local preference, defaulting to Dynamic."""
    value = str(value).strip().lower()
    return value if value in {"dynamic", "static"} else "dynamic"


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
        and coerce_bool_flag(appearance.get("animations_enabled"), True)
        and not coerce_bool_flag(appearance.get("reduce_motion"), False)
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


def _normalized_plays(raw_loop: Any, image_format: str) -> int:
    """Normalize the encoded loop count exactly as the decode path always has."""

    if raw_loop is None:
        return 1
    if not isinstance(raw_loop, int) or raw_loop < 0:
        raise ValueError("Invalid animation loop count")
    return raw_loop + 1 if image_format == "GIF" and raw_loop else raw_loop


def _shared_content_key(data: bytes) -> tuple[str, ...]:
    """Derive the content-identity decode key for unattributed image bytes."""

    return ("vi-decode-v1", "content", hashlib.sha256(data).hexdigest())


def prepare_expression(
    data: bytes,
    size: tuple[int, int],
    *,
    animate: bool,
    identity: tuple[str, ...] | None = None,
) -> PreparedExpression:
    """Decode validated image bytes within native limits and the shared RGBA budget.

    Args:
        data: Immutable bytes admitted by the Visual Identity resolver.
        size: Maximum prepared pixel dimensions, independent of source dimensions.
        animate: Decode the visible animation, or only encoded frame zero.
        identity: Optional decode identity carried by the resolver result. When
            it matches a retained shared decode, frames are resized straight
            from the composited originals with zero additional frame decoding.

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
    key = tuple(identity) if identity is not None else _shared_content_key(data)
    shared = _retained_shared_decode(key)
    if shared is not None and (
        shared.frames
        and len(shared.frames) == len(shared.durations_raw)
        and shared.size[0] * shared.size[1] > 0
    ):
        return _prepare_from_shared(
            shared, size, animate=animate, limits=(
                MAX_EXPRESSION_IMAGE_DIMENSION,
                MAX_EXPRESSION_FRAME_COUNT,
                MAX_EXPRESSION_ASSET_DECODED_PIXELS,
            )
        )
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
                plays = _normalized_plays(image.info.get("loop"), image.format or "")
                delays = []
                for index in range(start, start + selected_count):
                    _decode_frame(image, index)
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
                    _decode_frame(image, 0)
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


def _prepare_from_shared(
    shared: _SharedDecodedFrames,
    size: tuple[int, int],
    *,
    animate: bool,
    limits: tuple[int, int, int],
) -> PreparedExpression:
    """Build a PreparedExpression from retained composited frames, zero decodes.

    Budget sequencing, target math, delay validation, and both fallback modes
    mirror the decode path exactly so rendered output is pixel-identical; the
    retained frames are the same ``convert("RGBA")`` results that path built
    before resizing. The decoder-scratch reservation (``canvas * 3``) is kept
    for budget-decision parity even though no codec runs here.
    """

    from PIL import Image

    (
        max_dimension,
        max_frame_count,
        max_decoded_pixels,
    ) = limits
    width, height = shared.size
    canvas = width * height * 4
    if max(width, height) > max_dimension:
        raise ValueError("Character expression image limits exceeded")
    frames: list[Any] = []
    reserved = 0
    try:
        _adjust_budget(canvas * 3)
        reserved = canvas * 3
        count = len(shared.frames)
        if count > max_frame_count or width * height * count > max_decoded_pixels:
            raise ValueError("Character expression image limits exceeded")
        start = 1 if animate and shared.default_image else 0
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
        plays = _normalized_plays(shared.loop_raw, shared.image_format)
        delays = []
        for offset in range(selected_count):
            # Retained frames are shared read-only inputs: never closed here.
            frames.append(
                shared.frames[start + offset].resize(target, Image.Resampling.LANCZOS)
            )
            duration = shared.durations_raw[start + offset]
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
            frames.append(shared.frames[0].resize(target, Image.Resampling.LANCZOS))
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
