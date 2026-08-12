"""Structured video-generation metadata for Console transcript rows (task-3401.4).

Facts ABOUT a generated video: its slug name (the only durable reference --
ADR-044), the prompt/backend/model/seed that produced it, and its shape
(duration/fps/dimensions/ratio). Everything a tombstone card needs to render
and a regenerate action needs to rebuild the request after the video bytes
are gone.

Persisted as a namespaced top-level key in the LOCAL-ONLY
``messages.metadata_json`` column (schema v31, task-2364) -- no migration,
no path, no URL. The column is shared with
``Chat.message_metadata.MessageMetadata`` by mutual exclusion: a video
generation row never carries turn-provenance facts and a provenance row is
never a video row. Both parsers degrade gracefully on the foreign shape --
``MessageMetadata.from_json`` on a video payload yields an all-defaults
(``is_empty``) instance, and ``VideoGenerationMetadata.from_json`` on a
provenance payload returns ``None``.

Why not the v25 ``message_generation_metadata`` sidecar: its ``position``
column is defined by index alignment with the message's attachments, and a
video message has no attachments (bytes live in the ephemeral VideoStore) --
a position-0 row without attachment 0 would break every reader built on
that invariant.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from tldw_chatbook.Video_Generation.video_formats import canonical_video_extension

#: Top-level key namespacing this payload inside ``metadata_json``.
VIDEO_METADATA_TOP_KEY = "video_generation"


@dataclass(frozen=True, slots=True)
class VideoGenerationMetadata:
    """Structured facts about one generated video message.

    Attributes:
        name: Filesystem-safe slug naming the video (the conversation's only
            durable reference -- ADR-044). Non-empty at construction.
        prompt: Generation prompt text.
        negative_prompt: Negative prompt (``""`` when none).
        backend: Backend id that produced the video (e.g. ``"minimax"``).
        container: Canonical generated-video container (``"mp4"`` or ``"webm"``).
        model: Model identifier, or ``None`` when not known/recorded.
        seed: Seed used, or ``None`` when random/unknown.
        duration_seconds: Clip length, or ``None`` when unknown.
        fps: Frames per second, or ``None`` when unknown.
        width: Frame width in pixels, or ``None`` when unknown.
        height: Frame height in pixels, or ``None`` when unknown.
        ratio: Aspect ratio string (``"16:9"``, ``"adaptive"``), or ``None``.
        source_image_message_id: For image-to-video, the persisted id of the
            image message this video was animated from (task-3401.8), or
            ``None``.

    Raises:
        ValueError: If ``name`` or ``backend`` is empty -- refused at
            construction so a missing slug/backend fails at the call site,
            mirroring ``MessageMetadata``'s closed-vocabulary rule.
    """

    name: str
    prompt: str
    backend: str
    negative_prompt: str = ""
    model: str | None = None
    seed: int | None = None
    duration_seconds: float | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    ratio: str | None = None
    source_image_message_id: str | None = None
    container: str = "mp4"

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("name must be non-empty (it is the only durable reference)")
        if not str(self.backend).strip():
            raise ValueError("backend must be non-empty")
        canonical_video_extension(self.container)

    def to_json(self) -> str:
        """Serialize for the ``messages.metadata_json`` column.

        Returns:
            A stable (key-sorted) JSON object string namespaced under
            :data:`VIDEO_METADATA_TOP_KEY`.
        """
        payload = {
            "name": self.name,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "backend": self.backend,
            "container": self.container,
            "model": self.model,
            "seed": self.seed,
            "duration_seconds": self.duration_seconds,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "ratio": self.ratio,
            "source_image_message_id": self.source_image_message_id,
        }
        return json.dumps({VIDEO_METADATA_TOP_KEY: payload}, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str | None) -> "VideoGenerationMetadata | None":
        """Rebuild from a stored payload, degrading instead of raising.

        Runs on the resume path against durable data that may be a
        provenance payload, a newer build's shape, or corrupt -- none of
        which is worth failing a conversation load over.

        Args:
            raw: The stored ``metadata_json`` string, or ``None``.

        Returns:
            The decoded metadata, or ``None`` when the payload is missing,
            unparseable, or carries no :data:`VIDEO_METADATA_TOP_KEY` object.
            Numeric fields coerce to their declared types or drop to
            ``None``; a payload whose ``name``/``backend`` would fail
            construction also degrades to ``None``.
        """
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(data, dict):
            return None
        payload = data.get(VIDEO_METADATA_TOP_KEY)
        if not isinstance(payload, dict):
            return None
        container = "mp4" if "container" not in payload else payload["container"]
        try:
            return cls(
                name=_as_text(payload.get("name")),
                prompt=_as_text(payload.get("prompt")),
                negative_prompt=_as_text(payload.get("negative_prompt")),
                backend=_as_text(payload.get("backend")),
                container=container,
                model=_as_optional_text(payload.get("model")),
                seed=_as_optional_int(payload.get("seed")),
                duration_seconds=_as_optional_float(payload.get("duration_seconds")),
                fps=_as_optional_float(payload.get("fps")),
                width=_as_optional_int(payload.get("width")),
                height=_as_optional_int(payload.get("height")),
                ratio=_as_optional_text(payload.get("ratio")),
                source_image_message_id=_as_optional_text(payload.get("source_image_message_id")),
            )
        except ValueError:
            return None


def _as_text(value: Any) -> str:
    return str(value) if value else ""


def _as_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _as_optional_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None
