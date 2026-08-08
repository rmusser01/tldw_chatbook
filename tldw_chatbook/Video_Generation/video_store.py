"""Ephemeral, name-referenced storage for generated videos (task-3401.4, ADR-044).

Video bytes NEVER enter the database. They live under
``<user_data_dir>/generated_videos/<message_id>/<slug>.<ext>`` -- keyed by
the STABLE message id (not the ephemeral console session id, which would
orphan ttl-retained files across restarts). Conversations reference a video
only by a ``[video] <slug>`` content marker plus
``video_metadata.VideoGenerationMetadata``; this store resolves the name to
a live file and reports it missing after restart or expiry. Missing is a
normal state, not an error: the card renders a named tombstone and offers
regenerate.

Retention (``[video_generation] retention``):

- ``session`` (default): everything is wiped on app start -- the strictest
  reading of "not stored permanently".
- ``ttl``: files survive restarts up to ``retention_ttl_hours`` hours.

In BOTH modes a total size cap (``max_store_mb``) is enforced oldest-first,
so even a long-running session cannot grow the store unboundedly.
"""

from __future__ import annotations

import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from loguru import logger

from tldw_chatbook.Utils.paths import get_user_data_dir
from tldw_chatbook.Video_Generation.config import (
    get_video_generation_config,
)

VIDEO_MARKER_PREFIX = "[video] "
"""Prefix identifying a video card's content marker in a message row."""

_SLUG_MAX_LEN = 48
_SLUG_MAX_SUFFIX_ATTEMPTS = 99
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9._-]+$")
_NON_SLUG_CHARS = re.compile(r"[^a-z0-9]+")


def video_content_marker(slug: str) -> str:
    """Render the single-line content marker for a video message row."""
    return VIDEO_MARKER_PREFIX + slug


def parse_video_marker(content: str) -> str | None:
    """Extract the slug from a video marker line, or ``None``.

    Args:
        content: Message content to inspect.

    Returns:
        The slug when ``content`` starts with :data:`VIDEO_MARKER_PREFIX`
        followed by a non-empty slug; otherwise ``None``. Only the first
        whitespace-delimited token after the prefix is read.
    """
    if not content.startswith(VIDEO_MARKER_PREFIX):
        return None
    rest = content[len(VIDEO_MARKER_PREFIX):].strip()
    if not rest:
        return None
    return rest.split(None, 1)[0]


def slugify_prompt(prompt: str, *, max_words: int = 6) -> str:
    """Derive a filesystem-safe slug from a prompt.

    Lowercase, alphanumeric runs joined by dashes, first ``max_words``
    words, truncated to ``_SLUG_MAX_LEN`` chars without a trailing dash.
    Falls back to ``"clip"`` when nothing alphanumeric survives.
    """
    normalized = _NON_SLUG_CHARS.sub("-", prompt.strip().lower())
    words = [word for word in normalized.split("-") if word][:max_words]
    slug = "-".join(words)[:_SLUG_MAX_LEN].rstrip("-")
    return slug or "clip"


@dataclass(frozen=True)
class StoredVideo:
    """One video file present under the store root."""

    message_id: str
    slug: str
    path: Path
    size_bytes: int
    mtime: float


@dataclass(frozen=True)
class RetentionReport:
    """Outcome of one :meth:`VideoStore.enforce_retention` pass."""

    removed_files: int
    removed_bytes: int
    evicted: tuple[tuple[str, str], ...]  # (message_id, slug) pairs, oldest first


class VideoStore:
    """Resolves and retains ephemeral generated-video files.

    Args:
        root: Store root directory. Defaults to
            ``<user_data_dir>/generated_videos``. Tests inject a tmp path.
        config: Optional ``VideoGenerationConfig`` (retention/ttl/cap);
            defaults to the live config at call time.
    """

    def __init__(self, root: Path | None = None, *, config=None) -> None:
        self._root = (root or (get_user_data_dir() / "generated_videos")).expanduser()
        self._config = config

    @property
    def root(self) -> Path:
        return self._root

    def _get_config(self):
        return self._config if self._config is not None else get_video_generation_config()

    # -- path safety ------------------------------------------------------

    def _message_dir(self, message_id: str) -> Path:
        """Return the per-message directory, rejecting unsafe components.

        Raises:
            ValueError: If ``message_id`` contains anything outside
                ``[A-Za-z0-9._-]`` (path separators, traversal, whitespace).
        """
        if not message_id or not _SAFE_COMPONENT.match(message_id):
            raise ValueError(f"unsafe message_id component: {message_id!r}")
        return self._root / message_id

    def _video_path(self, message_id: str, slug: str, extension: str) -> Path:
        if not slug or not _SAFE_COMPONENT.match(slug):
            raise ValueError(f"unsafe slug component: {slug!r}")
        safe_ext = re.sub(r"[^a-z0-9]", "", extension.lower()) or "mp4"
        candidate = (self._message_dir(message_id) / f"{slug}.{safe_ext}")
        resolved_root = self._root.resolve()
        resolved = candidate.resolve()
        if resolved != resolved_root and resolved_root not in resolved.parents:
            raise ValueError(f"path escapes store root: {candidate}")
        return candidate

    # -- write/resolve ----------------------------------------------------

    def allocate_slug(self, message_id: str, prompt: str) -> str:
        """Return a collision-free slug for a new video under ``message_id``.

        First choice is :func:`slugify_prompt`; collisions (an earlier video
        file with the same name still present) get ``-2``, ``-3`` …

        Raises:
            ValueError: If no suffix below ``_SLUG_MAX_SUFFIX_ATTEMPTS`` is free.
        """
        base = slugify_prompt(prompt)
        for attempt in range(1, _SLUG_MAX_SUFFIX_ATTEMPTS + 1):
            slug = base if attempt == 1 else f"{base}-{attempt}"
            if self.resolve(message_id, slug) is None:
                return slug
        raise ValueError(f"no free slug for base {base!r} under message {message_id!r}")

    def save(
        self,
        message_id: str,
        slug: str,
        content: bytes,
        *,
        extension: str = "mp4",
    ) -> Path:
        """Write video bytes under the message's directory; return the path."""
        if not content:
            raise ValueError("refusing to save an empty video payload")
        path = self._video_path(message_id, slug, extension)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        logger.debug("VideoStore: saved {} bytes to {}", len(content), path.name)
        return path

    def resolve(self, message_id: str, slug: str, *, extension: str = "mp4") -> Path | None:
        """Resolve a name to a live file, or ``None`` when missing/expired.

        Missing is the normal post-restart/post-expiry state (ADR-044
        tombstone), never an error. Unsafe components resolve to ``None``
        rather than raising -- resolution is a read path against durable,
        possibly hand-edited names.
        """
        try:
            path = self._video_path(message_id, slug, extension)
        except ValueError:
            return None
        return path if path.is_file() else None

    def iter_stored(self) -> Iterator[StoredVideo]:
        """Yield every video file under the root (any message, any extension)."""
        if not self._root.is_dir():
            return
        for message_dir in sorted(self._root.iterdir()):
            if not message_dir.is_dir():
                continue
            for path in sorted(message_dir.iterdir()):
                if not path.is_file():
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    continue
                yield StoredVideo(
                    message_id=message_dir.name,
                    slug=path.stem,
                    path=path,
                    size_bytes=stat.st_size,
                    mtime=stat.st_mtime,
                )

    # -- retention --------------------------------------------------------

    def enforce_retention(self, *, now: float | None = None) -> RetentionReport:
        """Apply the configured retention policy; return what was removed.

        ``session`` mode removes everything (call on app start). ``ttl``
        mode removes files older than ``retention_ttl_hours``. Both modes
        then enforce ``max_store_mb`` oldest-first.
        """
        config = self._get_config()
        now = time.time() if now is None else now
        removed_files = 0
        removed_bytes = 0
        evicted: list[tuple[str, str]] = []

        stored = list(self.iter_stored())
        retention = getattr(config, "retention", "session")
        ttl_seconds = max(1, int(getattr(config, "retention_ttl_hours", 24))) * 3600

        survivors: list[StoredVideo] = []
        for video in stored:
            expired = retention == "session" or (now - video.mtime) > ttl_seconds
            if expired:
                removed_files, removed_bytes = self._remove(
                    video, removed_files, removed_bytes, evicted
                )
            else:
                survivors.append(video)

        max_bytes = max(1, int(getattr(config, "max_store_mb", 2048))) * 1024 * 1024
        total = sum(video.size_bytes for video in survivors)
        for video in sorted(survivors, key=lambda v: v.mtime):  # oldest first
            if total <= max_bytes:
                break
            total -= video.size_bytes
            removed_files, removed_bytes = self._remove(
                video, removed_files, removed_bytes, evicted
            )

        self._prune_empty_dirs()
        if evicted:
            logger.info(
                "VideoStore retention: removed {} file(s), {} byte(s), mode={}",
                removed_files,
                removed_bytes,
                retention,
            )
        return RetentionReport(
            removed_files=removed_files,
            removed_bytes=removed_bytes,
            evicted=tuple(evicted),
        )

    @staticmethod
    def _remove(
        video: StoredVideo,
        removed_files: int,
        removed_bytes: int,
        evicted: list[tuple[str, str]],
    ) -> tuple[int, int]:
        try:
            video.path.unlink()
            evicted.append((video.message_id, video.slug))
            return removed_files + 1, removed_bytes + video.size_bytes
        except OSError as exc:
            logger.warning("VideoStore: failed to remove {}: {}", video.path, exc)
            return removed_files, removed_bytes

    def _prune_empty_dirs(self) -> None:
        """Remove now-empty per-message directories (and the root's husk)."""
        if not self._root.is_dir():
            return
        for message_dir in sorted(self._root.iterdir()):
            if message_dir.is_dir():
                try:
                    message_dir.rmdir()  # only succeeds when empty
                except OSError:
                    pass

    def clear_all(self) -> None:
        """Remove the entire store (test teardown / explicit user wipe)."""
        if self._root.is_dir():
            shutil.rmtree(self._root, ignore_errors=True)
