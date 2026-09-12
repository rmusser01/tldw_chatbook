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

import os
import re
import shutil
import stat
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

import portalocker
from loguru import logger

from tldw_chatbook.Utils.paths import get_user_data_dir
from tldw_chatbook.Video_Generation.config import (
    get_video_store_policy,
)
from tldw_chatbook.Video_Generation.video_formats import (
    SUPPORTED_VIDEO_FORMATS,
    canonical_video_extension,
)

VIDEO_MARKER_PREFIX = "[video] "
"""Prefix identifying a video card's content marker in a message row."""

_SLUG_MAX_LEN = 48
_SLUG_MAX_SUFFIX_ATTEMPTS = 99
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9._-]+$")
_NON_SLUG_CHARS = re.compile(r"[^a-z0-9]+")
_ROOT_LEASE_TIMEOUT_SECONDS = 5.0
_ROOT_LEASE_POLL_SECONDS = 0.01
_STAGE_PREFIX = ".video-stage-"


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


@dataclass(frozen=True)
class VideoCapacityExceeded:
    """An ordinary payload cannot fit within the configured store cap."""

    size_bytes: int
    max_bytes: int


class VideoStoreSaveError(RuntimeError):
    """Managed publication or capacity enforcement failed."""


class VideoStoreBusyError(VideoStoreSaveError):
    """The root-scoped capacity lease was not acquired in time."""


class VideoPublicationGate:
    """Linearize cancellation against one managed-file publication.

    A caller that claims the gate holds it through the final commit. Cancellation
    therefore either prevents publication or observes a publication that already
    linearized; it can never interleave with the commit itself.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancelled = False

    def cancel(self) -> None:
        """Prevent a publication that has not already claimed the gate."""
        with self._lock:
            self._cancelled = True

    @contextmanager
    def claim_publication(self) -> Iterator[bool]:
        """Hold the cancellation lock across the caller's final commit."""
        with self._lock:
            yield not self._cancelled


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
        self._transaction_lock = threading.RLock()

    @property
    def root(self) -> Path:
        return self._root

    @property
    def capacity_bytes(self) -> int:
        """Configured capacity in bytes, with integer-MiB normalization."""
        config = self._get_config()
        return max(1, int(getattr(config, "max_store_mb", 2048))) * 1024 * 1024

    def _get_config(self):
        """Retention/capacity settings for this store.

        An injected ``config`` (tests, explicit callers) still wins. Otherwise
        this reads the secrets-free ``VideoStorePolicy`` rather than the full
        ``VideoGenerationConfig``: the store only ever reads ``retention``,
        ``retention_ttl_hours`` and ``max_store_mb``, and the full config's
        secret resolution put an OS-keyring read on the app's startup path
        (TASK-21111(b)).
        """
        return self._config if self._config is not None else get_video_store_policy()

    @property
    def _lease_path(self) -> Path:
        return self._root.parent / f".{self._root.name}.capacity.lock"

    @contextmanager
    def _root_lease(self):
        """Take the bounded interprocess lease for one capacity transaction."""
        try:
            self._root.parent.mkdir(parents=True, exist_ok=True)
            handle = self._lease_path.open("a+b")
        except OSError as exc:
            raise VideoStoreSaveError("managed store lease setup failed") from exc

        flags = portalocker.LockFlags.EXCLUSIVE | portalocker.LockFlags.NON_BLOCKING
        deadline = time.monotonic() + _ROOT_LEASE_TIMEOUT_SECONDS
        locked = False
        try:
            while True:
                try:
                    portalocker.lock(handle, flags)
                    locked = True
                    break
                except portalocker.exceptions.AlreadyLocked as exc:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise VideoStoreBusyError(
                            "generated video store is busy"
                        ) from exc
                    time.sleep(min(_ROOT_LEASE_POLL_SECONDS, remaining))
                except portalocker.exceptions.LockException as exc:
                    raise VideoStoreSaveError("managed store lease failed") from exc
                except Exception as exc:
                    raise VideoStoreSaveError("managed store lease failed") from exc
            yield
        finally:
            if locked:
                try:
                    portalocker.unlock(handle)
                except Exception as exc:
                    logger.warning(
                        "VideoStore: lease unlock failed ({})", type(exc).__name__
                    )
            try:
                handle.close()
            except Exception as exc:
                logger.warning(
                    "VideoStore: lease close failed ({})", type(exc).__name__
                )

    # -- path safety ------------------------------------------------------

    def _message_dir(self, message_id: str) -> Path:
        """Return the per-message directory, rejecting unsafe components.

        Raises:
            ValueError: If ``message_id`` contains anything outside
                ``[A-Za-z0-9._-]`` (path separators, traversal, whitespace).
        """
        if not message_id or not _SAFE_COMPONENT.fullmatch(message_id):
            raise ValueError(f"unsafe message_id component: {message_id!r}")
        return self._root / message_id

    def _video_path(self, message_id: str, slug: str, extension: str) -> Path:
        if not slug or not _SAFE_COMPONENT.fullmatch(slug):
            raise ValueError(f"unsafe slug component: {slug!r}")
        if slug.startswith(_STAGE_PREFIX):
            raise ValueError("slug uses reserved internal stage namespace")
        safe_ext = canonical_video_extension(extension)
        candidate = (self._message_dir(message_id) / f"{slug}.{safe_ext}")
        try:
            resolved_root = self._root.resolve()
            resolved = candidate.resolve()
        except (OSError, RuntimeError) as exc:
            raise VideoStoreSaveError(
                "managed video path resolution failed"
            ) from exc
        if resolved != resolved_root and resolved_root not in resolved.parents:
            raise ValueError(f"path escapes store root: {candidate}")
        return candidate

    def _ensure_safe_root(self, *, create: bool = False) -> Path | None:
        """Return the resolved real root, rejecting root links/reparse points."""
        if create:
            try:
                self._root.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise VideoStoreSaveError("managed store root setup failed") from exc
        try:
            metadata = self._root.lstat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise VideoStoreSaveError("managed store root validation failed") from exc
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or self._is_reparse(metadata)
        ):
            raise VideoStoreSaveError("managed store root is unsafe")
        try:
            return self._root.resolve(strict=True)
        except OSError as exc:
            raise VideoStoreSaveError("managed store root validation failed") from exc

    @staticmethod
    def _ensure_target_absent(target: Path) -> None:
        try:
            target.lstat()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise VideoStoreSaveError("managed video target validation failed") from exc
        raise VideoStoreSaveError("managed video target already exists")

    def _ensure_slug_absent(self, message_id: str, slug: str) -> None:
        """Require that no canonical container already owns this message slug."""
        for _container, _mime, extension in SUPPORTED_VIDEO_FORMATS:
            self._ensure_target_absent(
                self._video_path(message_id, slug, extension)
            )

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
            if all(
                self.resolve(message_id, slug, extension=extension) is None
                for _container, _mime, extension in SUPPORTED_VIDEO_FORMATS
            ):
                return slug
        raise ValueError(f"no free slug for base {base!r} under message {message_id!r}")

    def save(
        self,
        message_id: str,
        slug: str,
        content: bytes,
        *,
        extension: str,
        publication_gate: VideoPublicationGate | None = None,
    ) -> Path | VideoCapacityExceeded:
        """Publish a normal payload and evict only the old files needed for it."""
        if not content:
            raise ValueError("refusing to save an empty video payload")
        path = self._video_path(message_id, slug, extension)
        size_bytes = len(content)
        max_bytes = self.capacity_bytes
        if size_bytes > max_bytes:
            return VideoCapacityExceeded(
                size_bytes=size_bytes,
                max_bytes=max_bytes,
            )

        with self._transaction_lock:
            with self._root_lease():
                self._ensure_safe_root(create=True)
                self._cleanup_orphan_stages_unlocked()
                self._ensure_slug_absent(message_id, slug)
                try:
                    self._atomic_publish(
                        content,
                        path,
                        expected_size=size_bytes,
                        publication_gate=publication_gate,
                    )
                except (OSError, VideoStoreSaveError) as exc:
                    raise VideoStoreSaveError("managed video publication failed") from exc

                try:
                    self._enforce_save_capacity(path)
                except Exception as exc:
                    self._withdraw_new_target(path)
                    if isinstance(exc, VideoStoreSaveError):
                        raise
                    raise VideoStoreSaveError(
                        "managed video capacity enforcement failed"
                    ) from exc

        return path

    def adopt_oversized(
        self,
        message_id: str,
        slug: str,
        stream: BinaryIO,
        size_bytes: int,
        *,
        extension: str,
        publication_gate: VideoPublicationGate | None = None,
    ) -> Path:
        """Adopt one caller-owned over-cap stream as the sole managed video.

        The stream remains open and is rewound before this method returns or
        raises so the caller can retry or save it elsewhere.
        """
        if size_bytes <= 0:
            raise ValueError("refusing to adopt an empty video payload")
        try:
            path = self._video_path(message_id, slug, extension)
            try:
                stream.seek(0)
            except (OSError, ValueError) as exc:
                raise VideoStoreSaveError(
                    "managed video source is not rewindable"
                ) from exc
            with self._transaction_lock:
                with self._root_lease():
                    self._ensure_safe_root(create=True)
                    self._cleanup_orphan_stages_unlocked()
                    self._ensure_slug_absent(message_id, slug)
                    try:
                        self._atomic_publish(
                            stream,
                            path,
                            expected_size=size_bytes,
                            publication_gate=publication_gate,
                        )
                    except Exception as exc:
                        raise VideoStoreSaveError(
                            "managed video publication failed"
                        ) from exc

                    try:
                        for video in self._sorted_oldest(self._snapshot()):
                            if video.path != path:
                                self._checked_unlink(video)
                        final = self._snapshot()
                        if (
                            len(final) != 1
                            or final[0].path != path
                            or final[0].size_bytes != size_bytes
                        ):
                            raise VideoStoreSaveError(
                                "oversized video adoption verification failed"
                            )
                    except Exception as exc:
                        self._withdraw_new_target(path)
                        if isinstance(exc, VideoStoreSaveError):
                            raise
                        raise VideoStoreSaveError(
                            "oversized video adoption failed"
                        ) from exc
                    self._prune_empty_dirs_unlocked()
            return path
        finally:
            try:
                stream.seek(0)
            except (OSError, ValueError):
                pass

    def resolve(self, message_id: str, slug: str, *, extension: str) -> Path | None:
        """Resolve a name to a live file, or ``None`` when missing/expired.

        Missing is the normal post-restart/post-expiry state (ADR-044
        tombstone), never an error. Unsafe components resolve to ``None``
        rather than raising -- resolution is a read path against durable,
        possibly hand-edited names.
        """
        try:
            path = self._video_path(message_id, slug, extension)
        except (ValueError, VideoStoreSaveError):
            return None
        return path if self._is_safe_regular_file(path) else None

    def iter_stored(self) -> Iterator[StoredVideo]:
        """Return an iterator over one completed non-following snapshot."""
        with self._transaction_lock:
            with self._root_lease():
                snapshot = self._snapshot()
        return iter(snapshot)

    # -- capacity transaction helpers -----------------------------------

    @staticmethod
    def _is_reparse(metadata: os.stat_result) -> bool:
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        attributes = getattr(metadata, "st_file_attributes", 0)
        return bool(reparse_flag and attributes & reparse_flag)

    def _snapshot(self) -> tuple[StoredVideo, ...]:
        """Build a complete managed-file inventory without following links."""
        resolved_root = self._ensure_safe_root()
        if resolved_root is None:
            return ()
        videos: list[StoredVideo] = []
        try:
            with os.scandir(self._root) as entries:
                message_entries = sorted(entries, key=lambda entry: entry.name)
        except FileNotFoundError:
            return ()
        except OSError as exc:
            raise VideoStoreSaveError("managed store inventory failed") from exc

        for message_entry in message_entries:
            if not _SAFE_COMPONENT.fullmatch(message_entry.name):
                continue
            try:
                message_metadata = message_entry.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise VideoStoreSaveError("managed store inventory failed") from exc
            if (
                not stat.S_ISDIR(message_metadata.st_mode)
                or stat.S_ISLNK(message_metadata.st_mode)
                or self._is_reparse(message_metadata)
            ):
                continue
            message_path = Path(message_entry.path)
            try:
                resolved_message = message_path.resolve(strict=True)
                if resolved_message.parent != resolved_root:
                    continue
                with os.scandir(message_path) as entries:
                    file_entries = sorted(entries, key=lambda entry: entry.name)
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise VideoStoreSaveError("managed store inventory failed") from exc

            for file_entry in file_entries:
                if file_entry.name.startswith(_STAGE_PREFIX):
                    continue
                if not _SAFE_COMPONENT.fullmatch(file_entry.name):
                    continue
                try:
                    file_metadata = file_entry.stat(follow_symlinks=False)
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    raise VideoStoreSaveError("managed store inventory failed") from exc
                if (
                    not stat.S_ISREG(file_metadata.st_mode)
                    or stat.S_ISLNK(file_metadata.st_mode)
                    or self._is_reparse(file_metadata)
                ):
                    continue
                path = Path(file_entry.path)
                try:
                    if path.resolve(strict=True).parent != resolved_message:
                        continue
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    raise VideoStoreSaveError("managed store inventory failed") from exc
                videos.append(
                    StoredVideo(
                        message_id=message_entry.name,
                        slug=path.stem,
                        path=path,
                        size_bytes=file_metadata.st_size,
                        mtime=file_metadata.st_mtime,
                    )
                )
        return tuple(videos)

    def _cleanup_orphan_stages_unlocked(self) -> None:
        """Remove regular unpublished siblings inside one root transaction."""
        resolved_root = self._ensure_safe_root()
        if resolved_root is None:
            return
        try:
            with os.scandir(self._root) as entries:
                message_entries = sorted(entries, key=lambda entry: entry.name)
        except FileNotFoundError:
            return
        except OSError as exc:
            raise VideoStoreSaveError("managed stage cleanup failed") from exc

        for message_entry in message_entries:
            if not _SAFE_COMPONENT.fullmatch(message_entry.name):
                continue
            try:
                message_metadata = message_entry.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise VideoStoreSaveError("managed stage cleanup failed") from exc
            if (
                not stat.S_ISDIR(message_metadata.st_mode)
                or stat.S_ISLNK(message_metadata.st_mode)
                or self._is_reparse(message_metadata)
            ):
                continue
            message_path = Path(message_entry.path)
            try:
                resolved_message = message_path.resolve(strict=True)
                if resolved_message.parent != resolved_root:
                    continue
                with os.scandir(message_path) as entries:
                    file_entries = sorted(entries, key=lambda entry: entry.name)
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise VideoStoreSaveError("managed stage cleanup failed") from exc

            for file_entry in file_entries:
                if not file_entry.name.startswith(_STAGE_PREFIX):
                    continue
                if not _SAFE_COMPONENT.fullmatch(file_entry.name):
                    continue
                try:
                    file_metadata = file_entry.stat(follow_symlinks=False)
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    raise VideoStoreSaveError("managed stage cleanup failed") from exc
                if (
                    not stat.S_ISREG(file_metadata.st_mode)
                    or stat.S_ISLNK(file_metadata.st_mode)
                    or self._is_reparse(file_metadata)
                ):
                    continue
                path = Path(file_entry.path)
                try:
                    if path.resolve(strict=True).parent != resolved_message:
                        continue
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    raise VideoStoreSaveError("managed stage cleanup failed") from exc
                stage = StoredVideo(
                    message_id=message_entry.name,
                    slug=path.stem,
                    path=path,
                    size_bytes=file_metadata.st_size,
                    mtime=file_metadata.st_mtime,
                )
                try:
                    self._checked_unlink(stage)
                except Exception as exc:
                    raise VideoStoreSaveError("managed stage cleanup failed") from exc

    def _is_safe_regular_file(self, path: Path) -> bool:
        try:
            resolved_root = self._ensure_safe_root()
            if resolved_root is None:
                return False
            metadata = path.lstat()
            parent_metadata = path.parent.lstat()
            resolved_parent = path.parent.resolve(strict=True)
        except (OSError, VideoStoreSaveError):
            return False
        return (
            bool(_SAFE_COMPONENT.fullmatch(path.parent.name))
            and bool(_SAFE_COMPONENT.fullmatch(path.name))
            and stat.S_ISDIR(parent_metadata.st_mode)
            and not stat.S_ISLNK(parent_metadata.st_mode)
            and not self._is_reparse(parent_metadata)
            and resolved_parent.parent == resolved_root
            and stat.S_ISREG(metadata.st_mode)
            and not stat.S_ISLNK(metadata.st_mode)
            and not self._is_reparse(metadata)
        )

    @staticmethod
    def _commit_sibling(sibling: Path, target: Path) -> None:
        os.replace(sibling, target)

    def _atomic_publish(
        self,
        source: bytes | BinaryIO,
        target: Path,
        *,
        expected_size: int,
        publication_gate: VideoPublicationGate | None = None,
    ) -> None:
        resolved_root = self._ensure_safe_root(create=True)
        if resolved_root is None:  # pragma: no cover - create=True guarantees it
            raise VideoStoreSaveError("managed store root setup failed")
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            parent_metadata = target.parent.lstat()
            parent_is_safe = (
                stat.S_ISDIR(parent_metadata.st_mode)
                and not stat.S_ISLNK(parent_metadata.st_mode)
                and not self._is_reparse(parent_metadata)
                and target.parent.resolve(strict=True).parent == resolved_root
            )
        except OSError as exc:
            raise VideoStoreSaveError("managed video directory is unsafe") from exc
        if not parent_is_safe:
            raise VideoStoreSaveError("managed video directory is unsafe")
        sibling: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                delete=False,
                dir=target.parent,
                prefix=_STAGE_PREFIX,
                suffix=".tmp",
            ) as staged:
                sibling = Path(staged.name)
                if isinstance(source, bytes):
                    staged.write(source)
                else:
                    shutil.copyfileobj(source, staged)
                staged.flush()
                actual_size = os.fstat(staged.fileno()).st_size
                if actual_size != expected_size:
                    raise VideoStoreSaveError("managed video source size changed")
            if publication_gate is None:
                self._commit_sibling(sibling, target)
            else:
                with publication_gate.claim_publication() as active:
                    if not active:
                        raise VideoStoreSaveError(
                            "managed video publication cancelled"
                        )
                    self._commit_sibling(sibling, target)
            sibling = None
        finally:
            if sibling is not None:
                try:
                    sibling.unlink(missing_ok=True)
                except OSError:
                    logger.warning(
                        "VideoStore: failed to remove unpublished sibling ({})",
                        "OSError",
                    )

    @staticmethod
    def _sorted_oldest(
        videos: tuple[StoredVideo, ...] | list[StoredVideo],
    ) -> list[StoredVideo]:
        return sorted(videos, key=lambda item: (item.mtime, item.message_id, item.path.name))

    def _checked_unlink(self, video: StoredVideo) -> None:
        """Repeat non-following containment checks immediately before unlink."""
        path = video.path
        if path.parent.name != video.message_id or not self._is_safe_regular_file(path):
            raise VideoStoreSaveError("managed video deletion safety check failed")
        try:
            path.unlink()
        except OSError as exc:
            raise VideoStoreSaveError("managed video deletion failed") from exc

    def _withdraw_new_target(self, target: Path) -> None:
        try:
            metadata = target.lstat()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise VideoStoreSaveError("managed video rollback failed") from exc
        video = StoredVideo(
            message_id=target.parent.name,
            slug=target.stem,
            path=target,
            size_bytes=metadata.st_size,
            mtime=metadata.st_mtime,
        )
        try:
            self._checked_unlink(video)
        except Exception as exc:
            raise VideoStoreSaveError("managed video rollback failed") from exc
        if any(video.path == target for video in self._snapshot()):
            raise VideoStoreSaveError("managed video rollback verification failed")

    def _enforce_save_capacity(self, new_target: Path) -> None:
        snapshot = self._snapshot()
        if not any(video.path == new_target for video in snapshot):
            raise VideoStoreSaveError("managed video publication verification failed")
        total = sum(video.size_bytes for video in snapshot)
        victims = self._sorted_oldest(
            [video for video in snapshot if video.path != new_target]
        )
        for victim in victims:
            if total <= self.capacity_bytes:
                break
            self._checked_unlink(victim)
            total -= victim.size_bytes
        final = self._snapshot()
        if (
            not any(video.path == new_target for video in final)
            or sum(video.size_bytes for video in final) > self.capacity_bytes
        ):
            raise VideoStoreSaveError("managed video capacity verification failed")
        self._prune_empty_dirs_unlocked()

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
        retention = getattr(config, "retention", "session")
        ttl_seconds = max(1, int(getattr(config, "retention_ttl_hours", 24))) * 3600

        with self._transaction_lock:
            with self._root_lease():
                self._cleanup_orphan_stages_unlocked()
                stored = self._snapshot()
                survivors: list[StoredVideo] = []
                for video in stored:
                    expired = retention == "session" or (now - video.mtime) > ttl_seconds
                    if expired:
                        removed = self._remove_startup(video)
                        if removed:
                            removed_files += 1
                            removed_bytes += video.size_bytes
                            evicted.append((video.message_id, video.slug))
                        else:
                            survivors.append(video)
                    else:
                        survivors.append(video)

                total = sum(video.size_bytes for video in survivors)
                sole_oversized_exception = (
                    len(survivors) == 1 and total > self.capacity_bytes
                )
                if not sole_oversized_exception:
                    for video in self._sorted_oldest(survivors):
                        if total <= self.capacity_bytes:
                            break
                        if self._remove_startup(video):
                            total -= video.size_bytes
                            removed_files += 1
                            removed_bytes += video.size_bytes
                            evicted.append((video.message_id, video.slug))

                self._prune_empty_dirs_unlocked()
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

    def _remove_startup(self, video: StoredVideo) -> bool:
        try:
            self._checked_unlink(video)
            return True
        except Exception as exc:
            logger.warning(
                "VideoStore: startup removal failed ({})", type(exc).__name__
            )
            return False

    def _prune_empty_dirs_unlocked(self) -> None:
        """Remove now-empty per-message directories (and the root's husk)."""
        resolved_root = self._ensure_safe_root()
        if resolved_root is None:
            return
        try:
            with os.scandir(self._root) as root_entries:
                entries = sorted(root_entries, key=lambda entry: entry.name)
        except OSError:
            return
        for entry in entries:
            if not _SAFE_COMPONENT.fullmatch(entry.name):
                continue
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            if stat.S_ISDIR(metadata.st_mode) and not self._is_reparse(metadata):
                try:
                    Path(entry.path).rmdir()  # only succeeds when empty
                except OSError:
                    pass

    def clear_all(self) -> None:
        """Remove the entire store (test teardown / explicit user wipe)."""
        with self._transaction_lock:
            with self._root_lease():
                resolved_root = self._ensure_safe_root()
                if resolved_root is None:
                    return
                shutil.rmtree(self._root, ignore_errors=True)
