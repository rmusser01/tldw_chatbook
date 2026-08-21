"""Identity-pinned private staging for Persona Visual authoring assets."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import stat
from dataclasses import dataclass, field, replace
from io import BytesIO
from pathlib import Path, PurePosixPath
from uuid import uuid4

from PIL import Image

from tldw_chatbook.Utils.private_paths import secure_private_directory

from .assets import (
    MAX_ASSET_DECODED_PIXELS,
    PersonaVisualAssetMetadata,
    load_persona_visual_asset,
    validate_persona_visual_asset_set,
)
from .authoring import (
    PersonaVisualAuthoringDraft,
    PersonaVisualDraftAsset,
    inspect_persona_visual_draft,
)
from .contracts import (
    MAX_ASSET_DIMENSION,
    MAX_FRAME_DURATION_MS,
    MAX_FRAMES_PER_ANIMATION,
)

_STATE = re.compile(r"[a-z][a-z0-9_.:-]{0,95}\Z")
_MARKER_NAME = ".persona-visual-authoring"
_FORMATS = {
    "GIF": ("image/gif", ".gif"),
    "JPEG": ("image/jpeg", ".jpg"),
    "PNG": ("image/png", ".png"),
    "WEBP": ("image/webp", ".webp"),
}
_READ_CHUNK_BYTES = 64 * 1024


class PersonaVisualAuthoringWorkspaceError(ValueError):
    """Stable path-free private-staging failure."""

    __slots__ = ("category",)

    def __init__(self, category: str) -> None:
        self.category = (
            category
            if category
            in {
                "persona_visual_authoring_asset_invalid",
                "persona_visual_authoring_cleanup_denied",
                "persona_visual_authoring_staging_failed",
            }
            else "persona_visual_authoring_staging_failed"
        )
        super().__init__(self.category)


@dataclass(frozen=True, slots=True)
class PersonaVisualAuthoringWorkspace:
    """One private candidate retained only by its owning editor session."""

    profile_root: Path = field(repr=False, compare=False)
    relative_root: str
    identity: tuple[int, int] = field(repr=False)
    secret: str = field(repr=False)
    _assets: tuple[_WorkspaceAsset, ...] = field(default=(), repr=False)

    @property
    def asset_names(self) -> tuple[str, ...]:
        """Return the bounded path-free file inventory."""

        return tuple(asset.name for asset in self._assets)


@dataclass(frozen=True, slots=True)
class _WorkspaceAsset:
    name: str
    identity: tuple[int, int, int, int, int]
    sha256: str


def create_persona_visual_authoring_workspace(
    profile_root: os.PathLike[str] | str,
) -> PersonaVisualAuthoringWorkspace:
    """Create one verified-private, identity-pinned authoring candidate."""

    try:
        root = _absolute_root(profile_root)
        base = root / "persona_visual" / "authoring"
        privacy = secure_private_directory(base, create=True, application_owned=True)
        if not privacy.verified_private:
            raise ValueError
        name = f".draft-{uuid4().hex}"
        candidate = base / name
        candidate.mkdir(mode=0o700)
        (candidate / "assets").mkdir(mode=0o700)
        metadata = os.lstat(candidate)
        if not _private_directory(metadata):
            raise ValueError
        identity = (metadata.st_dev, metadata.st_ino)
        secret = os.urandom(32).hex()
        marker = _marker(secret, name, identity)
        _write_private(candidate / _MARKER_NAME, marker.encode("ascii"))
        return PersonaVisualAuthoringWorkspace(
            root,
            f"persona_visual/authoring/{name}",
            identity,
            secret,
        )
    except Exception:
        raise PersonaVisualAuthoringWorkspaceError(
            "persona_visual_authoring_staging_failed"
        ) from None


def stage_persona_visual_authoring_asset(
    workspace: PersonaVisualAuthoringWorkspace,
    data: bytes,
    *,
    state: str,
) -> tuple[PersonaVisualAuthoringWorkspace, PersonaVisualDraftAsset]:
    """Validate and add one replacement raster to an existing workspace."""

    try:
        if (
            type(workspace) is not PersonaVisualAuthoringWorkspace
            or type(data) is not bytes
            or not data
            or type(state) is not str
            or _STATE.fullmatch(state) is None
        ):
            raise ValueError
        mime_type, suffix, width, height, frame_count, duration_ms = _decode(data)
        metadata = PersonaVisualAssetMetadata(
            asset_key=f"{state}-{uuid4().hex}",
            role="frame",
            mime_type=mime_type,
            byte_count=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            width=width,
            height=height,
            frame_count=frame_count,
            duration_ms=duration_ms,
        )
        metadata = validate_persona_visual_asset_set((metadata,))[0]
        return _write_workspace_asset(workspace, data, metadata, suffix=suffix)
    except PersonaVisualAuthoringWorkspaceError:
        raise
    except Exception:
        raise PersonaVisualAuthoringWorkspaceError(
            "persona_visual_authoring_asset_invalid"
        ) from None


def adopt_persona_visual_draft_sources(
    workspace: PersonaVisualAuthoringWorkspace,
    draft: PersonaVisualAuthoringDraft,
    *,
    source_root: os.PathLike[str] | str,
) -> tuple[PersonaVisualAuthoringWorkspace, PersonaVisualAuthoringDraft]:
    """Copy one validated draft into the workspace without changing metadata."""

    if (
        type(workspace) is not PersonaVisualAuthoringWorkspace
        or workspace.asset_names
        or type(draft) is not PersonaVisualAuthoringDraft
    ):
        raise PersonaVisualAuthoringWorkspaceError(
            "persona_visual_authoring_staging_failed"
        )
    current = workspace
    copied: list[PersonaVisualDraftAsset] = []
    try:
        for asset in draft.assets:
            loaded = load_persona_visual_asset(
                source_root,
                storage_key=asset.source_storage_key,
                metadata=asset.metadata,
            )
            suffix = _FORMATS[_format_for_mime(asset.metadata.mime_type)][1]
            current, copied_asset = _write_workspace_asset(
                current,
                loaded.data,
                asset.metadata,
                suffix=suffix,
            )
            copied.append(copied_asset)
        adopted = replace(draft, assets=tuple(copied))
        inspect_persona_visual_draft(adopted)
        return current, adopted
    except Exception:
        cleanup_persona_visual_authoring_workspace(current)
        raise PersonaVisualAuthoringWorkspaceError(
            "persona_visual_authoring_staging_failed"
        ) from None


def cleanup_persona_visual_authoring_workspace(
    workspace: PersonaVisualAuthoringWorkspace,
) -> bool:
    """Delete only the exact issued workspace and its declared files."""

    if type(workspace) is not PersonaVisualAuthoringWorkspace:
        return False
    root_fd = base_fd = candidate_fd = assets_fd = -1
    try:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        root_fd = os.open(workspace.profile_root, flags)
        visual_fd = os.open("persona_visual", flags, dir_fd=root_fd)
        os.close(root_fd)
        root_fd = visual_fd
        base_fd = os.open("authoring", flags, dir_fd=root_fd)
        name = PurePosixPath(workspace.relative_root).name
        candidate_fd = os.open(name, flags, dir_fd=base_fd)
        opened = os.fstat(candidate_fd)
        named = os.stat(name, dir_fd=base_fd, follow_symlinks=False)
        if (
            not _private_directory(opened)
            or not _private_directory(named)
            or (opened.st_dev, opened.st_ino) != workspace.identity
            or (named.st_dev, named.st_ino) != workspace.identity
            or _read_marker(candidate_fd)
            != _marker(workspace.secret, name, workspace.identity)
        ):
            return False
        assets_fd = os.open("assets", flags, dir_fd=candidate_fd)
        expected = set(workspace.asset_names)
        if set(os.listdir(assets_fd)) != expected:
            return False
        for asset in workspace._assets:
            descriptor = os.open(
                asset.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=assets_fd
            )
            try:
                opened_asset = os.fstat(descriptor)
                named_asset = os.stat(
                    asset.name, dir_fd=assets_fd, follow_symlinks=False
                )
                if (
                    not _regular_file(opened_asset)
                    or not _regular_file(named_asset)
                    or _file_identity(opened_asset) != asset.identity
                    or _file_identity(named_asset) != asset.identity
                    or _digest_fd(descriptor, opened_asset.st_size) != asset.sha256
                    or _file_identity(os.fstat(descriptor)) != asset.identity
                ):
                    return False
            finally:
                os.close(descriptor)
        if set(os.listdir(assets_fd)) != expected:
            return False
        if set(os.listdir(candidate_fd)) != {_MARKER_NAME, "assets"}:
            return False
        for asset_name in workspace.asset_names:
            os.unlink(asset_name, dir_fd=assets_fd)
        os.close(assets_fd)
        assets_fd = -1
        os.rmdir("assets", dir_fd=candidate_fd)
        os.unlink(_MARKER_NAME, dir_fd=candidate_fd)
        opened = os.fstat(candidate_fd)
        named = os.stat(name, dir_fd=base_fd, follow_symlinks=False)
        if (opened.st_dev, opened.st_ino) != workspace.identity or (
            named.st_dev,
            named.st_ino,
        ) != workspace.identity:
            return False
        os.rmdir(name, dir_fd=base_fd)
        return True
    except Exception:
        return False
    finally:
        for descriptor in (assets_fd, candidate_fd, base_fd, root_fd):
            if descriptor >= 0:
                os.close(descriptor)


def _write_workspace_asset(
    workspace: PersonaVisualAuthoringWorkspace,
    data: bytes,
    metadata: PersonaVisualAssetMetadata,
    *,
    suffix: str,
) -> tuple[PersonaVisualAuthoringWorkspace, PersonaVisualDraftAsset]:
    name = f"{uuid4().hex}{suffix}"
    root_fd = base_fd = candidate_fd = assets_fd = descriptor = -1
    try:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        root_fd = os.open(workspace.profile_root, flags)
        visual_fd = os.open("persona_visual", flags, dir_fd=root_fd)
        os.close(root_fd)
        root_fd = visual_fd
        base_fd = os.open("authoring", flags, dir_fd=root_fd)
        candidate_name = PurePosixPath(workspace.relative_root).name
        candidate_fd = os.open(candidate_name, flags, dir_fd=base_fd)
        named_candidate = os.stat(candidate_name, dir_fd=base_fd, follow_symlinks=False)
        if (
            (os.fstat(candidate_fd).st_dev, os.fstat(candidate_fd).st_ino)
            != workspace.identity
            or (named_candidate.st_dev, named_candidate.st_ino) != workspace.identity
            or _read_marker(candidate_fd)
            != _marker(workspace.secret, candidate_name, workspace.identity)
        ):
            raise OSError
        assets_fd = os.open("assets", flags, dir_fd=candidate_fd)
        descriptor = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=assets_fd,
        )
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view[:_READ_CHUNK_BYTES])
            if written <= 0:
                raise OSError
            view = view[written:]
        os.fsync(descriptor)
        written_metadata = os.fstat(descriptor)
        pin = _WorkspaceAsset(
            name,
            _file_identity(written_metadata),
            hashlib.sha256(data).hexdigest(),
        )
    except Exception:
        if assets_fd >= 0:
            try:
                os.unlink(name, dir_fd=assets_fd)
            except OSError:
                pass
        raise
    finally:
        for file_descriptor in (
            descriptor,
            assets_fd,
            candidate_fd,
            base_fd,
            root_fd,
        ):
            if file_descriptor >= 0:
                os.close(file_descriptor)
    source_key = f"{workspace.relative_root}/assets/{name}"
    updated = replace(workspace, _assets=workspace._assets + (pin,))
    return updated, PersonaVisualDraftAsset(source_key, metadata)


def _decode(data: bytes) -> tuple[str, str, int, int, int, int | None]:
    with Image.open(BytesIO(data)) as image:
        if image.format not in _FORMATS:
            raise ValueError
        mime_type, suffix = _FORMATS[image.format]
        width, height = image.size
        frame_count = int(getattr(image, "n_frames", 1))
        if (
            width < 1
            or height < 1
            or width > MAX_ASSET_DIMENSION
            or height > MAX_ASSET_DIMENSION
            or frame_count < 1
            or frame_count > MAX_FRAMES_PER_ANIMATION
            or width * height * frame_count > MAX_ASSET_DECODED_PIXELS
        ):
            raise ValueError
        duration = 0
        for index in range(frame_count):
            image.seek(index)
            frame_duration = image.info.get("duration", 0)
            if type(frame_duration) is not int or frame_duration < 0:
                raise ValueError
            duration += frame_duration
            if duration > MAX_FRAME_DURATION_MS:
                raise ValueError
            image.load()
        return (
            mime_type,
            suffix,
            width,
            height,
            frame_count,
            duration or None,
        )


def _format_for_mime(mime_type: str) -> str:
    return next(name for name, (mime, _suffix) in _FORMATS.items() if mime == mime_type)


def _absolute_root(value: os.PathLike[str] | str) -> Path:
    raw = os.fspath(value)
    if type(raw) is not str or not raw or "\x00" in raw:
        raise ValueError
    root = Path(raw)
    if not root.is_absolute() or str(root) != raw:
        raise ValueError
    metadata = os.lstat(root)
    if not _private_directory(metadata):
        raise ValueError
    return root


def _private_directory(metadata: os.stat_result) -> bool:
    return (
        stat.S_ISDIR(metadata.st_mode)
        and metadata.st_uid == os.geteuid()
        and stat.S_IMODE(metadata.st_mode) & 0o077 == 0
    )


def _regular_file(metadata: os.stat_result) -> bool:
    return (
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_uid == os.geteuid()
        and stat.S_IMODE(metadata.st_mode) == 0o600
        and metadata.st_nlink == 1
    )


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _digest_fd(descriptor: int, expected_size: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    remaining = expected_size
    while remaining:
        chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, remaining))
        if not chunk:
            raise ValueError
        digest.update(chunk)
        remaining -= len(chunk)
    if os.read(descriptor, 1):
        raise ValueError
    return digest.hexdigest()


def _marker(secret: str, name: str, identity: tuple[int, int]) -> str:
    payload = f"{name}:{identity[0]}:{identity[1]}".encode("ascii")
    return hmac.new(bytes.fromhex(secret), payload, hashlib.sha256).hexdigest()


def _write_private(path: Path, data: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_marker(candidate_fd: int) -> str:
    descriptor = os.open(_MARKER_NAME, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=candidate_fd)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size != 64
        ):
            raise ValueError
        data = os.read(descriptor, 65)
        if len(data) != 64:
            raise ValueError
        return data.decode("ascii")
    finally:
        os.close(descriptor)


__all__ = [
    "PersonaVisualAuthoringWorkspace",
    "PersonaVisualAuthoringWorkspaceError",
    "adopt_persona_visual_draft_sources",
    "cleanup_persona_visual_authoring_workspace",
    "create_persona_visual_authoring_workspace",
    "stage_persona_visual_authoring_asset",
]
