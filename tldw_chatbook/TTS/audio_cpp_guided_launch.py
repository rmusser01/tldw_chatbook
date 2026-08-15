"""Private POSIX launch artifacts for guided audio.cpp configurations."""

from __future__ import annotations

import asyncio
import json
import os
import platform
import secrets
import socket
import stat
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, NoReturn, cast

from tldw_chatbook.Model_Artifacts.service import (
    ArtifactRef,
    LeasedArtifactHandle,
)
from tldw_chatbook.Model_Artifacts.store import managed_service
from tldw_chatbook.Utils.private_paths import secure_private_directory

from .audio_cpp_guided_config import (
    AudioCppAcceptedPackage,
    AudioCppBackendPreference,
    AudioCppManagedSetupSource,
    AudioCppSettingsConfig,
)
from .audio_cpp_managed_config import (
    AudioCppExpectedModel,
    AudioCppManagedLaunchConfig,
)
from .audio_cpp_package_scanner import (
    AudioCppPackageScanResult,
    AudioCppScanOutcome,
    scan_audio_cpp_package_root_async,
)
from .audio_cpp_recipes import (
    AUDIO_CPP_RECIPE_REGISTRY,
    AudioCppBackendEvidenceState,
    AudioCppPackageRecipe,
)
from .windows_artifact_fs import (
    OS_WINDOWS_ARTIFACT_FILESYSTEM,
    WindowsArtifactError,
    WindowsArtifactFilesystem,
    windows_audio_cpp_platform_supported,
)


AudioCppGuidedLaunchErrorCode = Literal[
    "configuration_invalid",
    "binary_invalid",
    "package_changed",
    "backend_unsupported",
    "port_unavailable",
    "artifact_changed",
    "artifact_create_failed",
    "artifact_cleanup_failed",
]

_ERROR_MESSAGES: dict[AudioCppGuidedLaunchErrorCode, str] = {
    "configuration_invalid": "The guided audio.cpp configuration is invalid",
    "binary_invalid": "The selected audio.cpp server executable is unavailable",
    "package_changed": "A guided audio.cpp model package requires review",
    "backend_unsupported": "The selected audio.cpp backend is unsupported",
    "port_unavailable": "A private audio.cpp loopback port is unavailable",
    "artifact_changed": "The generated audio.cpp configuration changed",
    "artifact_create_failed": "The generated audio.cpp configuration could not be created",
    "artifact_cleanup_failed": "The generated audio.cpp configuration could not be removed",
}
_PRIVATE_PORT_MIN = 49_152
_PRIVATE_PORT_MAX = 65_535
_PORT_ATTEMPTS = 128
_ARTIFACT_ATTEMPTS = 16
_ARTIFACT_FILE = "server.json"
_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_FILE_READ_FLAGS = (
    os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
)
_windows_artifact_filesystem: WindowsArtifactFilesystem | None = (
    OS_WINDOWS_ARTIFACT_FILESYSTEM if windows_audio_cpp_platform_supported() else None
)


class AudioCppGuidedLaunchError(ValueError):
    """Stable path-independent guided launch failure."""

    __slots__ = ("_cleanup_owner", "code")

    def __init__(
        self,
        code: AudioCppGuidedLaunchErrorCode,
        *,
        cleanup_owner: AudioCppGeneratedLaunchArtifact | None = None,
    ) -> None:
        self.code = code
        self._cleanup_owner = cleanup_owner
        super().__init__(_ERROR_MESSAGES[code])

    def take_cleanup_owner(self) -> AudioCppGeneratedLaunchArtifact | None:
        """Transfer a retained failed-cleanup owner exactly once."""

        owner = self._cleanup_owner
        self._cleanup_owner = None
        return owner


class AudioCppGeneratedLaunchArtifact:
    """Descriptor-bound ownership of one generated configuration directory."""

    __slots__ = (
        "_cleaned",
        "_digest",
        "_directory_fd",
        "_directory_identity",
        "_directory_name",
        "_fds_closed",
        "_file_identity",
        "_managed_handles",
        "_parent_fd",
        "_size",
        "server_json_path",
    )

    def __init__(
        self,
        *,
        parent_fd: int,
        directory_fd: int,
        directory_name: str,
        server_json_path: Path,
        directory_identity: tuple[int, int],
        file_identity: tuple[int, int],
        digest: str,
        size: int,
    ) -> None:
        self._parent_fd = parent_fd
        self._directory_fd = directory_fd
        self._directory_name = directory_name
        self.server_json_path = server_json_path
        self._directory_identity = directory_identity
        self._file_identity = file_identity
        self._digest = digest
        self._size = size
        self._cleaned = False
        self._fds_closed = False
        self._managed_handles: list[LeasedArtifactHandle] = []

    def retain_managed_handle(self, handle: LeasedArtifactHandle) -> None:
        """Take ownership of one exact managed lease before further work."""

        if self._cleaned:
            raise RuntimeError("generated artifact is already cleaned")
        self._managed_handles.append(handle)

    @property
    def privacy_posture(self) -> str:
        """Project only the privacy posture verified by the artifact owner."""

        return "posix_owner_only"

    @staticmethod
    def _identity(info: os.stat_result) -> tuple[int, int]:
        return info.st_dev, info.st_ino

    def _entry_is_owned(self) -> bool:
        if not self._directory_is_owned():
            return False
        file_info = os.stat(
            _ARTIFACT_FILE,
            dir_fd=self._directory_fd,
            follow_symlinks=False,
        )
        return (
            stat.S_ISREG(file_info.st_mode)
            and self._identity(file_info) == self._file_identity
            and file_info.st_nlink == 1
            and file_info.st_uid == os.geteuid()
            and stat.S_IMODE(file_info.st_mode) == 0o400
        )

    def _directory_is_owned(self) -> bool:
        directory_info = os.stat(
            self._directory_name,
            dir_fd=self._parent_fd,
            follow_symlinks=False,
        )
        descriptor_info = os.fstat(self._directory_fd)
        return (
            stat.S_ISDIR(directory_info.st_mode)
            and self._identity(directory_info) == self._directory_identity
            and stat.S_ISDIR(descriptor_info.st_mode)
            and self._identity(descriptor_info) == self._directory_identity
            and directory_info.st_uid == os.geteuid()
            and stat.S_IMODE(directory_info.st_mode) == 0o700
        )

    def validate(self) -> None:
        """Require the exact descriptor-bound file to remain unchanged."""
        failed = False
        file_fd = -1
        try:
            if self._cleaned or not self._entry_is_owned():
                failed = True
            else:
                file_fd = os.open(
                    _ARTIFACT_FILE,
                    _FILE_READ_FLAGS,
                    dir_fd=self._directory_fd,
                )
                info = os.fstat(file_fd)
                if (
                    self._identity(info) != self._file_identity
                    or info.st_nlink != 1
                    or info.st_size != self._size
                ):
                    failed = True
                else:
                    chunks: list[bytes] = []
                    while True:
                        chunk = os.read(file_fd, 64 * 1024)
                        if not chunk:
                            break
                        chunks.append(chunk)
                    failed = sha256(b"".join(chunks)).hexdigest() != self._digest
        except OSError:
            failed = True
        finally:
            if file_fd >= 0:
                os.close(file_fd)
        if failed:
            raise AudioCppGuidedLaunchError("artifact_changed") from None

    def cleanup(self) -> None:
        """Retry exact config removal, then release remaining managed leases."""
        if self._cleaned:
            return
        if not self._fds_closed:
            failed = False
            try:
                if not self._directory_is_owned():
                    failed = True
                else:
                    file_missing = False
                    try:
                        file_owned = self._entry_is_owned()
                    except FileNotFoundError:
                        file_missing = True
                        file_owned = False
                    if not file_missing and not file_owned:
                        failed = True
                    else:
                        if file_owned:
                            try:
                                os.unlink(_ARTIFACT_FILE, dir_fd=self._directory_fd)
                            except FileNotFoundError:
                                pass
                        if not self._directory_is_owned():
                            failed = True
                        else:
                            os.rmdir(self._directory_name, dir_fd=self._parent_fd)
            except OSError:
                failed = True
            if failed:
                raise AudioCppGuidedLaunchError("artifact_cleanup_failed") from None
            # POSIX does not make a failed close retry-safe: the numeric fd may
            # already have been released and reused. Relinquish the numbers once
            # attempted instead of risking a later close of an unrelated file.
            self._fds_closed = True
            for descriptor in (self._directory_fd, self._parent_fd):
                try:
                    os.close(descriptor)
                except OSError:
                    pass

        self._cleanup_managed_handles()
        self._cleaned = True

    def _cleanup_managed_handles(self) -> None:
        remaining: list[LeasedArtifactHandle] = []
        control_flow: BaseException | None = None
        for handle in self._managed_handles:
            try:
                handle.close()
            except BaseException as error:
                remaining.append(handle)
                if control_flow is None and not isinstance(error, Exception):
                    control_flow = error
        self._managed_handles = remaining
        if control_flow is not None:
            raise control_flow
        if remaining:
            raise AudioCppGuidedLaunchError("artifact_cleanup_failed") from None


class _WindowsGeneratedLaunchArtifact(AudioCppGeneratedLaunchArtifact):
    """Exact Windows handle owner for one generated launch configuration."""

    __slots__ = (
        "_windows_directory",
        "_windows_extras",
        "_windows_file",
        "_windows_parent",
    )

    def __init__(
        self,
        *,
        parent: Any,
        directory: Any | None,
        file: Any | None,
        server_json_path: Path,
        digest: str,
        size: int,
        extras: tuple[Any, ...] = (),
    ) -> None:
        super().__init__(
            parent_fd=-1,
            directory_fd=-1,
            directory_name=server_json_path.parent.name,
            server_json_path=server_json_path,
            directory_identity=(0, 0),
            file_identity=(0, 0),
            digest=digest,
            size=size,
        )
        self._windows_parent = parent
        self._windows_directory = directory
        self._windows_file = file
        self._windows_extras = list(extras)

    @property
    def privacy_posture(self) -> str:
        """Report protected only while every published owner still verifies."""

        parent = self._windows_parent
        directory = self._windows_directory
        file = self._windows_file
        if self._cleaned or parent is None or directory is None or file is None:
            return "unverified"
        owners = (parent, directory, file)
        return (
            "windows_account_protected"
            if all(
                owner.privacy_posture == "windows_account_protected"
                and owner.verify_private_acl()
                for owner in owners
            )
            else "unverified"
        )

    def validate(self) -> None:
        """Validate exact path identity, bounded bytes, contents, and DACL."""

        filesystem = _windows_artifact_filesystem
        failed = self.privacy_posture != "windows_account_protected"
        observed: Any = None
        try:
            if failed or filesystem is None or self._windows_file is None:
                failed = True
            else:
                names = tuple(
                    entry.name for entry in os.scandir(self.server_json_path.parent)
                )
                if names != (_ARTIFACT_FILE,):
                    failed = True
                else:
                    observed = filesystem.open_file_no_reparse(self.server_json_path)
                    self._windows_extras.append(observed)
                    if observed.identity != self._windows_file.identity:
                        failed = True
                    else:
                        data = self._windows_file.read(self._size + 1)
                        failed = (
                            len(data) != self._size
                            or sha256(data).hexdigest() != self._digest
                        )
        except (OSError, WindowsArtifactError):
            failed = True
        finally:
            if observed is not None:
                try:
                    observed.close()
                except WindowsArtifactError:
                    failed = True
                else:
                    self._windows_extras.remove(observed)
        if failed:
            raise AudioCppGuidedLaunchError("artifact_changed") from None

    @staticmethod
    def _close_windows(owner: Any) -> bool:
        try:
            owner.close()
        except WindowsArtifactError:
            return False
        return True

    def cleanup(self) -> None:
        """Remove exact Windows objects, close pins, then release leases."""

        if self._cleaned:
            return
        failed = False
        retained_extras: list[Any] = []
        for owner in self._windows_extras:
            if not self._close_windows(owner):
                retained_extras.append(owner)
        self._windows_extras = retained_extras
        failed = bool(retained_extras)

        if self._windows_file is not None:
            try:
                self._windows_file.delete_exact()
            except WindowsArtifactError:
                failed = True
            else:
                if self._close_windows(self._windows_file):
                    self._windows_file = None
                else:
                    failed = True

        directory_path = self.server_json_path.parent
        if self._windows_directory is not None and self._windows_file is None:
            try:
                if directory_path.exists() and tuple(directory_path.iterdir()):
                    failed = True
                else:
                    self._windows_directory.delete_exact()
            except (OSError, WindowsArtifactError):
                failed = True
            else:
                if self._close_windows(self._windows_directory):
                    self._windows_directory = None
                else:
                    failed = True

        if self._windows_directory is None and self._windows_parent is not None:
            if self._close_windows(self._windows_parent):
                self._windows_parent = None
            else:
                failed = True
        if failed:
            raise AudioCppGuidedLaunchError("artifact_cleanup_failed") from None
        self._cleanup_managed_handles()
        self._cleaned = True


def _normalize_architecture(value: str, *, system: str) -> str:
    folded = value.casefold()
    if system == "windows" and folded in {"x86", "i386", "i486", "i586", "i686"}:
        return "x86"
    if folded in {"arm64", "aarch64"}:
        return "arm64" if system == "darwin" else "aarch64"
    return {
        "amd64": "x86_64",
        "x64": "x86_64",
    }.get(folded, folded)


def _windows_pe_machine(owner: Any) -> int | None:
    header = owner.read(4096)
    if len(header) < 0x40 or header[:2] != b"MZ":
        return None
    offset = int.from_bytes(header[0x3C:0x40], "little")
    if offset < 0x40 or offset > len(header) - 6:
        return None
    if header[offset : offset + 4] != b"PE\0\0":
        return None
    return int.from_bytes(header[offset + 4 : offset + 6], "little")


def _validate_binary(
    path: str,
    *,
    system: str | None = None,
    architecture: str | None = None,
) -> Path | None:
    from tldw_chatbook.Utils.path_validation import validate_path_simple

    host_system = (platform.system() if system is None else system).casefold()
    try:
        candidate = validate_path_simple(path, require_exists=True)
        if host_system == "windows":
            filesystem = _windows_artifact_filesystem
            if (
                filesystem is None
                or not candidate.is_absolute()
                or candidate.suffix.casefold() != ".exe"
            ):
                return None
            host_architecture = _normalize_architecture(
                platform.machine() if architecture is None else architecture,
                system="windows",
            )
            expected_machine = {"x86": 0x014C, "x86_64": 0x8664}.get(host_architecture)
            if expected_machine is None:
                return None
            owner = filesystem.open_file_no_reparse(candidate)
            valid = (
                owner.identity.kind == "file"
                and owner.identity.reparse_tag == 0
                and _windows_pe_machine(owner) == expected_machine
            )
            try:
                owner.close()
            except WindowsArtifactError:
                try:
                    owner.close()
                except WindowsArtifactError:
                    return None
            return candidate if valid else None
        info = candidate.stat()
        if (
            not candidate.is_absolute()
            or not stat.S_ISREG(info.st_mode)
            or not os.access(candidate, os.X_OK)
        ):
            return None
    except (OSError, ValueError, WindowsArtifactError):
        return None
    return candidate


def _candidate_matches_accepted(
    scan: AudioCppPackageScanResult,
    accepted: AudioCppAcceptedPackage,
    recipe: AudioCppPackageRecipe,
) -> bool:
    if scan.outcome is not AudioCppScanOutcome.COMPLETE:
        return False
    matches = tuple(
        candidate
        for discovery in scan.discoveries
        for candidate in discovery.match.candidates
        if (
            candidate.recipe is recipe
            and candidate.canonical_root == accepted.canonical_root
            and candidate.canonical_root_identity == accepted.canonical_root_identity
            and candidate.configuration_identity == accepted.configuration_identity
            and candidate.weight_identity == accepted.weight_identity
        )
    )
    return len(matches) == 1


async def revalidate_audio_cpp_guided_packages(
    accepted_packages: tuple[AudioCppAcceptedPackage, ...],
) -> tuple[AudioCppPackageRecipe, ...]:
    """Recheck accepted package identities without launching audio.cpp.

    Args:
        accepted_packages: Immutable package snapshots accepted in Settings.

    Returns:
        The exact reviewed recipes for the still-current package identities.

    Raises:
        AudioCppGuidedLaunchError: If a recipe or local package no longer
            matches its accepted snapshot.
    """

    recipes: list[AudioCppPackageRecipe] = []
    for accepted in accepted_packages:
        invalid_recipe = False
        try:
            recipe = AUDIO_CPP_RECIPE_REGISTRY.validate_accepted(accepted)
        except (TypeError, ValueError):
            invalid_recipe = True
            recipe = None
        if invalid_recipe or recipe is None:
            raise AudioCppGuidedLaunchError("package_changed") from None

        scan_failed = False
        scan: AudioCppPackageScanResult | None = None
        try:
            if accepted.managed_artifact is None:
                scan = await scan_audio_cpp_package_root_async(accepted.canonical_root)
            else:
                scan = await scan_audio_cpp_package_root_async(
                    accepted.canonical_root,
                    expected_managed_artifact=accepted.managed_artifact,
                    expected_canonical_root=accepted.canonical_root,
                )
        except Exception:
            scan_failed = True
        if (
            scan_failed
            or scan is None
            or not _candidate_matches_accepted(scan, accepted, recipe)
        ):
            raise AudioCppGuidedLaunchError("package_changed") from None
        recipes.append(recipe)
    return tuple(recipes)


def _eligible_backends(
    recipes: tuple[AudioCppPackageRecipe, ...],
    *,
    system: str,
    architecture: str,
) -> frozenset[AudioCppBackendPreference]:
    accepted_states = {
        AudioCppBackendEvidenceState.EXPECTED,
        AudioCppBackendEvidenceState.VERIFIED,
    }
    supported: set[AudioCppBackendPreference] | None = None
    for recipe in recipes:
        current = {
            evidence.backend
            for evidence in recipe.backend_evidence
            if evidence.system == system
            and evidence.architecture == architecture
            and evidence.state in accepted_states
        }
        supported = current if supported is None else supported & current
    return frozenset(supported or ())


def _select_backend(
    preference: AudioCppBackendPreference,
    recipes: tuple[AudioCppPackageRecipe, ...],
    *,
    system: str,
    architecture: str,
) -> AudioCppBackendPreference | None:
    eligible = _eligible_backends(
        recipes,
        system=system,
        architecture=architecture,
    )
    if preference is not AudioCppBackendPreference.AUTO:
        return preference if preference in eligible else None
    order = (
        (
            AudioCppBackendPreference.METAL,
            AudioCppBackendPreference.VULKAN,
            AudioCppBackendPreference.CPU,
        )
        if system == "darwin"
        else (
            AudioCppBackendPreference.CUDA,
            AudioCppBackendPreference.HIP,
            AudioCppBackendPreference.VULKAN,
            AudioCppBackendPreference.CPU,
        )
    )
    return next((candidate for candidate in order if candidate in eligible), None)


def select_audio_cpp_guided_backend(
    preference: AudioCppBackendPreference,
    recipes: tuple[AudioCppPackageRecipe, ...],
    *,
    system: str | None = None,
    architecture: str | None = None,
) -> AudioCppBackendPreference | None:
    """Resolve one evidenced backend for the exact recipes and host.

    This pure selection seam is shared by passive Settings validation and
    deliberate launch so Save cannot promise a tuple that launch will reject.

    Args:
        preference: User-selected backend preference.
        recipes: Exact reviewed package recipes that must share one backend.
        system: Optional normalized host-system override for deterministic tests.
        architecture: Optional host-architecture override for deterministic tests.

    Returns:
        The selected evidenced backend, or ``None`` when the tuple is not
        supported on the host.
    """

    host_system = (platform.system() if system is None else system).casefold()
    if host_system not in {"darwin", "linux", "windows"}:
        return None
    host_architecture = _normalize_architecture(
        platform.machine() if architecture is None else architecture,
        system=host_system,
    )
    return _select_backend(
        preference,
        recipes,
        system=host_system,
        architecture=host_architecture,
    )


def _default_port_selector() -> int:
    span = _PRIVATE_PORT_MAX - _PRIVATE_PORT_MIN + 1
    for _ in range(_PORT_ATTEMPTS):
        candidate = _PRIVATE_PORT_MIN + secrets.randbelow(span)
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.bind(("127.0.0.1", candidate))
        except OSError:
            continue
        finally:
            listener.close()
        return candidate
    raise AudioCppGuidedLaunchError("port_unavailable")


def _selected_port(selector: Callable[[], int]) -> int | None:
    failed = False
    try:
        port = selector()
    except Exception:
        failed = True
        port = 0
    if (
        failed
        or type(port) is not int
        or not _PRIVATE_PORT_MIN <= port <= _PRIVATE_PORT_MAX
    ):
        return None
    return port


def _model_projection(
    accepted: AudioCppAcceptedPackage,
) -> dict[str, object]:
    projection = accepted.projection
    root = Path(accepted.canonical_root)
    path = (
        root
        if projection.model_relative_path is None
        else root / projection.model_relative_path
    )
    model: dict[str, object] = {
        "id": accepted.public_model_id,
        "family": projection.family,
        "path": str(path),
        "task": projection.task,
        "mode": projection.mode,
    }
    if projection.model_spec_override_relative_path is not None:
        model["model_spec_override"] = str(
            root / projection.model_spec_override_relative_path
        )
    if projection.busy_timeout_ms is not None:
        model["busy_timeout_ms"] = projection.busy_timeout_ms
    if projection.load_options:
        model["load_options"] = {
            option.name: option.value for option in projection.load_options
        }
    if projection.session_options:
        model["session_options"] = {
            option.name: option.value for option in projection.session_options
        }
    return model


def _server_document(
    settings: AudioCppSettingsConfig,
    backend: AudioCppBackendPreference,
    port: int,
) -> dict[str, object]:
    document: dict[str, object] = {
        "host": "127.0.0.1",
        "port": port,
        "backend": backend.value,
        "lazy_load": True,
        "log_request_body": False,
        "max_request_body_bytes": settings.guided_max_request_body_bytes,
        "busy_timeout_ms": settings.guided_busy_timeout_ms,
        "models": [
            _model_projection(accepted) for accepted in settings.guided_packages
        ],
    }
    if settings.guided_device is not None:
        document["device"] = settings.guided_device
    if settings.guided_threads is not None:
        document["threads"] = settings.guided_threads
    return document


def _remove_partial_artifact(
    *,
    parent_fd: int,
    directory_fd: int,
    directory_name: str,
) -> None:
    if directory_fd >= 0:
        try:
            os.unlink(_ARTIFACT_FILE, dir_fd=directory_fd)
        except OSError:
            pass
        try:
            os.close(directory_fd)
        except OSError:
            pass
    try:
        os.rmdir(directory_name, dir_fd=parent_fd)
    except OSError:
        pass
    try:
        os.close(parent_fd)
    except OSError:
        pass


def _windows_artifact_bytes(document: dict[str, object]) -> bytes:
    return (
        json.dumps(
            document,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _create_windows_artifact(
    root: Path,
    document: dict[str, object],
) -> AudioCppGeneratedLaunchArtifact | None:
    filesystem = _windows_artifact_filesystem
    if filesystem is None:
        return None
    parent: Any = None
    directory: Any = None
    file: Any = None
    extras: list[Any] = []
    directory_name = ""
    raw = _windows_artifact_bytes(document)
    server_path = root / "unpublished" / _ARTIFACT_FILE
    primary_error: BaseException | None = None
    try:
        root.mkdir(parents=True, exist_ok=True)
        parent = filesystem.protect_private_directory(root)  # type: ignore[attr-defined]
        for _ in range(_ARTIFACT_ATTEMPTS):
            candidate = f"generation-{secrets.token_hex(16)}"
            try:
                directory = filesystem.create_private_directory(root / candidate)
            except WindowsArtifactError as error:
                cleanup = error.take_cleanup_owner()
                if cleanup is not None:
                    extras.append(cleanup)
                    raise
                if (root / candidate).exists():
                    continue
                raise
            directory_name = candidate
            break
        if directory is None or not directory_name:
            raise WindowsArtifactError("unavailable")
        server_path = root / directory_name / _ARTIFACT_FILE
        file = filesystem.create_private_file(server_path, raw, read_only=True)
        artifact = _WindowsGeneratedLaunchArtifact(
            parent=parent,
            directory=directory,
            file=file,
            server_json_path=server_path,
            digest=sha256(raw).hexdigest(),
            size=len(raw),
            extras=tuple(extras),
        )
        artifact.validate()
        return artifact
    except BaseException as error:
        primary_error = error

    artifact = _WindowsGeneratedLaunchArtifact(
        parent=parent,
        directory=directory,
        file=file,
        server_json_path=server_path,
        digest=sha256(raw).hexdigest(),
        size=len(raw),
        extras=tuple(extras),
    )
    try:
        artifact.cleanup()
    except BaseException as cleanup_error:
        if not isinstance(primary_error, Exception):
            setattr(primary_error, _CLEANUP_OWNER_ATTRIBUTE, artifact)
            raise primary_error
        if not isinstance(cleanup_error, Exception):
            setattr(cleanup_error, _CLEANUP_OWNER_ATTRIBUTE, artifact)
            raise cleanup_error
        raise AudioCppGuidedLaunchError(
            "artifact_cleanup_failed",
            cleanup_owner=artifact,
        ) from None
    if primary_error is not None and not isinstance(primary_error, Exception):
        raise primary_error
    return None


def _create_artifact(
    root: Path,
    document: dict[str, object],
) -> AudioCppGeneratedLaunchArtifact | None:
    if _windows_artifact_filesystem is not None:
        return _create_windows_artifact(root, document)
    try:
        secured_root = secure_private_directory(
            root,
            create=True,
            application_owned=True,
        ).lexical_path
        parent_fd = os.open(secured_root, _DIRECTORY_FLAGS)
    except (OSError, RuntimeError, ValueError):
        return None

    directory_name = ""
    directory_fd = -1
    create_failed = False
    try:
        for _ in range(_ARTIFACT_ATTEMPTS):
            candidate = f"generation-{secrets.token_hex(16)}"
            try:
                os.mkdir(candidate, mode=0o700, dir_fd=parent_fd)
            except FileExistsError:
                continue
            directory_name = candidate
            break
        if not directory_name:
            create_failed = True
        else:
            directory_fd = os.open(
                directory_name,
                _DIRECTORY_FLAGS,
                dir_fd=parent_fd,
            )
            directory_info = os.fstat(directory_fd)
            if (
                not stat.S_ISDIR(directory_info.st_mode)
                or directory_info.st_uid != os.geteuid()
            ):
                create_failed = True
            else:
                os.fchmod(directory_fd, 0o700)
                raw = (
                    json.dumps(
                        document,
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("utf-8")
                    + b"\n"
                )
                file_fd = os.open(
                    _ARTIFACT_FILE,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                    dir_fd=directory_fd,
                )
                try:
                    offset = 0
                    while offset < len(raw):
                        written = os.write(file_fd, raw[offset:])
                        if written <= 0:
                            raise OSError
                        offset += written
                    os.fsync(file_fd)
                    os.fchmod(file_fd, 0o400)
                    file_info = os.fstat(file_fd)
                    if (
                        not stat.S_ISREG(file_info.st_mode)
                        or file_info.st_nlink != 1
                        or file_info.st_uid != os.geteuid()
                        or stat.S_IMODE(file_info.st_mode) != 0o400
                    ):
                        raise OSError
                finally:
                    os.close(file_fd)
                server_path = secured_root / directory_name / _ARTIFACT_FILE
                return AudioCppGeneratedLaunchArtifact(
                    parent_fd=parent_fd,
                    directory_fd=directory_fd,
                    directory_name=directory_name,
                    server_json_path=server_path,
                    directory_identity=(directory_info.st_dev, directory_info.st_ino),
                    file_identity=(file_info.st_dev, file_info.st_ino),
                    digest=sha256(raw).hexdigest(),
                    size=len(raw),
                )
    except (OSError, TypeError, ValueError):
        create_failed = True
    if create_failed:
        _remove_partial_artifact(
            parent_fd=parent_fd,
            directory_fd=directory_fd,
            directory_name=directory_name,
        )
    return None


def _default_artifact_root() -> Path:
    from tldw_chatbook.Utils.paths import get_user_data_dir

    return get_user_data_dir() / "audio_cpp" / "generated"


def _managed_reference(accepted: AudioCppAcceptedPackage) -> ArtifactRef | None:
    identity = accepted.managed_artifact
    if identity is None:
        return None
    return ArtifactRef(identity.artifact_id, identity.revision, identity.variant)


def _managed_root(
    leased: LeasedArtifactHandle,
    reference: ArtifactRef,
) -> Path | None:
    """Return the exact root path only from a matching acquired handle."""

    handle = leased.handle
    if handle.root != reference or reference not in handle.closure:
        return None
    paths = tuple(path for item, path in handle.paths if item == reference)
    if len(paths) != 1:
        return None
    root = Path(paths[0])
    return root if root.is_absolute() else None


async def _scan_matches_accepted(
    accepted: AudioCppAcceptedPackage,
    recipe: AudioCppPackageRecipe,
    *,
    canonical_root: Path,
) -> bool:
    try:
        if accepted.managed_artifact is None:
            scan = await scan_audio_cpp_package_root_async(canonical_root)
        else:
            scan = await scan_audio_cpp_package_root_async(
                canonical_root,
                expected_managed_artifact=accepted.managed_artifact,
                expected_canonical_root=str(canonical_root),
            )
    except Exception:
        return False
    return str(
        canonical_root
    ) == accepted.canonical_root and _candidate_matches_accepted(scan, accepted, recipe)


async def _cleanup_succeeded(artifact: AudioCppGeneratedLaunchArtifact) -> bool:
    try:
        await asyncio.to_thread(artifact.cleanup)
    except BaseException as error:
        if not isinstance(error, Exception):
            raise
        return False
    return True


_CLEANUP_OWNER_ATTRIBUTE = "_audio_cpp_generated_cleanup_owner"


def take_audio_cpp_guided_cleanup_owner(
    error: BaseException,
) -> AudioCppGeneratedLaunchArtifact | None:
    """Take an exact cleanup owner attached to preserved control flow.

    Args:
        error: The original exception object re-raised by materialization.

    Returns:
        The retained artifact owner, or ``None`` when cleanup succeeded.
    """

    take_owner = getattr(error, "take_cleanup_owner", None)
    if callable(take_owner):
        return cast(AudioCppGeneratedLaunchArtifact | None, take_owner())
    owner = getattr(error, _CLEANUP_OWNER_ATTRIBUTE, None)
    if not isinstance(owner, AudioCppGeneratedLaunchArtifact):
        return None
    setattr(error, _CLEANUP_OWNER_ATTRIBUTE, None)
    return owner


def _raise_control_after_cleanup(
    error: BaseException,
    artifact: AudioCppGeneratedLaunchArtifact,
) -> None:
    """Synchronously clean or attach ownership before bare control re-raise."""

    try:
        artifact.cleanup()
    except BaseException:
        setattr(error, _CLEANUP_OWNER_ATTRIBUTE, artifact)
    error.__traceback__ = None
    error.__context__ = None
    error.__cause__ = None
    raise error from None


def _managed_service_outcome() -> tuple[object | None, BaseException | None]:
    try:
        return managed_service(), None
    except BaseException as error:
        return None, error


def _managed_acquire_outcome(
    service: object,
    reference: ArtifactRef,
) -> tuple[LeasedArtifactHandle | None, BaseException | None]:
    try:
        activate = getattr(service, "activate")
        acquire = getattr(service, "acquire")
        activate(reference)
        return acquire(reference), None
    except BaseException as error:
        return None, error


async def _raise_guided_failure_after_cleanup(
    artifact: AudioCppGeneratedLaunchArtifact,
    code: AudioCppGuidedLaunchErrorCode,
) -> NoReturn:
    if not await _cleanup_succeeded(artifact):
        raise AudioCppGuidedLaunchError(
            "artifact_cleanup_failed",
            cleanup_owner=artifact,
        ) from None
    raise AudioCppGuidedLaunchError(code) from None


async def materialize_audio_cpp_guided_launch(
    settings: AudioCppSettingsConfig,
    *,
    artifact_root: Path | None = None,
    port_selector: Callable[[], int] | None = None,
    system: str | None = None,
    architecture: str | None = None,
) -> AudioCppManagedLaunchConfig:
    """Revalidate guided inputs and create one immutable launch snapshot.

    This is a deliberate-operation boundary. Merely loading or saving Settings
    never calls it.

    Args:
        settings: Persisted guided audio.cpp settings to revalidate.
        artifact_root: Optional private parent for the generated configuration.
        port_selector: Optional bounded loopback-port selector.
        system: Optional operating-system override used by deterministic tests.
        architecture: Optional machine-architecture override used by tests.

    Returns:
        An immutable managed launch snapshot owning its generated artifact.

    Raises:
        AudioCppGuidedLaunchError: If validation, backend selection, port
            allocation, artifact creation, or cancellation cleanup fails.
        asyncio.CancelledError: If the deliberate operation is cancelled after
            any completed artifact has been retired.
    """
    if (
        not isinstance(settings, AudioCppSettingsConfig)
        or settings.mode != "managed"
        or settings.managed_setup_source is not AudioCppManagedSetupSource.GUIDED
        or not settings.guided_packages
    ):
        raise AudioCppGuidedLaunchError("configuration_invalid") from None

    binary = await asyncio.to_thread(
        _validate_binary,
        settings.guided_binary_path,
        system=system,
        architecture=architecture,
    )
    if binary is None:
        raise AudioCppGuidedLaunchError("binary_invalid") from None
    recipes: list[AudioCppPackageRecipe] = []
    for accepted in settings.guided_packages:
        try:
            recipe = AUDIO_CPP_RECIPE_REGISTRY.validate_accepted(accepted)
        except (TypeError, ValueError):
            raise AudioCppGuidedLaunchError("package_changed") from None
        if accepted.managed_artifact is None and not await _scan_matches_accepted(
            accepted,
            recipe,
            canonical_root=Path(accepted.canonical_root),
        ):
            raise AudioCppGuidedLaunchError("package_changed") from None
        recipes.append(recipe)
    exact_recipes = tuple(recipes)

    host_system = (platform.system() if system is None else system).casefold()
    supported_host = (os.name == "posix" and host_system in {"darwin", "linux"}) or (
        host_system == "windows" and _windows_artifact_filesystem is not None
    )
    if not supported_host:
        raise AudioCppGuidedLaunchError("backend_unsupported") from None
    backend = select_audio_cpp_guided_backend(
        settings.guided_backend_preference,
        exact_recipes,
        system=host_system,
        architecture=architecture,
    )
    if backend is None:
        raise AudioCppGuidedLaunchError("backend_unsupported") from None

    selector = _default_port_selector if port_selector is None else port_selector
    port = await asyncio.to_thread(_selected_port, selector)
    if port is None:
        raise AudioCppGuidedLaunchError("port_unavailable") from None
    root = _default_artifact_root() if artifact_root is None else Path(artifact_root)
    document = _server_document(settings, backend, port)
    artifact_task = asyncio.create_task(
        asyncio.to_thread(_create_artifact, root, document)
    )
    try:
        artifact = await asyncio.shield(artifact_task)
    except asyncio.CancelledError as error:
        artifact = await asyncio.shield(artifact_task)
        if artifact is not None and not await _cleanup_succeeded(artifact):
            setattr(error, _CLEANUP_OWNER_ATTRIBUTE, artifact)
        raise
    if artifact is None:
        raise AudioCppGuidedLaunchError("artifact_create_failed") from None

    managed_failure = False
    cancellation: asyncio.CancelledError | None = None
    try:
        managed_packages = tuple(
            (accepted, recipe, reference)
            for accepted, recipe in zip(
                settings.guided_packages,
                exact_recipes,
                strict=True,
            )
            if (reference := _managed_reference(accepted)) is not None
        )
    except BaseException as error:
        if not isinstance(error, Exception):
            _raise_control_after_cleanup(error, artifact)
        await _raise_guided_failure_after_cleanup(artifact, "package_changed")
    if managed_packages:
        service_work = asyncio.to_thread(_managed_service_outcome)
        try:
            service_task = asyncio.create_task(service_work)
        except BaseException as error:
            service_work.close()
            if not isinstance(error, Exception):
                _raise_control_after_cleanup(error, artifact)
            await _raise_guided_failure_after_cleanup(artifact, "package_changed")
        try:
            service, service_error = await asyncio.shield(service_task)
        except asyncio.CancelledError as error:
            service, service_error = await asyncio.shield(service_task)
            if service_error is not None and not isinstance(service_error, Exception):
                _raise_control_after_cleanup(service_error, artifact)
            if not await _cleanup_succeeded(artifact):
                setattr(error, _CLEANUP_OWNER_ATTRIBUTE, artifact)
            raise
        if service_error is not None:
            if not isinstance(service_error, Exception):
                _raise_control_after_cleanup(service_error, artifact)
            await _raise_guided_failure_after_cleanup(artifact, "package_changed")
        assert service is not None
        for accepted, recipe, reference in managed_packages:
            acquire_work = asyncio.to_thread(
                _managed_acquire_outcome,
                service,
                reference,
            )
            try:
                acquire_task = asyncio.create_task(acquire_work)
            except BaseException as error:
                acquire_work.close()
                if not isinstance(error, Exception):
                    _raise_control_after_cleanup(error, artifact)
                managed_failure = True
                break
            try:
                leased, acquire_error = await asyncio.shield(acquire_task)
            except asyncio.CancelledError as error:
                cancellation = error
                leased, acquire_error = await asyncio.shield(acquire_task)
            if acquire_error is not None:
                if not isinstance(acquire_error, Exception):
                    _raise_control_after_cleanup(acquire_error, artifact)
                managed_failure = True
                break
            try:
                assert leased is not None
                artifact.retain_managed_handle(leased)
                canonical_root = _managed_root(leased, reference)
                matches = canonical_root is not None and await _scan_matches_accepted(
                    accepted,
                    recipe,
                    canonical_root=canonical_root,
                )
            except asyncio.CancelledError as error:
                cancellation = error
                break
            except BaseException as error:
                if not isinstance(error, Exception):
                    _raise_control_after_cleanup(error, artifact)
                managed_failure = True
                break
            if not matches:
                managed_failure = True
                break
            if cancellation is not None:
                break

    if cancellation is not None:
        if not await _cleanup_succeeded(artifact):
            setattr(cancellation, _CLEANUP_OWNER_ATTRIBUTE, artifact)
        raise cancellation
    if managed_failure:
        await _raise_guided_failure_after_cleanup(artifact, "package_changed")

    try:
        expected_models = tuple(
            AudioCppExpectedModel(
                model_id=accepted.public_model_id,
                family=accepted.projection.family,
                task=accepted.projection.task,
                mode=accepted.projection.mode,
                speech_capabilities=cast(
                    tuple[Literal["tts", "clone"], ...],
                    tuple(
                        capability
                        for capability in recipe.capabilities
                        if capability in {"tts", "clone"}
                    ),
                ),
            )
            for accepted, recipe in zip(
                settings.guided_packages,
                exact_recipes,
                strict=True,
            )
        )
        return AudioCppManagedLaunchConfig(
            binary_path=binary,
            server_json_path=artifact.server_json_path,
            working_directory=artifact.server_json_path.parent,
            base_url=f"http://127.0.0.1:{port}",
            startup_timeout_seconds=settings.managed_startup_timeout_seconds,
            health_check_interval_seconds=(
                settings.managed_health_check_interval_seconds
            ),
            termination_grace_seconds=settings.managed_termination_grace_seconds,
            expected_models=expected_models,
            generated_artifact=artifact,
        )
    except BaseException as error:
        if not isinstance(error, Exception):
            _raise_control_after_cleanup(error, artifact)
        await _raise_guided_failure_after_cleanup(artifact, "package_changed")


__all__ = (
    "AudioCppGeneratedLaunchArtifact",
    "AudioCppGuidedLaunchError",
    "AudioCppGuidedLaunchErrorCode",
    "materialize_audio_cpp_guided_launch",
    "revalidate_audio_cpp_guided_packages",
    "select_audio_cpp_guided_backend",
    "take_audio_cpp_guided_cleanup_owner",
)
