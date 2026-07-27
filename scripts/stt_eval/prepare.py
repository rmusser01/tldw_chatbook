"""Bounded, deterministic preparation of an immutable STT evaluation corpus."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import os
import random
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import wave
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import (
    BinaryIO,
    Callable,
    Iterator,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    cast,
)

import httpx
from pydantic import ValidationError

from scripts.stt_eval.io import (
    atomic_write_json,
    open_verified_file,
    read_bounded_regular_file,
    verify_file,
)
from scripts.stt_eval.schema import (
    AcquisitionMode,
    ArtifactFile,
    ConcatenationRecipe,
    ExperimentManifest,
    NoiseRecipe,
    PreparationManifest,
    PreparationReceipt,
    PreparationSource,
    SilenceRecipe,
    SourceArchiveIdentity,
    canonical_json,
    experiment_fingerprint,
)


USER_AGENT = "tldw-stt-eval/1"
RECEIPT_FILENAME = "receipt.json"
AUDIO_DIRECTORY = "audio"
CHUNK_SIZE = 1024 * 1024
MAX_RECEIPT_BYTES = 16 * 1024 * 1024
PCM_SAMPLE_RATE = 16_000
PCM_CHANNELS = 1
PCM_SAMPLE_WIDTH = 2


class PreparationError(RuntimeError):
    """Fail-closed corpus preparation error."""


class CommandRunner(Protocol):
    """The subprocess seam used by the real executable-boundary tests."""

    def __call__(
        self,
        arguments: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]: ...


class HttpStreamClient(Protocol):
    """Minimal streaming client contract accepted by preparation."""

    def stream(
        self, method: str, url: str, **kwargs: object
    ) -> AbstractContextManager[httpx.Response]: ...


FreeSpace = Callable[[Path], int]
Publisher = Callable[[Path, Path], None]


def _available_space(path: Path) -> int:
    return shutil.disk_usage(path).free


def _path_lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    except OSError as error:
        if error.errno not in {errno.EINVAL, errno.ENOTSUP, errno.EBADF}:
            raise
    finally:
        os.close(descriptor)


def _atomic_publish_directory(source: Path, destination: Path) -> None:
    """Atomically rename a directory without ever replacing a destination."""

    if source.parent != destination.parent:
        raise RuntimeError("atomic corpus publication requires one parent directory")
    if sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            renamex_np = libc.renamex_np
        except AttributeError as error:
            raise RuntimeError(
                "platform lacks an atomic no-replace directory rename"
            ) from error
        renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex_np.restype = ctypes.c_int
        result = renamex_np(
            os.fsencode(source),
            os.fsencode(destination),
            0x00000004,  # RENAME_EXCL
        )
    elif sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            renameat2 = libc.renameat2
        except AttributeError as error:
            raise RuntimeError(
                "platform lacks an atomic no-replace directory rename"
            ) from error
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,  # AT_FDCWD
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            1,  # RENAME_NOREPLACE
        )
    elif os.name == "nt":
        os.rename(source, destination)
        result = 0
    else:
        raise RuntimeError("platform lacks an atomic no-replace directory rename")

    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(
                f"destination already exists: {destination}"
            ) from None
        raise OSError(
            error_number,
            os.strerror(error_number),
            os.fspath(destination),
        )
    _fsync_directory(destination.parent)


@dataclass(frozen=True)
class PreparationPreflight:
    """Read-only acquisition, storage, and tool plan."""

    source_ids: tuple[str, ...]
    source_locations: tuple[str, ...]
    licenses: tuple[str, ...]
    transfer_bytes: int
    staging_bytes: int
    required_free_bytes: int
    available_bytes: int
    destination: Path
    required_local_inputs: tuple[Path, ...]
    missing_local_inputs: tuple[Path, ...]
    missing_tools: tuple[str, ...]


@dataclass(frozen=True)
class PrepareRequest:
    """Explicit local execution inputs and injectable external boundaries."""

    manifest: PreparationManifest
    experiment: ExperimentManifest
    destination: Path
    local_inputs: Mapping[str, Path]
    ffmpeg_executable: Path
    http_client: HttpStreamClient | None = None
    command_runner: CommandRunner = cast(CommandRunner, subprocess.run)
    free_space: FreeSpace = _available_space
    publisher: Publisher = _atomic_publish_directory

    def with_destination(self, destination: Path) -> "PrepareRequest":
        return replace(self, destination=Path(destination))

    def with_local_inputs(self, local_inputs: Mapping[str, Path]) -> "PrepareRequest":
        return replace(self, local_inputs=local_inputs)

    def with_free_space(self, available: int) -> "PrepareRequest":
        return replace(self, free_space=lambda _path: available)


def _is_executable(path: Path) -> bool:
    raw = os.fspath(path)
    if path.is_absolute() or os.sep in raw:
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            return False
        return (
            stat.S_ISREG(metadata.st_mode)
            and not stat.S_ISLNK(metadata.st_mode)
            and os.access(path, os.X_OK)
        )
    located = shutil.which(raw)
    return located is not None


def _prepared_artifacts(manifest: PreparationManifest) -> tuple[ArtifactFile, ...]:
    normalized = tuple(recipe.prepared_file for recipe in manifest.normalized_samples)
    derived = tuple(recipe.prepared_file for recipe in manifest.derived_recipes)
    return normalized + derived


def _validate_experiment_closure(request: PrepareRequest) -> None:
    if not isinstance(request.experiment, ExperimentManifest):
        raise PreparationError("validated ExperimentManifest is required")

    preparation_sources = {
        source.source_id: (
            source.dataset_family,
            source.repository,
            source.revision,
            source.source_url,
            source.license,
            source.archive,
        )
        for source in request.manifest.sources
    }
    experiment_sources = {
        source.source_id: (
            source.dataset_family,
            source.repository,
            source.revision,
            source.source_url,
            source.license,
            source.artifact,
        )
        for source in request.experiment.corpus.sources
    }
    if preparation_sources != experiment_sources:
        raise PreparationError(
            "preparation/experiment closure mismatch for source provenance"
        )

    preparation_samples = {
        recipe.sample_id: recipe.prepared_file
        for recipe in (
            *request.manifest.normalized_samples,
            *request.manifest.derived_recipes,
        )
    }
    experiment_samples = {
        sample.sample_id: sample.prepared_file
        for sample in request.experiment.corpus.samples
    }
    if preparation_samples != experiment_samples:
        raise PreparationError(
            "preparation/experiment closure mismatch for prepared samples"
        )

    experiment_sample_sources = {
        sample.sample_id: sample.source_id
        for sample in request.experiment.corpus.samples
    }
    if any(
        experiment_sample_sources[recipe.sample_id] != recipe.source_id
        for recipe in request.manifest.normalized_samples
    ):
        raise PreparationError(
            "preparation/experiment closure mismatch for sample provenance"
        )

    recipe_revisions = tuple(
        dict.fromkeys(
            recipe.recipe_revision
            for recipe in (
                *request.manifest.normalized_samples,
                *request.manifest.derived_recipes,
            )
        )
    )
    if recipe_revisions != request.experiment.corpus.derived_recipe_revisions:
        raise PreparationError(
            "preparation/experiment closure mismatch for recipe revisions"
        )


def preflight(request: PrepareRequest) -> PreparationPreflight:
    """Return a read-only plan; never transfer, extract, invoke, or publish."""

    _validate_experiment_closure(request)
    destination = Path(request.destination)
    parent = destination.parent
    if not parent.is_dir():
        available_bytes = 0
    else:
        available_bytes = request.free_space(parent)
    if isinstance(available_bytes, bool) or available_bytes < 0:
        raise PreparationError("available space must be a non-negative integer")

    transfer_bytes = sum(
        source.archive.size_bytes
        for source in request.manifest.sources
        if source.acquisition_mode is AcquisitionMode.VERIFIED_DOWNLOAD
    )
    archive_bytes = sum(
        source.archive.size_bytes for source in request.manifest.sources
    )
    maximum_extracted_bytes = request.manifest.limits.max_uncompressed_bytes * len(
        request.manifest.sources
    )
    prepared_bytes = sum(
        output.size_bytes for output in _prepared_artifacts(request.manifest)
    )
    staging_bytes = archive_bytes + maximum_extracted_bytes + prepared_bytes
    required_free_bytes = staging_bytes + request.manifest.limits.staging_headroom_bytes

    required_local_inputs: list[Path] = []
    missing_local_inputs: list[Path] = []
    locations: list[str] = []
    for source in request.manifest.sources:
        if source.acquisition_mode is AcquisitionMode.VERIFIED_DOWNLOAD:
            locations.append(source.source_url)
            continue
        supplied = request.local_inputs.get(source.source_id)
        input_path = Path(supplied) if supplied is not None else Path(source.source_id)
        required_local_inputs.append(input_path)
        locations.append(os.fspath(input_path))
        if supplied is None or not input_path.is_file() or input_path.is_symlink():
            missing_local_inputs.append(input_path)

    executable = Path(request.ffmpeg_executable)
    missing_tools = () if _is_executable(executable) else (os.fspath(executable),)
    return PreparationPreflight(
        source_ids=tuple(source.source_id for source in request.manifest.sources),
        source_locations=tuple(locations),
        licenses=tuple(source.license for source in request.manifest.sources),
        transfer_bytes=transfer_bytes,
        staging_bytes=staging_bytes,
        required_free_bytes=required_free_bytes,
        available_bytes=available_bytes,
        destination=destination,
        required_local_inputs=tuple(required_local_inputs),
        missing_local_inputs=tuple(missing_local_inputs),
        missing_tools=missing_tools,
    )


def prepare(
    request: PrepareRequest,
    *,
    execute: bool = False,
) -> Path | PreparationPreflight:
    """Preflight by default; mutate external state only through the explicit gate."""

    plan = preflight(request)
    if not execute:
        return plan

    destination = Path(request.destination)
    if _path_lexists(destination):
        if plan.missing_tools:
            raise PreparationError("missing tool(s): " + ", ".join(plan.missing_tools))
        current_ffmpeg_version = _ffmpeg_version(request)
        _verify_existing_destination(
            request,
            destination,
            current_ffmpeg_version,
        )
        return destination

    blockers: list[str] = []
    if plan.missing_local_inputs:
        blockers.append(
            "missing local input(s): "
            + ", ".join(os.fspath(path) for path in plan.missing_local_inputs)
        )
    if plan.missing_tools:
        blockers.append("missing tool(s): " + ", ".join(plan.missing_tools))
    if plan.available_bytes < plan.required_free_bytes:
        blockers.append(
            "insufficient free space: "
            f"required {plan.required_free_bytes}, available {plan.available_bytes}"
        )
    if blockers:
        raise PreparationError("; ".join(blockers))

    return _execute_preparation(request)


def _run_command(
    request: PrepareRequest,
    arguments: list[str],
) -> subprocess.CompletedProcess[str]:
    try:
        return request.command_runner(
            arguments,
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise PreparationError(f"ffmpeg invocation failed: {error}") from error


def _ffmpeg_version(request: PrepareRequest) -> str:
    arguments = [os.fspath(request.ffmpeg_executable), "-version"]
    result = _run_command(request, arguments)
    if result.returncode != 0:
        raise PreparationError(
            f"ffmpeg version check failed with exit code {result.returncode}"
        )
    first_line = result.stdout.splitlines()[0] if result.stdout else ""
    if not first_line:
        raise PreparationError("ffmpeg version check returned no version line")
    return first_line


def _exclusive_binary_writer(path: Path) -> BinaryIO:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise PreparationError(f"unsafe staging parent: {path.parent}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    return os.fdopen(descriptor, "wb")


@contextmanager
def _default_http_client() -> Iterator[httpx.Client]:
    timeout = httpx.Timeout(30.0, connect=10.0)
    with httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
        follow_redirects=False,
        max_redirects=3,
        trust_env=False,
    ) as client:
        yield client


def _download_archive(
    request: PrepareRequest,
    source: PreparationSource,
    destination: Path,
) -> None:
    digest = hashlib.sha256()
    byte_count = 0
    client_context: AbstractContextManager[HttpStreamClient]
    if request.http_client is None:
        client_context = _default_http_client()
    else:
        _reject_client_credentials(request.http_client)
        client_context = _borrowed_client(request.http_client)

    try:
        with client_context as client:
            with client.stream(
                "GET",
                source.source_url,
                headers={"User-Agent": USER_AGENT},
                follow_redirects=False,
            ) as response:
                if (
                    response.history
                    or str(response.url) != source.source_url
                    or response.is_redirect
                ):
                    raise PreparationError(f"redirect rejected for {source.source_id}")
                response.raise_for_status()
                with _exclusive_binary_writer(destination) as output:
                    for chunk in response.iter_bytes(CHUNK_SIZE):
                        if not chunk:
                            continue
                        byte_count += len(chunk)
                        if byte_count > source.archive.size_bytes:
                            raise PreparationError(
                                f"download size overflow for {source.source_id}"
                            )
                        digest.update(chunk)
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
    except PreparationError:
        raise
    except (httpx.HTTPError, OSError) as error:
        raise PreparationError(
            f"verified download failed for {source.source_id}: {error}"
        ) from error

    if byte_count != source.archive.size_bytes:
        raise PreparationError(
            f"download size mismatch for {source.source_id}: "
            f"expected {source.archive.size_bytes}, got {byte_count}"
        )
    actual_digest = digest.hexdigest()
    if actual_digest != source.archive.sha256:
        raise PreparationError(
            f"download SHA-256 mismatch for {source.source_id}: "
            f"expected {source.archive.sha256}, got {actual_digest}"
        )


@contextmanager
def _borrowed_client(client: HttpStreamClient) -> Iterator[HttpStreamClient]:
    yield client


def _reject_client_credentials(client: HttpStreamClient) -> None:
    headers = getattr(client, "headers", {})
    sensitive_headers = {
        "authorization",
        "proxy-authorization",
        "cookie",
    }
    if any(str(name).lower() in sensitive_headers for name in headers):
        raise PreparationError("injected HTTP client must not contain credentials")
    cookies = getattr(client, "cookies", ())
    if cookies:
        raise PreparationError("injected HTTP client must not contain credentials")
    if getattr(client, "_auth", None) is not None:
        raise PreparationError("injected HTTP client must not contain credentials")


def _archive_mode(filename: str) -> Literal["r:", "r:gz"]:
    if filename.endswith(".tar"):
        return "r:"
    if filename.endswith((".tar.gz", ".tgz")):
        return "r:gz"
    raise PreparationError("source archive format must be .tar, .tar.gz, or .tgz")


def _validate_member_name(name: str) -> None:
    posix = PurePosixPath(name)
    windows = PureWindowsPath(name)
    parts = name.split("/")
    windows_reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{number}" for number in range(1, 10)),
        *(f"LPT{number}" for number in range(1, 10)),
    }
    unsafe_windows_component = any(
        part.endswith((".", " "))
        or ":" in part
        or part.split(".", maxsplit=1)[0].upper() in windows_reserved
        for part in parts
    )
    if (
        not name
        or name.startswith("/")
        or name.endswith("/")
        or "\\" in name
        or posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or posix.as_posix() != name
        or any(part in {"", ".", ".."} for part in parts)
        or unsafe_windows_component
    ):
        raise PreparationError(f"unsafe archive member name: {name!r}")


def _member_is_sparse(member: tarfile.TarInfo) -> bool:
    sparse_headers = any(
        key.startswith(("GNU.sparse", "SCHILY.realsize")) for key in member.pax_headers
    )
    try:
        sparse = member.issparse()
    except AttributeError:
        sparse = False
    return bool(sparse or sparse_headers)


def _scan_archive(
    request: PrepareRequest,
    source: PreparationSource,
    stream: BinaryIO,
    staging: Path,
) -> tuple[tarfile.TarInfo, ...]:
    declared = {member.name: member for member in source.members}
    seen: set[str] = set()
    ambiguous_seen: set[str] = set()
    seen_prefixes: dict[tuple[str, ...], tuple[str, ...]] = {}
    members: list[tarfile.TarInfo] = []
    total = 0
    stream.seek(0)
    try:
        with tarfile.open(
            fileobj=stream,
            mode=_archive_mode(source.archive.filename),
        ) as archive:
            for archive_member in archive:
                if len(members) >= request.manifest.limits.max_member_count:
                    raise PreparationError("archive member count limit exceeded")
                _validate_member_name(archive_member.name)
                comparison_name = archive_member.name.casefold()
                if archive_member.name in seen or comparison_name in ambiguous_seen:
                    raise PreparationError(
                        f"duplicate archive member: {archive_member.name!r}"
                    )
                seen.add(archive_member.name)
                ambiguous_seen.add(comparison_name)
                exact_prefix: list[str] = []
                comparison_prefix: list[str] = []
                for part in archive_member.name.split("/"):
                    exact_prefix.append(part)
                    comparison_prefix.append(part.casefold())
                    key = tuple(comparison_prefix)
                    exact = tuple(exact_prefix)
                    previous = seen_prefixes.setdefault(key, exact)
                    if previous != exact:
                        raise PreparationError("ambiguous archive member parent paths")
                if not archive_member.isreg() or _member_is_sparse(archive_member):
                    raise PreparationError(
                        "archive members must be non-sparse regular files"
                    )
                expected = declared.get(archive_member.name)
                if expected is None:
                    raise PreparationError(
                        f"unknown archive member: {archive_member.name!r}"
                    )
                if archive_member.size != expected.size_bytes:
                    raise PreparationError(
                        f"archive member size mismatch: {archive_member.name!r}"
                    )
                if archive_member.size > request.manifest.limits.max_file_bytes:
                    raise PreparationError(
                        f"archive per-file byte limit exceeded: {archive_member.name!r}"
                    )
                total += archive_member.size
                if total > request.manifest.limits.max_uncompressed_bytes:
                    raise PreparationError("archive uncompressed byte limit exceeded")
                members.append(archive_member)
    except PreparationError:
        raise
    except (tarfile.TarError, OSError) as error:
        raise PreparationError(
            f"invalid source archive {source.source_id}: {error}"
        ) from error

    missing = set(declared) - seen
    if missing:
        raise PreparationError(
            "missing declared archive member(s): " + ", ".join(sorted(missing))
        )
    required_space = (
        sum(
            member.size_bytes
            for member in source.members
            if member.selected_for_preparation
        )
        + request.manifest.limits.staging_headroom_bytes
    )
    if request.free_space(staging) < required_space:
        raise PreparationError("insufficient staging free space during archive scan")
    return tuple(members)


def _extract_and_verify_archive(
    request: PrepareRequest,
    source: PreparationSource,
    stream: BinaryIO,
    staging: Path,
    output_root: Path,
) -> dict[tuple[str, str], Path]:
    scanned = _scan_archive(request, source, stream, staging)
    expected = {member.name: member for member in source.members}
    extracted: dict[tuple[str, str], Path] = {}
    total_actual = 0
    stream.seek(0)
    try:
        with tarfile.open(
            fileobj=stream,
            mode=_archive_mode(source.archive.filename),
        ) as archive:
            for scanned_member in scanned:
                archive_member = archive.next()
                if archive_member is None or archive_member.name != scanned_member.name:
                    raise PreparationError(
                        "archive member order changed between scan and extraction"
                    )
                declaration = expected[archive_member.name]
                member_stream = archive.extractfile(archive_member)
                if member_stream is None:
                    raise PreparationError(
                        f"unable to stream archive member {archive_member.name!r}"
                    )
                digest = hashlib.sha256()
                actual = 0
                output_path = output_root.joinpath(
                    *PurePosixPath(archive_member.name).parts
                )
                output: BinaryIO | None = None
                try:
                    if declaration.selected_for_preparation:
                        output = _exclusive_binary_writer(output_path)
                    while True:
                        chunk = member_stream.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        prospective = actual + len(chunk)
                        prospective_total = total_actual + len(chunk)
                        if (
                            prospective > declaration.size_bytes
                            or prospective > request.manifest.limits.max_file_bytes
                            or prospective_total
                            > request.manifest.limits.max_uncompressed_bytes
                        ):
                            raise PreparationError(
                                f"actual archive stream overflow: "
                                f"{archive_member.name!r}"
                            )
                        if (
                            request.free_space(staging)
                            < len(chunk)
                            + request.manifest.limits.staging_headroom_bytes
                        ):
                            raise PreparationError(
                                "insufficient staging free space during extraction"
                            )
                        actual = prospective
                        total_actual = prospective_total
                        digest.update(chunk)
                        if output is not None:
                            output.write(chunk)
                    if actual != declaration.size_bytes:
                        raise PreparationError(
                            f"archive member size mismatch: {archive_member.name!r}"
                        )
                    if digest.hexdigest() != declaration.sha256:
                        raise PreparationError(
                            f"archive member SHA-256 mismatch: {archive_member.name!r}"
                        )
                    if output is not None:
                        output.flush()
                        os.fsync(output.fileno())
                finally:
                    member_stream.close()
                    if output is not None:
                        output.close()
                if declaration.selected_for_preparation:
                    verify_file(
                        output_path,
                        expected_size=declaration.size_bytes,
                        expected_sha256=declaration.sha256,
                    )
                    extracted[(source.source_id, declaration.name)] = output_path
    except PreparationError:
        raise
    except (tarfile.TarError, OSError) as error:
        raise PreparationError(
            f"source extraction failed for {source.source_id}: {error}"
        ) from error
    return extracted


@contextmanager
def _verified_archive(
    source: PreparationSource,
    archive_path: Path,
) -> Iterator[BinaryIO]:
    try:
        with open_verified_file(
            archive_path,
            expected_size=source.archive.size_bytes,
            expected_sha256=source.archive.sha256,
        ) as verified:
            yield verified.stream
    except (FileNotFoundError, ValueError) as error:
        raise PreparationError(
            f"source archive verification failed for {source.source_id}: {error}"
        ) from error


def _verify_pcm_wave(path: Path) -> tuple[int, int]:
    try:
        with wave.open(os.fspath(path), "rb") as stream:
            if (
                stream.getnchannels() != PCM_CHANNELS
                or stream.getsampwidth() != PCM_SAMPLE_WIDTH
                or stream.getframerate() != PCM_SAMPLE_RATE
                or stream.getcomptype() != "NONE"
            ):
                raise PreparationError(
                    f"prepared audio is not PCM16 mono 16000 Hz: {path.name}"
                )
            frames = stream.getnframes()
            if frames <= 0:
                raise PreparationError(f"prepared audio has no frames: {path.name}")
            return frames, frames * PCM_SAMPLE_WIDTH
    except (EOFError, wave.Error, OSError) as error:
        raise PreparationError(
            f"prepared audio is not a valid WAV: {path.name}: {error}"
        ) from error


def _check_prepared_file_bound(
    request: PrepareRequest,
    artifact: ArtifactFile,
) -> None:
    if artifact.size_bytes > request.manifest.limits.max_file_bytes:
        raise PreparationError(
            f"prepared output exceeds per-file byte limit: {artifact.filename}"
        )


def _normalize_audio(
    request: PrepareRequest,
    source_path: Path,
    destination: Path,
    expected: ArtifactFile,
) -> None:
    _check_prepared_file_bound(request, expected)
    arguments = [
        os.fspath(request.ffmpeg_executable),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        os.fspath(source_path),
        "-map_metadata",
        "-1",
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(PCM_SAMPLE_RATE),
        "-ac",
        str(PCM_CHANNELS),
        "-n",
        os.fspath(destination),
    ]
    result = _run_command(request, arguments)
    if result.returncode != 0:
        raise PreparationError(
            "ffmpeg conversion failed for "
            f"{expected.filename} with exit code {result.returncode}: "
            f"{result.stderr.strip()}"
        )
    if not destination.is_file() or destination.is_symlink():
        raise PreparationError(
            f"ffmpeg did not produce a fresh regular output: {expected.filename}"
        )
    try:
        verify_file(
            destination,
            expected_size=expected.size_bytes,
            expected_sha256=expected.sha256,
        )
    except (FileNotFoundError, ValueError) as error:
        raise PreparationError(str(error)) from error
    _verify_pcm_wave(destination)


def _wave_frame_count(path: Path) -> int:
    frames, _ = _verify_pcm_wave(path)
    return frames


def _write_pcm_wave(
    request: PrepareRequest,
    destination: Path,
    frame_count: int,
    frame_chunks: Iterator[bytes],
    expected: ArtifactFile,
) -> None:
    _check_prepared_file_bound(request, expected)
    if frame_count <= 0:
        raise PreparationError("derived output must contain at least one frame")
    predicted_minimum = 44 + frame_count * PCM_SAMPLE_WIDTH
    if predicted_minimum > request.manifest.limits.max_file_bytes:
        raise PreparationError(
            f"prepared output exceeds per-file byte limit: {expected.filename}"
        )
    with _exclusive_binary_writer(destination) as raw:
        with wave.open(raw, "wb") as output:
            output.setnchannels(PCM_CHANNELS)
            output.setsampwidth(PCM_SAMPLE_WIDTH)
            output.setframerate(PCM_SAMPLE_RATE)
            written_frames = 0
            for chunk in frame_chunks:
                if len(chunk) % PCM_SAMPLE_WIDTH:
                    raise PreparationError("derived PCM chunk is not frame-aligned")
                chunk_frames = len(chunk) // PCM_SAMPLE_WIDTH
                if written_frames + chunk_frames > frame_count:
                    raise PreparationError("derived output frame overflow")
                output.writeframesraw(chunk)
                written_frames += chunk_frames
            if written_frames != frame_count:
                raise PreparationError("derived output frame count mismatch")
        raw.flush()
        os.fsync(raw.fileno())
    try:
        verify_file(
            destination,
            expected_size=expected.size_bytes,
            expected_sha256=expected.sha256,
        )
    except (FileNotFoundError, ValueError) as error:
        raise PreparationError(str(error)) from error
    _verify_pcm_wave(destination)


def _zero_chunks(frame_count: int) -> Iterator[bytes]:
    remaining = frame_count * PCM_SAMPLE_WIDTH
    zero_chunk = b"\0" * min(CHUNK_SIZE, max(PCM_SAMPLE_WIDTH, remaining))
    while remaining:
        chunk = zero_chunk[:remaining]
        yield chunk
        remaining -= len(chunk)


def _noise_chunks(
    source_path: Path,
    *,
    seed: int,
    amplitude: float,
    source_gain: float,
) -> Iterator[bytes]:
    generator = random.Random(seed)
    peak = round(32767 * amplitude)
    with wave.open(os.fspath(source_path), "rb") as source:
        while True:
            frames = source.readframes(CHUNK_SIZE // PCM_SAMPLE_WIDTH)
            if not frames:
                break
            output = bytearray()
            for index in range(0, len(frames), PCM_SAMPLE_WIDTH):
                sample = int.from_bytes(
                    frames[index : index + PCM_SAMPLE_WIDTH],
                    "little",
                    signed=True,
                )
                noisy = round(sample * source_gain) + generator.randint(-peak, peak)
                clipped = max(-32768, min(32767, noisy))
                output.extend(clipped.to_bytes(2, "little", signed=True))
            yield bytes(output)


def _concatenation_chunks(
    paths: Sequence[Path],
    gaps: Sequence[float],
) -> Iterator[bytes]:
    for index, path in enumerate(paths):
        if index:
            yield from _zero_chunks(round(gaps[index - 1] * PCM_SAMPLE_RATE))
        with wave.open(os.fspath(path), "rb") as source:
            while True:
                frames = source.readframes(CHUNK_SIZE // PCM_SAMPLE_WIDTH)
                if not frames:
                    break
                yield frames


def _derive_audio(
    request: PrepareRequest,
    audio_root: Path,
    samples: dict[str, Path],
) -> None:
    for recipe in request.manifest.derived_recipes:
        destination = audio_root / recipe.prepared_file.filename
        if isinstance(recipe, SilenceRecipe):
            frame_count = round(recipe.duration_seconds * PCM_SAMPLE_RATE)
            chunks = _zero_chunks(frame_count)
        elif isinstance(recipe, NoiseRecipe):
            source_path = samples.get(recipe.source_sample_id)
            if source_path is None:
                raise PreparationError(
                    f"noise recipe references unknown input {recipe.source_sample_id!r}"
                )
            frame_count = _wave_frame_count(source_path)
            chunks = _noise_chunks(
                source_path,
                seed=recipe.seed,
                amplitude=recipe.noise_amplitude,
                source_gain=recipe.source_gain,
            )
        elif isinstance(recipe, ConcatenationRecipe):
            try:
                paths = [samples[sample_id] for sample_id in recipe.source_sample_ids]
            except KeyError as error:
                raise PreparationError(
                    f"concatenation references unknown input {error.args[0]!r}"
                ) from error
            frame_count = sum(_wave_frame_count(path) for path in paths) + sum(
                round(gap * PCM_SAMPLE_RATE) for gap in recipe.silence_gaps_seconds
            )
            chunks = _concatenation_chunks(paths, recipe.silence_gaps_seconds)
        else:  # pragma: no cover - the discriminated schema closes this branch
            raise PreparationError("unknown derived recipe type")
        _write_pcm_wave(
            request,
            destination,
            frame_count,
            chunks,
            recipe.prepared_file,
        )
        samples[recipe.sample_id] = destination


def _expected_receipt(
    request: PrepareRequest,
    ffmpeg_version: str,
) -> PreparationReceipt:
    return PreparationReceipt(
        schema_version=1,
        status="complete",
        experiment_fingerprint=experiment_fingerprint(request.experiment),
        preparation_manifest_sha256=hashlib.sha256(
            canonical_json(request.manifest)
        ).hexdigest(),
        ffmpeg_executable=os.fspath(request.ffmpeg_executable),
        ffmpeg_version=ffmpeg_version,
        source_archives=tuple(
            SourceArchiveIdentity(
                source_id=source.source_id,
                archive=source.archive,
            )
            for source in request.manifest.sources
        ),
        prepared_files=_prepared_artifacts(request.manifest),
    )


def _load_receipt(destination: Path) -> PreparationReceipt:
    receipt_path = destination / RECEIPT_FILENAME
    try:
        payload = read_bounded_regular_file(
            receipt_path,
            max_bytes=MAX_RECEIPT_BYTES,
        )
        return PreparationReceipt.model_validate_json(payload)
    except (FileNotFoundError, OSError, ValueError, ValidationError) as error:
        raise PreparationError(
            f"existing destination has an invalid receipt: {error}"
        ) from error


def _verify_corpus_directory(
    request: PrepareRequest,
    destination: Path,
    receipt: PreparationReceipt,
    current_ffmpeg_version: str,
) -> None:
    if not destination.is_dir() or destination.is_symlink():
        raise PreparationError("existing destination is not a safe directory")
    expected_files = _prepared_artifacts(request.manifest)
    expected_source_archives = tuple(
        SourceArchiveIdentity(source_id=source.source_id, archive=source.archive)
        for source in request.manifest.sources
    )
    if receipt.ffmpeg_version != current_ffmpeg_version:
        raise PreparationError("completion receipt ffmpeg version mismatch")
    if (
        receipt.experiment_fingerprint != experiment_fingerprint(request.experiment)
        or receipt.preparation_manifest_sha256
        != hashlib.sha256(canonical_json(request.manifest)).hexdigest()
        or receipt.ffmpeg_executable != os.fspath(request.ffmpeg_executable)
        or receipt.source_archives != expected_source_archives
        or receipt.prepared_files != expected_files
    ):
        raise PreparationError("existing destination receipt identity mismatch")

    expected_relative = {
        f"{AUDIO_DIRECTORY}/{artifact.filename}" for artifact in expected_files
    }
    expected_relative.add(RECEIPT_FILENAME)
    actual_relative: set[str] = set()
    actual_directories: set[str] = set()
    try:
        for root, directories, filenames in os.walk(
            destination, topdown=True, followlinks=False
        ):
            root_path = Path(root)
            for directory in directories:
                directory_path = root_path / directory
                if directory_path.is_symlink():
                    raise PreparationError("existing destination contains a symlink")
                actual_directories.add(
                    directory_path.relative_to(destination).as_posix()
                )
            for filename in filenames:
                file_path = root_path / filename
                if file_path.is_symlink():
                    raise PreparationError("existing destination contains a symlink")
                actual_relative.add(file_path.relative_to(destination).as_posix())
    except OSError as error:
        raise PreparationError(
            f"unable to verify existing destination: {error}"
        ) from error
    if actual_relative != expected_relative or actual_directories != {AUDIO_DIRECTORY}:
        raise PreparationError("existing destination file set mismatch")

    for artifact in expected_files:
        try:
            verify_file(
                Path(AUDIO_DIRECTORY) / artifact.filename,
                root=destination,
                expected_size=artifact.size_bytes,
                expected_sha256=artifact.sha256,
            )
        except (FileNotFoundError, ValueError) as error:
            raise PreparationError(
                f"existing destination file verification failed: {error}"
            ) from error


def _verify_existing_destination(
    request: PrepareRequest,
    destination: Path,
    current_ffmpeg_version: str,
) -> None:
    try:
        receipt = _load_receipt(destination)
        _verify_corpus_directory(
            request,
            destination,
            receipt,
            current_ffmpeg_version,
        )
    except PreparationError as error:
        raise PreparationError(f"invalid existing destination: {error}") from error


def _fsync_tree(root: Path) -> None:
    directories: list[Path] = []
    for current_root, child_directories, filenames in os.walk(
        root, topdown=True, followlinks=False
    ):
        current = Path(current_root)
        directories.append(current)
        for directory in child_directories:
            path = current / directory
            if path.is_symlink():
                raise PreparationError("staged corpus contains a symlink")
        for filename in filenames:
            path = current / filename
            if path.is_symlink():
                raise PreparationError("staged corpus contains a symlink")
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    for staged_directory in reversed(directories):
        _fsync_directory(staged_directory)


def _execute_preparation(request: PrepareRequest) -> Path:
    destination = Path(request.destination)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            suffix=".staging",
            dir=destination.parent,
        )
    )
    staging.chmod(0o700)
    try:
        work = staging / ".work"
        archives_root = work / "archives"
        extracted_root = work / "extracted"
        audio_root = staging / AUDIO_DIRECTORY
        archives_root.mkdir(mode=0o700, parents=True)
        extracted_root.mkdir(mode=0o700)
        audio_root.mkdir(mode=0o700)

        ffmpeg_version = _ffmpeg_version(request)
        extracted: dict[tuple[str, str], Path] = {}
        for source in request.manifest.sources:
            source_staging_name = hashlib.sha256(
                source.source_id.encode("utf-8")
            ).hexdigest()
            if source.acquisition_mode is AcquisitionMode.VERIFIED_DOWNLOAD:
                archive_path = (
                    archives_root / source_staging_name / source.archive.filename
                )
                _download_archive(request, source, archive_path)
            else:
                supplied = request.local_inputs.get(source.source_id)
                if supplied is None:
                    raise PreparationError(
                        f"missing local input for {source.source_id}"
                    )
                archive_path = Path(supplied)
            with _verified_archive(source, archive_path) as archive_stream:
                extracted.update(
                    _extract_and_verify_archive(
                        request,
                        source,
                        archive_stream,
                        staging,
                        extracted_root / source_staging_name,
                    )
                )

        samples: dict[str, Path] = {}
        prepared_total = 0
        for recipe in request.manifest.normalized_samples:
            source_path = extracted.get((recipe.source_id, recipe.source_member))
            if source_path is None:
                raise PreparationError(
                    "normalization references an unavailable verified source member"
                )
            destination_path = audio_root / recipe.prepared_file.filename
            _normalize_audio(
                request,
                source_path,
                destination_path,
                recipe.prepared_file,
            )
            prepared_total += recipe.prepared_file.size_bytes
            if prepared_total > request.manifest.limits.max_uncompressed_bytes:
                raise PreparationError("prepared output cumulative byte limit exceeded")
            samples[recipe.sample_id] = destination_path

        _derive_audio(request, audio_root, samples)
        prepared_total = sum(
            artifact.size_bytes for artifact in _prepared_artifacts(request.manifest)
        )
        if prepared_total > request.manifest.limits.max_uncompressed_bytes:
            raise PreparationError("prepared output cumulative byte limit exceeded")

        shutil.rmtree(work)
        receipt = _expected_receipt(request, ffmpeg_version)
        atomic_write_json(staging / RECEIPT_FILENAME, receipt)
        staged_receipt = _load_receipt(staging)
        _verify_corpus_directory(
            request,
            staging,
            staged_receipt,
            ffmpeg_version,
        )
        _fsync_tree(staging)
        try:
            request.publisher(staging, destination)
        except (FileExistsError, OSError, RuntimeError) as error:
            raise PreparationError(f"corpus publication failed: {error}") from error
        return destination
    except PreparationError:
        raise
    except (OSError, ValueError, ValidationError) as error:
        raise PreparationError(f"corpus preparation failed: {error}") from error
    finally:
        if _path_lexists(staging):
            shutil.rmtree(staging, ignore_errors=True)


__all__ = [
    "PrepareRequest",
    "PreparationError",
    "PreparationPreflight",
    "prepare",
    "preflight",
]
