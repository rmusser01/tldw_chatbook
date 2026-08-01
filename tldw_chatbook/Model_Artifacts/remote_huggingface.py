"""Bounded metadata requests to Hugging Face's fixed public API origin."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping
from urllib.parse import quote

import httpx

from .service import (
    ArtifactDescriptor,
    ArtifactFile,
    ArtifactFormat,
    ArtifactRef,
    ArtifactRole,
    ProvenanceClass,
)

_API_MODELS_URL = "https://huggingface.co/api/models"
_MAX_METADATA_BYTES = 2 * 1024 * 1024
_MAX_SEARCH_RESULTS = 50
_MAX_QUERY_CHARACTERS = 256
_MAX_REPOSITORY_CHARACTERS = 96
_MAX_FILE_ENTRIES = 2_048
_MAX_GGUF_CANDIDATES = 100
_MAX_SHARDS = 64
_MAX_PATH_BYTES = 1_024
_MAX_LICENSE_CHARACTERS = 128
_MAX_LABEL_CHARACTERS = 160
_MAX_COUNTER = (2**63) - 1
_MAX_LAST_MODIFIED_CHARACTERS = 64
_MAX_ERROR_DETAILS = 20
_MAX_ERROR_DETAIL_CHARACTERS = 552
_REPOSITORY_COMPONENT_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SHARD_RE = re.compile(
    r"^(?P<stem>.+)-(?P<index>[0-9]{5})-of-(?P<count>[0-9]{5})\.gguf$"
)
_ERROR_CODES = frozenset(
    {
        "access_forbidden",
        "authentication_required",
        "invalid_query",
        "invalid_repository",
        "invalid_response",
        "invalid_token",
        "network_error",
        "no_eligible_gguf",
        "rate_limited",
        "remote_error",
        "repository_not_found",
        "response_too_large",
    }
)


@dataclass(frozen=True)
class RemoteModelSummary:
    """Safe metadata used to present a remote repository search result."""

    repository: str
    private: bool
    gated: Literal["none", "auto", "manual"]
    downloads: int | None = None
    likes: int | None = None
    last_modified: str | None = None


@dataclass(frozen=True)
class RemoteGGUFFile:
    """One LFS-backed GGUF payload eligible for managed acquisition."""

    upstream_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class RemoteGGUFCandidate:
    """One independently selectable GGUF file or complete shard set."""

    label: str
    files: tuple[RemoteGGUFFile, ...]
    total_bytes: int


@dataclass(frozen=True)
class ResolvedRemoteModel:
    """Pinned repository metadata and its bounded GGUF choices."""

    repository: str
    commit: str
    license_id: str
    review_url: str
    candidates: tuple[RemoteGGUFCandidate, ...]
    total_candidate_count: int
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedRemoteCatalog:
    """The one-item artifact catalog generated from a remote selection."""

    artifact: ArtifactDescriptor
    sources: Mapping[ArtifactRef, Mapping[str, str]]

    def descriptor(self, ref: ArtifactRef) -> ArtifactDescriptor:
        """Return the selected artifact when its exact reference is requested."""
        if ref != self.artifact.reference:
            raise KeyError(ref)
        return self.artifact


class RemoteDiscoveryError(RuntimeError):
    """A stable, display-safe failure from remote metadata discovery."""

    def __init__(
        self,
        code: str,
        *,
        retryable: bool = False,
        details: tuple[str, ...] = (),
    ) -> None:
        if code not in _ERROR_CODES:
            raise ValueError("unsupported remote discovery error code")
        _validate_error_details(details)
        super().__init__(code)
        self.code = code
        self.retryable = retryable
        self.details = details


class HuggingFaceRemoteAdapter:
    """Perform narrow, non-redirecting Hugging Face metadata searches."""

    def __init__(
        self,
        *,
        client_factory: Callable[[], httpx.AsyncClient] = httpx.AsyncClient,
    ) -> None:
        self._client_factory = client_factory

    async def search(
        self, query: str, *, token: str | None = None
    ) -> tuple[RemoteModelSummary, ...]:
        """Search the fixed Hugging Face models endpoint.

        Args:
            query: Free-text search input, limited after whitespace trimming.
            token: Optional Hugging Face token for this fixed-origin request.

        Returns:
            At most fifty validated remote model summaries.

        Raises:
            RemoteDiscoveryError: Input is invalid or remote metadata is unusable.
        """
        normalized_query = _validated_query(query)
        headers = _authorization_header(token)
        network_error: RemoteDiscoveryError | None = None
        try:
            async with self._client_factory() as client:
                async with client.stream(
                    "GET",
                    _API_MODELS_URL,
                    params={"search": normalized_query, "limit": str(_MAX_SEARCH_RESULTS)},
                    headers=headers,
                    follow_redirects=False,
                ) as response:
                    _raise_for_status(response.status_code)
                    payload = await _read_bounded_json(response)
        except RemoteDiscoveryError:
            raise
        except httpx.TimeoutException:
            network_error = RemoteDiscoveryError("network_error", retryable=True)
        except httpx.HTTPError:
            network_error = RemoteDiscoveryError("network_error", retryable=True)

        if network_error is not None:
            raise network_error

        return _parse_search_results(payload)

    async def resolve(
        self, repository: str, *, token: str | None = None
    ) -> ResolvedRemoteModel:
        """Resolve one exact repository into pinned GGUF candidates.

        Args:
            repository: Exact owner/repository identifier to resolve.
            token: Optional Hugging Face token for this fixed-origin request.

        Returns:
            Immutable repository metadata and at most one hundred candidates.

        Raises:
            RemoteDiscoveryError: The identifier or response is not usable.
        """
        if not is_exact_repository(repository):
            raise RemoteDiscoveryError("invalid_repository")
        headers = _authorization_header(token)
        network_error: RemoteDiscoveryError | None = None
        try:
            async with self._client_factory() as client:
                async with client.stream(
                    "GET",
                    f"{_API_MODELS_URL}/{repository}",
                    params={"blobs": "true"},
                    headers=headers,
                    follow_redirects=False,
                ) as response:
                    _raise_for_status(response.status_code)
                    payload = await _read_bounded_json(response)
        except RemoteDiscoveryError:
            raise
        except httpx.TimeoutException:
            network_error = RemoteDiscoveryError("network_error", retryable=True)
        except httpx.HTTPError:
            network_error = RemoteDiscoveryError("network_error", retryable=True)

        if network_error is not None:
            raise network_error
        return _parse_resolved_model(repository, payload)


def is_exact_repository(value: str) -> bool:
    """Return whether ``value`` is one bounded portable owner/repository pair."""
    if not isinstance(value, str) or len(value) > _MAX_REPOSITORY_CHARACTERS:
        return False
    parts = value.split("/")
    return len(parts) == 2 and all(
        _REPOSITORY_COMPONENT_RE.fullmatch(part)
        and "--" not in part
        and ".." not in part
        for part in parts
    )


def _validated_query(query: str) -> str:
    if not isinstance(query, str):
        raise RemoteDiscoveryError("invalid_query")
    normalized = query.strip()
    if not normalized or len(normalized) > _MAX_QUERY_CHARACTERS:
        raise RemoteDiscoveryError("invalid_query")
    return normalized


def _authorization_header(token: str | None) -> dict[str, str]:
    if token is None:
        return {}
    if not isinstance(token, str) or not token or "\r" in token or "\n" in token:
        raise RemoteDiscoveryError("invalid_token")
    return {"Authorization": f"Bearer {token}"}


def _raise_for_status(status_code: int) -> None:
    if 300 <= status_code < 400:
        raise RemoteDiscoveryError("remote_error")
    if status_code == 401:
        raise RemoteDiscoveryError("authentication_required")
    if status_code == 403:
        raise RemoteDiscoveryError("access_forbidden")
    if status_code == 404:
        raise RemoteDiscoveryError("repository_not_found")
    if status_code == 429:
        raise RemoteDiscoveryError("rate_limited", retryable=True)
    if 400 <= status_code < 500:
        raise RemoteDiscoveryError("remote_error")
    if status_code >= 500:
        raise RemoteDiscoveryError("remote_error", retryable=True)


async def _read_bounded_json(response: httpx.Response) -> object:
    """Read one decoded metadata JSON body without exceeding its byte budget."""
    chunks: list[bytes] = []
    size = 0
    async for chunk in response.aiter_bytes():
        size += len(chunk)
        if size > _MAX_METADATA_BYTES:
            raise RemoteDiscoveryError("response_too_large")
        chunks.append(chunk)
    try:
        return json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        parse_error = RemoteDiscoveryError("invalid_response")
    raise parse_error


def _validate_error_details(details: tuple[str, ...]) -> None:
    if type(details) is not tuple or len(details) > _MAX_ERROR_DETAILS:
        raise ValueError("invalid remote discovery error details")
    if any(
        not isinstance(detail, str)
        or len(detail) > _MAX_ERROR_DETAIL_CHARACTERS
        or not detail.isprintable()
        for detail in details
    ):
        raise ValueError("invalid remote discovery error details")


def _parse_search_results(payload: object) -> tuple[RemoteModelSummary, ...]:
    if not isinstance(payload, list) or len(payload) > _MAX_SEARCH_RESULTS:
        raise RemoteDiscoveryError("invalid_response")
    return tuple(_parse_search_result(item) for item in payload)


def _parse_search_result(item: object) -> RemoteModelSummary:
    if not isinstance(item, dict):
        raise RemoteDiscoveryError("invalid_response")
    repository = item.get("modelId")
    private = item.get("private")
    gated = item.get("gated")
    if not is_exact_repository(repository) or type(private) is not bool:
        raise RemoteDiscoveryError("invalid_response")
    normalized_gated = _normalized_gated(gated)
    if normalized_gated is None:
        raise RemoteDiscoveryError("invalid_response")
    return RemoteModelSummary(
        repository=repository,
        private=private,
        gated=normalized_gated,
        downloads=_optional_counter(item.get("downloads")),
        likes=_optional_counter(item.get("likes")),
        last_modified=_optional_last_modified(item.get("lastModified")),
    )


def _normalized_gated(value: object) -> Literal["none", "auto", "manual"] | None:
    if value is False:
        return "none"
    if value == "auto":
        return "auto"
    if value == "manual":
        return "manual"
    return None


def _optional_counter(value: object) -> int | None:
    if type(value) is int and 0 <= value <= _MAX_COUNTER:
        return value
    return None


def _optional_last_modified(value: object) -> str | None:
    if isinstance(value, str) and len(value) <= _MAX_LAST_MODIFIED_CHARACTERS:
        return value
    return None


def _parse_resolved_model(repository: str, payload: object) -> ResolvedRemoteModel:
    if not isinstance(payload, dict):
        raise RemoteDiscoveryError("invalid_response")
    commit = payload.get("sha")
    siblings = payload.get("siblings")
    if (
        not isinstance(commit, str)
        or _COMMIT_RE.fullmatch(commit) is None
        or not isinstance(siblings, list)
        or len(siblings) > _MAX_FILE_ENTRIES
    ):
        raise RemoteDiscoveryError("invalid_response")

    candidates, warnings = _gguf_candidates(repository, siblings)
    if not candidates:
        raise RemoteDiscoveryError("no_eligible_gguf", details=warnings)
    total_candidate_count = len(candidates)
    displayed_candidates = tuple(candidates[:_MAX_GGUF_CANDIDATES])
    review_url = f"https://huggingface.co/{repository}/tree/{commit}"
    return ResolvedRemoteModel(
        repository=repository,
        commit=commit,
        license_id=_license_id(payload.get("cardData")),
        review_url=review_url,
        candidates=displayed_candidates,
        total_candidate_count=total_candidate_count,
        warnings=warnings,
    )


def _license_id(card_data: object) -> str:
    if not isinstance(card_data, dict):
        return "NOASSERTION"
    license_id = card_data.get("license")
    if (
        not isinstance(license_id, str)
        or not license_id
        or len(license_id) > _MAX_LICENSE_CHARACTERS
        or license_id != license_id.strip()
        or not license_id.isprintable()
    ):
        return "NOASSERTION"
    return license_id


def _gguf_candidates(
    repository: str, siblings: list[object]
) -> tuple[list[RemoteGGUFCandidate], tuple[str, ...]]:
    singles: list[RemoteGGUFFile] = []
    shard_groups: dict[tuple[str, str, int], list[RemoteGGUFFile | None]] = {}
    invalid_groups: set[tuple[str, str, int]] = set()

    for sibling in siblings:
        path, remote_file = _parse_sibling(sibling)
        if path is None:
            continue
        shard = _shard_identity(path)
        if shard is None:
            if remote_file is not None:
                singles.append(remote_file)
            continue
        directory, stem, index, count = shard
        group_key = (directory, stem, count)
        shard_groups.setdefault(group_key, []).append(remote_file)
        if not (1 <= index <= count <= _MAX_SHARDS):
            invalid_groups.add(group_key)

    grouped: list[RemoteGGUFCandidate] = []
    warnings: list[str] = []
    for group_key in sorted(shard_groups):
        directory, stem, count = group_key
        if group_key in invalid_groups or not 1 <= count <= _MAX_SHARDS:
            continue
        members = shard_groups[group_key]
        valid_members = [member for member in members if member is not None]
        by_index: dict[int, RemoteGGUFFile] = {}
        duplicate = False
        for member in valid_members:
            assert member is not None
            match = _SHARD_RE.fullmatch(member.upstream_path.rsplit("/", 1)[-1])
            assert match is not None
            index = int(match.group("index"))
            if index in by_index:
                duplicate = True
            by_index[index] = member
        missing = tuple(index for index in range(1, count + 1) if index not in by_index)
        if missing and len(warnings) < _MAX_ERROR_DETAILS:
            warnings.append(_incomplete_warning(repository, directory, stem, missing))
        if (
            len(valid_members) != len(members)
            or duplicate
            or missing
            or len(by_index) != count
        ):
            continue
        files = tuple(by_index[index] for index in range(1, count + 1))
        grouped.append(
            RemoteGGUFCandidate(
                label=_candidate_label(repository, _group_display_path(directory, stem)),
                files=files,
                total_bytes=sum(item.size_bytes for item in files),
            )
        )

    candidates = [
        *(
            RemoteGGUFCandidate(
                label=_candidate_label(repository, item.upstream_path),
                files=(item,),
                total_bytes=item.size_bytes,
            )
            for item in singles
        ),
        *grouped,
    ]
    candidates.sort(key=lambda candidate: tuple(item.upstream_path for item in candidate.files))
    return candidates, tuple(warnings[:_MAX_ERROR_DETAILS])


def _parse_sibling(sibling: object) -> tuple[str | None, RemoteGGUFFile | None]:
    if not isinstance(sibling, dict):
        raise RemoteDiscoveryError("invalid_response")
    path = sibling.get("rfilename")
    if not isinstance(path, str):
        raise RemoteDiscoveryError("invalid_response")
    if not path.endswith(".gguf"):
        return None, None
    if not _is_valid_upstream_path(path):
        raise RemoteDiscoveryError("invalid_response")
    lfs = sibling.get("lfs")
    if not isinstance(lfs, dict):
        return path, None
    size_bytes = lfs.get("size")
    sha256 = lfs.get("sha256")
    if (
        type(size_bytes) is not int
        or size_bytes < 0
        or not isinstance(sha256, str)
        or _SHA256_RE.fullmatch(sha256) is None
    ):
        return path, None
    return path, RemoteGGUFFile(path, size_bytes, sha256)


def _shard_identity(path: str) -> tuple[str, str, int, int] | None:
    directory, separator, basename = path.rpartition("/")
    match = _SHARD_RE.fullmatch(basename if separator else path)
    if match is None:
        return None
    return (
        directory if separator else "",
        match.group("stem"),
        int(match.group("index")),
        int(match.group("count")),
    )


def _is_valid_upstream_path(path: str) -> bool:
    try:
        encoded_length = len(path.encode("utf-8"))
    except UnicodeEncodeError:
        return False
    return (
        0 < encoded_length <= _MAX_PATH_BYTES
        and path == path.strip()
        and "\\" not in path
        and all(
            component not in {"", ".", ".."}
            and component.isprintable()
            for component in path.split("/")
        )
    )


def _group_display_path(directory: str, stem: str) -> str:
    return f"{directory}/{stem}" if directory else stem


def _candidate_label(repository: str, path: str) -> str:
    return f"{repository} · {path}"[:_MAX_LABEL_CHARACTERS]


def _incomplete_warning(
    repository: str, directory: str, stem: str, missing: tuple[int, ...]
) -> str:
    missing_indexes = " ".join(f"{index:05d}" for index in missing)
    return (
        f"{_candidate_label(repository, _group_display_path(directory, stem))} "
        f"missing {missing_indexes}"
    )


def build_remote_catalog(
    resolved: ResolvedRemoteModel, candidate: RemoteGGUFCandidate
) -> ResolvedRemoteCatalog:
    """Map one resolved candidate into a pinned, inert managed artifact.

    Args:
        resolved: The immutable repository resolution that produced the candidate.
        candidate: A selected file or complete shard set from that resolution.

    Returns:
        A one-item catalog and per-file pinned Hugging Face source map.

    Raises:
        ValueError: The supplied immutable values cannot form a safe artifact.
    """
    if type(resolved) is not ResolvedRemoteModel or type(candidate) is not RemoteGGUFCandidate:
        raise ValueError("resolved and candidate must be remote discovery values")
    if not is_exact_repository(resolved.repository) or _COMMIT_RE.fullmatch(resolved.commit) is None:
        raise ValueError("resolved repository identity is invalid")
    files = tuple(sorted(candidate.files, key=lambda item: item.upstream_path))
    if not files or any(type(item) is not RemoteGGUFFile for item in files):
        raise ValueError("candidate files are invalid")
    if (
        candidate.total_bytes != sum(item.size_bytes for item in files)
        or any(
            not _is_valid_upstream_path(item.upstream_path)
            or type(item.size_bytes) is not int
            or item.size_bytes < 0
            or _SHA256_RE.fullmatch(item.sha256) is None
            for item in files
        )
    ):
        raise ValueError("candidate metadata is invalid")

    artifact_paths = _managed_paths(files)
    canonical_identity = json.dumps(
        {"repository": resolved.repository, "paths": [item.upstream_path for item in files]},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    reference = ArtifactRef(
        artifact_id=f"hf-gguf-{hashlib.sha256(canonical_identity).hexdigest()}",
        revision=resolved.commit,
        variant="not-declared",
    )
    artifact_files = tuple(
        ArtifactFile(path, remote_file.size_bytes, remote_file.sha256)
        for path, remote_file in zip(artifact_paths, files, strict=True)
    )
    source_items = tuple(
        (path, _pinned_payload_url(resolved, remote_file.upstream_path))
        for path, remote_file in zip(artifact_paths, files, strict=True)
    )
    sources = MappingProxyType({reference: MappingProxyType(dict(source_items))})
    artifact = ArtifactDescriptor(
        reference=reference,
        model_id=_bounded_candidate_label(candidate.label),
        role=ArtifactRole.ROOT,
        format=ArtifactFormat.GGUF,
        consumer="unassigned",
        model_family="unassigned",
        upstream_repository=resolved.repository,
        upstream_revision=resolved.commit,
        source_url=source_items[0][1],
        precision="not-declared",
        expected_installed_bytes=candidate.total_bytes,
        license_id=resolved.license_id,
        license_url=resolved.review_url,
        usage_notice="Runtime compatibility has not been verified. Configuration is required.",
        runtime_name="unassigned",
        runtime_version_constraint="none",
        supported_os=("unassigned",),
        supported_architectures=("unassigned",),
        provenance=(ProvenanceClass.LOCAL_INTEGRITY_RECORDED,),
        files=artifact_files,
    )
    return ResolvedRemoteCatalog(artifact=artifact, sources=sources)


def _bounded_candidate_label(label: str) -> str:
    if not isinstance(label, str) or not label or not label.isprintable():
        raise ValueError("candidate label is invalid")
    return label[:_MAX_LABEL_CHARACTERS]


def _managed_paths(files: tuple[RemoteGGUFFile, ...]) -> tuple[str, ...]:
    if len(files) == 1:
        return ("model.gguf",)
    paths: list[str] = []
    for remote_file in files:
        match = _SHARD_RE.fullmatch(remote_file.upstream_path.rsplit("/", 1)[-1])
        if match is None:
            raise ValueError("multi-file candidates must be standard shard sets")
        paths.append(
            "model-"
            f"{int(match.group('index')):05d}-of-{int(match.group('count')):05d}.gguf"
        )
    if len(set(paths)) != len(paths):
        raise ValueError("candidate shard paths are invalid")
    return tuple(paths)


def _pinned_payload_url(resolved: ResolvedRemoteModel, path: str) -> str:
    encoded_path = "/".join(quote(component, safe="-._~") for component in path.split("/"))
    return (
        f"https://huggingface.co/{resolved.repository}/resolve/{resolved.commit}/"
        f"{encoded_path}"
    )
