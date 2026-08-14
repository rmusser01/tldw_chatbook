#!/usr/bin/env python3
"""Refresh reviewed audio.cpp source facts at one explicit immutable commit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, BinaryIO, Callable
from urllib.parse import quote, urlsplit
import urllib.request

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_CATALOG_ROOT = _REPOSITORY_ROOT / "tldw_chatbook" / "TTS"
if str(_CATALOG_ROOT) not in sys.path:
    sys.path.insert(0, str(_CATALOG_ROOT))

from audio_cpp_artifact_catalog import (  # noqa: E402
    AudioCppArtifactPackage,
    AudioCppArtifactSourceFile,
    AudioCppArtifactSourceManifest,
    load_audio_cpp_artifact_source_manifest,
)


_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_OID_RE = re.compile(r"[0-9a-f]{40}\Z")
_MAX_TREE_BYTES = 8 * 1024 * 1024
_MAX_TREE_TOTAL_BYTES = 32 * 1024 * 1024
_MAX_TREE_PAGES = 32
_MAX_GIT_FILE_BYTES = 16 * 1024 * 1024
_CHUNK_BYTES = 1024 * 1024
_LINK_RE = re.compile(
    r'\s*<([^<>]+)>\s*;\s*rel=(?:"([^"]+)"|([A-Za-z][A-Za-z0-9_-]*))\s*\Z'
)
_DEFAULT_MANIFEST = (
    _REPOSITORY_ROOT / "tldw_chatbook" / "TTS" / "audio_cpp_artifact_manifest.json"
)
UrlOpen = Callable[[urllib.request.Request], BinaryIO]


def validate_commit(commit: str) -> str:
    """Require an explicit immutable Hugging Face commit.

    Args:
        commit: Candidate Hugging Face Git revision.

    Returns:
        The validated lowercase 40-character commit.

    Raises:
        ValueError: The revision is not an exact lowercase Git commit.
    """

    if _COMMIT_RE.fullmatch(commit) is None:
        raise ValueError("commit must be exactly 40 lowercase hexadecimal characters")
    return commit


def _read_bounded(response: BinaryIO, limit: int, label: str) -> bytes:
    headers = getattr(response, "headers", {})
    declared = headers.get("Content-Length") if headers is not None else None
    if declared is not None:
        if (
            type(declared) is not str
            or not declared.isascii()
            or not declared.isdigit()
        ):
            raise ValueError(f"{label} has an invalid Content-Length")
        try:
            declared_size = int(declared)
        except ValueError as exc:
            raise ValueError(f"{label} has an invalid Content-Length") from exc
        if declared_size > limit:
            raise ValueError(f"{label} exceeds the {limit}-byte limit")

    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = response.read(min(_CHUNK_BYTES, limit + 1 - total))
        if not chunk:
            return b"".join(chunks)
        total += len(chunk)
        if total > limit:
            raise ValueError(f"{label} exceeds the {limit}-byte limit")
        chunks.append(chunk)


def _request_json(
    url: str,
    urlopen: UrlOpen,
    limit: int,
    label: str,
) -> tuple[object, str | None, int]:
    request = urllib.request.Request(
        url, headers={"User-Agent": "tldw-chatbook-maintainer"}
    )
    with urlopen(request) as response:
        content = _read_bounded(response, limit, label)
        headers = getattr(response, "headers", {})
        link = headers.get("Link") if headers is not None else None
    try:
        return json.loads(content), link, len(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("repository tree is not valid JSON") from exc


def _next_page_url(
    link_header: object,
    repository: str,
    commit: str,
) -> str | None:
    if link_header is None:
        return None
    if type(link_header) is not str or not link_header:
        raise ValueError("pagination Link header is malformed")

    next_urls: list[str] = []
    for part in link_header.split(","):
        match = _LINK_RE.fullmatch(part)
        if match is None:
            raise ValueError("pagination Link header is malformed")
        relations = (match.group(2) or match.group(3)).split()
        if "next" in relations:
            next_urls.append(match.group(1))
    if not next_urls:
        return None
    if len(next_urls) != 1:
        raise ValueError("pagination Link header has multiple next links")

    next_url = next_urls[0]
    if any(character.isspace() for character in next_url):
        raise ValueError("pagination next link is unsafe")
    try:
        parsed = urlsplit(next_url)
    except ValueError as exc:
        raise ValueError("pagination next link is malformed") from exc
    expected_path = f"/api/models/{quote(repository, safe='/')}/tree/{commit}"
    if (
        parsed.scheme != "https"
        or parsed.netloc != "huggingface.co"
        or parsed.path != expected_path
        or not parsed.query
        or parsed.fragment
    ):
        raise ValueError("pagination next link changed origin, repository, or commit")
    return next_url


def _fetch_tree(
    repository: str,
    commit: str,
    urlopen: UrlOpen,
) -> dict[str, dict[str, Any]]:
    repository_path = quote(repository, safe="/")
    url = (
        f"https://huggingface.co/api/models/{repository_path}/tree/{commit}"
        "?recursive=true&expand=true"
    )
    entries: dict[str, dict[str, Any]] = {}
    seen_urls: set[str] = set()
    total_bytes = 0
    for page_index in range(_MAX_TREE_PAGES):
        if url in seen_urls:
            raise ValueError("repository tree pagination cycle")
        seen_urls.add(url)
        remaining = _MAX_TREE_TOTAL_BYTES - total_bytes
        if remaining <= 0:
            raise ValueError("repository tree aggregate byte limit exceeded")
        page_limit = min(_MAX_TREE_BYTES, remaining)
        label = (
            "repository tree aggregate byte limit"
            if remaining < _MAX_TREE_BYTES
            else "repository tree page"
        )
        raw, link, page_bytes = _request_json(url, urlopen, page_limit, label)
        total_bytes += page_bytes
        if type(raw) is not list:
            raise ValueError("repository tree must be an array")
        for item_index, value in enumerate(raw):
            if type(value) is not dict:
                raise ValueError(
                    f"repository tree entry {page_index}:{item_index} has unknown file shape"
                )
            path = value.get("path")
            if type(path) is not str or not path:
                raise ValueError(
                    f"repository tree entry {page_index}:{item_index} has unknown file shape"
                )
            if path in entries:
                raise ValueError(f"repository tree contains duplicate path {path!r}")
            entries[path] = value
        next_url = _next_page_url(link, repository, commit)
        if next_url is None:
            return entries
        if next_url in seen_urls:
            raise ValueError("repository tree pagination cycle")
        url = next_url
    raise ValueError("repository tree pagination page limit exceeded")


def _git_file_facts(
    repository: str,
    commit: str,
    path: str,
    entry: dict[str, Any],
    urlopen: UrlOpen,
) -> tuple[int, str]:
    size = entry.get("size")
    oid = entry.get("oid")
    if (
        type(size) is not int
        or size <= 0
        or type(oid) is not str
        or _GIT_OID_RE.fullmatch(oid) is None
        or size > _MAX_GIT_FILE_BYTES
    ):
        raise ValueError(f"unknown file shape for Git-managed file {path!r}")

    repository_path = quote(repository, safe="/")
    source_path = quote(path, safe="/")
    url = f"https://huggingface.co/{repository_path}/resolve/{commit}/{source_path}"
    request = urllib.request.Request(
        url, headers={"User-Agent": "tldw-chatbook-maintainer"}
    )
    digest = hashlib.sha256()
    received = 0
    with urlopen(request) as response:
        headers = getattr(response, "headers", {})
        declared = headers.get("Content-Length") if headers is not None else None
        if declared is not None and (not declared.isdigit() or int(declared) != size):
            raise ValueError(f"size mismatch for Git-managed file {path!r}")
        while True:
            chunk = response.read(_CHUNK_BYTES)
            if not chunk:
                break
            received += len(chunk)
            if received > size or received > _MAX_GIT_FILE_BYTES:
                raise ValueError(f"size mismatch for Git-managed file {path!r}")
            digest.update(chunk)
    if received != size:
        raise ValueError(f"size mismatch for Git-managed file {path!r}")
    return size, digest.hexdigest()


def _file_facts(
    repository: str,
    commit: str,
    path: str,
    entry: dict[str, Any],
    urlopen: UrlOpen,
) -> tuple[int, str]:
    if entry.get("type") != "file":
        raise ValueError(f"unknown file shape for requested path {path!r}")
    lfs = entry.get("lfs")
    if lfs is None:
        return _git_file_facts(repository, commit, path, entry, urlopen)
    if type(lfs) is not dict:
        raise ValueError(f"unknown file shape for LFS file {path!r}")
    size = lfs.get("size")
    oid = lfs.get("oid")
    if (
        type(size) is not int
        or size <= 0
        or type(oid) is not str
        or _SHA256_RE.fullmatch(oid) is None
        or type(entry.get("size")) is not int
        or entry.get("size") != size
    ):
        raise ValueError(f"unknown file shape for LFS file {path!r}")
    return size, oid


def _manifest_dict(manifest: AudioCppArtifactSourceManifest) -> dict[str, object]:
    return {
        "repository": manifest.repository,
        "commit": manifest.commit,
        "packages": [
            {
                "recipe_id": package.recipe_id,
                "recipe_revision": package.recipe_revision,
                "package_variant": package.package_variant,
                "artifact_id": package.artifact_id,
                "license_id": package.license_id,
                "license_url": package.license_url,
                "usage_notice": package.usage_notice,
                "files": [
                    {
                        "source_path": file.source_path,
                        "managed_path": file.managed_path,
                        "size_bytes": file.size_bytes,
                        "sha256": file.sha256,
                    }
                    for file in package.files
                ],
            }
            for package in manifest.packages
        ],
    }


def refresh_manifest_bytes(
    manifest_path: Path,
    commit: str,
    *,
    urlopen: UrlOpen = urllib.request.urlopen,
) -> bytes:
    """Refresh integrity facts while retaining reviewed mappings and licenses.

    Args:
        manifest_path: Existing reviewed manifest to refresh.
        commit: Exact immutable Hugging Face commit to query.
        urlopen: Bounded HTTP opener used for source evidence.

    Returns:
        Deterministic UTF-8 JSON bytes for the refreshed manifest.

    Raises:
        OSError: The manifest or remote source evidence cannot be read.
        TypeError: Manifest or source facts have an invalid type.
        ValueError: Manifest or source facts violate the bounded contract.
    """

    commit = validate_commit(commit)
    current = load_audio_cpp_artifact_source_manifest(
        manifest_path,
        expected_commit=None,
    )
    if not current.packages:
        refreshed = AudioCppArtifactSourceManifest(current.repository, commit, ())
    else:
        tree = _fetch_tree(current.repository, commit, urlopen)
        packages: list[AudioCppArtifactPackage] = []
        for package in current.packages:
            files: list[AudioCppArtifactSourceFile] = []
            for file in package.files:
                entry = tree.get(file.source_path)
                if entry is None:
                    raise ValueError(f"source path is absent: {file.source_path!r}")
                size, sha256 = _file_facts(
                    current.repository,
                    commit,
                    file.source_path,
                    entry,
                    urlopen,
                )
                files.append(
                    AudioCppArtifactSourceFile(
                        source_path=file.source_path,
                        managed_path=file.managed_path,
                        size_bytes=size,
                        sha256=sha256,
                    )
                )
            packages.append(
                AudioCppArtifactPackage(
                    recipe_id=package.recipe_id,
                    recipe_revision=package.recipe_revision,
                    package_variant=package.package_variant,
                    artifact_id=package.artifact_id,
                    license_id=package.license_id,
                    license_url=package.license_url,
                    usage_notice=package.usage_notice,
                    files=tuple(
                        sorted(
                            files,
                            key=lambda item: (item.managed_path, item.source_path),
                        )
                    ),
                )
            )
        refreshed = AudioCppArtifactSourceManifest(
            current.repository,
            commit,
            tuple(sorted(packages, key=lambda item: item.key)),
        )
    return (
        json.dumps(_manifest_dict(refreshed), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run the explicit-revision maintainer refresh command.

    ``--manifest`` and ``--output`` are explicit trusted-maintainer paths.
    The output path intentionally permits an arbitrary destination; this
    dependency-free command does not impose an application confinement root.

    Args:
        argv: Optional command arguments; defaults to ``sys.argv``.

    Returns:
        Zero after writing the refreshed manifest successfully.

    Raises:
        OSError: Standard output or the explicit output destination cannot be
            written.
        SystemExit: Argument parsing or manifest refresh fails.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        output = refresh_manifest_bytes(args.manifest, args.commit)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    if args.output is None:
        sys.stdout.buffer.write(output)
    else:
        args.output.write_bytes(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
