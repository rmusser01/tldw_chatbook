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
from urllib.parse import quote
import urllib.request

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_CATALOG_ROOT = _REPOSITORY_ROOT / "tldw_chatbook" / "TTS"
if str(_CATALOG_ROOT) not in sys.path:
    sys.path.insert(0, str(_CATALOG_ROOT))

from audio_cpp_artifact_catalog import (  # noqa: E402
    AudioCppArtifactPackage,
    AudioCppArtifactSourceFile,
    AudioCppArtifactSourceManifest,
    parse_audio_cpp_artifact_source_manifest,
)


_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_OID_RE = re.compile(r"[0-9a-f]{40}\Z")
_MAX_TREE_BYTES = 8 * 1024 * 1024
_MAX_GIT_FILE_BYTES = 16 * 1024 * 1024
_CHUNK_BYTES = 1024 * 1024
_DEFAULT_MANIFEST = (
    _REPOSITORY_ROOT
    / "tldw_chatbook"
    / "TTS"
    / "audio_cpp_artifact_manifest.json"
)
UrlOpen = Callable[[urllib.request.Request], BinaryIO]


def validate_commit(commit: str) -> str:
    """Require an explicit immutable Hugging Face commit."""

    if _COMMIT_RE.fullmatch(commit) is None:
        raise ValueError("commit must be exactly 40 lowercase hexadecimal characters")
    return commit


def _read_bounded(response: BinaryIO, limit: int, label: str) -> bytes:
    headers = getattr(response, "headers", {})
    declared = headers.get("Content-Length") if headers is not None else None
    if declared is not None:
        try:
            if int(declared) > limit:
                raise ValueError(f"{label} exceeds the {limit}-byte limit")
        except ValueError as exc:
            if "exceeds" in str(exc):
                raise
            raise ValueError(f"{label} has an invalid Content-Length") from exc

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


def _request_json(url: str, urlopen: UrlOpen) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": "tldw-chatbook-maintainer"})
    with urlopen(request) as response:
        content = _read_bounded(response, _MAX_TREE_BYTES, "repository tree")
    try:
        return json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("repository tree is not valid JSON") from exc


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
    raw = _request_json(url, urlopen)
    if type(raw) is not list:
        raise ValueError("repository tree must be an array")

    entries: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(raw):
        if type(value) is not dict:
            raise ValueError(f"repository tree entry {index} has unknown file shape")
        path = value.get("path")
        if type(path) is not str or not path:
            raise ValueError(f"repository tree entry {index} has unknown file shape")
        if path in entries:
            raise ValueError(f"repository tree contains duplicate path {path!r}")
        entries[path] = value
    return entries


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
    request = urllib.request.Request(url, headers={"User-Agent": "tldw-chatbook-maintainer"})
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
    """Refresh integrity facts while retaining reviewed mappings and licenses."""

    commit = validate_commit(commit)
    with manifest_path.open("r", encoding="utf-8") as handle:
        current = parse_audio_cpp_artifact_source_manifest(
            json.load(handle), expected_commit=None
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
                    files=tuple(sorted(files, key=lambda item: (item.managed_path, item.source_path))),
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
    """Run the explicit-revision maintainer refresh command."""

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
