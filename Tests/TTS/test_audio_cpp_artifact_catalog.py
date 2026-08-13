"""Boundaries for the pinned audio.cpp artifact-source manifest."""

from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
import urllib.request

import pytest


REPOSITORY = "audio-cpp/audio.cpp-gguf"
COMMIT = "597048d9a920592808d7d4e2acd7b9c4596a143a"


def _valid_manifest() -> dict[str, Any]:
    return {
        "repository": REPOSITORY,
        "commit": COMMIT,
        "packages": [
            {
                "recipe_id": "audio-cpp-0.5.1.example",
                "recipe_revision": 1,
                "package_variant": "example_q8_0",
                "artifact_id": "audio-cpp-example-q8-0",
                "license_id": "Apache-2.0",
                "license_url": "https://example.invalid/LICENSE",
                "usage_notice": "Review the upstream model license before use.",
                "files": [
                    {
                        "source_path": "example/model.gguf",
                        "managed_path": "model.gguf",
                        "size_bytes": 10,
                        "sha256": "a" * 64,
                    }
                ],
            }
        ],
    }


def _write_manifest(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_checked_in_manifest_is_exact_pinned_empty_header_and_network_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import (
        load_audio_cpp_artifact_source_manifest,
    )

    def fail_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("runtime manifest loading touched the network")

    monkeypatch.setattr(urllib.request, "urlopen", fail_network)
    catalog = load_audio_cpp_artifact_source_manifest()

    assert catalog.repository == REPOSITORY
    assert catalog.commit == COMMIT
    assert catalog.packages == ()
    manifest_path = (
        Path(__file__).parents[2]
        / "tldw_chatbook"
        / "TTS"
        / "audio_cpp_artifact_manifest.json"
    )
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == {
        "repository": REPOSITORY,
        "commit": COMMIT,
        "packages": [],
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repository", "audio-cpp/not-the-official-repository"),
        ("commit", "main"),
        ("commit", COMMIT.upper()),
        ("commit", "0" * 40),
    ],
)
def test_manifest_rejects_any_non_pinned_repository_or_commit(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import (
        load_audio_cpp_artifact_source_manifest,
    )

    payload = _valid_manifest()
    payload[field] = value

    with pytest.raises(ValueError, match=field):
        load_audio_cpp_artifact_source_manifest(_write_manifest(tmp_path, payload))


def test_manifest_rejects_duplicate_package_keys(tmp_path: Path) -> None:
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import (
        load_audio_cpp_artifact_source_manifest,
    )

    payload = _valid_manifest()
    duplicate = copy.deepcopy(payload["packages"][0])
    duplicate["artifact_id"] = "different-artifact-id"
    payload["packages"].append(duplicate)

    with pytest.raises(ValueError, match="duplicate package key"):
        load_audio_cpp_artifact_source_manifest(_write_manifest(tmp_path, payload))


@pytest.mark.parametrize("path_field", ["source_path", "managed_path"])
def test_manifest_rejects_duplicate_paths_within_a_package(
    tmp_path: Path,
    path_field: str,
) -> None:
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import (
        load_audio_cpp_artifact_source_manifest,
    )

    payload = _valid_manifest()
    duplicate = copy.deepcopy(payload["packages"][0]["files"][0])
    other_field = "managed_path" if path_field == "source_path" else "source_path"
    duplicate[other_field] = f"different/{duplicate[other_field]}"
    payload["packages"][0]["files"].append(duplicate)

    with pytest.raises(ValueError, match=f"duplicate {path_field}"):
        load_audio_cpp_artifact_source_manifest(_write_manifest(tmp_path, payload))


@pytest.mark.parametrize("path_field", ["source_path", "managed_path"])
@pytest.mark.parametrize("value", ["../model.gguf", "models/../model.gguf", "/model.gguf"])
def test_manifest_rejects_path_traversal_or_absolute_paths(
    tmp_path: Path,
    path_field: str,
    value: str,
) -> None:
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import (
        load_audio_cpp_artifact_source_manifest,
    )

    payload = _valid_manifest()
    payload["packages"][0]["files"][0][path_field] = value

    with pytest.raises(ValueError, match=path_field):
        load_audio_cpp_artifact_source_manifest(_write_manifest(tmp_path, payload))


@pytest.mark.parametrize(
    ("scope", "field"),
    [
        ("file", "size_bytes"),
        ("file", "sha256"),
        ("package", "license_id"),
        ("package", "license_url"),
        ("package", "usage_notice"),
    ],
)
def test_manifest_rejects_missing_integrity_or_reviewed_license_facts(
    tmp_path: Path,
    scope: str,
    field: str,
) -> None:
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import (
        load_audio_cpp_artifact_source_manifest,
    )

    payload = _valid_manifest()
    owner = (
        payload["packages"][0]["files"][0]
        if scope == "file"
        else payload["packages"][0]
    )
    del owner[field]

    with pytest.raises(ValueError, match=field):
        load_audio_cpp_artifact_source_manifest(_write_manifest(tmp_path, payload))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("recipe_revision", True),
        ("recipe_revision", 1.0),
        ("files", ()),
    ],
)
def test_manifest_uses_exact_json_types(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import (
        load_audio_cpp_artifact_source_manifest,
    )

    payload = _valid_manifest()
    payload["packages"][0][field] = value

    with pytest.raises((TypeError, ValueError), match=field):
        load_audio_cpp_artifact_source_manifest(_write_manifest(tmp_path, payload))


class _Response(io.BytesIO):
    def __init__(self, content: bytes) -> None:
        super().__init__(content)
        self.headers = {"Content-Length": str(len(content))}

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def test_refresh_is_immutable_bounded_and_byte_deterministic(tmp_path: Path) -> None:
    from scripts.refresh_audio_cpp_artifact_manifest import refresh_manifest_bytes

    git_content = b'{"x":1}\n'
    payload = _valid_manifest()
    payload["packages"][0]["files"] = [
        {
            "source_path": "example/model.gguf",
            "managed_path": "model.gguf",
            "size_bytes": 1,
            "sha256": "1" * 64,
        },
        {
            "source_path": "example/config.json",
            "managed_path": "config.json",
            "size_bytes": 1,
            "sha256": "2" * 64,
        },
    ]
    manifest_path = _write_manifest(tmp_path, payload)
    tree = json.dumps(
        [
            {
                "type": "file",
                "path": "example/config.json",
                "oid": "b" * 40,
                "size": len(git_content),
            },
            {
                "type": "file",
                "path": "example/model.gguf",
                "oid": "c" * 40,
                "size": 10,
                "lfs": {"oid": "d" * 64, "size": 10, "pointerSize": 127},
            },
        ]
    ).encode()
    opened_urls: list[str] = []

    def recorded_urlopen(request: object) -> _Response:
        url = getattr(request, "full_url", str(request))
        opened_urls.append(url)
        if "/api/models/" in url:
            return _Response(tree)
        if url.endswith("/resolve/" + COMMIT + "/example/config.json"):
            return _Response(git_content)
        raise AssertionError(f"unexpected URL: {url}")

    first = refresh_manifest_bytes(manifest_path, COMMIT, urlopen=recorded_urlopen)
    second = refresh_manifest_bytes(manifest_path, COMMIT, urlopen=recorded_urlopen)

    assert first == second
    assert b'"sha256": "' + hashlib.sha256(git_content).hexdigest().encode() in first
    assert b'"sha256": "' + b"d" * 64 in first
    assert json.loads(first)["packages"][0]["license_id"] == "Apache-2.0"
    assert opened_urls
    assert all(COMMIT in url for url in opened_urls)
    assert all("/main/" not in url and not url.endswith("/main") for url in opened_urls)


@pytest.mark.parametrize("commit", ["main", "A" * 40, "0" * 39, "0" * 41])
def test_refresh_refuses_non_exact_immutable_commit(commit: str) -> None:
    from scripts.refresh_audio_cpp_artifact_manifest import validate_commit

    with pytest.raises(ValueError, match="40 lowercase hexadecimal"):
        validate_commit(commit)


def test_refresh_refuses_unknown_requested_file_shapes(tmp_path: Path) -> None:
    from scripts.refresh_audio_cpp_artifact_manifest import refresh_manifest_bytes

    manifest_path = _write_manifest(tmp_path, _valid_manifest())
    tree = json.dumps(
        [{"type": "directory", "path": "example/model.gguf", "oid": "a" * 40}]
    ).encode()

    def recorded_urlopen(_request: object) -> _Response:
        return _Response(tree)

    with pytest.raises(ValueError, match="unknown file shape"):
        refresh_manifest_bytes(manifest_path, COMMIT, urlopen=recorded_urlopen)


def test_refresh_command_runs_directly_without_network_for_empty_manifest() -> None:
    repository_root = Path(__file__).parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "scripts/refresh_audio_cpp_artifact_manifest.py",
            "--commit",
            COMMIT,
            "--manifest",
            "tldw_chatbook/TTS/audio_cpp_artifact_manifest.json",
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "repository": REPOSITORY,
        "commit": COMMIT,
        "packages": [],
    }
