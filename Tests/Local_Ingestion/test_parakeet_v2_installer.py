"""Tests for the curated Parakeet v2 INT8 installer."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

import tldw_chatbook.Local_Ingestion.parakeet_v2_installer as installer


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def _tiny_files(payloads: dict[str, bytes]) -> tuple[installer.BundleFile, ...]:
    return tuple(
        installer.BundleFile(
            filename=filename,
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        for filename, payload in payloads.items()
    )


def _fake_network(monkeypatch, payloads: dict[str, bytes]) -> list[str]:
    requested: list[str] = []

    def open_url(request, *, timeout):
        del timeout
        filename = request.full_url.rsplit("/", 1)[-1]
        requested.append(filename)
        return _Response(payloads[filename])

    monkeypatch.setattr(installer, "_open_url", open_url)
    return requested


def test_curated_descriptor_is_pinned_to_verified_v2_int8() -> None:
    assert installer.PARAKEET_V2_REVISION == (
        "0bbb45a3365852604aef28b538a8f066f4ccaa85"
    )
    assert installer.PARAKEET_V2_LICENSE == "CC-BY-4.0"
    assert installer.PARAKEET_V2_TOTAL_BYTES == 661_191_781
    assert {file.filename for file in installer.PARAKEET_V2_FILES} == {
        "config.json",
        "vocab.txt",
        "encoder-model.int8.onnx",
        "decoder_joint-model.int8.onnx",
    }


def test_install_verifies_and_atomically_publishes_bundle(
    tmp_path: Path, monkeypatch
) -> None:
    payloads = {
        "config.json": b"config",
        "vocab.txt": b"vocab",
        "encoder-model.int8.onnx": b"encoder",
        "decoder_joint-model.int8.onnx": b"decoder",
    }
    monkeypatch.setattr(installer, "PARAKEET_V2_FILES", _tiny_files(payloads))
    requested = _fake_network(monkeypatch, payloads)
    progress: list[tuple[int, int, str]] = []
    destination = tmp_path / "installed"

    result = installer.install_verified_parakeet_v2(
        destination=destination,
        progress=lambda current, total, filename: progress.append(
            (current, total, filename)
        ),
    )

    assert result == destination
    assert requested == list(payloads)
    assert installer.verify_parakeet_v2_bundle(destination) is True
    receipt = json.loads(
        (destination / installer.VERIFICATION_RECEIPT).read_text(encoding="utf-8")
    )
    assert receipt["revision"] == installer.PARAKEET_V2_REVISION
    assert receipt["files"][0]["sha256"] == _tiny_files(payloads)[0].sha256
    assert progress[-1][0] == sum(map(len, payloads.values()))
    assert not list(tmp_path.glob(".parakeet-v2-install-*"))


def test_install_hash_failure_leaves_no_loadable_partial(
    tmp_path: Path, monkeypatch
) -> None:
    expected = {"config.json": b"expected"}
    monkeypatch.setattr(installer, "PARAKEET_V2_FILES", _tiny_files(expected))
    _fake_network(monkeypatch, {"config.json": b"corrupt!"})
    destination = tmp_path / "installed"

    with pytest.raises(installer.ParakeetInstallError, match="verification failed"):
        installer.install_verified_parakeet_v2(destination=destination)

    assert not destination.exists()
    assert not list(tmp_path.glob(".parakeet-v2-install-*"))


def test_valid_existing_bundle_is_reused_without_network(
    tmp_path: Path, monkeypatch
) -> None:
    payloads = {"config.json": b"config"}
    monkeypatch.setattr(installer, "PARAKEET_V2_FILES", _tiny_files(payloads))
    _fake_network(monkeypatch, payloads)
    destination = tmp_path / "installed"
    installer.install_verified_parakeet_v2(destination=destination)

    def unexpected_network(*_args, **_kwargs):
        raise AssertionError("network should not be used")

    monkeypatch.setattr(installer, "_open_url", unexpected_network)

    assert installer.install_verified_parakeet_v2(destination=destination) == destination
