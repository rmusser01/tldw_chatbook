"""Focused tests for shared, download-free Parakeet dispatch resolution."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_chatbook.Local_Ingestion.parakeet_v2_artifact import (
    parakeet_reference,
    parakeet_vad_reference,
)
from tldw_chatbook.Local_Ingestion.stt_batch_routing import PARAKEET_V2_MODEL
from tldw_chatbook.Model_Artifacts.store import managed_model_artifact_root
from tldw_chatbook.Model_Artifacts import ArtifactRef
from tldw_chatbook.STT.contracts import ExecutionDevice


def _configured_files(root: Path, precision: str) -> None:
    filenames = (
        (
            "config.json",
            "vocab.txt",
            "encoder-model.int8.onnx",
            "decoder_joint-model.int8.onnx",
        )
        if precision == "int8"
        else (
            "config.json",
            "vocab.txt",
            "encoder-model.onnx",
            "encoder-model.onnx.data",
            "decoder_joint-model.onnx",
        )
    )
    root.mkdir()
    for index, filename in enumerate(filenames):
        (root / filename).write_bytes(f"configured-{index}".encode())


@pytest.mark.parametrize("precision", ("int8", "f32"))
def test_configured_library_and_console_calls_share_exact_dispatch(
    tmp_path: Path,
    precision: str,
) -> None:
    from tldw_chatbook.STT.parakeet_dispatch import resolve_parakeet_dispatch

    model_root = tmp_path / precision
    _configured_files(model_root, precision)

    library = resolve_parakeet_dispatch(
        model_id=PARAKEET_V2_MODEL,
        precision=precision,
        model_dir=str(model_root),
    )
    console = resolve_parakeet_dispatch(
        model_id=PARAKEET_V2_MODEL,
        precision=precision,
        model_dir=model_root,
    )

    assert library == console
    assert library.identity.provider_id == "parakeet-onnx"
    assert library.identity.model_id == PARAKEET_V2_MODEL
    assert library.identity.precision == precision
    assert library.identity.device is ExecutionDevice.CPU
    assert (
        library.identity.root_revision
        == parakeet_reference(PARAKEET_V2_MODEL, precision).revision
    )
    assert library.identity.closure_fingerprint is None
    assert library.identity.local_snapshot_token == library.local_source.token
    assert library.managed_store_root == managed_model_artifact_root().absolute()
    assert library.managed_artifact_ref is None
    assert library.managed_dependency_refs == (
        (
            parakeet_vad_reference().artifact_id,
            parakeet_vad_reference().revision,
            parakeet_vad_reference().variant,
        ),
    )
    assert dict(library.option_updates) == {
        "transcription_model_dir": str(model_root.absolute())
    }


@pytest.mark.parametrize("precision", ("int8", "f32"))
def test_managed_library_and_console_calls_share_closure_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    precision: str,
) -> None:
    from tldw_chatbook.STT import parakeet_dispatch

    root = ArtifactRef("parakeet-v2", "managed-revision", precision)
    handle = SimpleNamespace(
        root=root,
        closure_fingerprint="managed-closure-fingerprint",
    )

    class _Lease:
        def __init__(self) -> None:
            self.handle = handle
            self.closed = False

        def close(self) -> None:
            self.closed = True

    leases: list[_Lease] = []

    class _Service:
        def acquire(self, reference: ArtifactRef) -> _Lease:
            assert reference == root
            lease = _Lease()
            leases.append(lease)
            return lease

    managed_root = tmp_path / "managed-store"
    monkeypatch.setattr(parakeet_dispatch, "parakeet_v2_managed_service", _Service)
    monkeypatch.setattr(
        parakeet_dispatch,
        "active_managed_parakeet_dir",
        lambda model, precision, *, service: tmp_path / "installed",
    )
    monkeypatch.setattr(
        parakeet_dispatch,
        "parakeet_reference",
        lambda model, precision: root,
    )
    monkeypatch.setattr(
        parakeet_dispatch,
        "managed_model_artifact_root",
        lambda: managed_root,
    )
    monkeypatch.setattr(
        parakeet_dispatch,
        "parakeet_v2_install_dir",
        lambda: tmp_path / "missing-legacy",
    )

    library = parakeet_dispatch.resolve_parakeet_dispatch(
        model_id=PARAKEET_V2_MODEL,
        precision=precision,
        model_dir=None,
    )
    console = parakeet_dispatch.resolve_parakeet_dispatch(
        model_id=PARAKEET_V2_MODEL,
        precision=precision,
        model_dir=None,
    )

    assert library == console
    assert library.identity.root_revision == "managed-revision"
    assert library.identity.closure_fingerprint == "managed-closure-fingerprint"
    assert library.identity.precision == precision
    assert library.identity.local_snapshot_token is None
    assert library.local_source is None
    assert library.managed_store_root == managed_root
    assert library.managed_artifact_ref == (
        "parakeet-v2",
        "managed-revision",
        precision,
    )
    assert library.managed_dependency_refs == ()
    assert dict(library.option_updates) == {}
    assert len(leases) == 2
    assert all(lease.closed for lease in leases)


def test_verified_legacy_library_and_console_calls_share_snapshot_and_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.STT import parakeet_dispatch

    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    paths = (legacy_root / ".tldw-verified.json",) + tuple(
        legacy_root / name
        for name in (
            "config.json",
            "vocab.txt",
            "encoder-model.int8.onnx",
            "decoder_joint-model.int8.onnx",
        )
    )
    for index, path in enumerate(paths):
        path.write_bytes(f"legacy-{index}".encode())

    monkeypatch.setattr(
        parakeet_dispatch,
        "active_managed_parakeet_dir",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(parakeet_dispatch, "parakeet_v2_install_dir", lambda: legacy_root)
    monkeypatch.setattr(
        parakeet_dispatch,
        "verify_parakeet_v2_bundle",
        lambda directory: Path(directory) == legacy_root,
    )

    library = parakeet_dispatch.resolve_parakeet_dispatch(
        model_id=PARAKEET_V2_MODEL,
        precision="int8",
        model_dir=None,
    )
    console = parakeet_dispatch.resolve_parakeet_dispatch(
        model_id=PARAKEET_V2_MODEL,
        precision="int8",
        model_dir=None,
    )

    assert library == console
    assert library.identity.root_revision is None
    assert library.identity.closure_fingerprint is None
    assert library.identity.local_snapshot_token == library.local_source.token
    assert dict(library.option_updates) == {
        "transcription_model_dir": str(legacy_root.absolute()),
        "_verify_legacy_parakeet_v2": True,
    }


def test_resolver_never_loads_download_layers_and_fails_without_installed_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.STT import parakeet_dispatch

    monkeypatch.delitem(
        sys.modules,
        "tldw_chatbook.Model_Artifacts.acquisition",
        raising=False,
    )
    monkeypatch.delitem(
        sys.modules,
        "tldw_chatbook.Model_Artifacts.fetch",
        raising=False,
    )
    monkeypatch.setattr(
        parakeet_dispatch,
        "active_managed_parakeet_dir",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        parakeet_dispatch,
        "parakeet_v2_install_dir",
        lambda: tmp_path / "missing",
    )
    monkeypatch.setattr(
        parakeet_dispatch,
        "verify_parakeet_v2_bundle",
        lambda _directory: False,
    )

    with pytest.raises(FileNotFoundError, match="No installed Parakeet artifact"):
        parakeet_dispatch.resolve_parakeet_dispatch(
            model_id=PARAKEET_V2_MODEL,
            precision="int8",
            model_dir=None,
        )

    assert "tldw_chatbook.Model_Artifacts.acquisition" not in sys.modules
    assert "tldw_chatbook.Model_Artifacts.fetch" not in sys.modules
