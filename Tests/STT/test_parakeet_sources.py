"""Focused source-preference and atomic external-selection tests."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from Tests.Model_Artifacts.test_service import (
    install_descriptor_payload,
    single_file_descriptor,
)
import tldw_chatbook.STT.parakeet_external as parakeet_external
from tldw_chatbook.Local_Ingestion.stt_batch_routing import (
    PARAKEET_V2_MODEL,
    PARAKEET_V3_MODEL,
)
from tldw_chatbook.Local_Ingestion.parakeet_v2_artifact import (
    parakeet_reference,
    parakeet_vad_reference,
)
from tldw_chatbook.Model_Artifacts.store import managed_model_artifact_root
from tldw_chatbook.Model_Artifacts.service import (
    ArtifactDescriptor,
    ArtifactFile,
    ArtifactFormat,
    ArtifactRef,
    ArtifactRole,
    ModelArtifactService,
    ProvenanceClass,
)
from tldw_chatbook.STT.executor import ModelIdentity
from tldw_chatbook.STT.contracts import ExecutionDevice
from tldw_chatbook.STT.parakeet_dispatch import ParakeetDispatch
from tldw_chatbook.STT.parakeet_external import (
    ExternalParakeetVerificationError,
    ExternalParakeetVerifier,
)
from tldw_chatbook.STT.parakeet_sources import (
    ParakeetSourceError,
    ParakeetSourceKey,
    ParakeetSourcePreference,
    ParakeetSourceService,
)


def _descriptor(
    key: ParakeetSourceKey, payload: bytes = b"model"
) -> ArtifactDescriptor:
    file = ArtifactFile("model.onnx", len(payload), hashlib.sha256(payload).hexdigest())
    version = "v2" if key.model_id == PARAKEET_V2_MODEL else "v3"
    return ArtifactDescriptor(
        reference=ArtifactRef(f"parakeet-{version}", "tiny-revision", key.precision),
        model_id=key.model_id,
        role=ArtifactRole.ROOT,
        format=ArtifactFormat.ONNX,
        consumer="stt",
        model_family="parakeet",
        upstream_repository="example/parakeet",
        upstream_revision="tiny-revision",
        source_url="https://example.invalid/model.onnx",
        precision=key.precision,
        expected_installed_bytes=len(payload),
        license_id="cc-by-4.0",
        license_url="https://example.invalid/license",
        usage_notice="test",
        runtime_name="onnx-asr",
        runtime_version_constraint="==0.12.0",
        supported_os=("linux", "darwin", "windows"),
        supported_architectures=("x86-64", "arm64"),
        provenance=(ProvenanceClass.CHATBOOK_CURATED,),
        files=(file,),
        dependencies=(ArtifactRef("silero-vad", "tiny-vad", "f32"),),
    )


def _materialize(root: Path, payload: bytes = b"model") -> None:
    root.mkdir()
    (root / "model.onnx").write_bytes(payload)


def _dispatch(key: ParakeetSourceKey, *, managed: bool) -> ParakeetDispatch:
    ref = (
        parakeet_reference(key.model_id, key.precision)
        if managed
        else _descriptor(key).reference
    )
    ref_tuple = (ref.artifact_id, ref.revision, ref.variant)
    return ParakeetDispatch(
        identity=ModelIdentity(
            provider_id="parakeet-onnx",
            model_id=key.model_id,
            root_revision=ref.revision if managed else None,
            closure_fingerprint="managed" if managed else None,
            precision=key.precision,
            device=ExecutionDevice.CPU,
        ),
        local_source=None,
        managed_store_root=Path("/managed") if managed else None,
        managed_artifact_ref=ref_tuple if managed else None,
        option_updates={},
    )


class _Config:
    def __init__(self, table: object = None, legacy: str = "") -> None:
        self.table = {} if table is None else table
        self.legacy = legacy
        self.writes: list[dict[str, dict[str, object]]] = []

    def read(self, section: str, key: str, default: object) -> object:
        if (section, key) == ("transcription", "parakeet_external_sources"):
            return self.table
        if (section, key) == ("transcription", "parakeet_onnx_model_dir"):
            return self.legacy
        return default

    def write(self, values) -> bool:
        copied = {
            section: dict(section_values) for section, section_values in values.items()
        }
        self.writes.append(copied)
        self.table = copied["transcription"]["parakeet_external_sources"]
        return True


def _service(
    config: _Config,
    *,
    active: set[ParakeetSourceKey] = frozenset(),
    vad_ready: bool = True,
    dispatches: list[tuple[str, str, object]] | None = None,
    write_settings=None,
) -> ParakeetSourceService:
    calls = dispatches if dispatches is not None else []

    def resolve_dispatch(*, model_id: str, precision: str, model_dir: object):
        key = ParakeetSourceKey.from_values(model_id, precision)
        calls.append((model_id, precision, model_dir))
        return _dispatch(key, managed=model_dir is None and key in active)

    return ParakeetSourceService(
        verifier=ExternalParakeetVerifier(),
        read_setting=config.read,
        write_settings=write_settings if write_settings is not None else config.write,
        descriptor_for=lambda model, precision: _descriptor(
            ParakeetSourceKey.from_values(model, precision)
        ),
        active_managed=lambda model, precision: (
            Path("/managed")
            if ParakeetSourceKey.from_values(model, precision) in active
            else None
        ),
        dispatch_resolver=resolve_dispatch,
        vad_ready=lambda: vad_ready,
    )


def test_exact_four_keys_repeat_model_and_precision_and_parse_strictly() -> None:
    table: dict[str, Any] = {}
    for key in ParakeetSourceKey:
        table[key.value] = {
            "model_id": key.model_id,
            "precision": key.precision,
            "preferred_source": "managed",
        }
    table["v2_int8"] = {**table["v2_int8"], "model_id": PARAKEET_V3_MODEL}
    config = _Config(table)
    service = _service(config)

    records = service.records()

    assert set(ParakeetSourceKey) == {
        ParakeetSourceKey.V2_INT8,
        ParakeetSourceKey.V2_F32,
        ParakeetSourceKey.V3_INT8,
        ParakeetSourceKey.V3_F32,
    }
    assert ParakeetSourceKey.V2_INT8 not in records
    for key, record in records.items():
        assert (record.model_id, record.precision) == (key.model_id, key.precision)
    service.close()


def test_prefer_managed_preserves_directory_and_stop_removes_only_directory(
    tmp_path: Path,
) -> None:
    key = ParakeetSourceKey.V2_INT8
    root = tmp_path / "remembered"
    config = _Config(
        {
            key.value: {
                "model_id": key.model_id,
                "precision": key.precision,
                "directory": str(root),
                "preferred_source": "external",
            }
        }
    )
    service = _service(config)

    service.prefer_managed(key)
    assert service.records()[key].directory == root
    assert service.records()[key].preferred_source is ParakeetSourcePreference.MANAGED
    service.stop_using_external(key)
    assert service.records()[key].directory is None
    assert service.records()[key].preferred_source is ParakeetSourcePreference.MANAGED
    assert len(config.writes) == 2
    service.close()


def test_prefer_managed_creates_exact_preference_without_external_directory() -> None:
    config = _Config()
    service = _service(config)

    service.prefer_managed(ParakeetSourceKey.V3_F32)

    record = service.records()[ParakeetSourceKey.V3_F32]
    assert record.directory is None
    assert record.preferred_source is ParakeetSourcePreference.MANAGED
    service.close()


def test_explicit_override_wins_over_preferred_managed(tmp_path: Path) -> None:
    key = ParakeetSourceKey.V2_INT8
    root = tmp_path / "override"
    _materialize(root)
    config = _Config(
        {
            key.value: {
                "model_id": key.model_id,
                "precision": key.precision,
                "preferred_source": "managed",
            }
        }
    )
    calls: list[tuple[str, str, object]] = []
    service = _service(config, active={key}, dispatches=calls)

    dispatch = service.resolve(key, override=root, scope_id="job")

    assert dispatch.local_source is not None
    assert dispatch.managed_artifact_ref is None
    assert dispatch.managed_store_root == managed_model_artifact_root().absolute()
    vad = parakeet_vad_reference()
    assert dispatch.managed_dependency_refs == (
        (vad.artifact_id, vad.revision, vad.variant),
    )
    assert calls == []
    service.close()


def test_external_dispatch_uses_the_injected_verified_vad_store(
    tmp_path: Path,
) -> None:
    key = ParakeetSourceKey.V2_INT8
    root = tmp_path / "override"
    _materialize(root)
    managed_service = ModelArtifactService(tmp_path / "custom-store")
    vad = parakeet_vad_reference()
    descriptor = single_file_descriptor(
        vad,
        ArtifactRole.DEPENDENCY,
        b"vad",
    )
    install_descriptor_payload(managed_service, tmp_path, descriptor, b"vad")
    config = _Config()
    service = ParakeetSourceService(
        verifier=ExternalParakeetVerifier(),
        read_setting=config.read,
        write_settings=config.write,
        descriptor_for=lambda model, precision: _descriptor(
            ParakeetSourceKey.from_values(model, precision)
        ),
        active_managed=lambda _model, _precision: None,
        managed_service=managed_service,
    )

    dispatch = service.resolve(key, override=root)

    assert dispatch.managed_store_root == managed_service.artifacts_path.parent
    assert dispatch.managed_artifact_ref is None
    assert dispatch.managed_dependency_refs == (
        (vad.artifact_id, vad.revision, vad.variant),
    )
    service.close()


def test_valid_external_resolution_fails_closed_when_managed_vad_is_missing(
    tmp_path: Path,
) -> None:
    key = ParakeetSourceKey.V2_INT8
    root = tmp_path / "override"
    _materialize(root)
    service = _service(_Config(), vad_ready=False)

    with pytest.raises(ParakeetSourceError) as caught:
        service.resolve(key, override=root)

    assert caught.value.code.value == "managed_vad_unavailable"
    assert str(root) not in str(caught.value)
    service.close()


def test_preferred_managed_never_tries_remembered_external(tmp_path: Path) -> None:
    key = ParakeetSourceKey.V2_INT8
    missing = tmp_path / "missing-remembered"
    config = _Config(
        {
            key.value: {
                "model_id": key.model_id,
                "precision": key.precision,
                "directory": str(missing),
                "preferred_source": "managed",
            }
        }
    )
    service = _service(config, active={key})

    dispatch = service.resolve(key)

    ref = parakeet_reference(key.model_id, key.precision)
    assert dispatch.managed_artifact_ref == (
        ref.artifact_id,
        ref.revision,
        ref.variant,
    )
    service.close()


def test_invalid_explicit_and_preferred_external_fail_without_fallback(
    tmp_path: Path,
) -> None:
    key = ParakeetSourceKey.V2_INT8
    missing = tmp_path / "private-missing"
    config = _Config(
        {
            key.value: {
                "model_id": key.model_id,
                "precision": key.precision,
                "directory": str(missing),
                "preferred_source": "external",
            }
        }
    )
    calls: list[tuple[str, str, object]] = []
    service = _service(config, active={key}, dispatches=calls)

    for override in (None, missing):
        with pytest.raises(ExternalParakeetVerificationError) as caught:
            service.resolve(key, override=override)
        assert str(missing) not in str(caught.value)
        assert str(missing) not in repr(caught.value)
    assert calls == []
    service.close()


def test_preferred_managed_is_authoritative_when_exact_active_root_is_missing() -> None:
    key = ParakeetSourceKey.V3_INT8
    config = _Config(
        {
            key.value: {
                "model_id": key.model_id,
                "precision": key.precision,
                "preferred_source": "managed",
            }
        }
    )
    calls: list[tuple[str, str, object]] = []
    service = _service(config, dispatches=calls)

    with pytest.raises(ParakeetSourceError) as caught:
        service.resolve(key)

    assert "managed_source_unavailable" in str(caught.value)
    assert calls == []
    service.close()


def test_no_preference_uses_active_managed_before_legacy() -> None:
    key = ParakeetSourceKey.V3_INT8
    config = _Config()
    calls: list[tuple[str, str, object]] = []
    service = _service(config, active={key}, dispatches=calls)

    dispatch = service.resolve(key)

    assert dispatch.managed_artifact_ref is not None
    assert calls == [(key.model_id, key.precision, None)]
    service.close()


def test_legacy_singular_path_only_migrates_v2_int8_after_vad_ready(
    tmp_path: Path,
) -> None:
    root = tmp_path / "legacy"
    _materialize(root)
    not_ready_config = _Config(legacy=str(root))
    not_ready = _service(not_ready_config, vad_ready=False)

    with pytest.raises(ParakeetSourceError):
        not_ready.resolve(ParakeetSourceKey.V2_INT8)
    assert not_ready_config.writes == []
    not_ready.close()

    config = _Config(legacy=str(root))
    service = _service(config)
    dispatch = service.resolve(ParakeetSourceKey.V2_INT8)
    assert dispatch.local_source is not None
    assert len(config.writes) == 1
    assert (
        service.records()[ParakeetSourceKey.V2_INT8].preferred_source
        is ParakeetSourcePreference.EXTERNAL
    )
    service.close()


def test_legacy_singular_path_is_not_considered_for_v3_or_exact_preference(
    tmp_path: Path,
) -> None:
    root = tmp_path / "legacy"
    _materialize(root)
    v3_calls: list[tuple[str, str, object]] = []
    v3 = _service(_Config(legacy=str(root)), dispatches=v3_calls)
    v3.resolve(ParakeetSourceKey.V3_INT8)
    assert v3_calls == [(PARAKEET_V3_MODEL, "int8", None)]
    v3.close()

    key = ParakeetSourceKey.V2_INT8
    config = _Config(
        {
            key.value: {
                "model_id": key.model_id,
                "precision": key.precision,
                "preferred_source": "managed",
            }
        },
        legacy=str(root),
    )
    preferred = _service(config)
    with pytest.raises(ParakeetSourceError):
        preferred.resolve(key)
    assert config.writes == []
    preferred.close()


def test_prepare_is_write_free_commit_writes_once_and_accept_requires_match(
    tmp_path: Path,
) -> None:
    key = ParakeetSourceKey.V2_F32
    root = tmp_path / "external"
    _materialize(root)
    config = _Config()
    service = _service(config)
    prepared = service.prepare_external(
        key,
        root,
        owner=("scope", "selection"),
    )

    commit = service.prepare_config_commit(prepared)
    assert config.writes == []
    assert commit.section_values["transcription"]["parakeet_external_sources"]

    service.commit_external(prepared)
    assert len(config.writes) == 1
    assert service.records()[key].directory == root.absolute()

    config.table = {}
    with pytest.raises(ParakeetSourceError):
        service.accept_committed(commit)
    service.close()


def test_config_writer_failure_is_fail_closed_and_path_private(tmp_path: Path) -> None:
    key = ParakeetSourceKey.V2_INT8
    root = tmp_path / "private-external"
    _materialize(root)
    config = _Config()

    def fail_write(_values) -> bool:
        raise OSError(str(root))

    service = _service(config, write_settings=fail_write)
    prepared = service.prepare_external(key, root, owner=("scope", "selection"))
    with pytest.raises(ParakeetSourceError) as caught:
        service.commit_external(prepared)

    assert caught.value.code.value == "config_write_failed"
    assert str(root) not in str(caught.value)
    assert config.table == {}
    service.close()


def test_prepare_config_commit_rechecks_snapshot_and_vad_before_any_write(
    tmp_path: Path,
) -> None:
    key = ParakeetSourceKey.V2_INT8
    root = tmp_path / "external"
    _materialize(root)
    config = _Config()
    service = _service(config, vad_ready=False)
    prepared = service.prepare_external(key, root, owner=("scope", "selection"))

    with pytest.raises(ParakeetSourceError):
        service.prepare_config_commit(prepared)
    assert config.writes == []

    service.close()
    ready = _service(config)
    prepared = ready.prepare_external(key, root, owner=("scope", "selection"))
    (root / "model.onnx").write_bytes(b"other")
    with pytest.raises(ParakeetSourceError):
        ready.prepare_config_commit(prepared)
    assert config.writes == []
    ready.close()


def test_retain_prepared_requires_unchanged_matching_selection(tmp_path: Path) -> None:
    key = ParakeetSourceKey.V2_INT8
    root = tmp_path / "external"
    _materialize(root)
    config = _Config()
    service = _service(config)
    prepared = service.prepare_external(key, root, owner=("scope", "prepare"))
    metadata = (root / "model.onnx").stat()
    (root / "model.onnx").touch()
    assert (root / "model.onnx").stat().st_mtime_ns >= metadata.st_mtime_ns

    with pytest.raises(ParakeetSourceError):
        service.retain_prepared("batch", prepared)
    service.close()


def test_release_scopes_except_releases_only_previously_observed_scopes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = ParakeetSourceKey.V2_INT8
    pending_root = tmp_path / "pending"
    active_root = tmp_path / "active"
    _materialize(pending_root)
    _materialize(active_root)
    service = _service(_Config())
    real_open = parakeet_external.os.open
    open_count = 0

    def counted_open(path, flags):
        nonlocal open_count
        open_count += 1
        return real_open(path, flags)

    monkeypatch.setattr(parakeet_external.os, "open", counted_open)
    pending = service.prepare_external(
        key,
        pending_root,
        owner=("scope", "pending"),
    )
    service.retain_prepared("pending", pending)
    assert open_count == 1

    service.release_scopes_except(set())
    service.prepare_external(key, pending_root)
    assert open_count == 1

    active = service.prepare_external(
        key,
        active_root,
        owner=("scope", "active"),
    )
    service.retain_prepared("active", active)
    service.release_scopes_except({"pending", "active"})
    service.prepare_external(key, pending_root)
    service.prepare_external(key, active_root)
    assert open_count == 2

    service.release_scopes_except({"active"})
    service.prepare_external(key, pending_root)
    service.prepare_external(key, active_root)
    service.close()

    assert open_count == 3
