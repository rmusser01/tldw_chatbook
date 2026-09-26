"""OmniVoice ONNX curated managed-artifact catalog."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_descriptor_shape() -> None:
    from tldw_chatbook.TTS import omnivoice_artifact_catalog as cat

    d = cat.omnivoice_onnx_descriptor()
    assert d.model_family == "omnivoice"
    assert d.precision == "int8hq"
    assert d.format.value == "onnx"
    assert d.role.value == "root"
    # The weights are Apache-2.0 (model card + export README); the binding
    # terms come from the Boson tokenizer license, which must be spelled out.
    assert "Apache-2.0" in d.usage_notice
    assert "CC-BY-NC" not in d.usage_notice
    assert "Higgs Audio 2 Community License" in d.usage_notice
    assert "100,000 annual active users" in d.usage_notice
    assert "Acceptable Use Policy" in d.usage_notice
    assert any("omnivoice_lm_int8_hq/model.onnx_data" == file.path for file in d.files)


def test_reference_pins_the_exact_upstream_revision() -> None:
    from tldw_chatbook.TTS import omnivoice_artifact_catalog as cat

    reference = cat.omnivoice_onnx_reference()
    assert reference.artifact_id == "omnivoice-onnx-int8hq"
    assert reference.variant == "int8hq"
    assert reference.revision == cat.OMNIVOICE_ONNX_REVISION
    assert len(cat.OMNIVOICE_ONNX_REVISION) == 40
    assert all(char in "0123456789abcdef" for char in cat.OMNIVOICE_ONNX_REVISION)


def test_required_layout_validated() -> None:
    from tldw_chatbook.TTS import omnivoice_artifact_catalog as cat

    assert "tokenizer.json" in cat.OMNIVOICE_ONNX_REQUIRED_PATHS
    assert len(cat.OMNIVOICE_ONNX_REQUIRED_PATHS) == 8
    assert {file.path for file in cat.omnivoice_onnx_files()} == set(
        cat.OMNIVOICE_ONNX_REQUIRED_PATHS
    )


def test_expected_installed_bytes_cover_every_declared_file() -> None:
    from tldw_chatbook.TTS import omnivoice_artifact_catalog as cat

    descriptor = cat.omnivoice_onnx_descriptor()
    assert descriptor.expected_installed_bytes == sum(
        file.size_bytes for file in descriptor.files
    )
    assert all(file.size_bytes > 0 for file in descriptor.files)


def test_source_map_covers_all_files() -> None:
    from tldw_chatbook.TTS import omnivoice_artifact_catalog as cat

    sources = cat.omnivoice_onnx_source_map()
    d = cat.omnivoice_onnx_descriptor()
    joined = json.dumps(sources[cat.omnivoice_onnx_reference()])
    for file in d.files:
        assert file.path in joined
        assert (
            f"https://huggingface.co/{cat.OMNIVOICE_ONNX_REPOSITORY}/resolve/"
            f"{cat.OMNIVOICE_ONNX_REVISION}/{file.path}"
        ) == sources[cat.omnivoice_onnx_reference()][file.path]


def test_registered_in_the_shared_curated_registry() -> None:
    from tldw_chatbook.Model_Artifacts.curated_registry import curated_registry
    from tldw_chatbook.TTS import omnivoice_artifact_catalog as cat

    registry = curated_registry()

    assert (
        registry.descriptor(cat.omnivoice_onnx_reference())
        == cat.omnivoice_onnx_descriptor()
    )
    assert (
        registry.sources(cat.omnivoice_onnx_reference())
        == (cat.omnivoice_onnx_source_map()[cat.omnivoice_onnx_reference()])
    )


from tldw_chatbook.TTS import omnivoice_artifact_catalog as cat


def _tree(root: Path) -> Path:
    for rel in cat.OMNIVOICE_ONNX_REQUIRED_PATHS:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(b"x")
    return root


def test_setup_state_engine_missing_wins() -> None:
    state = cat.omnivoice_setup_state(
        None, missing_modules=lambda: ["onnxruntime"], managed_root=lambda: None
    )
    assert state == "engine_missing"


def test_setup_state_model_missing_without_any_root() -> None:
    assert cat.omnivoice_setup_state(
        None, missing_modules=list, managed_root=lambda: None
    ) == "model_missing"


def test_setup_state_ready_from_configured_root(tmp_path: Path) -> None:
    root = _tree(tmp_path / "m")
    assert cat.omnivoice_setup_state(
        str(root), missing_modules=list, managed_root=lambda: None
    ) == "ready"


def test_setup_state_ready_from_managed_root(tmp_path: Path) -> None:
    root = _tree(tmp_path / "m")
    assert cat.omnivoice_setup_state(
        "", missing_modules=list, managed_root=lambda: root
    ) == "ready"


def test_setup_state_env_model_root_wins_like_the_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The engine prefers OMNIVOICE_MODEL_ROOT from the environment; the
    wizard must too, or an env-configured user is told to download 1.1 GB."""
    root = _tree(tmp_path / "env-root")
    monkeypatch.setenv("OMNIVOICE_MODEL_ROOT", str(root))
    assert cat.omnivoice_setup_state(
        "", missing_modules=list, managed_root=lambda: None
    ) == "ready"


def test_setup_state_broken_model_root_is_model_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OMNIVOICE_MODEL_ROOT", raising=False)
    partial = tmp_path / "partial"
    partial.mkdir()
    for model_root in (str(tmp_path / "does-not-exist"), str(partial), "bad\x00root"):
        assert cat.omnivoice_setup_state(
            model_root, missing_modules=list, managed_root=lambda: None
        ) == "model_missing"


def test_catalog_serves_only_the_omnivoice_descriptor() -> None:
    catalog = cat.OmniVoiceCatalog()
    assert catalog.descriptor(cat.omnivoice_onnx_reference()).model_id == "omnivoice-onnx-int8hq"
    with pytest.raises(KeyError):
        catalog.descriptor(object())


async def test_wrappers_pass_the_pinned_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list = []

    class FakeAcquisition:
        def __init__(self, service, **kwargs):
            calls.append(("init", service))

        async def preflight(self, ref, catalog, *, sources):
            calls.append(("preflight", ref, type(catalog).__name__, sources))
            return "REPORT"

        async def provision(self, ref, grant, catalog, *, sources, progress=None):
            calls.append(("provision", ref, grant, sources, progress))
            return ref

    class FakeReport:
        def grant(self):
            return "GRANT"

    class FakeService:
        def artifact_path(self, ref):
            return Path("/managed/omnivoice")

    import tldw_chatbook.Model_Artifacts.acquisition as acquisition

    monkeypatch.setattr(acquisition, "ArtifactAcquisitionService", FakeAcquisition)
    service = FakeService()
    ref = cat.omnivoice_onnx_reference()
    sources = cat.omnivoice_onnx_source_map()

    assert await cat.run_omnivoice_preflight(core=service, credential_resolver=object()) == "REPORT"
    path = await cat.run_omnivoice_provision(
        FakeReport(), core=service, credential_resolver=object(), progress=print
    )
    assert path == Path("/managed/omnivoice")
    assert calls[1] == ("preflight", ref, "OmniVoiceCatalog", sources)
    assert calls[3] == ("provision", ref, "GRANT", sources, print)
