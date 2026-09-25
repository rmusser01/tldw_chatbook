"""OmniVoice ONNX curated managed-artifact catalog."""

from __future__ import annotations

import json


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
