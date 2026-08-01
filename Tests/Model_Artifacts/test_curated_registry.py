"""Tests for the application-wide managed model registry and store."""

from pathlib import Path

import pytest


def test_store_root_matches_the_parakeet_adapter_value() -> None:
    """Moving the root does not orphan existing managed installs."""
    from tldw_chatbook.Local_Ingestion import parakeet_v2_artifact
    from tldw_chatbook.Model_Artifacts.store import managed_model_artifact_root

    assert (
        managed_model_artifact_root()
        == parakeet_v2_artifact.managed_model_artifact_root()
    )
    assert isinstance(managed_model_artifact_root(), Path)


def test_managed_service_uses_an_explicit_root(tmp_path: Path) -> None:
    """Tests and callers can bind the service to an isolated store."""
    from tldw_chatbook.Model_Artifacts.store import managed_service

    service = managed_service(tmp_path)
    assert service.artifacts_path == tmp_path / "artifacts"


def test_registry_lists_registered_descriptors_in_registration_order() -> None:
    """Curated entries are enumerable and remain catalog-compatible."""
    from tldw_chatbook.Local_Ingestion.parakeet_v2_artifact import (
        parakeet_v2_descriptor,
        parakeet_v2_reference,
        parakeet_v2_source_map,
    )
    from tldw_chatbook.Model_Artifacts.curated_registry import CuratedRegistry

    registry = CuratedRegistry()
    descriptor = parakeet_v2_descriptor()
    reference = parakeet_v2_reference()
    registry.register(descriptor, sources=parakeet_v2_source_map()[reference])

    assert registry.list() == (descriptor,)
    assert registry.descriptor(reference) is descriptor
    assert registry.sources(reference) == parakeet_v2_source_map()[reference]


def test_registry_descriptor_raises_keyerror_for_unknown_ref() -> None:
    """Unknown references cannot be synthesized through the registry."""
    from tldw_chatbook.Model_Artifacts.curated_registry import CuratedRegistry
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef

    with pytest.raises(KeyError):
        CuratedRegistry().descriptor(ArtifactRef("nope", "rev", "int8"))


def test_registry_does_not_import_the_acquisition_runtime_at_module_scope() -> None:
    """Worker-side imports keep acquisition and fetch out of their graph."""
    import inspect

    from tldw_chatbook.Model_Artifacts import curated_registry as module

    before_type_checking = inspect.getsource(module).split("if TYPE_CHECKING:")[0]
    assert "class CuratedRegistry:" in inspect.getsource(module)
    assert "from .acquisition import" not in before_type_checking


def test_default_registry_contains_parakeet_v2() -> None:
    """The existing production model is the first curated entry."""
    from tldw_chatbook.Local_Ingestion.parakeet_v2_artifact import (
        parakeet_v2_reference,
    )
    from tldw_chatbook.Model_Artifacts.curated_registry import curated_registry

    assert parakeet_v2_reference() in {
        descriptor.reference for descriptor in curated_registry().list()
    }
