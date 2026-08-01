"""Tests for the application-wide managed model registry and store."""

from pathlib import Path


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
