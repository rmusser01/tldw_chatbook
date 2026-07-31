"""TASK-595 Task 1: consume_source install semantics."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tldw_chatbook.Model_Artifacts.service import (
    ArtifactPathError,
    ModelArtifactService,
)

# Reuse the descriptor/payload builders from test_service.py
from Tests.Model_Artifacts.test_service import (
    artifact_file,
    descriptor,
    source_tree,
    install_inputs,
)


@pytest.fixture()
def service(tmp_path: Path) -> ModelArtifactService:
    """Create a service with a managed root."""
    return ModelArtifactService(tmp_path / "root")


def test_consume_source_moves_files_and_installs(service: ModelArtifactService, tmp_path: Path) -> None:
    """Files are moved (source emptied), install verifies and promotes."""
    # Use a descriptor with a single file to keep things simple
    desc = descriptor()
    # Create a source directory inside the service staging path
    source = Path(service.staging_path) / "managed" / "src"
    source.mkdir(parents=True)
    # Write the payload files to the source
    for file in desc.files:
        file_path = source / file.path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(b"x")  # matches artifact_file() default content

    ref = service.install(desc, source, consume_source=True)
    installed = service.artifact_path(ref)
    assert installed.exists()
    # Moved, not copied: the declared payload files are gone from source.
    for file in desc.files:
        assert not (source / file.path).exists()


def test_consume_source_outside_root_raises(service: ModelArtifactService, tmp_path: Path) -> None:
    """Raises ArtifactPathError when source is outside the service root."""
    desc = descriptor()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    # Write the payload files to the outside directory
    for file in desc.files:
        file_path = outside / file.path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(b"x")

    with pytest.raises(ArtifactPathError):
        service.install(desc, outside, consume_source=True)
    # Nothing consumed on refusal.
    for file in desc.files:
        assert (outside / file.path).exists()


def test_consume_source_exdev_falls_back_to_copy(service: ModelArtifactService, monkeypatch) -> None:
    """EXDEV inside the root degrades to copy+delete, still installing."""
    desc = descriptor()
    source = Path(service.staging_path) / "managed" / "src"
    source.mkdir(parents=True)
    # Write the payload files to the source
    for file in desc.files:
        file_path = source / file.path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(b"x")

    real_replace = os.replace

    def exdev_replace(src, dst, *a, **k):
        # Only raise EXDEV for payload files from the source directory,
        # not for manifest files from atomic_write_json.
        if Path(src).is_relative_to(source):
            raise OSError(18, "Invalid cross-device link")  # errno.EXDEV
        return real_replace(src, dst, *a, **k)

    monkeypatch.setattr("tldw_chatbook.Model_Artifacts.service.os.replace", exdev_replace)
    ref = service.install(desc, source, consume_source=True)
    monkeypatch.setattr("tldw_chatbook.Model_Artifacts.service.os.replace", real_replace)
    assert service.artifact_path(ref).exists()


def test_default_copy_behavior_unchanged(service: ModelArtifactService, tmp_path: Path) -> None:
    """consume_source=False keeps today's copy semantics: source intact."""
    desc = descriptor()
    source = tmp_path / "root" / "staging" / "src2"
    source.mkdir(parents=True)
    # Write the payload files to the source
    for file in desc.files:
        file_path = source / file.path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(b"x")

    service.install(desc, source)
    for file in desc.files:
        assert (source / file.path).exists()
