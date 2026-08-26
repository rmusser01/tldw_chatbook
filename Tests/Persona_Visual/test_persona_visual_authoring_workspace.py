"""Private staging tests for screen-owned Persona Visual authoring."""

from __future__ import annotations

import hashlib
import shutil
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from tldw_chatbook.Persona_Visual.assets import PersonaVisualAssetMetadata
from tldw_chatbook.Persona_Visual.authoring import (
    PersonaVisualAuthoringDraft,
    PersonaVisualDraftAsset,
    create_persona_visual_import_draft,
)
from tldw_chatbook.Persona_Visual.authoring_workspace import (
    PersonaVisualAuthoringWorkspaceError,
    adopt_persona_visual_draft_sources,
    cleanup_persona_visual_authoring_workspace,
    create_persona_visual_authoring_workspace,
    stage_persona_visual_authoring_asset,
)


def _png(color=(20, 40, 60, 255)) -> bytes:
    output = BytesIO()
    Image.new("RGBA", (2, 2), color).save(output, format="PNG")
    return output.getvalue()


def _import_draft(source_key: str, data: bytes) -> PersonaVisualAuthoringDraft:
    digest = hashlib.sha256(data).hexdigest()
    metadata = PersonaVisualAssetMetadata(
        asset_key="idle-frame",
        role="frame",
        mime_type="image/png",
        byte_count=len(data),
        sha256=digest,
        width=2,
        height=2,
        frame_count=1,
        duration_ms=None,
    )
    manifest = (
        '{"animations":{"idle":{"frames":[{"asset_id":"idle-frame"}]}},'
        '"authored_triggers":[],"fallbacks":{},"manifest_version":1,'
        '"renderer_type":"sprite_frames","state_catalog":{},'
        '"states":{"idle":{"animation_id":"idle"}}}'
    )
    return create_persona_visual_import_draft(
        persona_id="p-1",
        persona_revision=1,
        expected_identity=None,
        title="Imported",
        description="",
        manifest_json=manifest,
        assets=(PersonaVisualDraftAsset(source_key, metadata),),
    )


def test_stage_asset_returns_private_relative_source_and_path_free_repr(tmp_path: Path):
    profile_root = tmp_path / "profile"
    profile_root.mkdir(mode=0o700)
    workspace = create_persona_visual_authoring_workspace(profile_root)
    updated, asset = stage_persona_visual_authoring_asset(
        workspace,
        _png(),
        state="idle",
    )

    assert not asset.source_storage_key.startswith("/")
    assert (profile_root / asset.source_storage_key).is_file()
    assert asset.metadata.asset_key.startswith("idle-")
    assert asset.metadata.role == "frame"
    assert str(profile_root) not in repr(updated)
    assert str(profile_root) not in repr(asset)


def test_adopt_imported_sources_copies_bytes_and_rewrites_only_source_keys(
    tmp_path: Path,
):
    data = _png()
    imported_root = tmp_path / "imported"
    (imported_root / "assets").mkdir(parents=True)
    (imported_root / "assets" / "000.png").write_bytes(data)
    draft = _import_draft("assets/000.png", data)
    profile_root = tmp_path / "profile"
    profile_root.mkdir(mode=0o700)
    workspace = create_persona_visual_authoring_workspace(profile_root)

    updated, adopted = adopt_persona_visual_draft_sources(
        workspace,
        draft,
        source_root=imported_root,
    )

    assert adopted.manifest_json == draft.manifest_json
    assert adopted.source_kind == "imported"
    assert adopted.assets[0].metadata == draft.assets[0].metadata
    assert adopted.assets[0].source_storage_key != "assets/000.png"
    assert (profile_root / adopted.assets[0].source_storage_key).read_bytes() == data
    assert len(updated.asset_names) == 1


def test_cleanup_refuses_same_path_replacement_with_copied_content(tmp_path: Path):
    profile_root = tmp_path / "profile"
    profile_root.mkdir(mode=0o700)
    workspace, _asset = stage_persona_visual_authoring_asset(
        create_persona_visual_authoring_workspace(profile_root),
        _png(),
        state="idle",
    )
    issued = profile_root / workspace.relative_root
    backup = profile_root / "issued-backup"
    issued.rename(backup)
    shutil.copytree(backup, issued)

    assert cleanup_persona_visual_authoring_workspace(workspace) is False
    assert issued.is_dir()
    assert backup.is_dir()


def test_cleanup_deletes_only_the_exact_issued_workspace(tmp_path: Path):
    profile_root = tmp_path / "profile"
    profile_root.mkdir(mode=0o700)
    workspace, asset = stage_persona_visual_authoring_asset(
        create_persona_visual_authoring_workspace(profile_root),
        _png(),
        state="idle",
    )
    unrelated = profile_root / "keep.txt"
    unrelated.write_text("keep")

    assert cleanup_persona_visual_authoring_workspace(workspace) is True
    assert not (profile_root / asset.source_storage_key).exists()
    assert unrelated.read_text() == "keep"


def test_cleanup_refuses_replaced_staged_file(tmp_path: Path):
    profile_root = tmp_path / "profile"
    profile_root.mkdir(mode=0o700)
    workspace, asset = stage_persona_visual_authoring_asset(
        create_persona_visual_authoring_workspace(profile_root),
        _png(),
        state="idle",
    )
    staged = profile_root / asset.source_storage_key
    staged.unlink()
    staged.write_bytes(b"unrelated")
    staged.chmod(0o600)

    assert cleanup_persona_visual_authoring_workspace(workspace) is False
    assert staged.read_bytes() == b"unrelated"


def test_stage_refuses_replaced_assets_directory_without_external_write(
    tmp_path: Path,
):
    profile_root = tmp_path / "profile"
    profile_root.mkdir(mode=0o700)
    workspace = create_persona_visual_authoring_workspace(profile_root)
    candidate = profile_root / workspace.relative_root
    (candidate / "assets").rename(candidate / "assets-original")
    external = tmp_path / "external"
    external.mkdir()
    (candidate / "assets").symlink_to(external, target_is_directory=True)

    with pytest.raises(
        PersonaVisualAuthoringWorkspaceError,
        match="^persona_visual_authoring_asset_invalid$",
    ):
        stage_persona_visual_authoring_asset(workspace, _png(), state="idle")

    assert list(external.iterdir()) == []


def test_invalid_asset_fails_path_free_without_leaving_a_file(tmp_path: Path):
    profile_root = tmp_path / "profile"
    profile_root.mkdir(mode=0o700)
    workspace = create_persona_visual_authoring_workspace(profile_root)

    with pytest.raises(
        PersonaVisualAuthoringWorkspaceError,
        match="^persona_visual_authoring_asset_invalid$",
    ):
        stage_persona_visual_authoring_asset(workspace, b"not an image", state="idle")

    candidate = profile_root / workspace.relative_root
    assert sorted(path.name for path in candidate.iterdir()) == [
        ".persona-visual-authoring",
        "assets",
    ]
    assert list((candidate / "assets").iterdir()) == []
