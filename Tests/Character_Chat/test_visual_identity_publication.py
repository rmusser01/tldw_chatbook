"""Real filesystem/database coverage for immutable Visual Identity publication."""

from __future__ import annotations

import hashlib
import os
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

import tldw_chatbook.Character_Chat.visual_identity as visual_identity_module
from tldw_chatbook.Character_Chat.visual_identity import (
    VisualIdentityPublicationError,
    create_visual_identity_candidate,
    publish_visual_identity_candidate,
    resolve_visual_identity,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.VisualIdentity_DB import VisualIdentityRepository


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    stream = BytesIO()
    Image.new("RGB", (16, 16), color).save(stream, format="PNG")
    return stream.getvalue()


def _asset(
    root: Path,
    *,
    expression_key: str,
    original_label: str,
    color: tuple[int, int, int],
) -> dict[str, Any]:
    data = _png_bytes(color)
    relpath = f"characters/test/expressions/{original_label}.png"
    path = root / "assets" / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {
        "expression_key": expression_key,
        "original_expression_key": original_label,
        "display_label": original_label.title(),
        "source_filename": path.name,
        "storage_relpath": relpath,
        "content_type": "image/png",
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "width": 16,
        "height": 16,
        "source_context": {"fixture": True},
        "is_animated": False,
        "frame_count": 1,
        "duration_ms": None,
    }


@pytest.fixture
def publication_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    package_root = tmp_path / "installed-package"
    assets = [
        _asset(
            package_root,
            expression_key="neutral",
            original_label="neutral",
            color=(20, 30, 40),
        ),
        _asset(
            package_root,
            expression_key="thinking",
            original_label="thinking",
            color=(50, 60, 70),
        ),
        _asset(
            package_root,
            expression_key="custom:speaking",
            original_label="speaking",
            color=(80, 90, 100),
        ),
    ]
    monkeypatch.setattr(
        visual_identity_module.resources, "files", lambda _package: package_root
    )
    db = CharactersRAGDB(tmp_path / "publication.db", "publication-test")
    actor_id = db.add_character_card({"name": "Publication actor"})
    other_actor_id = db.add_character_card({"name": "Other actor"})
    assert actor_id is not None and other_actor_id is not None
    repository = VisualIdentityRepository(db)
    graph = repository.activate_pack(
        pack={
            "title": "Built-in fixture reactions",
            "description": "Publication fixture",
            "default_expression_key": "neutral",
            "source_kind": "builtin",
            "source_context": {"source_id": "fixture.builtin.reactions"},
        },
        manifest={"fixture": "built-in"},
        assets=assets,
        actor_kind="character",
        actor_id=actor_id,
    )
    with db.transaction():
        db.execute_query(
            """
            INSERT INTO visual_identity_bindings(
                owner_user_id, actor_kind, actor_id, pack_id, active_version_id
            ) VALUES (0, 'character', ?, ?, ?)
            """,
            (str(other_actor_id), graph["pack"]["id"], graph["version"]["id"]),
        )
    try:
        yield {
            "db": db,
            "actor_id": actor_id,
            "other_actor_id": other_actor_id,
            "graph": graph,
            "package_root": package_root,
            "user_root": tmp_path / "profile",
        }
    finally:
        db.close_connection()


def _active_identity(environment: dict[str, Any]) -> tuple[int, int]:
    graph = VisualIdentityRepository(environment["db"]).get_active_actor_pack(
        "character", environment["actor_id"]
    )
    assert graph is not None
    return int(graph["pack"]["id"]), int(graph["version"]["id"])


def test_replace_clear_and_generated_bytes_are_staging_only(
    publication_environment,
) -> None:
    environment = publication_environment
    before = _active_identity(environment)
    candidate = create_visual_identity_candidate(
        environment["db"], actor_kind="character", actor_id=environment["actor_id"]
    )

    candidate.stage_replacement("thinking", _png_bytes((1, 2, 3)), source="upload")
    candidate.stage_replacement(
        "custom:speaking", _png_bytes((4, 5, 6)), source="generated"
    )
    candidate.stage_clear("custom:speaking")

    assert _active_identity(environment) == before
    assert candidate.replaced_expression_keys == ("thinking",)
    assert candidate.cleared_expression_keys == ("custom:speaking",)
    assert not (environment["user_root"] / "visual_identities").exists()


def test_first_builtin_save_forks_once_and_atomically_switches_only_target_actor(
    publication_environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = publication_environment
    db = environment["db"]
    old_graph = environment["graph"]
    package_before = {
        path.relative_to(environment["package_root"]): path.read_bytes()
        for path in environment["package_root"].rglob("*")
        if path.is_file()
    }
    candidate = create_visual_identity_candidate(
        db, actor_kind="character", actor_id=environment["actor_id"]
    )
    candidate.stage_replacement("neutral", _png_bytes((11, 12, 13)))
    candidate.stage_replacement("thinking", _png_bytes((14, 15, 16)))
    secure_calls: list[tuple[Path, bool, bool]] = []
    original_secure = visual_identity_module.secure_private_directory

    def secure_spy(path, *, create, application_owned):
        secure_calls.append((Path(path), create, application_owned))
        return original_secure(path, create=create, application_owned=application_owned)

    replace_calls: list[tuple[Path, Path]] = []

    def replace_spy(source, destination):
        source_path, destination_path = Path(source), Path(destination)
        assert source_path.parent == destination_path.parent
        assert source_path.stat().st_dev == destination_path.parent.stat().st_dev
        replace_calls.append((source_path, destination_path))
        os.replace(source_path, destination_path)

    monkeypatch.setattr(visual_identity_module, "secure_private_directory", secure_spy)
    result = publish_visual_identity_candidate(
        db,
        candidate,
        user_data_dir=environment["user_root"],
        atomic_replace=replace_spy,
    )

    assert result.actor_kind == "character"
    assert result.actor_id == str(environment["actor_id"])
    assert result.old_pack_id == old_graph["pack"]["id"]
    assert result.old_version_id == old_graph["version"]["id"]
    assert result.new_pack_id != result.old_pack_id
    assert result.new_version_id != result.old_version_id
    assert len(replace_calls) == 1
    assert replace_calls[0][0].name.startswith(".staging-")
    assert any(
        path.name.startswith(".staging-") and create and application_owned
        for path, create, application_owned in secure_calls
    )
    manual_packs = db.execute_query(
        "SELECT * FROM visual_identity_packs WHERE source_kind = 'manual'"
    ).fetchall()
    assert len(manual_packs) == 1
    manual_versions = db.execute_query(
        "SELECT * FROM visual_identity_pack_versions WHERE pack_id = ?",
        (result.new_pack_id,),
    ).fetchall()
    assert len(manual_versions) == 1
    old_versions = db.execute_query(
        "SELECT * FROM visual_identity_pack_versions WHERE pack_id = ?",
        (result.old_pack_id,),
    ).fetchall()
    assert [row["id"] for row in old_versions] == [result.old_version_id]
    other_binding = VisualIdentityRepository(db).get_active_actor_pack(
        "character", environment["other_actor_id"]
    )
    assert other_binding is not None
    assert other_binding["pack"]["id"] == result.old_pack_id
    assert other_binding["version"]["id"] == result.old_version_id
    package_after = {
        path.relative_to(environment["package_root"]): path.read_bytes()
        for path in environment["package_root"].rglob("*")
        if path.is_file()
    }
    assert package_after == package_before


def test_later_save_appends_exactly_one_immutable_version(
    publication_environment,
) -> None:
    environment = publication_environment
    db = environment["db"]
    first = create_visual_identity_candidate(
        db, actor_kind="character", actor_id=environment["actor_id"]
    )
    first.stage_replacement("thinking", _png_bytes((21, 22, 23)))
    first_result = publish_visual_identity_candidate(
        db, first, user_data_dir=environment["user_root"]
    )
    second = create_visual_identity_candidate(
        db, actor_kind="character", actor_id=environment["actor_id"]
    )
    generated = _png_bytes((31, 32, 33))
    second.stage_replacement("thinking", generated, source="generated")
    second_result = publish_visual_identity_candidate(
        db, second, user_data_dir=environment["user_root"]
    )

    assert second_result.new_pack_id == first_result.new_pack_id
    assert second_result.old_version_id == first_result.new_version_id
    versions = db.execute_query(
        """
        SELECT id, version_number FROM visual_identity_pack_versions
         WHERE pack_id = ? ORDER BY version_number
        """,
        (first_result.new_pack_id,),
    ).fetchall()
    assert [(row["id"], row["version_number"]) for row in versions] == [
        (first_result.new_version_id, 1),
        (second_result.new_version_id, 2),
    ]
    assert Path(first_result.version_directory).is_dir()
    assert Path(second_result.version_directory).is_dir()
    assert first_result.version_directory != second_result.version_directory
    resolved = resolve_visual_identity(
        db,
        actor_kind="character",
        actor_id=environment["actor_id"],
        requested_state="thinking",
        user_data_dir=environment["user_root"],
    )
    assert resolved.image_bytes == generated


def test_cleared_expression_is_omitted_and_resolves_through_pack_default(
    publication_environment,
) -> None:
    environment = publication_environment
    candidate = create_visual_identity_candidate(
        environment["db"],
        actor_kind="character",
        actor_id=environment["actor_id"],
    )
    candidate.stage_clear("thinking")
    result = publish_visual_identity_candidate(
        environment["db"], candidate, user_data_dir=environment["user_root"]
    )

    graph = VisualIdentityRepository(environment["db"]).get_active_actor_pack(
        "character", environment["actor_id"]
    )
    assert graph is not None
    assert result.new_version_id == graph["version"]["id"]
    assert "thinking" not in {asset["expression_key"] for asset in graph["assets"]}
    resolved = resolve_visual_identity(
        environment["db"],
        actor_kind="character",
        actor_id=environment["actor_id"],
        requested_state="thinking",
        user_data_dir=environment["user_root"],
    )
    assert resolved.resolved_expression_key == "neutral"
    assert resolved.resolution_source == "pack_default"
    assert resolved.fallback_reason == "requested_unavailable"


def test_cancel_input_and_validation_failures_preserve_active_graph(
    publication_environment,
) -> None:
    environment = publication_environment
    before = _active_identity(environment)
    cancelled = create_visual_identity_candidate(
        environment["db"], actor_kind="character", actor_id=environment["actor_id"]
    )
    cancelled.stage_replacement("thinking", _png_bytes((1, 1, 1)))
    cancelled.cancel()
    with pytest.raises(
        VisualIdentityPublicationError, match="visual_identity_candidate_cancelled"
    ):
        publish_visual_identity_candidate(
            environment["db"],
            cancelled,
            user_data_dir=environment["user_root"],
        )

    invalid_input = create_visual_identity_candidate(
        environment["db"], actor_kind="character", actor_id=environment["actor_id"]
    )
    with pytest.raises(ValueError, match="visual_identity_expression_not_found"):
        invalid_input.stage_clear("custom:missing")

    invalid_bytes = create_visual_identity_candidate(
        environment["db"], actor_kind="character", actor_id=environment["actor_id"]
    )
    invalid_bytes.stage_replacement("thinking", b"not-an-image")
    with pytest.raises(
        VisualIdentityPublicationError, match="visual_identity_candidate_invalid"
    ):
        publish_visual_identity_candidate(
            environment["db"],
            invalid_bytes,
            user_data_dir=environment["user_root"],
        )

    assert _active_identity(environment) == before
    assert not list(environment["user_root"].rglob("manifest.json"))


def test_permission_or_atomic_replace_failure_preserves_active_and_cleans_staging(
    publication_environment,
) -> None:
    environment = publication_environment
    before = _active_identity(environment)
    candidate = create_visual_identity_candidate(
        environment["db"], actor_kind="character", actor_id=environment["actor_id"]
    )
    candidate.stage_replacement("thinking", _png_bytes((2, 2, 2)))

    def denied_replace(_source, _destination):
        raise PermissionError("private denied path")

    with pytest.raises(
        VisualIdentityPublicationError, match="visual_identity_publication_denied"
    ):
        publish_visual_identity_candidate(
            environment["db"],
            candidate,
            user_data_dir=environment["user_root"],
            atomic_replace=denied_replace,
        )

    assert _active_identity(environment) == before
    assert not list(environment["user_root"].rglob(".staging-*"))
    assert not list(environment["user_root"].rglob("manifest.json"))


def test_database_failure_leaves_only_known_unreferenced_cleanup_candidate(
    publication_environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = publication_environment
    before = _active_identity(environment)
    candidate = create_visual_identity_candidate(
        environment["db"], actor_kind="character", actor_id=environment["actor_id"]
    )
    candidate.stage_replacement("thinking", _png_bytes((3, 3, 3)))

    def fail_database(*_args, **_kwargs):
        raise RuntimeError("private database detail")

    monkeypatch.setattr(VisualIdentityRepository, "activate_pack", fail_database)
    with pytest.raises(
        VisualIdentityPublicationError, match="visual_identity_database_failed"
    ) as caught:
        publish_visual_identity_candidate(
            environment["db"],
            candidate,
            user_data_dir=environment["user_root"],
        )

    assert _active_identity(environment) == before
    cleanup_candidate = (
        environment["user_root"]
        / "visual_identities"
        / Path(caught.value.cleanup_candidate_relpath)
    )
    assert cleanup_candidate.is_dir()
    manifests = list(environment["user_root"].rglob("manifest.json"))
    assert manifests == [cleanup_candidate / "manifest.json"]
    assert not list(environment["user_root"].rglob(".staging-*"))


def test_package_resource_root_is_never_an_eligible_publication_root(
    publication_environment,
) -> None:
    environment = publication_environment
    before = _active_identity(environment)
    package_before = {
        path.relative_to(environment["package_root"]): path.read_bytes()
        for path in environment["package_root"].rglob("*")
        if path.is_file()
    }
    candidate = create_visual_identity_candidate(
        environment["db"], actor_kind="character", actor_id=environment["actor_id"]
    )
    candidate.stage_replacement("thinking", _png_bytes((4, 4, 4)))

    with pytest.raises(
        VisualIdentityPublicationError,
        match="visual_identity_package_root_immutable",
    ):
        publish_visual_identity_candidate(
            environment["db"],
            candidate,
            user_data_dir=environment["package_root"],
        )

    assert _active_identity(environment) == before
    package_after = {
        path.relative_to(environment["package_root"]): path.read_bytes()
        for path in environment["package_root"].rglob("*")
        if path.is_file()
    }
    assert package_after == package_before
    assert not (environment["package_root"] / "visual_identities").exists()
