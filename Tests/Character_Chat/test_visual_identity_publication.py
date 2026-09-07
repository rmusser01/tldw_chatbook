"""Real filesystem/database coverage for immutable Visual Identity publication."""

from __future__ import annotations

import errno
import hashlib
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

import tldw_chatbook.Character_Chat.visual_identity as visual_identity_module
from tldw_chatbook.Character_Chat.visual_identity import (
    MAX_EXPRESSION_ASSET_BYTES,
    VisualIdentityPublicationError,
    create_visual_identity_candidate,
    publish_visual_identity_candidate,
    resolve_visual_identity,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError
from tldw_chatbook.DB.VisualIdentity_DB import VisualIdentityRepository
from tldw_chatbook.Utils.private_paths import PrivatePathResult, PrivatePathStatus


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    stream = BytesIO()
    Image.new("RGB", (16, 16), color).save(stream, format="PNG")
    return stream.getvalue()


def _bmp_bytes(color: tuple[int, int, int]) -> bytes:
    stream = BytesIO()
    Image.new("RGB", (16, 16), color).save(stream, format="BMP")
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

    def replace_spy(source, destination, *, src_dir_fd, dst_dir_fd):
        assert isinstance(source, str) and isinstance(destination, str)
        assert src_dir_fd == dst_dir_fd
        replace_calls.append((Path(source), Path(destination)))
        os.replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

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

    def denied_replace(_source, _destination, **_kwargs):
        raise PermissionError("private denied path")

    with pytest.raises(
        VisualIdentityPublicationError, match="visual_identity_publication_denied"
    ) as caught:
        publish_visual_identity_candidate(
            environment["db"],
            candidate,
            user_data_dir=environment["user_root"],
            atomic_replace=denied_replace,
        )

    assert _active_identity(environment) == before
    assert caught.value.cleanup_candidate_relpath is None
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


def test_cancel_during_materialization_wins_before_atomic_commit(
    publication_environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = publication_environment
    before = _active_identity(environment)
    candidate = create_visual_identity_candidate(
        environment["db"], actor_kind="character", actor_id=environment["actor_id"]
    )
    candidate.stage_replacement("thinking", _png_bytes((5, 5, 5)))
    entered = threading.Event()
    release = threading.Event()
    original = visual_identity_module._materialize_visual_identity_candidate

    def blocked_materialize(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        visual_identity_module,
        "_materialize_visual_identity_candidate",
        blocked_materialize,
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            publish_visual_identity_candidate,
            environment["db"],
            candidate,
            user_data_dir=environment["user_root"],
        )
        assert entered.wait(5)
        with pytest.raises(
            VisualIdentityPublicationError,
            match="visual_identity_candidate_publishing",
        ):
            candidate.stage_replacement("neutral", _png_bytes((6, 6, 6)))
        candidate.cancel()
        release.set()
        with pytest.raises(
            VisualIdentityPublicationError,
            match="visual_identity_candidate_cancelled",
        ):
            future.result(timeout=5)

    assert _active_identity(environment) == before
    assert not list(environment["user_root"].rglob("manifest.json"))


def test_cleanup_unlinks_substituted_staging_name_without_deleting_target(
    publication_environment,
) -> None:
    environment = publication_environment
    before = _active_identity(environment)
    candidate = create_visual_identity_candidate(
        environment["db"], actor_kind="character", actor_id=environment["actor_id"]
    )
    candidate.stage_replacement("thinking", _png_bytes((7, 7, 7)))

    def substitute_then_deny(source, destination, *, src_dir_fd=None, dst_dir_fd=None):
        del destination, dst_dir_fd
        if src_dir_fd is None:
            root = Path(source).parent
            source_name = Path(source).name
            os.rename(root / source_name, root / ".detached-candidate")
            active = root / "active-version"
            active.mkdir()
            (active / "sentinel").write_text("active", encoding="utf-8")
            os.symlink(active.name, root / source_name, target_is_directory=True)
        else:
            source_name = str(source)
            os.rename(
                source_name,
                ".detached-candidate",
                src_dir_fd=src_dir_fd,
                dst_dir_fd=src_dir_fd,
            )
            os.mkdir("active-version", dir_fd=src_dir_fd)
            active_fd = os.open(
                "active-version",
                os.O_RDONLY | os.O_DIRECTORY,
                dir_fd=src_dir_fd,
            )
            try:
                sentinel_fd = os.open(
                    "sentinel",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=active_fd,
                )
                os.write(sentinel_fd, b"active")
                os.close(sentinel_fd)
            finally:
                os.close(active_fd)
            os.symlink("active-version", source_name, dir_fd=src_dir_fd)
        raise PermissionError("denied after substitution")

    with pytest.raises(
        VisualIdentityPublicationError, match="visual_identity_publication_denied"
    ):
        publish_visual_identity_candidate(
            environment["db"],
            candidate,
            user_data_dir=environment["user_root"],
            atomic_replace=substitute_then_deny,
        )

    assert _active_identity(environment) == before
    sentinels = list(environment["user_root"].rglob("active-version/sentinel"))
    assert len(sentinels) == 1
    assert sentinels[0].read_text(encoding="utf-8") == "active"


def test_parent_substitution_is_detected_without_writing_attacker_directory(
    publication_environment,
) -> None:
    environment = publication_environment
    before = _active_identity(environment)
    candidate = create_visual_identity_candidate(
        environment["db"], actor_kind="character", actor_id=environment["actor_id"]
    )
    candidate.stage_replacement("thinking", _png_bytes((15, 15, 15)))

    def substitute_parent_then_replace(source, destination, *, src_dir_fd, dst_dir_fd):
        versions_path = next(environment["user_root"].rglob("versions"))
        detached = versions_path.with_name("versions-detached")
        os.rename(versions_path, detached)
        versions_path.mkdir()
        (versions_path / "attacker-sentinel").write_text("untouched", encoding="utf-8")
        os.replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    with pytest.raises(
        VisualIdentityPublicationError, match="visual_identity_publication_denied"
    ) as caught:
        publish_visual_identity_candidate(
            environment["db"],
            candidate,
            user_data_dir=environment["user_root"],
            atomic_replace=substitute_parent_then_replace,
        )

    assert _active_identity(environment) == before
    assert caught.value.cleanup_candidate_relpath is None
    sentinel = next(environment["user_root"].rglob("attacker-sentinel"))
    assert sentinel.read_text(encoding="utf-8") == "untouched"
    assert not list(environment["user_root"].rglob("versions-detached/manifest.json"))
    assert not list(
        environment["db"]
        .execute_query(
            "SELECT id FROM visual_identity_packs WHERE source_kind = 'manual'"
        )
        .fetchall()
    )


@pytest.mark.parametrize("package_position", ["equal", "descendant"])
def test_publication_rejects_package_and_asset_root_overlap(
    publication_environment,
    monkeypatch: pytest.MonkeyPatch,
    package_position: str,
) -> None:
    environment = publication_environment
    profile_root = environment["user_root"]
    assets_root = profile_root / "visual_identities"
    package_root = (
        assets_root if package_position == "equal" else assets_root / "package"
    )
    package_root.mkdir(parents=True)
    monkeypatch.setattr(
        visual_identity_module.resources, "files", lambda _package: package_root
    )
    candidate = create_visual_identity_candidate(
        environment["db"], actor_kind="character", actor_id=environment["actor_id"]
    )
    candidate.stage_replacement("thinking", _png_bytes((8, 8, 8)))

    with pytest.raises(
        VisualIdentityPublicationError,
        match="visual_identity_package_root_immutable",
    ):
        publish_visual_identity_candidate(
            environment["db"], candidate, user_data_dir=profile_root
        )


def test_binding_delete_recreate_aba_is_rejected_after_filesystem_commit(
    publication_environment,
) -> None:
    environment = publication_environment
    db = environment["db"]
    candidate = create_visual_identity_candidate(
        db, actor_kind="character", actor_id=environment["actor_id"]
    )
    candidate.stage_replacement("thinking", _png_bytes((9, 9, 9)))

    def replace_then_aba(source, destination, **kwargs):
        os.replace(source, destination, **kwargs)
        with db.transaction():
            db.execute_query(
                "DELETE FROM visual_identity_bindings WHERE id = ?",
                (candidate.old_binding_id,),
            )
            db.execute_query(
                """
                INSERT INTO visual_identity_bindings(
                    owner_user_id, actor_kind, actor_id, pack_id, active_version_id
                ) VALUES (0, 'character', ?, ?, ?)
                """,
                (
                    candidate.actor_id,
                    candidate.old_pack_id,
                    candidate.old_version_id,
                ),
            )

    with pytest.raises(
        VisualIdentityPublicationError, match="visual_identity_binding_changed"
    ) as caught:
        publish_visual_identity_candidate(
            db,
            candidate,
            user_data_dir=environment["user_root"],
            atomic_replace=replace_then_aba,
        )

    assert caught.value.cleanup_candidate_relpath is not None
    manual_packs = db.execute_query(
        "SELECT id FROM visual_identity_packs WHERE source_kind = 'manual'"
    ).fetchall()
    assert manual_packs == []


def test_manual_version_binding_delete_recreate_aba_is_rejected(
    publication_environment,
) -> None:
    environment = publication_environment
    db = environment["db"]
    first = create_visual_identity_candidate(
        db, actor_kind="character", actor_id=environment["actor_id"]
    )
    first.stage_replacement("thinking", _png_bytes((16, 16, 16)))
    first_result = publish_visual_identity_candidate(
        db, first, user_data_dir=environment["user_root"]
    )
    candidate = create_visual_identity_candidate(
        db, actor_kind="character", actor_id=environment["actor_id"]
    )
    candidate.stage_replacement("thinking", _png_bytes((17, 17, 17)))

    def replace_then_aba(source, destination, **kwargs):
        os.replace(source, destination, **kwargs)
        with db.transaction():
            db.execute_query(
                "DELETE FROM visual_identity_bindings WHERE id = ?",
                (candidate.old_binding_id,),
            )
            db.execute_query(
                """
                INSERT INTO visual_identity_bindings(
                    owner_user_id, actor_kind, actor_id, pack_id, active_version_id
                ) VALUES (0, 'character', ?, ?, ?)
                """,
                (
                    candidate.actor_id,
                    candidate.old_pack_id,
                    candidate.old_version_id,
                ),
            )

    with pytest.raises(
        VisualIdentityPublicationError, match="visual_identity_binding_changed"
    ):
        publish_visual_identity_candidate(
            db,
            candidate,
            user_data_dir=environment["user_root"],
            atomic_replace=replace_then_aba,
        )

    versions = db.execute_query(
        "SELECT version_number FROM visual_identity_pack_versions WHERE pack_id = ?",
        (first_result.new_pack_id,),
    ).fetchall()
    assert [row["version_number"] for row in versions] == [1]


def test_competing_binding_winner_keeps_stable_conflict_category(
    publication_environment,
) -> None:
    environment = publication_environment
    db = environment["db"]
    repository = VisualIdentityRepository(db)
    competitor_assets = [
        _asset(
            environment["package_root"],
            expression_key=key,
            original_label=label,
            color=color,
        )
        for key, label, color in (
            ("neutral", "competitor-neutral", (13, 13, 13)),
            ("thinking", "competitor-thinking", (14, 14, 14)),
        )
    ]
    competitor = repository.activate_pack(
        pack={
            "title": "Competitor",
            "description": "Wins the race",
            "default_expression_key": "neutral",
            "source_kind": "builtin",
            "source_context": {},
        },
        manifest={"fixture": "competitor"},
        assets=competitor_assets,
        actor_kind="character",
        actor_id=environment["other_actor_id"],
    )
    candidate = create_visual_identity_candidate(
        db, actor_kind="character", actor_id=environment["actor_id"]
    )
    candidate.stage_replacement("thinking", _png_bytes((10, 10, 10)))

    def replace_then_compete(source, destination, **kwargs):
        os.replace(source, destination, **kwargs)
        with db.transaction():
            db.execute_query(
                """
                UPDATE visual_identity_bindings
                   SET pack_id = ?, active_version_id = ?, version = version + 1
                 WHERE id = ?
                """,
                (
                    competitor["pack"]["id"],
                    competitor["version"]["id"],
                    candidate.old_binding_id,
                ),
            )

    with pytest.raises(
        VisualIdentityPublicationError, match="visual_identity_binding_changed"
    ) as caught:
        publish_visual_identity_candidate(
            db,
            candidate,
            user_data_dir=environment["user_root"],
            atomic_replace=replace_then_compete,
        )

    assert caught.value.cleanup_candidate_relpath is not None
    active = repository.get_active_actor_pack("character", environment["actor_id"])
    assert active is not None
    assert active["pack"]["id"] == competitor["pack"]["id"]


def test_clearing_default_selects_a_retained_default_and_can_publish(
    publication_environment,
) -> None:
    environment = publication_environment
    candidate = create_visual_identity_candidate(
        environment["db"], actor_kind="character", actor_id=environment["actor_id"]
    )

    candidate.stage_clear("neutral")

    assert candidate.default_expression_key != "neutral"
    result = publish_visual_identity_candidate(
        environment["db"], candidate, user_data_dir=environment["user_root"]
    )
    graph = VisualIdentityRepository(environment["db"]).get_active_actor_pack(
        "character", environment["actor_id"]
    )
    assert graph is not None
    retained = {asset["expression_key"] for asset in graph["assets"]}
    assert graph["version"]["default_expression_key"] in retained
    assert result.new_version_id == graph["version"]["id"]


def test_clearing_last_retained_asset_is_rejected_immediately(
    publication_environment,
) -> None:
    environment = publication_environment
    candidate = create_visual_identity_candidate(
        environment["db"], actor_kind="character", actor_id=environment["actor_id"]
    )
    keys = [str(asset["expression_key"]) for asset in candidate.assets]
    for key in keys[:-1]:
        candidate.stage_clear(key)

    with pytest.raises(
        VisualIdentityPublicationError, match="visual_identity_candidate_empty"
    ):
        candidate.stage_clear(keys[-1])

    assert keys[-1] not in candidate.cleared_expression_keys


def test_projected_replacement_budget_rejects_eleventh_large_asset_without_retaining_it(
    publication_environment,
) -> None:
    environment = publication_environment
    db = environment["db"]
    assets = [
        _asset(
            environment["package_root"],
            expression_key=f"custom:bulk{index}",
            original_label=f"bulk-{index}",
            color=(index, index, index),
        )
        for index in range(11)
    ]
    VisualIdentityRepository(db).activate_pack(
        pack={
            "title": "Budget fixture",
            "description": "Eleven assets",
            "default_expression_key": "custom:bulk0",
            "source_kind": "builtin",
            "source_context": {},
        },
        manifest={"fixture": "budget"},
        assets=assets,
        actor_kind="character",
        actor_id=environment["actor_id"],
    )
    candidate = create_visual_identity_candidate(
        db, actor_kind="character", actor_id=environment["actor_id"]
    )
    large = b"x" * MAX_EXPRESSION_ASSET_BYTES
    for index in range(10):
        candidate.stage_replacement(f"custom:bulk{index}", large)

    with pytest.raises(
        VisualIdentityPublicationError, match="visual_identity_budget_exceeded"
    ):
        candidate.stage_replacement("custom:bulk10", large)

    assert "custom:bulk10" not in candidate.replaced_expression_keys
    assert len(candidate.replaced_expression_keys) == 10


def test_materialization_stops_as_soon_as_total_byte_budget_is_exceeded(
    publication_environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = publication_environment
    candidate = create_visual_identity_candidate(
        environment["db"], actor_kind="character", actor_id=environment["actor_id"]
    )
    candidate.stage_replacement("thinking", _png_bytes((11, 11, 11)))
    original_load = visual_identity_module.load_visual_identity_asset
    load_calls = 0

    def counted_load(*args, **kwargs):
        nonlocal load_calls
        load_calls += 1
        return original_load(*args, **kwargs)

    monkeypatch.setattr(visual_identity_module, "MAX_EXPRESSION_TOTAL_BYTES", 150)
    monkeypatch.setattr(
        visual_identity_module, "load_visual_identity_asset", counted_load
    )
    with pytest.raises(
        VisualIdentityPublicationError, match="visual_identity_budget_exceeded"
    ):
        publish_visual_identity_candidate(
            environment["db"], candidate, user_data_dir=environment["user_root"]
        )

    assert load_calls < len(candidate.assets)


def test_initial_database_failure_is_normalized_without_private_detail(
    publication_environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = publication_environment
    candidate = create_visual_identity_candidate(
        environment["db"], actor_kind="character", actor_id=environment["actor_id"]
    )
    candidate.stage_replacement("thinking", _png_bytes((12, 12, 12)))

    def fail_read(*_args, **_kwargs):
        raise CharactersRAGDBError("private /Users/name/profile.db")

    monkeypatch.setattr(VisualIdentityRepository, "get_active_actor_pack", fail_read)
    with pytest.raises(
        VisualIdentityPublicationError, match="^visual_identity_database_failed$"
    ) as caught:
        publish_visual_identity_candidate(
            environment["db"], candidate, user_data_dir=environment["user_root"]
        )

    assert "Users" not in str(caught.value)
    assert caught.value.cleanup_candidate_relpath is None


def test_profile_pack_ancestor_alias_outside_profile_is_rejected(
    publication_environment,
) -> None:
    environment = publication_environment
    before = _active_identity(environment)
    candidate = create_visual_identity_candidate(
        environment["db"], actor_kind="character", actor_id=environment["actor_id"]
    )
    candidate.stage_replacement("thinking", _png_bytes((18, 18, 18)))

    def escape_pack_then_replace(source, destination, **kwargs):
        pack_dir = next(environment["user_root"].rglob("profile-*"))
        escaped = environment["user_root"].parent / "escaped-profile-pack"
        os.rename(pack_dir, escaped)
        os.symlink(escaped, pack_dir, target_is_directory=True)
        os.replace(source, destination, **kwargs)

    with pytest.raises(
        VisualIdentityPublicationError, match="visual_identity_publication_denied"
    ):
        publish_visual_identity_candidate(
            environment["db"],
            candidate,
            user_data_dir=environment["user_root"],
            atomic_replace=escape_pack_then_replace,
        )

    assert _active_identity(environment) == before
    assert (
        not environment["db"]
        .execute_query(
            "SELECT id FROM visual_identity_packs WHERE source_kind = 'manual'"
        )
        .fetchall()
    )


@pytest.mark.parametrize("replacement_kind", ["directory", "symlink"])
def test_swapped_final_entry_is_rejected_without_touching_replacement(
    publication_environment, replacement_kind: str
) -> None:
    environment = publication_environment
    before = _active_identity(environment)
    candidate = create_visual_identity_candidate(
        environment["db"], actor_kind="character", actor_id=environment["actor_id"]
    )
    candidate.stage_replacement("thinking", _png_bytes((19, 19, 19)))

    def swap_final(source, destination, *, src_dir_fd, dst_dir_fd):
        os.replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )
        os.rename(
            destination,
            ".detached-final",
            src_dir_fd=dst_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )
        if replacement_kind == "directory":
            os.mkdir(destination, dir_fd=dst_dir_fd)
        else:
            os.mkdir("replacement-target", dir_fd=dst_dir_fd)
            target_fd = os.open(
                "replacement-target",
                os.O_RDONLY | os.O_DIRECTORY,
                dir_fd=dst_dir_fd,
            )
            try:
                sentinel = os.open(
                    "sentinel",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=target_fd,
                )
                os.close(sentinel)
            finally:
                os.close(target_fd)
            os.symlink("replacement-target", destination, dir_fd=dst_dir_fd)

    with pytest.raises(
        VisualIdentityPublicationError, match="visual_identity_publication_denied"
    ):
        publish_visual_identity_candidate(
            environment["db"],
            candidate,
            user_data_dir=environment["user_root"],
            atomic_replace=swap_final,
        )

    assert _active_identity(environment) == before
    if replacement_kind == "symlink":
        assert (
            len(list(environment["user_root"].rglob("replacement-target/sentinel")))
            == 1
        )


def test_initial_profile_ancestor_alias_window_is_rejected(
    publication_environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = publication_environment
    before = _active_identity(environment)
    candidate = create_visual_identity_candidate(
        environment["db"], actor_kind="character", actor_id=environment["actor_id"]
    )
    candidate.stage_replacement("thinking", _png_bytes((20, 20, 20)))
    original_secure = visual_identity_module.secure_private_directory
    swapped = False

    def secure_then_alias(path, *, create, application_owned):
        nonlocal swapped
        result = original_secure(
            path, create=create, application_owned=application_owned
        )
        if Path(path).name == "versions" and not swapped:
            swapped = True
            escaped = environment["user_root"].parent / "escaped-profile-root"
            os.rename(environment["user_root"], escaped)
            os.symlink(
                escaped,
                environment["user_root"],
                target_is_directory=True,
            )
        return result

    monkeypatch.setattr(
        visual_identity_module, "secure_private_directory", secure_then_alias
    )
    with pytest.raises(
        VisualIdentityPublicationError, match="visual_identity_publication_denied"
    ):
        publish_visual_identity_candidate(
            environment["db"], candidate, user_data_dir=environment["user_root"]
        )

    assert _active_identity(environment) == before


def test_initial_sqlite_error_is_private_and_resets_publication_state(
    publication_environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = publication_environment
    candidate = create_visual_identity_candidate(
        environment["db"], actor_kind="character", actor_id=environment["actor_id"]
    )
    candidate.stage_replacement("thinking", _png_bytes((21, 21, 21)))

    def fail_read(*_args, **_kwargs):
        raise sqlite3.OperationalError("private C:\\Users\\name\\profile.db")

    monkeypatch.setattr(VisualIdentityRepository, "get_active_actor_pack", fail_read)
    with pytest.raises(
        VisualIdentityPublicationError, match="^visual_identity_database_failed$"
    ) as caught:
        publish_visual_identity_candidate(
            environment["db"], candidate, user_data_dir=environment["user_root"]
        )

    assert "Users" not in str(caught.value)
    assert candidate._publishing is False


def test_post_filesystem_sqlite_error_reports_orphan_and_resets_state(
    publication_environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = publication_environment
    before = _active_identity(environment)
    candidate = create_visual_identity_candidate(
        environment["db"], actor_kind="character", actor_id=environment["actor_id"]
    )
    candidate.stage_replacement("thinking", _png_bytes((22, 22, 22)))

    def fail_commit(*_args, **_kwargs):
        raise sqlite3.OperationalError("private /Users/name/profile.db")

    monkeypatch.setattr(VisualIdentityRepository, "activate_pack", fail_commit)
    with pytest.raises(
        VisualIdentityPublicationError, match="^visual_identity_database_failed$"
    ) as caught:
        publish_visual_identity_candidate(
            environment["db"], candidate, user_data_dir=environment["user_root"]
        )

    assert _active_identity(environment) == before
    assert "Users" not in str(caught.value)
    assert caught.value.cleanup_candidate_relpath is not None
    assert candidate._publishing is False


def test_second_publisher_is_rejected_without_resetting_survivor(
    publication_environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = publication_environment
    candidate = create_visual_identity_candidate(
        environment["db"], actor_kind="character", actor_id=environment["actor_id"]
    )
    candidate.stage_replacement("thinking", _png_bytes((23, 23, 23)))
    entered = threading.Event()
    release = threading.Event()
    call_lock = threading.Lock()
    calls = 0
    original = visual_identity_module._materialize_visual_identity_candidate

    def block_first(*args, **kwargs):
        nonlocal calls
        with call_lock:
            calls += 1
            current = calls
        if current == 1:
            entered.set()
            assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        visual_identity_module,
        "_materialize_visual_identity_candidate",
        block_first,
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        survivor = executor.submit(
            publish_visual_identity_candidate,
            environment["db"],
            candidate,
            user_data_dir=environment["user_root"],
        )
        assert entered.wait(5)
        try:
            with pytest.raises(
                VisualIdentityPublicationError,
                match="visual_identity_candidate_publishing",
            ):
                publish_visual_identity_candidate(
                    environment["db"],
                    candidate,
                    user_data_dir=environment["user_root"],
                )
        finally:
            release.set()
        result = survivor.result(timeout=5)

    assert result.new_version_id != candidate.old_version_id


def test_replacing_cleared_original_default_restores_it_for_publication(
    publication_environment,
) -> None:
    environment = publication_environment
    candidate = create_visual_identity_candidate(
        environment["db"], actor_kind="character", actor_id=environment["actor_id"]
    )
    candidate.stage_clear("neutral")
    assert candidate.default_expression_key != "neutral"

    replacement = _png_bytes((24, 24, 24))
    candidate.stage_replacement("neutral", replacement)

    assert candidate.default_expression_key == "neutral"
    publish_visual_identity_candidate(
        environment["db"], candidate, user_data_dir=environment["user_root"]
    )
    resolved = resolve_visual_identity(
        environment["db"],
        actor_kind="character",
        actor_id=environment["actor_id"],
        requested_state="custom:missing",
        user_data_dir=environment["user_root"],
    )
    assert resolved.resolved_expression_key == "neutral"
    assert resolved.image_bytes == replacement


def test_forced_unverified_platform_fallback_uses_paths_and_usable_privacy(
    publication_environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = publication_environment
    candidate = create_visual_identity_candidate(
        environment["db"], actor_kind="character", actor_id=environment["actor_id"]
    )
    candidate.stage_replacement("thinking", _png_bytes((25, 25, 25)))
    original_secure = visual_identity_module.secure_private_directory

    def unverified_secure(path, *, create, application_owned):
        result = original_secure(
            path, create=create, application_owned=application_owned
        )
        return PrivatePathResult(
            result.lexical_path,
            PrivatePathStatus.UNVERIFIED_PLATFORM,
            reason="forced-fallback",
        )

    replace_calls: list[tuple[Path, Path]] = []

    def path_replace(source, destination):
        assert isinstance(source, Path) and isinstance(destination, Path)
        replace_calls.append((source, destination))
        os.replace(source, destination)

    monkeypatch.setattr(
        visual_identity_module, "_publication_posix_guards_available", lambda: False
    )
    monkeypatch.setattr(
        visual_identity_module, "secure_private_directory", unverified_secure
    )
    result = publish_visual_identity_candidate(
        environment["db"],
        candidate,
        user_data_dir=environment["user_root"],
        atomic_replace=path_replace,
    )

    assert len(replace_calls) == 1
    assert result.version_directory.is_dir()


def test_forced_fallback_rejects_swapped_final_symlink_without_touching_target(
    publication_environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = publication_environment
    before = _active_identity(environment)
    candidate = create_visual_identity_candidate(
        environment["db"], actor_kind="character", actor_id=environment["actor_id"]
    )
    candidate.stage_replacement("thinking", _png_bytes((26, 25, 25)))
    replacement_links: list[Path] = []

    def swap_final(source: Path, destination: Path) -> None:
        os.replace(source, destination)
        os.rename(destination, destination.with_name(".detached-final"))
        target = destination.with_name("replacement-target")
        target.mkdir()
        (target / "sentinel").write_bytes(b"replacement")
        destination.symlink_to(target, target_is_directory=True)
        replacement_links.append(destination)

    monkeypatch.setattr(
        visual_identity_module, "_publication_posix_guards_available", lambda: False
    )
    with pytest.raises(
        VisualIdentityPublicationError, match="visual_identity_publication_denied"
    ):
        publish_visual_identity_candidate(
            environment["db"],
            candidate,
            user_data_dir=environment["user_root"],
            atomic_replace=swap_final,
        )

    assert _active_identity(environment) == before
    target = next(environment["user_root"].rglob("replacement-target"))
    assert (target / "sentinel").read_bytes() == b"replacement"
    assert replacement_links[0].is_symlink()


@pytest.mark.skipif(os.name != "nt", reason="Windows-native fallback contract")
def test_windows_native_publication_uses_supported_path_operations(
    publication_environment,
) -> None:
    environment = publication_environment
    candidate = create_visual_identity_candidate(
        environment["db"], actor_kind="character", actor_id=environment["actor_id"]
    )
    candidate.stage_replacement("thinking", _png_bytes((26, 26, 26)))

    result = publish_visual_identity_candidate(
        environment["db"], candidate, user_data_dir=environment["user_root"]
    )

    assert result.version_directory.is_dir()


def test_directory_syncs_bracket_atomic_replace_before_database_commit(
    publication_environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = publication_environment
    candidate = create_visual_identity_candidate(
        environment["db"], actor_kind="character", actor_id=environment["actor_id"]
    )
    candidate.stage_replacement("thinking", _png_bytes((27, 27, 27)))
    events: list[str] = []
    original_activate = VisualIdentityRepository.activate_pack

    def sync_spy(_directory):
        events.append("sync")

    def replace_spy(source, destination, **kwargs):
        events.append("replace")
        os.replace(source, destination, **kwargs)

    def activate_spy(self, *args, **kwargs):
        events.append("db")
        return original_activate(self, *args, **kwargs)

    monkeypatch.setattr(visual_identity_module, "_sync_publication_directory", sync_spy)
    monkeypatch.setattr(VisualIdentityRepository, "activate_pack", activate_spy)
    publish_visual_identity_candidate(
        environment["db"],
        candidate,
        user_data_dir=environment["user_root"],
        atomic_replace=replace_spy,
    )

    assert events == ["sync", "replace", "sync", "db"]


def test_publish_rejects_caller_transaction_before_candidate_or_filesystem_work(
    publication_environment,
) -> None:
    environment = publication_environment
    candidate = create_visual_identity_candidate(
        environment["db"], actor_kind="character", actor_id=environment["actor_id"]
    )
    candidate.stage_replacement("thinking", _png_bytes((28, 28, 28)))

    with (
        pytest.raises(
            VisualIdentityPublicationError,
            match="^visual_identity_transaction_active$",
        ),
        environment["db"].transaction(),
    ):
        publish_visual_identity_candidate(
            environment["db"],
            candidate,
            user_data_dir=environment["user_root"],
        )

    assert candidate._publishing is False
    assert candidate._published is False
    assert not (environment["user_root"] / "visual_identities").exists()


def test_unsupported_valid_image_format_is_cleaned_and_candidate_is_reusable(
    publication_environment,
) -> None:
    environment = publication_environment
    candidate = create_visual_identity_candidate(
        environment["db"], actor_kind="character", actor_id=environment["actor_id"]
    )
    candidate.stage_replacement("thinking", _bmp_bytes((29, 29, 29)))

    with pytest.raises(
        VisualIdentityPublicationError, match="^visual_identity_candidate_invalid$"
    ):
        publish_visual_identity_candidate(
            environment["db"], candidate, user_data_dir=environment["user_root"]
        )

    assert candidate._publishing is False
    assert not list(environment["user_root"].rglob(".staging-*"))
    assert not list(environment["user_root"].rglob("manifest.json"))
    candidate.stage_replacement("thinking", _png_bytes((30, 30, 30)))
    result = publish_visual_identity_candidate(
        environment["db"], candidate, user_data_dir=environment["user_root"]
    )
    assert result.version_directory.is_dir()


@pytest.mark.parametrize("forced_fallback", [False, True])
def test_asset_name_swap_during_bounded_read_is_rejected(
    publication_environment,
    monkeypatch: pytest.MonkeyPatch,
    forced_fallback: bool,
) -> None:
    environment = publication_environment
    before = _active_identity(environment)
    candidate = create_visual_identity_candidate(
        environment["db"], actor_kind="character", actor_id=environment["actor_id"]
    )
    candidate.stage_replacement("thinking", _png_bytes((31, 31, 31)))
    original_read = os.read
    swapped = False
    replace_called = False

    def swap_name_then_read(descriptor: int, count: int) -> bytes:
        nonlocal swapped
        if not swapped:
            staging = next(environment["user_root"].rglob(".staging-*"), None)
            if staging is not None:
                asset = next(
                    path
                    for path in staging.iterdir()
                    if path.is_file() and path.name != "manifest.json"
                )
                os.rename(asset, asset.with_suffix(asset.suffix + ".detached"))
                asset.write_bytes(_png_bytes((32, 32, 32)))
                swapped = True
        return original_read(descriptor, count)

    def track_replace(source, destination, **kwargs):
        nonlocal replace_called
        replace_called = True
        os.replace(source, destination, **kwargs)

    if forced_fallback:
        monkeypatch.setattr(
            visual_identity_module,
            "_publication_posix_guards_available",
            lambda: False,
        )
    monkeypatch.setattr(visual_identity_module.os, "read", swap_name_then_read)
    with pytest.raises(
        VisualIdentityPublicationError, match="visual_identity_candidate_invalid"
    ):
        publish_visual_identity_candidate(
            environment["db"],
            candidate,
            user_data_dir=environment["user_root"],
            atomic_replace=track_replace,
        )

    assert swapped is True
    assert replace_called is False
    assert _active_identity(environment) == before


@pytest.mark.parametrize("forced_fallback", [False, True])
def test_post_rename_fifo_asset_is_rejected_without_blocking(
    publication_environment,
    monkeypatch: pytest.MonkeyPatch,
    forced_fallback: bool,
) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO contract requires os.mkfifo")
    if not forced_fallback and os.mkfifo not in os.supports_dir_fd:
        pytest.skip("descriptor-relative FIFO replacement is unavailable")
    environment = publication_environment
    before = _active_identity(environment)
    candidate = create_visual_identity_candidate(
        environment["db"], actor_kind="character", actor_id=environment["actor_id"]
    )
    candidate.stage_replacement("thinking", _png_bytes((33, 33, 33)))

    def replace_then_fifo(source, destination, **kwargs):
        os.replace(source, destination, **kwargs)
        if kwargs:
            final_fd = os.open(
                str(destination),
                os.O_RDONLY | os.O_DIRECTORY,
                dir_fd=kwargs["dst_dir_fd"],
            )
            try:
                asset_name = next(
                    name for name in os.listdir(final_fd) if name != "manifest.json"
                )
                os.unlink(asset_name, dir_fd=final_fd)
                os.mkfifo(asset_name, 0o600, dir_fd=final_fd)
            finally:
                os.close(final_fd)
        else:
            final_dir = Path(destination)
            asset = next(
                path for path in final_dir.iterdir() if path.name != "manifest.json"
            )
            asset.unlink()
            os.mkfifo(asset, 0o600)

    if forced_fallback:
        monkeypatch.setattr(
            visual_identity_module,
            "_publication_posix_guards_available",
            lambda: False,
        )
    with pytest.raises(
        VisualIdentityPublicationError, match="visual_identity_candidate_invalid"
    ) as caught:
        publish_visual_identity_candidate(
            environment["db"],
            candidate,
            user_data_dir=environment["user_root"],
            atomic_replace=replace_then_fifo,
        )

    if forced_fallback:
        assert caught.value.cleanup_candidate_relpath is None
    else:
        assert caught.value.cleanup_candidate_relpath is not None
    assert _active_identity(environment) == before


def test_forced_fallback_failure_does_not_promise_unverified_cleanup(
    publication_environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = publication_environment
    candidate = create_visual_identity_candidate(
        environment["db"], actor_kind="character", actor_id=environment["actor_id"]
    )
    candidate.stage_replacement("thinking", _png_bytes((34, 34, 34)))

    def deny_replace(_source, _destination):
        raise PermissionError("private fallback path")

    monkeypatch.setattr(
        visual_identity_module, "_publication_posix_guards_available", lambda: False
    )
    with pytest.raises(
        VisualIdentityPublicationError, match="visual_identity_publication_denied"
    ) as caught:
        publish_visual_identity_candidate(
            environment["db"],
            candidate,
            user_data_dir=environment["user_root"],
            atomic_replace=deny_replace,
        )

    assert caught.value.cleanup_candidate_relpath is None
    retained = list(environment["user_root"].rglob(".staging-*"))
    assert len(retained) == 1
    assert (retained[0] / "manifest.json").is_file()


def test_same_binding_row_away_and_back_revision_is_rejected(
    publication_environment,
) -> None:
    environment = publication_environment
    db = environment["db"]
    repository = VisualIdentityRepository(db)
    competitor = repository.activate_pack(
        pack={
            "title": "Revision competitor",
            "source_kind": "builtin",
            "source_context": {},
        },
        manifest={"fixture": "revision-competitor"},
        assets=[
            _asset(
                environment["package_root"],
                expression_key="neutral",
                original_label="revision-neutral",
                color=(35, 35, 35),
            )
        ],
        actor_kind="character",
        actor_id=environment["other_actor_id"],
    )
    candidate = create_visual_identity_candidate(
        db, actor_kind="character", actor_id=environment["actor_id"]
    )
    candidate.stage_replacement("thinking", _png_bytes((36, 36, 36)))

    def replace_then_away_back(source, destination, **kwargs):
        os.replace(source, destination, **kwargs)
        with db.transaction():
            db.execute_query(
                """
                UPDATE visual_identity_bindings
                   SET pack_id = ?, active_version_id = ?, version = version + 1
                 WHERE id = ?
                """,
                (
                    competitor["pack"]["id"],
                    competitor["version"]["id"],
                    candidate.old_binding_id,
                ),
            )
            db.execute_query(
                """
                UPDATE visual_identity_bindings
                   SET pack_id = ?, active_version_id = ?, version = version + 1
                 WHERE id = ?
                """,
                (
                    candidate.old_pack_id,
                    candidate.old_version_id,
                    candidate.old_binding_id,
                ),
            )

    with pytest.raises(
        VisualIdentityPublicationError, match="visual_identity_binding_changed"
    ):
        publish_visual_identity_candidate(
            db,
            candidate,
            user_data_dir=environment["user_root"],
            atomic_replace=replace_then_away_back,
        )


def test_manual_pack_revision_bump_is_rejected_before_append(
    publication_environment,
) -> None:
    environment = publication_environment
    db = environment["db"]
    first = create_visual_identity_candidate(
        db, actor_kind="character", actor_id=environment["actor_id"]
    )
    first.stage_replacement("thinking", _png_bytes((37, 37, 37)))
    first_result = publish_visual_identity_candidate(
        db, first, user_data_dir=environment["user_root"]
    )
    candidate = create_visual_identity_candidate(
        db, actor_kind="character", actor_id=environment["actor_id"]
    )
    candidate.stage_replacement("thinking", _png_bytes((38, 38, 38)))

    def replace_then_bump_pack(source, destination, **kwargs):
        os.replace(source, destination, **kwargs)
        with db.transaction():
            db.execute_query(
                "UPDATE visual_identity_packs SET version = version + 1 WHERE id = ?",
                (candidate.old_pack_id,),
            )

    with pytest.raises(
        VisualIdentityPublicationError, match="visual_identity_binding_changed"
    ):
        publish_visual_identity_candidate(
            db,
            candidate,
            user_data_dir=environment["user_root"],
            atomic_replace=replace_then_bump_pack,
        )

    versions = db.execute_query(
        "SELECT id FROM visual_identity_pack_versions WHERE pack_id = ?",
        (first_result.new_pack_id,),
    ).fetchall()
    assert len(versions) == 1


def test_shared_manual_pack_save_forks_target_and_preserves_other_actor(
    publication_environment,
) -> None:
    environment = publication_environment
    db = environment["db"]
    first_bytes = _png_bytes((39, 39, 39))
    first = create_visual_identity_candidate(
        db, actor_kind="character", actor_id=environment["actor_id"]
    )
    first.stage_replacement("thinking", first_bytes)
    first_result = publish_visual_identity_candidate(
        db, first, user_data_dir=environment["user_root"]
    )
    with db.transaction():
        db.execute_query(
            """
            UPDATE visual_identity_bindings
               SET pack_id = ?, active_version_id = ?, version = version + 1
             WHERE actor_kind = 'character' AND actor_id = ? AND status = 'active'
            """,
            (
                first_result.new_pack_id,
                first_result.new_version_id,
                str(environment["other_actor_id"]),
            ),
        )

    target_bytes = _png_bytes((40, 40, 40))
    candidate = create_visual_identity_candidate(
        db, actor_kind="character", actor_id=environment["actor_id"]
    )
    candidate.stage_replacement("thinking", target_bytes)
    result = publish_visual_identity_candidate(
        db, candidate, user_data_dir=environment["user_root"]
    )

    assert result.new_pack_id != first_result.new_pack_id
    target = resolve_visual_identity(
        db,
        actor_kind="character",
        actor_id=environment["actor_id"],
        requested_state="thinking",
        user_data_dir=environment["user_root"],
    )
    other = resolve_visual_identity(
        db,
        actor_kind="character",
        actor_id=environment["other_actor_id"],
        requested_state="thinking",
        user_data_dir=environment["user_root"],
    )
    other_graph = VisualIdentityRepository(db).get_active_actor_pack(
        "character", environment["other_actor_id"]
    )
    assert target.image_bytes == target_bytes
    assert other.image_bytes == first_bytes
    assert other_graph is not None
    assert other_graph["pack"]["active_version_id"] == first_result.new_version_id
    assert other_graph["version"]["id"] == first_result.new_version_id


def test_binding_added_during_manual_append_forces_conflict(
    publication_environment,
) -> None:
    environment = publication_environment
    db = environment["db"]
    first = create_visual_identity_candidate(
        db, actor_kind="character", actor_id=environment["actor_id"]
    )
    first.stage_replacement("thinking", _png_bytes((40, 41, 40)))
    first_result = publish_visual_identity_candidate(
        db, first, user_data_dir=environment["user_root"]
    )
    candidate = create_visual_identity_candidate(
        db, actor_kind="character", actor_id=environment["actor_id"]
    )
    candidate.stage_replacement("thinking", _png_bytes((40, 42, 40)))

    def replace_then_share(source, destination, **kwargs):
        os.replace(source, destination, **kwargs)
        with db.transaction():
            db.execute_query(
                """
                UPDATE visual_identity_bindings
                   SET pack_id = ?, active_version_id = ?, version = version + 1
                 WHERE actor_kind = 'character' AND actor_id = ? AND status = 'active'
                """,
                (
                    first_result.new_pack_id,
                    first_result.new_version_id,
                    str(environment["other_actor_id"]),
                ),
            )

    with pytest.raises(
        VisualIdentityPublicationError, match="visual_identity_binding_changed"
    ):
        publish_visual_identity_candidate(
            db,
            candidate,
            user_data_dir=environment["user_root"],
            atomic_replace=replace_then_share,
        )
    versions = db.execute_query(
        "SELECT id FROM visual_identity_pack_versions WHERE pack_id = ?",
        (first_result.new_pack_id,),
    ).fetchall()
    assert len(versions) == 1


def test_cleanup_removes_multiple_unreferenced_publication_candidates(
    publication_environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = publication_environment
    db = environment["db"]

    def fail_database(*_args, **_kwargs):
        raise sqlite3.OperationalError("private profile path")

    monkeypatch.setattr(VisualIdentityRepository, "activate_pack", fail_database)
    cleanup_relpaths: list[str] = []
    for color in ((41, 41, 41), (42, 42, 42)):
        candidate = create_visual_identity_candidate(
            db, actor_kind="character", actor_id=environment["actor_id"]
        )
        candidate.stage_replacement("thinking", _png_bytes(color))
        with pytest.raises(VisualIdentityPublicationError) as caught:
            publish_visual_identity_candidate(
                db, candidate, user_data_dir=environment["user_root"]
            )
        assert caught.value.cleanup_candidate_relpath is not None
        cleanup_relpaths.append(caught.value.cleanup_candidate_relpath)

    assert len(list(environment["user_root"].rglob("manifest.json"))) == 2
    for relpath in cleanup_relpaths:
        assert visual_identity_module.cleanup_visual_identity_publication_candidate(
            db, relpath, user_data_dir=environment["user_root"]
        )
    assert not list(environment["user_root"].rglob("manifest.json"))


def test_cleanup_refuses_active_referenced_version(
    publication_environment,
) -> None:
    environment = publication_environment
    candidate = create_visual_identity_candidate(
        environment["db"], actor_kind="character", actor_id=environment["actor_id"]
    )
    candidate.stage_replacement("thinking", _png_bytes((43, 43, 43)))
    result = publish_visual_identity_candidate(
        environment["db"], candidate, user_data_dir=environment["user_root"]
    )
    relpath = result.version_directory.relative_to(
        environment["user_root"] / "visual_identities"
    ).as_posix()

    with pytest.raises(
        VisualIdentityPublicationError, match="visual_identity_cleanup_referenced"
    ):
        visual_identity_module.cleanup_visual_identity_publication_candidate(
            environment["db"], relpath, user_data_dir=environment["user_root"]
        )

    assert result.version_directory.is_dir()


@pytest.mark.parametrize(
    "relpath",
    ["../outside", "/absolute", "packs/not-profile/versions/not-a-version"],
)
def test_cleanup_refuses_malformed_or_outside_relpaths(
    publication_environment, relpath: str
) -> None:
    environment = publication_environment
    with pytest.raises(
        VisualIdentityPublicationError, match="visual_identity_cleanup_denied"
    ):
        visual_identity_module.cleanup_visual_identity_publication_candidate(
            environment["db"], relpath, user_data_dir=environment["user_root"]
        )


def test_cleanup_refuses_package_root_and_unverified_fallback(
    publication_environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = publication_environment
    db = environment["db"]
    candidate = create_visual_identity_candidate(
        db, actor_kind="character", actor_id=environment["actor_id"]
    )
    candidate.stage_replacement("thinking", _png_bytes((44, 44, 44)))

    def fail_database(*_args, **_kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(VisualIdentityRepository, "activate_pack", fail_database)
    with pytest.raises(VisualIdentityPublicationError) as caught:
        publish_visual_identity_candidate(
            db, candidate, user_data_dir=environment["user_root"]
        )
    relpath = caught.value.cleanup_candidate_relpath
    assert relpath is not None

    original_guards = visual_identity_module._publication_posix_guards_available
    monkeypatch.setattr(
        visual_identity_module, "_publication_posix_guards_available", lambda: False
    )
    with pytest.raises(
        VisualIdentityPublicationError, match="visual_identity_cleanup_denied"
    ):
        visual_identity_module.cleanup_visual_identity_publication_candidate(
            db, relpath, user_data_dir=environment["user_root"]
        )
    assert (environment["user_root"] / "visual_identities" / relpath).is_dir()

    monkeypatch.setattr(
        visual_identity_module, "_publication_posix_guards_available", original_guards
    )
    with pytest.raises(
        VisualIdentityPublicationError,
        match="visual_identity_package_root_immutable",
    ):
        visual_identity_module.cleanup_visual_identity_publication_candidate(
            db, relpath, user_data_dir=environment["package_root"]
        )


def test_post_rename_parent_fsync_failure_reports_known_cleanup_candidate(
    publication_environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = publication_environment
    before = _active_identity(environment)
    candidate = create_visual_identity_candidate(
        environment["db"], actor_kind="character", actor_id=environment["actor_id"]
    )
    candidate.stage_replacement("thinking", _png_bytes((45, 45, 45)))
    original_sync = visual_identity_module._sync_publication_directory
    calls = 0

    def fail_parent_sync(directory):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError(errno.EIO, "private sync failure")
        original_sync(directory)

    monkeypatch.setattr(
        visual_identity_module, "_sync_publication_directory", fail_parent_sync
    )
    with pytest.raises(
        VisualIdentityPublicationError, match="visual_identity_publication_failed"
    ) as caught:
        publish_visual_identity_candidate(
            environment["db"], candidate, user_data_dir=environment["user_root"]
        )

    assert caught.value.cleanup_candidate_relpath is not None
    assert (
        environment["user_root"]
        / "visual_identities"
        / caught.value.cleanup_candidate_relpath
    ).is_dir()
    assert candidate._publishing is False
    assert _active_identity(environment) == before


@pytest.mark.parametrize("source_pack_change", ["archive", "revision"])
def test_fork_rejects_source_pack_status_or_revision_race(
    publication_environment,
    source_pack_change: str,
) -> None:
    environment = publication_environment
    db = environment["db"]
    candidate = create_visual_identity_candidate(
        db, actor_kind="character", actor_id=environment["actor_id"]
    )
    candidate.stage_replacement("thinking", _png_bytes((51, 51, 51)))

    def replace_then_change_source(source, destination, **kwargs):
        os.replace(source, destination, **kwargs)
        with db.transaction():
            db.execute_query(
                """
                UPDATE visual_identity_packs
                   SET status = ?, version = version + 1
                 WHERE id = ?
                """,
                (
                    "archived" if source_pack_change == "archive" else "active",
                    candidate.old_pack_id,
                ),
            )

    with pytest.raises(
        VisualIdentityPublicationError, match="visual_identity_binding_changed"
    ) as caught:
        publish_visual_identity_candidate(
            db,
            candidate,
            user_data_dir=environment["user_root"],
            atomic_replace=replace_then_change_source,
        )

    assert caught.value.cleanup_candidate_relpath is not None
    assert not db.execute_query(
        "SELECT id FROM visual_identity_packs WHERE source_kind = 'manual'"
    ).fetchall()


def test_posix_staging_cleanup_candidate_is_consumable(
    publication_environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = publication_environment
    candidate = create_visual_identity_candidate(
        environment["db"],
        actor_kind="character",
        actor_id=environment["actor_id"],
    )
    candidate.stage_replacement("thinking", _png_bytes((52, 52, 52)))
    original_discard = visual_identity_module._discard_staging_directory

    monkeypatch.setattr(
        visual_identity_module, "_discard_staging_directory", lambda *_args: False
    )
    with pytest.raises(VisualIdentityPublicationError) as caught:
        publish_visual_identity_candidate(
            environment["db"],
            candidate,
            user_data_dir=environment["user_root"],
            atomic_replace=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                PermissionError("denied")
            ),
        )

    relpath = caught.value.cleanup_candidate_relpath
    assert relpath is not None
    assert "/.staging-" in relpath
    monkeypatch.setattr(
        visual_identity_module, "_discard_staging_directory", original_discard
    )
    assert visual_identity_module.cleanup_visual_identity_publication_candidate(
        environment["db"], relpath, user_data_dir=environment["user_root"]
    )
    assert not (environment["user_root"] / "visual_identities" / relpath).exists()


@pytest.mark.parametrize("reference_column", ["storage_relpath", "preview_relpath"])
def test_cleanup_holds_write_reservation_until_reference_recheck(
    publication_environment,
    monkeypatch: pytest.MonkeyPatch,
    reference_column: str,
) -> None:
    environment = publication_environment
    db = environment["db"]
    candidate = create_visual_identity_candidate(
        db, actor_kind="character", actor_id=environment["actor_id"]
    )
    candidate.stage_replacement("thinking", _png_bytes((53, 53, 53)))
    monkeypatch.setattr(
        VisualIdentityRepository,
        "activate_pack",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(sqlite3.OperationalError()),
    )
    with pytest.raises(VisualIdentityPublicationError) as caught:
        publish_visual_identity_candidate(
            db, candidate, user_data_dir=environment["user_root"]
        )
    relpath = caught.value.cleanup_candidate_relpath
    assert relpath is not None

    original_discard = visual_identity_module._discard_pinned_directory
    injected = False

    def inject_reference_then_report_success(parent_fd, entry_name, pinned_fd):
        nonlocal injected
        if not injected:
            injected = True
            update_reference = (
                """
                UPDATE visual_identity_assets
                   SET preview_relpath = ?
                 WHERE id = (SELECT MIN(id) FROM visual_identity_assets)
                """
                if reference_column == "preview_relpath"
                else """
                UPDATE visual_identity_assets
                   SET storage_relpath = ?
                 WHERE id = (SELECT MIN(id) FROM visual_identity_assets)
                """
            )
            db.execute_query(
                update_reference,
                (f"{relpath}/manifest.json",),
            )
            return True
        return original_discard(parent_fd, entry_name, pinned_fd)

    monkeypatch.setattr(
        visual_identity_module,
        "_discard_pinned_directory",
        inject_reference_then_report_success,
    )
    with pytest.raises(
        VisualIdentityPublicationError, match="visual_identity_cleanup_referenced"
    ):
        visual_identity_module.cleanup_visual_identity_publication_candidate(
            db, relpath, user_data_dir=environment["user_root"]
        )


@pytest.mark.parametrize("reference_column", ["storage_relpath", "preview_relpath"])
def test_new_reference_writer_cannot_enter_during_cleanup(
    publication_environment,
    monkeypatch: pytest.MonkeyPatch,
    reference_column: str,
) -> None:
    environment = publication_environment
    db = environment["db"]
    candidate = create_visual_identity_candidate(
        db, actor_kind="character", actor_id=environment["actor_id"]
    )
    candidate.stage_replacement("thinking", _png_bytes((53, 54, 53)))
    monkeypatch.setattr(
        VisualIdentityRepository,
        "activate_pack",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(sqlite3.OperationalError()),
    )
    with pytest.raises(VisualIdentityPublicationError) as caught:
        publish_visual_identity_candidate(
            db, candidate, user_data_dir=environment["user_root"]
        )
    relpath = caught.value.cleanup_candidate_relpath
    assert relpath is not None

    discard_entered = threading.Event()
    release_discard = threading.Event()
    writer_attempting = threading.Event()
    writer_entered = threading.Event()
    original_discard = visual_identity_module._discard_pinned_directory

    def blocked_discard(parent_fd, entry_name, pinned_fd):
        discard_entered.set()
        assert release_discard.wait(5)
        return original_discard(parent_fd, entry_name, pinned_fd)

    def competing_reference_writer():
        connection = sqlite3.connect(db.db_path_str, timeout=5)
        try:
            writer_attempting.set()
            connection.execute("BEGIN IMMEDIATE")
            writer_entered.set()
            update_reference = (
                """
                UPDATE visual_identity_assets
                   SET preview_relpath = ?
                 WHERE id = (SELECT MIN(id) FROM visual_identity_assets)
                """
                if reference_column == "preview_relpath"
                else """
                UPDATE visual_identity_assets
                   SET storage_relpath = ?
                 WHERE id = (SELECT MIN(id) FROM visual_identity_assets)
                """
            )
            connection.execute(
                update_reference,
                (f"{relpath}/manifest.json",),
            )
            connection.rollback()
        finally:
            connection.close()

    monkeypatch.setattr(
        visual_identity_module, "_discard_pinned_directory", blocked_discard
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        cleanup = executor.submit(
            visual_identity_module.cleanup_visual_identity_publication_candidate,
            db,
            relpath,
            user_data_dir=environment["user_root"],
        )
        assert discard_entered.wait(5)
        writer = executor.submit(competing_reference_writer)
        assert writer_attempting.wait(5)
        try:
            assert not writer_entered.wait(0.2)
        finally:
            release_discard.set()
        assert cleanup.result(timeout=5)
        writer.result(timeout=5)
    assert writer_entered.is_set()


def test_final_name_is_revalidated_inside_repository_transaction(
    publication_environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = publication_environment
    before = _active_identity(environment)
    candidate = create_visual_identity_candidate(
        environment["db"],
        actor_kind="character",
        actor_id=environment["actor_id"],
    )
    candidate.stage_replacement("thinking", _png_bytes((54, 54, 54)))
    original_sync = visual_identity_module._sync_publication_directory
    sync_calls = 0

    def swap_after_second_sync(directory):
        nonlocal sync_calls
        original_sync(directory)
        sync_calls += 1
        if sync_calls == 2:
            versions = next(environment["user_root"].rglob("versions"))
            final = next(
                path
                for path in versions.iterdir()
                if path.is_dir() and not path.name.startswith(".staging-")
            )
            os.rename(final, versions / ".detached-final")
            final.mkdir()

    monkeypatch.setattr(
        visual_identity_module, "_sync_publication_directory", swap_after_second_sync
    )
    with pytest.raises(
        VisualIdentityPublicationError, match="visual_identity_publication_denied"
    ) as caught:
        publish_visual_identity_candidate(
            environment["db"],
            candidate,
            user_data_dir=environment["user_root"],
        )

    assert _active_identity(environment) == before
    assert caught.value.cleanup_candidate_relpath is None
    assert (
        not environment["db"]
        .execute_query(
            "SELECT id FROM visual_identity_packs WHERE source_kind = 'manual'"
        )
        .fetchall()
    )


def test_final_name_guard_runs_with_repository_write_reservation(
    publication_environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = publication_environment
    db = environment["db"]
    candidate = create_visual_identity_candidate(
        db, actor_kind="character", actor_id=environment["actor_id"]
    )
    candidate.stage_replacement("thinking", _png_bytes((54, 55, 54)))
    original_match = visual_identity_module._entry_matches_fd
    reservation_observed: list[bool] = []

    def observe_reservation(parent_fd, entry_name, pinned_fd):
        if db.get_connection().in_transaction:
            competitor = sqlite3.connect(db.db_path_str, timeout=0)
            try:
                try:
                    competitor.execute("BEGIN IMMEDIATE")
                except sqlite3.OperationalError:
                    reservation_observed.append(True)
                else:
                    reservation_observed.append(False)
                    competitor.rollback()
            finally:
                competitor.close()
        return original_match(parent_fd, entry_name, pinned_fd)

    monkeypatch.setattr(
        visual_identity_module, "_entry_matches_fd", observe_reservation
    )
    publish_visual_identity_candidate(
        db, candidate, user_data_dir=environment["user_root"]
    )

    assert reservation_observed == [True]


def test_cleanup_refuses_detached_directory_replaced_by_symlink(
    publication_environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = publication_environment
    db = environment["db"]
    candidate = create_visual_identity_candidate(
        db, actor_kind="character", actor_id=environment["actor_id"]
    )
    candidate.stage_replacement("thinking", _png_bytes((55, 55, 55)))
    monkeypatch.setattr(
        VisualIdentityRepository,
        "activate_pack",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(sqlite3.OperationalError()),
    )
    with pytest.raises(VisualIdentityPublicationError) as caught:
        publish_visual_identity_candidate(
            db, candidate, user_data_dir=environment["user_root"]
        )
    relpath = caught.value.cleanup_candidate_relpath
    assert relpath is not None
    original_discard = visual_identity_module._discard_pinned_directory

    def substitute_then_discard(parent_fd, entry_name, pinned_fd):
        os.rename(
            entry_name,
            ".detached-cleanup",
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        os.mkdir("replacement-target", dir_fd=parent_fd)
        os.symlink("replacement-target", entry_name, dir_fd=parent_fd)
        return original_discard(parent_fd, entry_name, pinned_fd)

    monkeypatch.setattr(
        visual_identity_module, "_discard_pinned_directory", substitute_then_discard
    )
    with pytest.raises(
        VisualIdentityPublicationError, match="visual_identity_cleanup_denied"
    ):
        visual_identity_module.cleanup_visual_identity_publication_candidate(
            db, relpath, user_data_dir=environment["user_root"]
        )

    versions = next(environment["user_root"].rglob("versions"))
    assert (versions / ".detached-cleanup" / "manifest.json").is_file()
    assert (versions / Path(relpath).name).is_symlink()


def test_cleanup_rejects_caller_owned_transaction(
    publication_environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = publication_environment
    db = environment["db"]
    candidate = create_visual_identity_candidate(
        db, actor_kind="character", actor_id=environment["actor_id"]
    )
    candidate.stage_replacement("thinking", _png_bytes((56, 56, 56)))
    monkeypatch.setattr(
        VisualIdentityRepository,
        "activate_pack",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(sqlite3.OperationalError()),
    )
    with pytest.raises(VisualIdentityPublicationError) as caught:
        publish_visual_identity_candidate(
            db, candidate, user_data_dir=environment["user_root"]
        )
    relpath = caught.value.cleanup_candidate_relpath
    assert relpath is not None

    with (
        db.transaction(),
        pytest.raises(
            VisualIdentityPublicationError,
            match="visual_identity_transaction_active",
        ),
    ):
        visual_identity_module.cleanup_visual_identity_publication_candidate(
            db, relpath, user_data_dir=environment["user_root"]
        )

    assert (environment["user_root"] / "visual_identities" / relpath).is_dir()


@pytest.mark.parametrize("preview_suffix", ["", "/preview.webp"])
def test_cleanup_refuses_preview_only_live_reference(
    publication_environment,
    monkeypatch: pytest.MonkeyPatch,
    preview_suffix: str,
) -> None:
    environment = publication_environment
    db = environment["db"]
    candidate = create_visual_identity_candidate(
        db, actor_kind="character", actor_id=environment["actor_id"]
    )
    candidate.stage_replacement("thinking", _png_bytes((57, 57, 57)))
    monkeypatch.setattr(
        VisualIdentityRepository,
        "activate_pack",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(sqlite3.OperationalError()),
    )
    with pytest.raises(VisualIdentityPublicationError) as caught:
        publish_visual_identity_candidate(
            db, candidate, user_data_dir=environment["user_root"]
        )
    relpath = caught.value.cleanup_candidate_relpath
    assert relpath is not None
    with db.transaction():
        db.execute_query(
            """
            UPDATE visual_identity_assets
               SET preview_relpath = ?
             WHERE id = (SELECT MIN(id) FROM visual_identity_assets)
            """,
            (f"{relpath}{preview_suffix}",),
        )

    with pytest.raises(
        VisualIdentityPublicationError, match="visual_identity_cleanup_referenced"
    ):
        visual_identity_module.cleanup_visual_identity_publication_candidate(
            db, relpath, user_data_dir=environment["user_root"]
        )

    assert (environment["user_root"] / "visual_identities" / relpath).is_dir()


def test_cleanup_fails_closed_when_name_swaps_during_enumeration(
    publication_environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = publication_environment
    db = environment["db"]
    candidate = create_visual_identity_candidate(
        db, actor_kind="character", actor_id=environment["actor_id"]
    )
    candidate.stage_replacement("thinking", _png_bytes((58, 58, 58)))
    monkeypatch.setattr(
        VisualIdentityRepository,
        "activate_pack",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(sqlite3.OperationalError()),
    )
    with pytest.raises(VisualIdentityPublicationError) as caught:
        publish_visual_identity_candidate(
            db, candidate, user_data_dir=environment["user_root"]
        )
    relpath = caught.value.cleanup_candidate_relpath
    assert relpath is not None
    candidate_path = environment["user_root"] / "visual_identities" / relpath
    detached = candidate_path.with_name(".detached-mid-cleanup")
    original_listdir = os.listdir
    swapped = False

    def swap_name_then_enumerate(directory_fd):
        nonlocal swapped
        filenames = original_listdir(directory_fd)
        if not swapped:
            swapped = True
            os.rename(candidate_path, detached)
            candidate_path.mkdir()
        return filenames

    monkeypatch.setattr(os, "listdir", swap_name_then_enumerate)
    with pytest.raises(
        VisualIdentityPublicationError, match="visual_identity_cleanup_denied"
    ):
        visual_identity_module.cleanup_visual_identity_publication_candidate(
            db, relpath, user_data_dir=environment["user_root"]
        )

    assert detached.is_dir()
    assert candidate_path.is_dir()
