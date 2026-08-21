"""Filesystem/SQLite tests for immutable Persona Visual publication."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
from dataclasses import FrozenInstanceError, replace
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

import tldw_chatbook.Persona_Visual.publication as publication_module
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Persona_Visual.assets import PersonaVisualAssetMetadata
from tldw_chatbook.Persona_Visual.contracts import MAX_ASSET_TOTAL_BYTES
from tldw_chatbook.Persona_Visual.publication import (
    PersonaVisualPublicationAssetSource,
    PersonaVisualPublicationError,
    PersonaVisualPublicationSnapshot,
    cleanup_persona_visual_publication_candidate,
    publish_persona_visual,
)
from tldw_chatbook.Persona_Visual.repository import PersonaVisualRepository


def _png_bytes(color: tuple[int, int, int] = (1, 2, 3)) -> bytes:
    output = BytesIO()
    Image.new("RGB", (4, 5), color).save(output, format="PNG")
    return output.getvalue()


def _manifest(*, frame_rate: int = 1) -> dict[str, Any]:
    return {
        "renderer_type": "sprite_frames",
        "manifest_version": 1,
        "states": {
            state: {"animation_id": "idle"}
            for state in ("idle", "listening", "thinking", "speaking", "error")
        },
        "animations": {
            "idle": {
                "frames": [{"asset_id": "idle"}],
                "preview_asset_id": "idle",
                "frame_rate": frame_rate,
            }
        },
        "state_catalog": {},
        "fallbacks": {},
        "authored_triggers": [],
    }


def _snapshot(
    source_root: Path,
    *,
    expected_identity=None,
    persona_revision: int = 7,
    color: tuple[int, int, int] = (1, 2, 3),
    frame_rate: int = 1,
    source_storage_key: str = "idle.png",
) -> PersonaVisualPublicationSnapshot:
    data = _png_bytes(color)
    source_path = source_root / source_storage_key
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(data)
    metadata = PersonaVisualAssetMetadata(
        asset_key="idle",
        role="frame",
        mime_type="image/png",
        byte_count=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        width=4,
        height=5,
        frame_count=1,
    )
    return PersonaVisualPublicationSnapshot(
        persona_id="persona-local-1",
        persona_revision=persona_revision,
        title="Operational states",
        description="Profile-private visuals",
        source_kind="manual",
        source_context=(("provenance", "publication-test"),),
        manifest_json=json.dumps(
            _manifest(frame_rate=frame_rate),
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        assets=(PersonaVisualPublicationAssetSource(source_storage_key, metadata),),
        expected_identity=expected_identity,
    )


@pytest.fixture
def environment(tmp_path: Path):
    source_root = tmp_path / "source-pack"
    profile_root = tmp_path / "profile"
    source_root.mkdir(mode=0o700)
    profile_root.mkdir(mode=0o700)
    db = CharactersRAGDB(tmp_path / "persona-visual-publication.db", "publication")
    repository = PersonaVisualRepository(db)
    try:
        yield repository, source_root, profile_root
    finally:
        db.close_connection()


def _publish(environment, snapshot, *, guard=lambda: True, **kwargs):
    repository, source_root, profile_root = environment
    return publish_persona_visual(
        repository,
        snapshot,
        source_root=source_root,
        profile_root=profile_root,
        authority_guard=guard,
        **kwargs,
    )


def test_first_activation_publishes_one_private_immutable_graph(environment) -> None:
    repository, source_root, profile_root = environment
    snapshot = _snapshot(source_root)

    result = _publish(environment, snapshot)

    assert result.old_identity is None
    assert result.cleanup_candidate is None
    assert (
        result.new_identity
        == repository.get_active_persona_pack(snapshot.persona_id).identity
    )
    manifest_rows = (
        repository.db.get_connection()
        .execute("SELECT storage_relpath FROM persona_visual_pack_versions")
        .fetchall()
    )
    asset_rows = (
        repository.db.get_connection()
        .execute("SELECT storage_relpath FROM persona_visual_assets")
        .fetchall()
    )
    assert len(manifest_rows) == len(asset_rows) == 1
    manifest = profile_root / manifest_rows[0][0]
    asset = profile_root / asset_rows[0][0]
    assert manifest.read_text(encoding="utf-8") == snapshot.manifest_json
    assert asset.read_bytes() == (source_root / "idle.png").read_bytes()
    assert not tuple(profile_root.rglob(".staging-*"))
    assert not hasattr(result, "path")
    with pytest.raises(FrozenInstanceError):
        result.cleanup_candidate = "changed"  # type: ignore[misc]


def test_later_publication_returns_exact_old_new_identity_without_referenced_cleanup(
    environment,
) -> None:
    repository, source_root, _profile_root = environment
    first = _publish(environment, _snapshot(source_root))
    second_snapshot = _snapshot(
        source_root,
        expected_identity=first.new_identity,
        color=(4, 5, 6),
        frame_rate=2,
    )

    second = _publish(environment, second_snapshot)

    assert second.old_identity == first.new_identity
    assert (
        second.new_identity
        == repository.get_active_persona_pack(second_snapshot.persona_id).identity
    )
    assert second.new_identity.binding_version == first.new_identity.binding_version + 1
    assert second.new_identity.pack_revision == first.new_identity.pack_revision + 1
    assert second.new_identity.version_number == first.new_identity.version_number + 1
    assert second.cleanup_candidate is None


def test_snapshot_is_deeply_immutable_and_rejects_noncanonical_nested_inputs(
    environment,
) -> None:
    _repository, source_root, _profile_root = environment
    snapshot = _snapshot(source_root)
    with pytest.raises(FrozenInstanceError):
        snapshot.title = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        snapshot.assets[0].source_storage_key = "changed.png"  # type: ignore[misc]

    with pytest.raises(PersonaVisualPublicationError, match="candidate_invalid"):
        _publish(environment, replace(snapshot, source_context=(("x", []),)))


@pytest.mark.parametrize("mutation", ("binding", "pack", "version", "persona"))
def test_source_identity_aba_is_rejected_before_database_activation(
    environment, mutation: str
) -> None:
    repository, source_root, _profile_root = environment
    first = _publish(environment, _snapshot(source_root))
    snapshot = _snapshot(
        source_root, expected_identity=first.new_identity, frame_rate=2
    )
    connection = repository.db.get_connection()
    identity = first.new_identity
    if mutation == "binding":
        connection.execute(
            "UPDATE persona_visual_bindings SET version = version + 2 WHERE id = ?",
            (identity.binding_id,),
        )
    elif mutation == "pack":
        connection.execute(
            "UPDATE persona_visual_packs SET version = version + 2 WHERE id = ?",
            (identity.pack_id,),
        )
    elif mutation == "version":
        connection.execute(
            "UPDATE persona_visual_pack_versions SET manifest_sha256 = ? WHERE id = ?",
            ("a" * 64, identity.pack_version_id),
        )
    else:
        snapshot = replace(snapshot, persona_revision=identity.persona_revision + 1)
    if mutation != "persona":
        connection.commit()

    with pytest.raises(PersonaVisualPublicationError, match="identity_changed"):
        _publish(environment, snapshot)

    assert not tuple((_profile_root / "persona_visual").rglob(".staging-*"))


def test_final_authority_guard_runs_after_filesystem_publish_and_rolls_back_db(
    environment,
) -> None:
    repository, source_root, profile_root = environment
    observed: list[bool] = []

    def stale_guard() -> bool:
        observed.append(
            any(profile_root.rglob("manifest.json"))
            and repository.db.get_connection().in_transaction
            and getattr(repository.db._local, "transaction_depth", 0) == 1
        )
        return False

    with pytest.raises(PersonaVisualPublicationError) as caught:
        _publish(environment, _snapshot(source_root), guard=stale_guard)

    assert observed == [True]
    assert repository.get_active_persona_pack("persona-local-1") is None
    assert caught.value.category == "persona_visual_authority_changed"
    assert caught.value.cleanup_candidate is not None


def test_source_name_or_inode_swap_during_replace_is_rejected(environment) -> None:
    repository, source_root, profile_root = environment
    snapshot = _snapshot(source_root)

    def replace_then_swap(source, destination, *, src_dir_fd, dst_dir_fd):
        os.replace(source, destination, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)
        source_root.rename(source_root.with_name("old-source"))
        source_root.mkdir()
        (source_root / "idle.png").write_bytes(_png_bytes((9, 9, 9)))

    with pytest.raises(PersonaVisualPublicationError) as caught:
        _publish(environment, snapshot, atomic_replace=replace_then_swap)

    assert repository.get_active_persona_pack(snapshot.persona_id) is None
    assert caught.value.category == "persona_visual_publication_denied"
    assert caught.value.cleanup_candidate is not None
    assert str(source_root) not in str(caught.value)
    assert str(profile_root) not in str(caught.value)


def test_source_intermediate_alias_and_same_inode_content_aba_are_rejected(
    environment,
) -> None:
    repository, source_root, _profile_root = environment
    snapshot = _snapshot(source_root, source_storage_key="assets/idle.png")

    def replace_then_swap_alias(source, destination, *, src_dir_fd, dst_dir_fd):
        os.replace(source, destination, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)
        original = source_root / "assets"
        original.rename(source_root / "old-assets")
        replacement = source_root / "assets"
        replacement.mkdir()
        (replacement / "idle.png").write_bytes(_png_bytes((8, 8, 8)))

    with pytest.raises(PersonaVisualPublicationError, match="publication_denied"):
        _publish(environment, snapshot, atomic_replace=replace_then_swap_alias)
    assert repository.get_active_persona_pack(snapshot.persona_id) is None

    # Rebuild a fresh package and mutate bytes in-place without replacing its inode.
    (source_root / "assets/idle.png").write_bytes(_png_bytes((1, 2, 3)))
    snapshot = _snapshot(source_root, source_storage_key="assets/idle.png")

    def replace_then_mutate(source, destination, *, src_dir_fd, dst_dir_fd):
        os.replace(source, destination, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)
        target = source_root / "assets/idle.png"
        with target.open("r+b") as stream:
            stream.seek(0)
            stream.write(_png_bytes((7, 7, 7)))
            stream.flush()
            os.fsync(stream.fileno())

    with pytest.raises(PersonaVisualPublicationError, match="publication_denied"):
        _publish(environment, snapshot, atomic_replace=replace_then_mutate)
    assert repository.get_active_persona_pack(snapshot.persona_id) is None


def test_final_inode_swap_inside_authority_guard_is_rejected(environment) -> None:
    repository, source_root, profile_root = environment
    snapshot = _snapshot(source_root)

    def swap_final() -> bool:
        final = next(profile_root.rglob("manifest.json")).parent
        final.rename(final.with_name("removed-final"))
        final.mkdir()
        return True

    with pytest.raises(PersonaVisualPublicationError, match="publication_denied"):
        _publish(environment, snapshot, guard=swap_final)

    assert repository.get_active_persona_pack(snapshot.persona_id) is None


@pytest.mark.parametrize("target_kind", ("manifest", "asset"))
def test_final_file_content_mutation_inside_authority_guard_is_rejected(
    environment, target_kind: str
) -> None:
    repository, source_root, profile_root = environment
    snapshot = _snapshot(source_root)

    def mutate_final_file() -> bool:
        if target_kind == "manifest":
            target = next(profile_root.rglob("manifest.json"))
            raw = target.read_bytes()
            changed = raw.replace(b'"frame_rate":1', b'"frame_rate":9')
            assert len(changed) == len(raw) and changed != raw
        else:
            target = next(profile_root.rglob("000.png"))
            changed = _png_bytes((9, 8, 7))
            assert len(changed) == target.stat().st_size
        with target.open("r+b") as stream:
            stream.seek(0)
            stream.write(changed)
            stream.flush()
            os.fsync(stream.fileno())
        return True

    with pytest.raises(PersonaVisualPublicationError) as caught:
        _publish(environment, snapshot, guard=mutate_final_file)

    assert caught.value.category == "persona_visual_publication_denied"
    assert caught.value.cleanup_candidate is not None
    assert repository.get_active_persona_pack(snapshot.persona_id) is None


@pytest.mark.parametrize(
    "layout", ("same", "profile_inside_source", "source_inside_profile")
)
def test_profile_and_package_root_overlap_is_rejected(
    tmp_path: Path, layout: str
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    if layout == "same":
        source_root = profile_root = root
    elif layout == "profile_inside_source":
        source_root, profile_root = root, root / "profile"
        profile_root.mkdir()
    else:
        profile_root, source_root = root, root / "source"
        source_root.mkdir()
    db = CharactersRAGDB(tmp_path / "overlap.db", "overlap")
    repository = PersonaVisualRepository(db)
    try:
        snapshot = _snapshot(source_root)
        with pytest.raises(PersonaVisualPublicationError, match="publication_denied"):
            publish_persona_visual(
                repository,
                snapshot,
                source_root=source_root,
                profile_root=profile_root,
                authority_guard=lambda: True,
            )
    finally:
        db.close_connection()


def test_materialization_budget_is_checked_before_profile_mutation(environment) -> None:
    _repository, source_root, profile_root = environment
    snapshot = _snapshot(source_root)
    oversized = replace(
        snapshot.assets[0].metadata,
        byte_count=MAX_ASSET_TOTAL_BYTES + 1,
    )

    with pytest.raises(PersonaVisualPublicationError, match="candidate_invalid"):
        _publish(
            environment,
            replace(
                snapshot,
                assets=(PersonaVisualPublicationAssetSource("idle.png", oversized),),
            ),
        )

    assert not (profile_root / "persona_visual").exists()


def test_missing_required_state_manifest_is_rejected_before_profile_mutation(
    environment,
) -> None:
    repository, source_root, profile_root = environment
    snapshot = _snapshot(source_root)
    manifest = json.loads(snapshot.manifest_json)
    del manifest["states"]["error"]
    invalid = replace(
        snapshot,
        manifest_json=json.dumps(
            manifest,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )

    with pytest.raises(PersonaVisualPublicationError) as caught:
        _publish(environment, invalid)

    assert caught.value.category == "persona_visual_candidate_invalid"
    assert caught.value.cleanup_candidate is None
    assert not (profile_root / "persona_visual").exists()
    assert repository.get_active_persona_pack(snapshot.persona_id) is None


@pytest.mark.parametrize(
    "private_context",
    (
        "../private",
        "C:\\private",
        "\x01private",
        "..",
        "{private",
        "[private",
        "~private",
    ),
)
def test_source_context_matches_repository_boundary_before_profile_mutation(
    environment, private_context: str
) -> None:
    repository, source_root, profile_root = environment
    snapshot = replace(
        _snapshot(source_root),
        source_context=(("provenance", private_context),),
    )

    with pytest.raises(PersonaVisualPublicationError) as caught:
        _publish(environment, snapshot)

    assert caught.value.category == "persona_visual_candidate_invalid"
    assert caught.value.cleanup_candidate is None
    assert not (profile_root / "persona_visual").exists()
    assert repository.get_active_persona_pack(snapshot.persona_id) is None


def test_database_failure_after_atomic_publish_returns_cleanup_candidate(
    environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, source_root, _profile_root = environment
    snapshot = _snapshot(source_root)

    def fail(*args, **kwargs):
        raise sqlite3.OperationalError("private database detail")

    monkeypatch.setattr(repository, "activate_new_pack", fail)
    with pytest.raises(PersonaVisualPublicationError) as caught:
        _publish(environment, snapshot)

    assert caught.value.category == "persona_visual_database_failed"
    assert caught.value.cleanup_candidate is not None
    assert "private database detail" not in str(caught.value)


def test_atomic_replace_failure_after_rename_returns_exact_cleanup_candidate(
    environment,
) -> None:
    repository, source_root, profile_root = environment

    def rename_then_fail(source, destination, *, src_dir_fd, dst_dir_fd):
        os.replace(source, destination, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)
        raise OSError("post-rename failure")

    with pytest.raises(PersonaVisualPublicationError) as caught:
        _publish(
            environment,
            _snapshot(source_root),
            atomic_replace=rename_then_fail,
        )

    assert caught.value.category == "persona_visual_publication_failed"
    assert caught.value.cleanup_candidate is not None
    assert (profile_root / caught.value.cleanup_candidate).is_dir()
    assert repository.get_active_persona_pack("persona-local-1") is None


def test_pre_rename_failure_returns_staging_token_only_when_pinned_cleanup_fails(
    environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, source_root, profile_root = environment
    monkeypatch.setattr(
        publication_module, "_delete_pinned_directory", lambda *_args: False
    )

    def fail_before_rename(*_args, **_kwargs):
        raise OSError("pre-rename failure")

    with pytest.raises(PersonaVisualPublicationError) as caught:
        _publish(
            environment,
            _snapshot(source_root),
            atomic_replace=fail_before_rename,
        )

    assert caught.value.cleanup_candidate is not None
    assert "/.staging-" in caught.value.cleanup_candidate
    assert (profile_root / caught.value.cleanup_candidate).is_dir()
    assert repository.get_active_persona_pack("persona-local-1") is None


def test_repository_partial_write_failure_rolls_back_complete_graph(
    environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, source_root, _profile_root = environment
    real_execute = repository.db.execute_query

    def fail_binding_insert(query, *args, **kwargs):
        if "INSERT INTO persona_visual_bindings" in query:
            raise sqlite3.OperationalError("injected partial write")
        return real_execute(query, *args, **kwargs)

    monkeypatch.setattr(repository.db, "execute_query", fail_binding_insert)
    with pytest.raises(PersonaVisualPublicationError) as caught:
        _publish(environment, _snapshot(source_root))

    assert caught.value.category == "persona_visual_database_failed"
    connection = repository.db.get_connection()
    for table in (
        "persona_visual_packs",
        "persona_visual_pack_versions",
        "persona_visual_assets",
        "persona_visual_bindings",
    ):
        assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_synchronous_cancellation_cannot_leave_a_half_updated_database_graph(
    environment,
) -> None:
    repository, source_root, _profile_root = environment

    def interrupt_guard() -> bool:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _publish(environment, _snapshot(source_root), guard=interrupt_guard)

    assert repository.get_active_persona_pack("persona-local-1") is None


def test_cleanup_begin_immediate_reserves_against_reference_insertion(
    environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, source_root, profile_root = environment
    with pytest.raises(PersonaVisualPublicationError) as caught:
        _publish(environment, _snapshot(source_root), guard=lambda: False)
    cleanup_candidate = caught.value.cleanup_candidate
    assert cleanup_candidate is not None
    blocked: list[bool] = []
    real_delete = publication_module._delete_pinned_directory

    def compete_then_delete(parent_fd, name, pinned_fd):
        competitor = sqlite3.connect(repository.db.db_path, timeout=0.01)
        try:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                competitor.execute("BEGIN IMMEDIATE")
            blocked.append(True)
        finally:
            competitor.close()
        return real_delete(parent_fd, name, pinned_fd)

    monkeypatch.setattr(
        publication_module, "_delete_pinned_directory", compete_then_delete
    )
    assert cleanup_persona_visual_publication_candidate(
        repository, cleanup_candidate, profile_root=profile_root
    )
    assert blocked and all(blocked)


def test_file_and_directory_fsync_precede_one_atomic_rename(
    environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    _repository, source_root, _profile_root = environment
    events: list[str] = []
    real_fsync = os.fsync

    def fsync_spy(fd: int) -> None:
        events.append(
            "dir_fsync" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file_fsync"
        )
        real_fsync(fd)

    def replace_spy(source, destination, *, src_dir_fd, dst_dir_fd):
        events.append("replace")
        os.replace(source, destination, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

    # The production call shape is the assertion; avoid monkeypatching os.replace.
    monkeypatch.setattr(publication_module.os, "fsync", fsync_spy)
    _publish(environment, _snapshot(source_root), atomic_replace=replace_spy)

    assert "file_fsync" in events
    replace_index = events.index("replace")
    assert "dir_fsync" in events[:replace_index]
    assert "dir_fsync" in events[replace_index + 1 :]


def test_successful_cleanup_deletes_only_unreferenced_old_pinned_directory(
    environment,
) -> None:
    repository, source_root, profile_root = environment
    with pytest.raises(PersonaVisualPublicationError) as caught:
        _publish(environment, _snapshot(source_root), guard=lambda: False)
    cleanup_candidate = caught.value.cleanup_candidate
    assert cleanup_candidate is not None
    old_dir = profile_root / cleanup_candidate
    sentinel = old_dir.parent / "sibling"
    sentinel.mkdir()

    assert (
        cleanup_persona_visual_publication_candidate(
            repository, cleanup_candidate, profile_root=profile_root
        )
        is True
    )
    assert not old_dir.exists()
    assert sentinel.is_dir()


def test_cleanup_refuses_active_reference_invalid_tokens_and_symlink_substitution(
    environment,
) -> None:
    repository, source_root, profile_root = environment
    _publish(environment, _snapshot(source_root))
    active_token = (
        repository.db.get_connection()
        .execute("SELECT storage_relpath FROM persona_visual_pack_versions")
        .fetchone()[0]
    )
    active_token = str(Path(active_token).parent)

    with pytest.raises(PersonaVisualPublicationError, match="cleanup_referenced"):
        cleanup_persona_visual_publication_candidate(
            repository, active_token, profile_root=profile_root
        )
    for token in ("../escape", ".staging-deadbeef", str(profile_root)):
        with pytest.raises(PersonaVisualPublicationError, match="cleanup_denied"):
            cleanup_persona_visual_publication_candidate(
                repository, token, profile_root=profile_root
            )

    target = profile_root / "target"
    target.mkdir()
    candidate = (
        profile_root / "persona_visual/packs/" / ("a" * 32) / "versions" / ("b" * 32)
    )
    candidate.parent.mkdir(parents=True)
    candidate.symlink_to(target, target_is_directory=True)
    with pytest.raises(PersonaVisualPublicationError, match="cleanup_denied"):
        cleanup_persona_visual_publication_candidate(
            repository,
            candidate.relative_to(profile_root).as_posix(),
            profile_root=profile_root,
        )
    assert target.is_dir()


def test_cleanup_rechecks_manifest_and_asset_references_after_pinned_deletion(
    environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, source_root, profile_root = environment
    first = _publish(environment, _snapshot(source_root))
    with pytest.raises(PersonaVisualPublicationError) as caught:
        _publish(
            environment,
            _snapshot(source_root, expected_identity=first.new_identity, frame_rate=2),
            guard=lambda: False,
        )
    cleanup_candidate = caught.value.cleanup_candidate
    assert cleanup_candidate is not None
    real_delete = publication_module._delete_pinned_directory

    def delete_then_reference(parent_fd, name, pinned_fd):
        deleted = real_delete(parent_fd, name, pinned_fd)
        repository.db.get_connection().execute(
            "UPDATE persona_visual_pack_versions SET storage_relpath = ? WHERE id = ?",
            (f"{cleanup_candidate}/manifest.json", first.new_identity.pack_version_id),
        )
        return deleted

    monkeypatch.setattr(
        publication_module, "_delete_pinned_directory", delete_then_reference
    )
    with pytest.raises(PersonaVisualPublicationError, match="cleanup_referenced"):
        cleanup_persona_visual_publication_candidate(
            repository, cleanup_candidate, profile_root=profile_root
        )


def test_publication_and_cleanup_fail_closed_without_posix_descriptor_guards(
    environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, source_root, profile_root = environment
    snapshot = _snapshot(source_root)
    monkeypatch.setattr(publication_module, "_posix_guards_available", lambda: False)

    with pytest.raises(PersonaVisualPublicationError, match="publication_denied"):
        _publish(environment, snapshot)
    assert repository.get_active_persona_pack(snapshot.persona_id) is None
    with pytest.raises(PersonaVisualPublicationError, match="cleanup_denied"):
        cleanup_persona_visual_publication_candidate(
            repository,
            "persona_visual/packs/" + "a" * 32 + "/versions/" + "b" * 32,
            profile_root=profile_root,
        )
