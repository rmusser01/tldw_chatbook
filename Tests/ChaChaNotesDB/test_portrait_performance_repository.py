"""Real SQLite proof of atomic source/revision-fenced local attachments."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

import pytest

from Tests.Character_Chat.test_portrait_performance import png
from tldw_chatbook.Character_Chat.portrait_performance import (
    FRAME_ROLES,
    PerformanceConflict,
    capture_performance_source,
    normalize_performance_frames,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def performance_case(tmp_path):
    db = CharactersRAGDB(tmp_path / "performance.db", "performance-test")
    character_id = db.add_character_card({"name": "Ada", "image": png()})
    source, _ = capture_performance_source(
        db, character_id=character_id, requested_state="idle"
    )
    frames = normalize_performance_frames({role: png() for role in FRAME_ROLES})
    try:
        yield db, source, frames
    finally:
        db.close()


def test_remove_reapply_does_not_reuse_revision(performance_case):
    db, source, frames = performance_case
    first = db.apply_character_portrait_performance(source, frames, expected_revision=0)
    absent = db.remove_character_portrait_performance(source, expected_revision=first)
    record = db.get_character_portrait_performance(
        character_id=source.character_id, expression_key="neutral"
    )
    assert not record.active and record.frames == () and record.revision == absent
    second = db.apply_character_portrait_performance(
        source, frames, expected_revision=absent
    )
    assert 0 < first < absent < second
    with pytest.raises(PerformanceConflict):
        db.apply_character_portrait_performance(source, frames, expected_revision=first)
    assert db.get_character_card_by_id(source.character_id)["image"] == png()


def test_source_change_requires_explicit_new_source_replacement(performance_case):
    db, source, frames = performance_case
    first = db.apply_character_portrait_performance(source, frames, expected_revision=0)
    card = db.get_character_card_by_id(source.character_id)
    db.update_character_card(
        source.character_id, {"image": png("blue")}, card["version"]
    )
    with pytest.raises(PerformanceConflict):
        db.apply_character_portrait_performance(source, frames, expected_revision=first)
    with pytest.raises(PerformanceConflict):
        db.remove_character_portrait_performance(source, expected_revision=first)
    old = db.get_character_portrait_performance(
        character_id=source.character_id, expression_key="neutral"
    )
    assert old.source == source and old.revision == first
    current, _ = capture_performance_source(
        db, character_id=source.character_id, requested_state="idle"
    )
    assert current.source_sha256 != source.source_sha256
    assert (
        db.apply_character_portrait_performance(
            current, frames, expected_revision=first
        )
        > first
    )


@pytest.mark.parametrize(
    "forgery", ["hash", "dimensions", "role", "bytes", "duplicate"]
)
def test_apply_rechecks_frame_metadata_and_roles(performance_case, forgery):
    db, source, frames = performance_case
    changes = {
        "hash": {"sha256": "0" * 64},
        "dimensions": {"width": 9},
        "role": {"role": "bogus"},
        "bytes": {"png": b"broken"},
    }
    bad = (
        frames[1] if forgery == "duplicate" else replace(frames[0], **changes[forgery]),
    ) + frames[1:]
    with pytest.raises(ValueError):
        db.apply_character_portrait_performance(source, bad, expected_revision=0)
    assert (
        db.get_character_portrait_performance(
            character_id=source.character_id, expression_key="neutral"
        )
        is None
    )


def test_failed_child_insert_rolls_back_parent_and_old_frames(performance_case):
    db, source, frames = performance_case
    first = db.apply_character_portrait_performance(source, frames, expected_revision=0)
    with db.transaction() as cursor:
        cursor.execute(
            "CREATE TEMP TRIGGER fail_frame BEFORE INSERT ON character_portrait_performance_frames WHEN NEW.role='open' BEGIN SELECT RAISE(ABORT, 'fixture'); END"
        )
    with pytest.raises(Exception, match="fixture"):
        db.apply_character_portrait_performance(source, frames, expected_revision=first)
    current = db.get_character_portrait_performance(
        character_id=source.character_id, expression_key="neutral"
    )
    assert current.revision == first and current.frames == frames


def test_two_connections_cannot_overwrite_same_revision(performance_case):
    db, source, frames = performance_case
    barrier = Barrier(2)

    def apply():
        other = CharactersRAGDB(db.db_path_str, "racer")
        try:
            barrier.wait(timeout=10)
            try:
                return other.apply_character_portrait_performance(
                    source, frames, expected_revision=0
                )
            except PerformanceConflict:
                return "conflict"
        finally:
            other.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: apply(), range(2)))
    assert results.count("conflict") == 1
    assert sum(isinstance(value, int) for value in results) == 1


def test_deleted_owner_is_rejected_and_restore_preserves_frames(performance_case):
    db, source, frames = performance_case
    first = db.apply_character_portrait_performance(source, frames, expected_revision=0)
    card = db.get_character_card_by_id(source.character_id)
    assert db.soft_delete_character_card(source.character_id, card["version"])
    with pytest.raises(PerformanceConflict):
        db.apply_character_portrait_performance(source, frames, expected_revision=first)
    with pytest.raises(PerformanceConflict):
        db.remove_character_portrait_performance(source, expected_revision=first)
    assert db.restore_character_card(source.character_id, card["version"] + 1)
    assert (
        db.get_character_portrait_performance(
            character_id=source.character_id, expression_key="neutral"
        ).frames
        == frames
    )
    with db.transaction() as cursor:
        cursor.execute("DELETE FROM character_cards WHERE id=?", (source.character_id,))
    assert (
        db.get_character_portrait_performance(
            character_id=source.character_id, expression_key="neutral"
        )
        is None
    )
    assert (
        db.execute_query(
            "SELECT count(*) FROM character_portrait_performance_frames"
        ).fetchone()[0]
        == 0
    )


def test_sqlite_backup_preserves_attachment_without_sync_side_effect(
    performance_case, tmp_path
):
    db, source, frames = performance_case
    before = db.execute_query("SELECT count(*) FROM sync_log").fetchone()[0]
    revision = db.apply_character_portrait_performance(
        source, frames, expected_revision=0
    )
    assert db.execute_query("SELECT count(*) FROM sync_log").fetchone()[0] == before
    destination = tmp_path / "backup.db"
    assert db.backup_database(str(destination))
    restored = CharactersRAGDB(destination, "restored")
    try:
        record = restored.get_character_portrait_performance(
            character_id=source.character_id, expression_key="neutral"
        )
        assert (
            record.source == source
            and record.revision == revision
            and record.frames == frames
        )
        assert restored.get_character_card_by_id(source.character_id)["image"] == png()
    finally:
        restored.close()


def test_legacy_expression_keeps_actual_identity_and_bytes(performance_case):
    db, source, frames = performance_case
    legacy = png("green")
    db.set_character_expression_image(
        source.character_id, "speaking", legacy, "image/png"
    )
    selected, data = capture_performance_source(
        db, character_id=source.character_id, requested_state="speaking"
    )
    assert selected.source_kind == "legacy_expression"
    assert selected.expression_key == "custom:speaking" and data == legacy
    db.apply_character_portrait_performance(selected, frames, expected_revision=0)
    assert db.get_character_expression_image(source.character_id, "speaking") == legacy


def test_nested_failed_apply_cannot_commit_partial_replacement(performance_case):
    db, source, frames = performance_case
    revision = db.apply_character_portrait_performance(
        source, frames, expected_revision=0
    )
    with db.transaction() as cursor:
        cursor.execute(
            "CREATE TEMP TRIGGER fail_nested_frame BEFORE INSERT ON character_portrait_performance_frames WHEN NEW.role='open' BEGIN SELECT RAISE(ABORT, 'nested fixture'); END"
        )
        with pytest.raises(Exception, match="nested fixture"):
            db.apply_character_portrait_performance(
                source, frames, expected_revision=revision
            )
    record = db.get_character_portrait_performance(
        character_id=source.character_id, expression_key="neutral"
    )
    assert record.revision == revision and record.frames == frames


def test_source_changed_during_normalization_cannot_publish(
    performance_case, monkeypatch
):
    from tldw_chatbook.Character_Chat import portrait_performance

    db, source, frames = performance_case
    original = portrait_performance.normalize_performance_frames

    def change_source(values):
        result = original(values)
        card = db.get_character_card_by_id(source.character_id)
        db.update_character_card(
            source.character_id, {"image": png("blue")}, card["version"]
        )
        return result

    monkeypatch.setattr(
        portrait_performance, "normalize_performance_frames", change_source
    )
    with pytest.raises(PerformanceConflict):
        db.apply_character_portrait_performance(source, frames, expected_revision=0)
    assert (
        db.get_character_portrait_performance(
            character_id=source.character_id, expression_key="neutral"
        )
        is None
    )


def test_immutable_source_and_history_agree_and_rebinding_during_apply_is_stale(
    performance_case, tmp_path, monkeypatch
):
    from Tests.Character_Chat.test_visual_identity_resolution import _activate
    from tldw_chatbook.Character_Chat import portrait_performance as performance
    from tldw_chatbook.Character_Chat import visual_identity

    db, original, frames = performance_case
    profile = tmp_path / "profile"
    monkeypatch.setattr(visual_identity, "get_user_data_dir", lambda: profile)
    graph = _activate(db, profile, original.character_id, ("neutral", "happy"))
    source, _ = capture_performance_source(
        db, character_id=original.character_id, requested_state="happy"
    )
    assert source.source_kind == "visual_identity"
    historical = visual_identity.resolve_historical_visual_identity(
        db,
        actor_id=original.character_id,
        pack_id=source.source_ref[0],
        pack_version_id=source.source_ref[1],
        asset_id=source.source_ref[2],
        expression_key="happy",
        expression_id=None,
        user_data_dir=profile,
    )
    assert performance.source_from_resolution(historical) == source
    assert graph["pack"]["id"] == source.source_ref[0]
    revision = db.apply_character_portrait_performance(
        source, frames, expected_revision=0
    )
    normalize = performance.normalize_performance_frames

    def rebind(values):
        result = normalize(values)
        _activate(db, profile, source.character_id, ("neutral", "happy"))
        return result

    monkeypatch.setattr(performance, "normalize_performance_frames", rebind)
    with pytest.raises(PerformanceConflict):
        db.apply_character_portrait_performance(
            source, frames, expected_revision=revision
        )
    record = db.get_character_portrait_performance(
        character_id=source.character_id, expression_key="happy"
    )
    assert record.revision == revision and record.frames == frames


def test_nested_failed_remove_preserves_frames(performance_case):
    db, source, frames = performance_case
    revision = db.apply_character_portrait_performance(
        source, frames, expected_revision=0
    )
    with db.transaction() as cursor:
        cursor.execute(
            "CREATE TEMP TRIGGER fail_remove BEFORE UPDATE ON character_portrait_performances WHEN NEW.active=0 BEGIN SELECT RAISE(ABORT, 'remove fixture'); END"
        )
        with pytest.raises(Exception, match="remove fixture"):
            db.remove_character_portrait_performance(source, expected_revision=revision)
    record = db.get_character_portrait_performance(
        character_id=source.character_id, expression_key="neutral"
    )
    assert record.revision == revision and record.frames == frames and record.active
