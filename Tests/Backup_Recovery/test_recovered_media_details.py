"""Passive recovered-media details and stale explicit deletion reviews."""

import dataclasses
import shutil
import sqlite3

import pytest

from Tests.Backup_Recovery.test_participant_lifetimes import (
    local_root as local_root,  # noqa: PLC0414
)
from Tests.Backup_Recovery.test_recovered_media import media as media  # noqa: PLC0414
from Tests.Backup_Recovery.test_recovered_media import retain


def test_absent_details_do_not_create_catalog(tmp_path, local_root):
    from tldw_chatbook.Backup_Recovery import recovered_media as owner

    root = tmp_path / "absent"
    assert owner.list_recovered_media(root) is None
    assert not root.exists()
    root.mkdir(mode=0o700)
    assert owner.list_recovered_media(root) is None
    assert not (root / "catalog.sqlite3").exists()


def test_passive_details_preserve_catalog_and_pending_operations(media, monkeypatch):
    from tldw_chatbook.Backup_Recovery import recovered_media as owner

    store, source = media
    asset = retain(store, source)
    before = store.db_path.read_bytes()

    def forbidden(*args, **kwargs):
        pytest.fail(
            "passive details invoked constructor, migration, recovery or payload IO"
        )

    monkeypatch.setattr(owner.RecoveredMedia, "__init__", forbidden)
    monkeypatch.setattr(owner.RecoveredMedia, "recover", forbidden)
    monkeypatch.setattr(owner, "migrate", forbidden)
    monkeypatch.setattr(owner, "_read", forbidden)
    details = owner.list_recovered_media(store.root)
    assert details.total == 1 and not details.has_more
    row = details.assets[0]
    assert row.asset_id == asset and row.size == len(source.read_bytes())
    assert row.state == "ready" and row.reference_count == 1 and not row.orphan_eligible
    assert store.db_path.read_bytes() == before


def test_list_pagination_and_exact_review_disclose_all_profiles(media):
    from tldw_chatbook.Backup_Recovery import recovered_media as owner

    store, source = media
    asset = retain(store, source)
    store.add_reference(
        asset,
        profile="historical",
        message="other",
        slug="alias",
        media_type="image/png",
    )
    store.hold(asset, "recovery-copy")
    retain(store, source, message="second")
    details = owner.list_recovered_media(store.root, limit=1)
    assert details.total == 2 and len(details.assets) == 1 and details.has_more
    second = owner.list_recovered_media(store.root, limit=1, offset=1)
    assert len(second.assets) == 1 and not second.has_more
    assert second.assets[0].asset_id != details.assets[0].asset_id
    review = owner.review_recovered_asset(store.root, asset)
    assert review.references == (
        ("historical", "other", "alias", "image/png"),
        ("p", "m", "same", "image/png"),
    )
    assert review.holds == ("recovery-copy",)
    with pytest.raises(dataclasses.FrozenInstanceError):
        review.root = source
    with pytest.raises(ValueError, match="recovered_asset_held"):
        owner.delete_reviewed_asset(review)
    assert store.resolve(asset)[0] == "ready"


@pytest.mark.parametrize("change", ["reference", "hold", "catalog", "root"])
def test_reviewed_delete_refuses_changed_sources_or_effects(media, change):
    from tldw_chatbook.Backup_Recovery import recovered_media as owner

    store, source = media
    asset = retain(store, source)
    review = owner.review_recovered_asset(store.root, asset)
    if change == "reference":
        store.add_reference(
            asset, profile="p", message="new", slug="same", media_type="image/png"
        )
    elif change == "hold":
        store.hold(asset, "new-hold")
    elif change == "catalog":
        replacement = store.root / "replacement.sqlite3"
        shutil.copyfile(store.db_path, replacement)
        replacement.chmod(0o600)
        replacement.replace(store.db_path)
    else:
        original = store.root.with_name("original")
        store.root.rename(original)
        shutil.copytree(original, store.root)
    with pytest.raises((ValueError, RuntimeError, OSError)):
        owner.delete_reviewed_asset(review)
    with sqlite3.connect(store.db_path) as connection:
        assert connection.execute(
            "SELECT state FROM assets WHERE asset_id=?", (asset,)
        ).fetchone() == ("ready",)
        assert not connection.execute("SELECT 1 FROM tombstones").fetchall()
    assert (store.root / (asset + ".payload")).read_bytes() == source.read_bytes()


def test_reviewed_delete_preserves_deleted_references_and_recorded_size(media):
    from tldw_chatbook.Backup_Recovery import recovered_media as owner

    store, source = media
    asset = retain(store, source)
    review = owner.review_recovered_asset(store.root, asset)
    owner.delete_reviewed_asset(review)
    assert store.resolve_reference(
        profile="p", message="m", slug="same", media_type="image/png"
    ) == ("deleted", None)
    details = owner.list_recovered_media(store.root)
    assert details.assets[0].state == "deleted"
    assert details.assets[0].size == len(source.read_bytes())
    assert not details.assets[0].orphan_eligible
    assert not (store.root / (asset + ".payload")).exists()


def test_reviewed_cleanup_rechecks_eligibility(media):
    from tldw_chatbook.Backup_Recovery import recovered_media as owner

    store, source = media
    asset = retain(store, source)
    store.release_reference(
        profile="p", message="m", slug="same", media_type="image/png"
    )
    review = owner.review_recovered_asset(store.root, asset)
    assert review.asset.orphan_eligible
    store.hold(asset, "new")
    with pytest.raises(ValueError, match="recovered_review_changed"):
        owner.cleanup_reviewed_asset(review)
    store.release_hold(asset, "new")
    assert owner.cleanup_reviewed_asset(owner.review_recovered_asset(store.root, asset))
    assert store.resolve(asset) == ("deleted", None)


def test_invalid_deleted_metadata_cannot_be_reported_as_valid_deletion(media):
    from tldw_chatbook.Backup_Recovery import recovered_media as owner

    store, source = media
    asset = retain(store, source)
    store.delete(asset)
    with sqlite3.connect(store.db_path) as connection:
        connection.execute("UPDATE tombstones SET references_json='[]'")
    with pytest.raises(ValueError, match="invalid_recovered_tombstone"):
        owner.list_recovered_media(store.root)


def test_invalid_pagination_refuses_before_creation(tmp_path, local_root):
    from tldw_chatbook.Backup_Recovery import recovered_media as owner

    for kwargs in ({"limit": 0}, {"limit": 101}, {"offset": -1}):
        with pytest.raises(ValueError, match="recovered_details_limit"):
            owner.list_recovered_media(tmp_path / "missing", **kwargs)
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize("unsafe", ["root_link", "catalog_link", "catalog_mode"])
def test_unsafe_catalog_is_not_reported_as_absent(tmp_path, local_root, unsafe):
    from tldw_chatbook.Backup_Recovery import recovered_media as owner

    root = tmp_path / "recovered"
    if unsafe == "root_link":
        root.symlink_to(tmp_path / "missing", target_is_directory=True)
    else:
        root.mkdir(mode=0o700)
        catalog = root / "catalog.sqlite3"
        if unsafe == "catalog_link":
            catalog.symlink_to(tmp_path / "missing")
        else:
            catalog.write_bytes(b"unreadable catalog")
            catalog.chmod(0)
    with pytest.raises((ValueError, RuntimeError, OSError)):
        owner.list_recovered_media(root)


def test_passive_details_do_not_replay_pending_retain(media, monkeypatch):
    from tldw_chatbook.Backup_Recovery import recovered_media as owner

    store, source = media

    def stop(*args):
        raise RuntimeError("interrupted retain")

    with monkeypatch.context() as patch:
        patch.setattr(store, "_finish_retain", stop)
        with pytest.raises(RuntimeError, match="interrupted retain"):
            retain(store, source)
    with sqlite3.connect(store.db_path) as connection:
        before = connection.execute("SELECT asset_id,kind FROM operations").fetchall()
    assert len(before) == 1 and before[0][1] == "retain"
    details = owner.list_recovered_media(store.root)
    assert details.assets[0].state == "pending"
    assert not details.assets[0].orphan_eligible
    with sqlite3.connect(store.db_path) as connection:
        assert (
            connection.execute("SELECT asset_id,kind FROM operations").fetchall()
            == before
        )


def test_large_reference_review_refuses_instead_of_truncating(media):
    from tldw_chatbook.Backup_Recovery import recovered_media as owner

    store, source = media
    asset = retain(store, source)
    # A valid existing catalog may contain more refs than one bounded UI review.
    with sqlite3.connect(store.db_path) as connection:
        connection.executemany(
            "INSERT INTO refs VALUES (?,?,?,?,?)",
            (("p", f"message-{i}", "same", "image/png", asset) for i in range(1000)),
        )
    details = owner.list_recovered_media(store.root)
    assert details.assets[0].reference_count == 1001
    with pytest.raises(ValueError, match="recovered_details_limit"):
        owner.review_recovered_asset(store.root, asset)
    assert store.resolve(asset)[0] == "ready"


def test_oversized_reference_metadata_refuses_exact_review(media):
    from tldw_chatbook.Backup_Recovery import recovered_media as owner

    store, source = media
    asset = retain(store, source)
    with sqlite3.connect(store.db_path) as connection:
        connection.execute("UPDATE refs SET message=?", ("x" * 4097,))
    with pytest.raises(ValueError, match="invalid_recovered_reference"):
        owner.review_recovered_asset(store.root, asset)
    assert store.resolve(asset)[0] == "ready"
