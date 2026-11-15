"""Current native Core82/Subscription3 composition keeps exact catalog guards."""

import sqlite3
from contextlib import closing

import pytest


def _current_hybrid(path):
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB

    subscriptions = SubscriptionsDB(path)
    subscriptions.close()
    core = CharactersRAGDB(path, "native-hybrid-catalog-test")
    core.close_connection()


def test_actual_current_native_hybrid_is_readable_without_source_changes(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.DB.recovery_operations import recovery_adapters

    monkeypatch.setattr(
        bootstrap, "default_bootstrap_root", lambda: tmp_path / "bootstrap"
    )
    path = tmp_path / "hybrid.db"
    _current_hybrid(path)
    before = path.read_bytes()
    owner = next(
        row for row in recovery_adapters() if row.owner_id == "db.subscriptions"
    )
    assert owner.validate(path) == ()
    assert path.read_bytes() == before


@pytest.mark.parametrize("alteration", ["extra_index", "wrong_core_stamp"])
def test_current_hybrid_catalog_refuses_unknown_objects_and_wrong_stamp(
    tmp_path, monkeypatch, alteration
):
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.DB.recovery_operations import recovery_adapters

    monkeypatch.setattr(
        bootstrap, "default_bootstrap_root", lambda: tmp_path / "bootstrap"
    )
    path = tmp_path / "hybrid.db"
    _current_hybrid(path)
    owner = next(
        row for row in recovery_adapters() if row.owner_id == "db.subscriptions"
    )
    with (
        closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as source,
        closing(sqlite3.connect(":memory:")) as candidate,
    ):
        source.backup(candidate)
        # Build a negative candidate before the native validator installs
        # its read-only authorizer; never remove that restriction afterward.
        if alteration == "extra_index":
            candidate.execute(
                "CREATE INDEX unexpected_hybrid_index ON subscriptions(source)"
            )
            expected = "unsupported_schema"
        else:
            candidate.execute(
                "UPDATE db_schema_version SET version=81 WHERE schema_name='rag_char_chat_schema'"
            )
            expected = "unsupported_schema_version"
        assert owner._validate_catalog(candidate) == (expected,)
