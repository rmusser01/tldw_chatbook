"""Native uniqueness boundaries keep exact source dependencies visible."""

import sqlite3
from contextlib import closing

from Tests.Backup_Recovery.test_mobile_content_graph import _seed_core, content_reads


def test_embedded_nul_nocase_conflict_refuses_exact_source(tmp_path, monkeypatch):
    source, canonical = "a\x00x", "A\x00y"
    with closing(sqlite3.connect(":memory:")) as connection:
        assert (
            connection.execute(
                "SELECT ? = ? COLLATE NOCASE", (source, canonical)
            ).fetchone()[0]
            == 1
        )

    def seed(connection, purpose):
        _seed_core(connection, purpose)
        connection.execute(
            "UPDATE keywords SET keyword=? WHERE id=11",
            (source if purpose == "legacy" else canonical,),
        )

    with content_reads(tmp_path, monkeypatch, seed) as reads:
        plan = reads.content_graph_plan()
        assert any(
            b.table == "keywords"
            and b.original == (11,)
            and b.dependency_domain == "native_nocase_text"
            for b in plan.blockers
        )
        assert not any(
            row.table == "keywords" and row.original == (11,) for row in plan.rows
        )
        assert any(row.table == "world_books" for row in plan.rows)
        original = next(
            dict(row.values)
            for row in plan.source_rows
            if row.table == "keywords" and row.original == (11,)
        )
        assert original["keyword"] == source


def test_notes_locator_relaxation_keeps_dictionary_file_dependency_guard(
    tmp_path, monkeypatch
):
    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from Tests.Backup_Recovery.test_mobile_consolidation_preview import _stores
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        preview_mobile_consolidation,
    )

    original_seed = mobile_consolidation_fixture._seed

    def native_seed(stores, **kwargs):
        def dictionary_file(db, legacy):
            if legacy:
                with db.transaction() as cursor:
                    cursor.execute(
                        "INSERT INTO chat_dictionaries(name,file_path) VALUES ('source file dictionary','/missing/dictionary.json')"
                    )

        return original_seed(stores, **kwargs, chat_seed=dictionary_file)

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", native_seed)
    source = tmp_path / "mobile"
    mobile_consolidation_fixture.seed_legacy_mobile(source)
    preview = preview_mobile_consolidation(source, _stores(tmp_path))
    assert "mobile_missing_file_dependency" in preview.issues
