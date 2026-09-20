import sqlite3
from contextlib import contextmanager
from unittest.mock import Mock

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Library.library_artifacts_catalog import LibraryArtifactsCatalog
from tldw_chatbook.Library.library_artifacts_state import ArtifactKey, ArtifactScope
from tldw_chatbook.Subscriptions.briefing_keep import keep_briefing
from tldw_chatbook.Subscriptions.watchlist_bundle_service import WatchlistBundleService


def test_kept_is_readable_without_its_source(tmp_path):
    subs = SubscriptionsDB(tmp_path / "subs.db", "artifact-test")
    kept = CharactersRAGDB(tmp_path / "kept.db", client_id="artifact-test")
    try:
        manager = WatchlistBundleService(subs)
        watch = int(manager.create("Weekly digest")["id"])
        live = subs.insert_briefing(watch)
        subs.update_briefing(live, status="complete", body_markdown="# Saved body")
        saved = keep_briefing(subs, kept, live, origin="manual")
        manager.delete(watch)
        assert subs.get_briefing(live) is None
        catalog = LibraryArtifactsCatalog(subscriptions_db=None, chachanotes_db=kept)
        page = catalog.read_page(ArtifactScope(kept_only=True))
        key = ArtifactKey("kept_report", saved["kept_id"])
        assert page.total == 1
        assert [row.key for row in page.items] == [key]
        assert catalog.read_detail(key).body == "# Saved body"
    finally:
        kept.close_connection()
        subs.close()


@pytest.fixture
def owners(tmp_path):
    subs = SubscriptionsDB(tmp_path / "subs.db", "artifact-test")
    kept = CharactersRAGDB(tmp_path / "kept.db", client_id="artifact-test")
    try:
        yield subs, kept
    finally:
        kept.close_connection()
        subs.close()


def seed(owners, count=45):
    subs, kept = owners
    with subs.transaction() as conn:
        watch = conn.execute(
            "INSERT INTO watchlists(name) VALUES ('Daily news')"
        ).lastrowid
    expected = []
    for index in range(count):
        live = subs.insert_briefing(watch)
        subs.update_briefing(live, status="complete", body_markdown=f"Live {index}")
        live_time = f"2026-01-01 00:{index:02}:01"
        kept_time = f"2026-01-01 00:{index:02}:00"
        with subs.transaction() as conn:
            conn.execute(
                "UPDATE briefings SET created_at=? WHERE id=?", (live_time, live)
            )
        saved = kept.create_kept_briefing(
            source_briefing_id=live,
            watchlist_name="Saved news",
            body_markdown=f"Saved {index}",
            origin="manual",
            original_created_at=kept_time,
        )
        expected[0:0] = [
            ArtifactKey("live_report", live),
            ArtifactKey("kept_report", saved),
        ]
    return expected


def catalog_for(owners):
    return LibraryArtifactsCatalog(subscriptions_db=owners[0], chachanotes_db=owners[1])


def test_pages_and_direct_locator_are_complete_bounded_and_namespaced(
    owners, monkeypatch
):
    expected = seed(owners)
    catalog = catalog_for(owners)
    page = catalog.read_page(ArtifactScope())
    pages = [page]
    seen = list(page.items)
    while page.start + len(page.items) < page.total:
        page = catalog.read_page(page.scope, boundary=page.items[-1].order_key)
        assert page.start == len(seen)
        assert 0 < len(page.items) <= 20
        seen.extend(page.items)
        pages.append(page)
    assert [row.key for row in seen] == expected
    assert len({row.key for row in seen}) == 90
    previous = catalog.read_page(
        page.scope, boundary=page.items[0].order_key, direction="before"
    )
    assert previous.start == 60
    assert previous.items == pages[-2].items
    last = catalog.read_page(page.scope, direction="before")
    assert last.start == 70
    assert [row.key for row in last.items] == expected[-20:]
    counts = []
    for owner in owners:
        original = owner.read_artifact_window

        def measured(*args, _original=original, **kwargs):
            result = _original(*args, **kwargs)
            counts.append(len(result.items))
            return result

        monkeypatch.setattr(owner, "read_artifact_window", measured)
    located = catalog.locate(ArtifactScope(), expected[-1])
    assert located.start == 80
    assert [row.key for row in located.items] == expected[-10:]
    assert len(counts) <= 6
    assert all(n <= 20 for n in counts)


def test_kept_scope_never_touches_subscriptions_and_failures_are_explicit(owners):
    seed(owners, 1)
    failing = Mock()
    failing.artifact_read_snapshot.side_effect = sqlite3.OperationalError("failed")
    catalog = LibraryArtifactsCatalog(
        subscriptions_db=failing, chachanotes_db=owners[1]
    )
    assert catalog.read_page(ArtifactScope(kept_only=True)).total == 1
    assert failing.mock_calls == []
    from tldw_chatbook.Library.library_artifacts_catalog import ArtifactReadError

    with pytest.raises(ArtifactReadError):
        catalog.read_page(ArtifactScope())


def test_missing_target_query_and_deleted_anchor(owners):
    expected = seed(owners, 25)
    catalog = catalog_for(owners)
    page = catalog.read_page(ArtifactScope())
    anchor = page.items[-1]
    owners[1].delete_kept_briefing(anchor.key.native_id)
    next_page = catalog.read_page(page.scope, boundary=anchor.order_key)
    assert next_page.start == 19
    assert [row.key for row in next_page.items] == expected[20:40]
    assert catalog.locate(ArtifactScope(), anchor.key) is None
    assert catalog.locate(ArtifactScope(query="not found"), expected[0]) is None
    assert catalog.read_page(ArtifactScope(query="not found")).items == ()
    assert catalog.read_page(ArtifactScope(query="  SAVED  ")).total == 24


def test_owner_metadata_projects_no_bodies_or_scripts(owners):
    seed(owners, 1)
    traces = []
    for owner in owners:
        connection = (
            owner.conn if isinstance(owner, SubscriptionsDB) else owner.get_connection()
        )
        connection.set_trace_callback(traces.append)
    try:
        page = catalog_for(owners).read_page(ArtifactScope())
    finally:
        for owner in owners:
            connection = (
                owner.conn
                if isinstance(owner, SubscriptionsDB)
                else owner.get_connection()
            )
            connection.set_trace_callback(None)
    assert page.total == 2
    selects = [
        query.lower()
        for query in traces
        if query.lstrip().lower().startswith(("select", "with"))
    ]
    assert selects
    assert all(
        "body_markdown" not in query
        and "turns_json" not in query
        and "audio_path" not in query
        and "select *" not in query
        for query in selects
    )
    assert catalog_for(owners).read_detail(page.items[0].key).body == "Live 0"
    assert catalog_for(owners).read_detail(page.items[1].key).body == "Saved 0"


@pytest.mark.parametrize("bad_limit", [0, 21, True, 1.5])
def test_owner_rejects_invalid_limit(owners, bad_limit):
    for owner in owners:
        with pytest.raises(ValueError):
            owner.read_artifact_window(
                ArtifactScope(), boundary=None, direction="after", limit=bad_limit
            )


@pytest.mark.parametrize("borrowed", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_snapshot_leaves_transaction_owner_in_control(owners, borrowed, failure):
    subs, _ = owners
    seed(owners, 1)

    @contextmanager
    def native():
        subs.conn.execute("BEGIN")
        try:
            yield subs.conn
        finally:
            subs.conn.rollback()

    with native() if borrowed else subs.transaction() as conn:
        conn.execute("UPDATE watchlists SET name='Pending'")
        depth = getattr(subs._local, "transaction_depth", 0)
        try:
            with subs.artifact_read_snapshot():
                assert subs.get_briefing(1) is not None
                subs.list_briefing_scripts(1)
                assert (
                    subs.read_artifact_window(
                        ArtifactScope(), boundary=None, direction="after", limit=20
                    ).total
                    == 1
                )
                if failure:
                    raise RuntimeError("abort read")
        except RuntimeError:
            pass
        assert conn.in_transaction
        assert subs._local.transaction_depth == depth
        conn.rollback()
    assert (
        subs.conn.execute("SELECT name FROM watchlists").fetchone()[0] == "Daily news"
    )


def test_revision_is_independent_of_requested_sort(owners):
    seed(owners, 1)
    catalog = catalog_for(owners)
    for row in catalog.read_page(ArtifactScope(sort="title")).items:
        assert row.revision == catalog.read_detail(row.key).revision


def test_ascii_title_order_literal_search_and_timestamp_fallback(owners):
    _, kept = owners
    names = ["éclair", "Zulu", "alpha", "Éclair", "a%literal"]
    keys = []
    for index, name in enumerate(names):
        keys.append(
            ArtifactKey(
                "kept_report",
                kept.create_kept_briefing(
                    source_briefing_id=index + 1,
                    watchlist_name=name,
                    body_markdown="body",
                    origin="manual",
                ),
            )
        )
    catalog = catalog_for(owners)
    page = catalog.read_page(ArtifactScope(sort="title"))
    assert [row.key for row in page.items] == [
        keys[4],
        keys[2],
        keys[1],
        keys[3],
        keys[0],
    ]
    assert [row.key for row in catalog.read_page(ArtifactScope(query="ÉCL")).items] == [
        keys[3]
    ]
    assert [row.key for row in catalog.read_page(ArtifactScope(query="%")).items] == [
        keys[4]
    ]
    times = [
        None,
        "2026-01-01T01:00:00+01:00",
        "2026-01-01 00:00:00",
        "invalid",
        "2026-01-01T00:00:00.999999Z",
    ]
    with kept.transaction() as conn:
        for key, timestamp in zip(keys, times):
            conn.execute(
                "UPDATE kept_briefings SET original_created_at=?, kept_at=? WHERE id=?",
                (timestamp, "2026-01-02 00:00:00", key.native_id),
            )
    page = catalog.read_page(ArtifactScope())
    assert [row.key for row in page.items] == [
        keys[0],
        keys[1],
        keys[2],
        keys[4],
        keys[3],
    ]
    assert (
        page.items[1].order_key[0]
        == page.items[2].order_key[0]
        == page.items[3].order_key[0]
    )
    assert page.items[-1].type_label == "Report"


@pytest.mark.parametrize("operation", ["page", "locate"])
def test_wal_writer_cannot_split_metadata_rows_or_locator(owners, operation):
    import threading

    subs, _ = owners
    keys = seed(owners, 2)
    target = keys[0]
    go = threading.Event()
    ready = threading.Event()
    done = threading.Event()
    errors = []
    callback_errors = []

    def writer():
        connection = sqlite3.connect(subs.db_path, timeout=1)
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            ready.set()
            assert go.wait(5)
            connection.execute("DELETE FROM briefings WHERE id=?", (target.native_id,))
            connection.execute(
                "INSERT INTO briefings(watchlist_id, status, created_at) SELECT id, 'complete', '2030-01-01' FROM watchlists LIMIT 1"
            )
            connection.commit()
        except Exception as exc:  # noqa: BLE001 - relay worker failures to the test thread
            errors.append(exc)
        finally:
            connection.close()
            ready.set()
            done.set()

    worker = threading.Thread(target=writer)
    fired = False

    def trace(statement):
        nonlocal fired
        wanted = (
            "SELECT COUNT(*)" if operation == "locate" else "SELECT id, title, status"
        )
        if (
            not fired
            and statement.lstrip().startswith("WITH metadata")
            and wanted in statement
        ):
            fired = True
            go.set()
            if not done.wait(5):
                callback_errors.append("writer blocked by browse")

    worker.start()
    try:
        assert ready.wait(5)
        subs.conn.set_trace_callback(trace)
        catalog = catalog_for(owners)
        page = (
            catalog.read_page(ArtifactScope())
            if operation == "page"
            else catalog.locate(ArtifactScope(), target)
        )
        assert page is not None
        assert [row.key for row in page.items] == keys
        assert page.total == 4 and page.start == 0
        assert fired and not errors and not callback_errors
        subs.conn.set_trace_callback(None)
        refreshed = catalog.read_page(ArtifactScope())
        assert refreshed.total == 4
        assert target not in [row.key for row in refreshed.items]
    finally:
        go.set()
        subs.conn.set_trace_callback(None)
        worker.join(5)
        assert not worker.is_alive()


def test_empty_end_cursor_recovers_once(owners, monkeypatch):
    from dataclasses import replace

    from tldw_chatbook.Library.library_artifacts_catalog import ArtifactReadError

    seed(owners, 1)
    catalog = catalog_for(owners)
    page = catalog.read_page(ArtifactScope())
    assert (
        catalog.read_page(page.scope, boundary=page.items[-1].order_key).items
        == page.items
    )
    assert (
        catalog.read_page(
            page.scope, boundary=page.items[0].order_key, direction="before"
        ).items
        == page.items
    )
    original = owners[0].read_artifact_window

    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        return replace(result, items=())

    monkeypatch.setattr(owners[0], "read_artifact_window", corrupt)
    with pytest.raises(ArtifactReadError):
        catalog.read_page(ArtifactScope())


def test_duplicate_owner_identities_are_rejected(owners, monkeypatch):
    from dataclasses import replace

    from tldw_chatbook.Library.library_artifacts_catalog import ArtifactReadError

    seed(owners, 2)
    original = owners[0].read_artifact_window

    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        return replace(result, items=(result.items[0], result.items[0]))

    monkeypatch.setattr(owners[0], "read_artifact_window", corrupt)
    with pytest.raises(ArtifactReadError):
        catalog_for(owners).read_page(ArtifactScope())


@pytest.mark.parametrize("stored_form", ["absolute", "relative"])
def test_selected_live_audio_uses_exact_owner_metadata_and_safe_existing_path(
    owners, tmp_path, monkeypatch, stored_form
):
    from tldw_chatbook.Subscriptions import briefing_audio

    subs, _ = owners
    seed(owners, 1)
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    audio_file = audio_root / "episode.wav"
    audio_file.write_bytes(b"audio")
    stored = str(audio_file) if stored_form == "absolute" else audio_file.name
    monkeypatch.setattr(briefing_audio, "briefing_audio_dir", lambda: audio_root)
    with subs.transaction() as conn:
        script = conn.execute(
            "INSERT INTO briefing_scripts(briefing_id, preset_name, status, roster_snapshot_json) VALUES (1, 'Cast', 'complete', '[]')"
        ).lastrowid
        conn.execute(
            "INSERT INTO briefing_audio(script_id, voice_snapshot_json, status, file_path) VALUES (?, '[]', 'complete', ?)",
            (script, stored),
        )
    catalog = catalog_for(owners)
    assert catalog.read_detail(ArtifactKey("live_report", 1)).can_play
    assert subs.get_artifact_audio_path(ArtifactKey("live_report", 1)) == stored
    audio_file.unlink()
    assert not catalog.read_detail(ArtifactKey("live_report", 1)).can_play
    with subs.transaction() as conn:
        conn.execute(
            "UPDATE briefing_audio SET file_path=?", (str(tmp_path / "unsafe.wav"),)
        )
    assert not catalog.read_detail(ArtifactKey("live_report", 1)).can_play


@pytest.mark.parametrize("direction", ["after", "before"])
def test_null_boundary_rejects_impossible_rank_envelope(owners, monkeypatch, direction):
    from dataclasses import replace

    from tldw_chatbook.Library.library_artifacts_catalog import ArtifactReadError

    seed(owners, 25)
    original = owners[0].read_artifact_window

    def corrupt(*args, **kwargs):
        window = original(*args, **kwargs)
        return replace(
            window, before_boundary=1 if direction == "after" else window.total - 1
        )

    monkeypatch.setattr(owners[0], "read_artifact_window", corrupt)
    with pytest.raises(ArtifactReadError):
        catalog_for(owners).read_page(ArtifactScope(), direction=direction)


@pytest.mark.parametrize("borrowed", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_kept_metadata_preserves_borrowed_and_nested_transactions(
    owners, borrowed, failure
):
    _, kept = owners
    seed(owners, 1)
    connection = kept.get_connection()

    @contextmanager
    def native():
        connection.execute("BEGIN")
        try:
            yield connection
        finally:
            connection.rollback()

    with native() if borrowed else kept.transaction() as cursor:
        cursor.execute("UPDATE kept_briefings SET watchlist_name='Pending'")
        depth = getattr(kept._local, "transaction_depth", 0)
        try:
            with kept.transaction():
                assert kept.get_kept_briefing(1) is not None
                kept.list_kept_scripts(1)
                assert (
                    kept.read_artifact_window(
                        ArtifactScope(), boundary=None, direction="after", limit=20
                    ).total
                    == 1
                )
                if failure:
                    raise RuntimeError("abort read")
        except RuntimeError:
            pass
        assert connection.in_transaction
        assert kept._local.transaction_depth == depth
        connection.rollback()
    assert kept.get_kept_briefing(1)["watchlist_name"] == "Saved news"
