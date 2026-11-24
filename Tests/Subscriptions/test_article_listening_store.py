"""Article admission, snapshot identity, source fences and independent bookmarks."""

from contextlib import closing
from dataclasses import replace
from importlib import import_module
import sqlite3

import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions.watchlist_bundle_service import WatchlistBundleService
from tldw_chatbook.TTS.controlled_wav_profile import WavBookmark


@pytest.fixture
def records(tmp_path):
    with closing(SubscriptionsDB(tmp_path / "subs.db")) as db:
        with db.transaction() as c:
            c.execute(
                "INSERT INTO subscriptions(id,name,type,source) VALUES (1,'a','rss','local')"
            )
            c.executemany(
                "INSERT INTO subscription_items(id,subscription_id,url,title,content) "
                "VALUES (?,1,?,'title','same article')",
                [(1, "a"), (2, "b"), (3, "c")],
            )
            c.execute("INSERT INTO watchlists(id,name) VALUES (1,'watch')")
            c.execute(
                "INSERT INTO watchlist_sources(watchlist_id,subscription_id) VALUES (1,1)"
            )
            c.execute(
                "INSERT INTO briefings(id,watchlist_id,status,body_markdown) "
                "VALUES (1,1,'complete','[item 3] [item 2] [item 1]')"
            )
            c.executemany(
                "INSERT INTO briefing_items(briefing_id,item_id,live_item_id,"
                "source_id,cited,citation_position,provenance_version) "
                "VALUES (1,?,?,1,1,?,2)",
                [(1, 1, 1), (2, 2, 0)],
            )
        yield db


def api():
    import importlib.util

    name = "tldw_chatbook.Subscriptions.article_listening_store"
    assert importlib.util.find_spec(name) is not None, "missing listening store"
    return import_module(name)


def setup_store(db):
    m = api()
    binding = [
        m.ListeningAuthority(
            m.listening_profile_key(db.db_path.with_name("config.toml")), "local", 7
        )
    ]
    store = m.ArticleListeningStore(db, current_authority=lambda: binding[0])
    voice = m.FrozenVoice("openai", "tts-1", "alloy")
    return m, binding, store, voice


def prepare(store, authority, voice, *, brief=False, item=1):
    method = store.prepare_brief if brief else store.prepare_item
    return method(
        1 if brief else item,
        authority=authority,
        voice=voice,
        converter="speech-v1",
        prepare_text=str.strip,
    )


def test_cas_clear_aba_restoration_and_explicit_authority(records):
    m, binding, store, voice = setup_store(records)
    a = binding[0]
    candidate = prepare(store, a, voice)
    session = store.replace_session(candidate, expected_revision=None)
    assert session.foreground.text == "same article"
    with closing(SubscriptionsDB(records.db_path)) as other:
        peer = m.ArticleListeningStore(other, current_authority=lambda: binding[0])
        assert peer.load_session() == session
        with pytest.raises(m.ListeningConflict):
            peer.replace_session(candidate, expected_revision=None)
        peer.clear(expected_revision=session.revision)
    assert store.load_session() is None
    with pytest.raises(m.ListeningConflict):
        store.replace_session(candidate, expected_revision=None)
    fresh = prepare(store, a, voice)
    binding[0] = replace(a, policy_epoch=8)
    with pytest.raises(m.ListeningConflict):
        store.replace_session(fresh, expected_revision=None)
    for item in [True, "server:watchlist_item:1", "local:subscription:1", 0]:
        with pytest.raises(ValueError):
            prepare(store, binding[0], voice, item=item)
    with pytest.raises(ValueError):
        replace(voice, voice="C:/Users/private/reference.wav")


def test_brief_admission_authorizes_before_body_read_and_does_not_publish_stale(
    records,
):
    m, binding, store, voice = setup_store(records)
    a = binding[0]
    candidate = prepare(store, a, voice, brief=True)
    assert [e.item_id for e in candidate.queue] == [2, 1]
    with records.transaction() as c:
        c.execute(
            "UPDATE briefing_items SET provenance_version=1,cited=0,citation_position=NULL"
        )
    legacy = prepare(store, a, voice, brief=True)
    assert [e.item_id for e in legacy.queue] == [None, 2, 1]
    saved = store.replace_session(legacy, expected_revision=None)
    assert saved.queue[0].availability == "unavailable"
    with pytest.raises(m.ListeningConflict):
        store.replace_session(candidate, expected_revision=saved.revision)
    replacement = prepare(store, a, voice, item=3)
    with records.transaction() as c:
        c.execute("DELETE FROM subscription_items WHERE id=3")
    with pytest.raises(m.ListeningConflict):
        store.replace_session(replacement, expected_revision=saved.revision)
    assert store.load_session() == saved


def test_checkpoints_share_only_same_source_audio_and_rollback_both_bookmarks(records):
    m, binding, store, voice = setup_store(records)
    a = binding[0]
    candidate = prepare(store, a, voice, brief=True)
    foreground = prepare(store, a, voice, item=2).foreground
    session = store.replace_session(
        replace(candidate, foreground=foreground), expected_revision=None
    )
    reservation = store.reserve_audio(
        session.foreground.entry_id, expected_revision=session.revision
    )
    metadata = m.AudioMetadata("a" * 64, 48000, 12, 12012, 48092)
    session = store.complete_audio(reservation, metadata)
    session = store.attach_audio(
        session.queue[0].entry_id,
        reservation.generation,
        expected_revision=session.revision,
    )
    with pytest.raises(m.ListeningConflict):
        store.attach_audio(
            session.queue[1].entry_id,
            reservation.generation,
            expected_revision=session.revision,
        )
    points = tuple(
        (entry.entry_id, WavBookmark("a" * 64, 48000, frame))
        for entry, frame in [(session.foreground, 8), (session.queue[0], 4)]
    )
    update = m.ListeningCheckpoint(
        a, session.incarnation, session.playback_generation, 0, True, "paused", points
    )
    session = store.checkpoint(update, expected_revision=session.revision)
    assert [session.foreground.bookmark.frame, session.queue[0].bookmark.frame] == [
        8,
        4,
    ]
    with pytest.raises(m.ListeningConflict):
        store.checkpoint(
            replace(
                update,
                playback_generation=session.playback_generation,
                bookmarks=(
                    (points[0][0], WavBookmark("a" * 64, 48000, 0)),
                    (points[1][0], WavBookmark("a" * 64, 48000, 13)),
                ),
            ),
            expected_revision=session.revision,
        )
    assert store.load_session() == session
    store.mark_audio_missing(reservation.generation)
    missing = store.load_session()
    assert missing.foreground.availability == "missing"
    assert missing.foreground.bookmark == points[0][1]
    next_job = store.reserve_audio(
        missing.foreground.entry_id, expected_revision=missing.revision, regenerate=True
    )
    assert next_job.file_key != reservation.file_key
    refreshed = store.complete_audio(next_job, replace(metadata, sha256="b" * 64))
    assert refreshed.foreground.bookmark.frame == 0
    assert refreshed.queue[0].bookmark == points[1][1]


def test_deletion_scrubs_sources_retires_queues_and_keeps_foreground(records):
    m, binding, store, voice = setup_store(records)
    a = binding[0]
    candidate = prepare(store, a, voice, brief=True)
    candidate = replace(
        candidate, foreground=prepare(store, a, voice, item=3).foreground
    )
    session = store.replace_session(candidate, expected_revision=None)
    job = store.reserve_audio(
        session.queue[0].entry_id, expected_revision=session.revision
    )
    service = WatchlistBundleService(records)
    service.remove_source(1, 1)
    assert store.load_session() == session
    with pytest.raises(sqlite3.OperationalError), records.transaction() as c:
        c.execute("DELETE FROM subscription_items WHERE id=2")
    assert store.load_session() == session
    with pytest.raises(RuntimeError), records.destructive_transaction() as c:
        c.execute("DELETE FROM subscription_items WHERE id=2")
        raise RuntimeError("rollback")
    assert store.load_session() == session
    with store.mutation_lease() as lease:
        with pytest.raises(m.ListeningBusy):
            records.delete_subscription(1)
        with (
            pytest.raises((sqlite3.Error, m.ListeningBusy)),
            records.transaction() as c,
        ):
            c.execute("DELETE FROM subscription_items WHERE id=2")
        with records.transaction() as c:
            c.execute("UPDATE subscription_items SET status='reviewed' WHERE id=2")
        records.delete_article(2, lease=lease)
    deleted = store.load_session()
    e = deleted.queue[0]
    assert (e.availability, e.text, e.item_id, e.bookmark) == (
        "unavailable",
        None,
        None,
        None,
    )
    assert store.pending_deletions() == ((job.generation, job.file_key),)
    with pytest.raises(m.ListeningConflict):
        store.complete_audio(job, m.AudioMetadata("b" * 64, 48000, 5, 12005, 48064))
    service.delete(1)
    kept = store.load_session()
    assert kept.foreground == deleted.foreground
    assert (kept.brief_id, kept.queue, kept.resume_queue) == (None, (), False)


def test_subprocess_death_releases_same_lock_and_revocation_pins_borrower(records):
    import subprocess
    import sys
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    from threading import Event

    m, binding, store, _voice = setup_store(records)
    child_code = """
from pathlib import Path
import sys
from tldw_chatbook.Subscriptions.article_listening_lease import hold_listening
with hold_listening(Path(sys.argv[1]),profile=sys.argv[2]) as lease:
    print("held",flush=True)
    sys.stdin.readline()
"""
    child = subprocess.Popen(
        [
            sys.executable,
            "-B",
            "-c",
            child_code,
            str(records.db_path),
            binding[0].profile,
        ],
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        text=True,
    )
    try:
        with ThreadPoolExecutor() as pool:
            assert pool.submit(child.stdout.readline).result(20).strip() == "held"
        with pytest.raises(m.ListeningBusy):
            with store.mutation_lease():
                pass
        from tldw_chatbook.Subscriptions.article_listening_lease import lock_path

        lock = lock_path(Path(records.db_path))
        identity = lock.stat().st_ino
    finally:
        child.kill()
        child.communicate(timeout=10)
    assert lock.stat().st_ino == identity
    entered, release = Event(), Event()
    with ThreadPoolExecutor() as pool:
        with store.mutation_lease() as lease:
            alias = (
                records.db_path.parent / "alias-parent" / ".." / records.db_path.name
            )
            with lease.borrow(alias, profile=binding[0].profile):
                pass

            def borrow():
                with lease.borrow(Path(records.db_path), profile=binding[0].profile):
                    entered.set()
                    assert release.wait(10)

            borrower = pool.submit(borrow)
            assert entered.wait(10)
            lease.revoke()
            with (
                pytest.raises(m.ListeningBusy),
                lease.borrow(Path(records.db_path), profile=binding[0].profile),
            ):
                pass
            with pytest.raises(m.ListeningBusy), store.mutation_lease():
                pass
            release.set()
            borrower.result(10)
    assert lock.stat().st_ino == identity


def test_lease_is_minted_physical_profile_custody_and_gate_drains(records, tmp_path):
    import asyncio
    from time import monotonic
    from concurrent.futures import ThreadPoolExecutor
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Subscriptions.article_listening_lease import (
        ListeningLease,
        listening_producer,
    )

    m, binding, store, _ = setup_store(records)
    with pytest.raises(TypeError):
        ListeningLease(store.database, "profile", object(), object())
    with store.mutation_lease() as lease:
        with (
            pytest.raises(m.ListeningBusy),
            lease.borrow(store.database, profile="different"),
        ):
            pass
        with (
            pytest.raises(m.ListeningBusy),
            lease.borrow(tmp_path / "other.db", profile=binding[0].profile),
        ):
            pass
        listening_producer._maintenance_close_admission()
        try:
            assert not asyncio.run(
                listening_producer._maintenance_drain(monotonic() + 0.01)
            )

            def try_hold():
                with store.mutation_lease():
                    pass

            with (
                ThreadPoolExecutor() as pool,
                pytest.raises(RecoveryRequired, match="runtime_producer_paused"),
            ):
                pool.submit(try_hold).result(5)
        finally:
            lease.revoke()
    assert asyncio.run(listening_producer._maintenance_drain(monotonic() + 1))
    listening_producer._maintenance_resume()


def test_empty_replacement_and_global_source_deletion_with_foreign_owner(records):
    m, binding, store, voice = setup_store(records)
    a = binding[0]
    saved = store.replace_session(
        prepare(store, a, voice, brief=True), expected_revision=None
    )
    empty = prepare(store, a, voice, brief=True)
    unavailable = m.EntryCandidate(None, None, None, None, None, None)
    with pytest.raises(m.ListeningConflict):
        store.replace_session(
            replace(empty, queue=(unavailable, unavailable)),
            expected_revision=saved.revision,
        )
    assert store.load_session() == saved
    with closing(SubscriptionsDB(records.db_path)) as other:
        with store.mutation_lease() as lease:
            with pytest.raises(m.ListeningBusy):
                other.delete_subscription(1)
            with pytest.raises(sqlite3.Error), other.transaction() as c:
                c.execute("DELETE FROM subscriptions WHERE id=1")
            other.delete_subscription(1, lease=lease)
    scrubbed = store.load_session()
    assert len(scrubbed.queue) == 2 and all(
        e.availability == "unavailable" for e in scrubbed.queue
    )
    assert all(
        e.voice is None and e.title is None and e.text is None for e in scrubbed.queue
    )


@pytest.mark.parametrize(
    "tamper",
    [
        "UPDATE listening_session SET profile='bogus'",
        "UPDATE listening_entries SET speech_hash=lower(hex(zeroblob(32)))",
        "UPDATE listening_entries SET voice='{}'",
        "UPDATE listening_entries SET item_id=1 WHERE slot=0",
    ],
)
def test_recovery_rejects_corrupt_listening_rows(records, tamper):
    from threading import Event
    from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate
    from tldw_chatbook.DB.recovery_operations import recovery_adapters

    m, binding, store, voice = setup_store(records)
    store.replace_session(
        prepare(store, binding[0], voice, brief=True), expected_revision=None
    )
    owner = next(o for o in recovery_adapters() if o.owner_id == "db.subscriptions")
    if tamper.startswith("DELETE"):
        with records.transaction() as c:
            c.execute("DELETE FROM listening_entries")
    with records.transaction() as c:
        c.execute(tamper)
    records.close()
    assert validate_candidate(owner, records.db_path, Event(), migrate=False) == (
        "invalid_listening_state",
    )


def test_shared_database_has_private_sessions_and_cleanup_by_selector(records):
    m, binding, first, voice = setup_store(records)
    a = binding[0]
    b = m.ListeningAuthority(
        m.listening_profile_key(records.db_path.with_name("other.toml")), "local", 0
    )
    with pytest.raises(ValueError):
        m.ListeningAuthority("display-name", "local", 0)
    first_saved = first.replace_session(
        prepare(first, a, voice), expected_revision=None
    )
    second = m.ArticleListeningStore(records, current_authority=lambda: b)
    assert second.load_session() is None
    second_saved = second.replace_session(
        prepare(second, b, voice), expected_revision=None
    )
    assert first.load_session() == first_saved
    first_job = first.reserve_audio(
        first_saved.foreground.entry_id, expected_revision=first_saved.revision
    )
    second_job = second.reserve_audio(
        second_saved.foreground.entry_id, expected_revision=second_saved.revision
    )
    with pytest.raises(m.ListeningConflict):
        second.reserve_audio(
            first_saved.foreground.entry_id, expected_revision=second_saved.revision
        )
    records.delete_article(1)
    assert first.pending_deletions() == ((first_job.generation, first_job.file_key),)
    assert second.pending_deletions() == ((second_job.generation, second_job.file_key),)
    assert first.load_session().foreground.availability == "unavailable"
    assert second.load_session().foreground.availability == "unavailable"
    first.clear(expected_revision=first.load_session().revision)
    assert first.load_session() is None and second.load_session() is not None


def test_memory_binding_is_explicit_and_private_to_one_database(tmp_path):
    m = api()
    a = m.ListeningAuthority(m.listening_profile_key(tmp_path / "cfg"), "local", 1)
    with (
        closing(SubscriptionsDB(":memory:")) as db,
        pytest.raises(ValueError, match="memory_listening_binding_required"),
    ):
        m.ArticleListeningStore(db, current_authority=lambda: a)
    with (
        closing(
            SubscriptionsDB(":memory:", memory_listening_binding=tmp_path)
        ) as first,
        closing(
            SubscriptionsDB(":memory:", memory_listening_binding=tmp_path)
        ) as second,
    ):
        f = m.ArticleListeningStore(first, current_authority=lambda: a)
        g = m.ArticleListeningStore(second, current_authority=lambda: a)
        assert f.database != g.database
        with (
            f.mutation_lease() as lease,
            pytest.raises(m.ListeningBusy),
            second.listening_transaction(lease=lease, profile=a.profile),
        ):
            pass


def test_pending_claim_can_be_retired_before_late_completion(records):
    m, binding, store, voice = setup_store(records)
    session = store.replace_session(
        prepare(store, binding[0], voice), expected_revision=None
    )
    claim = store.reserve_audio(
        session.foreground.entry_id, expected_revision=session.revision
    )
    with store.mutation_lease() as lease:
        store.check_reservation(claim, lease=lease)
        store.cancel_audio(claim, lease=lease)
        with pytest.raises(m.ListeningConflict):
            store.check_reservation(claim, lease=lease)
    with pytest.raises(m.ListeningConflict):
        store.complete_audio(claim, m.AudioMetadata("a" * 64, 48000, 5, 12005, 48064))
    assert store.pending_deletions() == ((claim.generation, claim.file_key),)


def test_live_id_and_source_have_to_agree_before_speech_preparation(records):
    _, binding, store, voice = setup_store(records)
    with records.transaction() as c:
        c.execute("UPDATE briefing_items SET live_item_id=3 WHERE item_id=2")
        c.execute("UPDATE subscription_items SET content='not selected' WHERE id=3")
    seen = []

    def prepare_text(text):
        seen.append(text)
        return text

    candidate = store.prepare_brief(
        1, authority=binding[0], voice=voice, converter="v1", prepare_text=prepare_text
    )
    assert seen == ["same article"]
    assert [e.item_id for e in candidate.queue] == [None, 1]


def test_cross_process_maintenance_waits_for_physical_listening_storage(
    records, tmp_path
):
    import os
    import subprocess
    import sys
    from queue import Queue
    from threading import Thread

    home = tmp_path / "private-home"
    home.mkdir(mode=0o700)
    env = dict(
        os.environ,
        HOME=str(home),
        USERPROFILE=str(home),
        TLDW_CONFIG_PATH=str(home / "settings.toml"),
    )
    holder_code = """
from pathlib import Path
import sys
from tldw_chatbook.Subscriptions.article_listening_lease import hold_listening
with hold_listening(Path(sys.argv[1]),profile=None):
    print("held",flush=True)
    sys.stdin.readline()
"""
    maintenance_code = """
from tldw_chatbook.Backup_Recovery.admission import AdmissionTimeout
from tldw_chatbook.Backup_Recovery.bootstrap import default_bootstrap_root
from tldw_chatbook.Backup_Recovery.control_records import admission_authority, UNBOUND_NAMESPACE
authority=admission_authority(default_bootstrap_root())
try:
    with authority.maintenance((UNBOUND_NAMESPACE,),0.5):
        print("exclusive")
except AdmissionTimeout:
    print("busy")
"""
    child = subprocess.Popen(
        [sys.executable, "-B", "-c", holder_code, str(records.db_path)],
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )

    def probe():
        result = subprocess.run(
            [sys.executable, "-B", "-c", maintenance_code],
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        return result.stdout.strip()

    try:
        result = Queue()
        Thread(target=lambda: result.put(child.stdout.readline()), daemon=True).start()
        assert result.get(timeout=20).strip() == "held"
        assert probe() == "busy"
    finally:
        child.kill()
        child.communicate(timeout=10)
    assert probe() == "exclusive"


def test_caught_nested_replacement_error_never_commits_a_partial_transition(records):
    m, binding, store, voice = setup_store(records)
    a = binding[0]
    saved = store.replace_session(prepare(store, a, voice), expected_revision=None)
    offered = prepare(store, a, voice, brief=True)
    # The final foreground lookup fails after the two valid queue inserts.
    with (
        store.mutation_lease() as lease,
        records.listening_transaction(lease=lease, profile=a.profile),
    ):
        with pytest.raises(m.ListeningConflict):
            store.replace_session(
                replace(
                    offered,
                    foreground=replace(
                        prepare(store, a, voice).foreground, item_id=999
                    ),
                ),
                expected_revision=saved.revision,
                lease=lease,
            )
    assert store.load_session() == saved


def test_approved_queue_limits_leave_room_for_independent_foreground(records):
    m, binding, store, voice = setup_store(records)
    candidate = prepare(store, binding[0], voice, brief=True)
    unit = replace(candidate.queue[0], text="x" * 8192)
    foreground = replace(unit, text="x" * (256 * 1024))
    queue = (unit,) * 256
    assert len(replace(candidate, queue=queue, foreground=foreground).queue) == 256
    with pytest.raises(ValueError):
        replace(candidate, queue=(*queue, unit))
    with pytest.raises(ValueError):
        replace(candidate, queue=(replace(unit, text=unit.text + "x"), *queue[1:]))
    with pytest.raises(ValueError):
        replace(foreground, text=foreground.text + "x")


def test_foreground_prepared_under_other_selector_cannot_enter_session(records):
    m, binding, store, voice = setup_store(records)
    other_authority = m.ListeningAuthority(
        m.listening_profile_key(records.db_path.with_name("foreign.toml")), "local", 7
    )
    other = m.ArticleListeningStore(records, current_authority=lambda: other_authority)
    foreign = prepare(other, other_authority, voice).foreground
    queue = prepare(store, binding[0], voice, brief=True)
    with pytest.raises(m.ListeningConflict):
        store.replace_session(
            replace(queue, foreground=foreign), expected_revision=None
        )
    assert store.load_session() is None
