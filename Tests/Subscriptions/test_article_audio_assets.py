"""Retained article WAVs, private publication and source-owned cleanup."""

import errno
import hashlib
import io
import os
import shutil
import struct
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import replace
from importlib import import_module, util

import pytest

from Tests.Subscriptions.test_article_listening_store import (
    prepare,
    records,  # noqa: F401 - shared real SQLite fixture
    setup_store,
)
from tldw_chatbook.Backup_Recovery.native_files import create_private_directory
from tldw_chatbook.Subscriptions import article_listening_models as models

native_effects = pytest.mark.skipif(
    os.name != "nt",
    reason="same-object article file effects require native Windows NTFS",
)


def wav(frames=8, *, extra=b""):
    samples = bytes([1, 2, 3, 4]) * frames + bytes(48000)
    fmt = struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 2, 48000, 192000, 4, 16)
    body = b"WAVE" + extra + fmt + struct.pack("<4sI", b"data", len(samples)) + samples
    return struct.pack("<4sI", b"RIFF", len(body)) + body


def metadata(data, frames=8):
    return models.AudioMetadata(
        hashlib.sha256(data).hexdigest(), 48000, frames, frames + 12000, len(data)
    )


def api():
    name = "tldw_chatbook.Subscriptions.article_audio_assets"
    assert util.find_spec(name) is not None, "article asset owner missing"
    return import_module(name)


def intercept_handle(monkeypatch, method, hook):
    """Inject an observation at an actual held-file effect in the asset owner."""
    original = api().held_regular_file

    @contextmanager
    def instrumented(*args, **kwargs):
        with original(*args, **kwargs) as held:

            class Proxy:
                def __getattr__(self, name):
                    actual = getattr(held, name)
                    if name == method:
                        return lambda *values: hook(actual, *values)
                    return actual

            yield Proxy()

    monkeypatch.setattr(api(), "held_regular_file", instrumented)


@pytest.fixture
def listening(records, tmp_path):  # noqa: F811
    _, authorities, store, voice = setup_store(records)
    queue = prepare(store, authorities[0], voice, brief=True)
    same_item = prepare(store, authorities[0], voice, item=2).foreground
    session = store.replace_session(
        replace(queue, foreground=same_item), expected_revision=None
    )
    root = tmp_path / "private-profile"
    create_private_directory(root)
    return store, session, root


def publish(store, session, root, entry, *, regenerate=False, frames=8):
    a = api()
    claim = store.reserve_audio(
        entry.entry_id, expected_revision=session.revision, regenerate=regenerate
    )
    owner = a.ArticleAudioAssets(store, profile_data_root=root, reservation=claim)
    data = wav(frames)
    candidate = a.GeneratedArticleAudio(data, metadata(data, frames).sha256, frames)
    return owner.publish(
        candidate, source_item_id=entry.item_id, generation=claim.generation
    )


def test_shared_validator_checks_complete_pinned_stream_and_read_budget():
    assert hasattr(
        models, "validate_article_audio"
    ), "shared article WAV validation missing"
    data = wav()
    stream = io.BytesIO(data)
    stream.seek(9)
    models.validate_article_audio(stream, metadata(data))
    for bad, m in [
        (data[:-1], metadata(data)),
        (data, replace(metadata(data), sha256="0" * 64)),
        (data[:-1] + b"x", metadata(data[:-1] + b"x")),
        (
            data.replace(b"WAVE", b"wave", 1),
            metadata(data.replace(b"WAVE", b"wave", 1)),
        ),
    ]:
        with pytest.raises(ValueError, match="invalid_article_audio"):
            models.validate_article_audio(io.BytesIO(bad), m)
    many = wav(extra=struct.pack("<4sI", b"JUNK", 0) * 1100)
    with pytest.raises(ValueError, match="invalid_article_audio"):
        models.validate_article_audio(io.BytesIO(many), metadata(many))


@native_effects
def test_physical_share_bookmarks_and_distinct_source_ownership(listening):
    store, session, root = listening
    owner = api().ArticleAudioAssets(store, profile_data_root=root)
    first = publish(store, session, root, session.foreground)
    assert first.wav.path.parent == root / "article_audio"
    assert first.wav.path.read_bytes() == wav()
    assert first.byte_size == len(wav())
    assert owner.read_verified(first.generation) == first
    session = store.load_session()
    assert (
        owner.reuse(session.queue[0].entry_id, expected_revision=session.revision)
        == first
    )
    session = store.load_session()
    assert (
        owner.reuse(session.queue[1].entry_id, expected_revision=session.revision)
        is None
    )
    second = publish(store, session, root, session.queue[1])
    assert first.generation != second.generation
    assert first.wav.path != second.wav.path
    assert first.wav.sha256 == second.wav.sha256
    session = store.load_session()
    points = (
        (session.foreground.entry_id, models.WavBookmark(first.wav.sha256, 48000, 3)),
        (session.queue[0].entry_id, models.WavBookmark(first.wav.sha256, 48000, 6)),
    )
    session = store.checkpoint(
        models.ListeningCheckpoint(
            session.authority,
            session.incarnation,
            session.playback_generation,
            0,
            True,
            "completed",
            points,
        ),
        expected_revision=session.revision,
    )
    first.wav.path.write_bytes(b"damaged")
    assert owner.prune(2**62).removed == 0
    assert owner.reconcile().missing == 1
    missing = store.load_session()
    assert (missing.foreground.bookmark.frame, missing.queue[0].bookmark.frame) == (
        3,
        6,
    )
    assert missing.foreground.availability == "missing"
    new = publish(store, missing, root, missing.foreground, regenerate=True)
    ready = store.load_session()
    assert ready.foreground.bookmark.frame == 0
    assert ready.queue[0].bookmark == points[1][1]
    assert new.wav.path != first.wav.path
    assert second.wav.path.exists()


def test_forged_reservation_and_unadmitted_borrow_cannot_publish(listening):
    store, session, root = listening
    a = api()
    claim = store.reserve_audio(
        session.foreground.entry_id, expected_revision=session.revision
    )
    data = wav()
    candidate = a.GeneratedArticleAudio(data, metadata(data).sha256, 8)

    def attempt(reservation, lease=None):
        return a.ArticleAudioAssets(
            store, profile_data_root=root, reservation=reservation, lease=lease
        ).publish(candidate, source_item_id=claim.item_id, generation=claim.generation)

    with pytest.raises(models.ListeningConflict):
        attempt(replace(claim))
    with store.mutation_lease() as lease, pytest.raises(store_module().ListeningBusy):
        attempt(claim, lease)
    assert not (root / "article_audio").exists()
    with store.db.destructive_transaction() as c:
        c.execute("DELETE FROM subscription_items WHERE id=?", (claim.item_id,))
    with pytest.raises(models.ListeningConflict):
        attempt(claim)
    assert not (root / "article_audio").exists()


def store_module():
    return import_module("tldw_chatbook.Subscriptions.article_listening_store")


@pytest.mark.parametrize("boundary", ["unsealed", "staged", "published"])
@native_effects
def test_dead_physical_publisher_leaves_only_owned_retryable_evidence(
    listening, boundary
):
    store, session, root = listening
    a = api()
    code = """
import os, sys
from pathlib import Path
from Tests.Subscriptions.test_article_audio_assets import wav, metadata
from Tests.Subscriptions.test_article_listening_store import setup_store
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions import article_audio_assets as a
_, _, store, _ = setup_store(SubscriptionsDB(Path(sys.argv[1])))
session = store.load_session()
claim = store.reserve_audio(session.foreground.entry_id, expected_revision=session.revision)
real_seal = store.seal_audio_custody
def die(*args, **kwargs):
    if sys.argv[3] == 'staged':
        real_seal(*args, **kwargs)
    os._exit(54)
if sys.argv[3] == 'published':
    store.complete_audio = die
else:
    store.seal_audio_custody = die
data = wav()
a.ArticleAudioAssets(store, profile_data_root=Path(sys.argv[2]), reservation=claim).publish(
    a.GeneratedArticleAudio(data, metadata(data).sha256, 8), source_item_id=claim.item_id, generation=claim.generation)
"""
    child = subprocess.run(
        [sys.executable, "-B", "-c", code, str(store.database), str(root), boundary],
        timeout=60,
        capture_output=True,
        text=True,
    )
    assert child.returncode == 54, child.stderr
    names = list((root / "article_audio").iterdir())
    assert len(names) == 1 and names[0].name.endswith(
        ".wav" if boundary == "published" else ".stage"
    )
    assert store.load_session().foreground.artifact is None
    receipt = a.ArticleAudioAssets(store, profile_data_root=root).reconcile()
    assert receipt.interrupted == 1
    if boundary == "unsealed":
        assert receipt.uncertain == receipt.retry == 1 and names[0].exists()
    else:
        assert receipt.removed == 1 and not names[0].exists()
    assert store.load_session().foreground.artifact is None


@native_effects
def test_collision_never_transfers_or_unlinks_existing_name(listening):
    store, session, root = listening
    a = api()
    claim = store.reserve_audio(
        session.foreground.entry_id, expected_revision=session.revision
    )
    owner = a.ArticleAudioAssets(store, profile_data_root=root, reservation=claim)
    create_private_directory(root / "article_audio")
    foreign = root / "article_audio" / claim.file_key
    foreign.write_bytes(b"prior owner")
    data = wav()
    with pytest.raises(a.ArticleAudioError, match="collision"):
        owner.publish(
            a.GeneratedArticleAudio(data, metadata(data).sha256, 8),
            source_item_id=claim.item_id,
            generation=claim.generation,
        )
    assert owner.reconcile().uncertain >= 1
    assert foreign.read_bytes() == b"prior owner"
    assert store.load_session().foreground.artifact is None


@native_effects
def test_missing_file_retains_bookmark_and_linked_file_refuses_cleanup(listening):
    store, session, root = listening
    a = api()
    first = publish(store, session, root, session.foreground)
    owner = a.ArticleAudioAssets(store, profile_data_root=root)
    external = root / "outside.wav"
    first.wav.path.rename(external)
    assert owner.reconcile().missing == 1
    missing = store.load_session()
    assert missing.foreground.artifact == first.generation
    assert missing.foreground.bookmark.sha256 == first.wav.sha256
    os.link(external, first.wav.path)
    store.db.delete_article(first.source_item_id)
    assert owner.reconcile().retry == 1
    assert external.read_bytes() == wav() and first.wav.path.exists()
    assert store.pending_deletions()


@native_effects
def test_retention_uses_last_use_then_generation_and_excludes_references(
    listening, monkeypatch
):
    store, session, root = listening
    a = api()
    owner = a.ArticleAudioAssets(store, profile_data_root=root)
    first = publish(store, session, root, session.foreground)
    second = publish(
        store, store.load_session(), root, session.foreground, regenerate=True
    )
    third = publish(
        store, store.load_session(), root, session.foreground, regenerate=True
    )
    with store.db.transaction() as c:
        c.execute("UPDATE article_audio SET last_used=1000000")
    monkeypatch.setattr(a, "MAX_UNREFERENCED_BYTES", first.byte_size)
    assert owner.prune(1000000).removed == 1
    assert (
        not first.wav.path.exists()
        and second.wav.path.is_file()
        and third.wav.path.is_file()
    )
    assert owner.prune(1000000 + 7 * 86400).removed == 1
    assert third.wav.path.is_file()
    store.clear(expected_revision=store.load_session().revision)
    assert owner.prune(1000000 + 7 * 86400).removed == 1
    assert not third.wav.path.exists()


@native_effects
def test_user_read_refreshes_retention_but_integrity_scan_preserves_age(
    listening, monkeypatch
):
    store, session, root = listening
    a = api()
    owner = a.ArticleAudioAssets(store, profile_data_root=root)
    first = publish(store, session, root, session.foreground)
    second = publish(
        store, store.load_session(), root, session.foreground, regenerate=True
    )
    with store.db.transaction() as c:
        now = c.execute("SELECT unixepoch()").fetchone()[0]
        c.execute("UPDATE article_audio SET last_used=? + generation", (now - 86400,))
    aged = {r["generation"]: r["last_used"] for r in store.audio_inventory()}
    assert owner.reconcile().missing == 0
    assert {r["generation"]: r["last_used"] for r in store.audio_inventory()} == aged
    assert owner.read_verified(first.generation) == first
    store.clear(expected_revision=store.load_session().revision)
    monkeypatch.setattr(a, "MAX_UNREFERENCED_BYTES", first.byte_size)
    assert owner.prune(now).removed == 1
    assert first.wav.path.is_file() and not second.wav.path.exists()


@native_effects
def test_cleanup_obeys_commit_retries_and_preserves_rowless_files(
    listening, monkeypatch
):
    store, session, root = listening
    a = api()
    owner = a.ArticleAudioAssets(store, profile_data_root=root)
    first = publish(store, session, root, session.foreground)
    orphan = first.wav.path.with_name("a" * 32 + ".wav")
    orphan.write_bytes(b"foreign")
    with pytest.raises(RuntimeError, match="rollback"):
        with store.db.destructive_transaction() as c:
            c.execute(
                "DELETE FROM subscription_items WHERE id=?", (first.source_item_id,)
            )
            with pytest.raises(models.ListeningConflict, match="committed"):
                owner.reconcile()
            raise RuntimeError("rollback")
    assert first.wav.path.read_bytes() == wav()
    store.db.delete_article(first.source_item_id)
    with monkeypatch.context() as patch:

        def refuse(original):
            raise PermissionError("sharing")

        intercept_handle(patch, "delete", refuse)
        assert owner.reconcile().retry == 1
    assert store.pending_deletions()
    receipt = owner.reconcile()
    assert receipt.removed == 1 and receipt.uncertain >= 1
    assert not first.wav.path.exists() and orphan.read_bytes() == b"foreign"
    assert not store.pending_deletions()


def test_complete_wav_cap_precedes_hashing_and_parsing():
    with pytest.raises(ValueError, match="size"):
        models.GeneratedArticleAudio(bytes(64 * 1024 * 1024 + 1), "0" * 64, 8)
    data = wav()

    class TooLarge:
        def seek(self, offset, whence):
            return 64 * 1024 * 1024 + 1

        def read(self, size):
            pytest.fail("oversize file was read")

    with pytest.raises(ValueError, match="invalid_article_audio"):
        models.validate_article_audio(TooLarge(), metadata(data))
    with pytest.raises(ValueError):
        models.GeneratedArticleAudio(bytearray(data), metadata(data).sha256, 8)


def owner_for(listening, lease=None, claim=None):
    store, session, root = listening
    if claim is None:
        claim = store.reserve_audio(
            session.foreground.entry_id, expected_revision=session.revision, lease=lease
        )
    data = wav()
    a = api()
    return (
        a.ArticleAudioAssets(
            store, profile_data_root=root, reservation=claim, lease=lease
        ),
        claim,
        a.GeneratedArticleAudio(data, metadata(data).sha256, 8),
    )


@native_effects
def test_disk_full_and_cas_race_preserve_bounded_retryable_state(
    listening, monkeypatch
):
    store, session, root = listening
    a = api()
    owner, claim, data = owner_for(listening)

    def full(*args, **kwargs):
        raise OSError(errno.ENOSPC, str(root))

    with monkeypatch.context() as patch:
        patch.setattr(a, "held_regular_file", full)
        with pytest.raises(a.ArticleAudioError, match="^article_audio_disk_full$"):
            owner.publish(
                data, source_item_id=claim.item_id, generation=claim.generation
            )
    assert owner.reconcile().interrupted == 1
    owner, claim, data = owner_for(listening)
    real = store.complete_audio

    def source_changed(claim, m, *, lease, custody_witness):
        store.clear(expected_revision=session.revision, lease=lease)
        return real(claim, m, lease=lease, custody_witness=custody_witness)

    monkeypatch.setattr(store, "complete_audio", source_changed)
    with pytest.raises(models.ListeningConflict):
        owner.publish(data, source_item_id=claim.item_id, generation=claim.generation)
    assert store.load_session() is None
    assert (owner.root / claim.file_key).is_file()
    assert owner.reconcile().removed == 1
    assert not (owner.root / claim.file_key).exists()


def test_idle_claim_cannot_revive_after_same_path_restore(listening):
    store, _session, root = listening
    owner, claim, data = owner_for(listening)
    path = store.database
    store.db.close()
    replacement = root / "snapshot.db"
    shutil.copyfile(path, replacement)
    replacement.replace(path)
    with pytest.raises(models.ListeningConflict):
        owner.publish(data, source_item_id=claim.item_id, generation=claim.generation)
    assert not owner.root.exists()


@native_effects
def test_borrowed_worker_prevents_same_holder_sweep_during_publication(
    listening, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor

    store, _session, _root = listening
    owner, claim, data = owner_for(listening)
    paths = (
        owner.root,
        owner.root / claim.file_key,
        owner.root / ("." + claim.file_key + ".stage"),
    )
    with store.mutation_lease(related_paths=paths) as lease:
        active, _, _ = owner_for(listening, lease, claim)

        def intercept(original, target):
            with pytest.raises(store_module().ListeningBusy):
                active.reconcile()
            with pytest.raises(store_module().ListeningBusy):
                owner.reconcile()
            original(target)

        intercept_handle(monkeypatch, "rename_new", intercept)
        with ThreadPoolExecutor() as pool:
            asset = pool.submit(
                active.publish,
                data,
                source_item_id=claim.item_id,
                generation=claim.generation,
            ).result(30)
    assert asset.wav.path.read_bytes() == wav()


@native_effects
def test_cleanup_unlinks_only_in_pinned_root_during_rename(listening, monkeypatch):
    store, session, root = listening
    a = api()
    audio = publish(store, session, root, session.foreground)
    store.db.delete_article(audio.source_item_id)
    owner = a.ArticleAudioAssets(store, profile_data_root=root)
    check = owner._check_directory
    saved = root / "saved"
    denied = []

    def swap_after_check(directory):
        check(directory)
        if not saved.exists():
            try:
                owner.root.rename(saved)
            except PermissionError as error:
                denied.append(error.winerror if os.name == "nt" else error.errno)
                raise
            create_private_directory(owner.root)
            audio.wav.path.write_bytes(b"new tree")

    monkeypatch.setattr(owner, "_check_directory", swap_after_check)
    receipt = owner.reconcile()
    if denied:
        assert os.name == "nt" and denied[0] in (5, 32)
        assert receipt.retry == 1 and store.pending_deletions()
        assert audio.wav.path.read_bytes() == wav()
    else:
        assert audio.wav.path.read_bytes() == b"new tree"


@pytest.mark.parametrize(
    "bad", ["../other.wav", "CON.wav", "D:foo.wav", "bad.wav:ads", "A" * 32 + ".wav"]
)
def test_unsafe_restored_name_is_refused_before_path_work(listening, bad):
    store, _session, root = listening
    owner, claim, _ = owner_for(listening)
    with store.db.transaction() as c:
        c.execute("PRAGMA ignore_check_constraints=ON")
        c.execute(
            "UPDATE article_audio SET file_key=? WHERE generation=?",
            (bad, claim.generation),
        )
        c.execute("PRAGMA ignore_check_constraints=OFF")
    with pytest.raises(api().ArticleAudioError, match="unsafe_article_audio_key"):
        owner.reconcile()
    assert not (root / "article_audio").exists()


@native_effects
def test_cleanup_leaves_late_tombstone_outside_admitted_names(listening, monkeypatch):
    store, session, root = listening
    owner, claim, _ = owner_for(listening)
    original = store.pending_deletions
    paths = (
        owner.root,
        owner.root / claim.file_key,
        owner.root / ("." + claim.file_key + ".stage"),
    )
    late = []
    store.cancel_audio(claim)
    with store.mutation_lease(related_paths=paths) as lease:
        owner = api().ArticleAudioAssets(store, profile_data_root=root, lease=lease)

        def additional_tombstone():
            other = store.reserve_audio(
                session.queue[1].entry_id,
                expected_revision=session.revision,
                lease=lease,
            )
            store.cancel_audio(other, lease=lease)
            late.append((other.generation, other.file_key))
            return original()

        monkeypatch.setattr(store, "pending_deletions", additional_tombstone)
        assert owner.reconcile().removed == 1
    assert original() == tuple(late)


def test_ambiguous_two_names_are_preserved_for_owned_reconciliation(listening):
    owner, claim, _ = owner_for(listening)
    create_private_directory(owner.root)
    final = owner.root / claim.file_key
    stage = owner.root / ("." + claim.file_key + ".stage")
    final.write_bytes(b"prior final")
    stage.write_bytes(b"interrupted")
    report = owner.reconcile()
    assert report.uncertain and report.retry
    assert final.read_bytes() == b"prior final" and stage.read_bytes() == b"interrupted"
    assert owner.store.pending_deletions() == ((claim.generation, claim.file_key),)


@native_effects
def test_absent_profile_directory_projects_db_only_restore_as_missing(listening):
    store, session, root = listening
    audio = publish(store, session, root, session.foreground)
    root.rename(root.with_name("saved-outside-restored-profile"))
    owner = api().ArticleAudioAssets(store, profile_data_root=root)
    assert owner.reconcile().missing == 1
    assert not root.exists()
    assert store.load_session().foreground.bookmark.sha256 == audio.wav.sha256


@pytest.mark.parametrize(
    ("changed", "revoked"),
    [("stage", None), ("final", None), ("stage", "cancel"), ("final", "delete")],
)
@native_effects
def test_effect_handle_keeps_issued_leaf_through_publication(
    listening, monkeypatch, changed, revoked
):
    owner, claim, data = owner_for(listening)
    stage = owner.root / ("." + claim.file_key + ".stage")
    final = owner.root / claim.file_key
    source = stage if changed == "stage" else final
    paths = (owner.root, stage, final)
    observations = []
    with owner.store.mutation_lease(related_paths=paths) as lease:
        active, _, _ = owner_for(listening, lease, claim)

        def check_native_boundary(rename, target):
            if changed == "final":
                rename(target)
            with pytest.raises(PermissionError):
                source.rename(owner.root.parent / "displaced-private-object")
            observations.append(source.name)
            if revoked == "cancel":
                owner.store.cancel_audio(claim, lease=lease)
            elif revoked == "delete":
                owner.store.db.delete_article(claim.item_id, lease=lease)
            if changed == "stage":
                rename(target)

        intercept_handle(monkeypatch, "rename_new", check_native_boundary)
        if revoked:
            with pytest.raises(models.ListeningConflict):
                active.publish(
                    data, source_item_id=claim.item_id, generation=claim.generation
                )
        else:
            active.publish(
                data, source_item_id=claim.item_id, generation=claim.generation
            )
    assert observations == [source.name]
    assert not (owner.root.parent / "displaced-private-object").exists()
    assert final.read_bytes() == wav()
    if revoked:
        assert owner.reconcile().removed == 1 and not final.exists()


@native_effects
def test_linked_tree_refuses_publication_without_touching_target(listening):
    owner, claim, data = owner_for(listening)
    external = owner.root.parent / "outside-tree"
    create_private_directory(external)
    if os.name == "nt":
        import _winapi

        _winapi.CreateJunction(str(external), str(owner.root))
    else:
        owner.root.symlink_to(external, target_is_directory=True)
    with pytest.raises(api().ArticleAudioError):
        owner.publish(data, source_item_id=claim.item_id, generation=claim.generation)
    assert not list(external.iterdir())


@native_effects
def test_post_rename_flush_error_preserves_journal_and_unattached_file(
    listening, monkeypatch
):
    owner, claim, data = owner_for(listening)
    a = api()

    def interrupted(rename, target):
        rename(target)
        raise FileExistsError

    intercept_handle(monkeypatch, "rename_new", interrupted)
    with pytest.raises(a.ArticleAudioError, match="publication_uncertain"):
        owner.publish(data, source_item_id=claim.item_id, generation=claim.generation)
    assert (owner.root / claim.file_key).read_bytes() == wav()
    assert owner.store.audio_inventory(claim.generation)[0]["status"] == "pending"
    assert owner.store.load_session().foreground.artifact is None
    assert owner.reconcile().removed == 1


def test_absent_generation_never_reads_another_ready_article(listening):
    store, session, root = listening
    claim = store.reserve_audio(
        session.foreground.entry_id, expected_revision=session.revision
    )
    store.complete_audio(claim, metadata(wav()))
    with pytest.raises(ValueError):
        api().ArticleAudioAssets(store, profile_data_root=root).read_verified(None)


def test_wrong_lease_refused_before_any_pending_delete_work(listening):
    store, _, root = listening
    owner, claim, _ = owner_for(listening)
    store.cancel_audio(claim)
    create_private_directory(owner.root)
    leaf = owner.root / claim.file_key
    leaf.write_bytes(b"private")
    with pytest.raises(ValueError, match="invalid_listening_lease"):
        api().ArticleAudioAssets(
            store, profile_data_root=root, lease=object()
        ).reconcile()
    assert leaf.read_bytes() == b"private"
    assert store.pending_deletions()


@native_effects
def test_borrowed_sweep_preserves_preparing_claim(listening):
    store, session, root = listening
    claim = store.reserve_audio(
        session.foreground.entry_id, expected_revision=session.revision
    )
    directory = root / "article_audio"
    paths = (
        directory,
        directory / claim.file_key,
        directory / ("." + claim.file_key + ".stage"),
    )
    with store.mutation_lease(related_paths=paths) as lease:
        owner, _, data = owner_for(listening, lease, claim)
        assert owner.reconcile().interrupted == 0
        audio = owner.publish(
            data, source_item_id=claim.item_id, generation=claim.generation
        )
    assert audio.wav.path.read_bytes() == wav()


@pytest.mark.parametrize(
    "reason",
    [
        "cancel-never-staged",
        pytest.param("delete-replaced", marks=native_effects),
        pytest.param("ttl-missing", marks=native_effects),
        pytest.param("changed-content", marks=native_effects),
    ],
)
def test_reopened_cleanup_preserves_unproven_singleton(listening, reason):
    store, session, root = listening
    if reason == "cancel-never-staged":
        owner, claim, _ = owner_for(listening)
        store.cancel_audio(claim)
        create_private_directory(owner.root)
        path = owner.root / claim.file_key
        path.write_bytes(wav())
    else:
        audio = publish(store, session, root, session.foreground)
        path = audio.wav.path
        if reason == "changed-content":
            path.write_bytes(b"other content")
        else:
            path.rename(root / "previous-object")
            path.write_bytes(wav())
        if reason == "ttl-missing":
            store.mark_audio_missing(audio.generation)
            store.clear(expected_revision=store.load_session().revision)
        else:
            store.db.delete_article(audio.source_item_id)
    prior = path.read_bytes()
    store.db.close()
    _, _, reopened, _ = setup_store(store.db)
    owner = api().ArticleAudioAssets(reopened, profile_data_root=root)
    report = owner.prune(2**62)
    assert report.uncertain == 1 and report.retry == 1 and not report.removed
    assert path.read_bytes() == prior and reopened.pending_deletions()


@native_effects
def test_generation_witness_is_sealed_and_preserved_on_source_delete(listening):
    store, session, root = listening
    audio = publish(store, session, root, session.foreground)
    row = store.audio_inventory(audio.generation)[0]
    assert "custody_witness" in row.keys()
    witness = row["custody_witness"]
    assert len(witness) == 64 and all(c in "0123456789abcdef" for c in witness)
    store.db.delete_article(audio.source_item_id)
    row = store.audio_inventory(audio.generation)[0]
    assert row["custody_witness"] == witness and row["sha256"] is None
    store.db.close()
    _, _, reopened, _ = setup_store(store.db)
    assert (
        api().ArticleAudioAssets(reopened, profile_data_root=root).reconcile().removed
        == 1
    )
    assert not audio.wav.path.exists()


@native_effects
def test_absent_name_requires_successful_directory_barrier_on_retry(
    listening, monkeypatch
):
    store, session, root = listening
    audio = publish(store, session, root, session.foreground)
    store.db.delete_article(audio.source_item_id)
    audio.wav.path.unlink()  # State after a successful disposition with an uncertain flush.
    owner = api().ArticleAudioAssets(store, profile_data_root=root)
    with monkeypatch.context() as patch:

        def unconfirmed(fd):
            raise OSError("barrier_failed")

        patch.setattr(api(), "flush_regular_parent", unconfirmed, raising=False)
        assert owner.reconcile().retry == 1
    assert store.pending_deletions()
    assert owner.reconcile().removed == 1 and not store.pending_deletions()


def test_unqualified_volume_refuses_before_creating_article_tree(
    listening, monkeypatch
):
    owner, claim, data = owner_for(listening)
    if os.name == "nt":

        def unavailable(_parent):
            raise OSError("regular_file_effects_unqualified")

        monkeypatch.setattr(api(), "require_regular_file_effects", unavailable)
    with pytest.raises(
        api().ArticleAudioError, match="article_audio_effects_unqualified"
    ):
        owner.publish(data, source_item_id=claim.item_id, generation=claim.generation)
    assert not owner.root.exists()


@native_effects
@pytest.mark.parametrize(
    "reason", ["bad-digest", "cancel", "source-deletion", "profile-switch"]
)
def test_live_rejected_stage_disposed_after_committed_revocation(
    listening, reason, monkeypatch
):
    store, _, _ = listening
    owner, claim, candidate = owner_for(listening)
    if reason == "bad-digest":
        candidate = replace(candidate, sha256="0" * 64)
    else:
        original = store.seal_audio_custody

        def revoke(*args, **kwargs):
            if reason == "cancel":
                store.cancel_audio(claim, lease=kwargs["lease"])
            elif reason == "source-deletion":
                store.db.delete_article(claim.item_id, lease=kwargs["lease"])
            else:
                monkeypatch.setattr(
                    store,
                    "_current",
                    lambda: replace(claim.authority, profile="f" * 24),
                )
            return original(*args, **kwargs)

        monkeypatch.setattr(store, "seal_audio_custody", revoke)
    observed = []

    def dispose(actual):
        assert not store.db.conn.in_transaction
        row = store.db.conn.execute(
            "SELECT status,custody_witness FROM article_audio WHERE generation=?",
            (claim.generation,),
        ).fetchone()
        observed.append((row[0], len(row[1])))
        actual()

    intercept_handle(monkeypatch, "delete", dispose)
    with pytest.raises((models.ListeningConflict, api().ArticleAudioError)):
        owner.publish(
            candidate, source_item_id=claim.item_id, generation=claim.generation
        )
    assert observed == [("pending_delete", 64)]
    assert list(owner.root.iterdir()) == []


@native_effects
@pytest.mark.parametrize(
    "after_failure", ["retry-stage", "move-to-final", "replace-stage"]
)
def test_failed_rejected_stage_disposition_only_retries_proven_stage(
    listening, after_failure, monkeypatch
):
    store, _, root = listening
    owner, claim, candidate = owner_for(listening)

    def failure(actual):
        raise OSError("disposition_failed")

    with monkeypatch.context() as patch:
        intercept_handle(patch, "delete", failure)
        with pytest.raises(api().ArticleAudioError, match="storage_unavailable"):
            owner.publish(
                replace(candidate, sha256="0" * 64),
                source_item_id=claim.item_id,
                generation=claim.generation,
            )
    row = store.audio_inventory(claim.generation)[0]
    assert row["status"] == "pending_delete" and len(row["custody_witness"]) == 64
    stage = owner.root / ("." + claim.file_key + ".stage")
    final = owner.root / claim.file_key
    if after_failure == "move-to-final":
        stage.rename(final)
    elif after_failure == "replace-stage":
        stage.rename(root / "original-stage")
        stage.write_bytes(candidate.wav_bytes)
    store.db.close()
    _, _, reopened, _ = setup_store(store.db)
    receipt = api().ArticleAudioAssets(reopened, profile_data_root=root).reconcile()
    if after_failure == "retry-stage":
        assert (
            receipt.removed == 1
            and not stage.exists()
            and not reopened.pending_deletions()
        )
    else:
        assert receipt.uncertain == receipt.retry == 1 and reopened.pending_deletions()
        assert (
            final if after_failure == "move-to-final" else stage
        ).read_bytes() == candidate.wav_bytes


@native_effects
def test_unqualified_prune_preserves_ready_metadata_and_verified_reuse(
    listening, monkeypatch
):
    store, session, root = listening
    owner = api().ArticleAudioAssets(store, profile_data_root=root)
    audio = publish(store, session, root, session.foreground)
    store.clear(expected_revision=store.load_session().revision)
    before = dict(store.audio_inventory(audio.generation)[0])
    assert before["status"] == "ready" and not before["referenced"]

    def refuse(_parent):
        raise OSError("regular_file_effects_unqualified")

    monkeypatch.setattr(api(), "require_regular_file_effects", refuse)
    monkeypatch.setattr(
        import_module("tldw_chatbook.Backup_Recovery.native_files"),
        "require_regular_file_effects",
        refuse,
    )
    with pytest.raises(api().ArticleAudioError, match="effects_unqualified"):
        owner.prune(2**62)
    assert dict(store.audio_inventory(audio.generation)[0]) == before
    assert owner.read_verified(audio.generation) == audio
    _, _, _, voice = setup_store(store.db)
    session = store.replace_session(
        prepare(store, session.authority, voice, item=audio.source_item_id),
        expected_revision=None,
    )
    assert (
        owner.reuse(session.foreground.entry_id, expected_revision=session.revision)
        == audio
    )


@native_effects
def test_post_seal_error_preserves_existing_publication_witness(listening, monkeypatch):
    owner, claim, candidate = owner_for(listening)
    seal = owner.store.seal_audio_custody
    saved = []

    def committed_then_failed(claim, witness, *, lease):
        seal(claim, witness, lease=lease)
        saved.append(witness)
        raise OSError("after_commit")

    monkeypatch.setattr(owner.store, "seal_audio_custody", committed_then_failed)
    with pytest.raises(api().ArticleAudioError, match="storage_unavailable"):
        owner.publish(
            candidate, source_item_id=claim.item_id, generation=claim.generation
        )
    row = owner.store.audio_inventory(claim.generation)[0]
    assert row["status"] == "pending" and saved == [row["custody_witness"]]
    assert len(list(owner.root.iterdir())) == 1
    assert owner.reconcile().removed == 1 and not list(owner.root.iterdir())
