"""Tests for the feed-directory writer (spec #2 phase 3, Task 4).

`export_feed_directory` is the deliverable of the whole spec #2 podcast
feed: it copies a watchlist's audio episodes out of the private
`briefing_audio_dir()` into a directory the user chose, alongside a
`feed.xml` RSS document (Task 3's `build_feed_xml`). Every test here
patches `get_user_data_dir` to `tmp_path` -- this repo has had three
separate incidents of test scaffolding touching live user files, and
`audio_file_path_is_safe` (imported from `Subscriptions.briefing_audio`,
moved there from `artifacts_pane` in this task's review round 1) resolves
against `briefing_audio_dir()`, which reads that setting.

The load-bearing test in this file is
`test_an_unsafe_file_path_is_skipped_without_any_filesystem_access`: it
proves the safety check runs BEFORE any read of the underlying file, not
merely that an unsafe file never ends up copied -- a real, readable file
sits at the "unsafe" path, and the test spies on both `Path.exists` and
`shutil.copy2` to prove neither is ever invoked for it.

Same harness as `test_briefing_feed_query.py`: a real `SubscriptionsDB`,
`WatchlistBundleService` for watchlist creation, `insert_briefing` /
`insert_briefing_script` / `create_briefing_audio` / `update_briefing_audio`
for the rest of the chain.
"""

from __future__ import annotations

import os
import stat
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions import briefing_audio, briefing_export
from tldw_chatbook.Subscriptions.briefing_export import (
    FeedExportResult,
    export_feed_directory,
)
from tldw_chatbook.Subscriptions.watchlist_bundle_service import WatchlistBundleService

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 8, 1, 12, 0, 0, tzinfo=timezone.utc)


# --- shared harness ---------------------------------------------------------


def _patch_user_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Redirect `briefing_audio_dir()` into `tmp_path` -- never real storage."""
    monkeypatch.setattr(briefing_audio, "get_user_data_dir", lambda: tmp_path)


def _db(tmp_path: Path) -> SubscriptionsDB:
    return SubscriptionsDB(tmp_path / "subs.db", "test")


def _make_watchlist(db: SubscriptionsDB, name: str = "w") -> int:
    return WatchlistBundleService(db).create(name=name)["id"]


def _make_briefing(db: SubscriptionsDB, watchlist_id: int, *, created_at: str) -> int:
    briefing_id = db.insert_briefing(watchlist_id)
    with db.transaction() as conn:
        conn.execute(
            "UPDATE briefings SET created_at = ? WHERE id = ?",
            (created_at, briefing_id),
        )
    return briefing_id


def _make_script(db: SubscriptionsDB, briefing_id: int, *, preset_name: str = "p") -> int:
    return db.insert_briefing_script(
        briefing_id, preset_id=None, preset_name=preset_name, roster_snapshot_json="[]"
    )


def _seed_episode(
    db: SubscriptionsDB,
    watchlist_id: int,
    *,
    audio_dir: Path,
    created_at: str = "2026-01-01 00:00:00",
    preset_name: str = "Two Host Debate",
    write_real_file: bool = True,
    file_path: str | None = None,
    covers_from_ts: str | None = None,
) -> tuple[int, Path]:
    """Build one full watchlist -> briefing -> script -> complete-audio chain.

    Writes a real, readable `.wav`-shaped file under `audio_dir` unless
    `write_real_file` is False (the "vanished file" case) or `file_path` is
    given explicitly (the "path points outside `audio_dir`" case).

    Returns:
        `(audio_id, source_path)`.
    """
    briefing_id = _make_briefing(db, watchlist_id, created_at=created_at)
    if covers_from_ts is not None:
        db.update_briefing(briefing_id, covers_from_ts=covers_from_ts)
    script_id = _make_script(db, briefing_id, preset_name=preset_name)
    audio_id = db.create_briefing_audio(script_id, voice_snapshot_json="[]", status="complete")
    source_path = (
        Path(file_path)
        if file_path is not None
        else audio_dir / f"script-{script_id}-audio-{audio_id}.wav"
    )
    if write_real_file:
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(b"RIFF....WAVEfmt fake-audio-bytes")
    db.update_briefing_audio(
        audio_id,
        file_path=str(source_path),
        duration_seconds=90.0,
        turn_count=4,
    )
    return audio_id, source_path


# --- happy path --------------------------------------------------------------


def test_happy_path_writes_feed_xml_plus_one_file_per_episode_and_returns_the_count(
    tmp_path, monkeypatch
):
    _patch_user_data_dir(monkeypatch, tmp_path)
    db = _db(tmp_path)
    watchlist_id = _make_watchlist(db, "Morning AI Brief")
    audio_dir = briefing_audio.briefing_audio_dir()

    _seed_episode(db, watchlist_id, audio_dir=audio_dir, created_at="2026-01-01 00:00:00")
    _seed_episode(db, watchlist_id, audio_dir=audio_dir, created_at="2026-01-02 00:00:00")

    destination = tmp_path / "export"
    destination.mkdir()

    result = export_feed_directory(
        db,
        watchlist_id,
        destination=destination,
        watchlist_name="Morning AI Brief",
        now=_NOW,
    )

    assert isinstance(result, FeedExportResult)
    assert result.episode_count == 2
    assert result.skipped == []
    assert result.directory == destination.resolve()
    assert (destination / "feed.xml").exists()

    written_audio_files = [p for p in destination.iterdir() if p.name != "feed.xml"]
    assert len(written_audio_files) == 2

    tree = ET.fromstring((destination / "feed.xml").read_bytes())
    items = tree.findall("./channel/item")
    assert len(items) == 2


def test_length_bytes_matches_the_actual_copied_destination_files_size(
    tmp_path, monkeypatch
):
    """The enclosure's `length` must reflect the file that actually landed in
    the destination directory. Task 4 review round 1: the original version
    of this test stated the SOURCE file before the copy ran and compared
    against that -- since a correct copy leaves source and destination
    byte-identical, that oracle cannot tell a (correct) dest-stat
    implementation from a (wrong) source-stat one, and the test's own name
    over-claimed what it checked. This stats the destination file
    `export_feed_directory` actually produced, found via a real directory
    listing after the call.
    """
    _patch_user_data_dir(monkeypatch, tmp_path)
    db = _db(tmp_path)
    watchlist_id = _make_watchlist(db)
    audio_dir = briefing_audio.briefing_audio_dir()
    _seed_episode(db, watchlist_id, audio_dir=audio_dir)

    destination = tmp_path / "export"
    destination.mkdir()

    export_feed_directory(
        db, watchlist_id, destination=destination, watchlist_name="W", now=_NOW
    )

    episode_files = [p for p in destination.iterdir() if p.name != "feed.xml"]
    assert len(episode_files) == 1
    actual_dest_size = episode_files[0].stat().st_size

    tree = ET.fromstring((destination / "feed.xml").read_bytes())
    enclosure = tree.find("./channel/item/enclosure")
    assert int(enclosure.get("length")) == actual_dest_size


def test_filenames_combine_stem_and_audio_id_so_identical_titles_cannot_collide(
    tmp_path, monkeypatch
):
    _patch_user_data_dir(monkeypatch, tmp_path)
    db = _db(tmp_path)
    watchlist_id = _make_watchlist(db)
    audio_dir = briefing_audio.briefing_audio_dir()

    _seed_episode(
        db,
        watchlist_id,
        audio_dir=audio_dir,
        created_at="2026-01-01 00:00:00",
        preset_name="Same Title",
    )
    _seed_episode(
        db,
        watchlist_id,
        audio_dir=audio_dir,
        created_at="2026-01-02 00:00:00",
        preset_name="Same Title",
    )

    destination = tmp_path / "export"
    destination.mkdir()

    result = export_feed_directory(
        db, watchlist_id, destination=destination, watchlist_name="W", now=_NOW
    )

    assert result.episode_count == 2
    written_names = sorted(p.name for p in destination.iterdir() if p.name != "feed.xml")
    assert len(written_names) == 2
    assert written_names[0] != written_names[1]


def test_episode_titles_lead_with_the_date_so_identical_presets_are_distinguishable(
    tmp_path, monkeypatch
):
    """Task 4 review round 1: `preset_name` alone made every episode
    rendered from the same preset identical in a podcast client's episode
    list, distinguishable only by whatever date field that client happens
    to expose. Leading the title with the publish date fixes this
    directly."""
    _patch_user_data_dir(monkeypatch, tmp_path)
    db = _db(tmp_path)
    watchlist_id = _make_watchlist(db)
    audio_dir = briefing_audio.briefing_audio_dir()

    _seed_episode(
        db,
        watchlist_id,
        audio_dir=audio_dir,
        created_at="2026-01-01 00:00:00",
        preset_name="Same Preset",
    )
    _seed_episode(
        db,
        watchlist_id,
        audio_dir=audio_dir,
        created_at="2026-01-02 00:00:00",
        preset_name="Same Preset",
    )

    destination = tmp_path / "export"
    destination.mkdir()

    export_feed_directory(
        db, watchlist_id, destination=destination, watchlist_name="W", now=_NOW
    )

    tree = ET.fromstring((destination / "feed.xml").read_bytes())
    titles = [item.findtext("title") for item in tree.findall("./channel/item")]

    assert len(titles) == 2
    assert titles[0] != titles[1], "identical-preset episodes must not share a title"
    assert all("Same Preset" in title for title in titles)
    assert "Jan 01, 2026" in titles[0] or "Jan 01, 2026" in titles[1]
    assert "Jan 02, 2026" in titles[0] or "Jan 02, 2026" in titles[1]


def test_episode_description_includes_covers_from_ts_when_present(tmp_path, monkeypatch):
    """Task 4 review round 1: `covers_from_ts` is present on every
    `list_watchlist_audio_episodes` row but was previously unused -- fed
    into the description so an episode says what period it actually
    covers, which a title/date alone cannot."""
    _patch_user_data_dir(monkeypatch, tmp_path)
    db = _db(tmp_path)
    watchlist_id = _make_watchlist(db)
    audio_dir = briefing_audio.briefing_audio_dir()

    _seed_episode(
        db,
        watchlist_id,
        audio_dir=audio_dir,
        covers_from_ts="2025-12-25T00:00:00+00:00",
    )

    destination = tmp_path / "export"
    destination.mkdir()

    export_feed_directory(
        db, watchlist_id, destination=destination, watchlist_name="W", now=_NOW
    )

    tree = ET.fromstring((destination / "feed.xml").read_bytes())
    description = tree.findtext("./channel/item/description")
    assert "2025-12-25T00:00:00+00:00" in description


# --- the load-bearing security test ------------------------------------------


def test_an_unsafe_file_path_is_skipped_without_any_filesystem_access(tmp_path, monkeypatch):
    """A real, readable file sits at the "unsafe" path -- a wrong-order
    implementation would happily read/copy it. `audio_file_path_is_safe` is
    forced to return `False` regardless, and this asserts neither
    `Path.exists` nor `shutil.copy2` is ever invoked against it: the safety
    check must run before ANY filesystem probe, not just before the copy.
    """
    _patch_user_data_dir(monkeypatch, tmp_path)
    db = _db(tmp_path)
    watchlist_id = _make_watchlist(db)
    audio_dir = briefing_audio.briefing_audio_dir()

    audio_id, source_path = _seed_episode(db, watchlist_id, audio_dir=audio_dir)

    monkeypatch.setattr(briefing_export, "audio_file_path_is_safe", lambda _path: False)

    exists_calls: list[Path] = []
    real_exists = Path.exists

    def _spy_exists(self, *args, **kwargs):
        exists_calls.append(self)
        return real_exists(self, *args, **kwargs)

    monkeypatch.setattr(Path, "exists", _spy_exists)

    copy_mock = Mock(side_effect=AssertionError("shutil.copy2 must not run for an unsafe path"))
    monkeypatch.setattr(briefing_export.shutil, "copy2", copy_mock)

    destination = tmp_path / "export"
    destination.mkdir()

    result = export_feed_directory(
        db, watchlist_id, destination=destination, watchlist_name="W", now=_NOW
    )

    assert result.episode_count == 0
    assert len(result.skipped) == 1
    assert str(audio_id) in result.skipped[0]
    copy_mock.assert_not_called()
    assert source_path not in exists_calls, (
        "the source file's own .exists() was probed -- the safety check "
        "must run before any filesystem access, not just before the copy"
    )


def test_a_vanished_source_file_is_skipped_with_a_reason_and_others_still_export(
    tmp_path, monkeypatch
):
    _patch_user_data_dir(monkeypatch, tmp_path)
    db = _db(tmp_path)
    watchlist_id = _make_watchlist(db)
    audio_dir = briefing_audio.briefing_audio_dir()

    vanished_audio_id, _vanished_path = _seed_episode(
        db,
        watchlist_id,
        audio_dir=audio_dir,
        created_at="2026-01-01 00:00:00",
        write_real_file=False,
    )
    _present_audio_id, _present_path = _seed_episode(
        db, watchlist_id, audio_dir=audio_dir, created_at="2026-01-02 00:00:00"
    )

    destination = tmp_path / "export"
    destination.mkdir()

    result = export_feed_directory(
        db, watchlist_id, destination=destination, watchlist_name="W", now=_NOW
    )

    assert result.episode_count == 1
    assert len(result.skipped) == 1
    assert str(vanished_audio_id) in result.skipped[0]


def test_a_malformed_briefing_created_at_is_skipped_with_a_clear_reason_others_still_export(
    tmp_path, monkeypatch
):
    """Task 4 review round 1: no test previously exercised this skip path
    (`_published_from_briefing_created_at`'s `ValueError`), and its reason
    string used to be the bare exception type name (`"audio 3: ValueError"`)
    -- opaque, and Task 5 surfaces `skipped` reasons verbatim in its "N of M
    exported" message. This both covers the path and pins that the reason
    says what actually went wrong, in plain words.
    """
    _patch_user_data_dir(monkeypatch, tmp_path)
    db = _db(tmp_path)
    watchlist_id = _make_watchlist(db)
    audio_dir = briefing_audio.briefing_audio_dir()

    bad_audio_id, _bad_path = _seed_episode(
        db, watchlist_id, audio_dir=audio_dir, created_at="not-a-timestamp"
    )
    _good_audio_id, _good_path = _seed_episode(
        db, watchlist_id, audio_dir=audio_dir, created_at="2026-01-02 00:00:00"
    )

    destination = tmp_path / "export"
    destination.mkdir()

    result = export_feed_directory(
        db, watchlist_id, destination=destination, watchlist_name="W", now=_NOW
    )

    assert result.episode_count == 1
    assert len(result.skipped) == 1
    reason = result.skipped[0]
    assert str(bad_audio_id) in reason
    assert "ValueError" not in reason, "the reason must say what went wrong, not a bare exception type"
    assert "timestamp" in reason.lower()


def test_an_export_with_every_episode_skipped_still_writes_a_valid_empty_feed(
    tmp_path, monkeypatch
):
    """Task 4 review round 1: an all-skipped export must still produce a
    valid `feed.xml` with zero items -- the honest "0 of N exported"
    outcome, not a missing or malformed feed."""
    _patch_user_data_dir(monkeypatch, tmp_path)
    db = _db(tmp_path)
    watchlist_id = _make_watchlist(db)
    audio_dir = briefing_audio.briefing_audio_dir()
    _seed_episode(db, watchlist_id, audio_dir=audio_dir, write_real_file=False)

    destination = tmp_path / "export"
    destination.mkdir()

    result = export_feed_directory(
        db, watchlist_id, destination=destination, watchlist_name="W", now=_NOW
    )

    assert result.episode_count == 0
    assert len(result.skipped) == 1
    assert (destination / "feed.xml").exists()
    tree = ET.fromstring((destination / "feed.xml").read_bytes())
    assert tree.tag == "rss"
    assert tree.findall("./channel/item") == []


# --- feed.xml atomicity -------------------------------------------------------


def test_no_partial_file_remains_after_a_successful_export(tmp_path, monkeypatch):
    """Task 4 review round 1: absence of a `.partial` file alone does not
    pin atomicity -- a plain `write_bytes` also leaves none behind (the
    reviewer confirmed this by mutation). The `os.replace` spy below is
    what actually ties this test to the atomic-rename mechanism: it asserts
    the real publish step ran with the expected (partial, final) pair, not
    just that no stray file happens to remain afterward.
    """
    _patch_user_data_dir(monkeypatch, tmp_path)
    db = _db(tmp_path)
    watchlist_id = _make_watchlist(db)
    audio_dir = briefing_audio.briefing_audio_dir()
    _seed_episode(db, watchlist_id, audio_dir=audio_dir)

    destination = tmp_path / "export"
    destination.mkdir()

    real_replace = briefing_export.os.replace
    replace_spy = Mock(wraps=real_replace)
    monkeypatch.setattr(briefing_export.os, "replace", replace_spy)

    export_feed_directory(
        db, watchlist_id, destination=destination, watchlist_name="W", now=_NOW
    )

    resolved_destination = destination.resolve()
    replace_spy.assert_called_once_with(
        resolved_destination / "feed.xml.partial", resolved_destination / "feed.xml"
    )
    assert (destination / "feed.xml").exists()
    assert not (destination / "feed.xml.partial").exists()


def test_a_failure_mid_write_leaves_the_previous_feed_xml_intact(tmp_path, monkeypatch):
    _patch_user_data_dir(monkeypatch, tmp_path)
    db = _db(tmp_path)
    watchlist_id = _make_watchlist(db)
    audio_dir = briefing_audio.briefing_audio_dir()
    _seed_episode(db, watchlist_id, audio_dir=audio_dir)

    destination = tmp_path / "export"
    destination.mkdir()

    export_feed_directory(
        db, watchlist_id, destination=destination, watchlist_name="W", now=_NOW
    )
    previous_bytes = (destination / "feed.xml").read_bytes()

    # Simulate a crash between the write and the atomic rename.
    monkeypatch.setattr(
        briefing_export.os, "replace", Mock(side_effect=OSError("simulated crash"))
    )

    with pytest.raises(OSError):
        export_feed_directory(
            db, watchlist_id, destination=destination, watchlist_name="W", now=_NOW
        )

    assert (destination / "feed.xml").read_bytes() == previous_bytes
    assert not (destination / "feed.xml.partial").exists()


def test_re_exporting_to_the_same_directory_overwrites_cleanly(tmp_path, monkeypatch):
    """No `"xb"`-style refusal -- the phase-2b private-write helpers must
    not be used here (Decision 2)."""
    _patch_user_data_dir(monkeypatch, tmp_path)
    db = _db(tmp_path)
    watchlist_id = _make_watchlist(db)
    audio_dir = briefing_audio.briefing_audio_dir()
    _seed_episode(db, watchlist_id, audio_dir=audio_dir, created_at="2026-01-01 00:00:00")

    destination = tmp_path / "export"
    destination.mkdir()

    first = export_feed_directory(
        db, watchlist_id, destination=destination, watchlist_name="W", now=_NOW
    )
    assert first.episode_count == 1

    _seed_episode(db, watchlist_id, audio_dir=audio_dir, created_at="2026-01-02 00:00:00")

    second = export_feed_directory(
        db, watchlist_id, destination=destination, watchlist_name="W", now=_NOW
    )

    assert second.episode_count == 2
    tree = ET.fromstring((destination / "feed.xml").read_bytes())
    assert len(tree.findall("./channel/item")) == 2


# --- Decision 2: never route the destination through private_paths ----------


@pytest.mark.skipif(os.name == "nt", reason="POSIX file mode semantics")
def test_destination_directory_mode_is_never_forced_to_0o700(tmp_path, monkeypatch):
    _patch_user_data_dir(monkeypatch, tmp_path)
    db = _db(tmp_path)
    watchlist_id = _make_watchlist(db)
    audio_dir = briefing_audio.briefing_audio_dir()
    _seed_episode(db, watchlist_id, audio_dir=audio_dir)

    destination = tmp_path / "export"
    destination.mkdir()
    os.chmod(destination, 0o755)

    export_feed_directory(
        db, watchlist_id, destination=destination, watchlist_name="W", now=_NOW
    )

    mode = stat.S_IMODE(os.stat(destination).st_mode)
    assert mode == 0o755, "the user's destination directory must never be chmodded (Decision 2)"


@pytest.mark.skipif(os.name == "nt", reason="POSIX file mode semantics")
def test_exported_feed_xml_is_not_left_at_private_storage_permissions(tmp_path, monkeypatch):
    """Task 4 review round 1 (IMPORTANT finding): `feed.xml`'s atomic-write
    partial used to be opened at `0o600`, and `os.replace` preserves the
    replaced-in file's mode, so every exported `feed.xml` landed `0o600` --
    unreadable to anything but the exporting account, defeating the whole
    point of a folder meant to be synced, zipped, or served. Fixed by
    opening the partial at `_EXPORTED_FILE_MODE` (`0o644`) with an explicit
    `fchmod`, deterministic regardless of the process umask.
    """
    _patch_user_data_dir(monkeypatch, tmp_path)
    db = _db(tmp_path)
    watchlist_id = _make_watchlist(db)
    audio_dir = briefing_audio.briefing_audio_dir()
    _seed_episode(db, watchlist_id, audio_dir=audio_dir)

    destination = tmp_path / "export"
    destination.mkdir()

    export_feed_directory(
        db, watchlist_id, destination=destination, watchlist_name="W", now=_NOW
    )

    feed_mode = stat.S_IMODE(os.stat(destination / "feed.xml").st_mode)
    assert feed_mode & 0o044 == 0o044, (
        f"feed.xml must be group/other readable, got {oct(feed_mode)}"
    )


@pytest.mark.skipif(os.name == "nt", reason="POSIX file mode semantics")
def test_a_copied_episode_file_relaxes_the_privately_stored_sources_mode(
    tmp_path, monkeypatch
):
    """Task 4 review round 1 (IMPORTANT finding): production audio is
    written by `Utils.private_paths.atomic_private_write_bytes` at `0o600`
    (application-owned, private-storage semantics), and `shutil.copy2`
    preserves the SOURCE file's mode along with its data -- so every
    exported episode silently inherited that private mode into the user's
    own folder too. This reproduces the real scenario by explicitly
    chmod-ing the seeded source to `0o600` (what production actually
    writes) and asserts the COPY in the destination directory is relaxed to
    a normal, group/other-readable mode instead of inheriting it.
    """
    _patch_user_data_dir(monkeypatch, tmp_path)
    db = _db(tmp_path)
    watchlist_id = _make_watchlist(db)
    audio_dir = briefing_audio.briefing_audio_dir()
    _audio_id, source_path = _seed_episode(db, watchlist_id, audio_dir=audio_dir)
    os.chmod(source_path, 0o600)

    destination = tmp_path / "export"
    destination.mkdir()

    export_feed_directory(
        db, watchlist_id, destination=destination, watchlist_name="W", now=_NOW
    )

    episode_files = [p for p in destination.iterdir() if p.name != "feed.xml"]
    assert len(episode_files) == 1
    episode_mode = stat.S_IMODE(os.stat(episode_files[0]).st_mode)
    assert episode_mode & 0o044 == 0o044, (
        f"a copied episode file must be group/other readable, not inherit the "
        f"private source's mode, got {oct(episode_mode)}"
    )
    # The source itself is untouched -- only the COPY is relaxed.
    source_mode = stat.S_IMODE(os.stat(source_path).st_mode)
    assert source_mode == 0o600


def test_a_destination_failing_validate_path_simple_raises_before_anything_is_written(
    tmp_path, monkeypatch
):
    _patch_user_data_dir(monkeypatch, tmp_path)
    db = _db(tmp_path)
    watchlist_id = _make_watchlist(db)
    audio_dir = briefing_audio.briefing_audio_dir()
    _seed_episode(db, watchlist_id, audio_dir=audio_dir)

    real_query = db.list_watchlist_audio_episodes
    query_spy = Mock(wraps=real_query)
    monkeypatch.setattr(db, "list_watchlist_audio_episodes", query_spy)

    hostile_destination = Path(str(tmp_path / "export") + ";rm -rf ~")

    with pytest.raises(ValueError):
        export_feed_directory(
            db,
            watchlist_id,
            destination=hostile_destination,
            watchlist_name="W",
            now=_NOW,
        )

    query_spy.assert_not_called()
    assert not hostile_destination.exists()
