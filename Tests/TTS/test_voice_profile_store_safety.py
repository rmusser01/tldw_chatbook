"""Voice-profile stores must not destroy profiles they merely failed to read.

TASK-32893. Both managers (``ChatterboxVoiceManager`` and
``HiggsVoiceProfileManager``) keep their profiles in one JSON file, and the
shared store answered a READ ERROR with ``{}`` -- the same answer as "this
user has no profiles". The next write then replaced a file that was only
locked, permission-denied, or truncated, and every profile in it was gone.
Chatterbox also had no pre-write backup at all, and the Higgs backups that did
exist were named with naive LOCAL time and selected by ``sorted(glob(...))``,
i.e. by filename bytes, so "restore the most recent backup" could restore an
older one.

Since TASK-32863 both managers subclass ``VoiceManagerBase``, so all four
properties are pinned on the one shared implementation -- the parametrized
fixture proves each backend really inherits it.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tldw_chatbook.TTS.backends.chatterbox_voice_manager import ChatterboxVoiceManager
from tldw_chatbook.TTS.backends.higgs_voice_manager import HiggsVoiceProfileManager

_PROFILES = {"keeper": {"display_name": "Keeper", "reference_audio": "keeper.wav"}}

#: ``%Y%m%dT%H%M%S.%fZ`` -- UTC, ``Z``-suffixed (ADR-173), filename-safe. The
#: sub-second field is what stops two saves in one second sharing a path.
_UTC_BACKUP_NAME = re.compile(r"_backup_\d{8}T\d{6}\.\d{6}Z\.json$")


@pytest.fixture(params=["chatterbox", "higgs"])
def manager(request, tmp_path: Path):
    """One manager of each kind -- the defect was identical in both."""
    if request.param == "chatterbox":
        return ChatterboxVoiceManager(tmp_path / "chatterbox")
    return HiggsVoiceProfileManager(tmp_path / "higgs")


def _fresh(manager):
    """A second manager over the same directory (an empty read-through cache)."""
    return type(manager)(manager.voice_samples_dir)


def test_an_unreadable_profile_store_is_never_read_as_an_empty_one(manager) -> None:
    """A store that exists but cannot be parsed must raise, not answer ``{}``."""
    assert manager.save_profiles(dict(_PROFILES)) is True
    manager.profiles_file.write_text("{ this is not json", encoding="utf-8")

    with pytest.raises(Exception) as caught:
        _fresh(manager).load_profiles()
    assert not isinstance(caught.value, AssertionError)

    # An absent store is still the ordinary empty answer, not an error.
    manager.profiles_file.unlink()
    assert _fresh(manager).load_profiles() == {}


def test_a_failed_read_does_not_let_the_next_write_destroy_the_store(
    manager, tmp_path: Path
) -> None:
    """The point of the raise: the bytes on disk SURVIVE.

    Before the fix, ``load_profiles`` returned ``{}`` for an unreadable store
    and ``create_profile``/``delete_profile`` happily wrote that ``{}`` back
    over it.
    """
    assert manager.save_profiles(dict(_PROFILES)) is True
    original = manager.profiles_file.read_bytes()
    # Simulate "readable file, unreadable content" -- a truncated write, the
    # shape a crashed or out-of-space save leaves behind.
    truncated = original[: len(original) // 2]
    manager.profiles_file.write_bytes(truncated)

    reopened = _fresh(manager)
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF....WAVEfmt ")
    ok, _message = reopened.create_profile("newcomer", str(audio))

    assert ok is False, "a profile cannot be created on top of an unreadable store"
    assert manager.profiles_file.read_bytes() == truncated, (
        "the unreadable store must be left exactly as found, not overwritten"
    )


def test_saving_profiles_keeps_a_backup_of_what_was_there(manager) -> None:
    """Chatterbox kept no backup at all; now both backends inherit one."""
    assert manager.save_profiles(dict(_PROFILES)) is True
    assert manager.save_profiles({"replacement": {"display_name": "Replacement"}})

    backups = manager._list_backups()
    assert backups, "the overwriting save must have left a backup behind"
    assert json.loads(backups[-1].read_text(encoding="utf-8")) == _PROFILES


def test_backup_names_are_utc_with_a_z_suffix(manager) -> None:
    """ADR-173: UTC, ``Z``-suffixed. They were ``datetime.now()`` -- local."""
    assert manager.save_profiles(dict(_PROFILES)) is True
    assert manager.save_profiles({"replacement": {}}) is True

    backups = manager._list_backups()
    assert backups
    for path in backups:
        assert _UTC_BACKUP_NAME.search(path.name), path.name


def test_the_newest_backup_is_chosen_by_timestamp_not_by_glob_order(
    tmp_path: Path,
) -> None:
    """Glob order is filename-byte order, which is not chronological order.

    A store whose backup directory holds both a legacy naive-local name and a
    new UTC name is the plain case: ``sorted(glob(...))[-1]`` picks the legacy
    ``2026*1*231_...`` name over the newer ``2026*0*101T...Z`` one purely on
    the fifth character, and restoring "the latest backup" then replaced the
    user's newer profiles with older ones.
    """
    manager = HiggsVoiceProfileManager(tmp_path / "higgs")
    manager.backup_dir.mkdir(parents=True, exist_ok=True)

    stale = manager.backup_dir / "voice_profiles_backup_20261231_235900.json"
    stale.write_text(json.dumps({"stale": {}}), encoding="utf-8")
    os.utime(stale, (0, datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()))

    newest = manager.backup_dir / "voice_profiles_backup_20260101T000000Z.json"
    newest.write_text(json.dumps({"newest": {}}), encoding="utf-8")

    assert sorted(manager.backup_dir.glob("voice_profiles_backup_*.json"))[-1] == stale, (
        "precondition: glob order really does put the stale backup last"
    )

    ok, message = manager.restore_from_backup()

    assert ok is True, message
    assert manager.load_profiles() == {"newest": {}}


def test_a_burst_of_saves_keeps_one_backup_per_save(manager) -> None:
    """Three saves inside one second must leave three distinct backups.

    The stamp used to be second-precision, so every save in the same second
    resolved to the SAME backup path and ``voice_files.copy`` replaced it.
    A burst of edits -- an import, a multi-select delete -- therefore kept one
    recovery point instead of the configured ten.
    """
    assert manager.save_profiles({"first": {}}) is True
    assert manager.save_profiles({"second": {}}) is True
    assert manager.save_profiles({"third": {}}) is True

    backups = manager._list_backups()
    assert [json.loads(path.read_text(encoding="utf-8")) for path in backups] == [
        {"first": {}},
        {"second": {}},
    ], "each overwriting save must preserve the state it replaced"
