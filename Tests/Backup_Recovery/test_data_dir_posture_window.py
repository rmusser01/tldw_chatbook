"""One positive look at the data directory's path posture covers a second on Windows.

``config.get_user_data_dir()`` reuses its resolution only while the posture of
every path component is unchanged, and looked on every call. On Windows a look
is one native open per component: 32 opens a call, 23 to 58 calls in one
Console turn (TASK-40608, counted 2026-11-27). ``profile_paths.data_dir_holds``
gives that look the storage layer's one-second window there (owner decisions
O1 and O3). These tests set the window themselves, so they run on every
platform; elsewhere the default is no window.
"""

from __future__ import annotations

import os
import time

import pytest

from tldw_chatbook.Backup_Recovery import profile_paths
from tldw_chatbook.Backup_Recovery import storage_admission as storage

STAMPS = ((1, 2, 3, 4, 5), (1, 6, 3, 4, 5))
PATHS = ("root", "leaf")


class _Posture:
    """Stands in for ``config._user_data_dir_stamps``."""

    def __init__(self, stamps=STAMPS, *, seconds: float = 0.0):
        self.stamps = stamps
        self.seconds = seconds
        self.looks = 0

    def __call__(self, paths):
        assert paths == PATHS
        self.looks += 1
        if self.seconds:
            time.sleep(self.seconds)
        return self.stamps


def _memo(stamps=STAMPS):
    return (("generation", "settings"), stamps, "directory")


@pytest.fixture(autouse=True)
def _windows(monkeypatch):
    """Both windows on, nothing held: each test then changes what it is about."""
    monkeypatch.setattr(profile_paths, "_data_dir_held", None)
    monkeypatch.setattr(profile_paths, "_DATA_DIR_HELD_SECONDS", 30.0)
    monkeypatch.setattr(storage, "_EVIDENCE_OBSERVATION_SECONDS", 30.0)


def test_the_window_is_on_for_windows_only(monkeypatch):
    monkeypatch.undo()
    assert profile_paths._DATA_DIR_HELD_SECONDS == (1.0 if os.name == "nt" else 0.0)


def test_inside_the_window_one_look_serves_every_call():
    memo, posture = _memo(), _Posture()

    assert all(profile_paths.data_dir_holds(memo, PATHS, posture) for _ in range(6))

    assert posture.looks == 1


def test_with_no_window_every_call_looks(monkeypatch):
    monkeypatch.setattr(profile_paths, "_DATA_DIR_HELD_SECONDS", 0.0)
    memo, posture = _memo(), _Posture()

    assert all(profile_paths.data_dir_holds(memo, PATHS, posture) for _ in range(4))

    assert posture.looks == 4
    assert profile_paths._data_dir_held is None


def test_a_changed_posture_is_refused_once_the_window_ends(monkeypatch):
    monkeypatch.setattr(profile_paths, "_DATA_DIR_HELD_SECONDS", 0.2)
    memo, posture = _memo(), _Posture()
    assert profile_paths.data_dir_holds(memo, PATHS, posture)
    posture.stamps = ((1, 2, 3, 0o755, 5), STAMPS[1])  # a repermissioned component
    assert profile_paths.data_dir_holds(memo, PATHS, posture)  # inside the window

    time.sleep(0.25)

    assert not profile_paths.data_dir_holds(memo, PATHS, posture)


def test_a_refusal_is_not_remembered_as_a_confirmation():
    memo, posture = _memo(), _Posture(((9, 9, 9, 9, 9), STAMPS[1]))

    assert not profile_paths.data_dir_holds(memo, PATHS, posture)
    assert not profile_paths.data_dir_holds(memo, PATHS, posture)

    assert posture.looks == 2


def test_a_refusal_ends_an_earlier_confirmation():
    """A change one caller saw at once is not hidden from the next by the window."""
    memo, posture = _memo(), _Posture()
    assert profile_paths.data_dir_holds(memo, PATHS, posture)
    posture.stamps = ((9, 9, 9, 9, 9), STAMPS[1])
    with storage.fresh_observation():
        assert not profile_paths.data_dir_holds(memo, PATHS, posture)

    assert not profile_paths.data_dir_holds(memo, PATHS, posture)


def test_an_equal_memo_from_another_resolution_is_looked_at_itself():
    """Identity, not equality: a new resolution has not been confirmed yet."""
    first, second, posture = _memo(), _memo(), _Posture()
    assert first == second and first is not second
    assert profile_paths.data_dir_holds(first, PATHS, posture)

    assert profile_paths.data_dir_holds(second, PATHS, posture)

    assert posture.looks == 2


def test_the_window_starts_when_the_confirming_look_began(monkeypatch):
    """A slow look must not extend the time its result is believed."""
    monkeypatch.setattr(profile_paths, "_DATA_DIR_HELD_SECONDS", 0.2)
    memo, posture = _memo(), _Posture(seconds=0.25)
    assert profile_paths.data_dir_holds(memo, PATHS, posture)
    posture.seconds = 0.0

    assert profile_paths.data_dir_holds(memo, PATHS, posture)

    assert posture.looks == 2


def test_a_fresh_observation_looks_whatever_the_window():
    memo, posture = _memo(), _Posture()
    assert profile_paths.data_dir_holds(memo, PATHS, posture)

    with storage.fresh_observation():
        assert profile_paths.data_dir_holds(memo, PATHS, posture)
        posture.stamps = ((9, 9, 9, 9, 9), STAMPS[1])
        assert not profile_paths.data_dir_holds(memo, PATHS, posture)

    assert posture.looks == 3


def test_no_window_while_the_storage_window_is_off(monkeypatch):
    """Tests that select the per-call derivation there select it here too."""
    monkeypatch.setattr(storage, "_EVIDENCE_OBSERVATION_SECONDS", 0.0)
    memo, posture = _memo(), _Posture()

    assert all(profile_paths.data_dir_holds(memo, PATHS, posture) for _ in range(3))

    assert posture.looks == 3
