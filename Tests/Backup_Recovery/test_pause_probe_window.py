"""A storage hold is asked "is maintenance waiting?" at most once a window.

On Windows a raw operation reuses a quiet answer for a second, because the
app's maintenance monitor asks the same native question once a second
(TASK-40608 D1, owner decisions O1 and O3). That answer was kept in one slot
for the last tuple of holds, so two operations that alternate, each with its
own holds, both asked every time: 49 to 80 probes in one Console turn, each
opening the control folder, its registry lock and every gate (counted
2026-11-27). It is now kept per hold, and a wait for the config lock asks
through the same window. These tests set the window themselves, so they run
on every platform; elsewhere the default is no window.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from tldw_chatbook.Backup_Recovery import bootstrap, config_participants
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage


class _Authority:
    def __init__(self, answer: bool = False):
        self.answer = answer
        self.asked = 0

    def pause_requested(self, names):
        self.asked += 1
        return self.answer


class _Hold:
    """What the probe reads of a storage hold; weakly referenceable, as that is."""

    def __init__(self, authority):
        self.authority = authority
        self.names = ("fixture",)


def _hold(answer: bool = False):
    authority = _Authority(answer)
    return _Hold(authority), authority


@pytest.fixture(autouse=True)
def _window(monkeypatch):
    monkeypatch.setattr(storage, "_EVIDENCE_OBSERVATION_SECONDS", 30.0)
    monkeypatch.setattr(raw, "_quiet_pause_probes", {})
    assert storage._pause is None


def test_with_no_window_every_call_asks(monkeypatch):
    monkeypatch.setattr(storage, "_EVIDENCE_OBSERVATION_SECONDS", 0.0)
    hold, authority = _hold()

    assert not any(raw._raw_pause_requested((hold,)) for _ in range(4))

    assert authority.asked == 4


def test_one_operation_asks_once_a_window():
    hold, authority = _hold()

    assert not any(raw._raw_pause_requested((hold,)) for _ in range(6))

    assert authority.asked == 1


def test_two_alternating_operations_each_ask_once_a_window():
    """One slot for the last tuple made both ask on every call."""
    first, first_authority = _hold()
    second, second_authority = _hold()

    for _ in range(5):
        assert not raw._raw_pause_requested((first,))
        assert not raw._raw_pause_requested((second,))

    assert (first_authority.asked, second_authority.asked) == (1, 1)


def test_holds_already_asked_alone_are_not_asked_again_together():
    first, first_authority = _hold()
    second, second_authority = _hold()
    assert not raw._raw_pause_requested((first,))
    assert not raw._raw_pause_requested((second,))

    assert not raw._raw_pause_requested((first, None, second))

    assert (first_authority.asked, second_authority.asked) == (1, 1)


def test_a_hold_not_asked_yet_is_asked_even_beside_a_quiet_one():
    first, first_authority = _hold()
    second, second_authority = _hold()
    assert not raw._raw_pause_requested((first,))

    assert not raw._raw_pause_requested((first, second))

    assert second_authority.asked == 1


def test_a_waiting_request_is_reported_and_asked_about_again():
    hold, authority = _hold()
    assert not raw._raw_pause_requested((hold,))
    quiet, quiet_authority = _hold()
    assert not raw._raw_pause_requested((quiet,))
    waiting, waiting_authority = _hold(True)

    assert raw._raw_pause_requested((waiting,))
    assert raw._raw_pause_requested((waiting,))

    assert waiting_authority.asked == 2
    # Another hold's quiet answer says nothing about this one.
    assert raw._raw_pause_requested((quiet, waiting))


def test_a_request_seen_beside_a_quiet_hold_ends_its_quiet_answer():
    quiet, quiet_authority = _hold()
    assert not raw._raw_pause_requested((quiet,))
    waiting, _ = _hold(True)
    assert raw._raw_pause_requested((quiet, waiting))
    asked = quiet_authority.asked

    assert not raw._raw_pause_requested((quiet,))

    assert quiet_authority.asked == asked + 1


def test_each_hold_is_asked_again_once_the_window_ends(monkeypatch):
    monkeypatch.setattr(storage, "_EVIDENCE_OBSERVATION_SECONDS", 0.2)
    hold, authority = _hold()
    assert not raw._raw_pause_requested((hold,))
    authority.answer = True
    assert not raw._raw_pause_requested((hold,))  # inside the window

    time.sleep(0.25)

    assert raw._raw_pause_requested((hold,))
    assert authority.asked == 2


def test_a_local_pause_is_asked_about_whatever_the_window(monkeypatch):
    hold, authority = _hold()
    assert not raw._raw_pause_requested((hold,))
    monkeypatch.setattr(storage, "_pause", object())
    authority.answer = True

    assert raw._raw_pause_requested((hold,))


def test_a_new_hold_at_a_dead_holds_address_is_asked(monkeypatch):
    """An answer belongs to the hold it was asked of, not to its address."""
    hold, _ = _hold()
    assert not raw._raw_pause_requested((hold,))
    newcomer, authority = _hold()
    probes = raw._quiet_pause_probes
    probes[id(newcomer)] = probes.pop(id(hold))  # as if the address had been reused

    assert not raw._raw_pause_requested((newcomer,))

    assert authority.asked == 1


def test_the_quiet_answers_do_not_keep_a_hold_alive():
    import gc
    import weakref

    hold, _ = _hold()
    assert not raw._raw_pause_requested((hold,))
    alive = weakref.ref(hold)

    del hold
    gc.collect()

    assert alive() is None


def test_a_hold_that_cannot_be_weakly_referenced_is_asked_every_time():
    authority = _Authority()
    hold = SimpleNamespace(authority=authority, names=("fixture",))

    assert not any(raw._raw_pause_requested((hold,)) for _ in range(3))

    assert authority.asked == 3


def test_the_quiet_answers_stay_bounded():
    holds = [_hold()[0] for _ in range(raw._QUIET_PAUSE_PROBES_MAX * 3)]

    for hold in holds:
        assert not raw._raw_pause_requested((hold,))

    assert len(raw._quiet_pause_probes) <= raw._QUIET_PAUSE_PROBES_MAX


def _lock_wait(monkeypatch, *holds):
    source = object()
    state = SimpleNamespace(source=source, participant=None, holds=holds)
    monkeypatch.setattr(raw, "check", lambda operation: state)
    return source


def test_a_config_lock_wait_asks_once_a_window(monkeypatch):
    hold, authority = _hold()
    source = _lock_wait(monkeypatch, hold, None)

    for _ in range(5):  # one before each attempt and one after taking the lock
        config_participants.check_lock_wait(source, object())

    assert authority.asked == 1


def test_a_config_lock_wait_still_refuses_a_waiting_request(monkeypatch):
    hold, authority = _hold(True)
    source = _lock_wait(monkeypatch, hold)

    with pytest.raises(bootstrap.RecoveryRequired, match="storage_locally_paused"):
        config_participants.check_lock_wait(source, object())
    with pytest.raises(bootstrap.RecoveryRequired, match="storage_locally_paused"):
        config_participants.check_lock_wait(source, object())

    assert authority.asked == 2


def test_with_no_window_a_config_lock_wait_asks_every_time(monkeypatch):
    monkeypatch.setattr(storage, "_EVIDENCE_OBSERVATION_SECONDS", 0.0)
    hold, authority = _hold()
    source = _lock_wait(monkeypatch, hold)

    for _ in range(3):
        config_participants.check_lock_wait(source, object())

    assert authority.asked == 3
