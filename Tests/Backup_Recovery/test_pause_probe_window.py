"""Only the pause question asked immediately before may be reused, and only for a second.

On Windows a raw operation may reuse a quiet "is maintenance waiting?" answer
for a second (TASK-40608 D1). It is kept in one slot for the last holds asked
about, so it is reused only when the very same holds are asked about again
with nothing else in between.

That narrowness is deliberate. On 2026-11-27 the answer was kept per hold
instead, which cut 49-80 native probes a Console turn to 12, and a config read
made within a second of a quiet answer then no longer saw a backup's request
before the maintenance monitor did:
``Tests/Backup_Recovery/test_scheduler_native_pause_intent.py`` failed, the
scheduler having dispatched due work under a pending native intent. The
interleaving test below is that case in small. These tests set the window
themselves, so they run on every platform; elsewhere the default is no window.
"""

from __future__ import annotations

import gc
import time
import weakref
from types import SimpleNamespace

import pytest

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
    monkeypatch.setattr(raw, "_quiet_pause_probe", None)
    assert storage._pause is None


def test_with_no_window_every_call_asks(monkeypatch):
    monkeypatch.setattr(storage, "_EVIDENCE_OBSERVATION_SECONDS", 0.0)
    hold, authority = _hold()

    assert not any(raw._raw_pause_requested((hold,)) for _ in range(4))

    assert authority.asked == 4


def test_the_same_question_asked_again_at_once_is_not_asked_natively():
    hold, authority = _hold()

    assert not any(raw._raw_pause_requested((hold, None)) for _ in range(6))

    assert authority.asked == 1


def test_a_quiet_answer_does_not_survive_another_operations_question():
    """The scheduler case: a request that arrives while other work runs is seen at once."""
    config, config_authority = _hold()
    other, _ = _hold()
    assert not raw._raw_pause_requested((config,))
    assert not raw._raw_pause_requested((other,))
    config_authority.answer = True  # a backup now waits on the config hold

    assert raw._raw_pause_requested((config,))


def test_holds_asked_alone_are_asked_again_together():
    first, first_authority = _hold()
    second, second_authority = _hold()
    assert not raw._raw_pause_requested((first,))
    assert not raw._raw_pause_requested((second,))

    assert not raw._raw_pause_requested((first, second))

    assert (first_authority.asked, second_authority.asked) == (2, 2)


def test_a_waiting_request_is_reported_and_never_remembered():
    hold, authority = _hold(True)

    assert raw._raw_pause_requested((hold,))
    assert raw._raw_pause_requested((hold,))

    assert authority.asked == 2
    assert raw._quiet_pause_probe is None


def test_the_question_is_asked_again_once_the_window_ends(monkeypatch):
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
    """An answer belongs to the holds it was asked of, not to their addresses."""
    hold, _ = _hold()
    assert not raw._raw_pause_requested((hold,))
    references, asked_at = raw._quiet_pause_probe
    del hold
    gc.collect()
    assert references[0]() is None
    newcomer, authority = _hold()
    # As if the newcomer had been allocated where the dead hold was.
    monkeypatch.setattr(raw, "_quiet_pause_probe", (references, asked_at))

    assert not raw._raw_pause_requested((newcomer,))

    assert authority.asked == 1


def test_the_quiet_answer_does_not_keep_a_hold_alive():
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
