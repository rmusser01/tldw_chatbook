"""InterruptRoundHost unit tests -- no ConsoleChatController anywhere.

Sub-project C1 (spec: 2026-08-20-console-interrupt-host-design.md): the
host must be testable against a minimal seams double, which is exactly
the surface it is allowed to touch on the controller.
"""

from __future__ import annotations

import threading
import time

import pytest

from tldw_chatbook.Chat.console_interrupt_rounds import (
    KIND_SETTER_ATTRS,
    InterruptRoundHost,
)


class FakeStore:
    def __init__(self) -> None:
        self.active_session_id = "sess-A"


class FakeApp:
    def call_from_thread(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)


class FakeSeams:
    """The exact controller surface the host may touch, and nothing more."""

    def __init__(self) -> None:
        self.app = FakeApp()
        self.store = FakeStore()
        self.mounted: dict[str, list] = {k: [] for k in KIND_SETTER_ATTRS}
        for kind, attr in KIND_SETTER_ATTRS.items():
            setattr(self, attr, self.mounted[kind].append)


@pytest.fixture
def host():
    return InterruptRoundHost(FakeSeams())


def _payload(round_id, session_id="sess-A", **extra):
    return {"round_id": round_id, "session_id": session_id, **extra}


def test_park_returns_head_only_for_the_oldest_round(host):
    assert host.park_round_payload("approval", "r1", _payload("r1")) is True
    assert host.park_round_payload("approval", "r2", _payload("r2")) is False


def test_kinds_do_not_share_heads(host):
    host.park_round_payload("approval", "r1", _payload("r1"))
    assert host.park_round_payload("skill_install", "r2", _payload("r2")) is True


def test_unpark_promotes_the_next_round(host):
    host.park_round_payload("approval", "r1", _payload("r1"))
    host.park_round_payload("approval", "r2", _payload("r2"))
    host.unpark_round_payload("approval", "r1")
    head = host.head_round_payload("approval", "sess-A")
    assert head is not None and head["round_id"] == "r2"


def test_head_returns_remaining_time_snapshot_without_mutating_the_stored_payload(host):
    stored = _payload(
        "r1", timeout_seconds=30.0, deadline_monotonic=time.monotonic() + 5.0
    )
    host.park_round_payload("approval", "r1", stored)
    head = host.head_round_payload("approval", "sess-A")
    assert 0.0 < head["timeout_seconds"] <= 5.0
    assert stored["timeout_seconds"] == 30.0


def test_remount_head_none_session_resolves_the_active_session(host):
    seams = host._seams
    host.park_round_payload("approval", "r1", _payload("r1", session_id="sess-A"))
    seams.store.active_session_id = "sess-A"
    host.remount_head("approval", None)
    assert seams.mounted["approval"][-1]["round_id"] == "r1"
    seams.store.active_session_id = "sess-B"
    host.remount_head("approval", None)
    assert seams.mounted["approval"][-1] is None


def test_remount_head_mismatched_session_is_a_no_op(host):
    seams = host._seams
    host.park_round_payload("approval", "r1", _payload("r1", session_id="sess-B"))
    host.remount_head("approval", "sess-B")  # active is sess-A
    assert seams.mounted["approval"] == []


def test_missing_setter_attr_is_a_safe_no_op(host):
    seams = host._seams
    delattr(seams, KIND_SETTER_ATTRS["question"])
    host.park_round_payload("question", "q1", _payload("q1"))
    host.remount_head("question", "sess-A")  # must not raise


class FakeSeamsFull(FakeSeams):
    """Adds the probe/badge surface run_round touches."""

    def __init__(self) -> None:
        super().__init__()
        self.cancelled = False
        self.badges: list[tuple[str, str, str]] = []
        self.park_pending_approval = None

    def _is_session_cancelled(self, session_id, *, cancel_event=None, visit_event=None):
        return self.cancelled

    def add_pending_round(self, session_id, round_id):
        self.badges.append(("add", session_id, round_id))

    def discard_pending_round(self, session_id, round_id):
        self.badges.append(("discard", session_id, round_id))


def test_run_round_decided_when_the_event_is_set():
    host = InterruptRoundHost(FakeSeamsFull())
    state = {"event": threading.Event(), "session_id": "sess-A"}
    state["event"].set()  # pre-resolved: loop exits immediately
    outcome = host.run_round(
        "approval", "r1", _payload("r1"), state,
        session_id="sess-A", owning_session_id="sess-A",
        deadline=None, is_parked=False,
    )
    assert outcome == "decided"
    assert host.registries["approval"] == {}
    assert host.payloads["approval"] == {}


def test_run_round_times_out_and_calls_on_timeout():
    host = InterruptRoundHost(FakeSeamsFull())
    fired = []
    state = {"event": threading.Event(), "session_id": "sess-A"}
    outcome = host.run_round(
        "approval", "r1", _payload("r1"), state,
        session_id="sess-A", owning_session_id="sess-A",
        deadline=time.monotonic() - 1.0, is_parked=False,
        on_timeout=lambda: fired.append("t"),
    )
    assert outcome == "timeout" and fired == ["t"]


def test_run_round_cancelled_calls_on_cancelled():
    seams = FakeSeamsFull()
    seams.cancelled = True
    host = InterruptRoundHost(seams)
    fired = []
    state = {"event": threading.Event(), "session_id": "sess-A"}
    outcome = host.run_round(
        "approval", "r1", _payload("r1"), state,
        session_id="sess-A", owning_session_id="sess-A",
        deadline=None, is_parked=False,
        on_cancelled=lambda: fired.append("c"),
    )
    assert outcome == "cancelled" and fired == ["c"]


def test_run_round_revoked_wins_over_decided():
    host = InterruptRoundHost(FakeSeamsFull())
    state = {"event": threading.Event(), "session_id": "sess-A", "revoked": True}
    state["event"].set()
    outcome = host.run_round(
        "approval", "r1", _payload("r1"), state,
        session_id="sess-A", owning_session_id="sess-A",
        deadline=None, is_parked=False,
    )
    assert outcome == "revoked"


def test_run_round_teardown_promotes_the_queued_sibling():
    seams = FakeSeamsFull()
    host = InterruptRoundHost(seams)
    host.park_round_payload("approval", "r0", _payload("r0"))  # will be head
    state = {"event": threading.Event(), "session_id": "sess-A"}
    state["event"].set()
    host.run_round(
        "approval", "r1", _payload("r1"), state,
        session_id="sess-A", owning_session_id="sess-A",
        deadline=None, is_parked=False,
    )
    assert seams.mounted["approval"][-1]["round_id"] == "r0"


def test_run_round_badge_add_and_discard_bracket_the_wait():
    seams = FakeSeamsFull()
    host = InterruptRoundHost(seams)
    state = {"event": threading.Event(), "session_id": "sess-A"}
    state["event"].set()
    host.run_round(
        "approval", "r1", _payload("r1"), state,
        session_id="sess-A", owning_session_id="sess-A",
        deadline=None, is_parked=False,
    )
    assert seams.badges == [("add", "sess-A", "r1"), ("discard", "sess-A", "r1")]


def test_run_round_legacy_none_session_skips_badge_and_park():
    seams = FakeSeamsFull()
    host = InterruptRoundHost(seams)
    state = {"event": threading.Event(), "session_id": ""}
    state["event"].set()
    host.run_round(
        "approval", "r1", _payload("r1", session_id=""), state,
        session_id=None, owning_session_id="",
        deadline=None, is_parked=False,
    )
    assert seams.badges == []
    assert host.payloads["approval"] == {}


# --- task-31384: teardown retention, revocation sweep, post-remount hook ------


def test_on_teardown_true_keeps_the_payload_parked():
    seams = FakeSeamsFull()
    host = InterruptRoundHost(seams)
    state = {"event": threading.Event(), "run_id": "run-1"}
    payload = _payload("r1")
    state["event"].set()
    host.run_round(
        "approval", "r1", payload, state,
        session_id="sess-A", owning_session_id="sess-A", deadline=None, is_parked=False,
        on_teardown=lambda: True,
    )
    assert host.head_round_payload("approval", "sess-A") is payload, "retained past teardown"
    host.unpark_round_payload("approval", "r1")
    assert host.head_round_payload("approval", "sess-A") is None


def test_revoke_for_run_sweeps_only_the_named_kinds_and_run():
    host = InterruptRoundHost(FakeSeams())
    stamped = []
    for kind, rid, run in [("approval", "a1", "run-x"), ("question", "q1", "run-x"), ("skill_script", "s1", "run-y"), ("skill_install", "i1", "run-x")]:
        host.registries[kind][rid] = {"event": threading.Event(), "run_id": run, "session_id": "sess-A", "revoked": False}
        host.park_round_payload(kind, rid, _payload(rid))
    swept = host.revoke_for_run("run-x", {"approval": lambda st: stamped.append("a"), "question": lambda st: stamped.append("q"), "skill_script": lambda st: stamped.append("s")})
    assert swept == {"approval": [("a1", "sess-A")], "question": [("q1", "sess-A")], "skill_script": []}
    assert sorted(stamped) == ["a", "q"]
    assert "a1" not in host.registries["approval"] and host.head_round_payload("approval", "sess-A") is None
    assert "q1" not in host.registries["question"]
    assert "s1" in host.registries["skill_script"], "other run untouched"
    assert "i1" in host.registries["skill_install"], "kind not in the stamp map is never swept"
    assert host.revoke_for_run("", {"approval": lambda st: None}) == {"approval": []}


def test_remount_for_session_pushes_every_kind_and_runs_the_hook_on_a_head(host):
    seams = host._seams
    host.park_round_payload("approval", "r1", _payload("r1"))
    hooked = []
    host.remount_for_session("sess-A", after={"approval": hooked.append, "question": hooked.append})
    assert [p["round_id"] for p in seams.mounted["approval"] if p] == ["r1"]
    assert seams.mounted["question"] == [None]
    assert [p["round_id"] for p in hooked] == ["r1"], "hook fires only for a pushed head"


def test_before_wait_runs_once_inside_the_wait_before_polling():
    seams = FakeSeamsFull()
    host = InterruptRoundHost(seams)
    order = []
    state = {"event": threading.Event(), "run_id": "run-1"}

    def _before():
        order.append("before")
        state["event"].set()

    outcome = host.run_round(
        "approval", "r1", _payload("r1"), state,
        session_id="sess-A", owning_session_id="sess-A", deadline=None, is_parked=False,
        before_wait=_before,
    )
    assert outcome == "decided" and order == ["before"]


def test_remount_for_session_kinds_filter_leaves_other_kinds_alone(host):
    seams = host._seams
    host.park_round_payload("approval", "r1", _payload("r1"))
    host.remount_for_session("sess-A", kinds=("question", "skill_script"))
    assert seams.mounted["approval"] == [] and seams.mounted["question"] == [None]


# --- review round: hooks the reviewer found missing --------------------------


def test_after_remount_hook_fires_for_a_teardown_promoted_head():
    seams = FakeSeamsFull()
    host = InterruptRoundHost(seams)
    fired = []
    host.after_remount["approval"] = fired.append
    host.park_round_payload("approval", "r0", _payload("r0"))
    state = {"event": threading.Event(), "session_id": "sess-A"}
    state["event"].set()
    host.run_round(
        "approval", "r1", _payload("r1"), state,
        session_id="sess-A", owning_session_id="sess-A",
        deadline=None, is_parked=False,
    )
    assert [p["round_id"] for p in fired] == ["r0"]
    # An explicit `after` still wins over the registry.
    explicit = []
    host.remount_head("approval", "sess-A", after=explicit.append)
    assert [p["round_id"] for p in explicit] == ["r0"] and len(fired) == 1


def test_on_outcome_runs_before_the_registry_pop_and_unpark():
    host = InterruptRoundHost(FakeSeamsFull())
    seen = []
    state = {"event": threading.Event(), "session_id": "sess-A"}
    state["event"].set()

    def _on_outcome(outcome):
        seen.append((outcome, "r1" in host.registries["approval"], "r1" in host.payloads["approval"]))

    host.run_round(
        "approval", "r1", _payload("r1"), state,
        session_id="sess-A", owning_session_id="sess-A",
        deadline=None, is_parked=False, on_outcome=_on_outcome,
    )
    assert seen == [("decided", True, True)]
    assert "r1" not in host.registries["approval"]


def test_announce_detached_true_skips_the_mount_and_false_mounts():
    for announced in (True, False):
        seams = FakeSeamsFull()
        host = InterruptRoundHost(seams)
        state = {"event": threading.Event(), "session_id": "sess-A"}
        state["event"].set()
        host.run_round(
            "approval", "r1", _payload("r1"), state,
            session_id="sess-A", owning_session_id="sess-A",
            deadline=None, is_parked=False,
            announce_detached=lambda: announced,
        )
        mounted = [p for p in seams.mounted["approval"] if p is not None]
        assert bool(mounted) is (not announced)


def test_a_state_revoked_before_host_entry_is_never_resurrected():
    seams = FakeSeamsFull()
    host = InterruptRoundHost(seams)
    outcomes = []
    # The bridge pre-registered, a sweep popped and stamped it, then run_round runs.
    state = {"event": threading.Event(), "session_id": "sess-A", "revoked": True}
    state["event"].set()
    result = host.run_round(
        "approval", "r1", _payload("r1"), state,
        session_id="sess-A", owning_session_id="sess-A",
        deadline=None, is_parked=False, on_outcome=outcomes.append,
        before_wait=lambda: outcomes.append("before_wait"),
    )
    assert result == "revoked" and outcomes == ["revoked"]
    assert host.registries["approval"] == {} and host.payloads["approval"] == {}
    assert seams.mounted["approval"] == [] and seams.badges == []
