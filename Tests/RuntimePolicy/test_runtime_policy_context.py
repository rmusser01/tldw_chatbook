from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
import threading

import pytest

import tldw_chatbook.runtime_policy.bootstrap as bootstrap
from tldw_chatbook.runtime_policy.bootstrap import RuntimePolicyContext
from tldw_chatbook.runtime_policy.types import RuntimeSourceState


class RecordingStore:
    def __init__(self, events: list[tuple[str, RuntimeSourceState]]) -> None:
        self.events = events
        self.saved_states: list[RuntimeSourceState] = []

    def save(self, state: RuntimeSourceState) -> None:
        self.events.append(("persist", state))
        self.saved_states.append(state)


class RaisingStore:
    def __init__(self, message: str = "persistence failed") -> None:
        self.message = message
        self.calls = 0

    def save(self, state: RuntimeSourceState) -> None:
        self.calls += 1
        raise OSError(self.message)


def test_commit_persists_before_publish_and_advances_one_snapshot() -> None:
    events: list[tuple[str, RuntimeSourceState]] = []
    initial = RuntimeSourceState()
    candidate = replace(initial, active_source="server")
    store = RecordingStore(events)
    context = RuntimePolicyContext(
        initial,
        store,
        publish=lambda state: events.append(("publish", state)),
    )

    assert context.commit_state(candidate, expected_revision=0) is True

    assert events == [("persist", candidate), ("publish", candidate)]
    assert context.snapshot() == (candidate, 1)
    assert context.state is candidate


def test_stale_commit_returns_false_without_persistence_or_publication() -> None:
    events: list[tuple[str, RuntimeSourceState]] = []
    initial = RuntimeSourceState()
    candidate = replace(initial, active_source="server")
    store = RecordingStore(events)
    context = RuntimePolicyContext(
        initial,
        store,
        publish=lambda state: events.append(("publish", state)),
    )

    assert context.commit_state(candidate, expected_revision=9) is False

    assert context.snapshot() == (initial, 0)
    assert store.saved_states == []
    assert events == []


def test_persistence_failure_leaves_state_revision_and_projection_unchanged() -> None:
    initial = RuntimeSourceState()
    candidate = replace(initial, active_source="server")
    projected: list[RuntimeSourceState] = []
    store = RaisingStore()
    context = RuntimePolicyContext(initial, store, publish=projected.append)

    with pytest.raises(OSError, match="persistence failed"):
        context.commit_state(candidate, expected_revision=0)

    assert context.snapshot() == (initial, 0)
    assert context.state is initial
    assert store.calls == 1
    assert projected == []


def test_projection_failure_is_contained_after_durable_commit_without_payload_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_sentinel = "PROJECTION-STATE-SENTINEL-117"
    exception_sentinel = "PROJECTION-EXCEPTION-SENTINEL-8aa"
    initial = RuntimeSourceState()
    candidate = replace(
        initial,
        active_source="server",
        active_server_id=state_sentinel,
        server_configured=True,
    )
    events: list[tuple[str, RuntimeSourceState]] = []
    store = RecordingStore(events)

    def raising_publish(state: RuntimeSourceState) -> None:
        raise ValueError(exception_sentinel)

    messages: list[str] = []
    sink = bootstrap.logger.add(
        lambda message: messages.append(message.record["message"]),
        level="WARNING",
    )
    try:
        context = RuntimePolicyContext(initial, store, publish=raising_publish)
        assert context.commit_state(candidate, expected_revision=0) is True
    finally:
        bootstrap.logger.remove(sink)

    assert context.snapshot() == (candidate, 1)
    assert store.saved_states == [candidate]
    assert len(messages) == 1
    assert "ValueError" in messages[0]
    assert exception_sentinel not in messages[0]
    assert state_sentinel not in messages[0]


def test_context_state_is_read_only_and_runtime_state_is_immutable() -> None:
    context = RuntimePolicyContext(RuntimeSourceState(), RecordingStore([]))

    with pytest.raises(AttributeError):
        context.state = RuntimeSourceState(active_source="server")
    with pytest.raises(FrozenInstanceError):
        context.state.active_source = "server"

    assert not hasattr(context, "persist")
    assert not hasattr(context, "store")


def test_foreign_thread_mutation_rejects_without_thread_identifiers() -> None:
    initial = RuntimeSourceState()
    candidate = replace(initial, active_source="server")
    events: list[tuple[str, RuntimeSourceState]] = []
    context = RuntimePolicyContext(initial, RecordingStore(events))

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            context.commit_state,
            candidate,
            expected_revision=0,
        )
        with pytest.raises(RuntimeError) as caught:
            future.result()

    assert str(caught.value) == "runtime policy mutation requires the owner thread"
    assert not any(character.isdigit() for character in str(caught.value))
    assert context.snapshot() == (initial, 0)
    assert events == []


def test_snapshot_reads_remain_coherent_during_owner_thread_commits() -> None:
    context = RuntimePolicyContext(RuntimeSourceState(), RecordingStore([]))
    reader_started = threading.Event()
    commits_finished = threading.Event()

    def read_snapshots() -> deque[tuple[RuntimeSourceState, int]]:
        observed: deque[tuple[RuntimeSourceState, int]] = deque(maxlen=256)
        reader_started.set()
        while not commits_finished.is_set():
            observed.append(context.snapshot())
        observed.append(context.snapshot())
        return observed

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(read_snapshots)
        try:
            assert reader_started.wait(timeout=2)
            for revision in range(1, 31):
                candidate = RuntimeSourceState(
                    active_source="server",
                    active_server_id=str(revision),
                    server_configured=True,
                )
                assert context.commit_state(
                    candidate,
                    expected_revision=revision - 1,
                )
        finally:
            commits_finished.set()
        observed = future.result(timeout=2)

    assert observed
    for state, revision in observed:
        if revision == 0:
            assert state == RuntimeSourceState()
        else:
            assert state.active_server_id == str(revision)
            assert state.active_source == "server"
