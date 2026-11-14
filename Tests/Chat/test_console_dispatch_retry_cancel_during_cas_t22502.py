"""TASK-22502: a cancel during the checkpoint CAS hop must not wedge the retry.

TASK-22205's adversarial review (probe P2): a cancellation that lands while
the pre-dispatch checkpoint CAS await is in flight cancels the EFFECT
coroutine but not the ``to_thread`` worker -- the surviving thread commits
``dispatch_started`` while the effect ledger records the transition as
abandoned. ``retry_dispatch_recovery`` then re-runs the transition with its
ACCEPTED expectations, CONFLICTS against the already-transitioned row, and
fails "Accepted turn is retained for recovery" on EVERY retry: an
in-process wedge until restart (the restart path's
``reconcile_for_session`` classifies the same DB state correctly).

Not reachable by any production path today (only ``close_session`` -- which
retires the continuation -- and shutdown cancel that task); these tests
force the interleave so the hardening cannot silently rot.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_first_send_atomicity import (
    _controller,
)
from Tests.private_profile import private_profile_test


@private_profile_test
async def test_retry_recovers_after_cancel_during_checkpoint_cas(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A DISPATCH_STARTED checkpoint must satisfy the transition effect.

    Forces the exact DB state a cancel-during-CAS interleave leaves (TASK-22205
    probe P2): the surviving worker thread's CAS commits ``dispatch_started``
    while the effect re-run still holds its ACCEPTED expectations. The
    transition effect must accept the already-committed owner instead of
    conflicting on every retry -- the in-process wedge until restart, whose
    restart-side classification ``reconcile_for_session`` already handles.

    The duplicate committer here is the test thread (deterministic) rather
    than a cancelled coroutine's surviving ``to_thread`` worker; the DB state
    the effect's CAS then meets -- and must tolerate -- is identical.
    """
    db, store, controller, gateway = _controller(tmp_path)
    repository = store.persistence.console_dispatch_repository
    session_id = "session-1"

    # threading.Event, not asyncio.Event: the wrapper runs on the
    # to_thread worker while the assertions drive the loop.
    cas_entered = threading.Event()
    cas_release = threading.Event()
    real_cas_state = repository.cas_state
    captured: dict[str, object] = {}

    def gated_cas_state(transition):
        cas_entered.set()
        captured["transition"] = transition
        cas_release.wait(timeout=10)
        return real_cas_state(transition)

    monkeypatch.setattr(repository, "cas_state", gated_cas_state)

    submit_task = asyncio.create_task(
        controller.submit_draft("cancel during CAS", session_id=session_id)
    )
    while not cas_entered.is_set():
        assert not submit_task.done(), "submit finished before the CAS was entered"
        await asyncio.sleep(0.01)

    # The surviving-thread commit: while the effect's own CAS is held on
    # the worker, commit the SAME transition through the repository from
    # the test thread -- the exact DB state a cancel-during-CAS leaves
    # (dispatch_started durably committed, revision bumped) with the
    # effect's re-run still holding its pre-transition expectations.
    transition = captured["transition"]
    committed = real_cas_state(transition)
    assert committed.status.value == "committed", (
        "the surviving-thread simulation did not commit dispatch_started"
    )
    conversation_id = (
        db.get_connection().execute("SELECT id FROM conversations").fetchone()["id"]
    )
    live = repository.read_for_session(conversation_id)
    assert live.checkpoint is not None
    assert live.checkpoint.state.value == "dispatch_started"

    # Release the held CAS: the effect's own transition now CONFLICTS with
    # the already-committed owner. Pre-fix this raised and abandoned the
    # effect on every resume (the in-process wedge); the fix accepts the
    # already-DISPATCH_STARTED checkpoint and the turn completes.
    cas_release.set()
    result = await submit_task

    assert result.accepted is True, (
        f"the send wedged against the already-dispatch_started checkpoint: "
        f"{result.visible_copy!r}"
    )
    assert gateway.calls == 1, "the provider was never entered"
    preparation_id = live.checkpoint.preparation_id
    fingerprint = store.durable_acceptance_fingerprint_for(preparation_id)
    from tldw_chatbook.Chat.console_chat_store import (
        ConsoleDurableAcceptanceRetired,
    )

    try:
        effects = store.durable_postcommit_effects_for(
            preparation_id, fingerprint=fingerprint
        )
    except ConsoleDurableAcceptanceRetired:
        # A fully settled turn retires its acceptance -- equally complete.
        effects = None
    if effects is not None:
        assert "checkpoint_transition" in effects.completed, (
            "the ledger effect never completed against the accepted checkpoint"
        )
    settled = store.dispatch_recovery_for_session(session_id)
    assert settled is None or not settled.recovery_needed, (
        "a recovery owner survived a completed turn"
    )


@private_profile_test
async def test_memory_backed_persistence_never_offloads_durable_calls(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> None:
    """The ``:memory:`` offload gate: per-connection DBs stay on the loop.

    A ``:memory:`` CharactersRAGDB is per-connection -- a worker thread
    would see an empty, unmigrated database -- so ``_run_durable_db_call``
    must keep those transactions inline. Also pins the fail-safe default
    for persistence shapes without a ``.db`` attribute (test fakes).
    """
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    file_db = CharactersRAGDB(tmp_path / "file.sqlite", client_id="t22502-file")
    file_controller = ConsoleChatController(
        store=ConsoleChatStore(persistence=ChatPersistenceService(file_db)),
        provider_gateway=SimpleNamespace(),
        provider="llama_cpp",
        model="test-model",
    )
    assert file_controller._durable_db_call_offloadable() is True

    memory_db = CharactersRAGDB(":memory:", client_id="t22502-memory")
    memory_controller = ConsoleChatController(
        store=ConsoleChatStore(persistence=ChatPersistenceService(memory_db)),
        provider_gateway=SimpleNamespace(),
        provider="llama_cpp",
        model="test-model",
    )
    assert memory_controller._durable_db_call_offloadable() is False

    fake_controller = ConsoleChatController(
        store=ConsoleChatStore(persistence=SimpleNamespace()),
        provider_gateway=SimpleNamespace(),
        provider="llama_cpp",
        model="test-model",
    )
    assert fake_controller._durable_db_call_offloadable() is True, (
        "a persistence shape without .db has no connection to starve; the "
        "documented contract (TASK-22205) lets test fakes thread harmlessly"
    )
