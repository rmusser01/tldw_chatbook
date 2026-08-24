"""Task 15 fix round 4: close/disposal recovery teardown ratchets."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable

import pytest

from Tests.Chat.test_console_automatic_library_preparation import (
    _PolicyCoordinator,
    _capture_staged_evidence,
    _real_retrieval_controller_for_launch,
    _staged_evidence_launch,
)
from Tests.Chat.test_console_dispatch_recovery import _restored_store
from Tests.Chat.test_console_durable_turn_fix_round2 import _submit_queued
from Tests.Chat.test_console_durable_turn_fix_round1 import (
    _install_real_effect_failure,
)
from Tests.Chat.test_console_first_send_atomicity import _controller
from tldw_chatbook.Chat.console_library_policy import ConsoleAutoRetrieve
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.UI.Console_Modules import retrieval as retrieval_module


def _checkpoint_rows(db: Any) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        tuple(row)
        for row in db.get_connection()
        .execute("SELECT * FROM console_dispatch_checkpoints ORDER BY created_at")
        .fetchall()
    )


async def _accepted_evidence_recovery(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    release_override: Callable[[object, object], None] | None = None,
):
    db, store, controller, gateway = _controller(tmp_path)
    original = _staged_evidence_launch("original")
    newer = _staged_evidence_launch("newer")
    evidence_state: dict[str, object] = {"launch": original, "released": []}
    retrieval = _real_retrieval_controller_for_launch(evidence_state)
    if release_override is not None:
        monkeypatch.setattr(
            retrieval,
            "_release_frozen_console_staged_rag",
            release_override,
        )
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.NEVER)
    controller._rag_capture_provider = retrieval._capture_console_staged_rag
    monkeypatch.setattr(
        retrieval_module,
        "capture_console_staged_evidence_for_chat",
        _capture_staged_evidence,
    )
    _install_real_effect_failure(controller, store, "identity_publication", monkeypatch)
    first = await controller.submit_draft(
        "close-session private evidence", session_id="session-1"
    )
    assert first.accepted is True
    assert first.preparation_id is not None
    assert first.assistant_message_id is not None
    assert evidence_state["released"] == []
    evidence_state["launch"] = newer
    return db, store, controller, gateway, first, evidence_state, original, newer


@pytest.mark.asyncio
async def test_close_session_release_fault_cannot_skip_owner_cleanup(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_attempts = 0

    def fail_release(_launch: object, _result: object) -> None:
        nonlocal release_attempts
        release_attempts += 1
        raise RuntimeError("injected close evidence release failure")

    (
        db,
        store,
        controller,
        gateway,
        first,
        evidence_state,
        _original,
        newer,
    ) = await _accepted_evidence_recovery(
        tmp_path,
        monkeypatch,
        release_override=fail_release,
    )
    checkpoint_before = _checkpoint_rows(db)

    activated = controller.close_session("session-1")

    assert activated is None
    assert release_attempts == 1
    assert evidence_state["released"] == []
    assert evidence_state["launch"] is newer
    assert controller._durable_postcommit_continuations == {}
    assert store.durable_content_retention_count() == 0
    assert store.preparation_by_id(first.preparation_id) is None
    assert controller.preparation_outcome(first.preparation_id) is None
    assert store.sessions() == []
    assert store.active_session_id is None
    assert _checkpoint_rows(db) == checkpoint_before
    assert gateway.calls == 0


@pytest.mark.asyncio
async def test_close_session_releases_exact_evidence_once_and_preserves_replacement(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        db,
        store,
        controller,
        gateway,
        first,
        evidence_state,
        original,
        newer,
    ) = await _accepted_evidence_recovery(tmp_path, monkeypatch)
    checkpoint_before = _checkpoint_rows(db)

    activated = controller.close_session("session-1")

    assert activated is None
    released = evidence_state["released"]
    assert isinstance(released, list)
    assert len(released) == 1
    assert released[0][0] is original
    assert evidence_state["launch"] is newer
    assert controller._durable_postcommit_continuations == {}
    assert store.durable_content_retention_count() == 0
    assert store.preparation_by_id(first.preparation_id) is None
    assert store.sessions() == []
    assert _checkpoint_rows(db) == checkpoint_before
    assert gateway.calls == 0


@pytest.mark.asyncio
async def test_app_disposal_drops_runtime_projection_but_loader_rehydrates_sqlite(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db, store, controller, gateway = _controller(tmp_path)
    _install_real_effect_failure(controller, store, "identity_publication", monkeypatch)
    _entry_id, first = await _submit_queued(controller, "restart-only recovery")
    assert first.accepted is True
    assert first.assistant_message_id is not None
    recovery = store.dispatch_recovery_for_session("session-1")
    assert recovery is not None
    store.mark_dispatch_recovery_needed("session-1", first.assistant_message_id)
    assert store.dispatch_recovery_needs_queue_hydration("session-1") is True
    claimed = store.claim_dispatch_recovery_action(
        "session-1", recovery.actions[0].action_id
    )
    assert claimed is not None
    assert store._dispatch_recovery_message_baselines
    checkpoint_before = _checkpoint_rows(db)
    conversation_id = str(
        db.get_connection().execute("SELECT id FROM conversations").fetchone()[0]
    )

    runtime = ConsoleRuntime(SimpleNamespace(persona_buddy_controller=None))
    runtime.set_chat_store(store)
    runtime.set_provider_gateway(gateway)
    runtime.set_chat_controller(controller)
    await runtime.dispose()

    assert store.dispatch_recovery_for_session("session-1") is None
    assert store.dispatch_recovery_for_presentation("session-1") is None
    assert store.dispatch_recovery_blocks_submission("session-1") is False
    assert store.dispatch_recovery_needs_queue_hydration("session-1") is False
    assert store._dispatch_recovery_message_baselines == {}
    assert _checkpoint_rows(db) == checkpoint_before
    assert gateway.calls == 0

    restarted, restarted_session_id = _restored_store(db, conversation_id)
    rehydrated = restarted.dispatch_recovery_for_session(restarted_session_id)
    assert rehydrated is not None
    assert rehydrated.assistant_message_id == first.assistant_message_id
    assert rehydrated.actions
    assert (
        restarted.dispatch_recovery_needs_queue_hydration(restarted_session_id) is True
    )
    assert _checkpoint_rows(db) == checkpoint_before
