"""Task 15 fix round 3: bounded teardown and prerequisite-safe Discard."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from Tests.console_resource_fixtures import (
    close_owned_console_resources as close_owned_console_resources,
)

from Tests.Chat.test_console_automatic_library_preparation import (
    _PolicyCoordinator,
    _capture_staged_evidence,
    _real_retrieval_controller_for_launch,
    _staged_evidence_launch,
)
from Tests.Chat.test_console_dispatch_recovery_fix_round2 import (
    _assert_exact_postcommit_recovery,
)
from Tests.Chat.test_console_durable_turn_fix_round1 import (
    _install_real_effect_failure,
)
from Tests.Chat.test_console_first_send_atomicity import _controller
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_library_policy import ConsoleAutoRetrieve
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.UI.Console_Modules import retrieval as retrieval_module


_EARLY_POSTCOMMIT_EFFECTS = (
    "identity_publication",
    "durable_owner_publication",
    "staged_input_clearing",
    "preparation_publication",
)


def _checkpoint_count(db: Any) -> int:
    return int(
        db.get_connection()
        .execute("SELECT COUNT(*) FROM console_dispatch_checkpoints")
        .fetchone()[0]
    )


@pytest.mark.asyncio
async def test_app_disposal_retires_live_postcommit_bodies_even_without_tasks(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db, store, controller, gateway = _controller(tmp_path)
    _install_real_effect_failure(controller, store, "accepted_hook", monkeypatch)
    first = await controller.submit_draft(
        "app-disposal private body", session_id="session-1"
    )
    assert first.accepted is True
    assert first.preparation_id is not None
    assert len(controller._durable_postcommit_continuations) == 1
    assert store.durable_content_retention_count() == 2
    assert _checkpoint_count(db) == 1

    runtime = ConsoleRuntime(SimpleNamespace(persona_buddy_controller=None))
    runtime.set_chat_store(store)
    runtime.set_provider_gateway(gateway)
    runtime.set_chat_controller(controller)
    await runtime.dispose()

    assert controller._durable_postcommit_continuations == {}
    assert store.durable_content_retention_count() == 0
    assert store.durable_tombstone_count() == 1
    assert store.preparation_by_id(first.preparation_id) is None
    assert controller.preparation_outcome(first.preparation_id) is None
    assert _checkpoint_count(db) == 1
    assert gateway.calls == 0
    body_free_retention = repr(store.durable_retention_debug_snapshot())
    assert "app-disposal private body" not in body_free_retention


@pytest.mark.asyncio
async def test_app_disposal_releases_only_its_frozen_evidence_lease(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db, store, controller, gateway = _controller(tmp_path)
    original = _staged_evidence_launch("original")
    newer = _staged_evidence_launch("newer")
    evidence_state: dict[str, object] = {"launch": original, "released": []}
    retrieval = _real_retrieval_controller_for_launch(evidence_state)
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.NEVER)
    controller._rag_capture_provider = retrieval._capture_console_staged_rag
    monkeypatch.setattr(
        retrieval_module,
        "capture_console_staged_evidence_for_chat",
        _capture_staged_evidence,
    )
    _install_real_effect_failure(controller, store, "identity_publication", monkeypatch)
    first = await controller.submit_draft(
        "app-disposal evidence body", session_id="session-1"
    )
    assert first.accepted is True
    assert first.preparation_id is not None
    assert evidence_state["released"] == []
    evidence_state["launch"] = newer

    runtime = ConsoleRuntime(SimpleNamespace(persona_buddy_controller=None))
    runtime.set_chat_store(store)
    runtime.set_provider_gateway(gateway)
    runtime.set_chat_controller(controller)
    await runtime.dispose()

    released = evidence_state["released"]
    assert isinstance(released, list)
    assert len(released) == 1
    assert released[0][0] is original
    assert evidence_state["launch"] is newer
    assert controller._durable_postcommit_continuations == {}
    assert store.durable_content_retention_count() == 0
    assert store.preparation_by_id(first.preparation_id) is None
    assert _checkpoint_count(db) == 1


@pytest.mark.asyncio
async def test_app_disposal_scrubs_content_when_evidence_release_raises(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db, store, controller, gateway = _controller(tmp_path)
    evidence_state: dict[str, object] = {
        "launch": _staged_evidence_launch("faulting"),
        "released": [],
    }
    retrieval = _real_retrieval_controller_for_launch(evidence_state)
    release_attempts = 0

    def fail_release(_launch: object, _result: object) -> None:
        nonlocal release_attempts
        release_attempts += 1
        raise RuntimeError("injected evidence release failure")

    monkeypatch.setattr(
        retrieval,
        "_release_frozen_console_staged_rag",
        fail_release,
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
        "release-fault private body", session_id="session-1"
    )
    assert first.accepted is True
    assert first.preparation_id is not None

    runtime = ConsoleRuntime(SimpleNamespace(persona_buddy_controller=None))
    runtime.set_chat_store(store)
    runtime.set_provider_gateway(gateway)
    runtime.set_chat_controller(controller)
    await runtime.dispose()

    assert release_attempts == 1
    assert controller._durable_postcommit_continuations == {}
    assert store.durable_content_retention_count() == 0
    assert store.preparation_by_id(first.preparation_id) is None
    assert _checkpoint_count(db) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("effect_name", _EARLY_POSTCOMMIT_EFFECTS)
async def test_discard_finishes_prerequisites_without_dispatch_and_unwedges_next_send(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    effect_name: str,
) -> None:
    db, store, controller, gateway = _controller(tmp_path)
    counts = _install_real_effect_failure(controller, store, effect_name, monkeypatch)
    session = store.sessions()[0]
    session.draft = "retained draft"

    first = await controller.submit_draft(session.draft, session_id="session-1")
    assert first.accepted is True
    assert first.preparation_id is not None
    assert counts == {"attempts": 1, "successes": 0}
    assert session.draft == (
        "" if effect_name == "preparation_publication" else "retained draft"
    )
    _assert_exact_postcommit_recovery(
        controller,
        assistant_message_id=first.assistant_message_id or "",
    )

    discarded = await controller.discard_dispatch_recovery("session-1")

    assert discarded.accepted is True
    assert counts == {"attempts": 2, "successes": 1}
    assert gateway.calls == 0
    assert session.persisted_conversation_id is not None
    assert session.draft == ""
    assert _checkpoint_count(db) == 0
    assert store.dispatch_recovery_for_session("session-1") is None
    assert store.preparation_by_id(first.preparation_id) is None
    assert controller.preparation_outcome(first.preparation_id) is None
    assert controller._durable_postcommit_continuations == {}
    assert store.durable_content_retention_count() == 0
    tombstone = store.durable_retention_debug_snapshot()[0]
    assert "preparation_publication" in tombstone.completed
    assert "checkpoint_transition" not in tombstone.completed
    assert "provider_entry" not in tombstone.completed

    next_send = await controller.submit_draft(
        "next send is not wedged", session_id="session-1"
    )
    assert next_send.accepted is True
    assert "Another send is still preparing" not in next_send.visible_copy
    assert gateway.calls == 1


@pytest.mark.asyncio
async def test_discard_prerequisite_failure_retains_every_recovery_owner(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db, store, controller, gateway = _controller(tmp_path)
    attempts = 0
    original_identity_publication = store.publish_durable_turn_identity

    def fail_identity(*_args: Any, **_kwargs: Any) -> None:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("injected persistent identity publication")

    monkeypatch.setattr(store, "publish_durable_turn_identity", fail_identity)
    first = await controller.submit_draft("retained body", session_id="session-1")
    assert first.accepted is True
    assert first.preparation_id is not None
    before_rows = tuple(
        db.get_connection().execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("conversations", "messages", "console_dispatch_checkpoints")
    )

    discarded = await controller.discard_dispatch_recovery("session-1")

    assert discarded.accepted is False
    assert attempts == 2
    assert gateway.calls == 0
    assert (
        tuple(
            db.get_connection().execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("conversations", "messages", "console_dispatch_checkpoints")
        )
        == before_rows
    )
    _assert_exact_postcommit_recovery(
        controller,
        assistant_message_id=first.assistant_message_id or "",
    )
    assert first.preparation_id in controller._durable_postcommit_continuations
    assert store.durable_content_retention_count() == 2

    monkeypatch.setattr(
        store,
        "publish_durable_turn_identity",
        original_identity_publication,
    )
    retried_discard = await controller.discard_dispatch_recovery("session-1")
    assert retried_discard.accepted is True
    assert gateway.calls == 0
    assert _checkpoint_count(db) == 0
    assert store.preparation_by_id(first.preparation_id) is None
    assert controller._durable_postcommit_continuations == {}
    assert store.durable_content_retention_count() == 0


@pytest.mark.asyncio
async def test_replacement_controller_cannot_discard_unfinished_live_continuation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db, store, controller, gateway = _controller(tmp_path)
    counts = _install_real_effect_failure(
        controller, store, "identity_publication", monkeypatch
    )
    first = await controller.submit_draft("retained body", session_id="session-1")
    assert first.accepted is True
    replacement = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="llama_cpp",
        model="test-model",
        agent_runtime_enabled=False,
    )

    refused = await replacement.discard_dispatch_recovery("session-1")

    assert refused.accepted is False
    assert "continuation is unavailable" in refused.visible_copy.lower()
    assert counts == {"attempts": 1, "successes": 0}
    assert gateway.calls == 0
    assert _checkpoint_count(db) == 1
    _assert_exact_postcommit_recovery(
        controller,
        assistant_message_id=first.assistant_message_id or "",
    )


@pytest.mark.asyncio
async def test_discard_releases_only_exact_frozen_evidence_once(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db, store, controller, gateway = _controller(tmp_path)
    original = _staged_evidence_launch("original")
    newer = _staged_evidence_launch("newer")
    evidence_state: dict[str, object] = {"launch": original, "released": []}
    retrieval = _real_retrieval_controller_for_launch(evidence_state)
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.NEVER)
    controller._rag_capture_provider = retrieval._capture_console_staged_rag
    monkeypatch.setattr(
        retrieval_module,
        "capture_console_staged_evidence_for_chat",
        _capture_staged_evidence,
    )
    _install_real_effect_failure(controller, store, "identity_publication", monkeypatch)

    first = await controller.submit_draft("evidence body", session_id="session-1")
    assert first.accepted is True
    assert evidence_state["released"] == []
    evidence_state["launch"] = newer

    discarded = await controller.discard_dispatch_recovery("session-1")
    repeated = await controller.discard_dispatch_recovery("session-1")

    assert discarded.accepted is True
    assert repeated.accepted is False
    released = evidence_state["released"]
    assert isinstance(released, list)
    assert len(released) == 1
    assert released[0][0] is original
    assert evidence_state["launch"] is newer
    assert gateway.calls == 0
    assert _checkpoint_count(db) == 0
