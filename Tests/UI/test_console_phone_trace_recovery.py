"""Phone trace recovery keeps the captured chat and leaves composer text/revision alone.

Real received custody, controller, store, SQLite and provider gateway; only the
provider adapter and the deliberately missing frozen trace request are doubled.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from functools import partial
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_received_intent_transitions import _command_owner, _outcome
from Tests.Chat.test_console_trace_first_send_atomicity import _force_capture_on
from Tests.private_profile import private_profile_test
from Tests.UI import test_console_compaction_hold_flow as hold_controls
from tldw_chatbook import config
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole, ConsoleRunStatus
from tldw_chatbook.Chat.console_turn_preparation import (
    ConsolePreparationPauseKind,
    ConsoleTurnPreparationState,
)


@pytest.mark.parametrize("draft", ["", "desktop draft", "phone question", "late edit"])
@private_profile_test
async def test_phone_trace_cancel_preserves_both_chat_drafts(
    request, monkeypatch, tmp_path, draft
):
    """Native Cancel must not restore phone text into either desktop composer."""
    monkeypatch.setattr(
        hold_controls,
        "_build_test_app",
        partial(
            hold_controls._build_test_app, user_data_dir=config.get_user_data_dir()
        ),
    )
    _, database, host, adapter, adapter_calls = await hold_controls._held_console(
        monkeypatch, tmp_path
    )
    console = None
    try:
        async with host.run_test(size=(100, 32)) as pilot:
            console, controller, phone, runtime, _ = await hold_controls._ask_console(
                host, pilot, monkeypatch, adapter
            )
            gateway = controller.provider_gateway
            monkeypatch.setattr(
                gateway,
                "resolve_for_send",
                type(gateway).resolve_for_send.__get__(gateway),
            )
            store = controller.store
            store.replace_session_settings(
                phone.id, replace(phone.settings, streaming=False)
            )
            store.set_session_draft(
                phone.id, "before late edit" if draft == "late edit" else draft
            )
            desktop = store.create_session()
            await console._sync_native_console_chat_ui()
            console._console_composer_or_none().load_draft("active desktop untouched")
            console._session._sync_console_session_draft()
            before_send = tuple(
                (s.id, s.draft, s.draft_revision) for s in store.sessions()
            )
            _force_capture_on(monkeypatch)
            monkeypatch.setattr(
                controller, "_build_durable_trace_request", lambda **kwargs: None
            )
            commands, target, revisions = _command_owner(
                SimpleNamespace(session=phone, runtime=runtime)
            )
            turn_id = await commands.send(
                target,
                "phone question",
                "phone-trace-cancel",
                expected_revisions=revisions,
            )
            record = runtime._turn_custody[turn_id]
            proof = record.received_intent.recovery_user_origin
            recovery = runtime.ensure_archive_recovery()
            assert recovery._proofs[id(proof)] is proof
            outcome = await asyncio.wait_for(_outcome(record), 30)
            assert not isinstance(outcome, BaseException), outcome
            assert outcome.accepted
            paused = store.preparation_for_session(phone.id)
            assert (
                paused is not None
                and paused.state is ConsoleTurnPreparationState.PAUSED
            )
            assert paused.pause_kind is ConsolePreparationPauseKind.TRACE_CALL
            continuation = controller._durable_postcommit_continuations[
                paused.preparation_id
            ]
            assert continuation.prepared.preserve_composer
            assert adapter_calls == []
            assert (
                tuple((s.id, s.draft, s.draft_revision) for s in store.sessions())
                == before_send
            )
            assert [
                m.content
                for m in store.messages_for_session(phone.id)
                if m.role is ConsoleMessageRole.USER
            ] == ["phone question"]
            assert (
                id(proof) not in recovery._proofs
                and id(proof) not in recovery._accepted_proofs
            )
            if draft == "late edit":
                store.set_session_draft(phone.id, draft)
                store.set_session_draft(desktop.id, "active late edit")
            before_cancel = tuple(
                (s.id, s.draft, s.draft_revision) for s in store.sessions()
            )
            cancelled = controller.cancel_library_preparation(paused.preparation_id)
            assert cancelled.visible_copy == "Trace-captured send canceled."
            assert (
                tuple((s.id, s.draft, s.draft_revision) for s in store.sessions())
                == before_cancel
            )
            assert store.active_session_id == desktop.id
            assert store.preparation_for_session(phone.id) is None
            assert controller.run_state_for(phone.id).status is ConsoleRunStatus.STOPPED
            assert (
                paused.preparation_id
                not in controller._durable_postcommit_continuations
            )
            assert adapter_calls == []
            stale = controller.cancel_library_preparation(paused.preparation_id)
            assert not stale.accepted
            assert (
                tuple((s.id, s.draft, s.draft_revision) for s in store.sessions())
                == before_cancel
            )
    finally:
        if console is not None:
            await console._console_runtime().dispose()
        database.close()


@pytest.mark.parametrize("rebuild", [True, False])
@private_profile_test
async def test_named_phone_trace_retry_uses_frozen_turn_after_tab_switch(
    request, monkeypatch, tmp_path, rebuild
):
    """Retry uses the committed phone input, never the newly selected draft."""
    monkeypatch.setattr(
        hold_controls,
        "_build_test_app",
        partial(
            hold_controls._build_test_app, user_data_dir=config.get_user_data_dir()
        ),
    )
    app, database, host, adapter, adapter_calls = await hold_controls._held_console(
        monkeypatch, tmp_path
    )
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="trace-phone", name="Trace phone")
    registry.set_active_workspace("trace-phone")
    assert config.save_settings_to_cli_config({"console": {"exchange_capture": True}})
    app.app_config = config.load_settings(force_reload=True)
    console = None
    try:
        async with host.run_test(size=(100, 32)) as pilot:
            console, controller, phone, runtime, _ = await hold_controls._ask_console(
                host, pilot, monkeypatch, adapter
            )
            assert phone.workspace_id == "trace-phone"
            gateway = controller.provider_gateway
            monkeypatch.setattr(
                gateway,
                "resolve_for_send",
                type(gateway).resolve_for_send.__get__(gateway),
            )
            store = controller.store
            store.replace_session_settings(
                phone.id, replace(phone.settings, streaming=False)
            )
            store.set_session_draft(phone.id, "target desktop draft")
            real_build = controller._build_durable_trace_request
            build_count = 0

            def missing_first_request(**kwargs):
                nonlocal build_count
                build_count += 1
                return real_build(**kwargs) if rebuild and build_count > 1 else None

            monkeypatch.setattr(
                controller, "_build_durable_trace_request", missing_first_request
            )
            commands, target, revisions = _command_owner(
                SimpleNamespace(session=phone, runtime=runtime)
            )
            turn_id = await commands.send(
                target,
                "frozen phone question",
                "phone-trace-retry",
                expected_revisions=revisions,
            )
            outcome = await asyncio.wait_for(
                _outcome(runtime._turn_custody[turn_id]), 30
            )
            assert not isinstance(outcome, BaseException), outcome
            assert outcome.accepted
            paused = store.preparation_for_session(phone.id)
            assert (
                paused is not None
                and paused.pause_kind is ConsolePreparationPauseKind.TRACE_CALL
            )
            continuation = controller._durable_postcommit_continuations[
                paused.preparation_id
            ]
            assert continuation.prepared.preserve_composer
            assert adapter_calls == []
            saved = next(s for s in store.sessions() if s.id == phone.id)
            assert saved.persisted_conversation_id is not None
            assert (
                database.get_conversation_by_id(saved.persisted_conversation_id)[
                    "workspace_id"
                ]
                == "trace-phone"
            )
            desktop = store.create_session()
            await console._sync_native_console_chat_ui()
            console._console_composer_or_none().load_draft(
                "other chat after phone commit"
            )
            console._session._sync_console_session_draft()
            store.set_session_draft(phone.id, "late target desktop draft")
            before = tuple((s.id, s.draft, s.draft_revision) for s in store.sessions())
            retried = await controller.retry_library_preparation(paused.preparation_id)
            assert build_count == 2
            if rebuild:
                assert retried.accepted and retried.provider_started
                assert len(adapter_calls) == 1
                assert adapter_calls[0]["streaming"] is False
                assert "frozen phone question" in str(
                    adapter_calls[0]["messages_payload"]
                )
                assert "other chat after phone commit" not in str(
                    adapter_calls[0]["messages_payload"]
                )
                assert "late target desktop draft" not in str(
                    adapter_calls[0]["messages_payload"]
                )
                assert store.preparation_for_session(phone.id) is None
                assert any(
                    m.role is ConsoleMessageRole.ASSISTANT
                    and m.content.startswith("answer-1 ")
                    for m in store.messages_for_session(phone.id)
                )
            else:
                assert not retried.accepted and not retried.provider_started
                still = store.preparation_for_session(phone.id)
                assert (
                    still is not None and still.preparation_id == paused.preparation_id
                )
                assert still.state is ConsoleTurnPreparationState.PAUSED
                assert adapter_calls == []
                cancelled = controller.cancel_library_preparation(paused.preparation_id)
                assert cancelled.visible_copy == "Trace-captured send canceled."
            assert [
                m.content
                for m in store.messages_for_session(phone.id)
                if m.role is ConsoleMessageRole.USER
            ] == (["frozen phone question"] if rebuild else [])
            assert (
                tuple((s.id, s.draft, s.draft_revision) for s in store.sessions())
                == before
            )
            assert store.active_session_id == desktop.id
    finally:
        if console is not None:
            await console._console_runtime().dispose()
        database.close()
