"""Mounted proof that accepted Console turns survive real screen navigation.

These tests deliberately cross the production ownership boundary instead of
calling ``ChatScreen`` teardown helpers.  A real ``TldwCli`` mounts Console,
the runtime accepts the turn, ``NavigateToScreen`` removes the view, and the
same app-owned store/controller finish the work before a fresh Console screen
reconciles it.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, replace
from typing import Any
from uuid import uuid4

import pytest
from loguru import logger

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_fleet_wake_wiring import _attach_real_dbs
from Tests.UI.test_console_headless_approval import (
    _arm,
    _armed_round_ids,
    _risk_row,
    _wait_for_round,
)
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_console_store_continuity import (
    _StallingWakeGateway,
    _db_chain,
    _navigate,
    _rendered_text,
    _seed_console,
)
from tldw_chatbook.Agents.agent_models import (
    RUN_DONE,
    STEP_SPAWN,
    STEP_TOOL_RESULT,
    AgentStep,
    RunOutcome,
)
from tldw_chatbook.Chat.attachment_core import PendingAttachment
from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch
from tldw_chatbook.Chat.message_metadata import terminal_receipt_id_for_message
from tldw_chatbook.Chat.console_turn_context import ConsoleTurnCustodyRequest
from tldw_chatbook.Character_Chat.Character_Chat_Lib import (
    export_conversation_to_json,
)
from tldw_chatbook.UI.Console_Modules.wiring import _admit_console_turn_to_runtime
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


@dataclass
class _DetachedViewCounters:
    dom_queries: int = 0
    transcript_polls: int = 0
    shell_updates: int = 0


class _TwoChunkGateway(_StallingWakeGateway):
    """Direct-provider double with barriers after each streamed delta."""

    def __init__(self) -> None:
        super().__init__()
        self.chunk_one = ""
        self.chunk_two = ""
        self.two_chunk_mode = False
        self.first_chunk = asyncio.Event()
        self.release_second = asyncio.Event()
        self.second_chunk = asyncio.Event()
        self.release_terminal = asyncio.Event()

    def arm_two_chunks(self, marker: str) -> None:
        self.chunk_one = f"{marker}-A"
        self.chunk_two = f"{marker}-B"
        self.two_chunk_mode = True
        self.first_chunk = asyncio.Event()
        self.release_second = asyncio.Event()
        self.second_chunk = asyncio.Event()
        self.release_terminal = asyncio.Event()

    async def stream_chat(self, resolution, messages, **kwargs):
        self.payloads.append([dict(message) for message in messages])
        if not self.two_chunk_mode:
            yield self.reply
            return
        yield self.chunk_one
        self.first_chunk.set()
        await self.release_second.wait()
        yield self.chunk_two
        self.second_chunk.set()
        await self.release_terminal.wait()


class _TwoChunkAgentBridge:
    """Agent-bridge seam that blocks on the worker thread between deltas."""

    def __init__(
        self,
        store,
        marker: str,
        *,
        two_chunk_mode: bool,
        steps: tuple[AgentStep, ...] = (),
    ) -> None:
        self.store = store
        self.marker = marker
        self.two_chunk_mode = two_chunk_mode
        self.first_chunk = threading.Event()
        self.release_second = threading.Event()
        self.second_chunk = threading.Event()
        self.release_terminal = threading.Event()
        self.calls: list[dict[str, Any]] = []
        self.recorded_assistant_ids: list[tuple[str, str]] = []
        self.steps = steps
        self.last_outcome: RunOutcome | None = None

    def native_tool_schemas(self):
        return []

    def subagent_counts(self, conversation_ids):
        return {conversation_id: 0 for conversation_id in conversation_ids}

    def record_run_assistant_message(self, run_id: str, message_id: str) -> None:
        self.recorded_assistant_ids.append((run_id, message_id))

    def run_reply(self, **kwargs):
        self.calls.append(dict(kwargs))
        message_id = kwargs["assistant_message_id"]
        first = f"{self.marker}-A"
        second = f"{self.marker}-B"
        if not self.two_chunk_mode:
            self.store.append_stream_chunk(message_id, first)
            self.store.append_stream_chunk(message_id, second)
            self.last_outcome = RunOutcome(
                status=RUN_DONE,
                steps=list(self.steps),
                final_text=first + second,
            )
            return "run-navigation", self.last_outcome
        self.store.append_stream_chunk(message_id, first)
        self.first_chunk.set()
        if not self.release_second.wait(10):
            raise TimeoutError("test never released the agent's second delta")
        self.store.append_stream_chunk(message_id, second)
        self.second_chunk.set()
        if not self.release_terminal.wait(10):
            raise TimeoutError("test never released agent terminalization")
        self.last_outcome = RunOutcome(
            status=RUN_DONE,
            steps=list(self.steps),
            final_text=first + second,
        )
        return "run-navigation", self.last_outcome


class _WorkerTerminalBridge:
    """Agent seam that enters terminal persistence on its worker thread."""

    def __init__(self, store, final_content: str) -> None:
        self.store = store
        self.final_content = final_content
        self.calls = 0

    def native_tool_schemas(self):
        return []

    def record_run_assistant_message(self, _run_id: str, _message_id: str) -> None:
        return None

    def run_reply(self, **kwargs):
        self.calls += 1
        message_id = kwargs["assistant_message_id"]
        self.store.append_stream_chunk(message_id, self.final_content)
        self.store.mark_message_complete(message_id)
        return "run-terminal-transaction", RunOutcome(
            status=RUN_DONE,
            steps=[],
            final_text=self.final_content,
        )


def _build_navigation_app(tmp_path):
    """Build a production-shaped app with isolated file-backed data."""
    app = _build_test_app()
    _attach_real_dbs(app, tmp_path)
    _configure_native_ready_console(app)
    gateway = _TwoChunkGateway()
    app.console_provider_gateway_factory = lambda: gateway
    app.app_config.setdefault("console", {})["agent_runtime"] = False
    # Keep the injected test snapshot authoritative rather than reloading a
    # developer profile after the mounted app begins refreshing settings.
    app.app_config.pop("logging", None)
    return app, gateway


def _install_detached_view_counters(monkeypatch, app):
    """Count forbidden work against the outgoing, already-detached view."""
    state: dict[str, Any] = {
        "outgoing": None,
        "detached": False,
    }
    counters = _DetachedViewCounters()

    original_query = ChatScreen.query
    original_query_one = ChatScreen.query_one
    original_sync = ChatScreen._sync_native_console_chat_ui

    def counted_query(screen, *args, **kwargs):
        if state["detached"] and screen is state["outgoing"]:
            counters.dom_queries += 1
        return original_query(screen, *args, **kwargs)

    def counted_query_one(screen, *args, **kwargs):
        if state["detached"] and screen is state["outgoing"]:
            counters.dom_queries += 1
        return original_query_one(screen, *args, **kwargs)

    async def counted_sync(screen, *args, **kwargs):
        if state["detached"] and screen is state["outgoing"]:
            counters.transcript_polls += 1
        return await original_sync(screen, *args, **kwargs)

    monkeypatch.setattr(ChatScreen, "query", counted_query)
    monkeypatch.setattr(ChatScreen, "query_one", counted_query_one)
    monkeypatch.setattr(ChatScreen, "_sync_native_console_chat_ui", counted_sync)

    original_projection = app.set_console_attention_projection

    def counted_projection(needs_attention: bool) -> None:
        if state["detached"]:
            counters.shell_updates += 1
        original_projection(needs_attention)

    monkeypatch.setattr(app, "set_console_attention_projection", counted_projection)
    return state, counters


async def _wait_thread_event(event: threading.Event) -> None:
    assert await asyncio.to_thread(event.wait, 5), (
        "worker-thread barrier was not reached"
    )


def _unique_terminal_rows(store, session_id: str, content: str):
    return [
        message
        for message in store.messages_for_session(session_id)
        if message.content == content
    ]


async def _assert_hidden_turn_reconciles_all_four_layers(
    *,
    app,
    pilot,
    controller,
    store,
    runtime,
    gateway,
    session_id: str,
    conversation_id: str,
    final_content: str,
    submit_follow_up: bool = True,
):
    """Assert live, mounted, durable, and continuation evidence separately."""
    terminal_rows = _unique_terminal_rows(store, session_id, final_content)
    assert len(terminal_rows) == 1
    receipt_id = terminal_receipt_id_for_message(terminal_rows[0])
    assert receipt_id
    marks_service = runtime._console_local_marks_service()
    assert (conversation_id, receipt_id) in marks_service.list_console_unseen_marks()
    assert [row[2] for row in _db_chain(app.chachanotes_db, conversation_id)].count(
        final_content
    ) == 1

    reopened = await _navigate(app, pilot, "chat", expect="ChatScreen")
    await pilot.pause()
    assert _rendered_text(reopened).count(final_content) == 1
    assert (conversation_id, receipt_id) not in (
        marks_service.list_console_unseen_marks()
    )

    if not submit_follow_up:
        provider_payload = controller._provider_messages_for_session(session_id)
        payload_text = "\n".join(
            str(entry.get("content", "")) for entry in provider_payload
        )
        assert payload_text.count(final_content) == 1
        return reopened

    controller._agent_runtime_enabled = False
    gateway.two_chunk_mode = False
    gateway.stall = False
    gateway.stall_stream = False
    gateway.reply = "PHASE-FOLLOW-UP"
    gateway.payloads.clear()
    follow_up = await controller.submit_draft(
        "continue the same phase-tested lineage",
        session_id=session_id,
    )
    assert follow_up.accepted
    payload_text = "\n".join(
        str(entry.get("content", "")) for entry in gateway.payloads[-1]
    )
    assert payload_text.count(final_content) == 1
    return reopened


@pytest.mark.asyncio
async def test_pre_durable_custody_does_not_publish_sensitive_inputs(
    tmp_path,
    monkeypatch,
):
    """Navigation exposes only opaque custody and sanitized shell state."""
    app, gateway = _build_navigation_app(tmp_path)
    notifications: list[str] = []
    log_records: list[str] = []
    sink_id = logger.add(
        lambda message: log_records.append(str(message)),
        level="DEBUG",
        format="{message} {extra}",
    )

    try:
        async with app.run_test(size=(160, 48), notifications=True) as pilot:
            chat, _controller, store, session_id, conversation_id = await _seed_console(
                app, pilot, gateway
            )
            runtime = chat._console_runtime()
            monkeypatch.setattr(
                app,
                "notify",
                lambda message, *args, **kwargs: notifications.append(str(message)),
            )

            secrets = (
                "PRIVACY-PROMPT-CANARY",
                "/private/PRIVACY-ATTACHMENT-PATH-CANARY.bin",
                "PRIVACY-ATTACHMENT-NAME-CANARY.bin",
                "PRIVACY-ATTACHMENT-BYTES-CANARY",
                "PRIVACY-RAG-CONTEXT-CANARY",
                "PRIVACY-TOOL-ARGUMENT-CANARY",
                "PRIVACY-TOOL-RESULT-CANARY",
                "PRIVACY-CREDENTIAL-CANARY",
                "PRIVACY-EXCEPTION-BODY-CANARY",
            )
            attachment = PendingAttachment(
                file_path=secrets[1],
                display_name=secrets[2],
                file_type="binary",
                insert_mode="attachment",
                data=secrets[3].encode(),
                mime_type="application/octet-stream",
                original_size=len(secrets[3]),
                processed_size=len(secrets[3]),
                attachment_id=str(uuid4()),
            )
            assert store.add_pending_attachment(session_id, attachment)

            staged_evidence = ConsoleLiveWorkLaunch.from_values(
                source=secrets[4],
                title=secrets[2],
                payload={
                    "tool_arguments": secrets[5],
                    "tool_result": secrets[6],
                    "exception": secrets[8],
                },
            )
            runtime.stage_console_staged_evidence(staged_evidence)
            base_configuration = chat._session._build_console_turn_execution_context(
                session_id
            )
            configuration = replace(
                base_configuration,
                provider_selection=replace(
                    base_configuration.provider_selection,
                    system_prompt=secrets[7],
                ),
                rag_defaults={"context": secrets[4]},
                tool_configuration={
                    "arguments": secrets[5],
                    "result": secrets[6],
                },
                provider_payload_settings={
                    "credential": secrets[7],
                    "exception": secrets[8],
                },
            )
            request = ConsoleTurnCustodyRequest(
                turn_id=str(uuid4()),
                session_id=session_id,
                draft=secrets[0],
                configuration=configuration,
                attachment_ids=(attachment.attachment_id,),
                staged_evidence_launch=staged_evidence,
            )

            entered_custody = asyncio.Event()
            release_custody = asyncio.Event()

            async def hold_before_durable_acceptance(*_args, **_kwargs):
                entered_custody.set()
                await release_custody.wait()
                return None

            monkeypatch.setattr(
                runtime,
                "_run_custodied_turn",
                hold_before_durable_acceptance,
            )
            connection = app.chachanotes_db.get_connection()
            connection.execute("DELETE FROM sync_log")
            connection.commit()
            gateway.payloads.clear()
            notifications.clear()
            log_records.clear()

            turn_id = runtime.accept_turn(request)
            task = runtime._turn_custody[turn_id].task
            assert task is not None
            await asyncio.wait_for(entered_custody.wait(), timeout=3)

            try:
                await _navigate(
                    app,
                    pilot,
                    "library",
                    expect="LibraryScreen",
                    allow_confirmation=False,
                )
                record = runtime._turn_custody[turn_id]
                nav_labels = [str(button.label) for button in app.query(".nav-button")]
                exported = export_conversation_to_json(
                    app.chachanotes_db,
                    conversation_id,
                )
                assert exported is not None
                assert gateway.payloads == []
                owner_surfaces = "\n".join(
                    (
                        repr(request),
                        repr(record),
                        repr(record.inputs),
                        repr(runtime._turn_custody),
                        repr(log_records),
                        repr(notifications),
                        repr(nav_labels),
                        repr(app.chachanotes_db.get_sync_log_entries()),
                        exported,
                        repr(gateway.payloads),
                    )
                )
                for secret in secrets:
                    assert secret not in owner_surfaces
            finally:
                release_custody.set()
                await asyncio.wait_for(task, timeout=3)
    finally:
        logger.remove(sink_id)


@pytest.mark.parametrize("origin", ["ordinary", "main_agent"])
@pytest.mark.parametrize("race", ["before_task", "during_stream"])
@pytest.mark.asyncio
async def test_accepted_turn_survives_both_outer_navigation_races(
    tmp_path,
    monkeypatch,
    origin,
    race,
):
    """Every accepted outer path finishes once with no detached view work."""
    app, gateway = _build_navigation_app(tmp_path)
    counter_state, counters = _install_detached_view_counters(monkeypatch, app)

    async with app.run_test(size=(160, 48), notifications=True) as pilot:
        chat, controller, store, session_id, conversation_id = await _seed_console(
            app, pilot, gateway
        )
        runtime = chat._console_runtime()
        marker = f"NAV-{origin.upper()}-{race.upper()}"
        final_content = f"{marker}-A{marker}-B"
        agent_bridge = None
        if origin == "main_agent":
            agent_bridge = _TwoChunkAgentBridge(
                store,
                marker,
                two_chunk_mode=race == "during_stream",
            )
            app.app_config["console"]["agent_runtime"] = True
            monkeypatch.setattr(
                chat,
                "_ensure_console_agent_bridge",
                lambda: agent_bridge,
            )
            controller._agent_runtime_enabled = True
            controller._agent_bridge = agent_bridge
        else:
            gateway.reply = final_content
            if race == "during_stream":
                gateway.arm_two_chunks(marker)

        pretask_entered = asyncio.Event()
        pretask_release = asyncio.Event()
        if race == "before_task":
            original_run = runtime._run_custodied_turn

            async def hold_before_domain_task(*args, **kwargs):
                pretask_entered.set()
                await pretask_release.wait()
                return await original_run(*args, **kwargs)

            monkeypatch.setattr(runtime, "_run_custodied_turn", hold_before_domain_task)

        turn_id = _admit_console_turn_to_runtime(chat, marker, session_id)
        custody = runtime._turn_custody[turn_id]
        task = custody.task
        assert task is not None
        if race == "before_task":
            await asyncio.wait_for(pretask_entered.wait(), timeout=3)
        elif origin == "ordinary":
            await asyncio.wait_for(gateway.first_chunk.wait(), timeout=3)
        else:
            assert agent_bridge is not None
            await _wait_thread_event(agent_bridge.first_chunk)

        counter_state["outgoing"] = chat
        await _navigate(
            app,
            pilot,
            "library",
            expect="LibraryScreen",
            allow_confirmation=False,
        )
        assert runtime.view is None
        assert runtime.chat_store is store
        assert runtime.chat_controller is controller
        assert not task.done()
        counter_state["detached"] = True

        if race == "before_task":
            pretask_release.set()
        elif origin == "ordinary":
            gateway.release_second.set()
            await asyncio.wait_for(gateway.second_chunk.wait(), timeout=3)
            assert counters.shell_updates == 0, (
                "a detached stream delta drove shell navigation state"
            )
            gateway.release_terminal.set()
        else:
            assert agent_bridge is not None
            agent_bridge.release_second.set()
            await _wait_thread_event(agent_bridge.second_chunk)
            assert counters.shell_updates == 0, (
                "a detached agent delta drove shell navigation state"
            )
            agent_bridge.release_terminal.set()

        outcome = await asyncio.wait_for(task, timeout=8)
        assert outcome.accepted
        await pilot.pause()

        # Layer 1: the app-owned store advanced while there was no Console.
        terminal_rows = _unique_terminal_rows(store, session_id, final_content)
        assert len(terminal_rows) == 1, (
            [
                (message.role.value, message.status, message.content)
                for message in store.messages_for_session(session_id)
            ],
            len(agent_bridge.calls) if agent_bridge is not None else None,
        )
        assert counters.dom_queries == 0
        assert counters.transcript_polls == 0
        # A terminal receipt may publish one coalesced attention transition;
        # the delta assertions above prove tokens themselves publish none.
        assert counters.shell_updates <= 1

        # Layer 3a: hidden terminalization committed one durable row and one
        # exact local-only receipt before a returning view acknowledged it.
        db_rows = _db_chain(app.chachanotes_db, conversation_id)
        assert [row[2] for row in db_rows].count(final_content) == 1
        marks_service = runtime._console_local_marks_service()
        hidden_marks = marks_service.list_console_unseen_marks()
        receipt_id = terminal_receipt_id_for_message(terminal_rows[0])
        assert receipt_id
        assert (conversation_id, receipt_id) in hidden_marks

        # Layer 2: a new screen, not a cached one, renders the identified row
        # once and acknowledges only the matching receipt after it mounted.
        counter_state["detached"] = False
        reopened = await _navigate(app, pilot, "chat", expect="ChatScreen")
        assert reopened is not chat
        await pilot.pause()
        rendered = _rendered_text(reopened)
        assert rendered.count(final_content) == 1
        assert (conversation_id, receipt_id) not in (
            marks_service.list_console_unseen_marks()
        )

        # Layer 4: the next provider request continues the same lineage and
        # narrates the prior terminal reply exactly once.
        controller._agent_runtime_enabled = False
        gateway.two_chunk_mode = False
        gateway.reply = "FOLLOW-UP-ACK"
        gateway.payloads.clear()
        follow_up = await controller.submit_draft(
            f"follow after {marker}",
            session_id=session_id,
        )
        assert follow_up.accepted
        assert gateway.payloads
        payload_text = "\n".join(
            str(entry.get("content", "")) for entry in gateway.payloads[-1]
        )
        assert payload_text.count(final_content) == 1


@pytest.mark.asyncio
async def test_return_during_live_turn_keeps_fresh_view_reconciling(tmp_path):
    """A view mounted mid-turn keeps polling through terminalization.

    Regression for the live hidden-approval journey: Console was reopened
    while the turn waited for a decision, but the fresh screen never armed
    its transcript sync timer.  After approval, the app-owned store and DB
    completed while that mounted view stayed stuck on ``Thinking`` until a
    second navigation forced another initial reconciliation.
    """
    app, gateway = _build_navigation_app(tmp_path)

    async with app.run_test(size=(160, 48), notifications=True) as pilot:
        outgoing, controller, store, session_id, _conversation_id = await _seed_console(
            app, pilot, gateway
        )
        runtime = outgoing._console_runtime()
        marker = "RETURN-DURING-LIVE-TURN"
        final_content = f"{marker}-A{marker}-B"
        gateway.arm_two_chunks(marker)

        turn_id = _admit_console_turn_to_runtime(
            outgoing,
            "continue after returning to Console",
            session_id,
        )
        task = runtime._turn_custody[turn_id].task
        assert task is not None
        await asyncio.wait_for(gateway.first_chunk.wait(), timeout=3)

        await _navigate(
            app,
            pilot,
            "library",
            expect="LibraryScreen",
            allow_confirmation=False,
        )
        reopened = await _navigate(app, pilot, "chat", expect="ChatScreen")
        await pilot.pause()
        assert reopened is not outgoing
        assert runtime.view is reopened
        assert not task.done()

        gateway.release_second.set()
        await asyncio.wait_for(gateway.second_chunk.wait(), timeout=3)
        gateway.release_terminal.set()
        result = await asyncio.wait_for(task, timeout=8)
        assert result.accepted
        assert len(_unique_terminal_rows(store, session_id, final_content)) == 1

        for _ in range(30):
            if _rendered_text(reopened).count(final_content) == 1:
                break
            await pilot.pause(0.1)
        assert _rendered_text(reopened).count(final_content) == 1
        assert controller.in_flight_run_count() == 0


@pytest.mark.parametrize("phase", ["async_preparation", "provider_readiness"])
@pytest.mark.asyncio
async def test_ordinary_turn_survives_navigation_during_async_interior_phase(
    tmp_path,
    monkeypatch,
    phase,
):
    """Preparation and readiness remain owned by the detached runtime task."""
    app, gateway = _build_navigation_app(tmp_path)

    async with app.run_test(size=(160, 48), notifications=True) as pilot:
        chat, controller, store, session_id, conversation_id = await _seed_console(
            app, pilot, gateway
        )
        runtime = chat._console_runtime()
        entered = asyncio.Event()
        release = asyncio.Event()
        final_content = f"INTERIOR-{phase.upper()}"
        gateway.reply = final_content

        if phase == "provider_readiness":
            original_resolve = gateway.resolve_for_send

            async def hold_provider_readiness(selection):
                entered.set()
                await release.wait()
                return await original_resolve(selection)

            monkeypatch.setattr(gateway, "resolve_for_send", hold_provider_readiness)
        else:
            original_capture = controller._capture_rag_context

            async def hold_async_preparation(*args, **kwargs):
                entered.set()
                await release.wait()
                return await original_capture(*args, **kwargs)

            monkeypatch.setattr(
                controller,
                "_capture_rag_context",
                hold_async_preparation,
            )

        turn_id = _admit_console_turn_to_runtime(chat, phase, session_id)
        task = runtime._turn_custody[turn_id].task
        assert task is not None
        await asyncio.wait_for(entered.wait(), timeout=3)

        await _navigate(
            app,
            pilot,
            "library",
            expect="LibraryScreen",
            allow_confirmation=False,
        )
        assert runtime.view is None
        assert not task.done()
        release.set()
        result = await asyncio.wait_for(task, timeout=8)
        assert result.accepted
        await pilot.pause()

        await _assert_hidden_turn_reconciles_all_four_layers(
            app=app,
            pilot=pilot,
            controller=controller,
            store=store,
            runtime=runtime,
            gateway=gateway,
            session_id=session_id,
            conversation_id=conversation_id,
            final_content=final_content,
        )


@pytest.mark.parametrize(
    ("work_kind", "step_kind", "tool_name"),
    [
        ("tool", STEP_TOOL_RESULT, "fs_read"),
        ("image", STEP_TOOL_RESULT, "generate_image"),
        ("video", STEP_TOOL_RESULT, "generate_video"),
        ("delegated", STEP_SPAWN, ""),
    ],
)
@pytest.mark.asyncio
async def test_agent_interior_work_survives_real_navigation_with_origin_evidence(
    tmp_path,
    monkeypatch,
    work_kind,
    step_kind,
    tool_name,
):
    """Each worker origin crosses navigation with its own trace evidence."""
    app, gateway = _build_navigation_app(tmp_path)
    counter_state, counters = _install_detached_view_counters(monkeypatch, app)

    async with app.run_test(size=(160, 48), notifications=True) as pilot:
        chat, controller, store, session_id, conversation_id = await _seed_console(
            app, pilot, gateway
        )
        runtime = chat._console_runtime()
        marker = f"WORK-{work_kind.upper()}"
        final_content = f"{marker}-A{marker}-B"
        step = AgentStep(
            1,
            step_kind,
            summary=f"{work_kind} completed",
            tool_name=tool_name,
            result=f"{work_kind}-result",
        )
        bridge = _TwoChunkAgentBridge(
            store,
            marker,
            two_chunk_mode=True,
            steps=(step,),
        )
        app.app_config["console"]["agent_runtime"] = True
        monkeypatch.setattr(chat, "_ensure_console_agent_bridge", lambda: bridge)
        controller._agent_runtime_enabled = True
        controller._agent_bridge = bridge

        turn_id = _admit_console_turn_to_runtime(chat, work_kind, session_id)
        task = runtime._turn_custody[turn_id].task
        assert task is not None
        await _wait_thread_event(bridge.first_chunk)

        counter_state["outgoing"] = chat
        await _navigate(
            app,
            pilot,
            "library",
            expect="LibraryScreen",
            allow_confirmation=False,
        )
        counter_state["detached"] = True
        assert runtime.view is None
        bridge.release_second.set()
        await _wait_thread_event(bridge.second_chunk)
        assert counters.dom_queries == 0
        assert counters.transcript_polls == 0
        assert counters.shell_updates == 0
        bridge.release_terminal.set()

        result = await asyncio.wait_for(task, timeout=8)
        assert result.accepted
        assert bridge.last_outcome is not None
        assert [(item.kind, item.tool_name) for item in bridge.last_outcome.steps] == [
            (step_kind, tool_name)
        ]
        await pilot.pause()

        counter_state["detached"] = False
        await _assert_hidden_turn_reconciles_all_four_layers(
            app=app,
            pilot=pilot,
            controller=controller,
            store=store,
            runtime=runtime,
            gateway=gateway,
            session_id=session_id,
            conversation_id=conversation_id,
            final_content=final_content,
        )


@pytest.mark.asyncio
async def test_claimed_queued_continuation_drains_while_console_is_hidden(
    tmp_path,
    monkeypatch,
):
    """A real FIFO claim crosses the runtime handoff with no mounted view."""
    app, gateway = _build_navigation_app(tmp_path)

    async with app.run_test(size=(160, 48), notifications=True) as pilot:
        chat, controller, store, session_id, conversation_id = await _seed_console(
            app, pilot, gateway
        )
        runtime = chat._console_runtime()
        gateway.arm_two_chunks("QUEUE-INITIAL")
        first_turn_id = _admit_console_turn_to_runtime(
            chat,
            "first turn owns the chain",
            session_id,
        )
        first_task = runtime._turn_custody[first_turn_id].task
        assert first_task is not None
        await asyncio.wait_for(gateway.first_chunk.wait(), timeout=3)

        queue_snapshot = controller.prompt_queue_registry.snapshot(session_id)
        queued = controller.queue_prompt(
            session_id,
            text="queued continuation survives navigation",
            expected_revision=queue_snapshot.revision,
        )
        assert queued.applied
        assert queued.entry_id

        queued_claimed = asyncio.Event()
        release_queued_handoff = asyncio.Event()
        original_submit_queued = runtime._submit_queued_turn

        async def hold_claimed_queue_handoff(*args, **kwargs):
            queued_claimed.set()
            await release_queued_handoff.wait()
            return await original_submit_queued(*args, **kwargs)

        monkeypatch.setattr(
            runtime,
            "_submit_queued_turn",
            hold_claimed_queue_handoff,
        )
        controller.prompt_queue_coordinator.bind_runtime_submitter(
            runtime._submit_queued_turn
        )

        await _navigate(
            app,
            pilot,
            "library",
            expect="LibraryScreen",
            allow_confirmation=False,
        )
        assert runtime.view is None
        gateway.release_second.set()
        await asyncio.wait_for(gateway.second_chunk.wait(), timeout=3)
        gateway.release_terminal.set()
        await asyncio.wait_for(queued_claimed.wait(), timeout=5)
        claimed_snapshot = controller.prompt_queue_registry.snapshot(session_id)
        assert claimed_snapshot.claimed_count == 1
        assert claimed_snapshot.waiting_count == 0
        assert not first_task.done()

        queued_final = "QUEUED-CONTINUATION-FINAL"
        gateway.two_chunk_mode = False
        gateway.reply = queued_final
        release_queued_handoff.set()
        result = await asyncio.wait_for(first_task, timeout=10)
        assert result.accepted
        assert controller.prompt_queue_registry.snapshot(session_id).total_count == 0
        await pilot.pause()

        await _assert_hidden_turn_reconciles_all_four_layers(
            app=app,
            pilot=pilot,
            controller=controller,
            store=store,
            runtime=runtime,
            gateway=gateway,
            session_id=session_id,
            conversation_id=conversation_id,
            final_content=queued_final,
        )


@pytest.mark.asyncio
async def test_navigation_during_terminal_row_and_receipt_transaction_commits_once(
    tmp_path,
    monkeypatch,
):
    """Navigation may land after the live row mutates but before SQLite commits."""
    app, gateway = _build_navigation_app(tmp_path)

    async with app.run_test(size=(160, 48), notifications=True) as pilot:
        chat, controller, store, session_id, conversation_id = await _seed_console(
            app, pilot, gateway
        )
        runtime = chat._console_runtime()
        final_content = "TERMINAL-TRANSACTION-NAVIGATION"
        bridge = _WorkerTerminalBridge(store, final_content)
        app.app_config["console"]["agent_runtime"] = True
        monkeypatch.setattr(chat, "_ensure_console_agent_bridge", lambda: bridge)
        controller._agent_runtime_enabled = True
        controller._agent_bridge = bridge

        transaction_entered = threading.Event()
        release_transaction = threading.Event()
        persistence = store.persistence
        assert persistence is not None
        dispatch_repository = persistence.console_dispatch_repository
        assert dispatch_repository is not None
        original_settle = dispatch_repository.settle_with_assistant

        def hold_atomic_terminal_update(settlement):
            if (
                settlement.terminal_receipt_id
                and settlement.terminal_state == "complete"
                and settlement.content == final_content
            ):
                live_rows = _unique_terminal_rows(store, session_id, final_content)
                assert len(live_rows) == 1
                assert live_rows[0].status == "complete"
                transaction_entered.set()
                if not release_transaction.wait(10):
                    raise TimeoutError("test never released terminal persistence")
            return original_settle(settlement)

        monkeypatch.setattr(
            dispatch_repository,
            "settle_with_assistant",
            hold_atomic_terminal_update,
        )

        turn_id = _admit_console_turn_to_runtime(
            chat,
            "navigate during terminal commit",
            session_id,
        )
        task = runtime._turn_custody[turn_id].task
        assert task is not None
        await _wait_thread_event(transaction_entered)
        assert not task.done()
        assert [row[2] for row in _db_chain(app.chachanotes_db, conversation_id)].count(
            final_content
        ) == 0

        try:
            await _navigate(
                app,
                pilot,
                "library",
                expect="LibraryScreen",
                allow_confirmation=False,
            )
            assert runtime.view is None
        finally:
            release_transaction.set()

        result = await asyncio.wait_for(task, timeout=8)
        assert result.accepted
        assert bridge.calls == 1
        await pilot.pause()

        await _assert_hidden_turn_reconciles_all_four_layers(
            app=app,
            pilot=pilot,
            controller=controller,
            store=store,
            runtime=runtime,
            gateway=gateway,
            session_id=session_id,
            conversation_id=conversation_id,
            final_content=final_content,
            submit_follow_up=False,
        )


@pytest.mark.asyncio
async def test_stale_attachment_generation_cannot_detach_successor(tmp_path):
    """A delayed outgoing teardown cannot clear a newly attached view."""
    app, gateway = _build_navigation_app(tmp_path)

    async with app.run_test(size=(160, 48), notifications=True) as pilot:
        (
            outgoing,
            _controller,
            _store,
            _session_id,
            _conversation_id,
        ) = await _seed_console(app, pilot, gateway)
        runtime = outgoing._console_runtime()
        stale_generation = outgoing._console_runtime_attachment_generation

        await _navigate(
            app,
            pilot,
            "library",
            expect="LibraryScreen",
            allow_confirmation=False,
        )
        successor = await _navigate(app, pilot, "chat", expect="ChatScreen")
        successor_generation = successor._console_runtime_attachment_generation
        assert successor is not outgoing
        assert successor_generation != stale_generation
        assert runtime.view is successor

        detached = await runtime.leave_console(outgoing, stale_generation)
        assert detached is False
        assert runtime.view is successor
        assert runtime._attached_generation == successor_generation


@pytest.mark.asyncio
async def test_hidden_completion_and_background_approval_reconcile_by_session(
    tmp_path,
):
    """Two concurrent sessions keep outcomes, cards, and attention isolated."""
    app, gateway = _build_navigation_app(tmp_path)
    approval_thread = None
    approval_round_id = None

    async with app.run_test(size=(160, 48), notifications=True) as pilot:
        chat, controller, store, turn_session_id, conversation_id = await _seed_console(
            app, pilot, gateway
        )
        runtime = chat._console_runtime()
        approval_session = controller.new_session(title="Approval session")
        controller.switch_session(turn_session_id)
        assert store.active_session_id == turn_session_id

        gateway.arm_two_chunks("SESSION-A")
        final_content = "SESSION-A-ASESSION-A-B"
        turn_id = _admit_console_turn_to_runtime(
            chat,
            "complete while another session waits",
            turn_session_id,
        )
        turn_task = runtime._turn_custody[turn_id].task
        assert turn_task is not None
        await asyncio.wait_for(gateway.first_chunk.wait(), timeout=3)

        await _navigate(
            app,
            pilot,
            "library",
            expect="LibraryScreen",
            allow_confirmation=False,
        )
        assert runtime.view is None

        approval_thread, _approval_box = _arm(
            controller,
            approval_session.id,
            call=_risk_row(),
        )
        assert await _wait_for_round(controller, approval_session.id)
        approval_round_id = _armed_round_ids(controller, approval_session.id)[0]
        assert runtime.console_needs_attention is True
        assert app.console_needs_attention is True

        gateway.release_second.set()
        await asyncio.wait_for(gateway.second_chunk.wait(), timeout=3)
        gateway.release_terminal.set()
        result = await asyncio.wait_for(turn_task, timeout=8)
        assert result.accepted

        terminal_rows = _unique_terminal_rows(
            store,
            turn_session_id,
            final_content,
        )
        assert len(terminal_rows) == 1
        receipt_id = terminal_receipt_id_for_message(terminal_rows[0])
        assert receipt_id
        marks_service = runtime._console_local_marks_service()
        assert (conversation_id, receipt_id) in (
            marks_service.list_console_unseen_marks()
        )

        reopened = await _navigate(app, pilot, "chat", expect="ChatScreen")
        await pilot.pause()
        assert store.active_session_id == turn_session_id
        assert _rendered_text(reopened).count(final_content) == 1
        assert not any(card.display for card in reopened.query("#chat-approval-card"))
        assert (conversation_id, receipt_id) not in (
            marks_service.list_console_unseen_marks()
        )
        # Visiting A acknowledges only A's terminal receipt. B's unresolved
        # review continues to own the app-wide marker.
        assert controller.has_pending_approval_round(approval_session.id)
        assert runtime.console_needs_attention is True
        assert app.console_needs_attention is True

        controller.switch_session(approval_session.id)
        await reopened._sync_native_console_chat_ui()
        await pilot.pause()
        assert final_content not in _rendered_text(reopened)
        assert any(card.display for card in reopened.query("#chat-approval-card"))

        controller.resolve_pending_approval(
            {"builtin__write_file": "deny"},
            round_id=approval_round_id,
        )
        await asyncio.to_thread(approval_thread.join, 5)
        assert not approval_thread.is_alive()
        approval_round_id = None
        assert not controller.has_pending_approval_round(approval_session.id)
        assert runtime.recompute_console_attention(force_projection=True) is False
        assert app.console_needs_attention is False

    if approval_thread is not None and approval_thread.is_alive():
        if approval_round_id is not None:
            controller.resolve_pending_approval(
                {"builtin__write_file": "deny"},
                round_id=approval_round_id,
            )
        await asyncio.to_thread(approval_thread.join, 5)
