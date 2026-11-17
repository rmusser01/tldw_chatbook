"""The actual contained backend, separate from parent UI qualification."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import time

import pytest

from Tests.private_profile import private_profile_test


class _ObservedStderr:
    """Observe only bytes the original diagnostic monitor actually consumes."""

    def __init__(self, original, tail):
        self.original, self.tail = original, tail

    def read(self, count):
        value = self.original.read(count)
        self.tail.extend(value)
        del self.tail[:-131072]
        return value

    def close(self):
        return self.original.close()

    @property
    def closed(self):
        return self.original.closed


_ACCEPTED_DRAFT = "Original accepted process input"
_ACCEPTED_REPLY = "ORIGINAL-ACCEPTED-PROCESS-REPLY"


@pytest.fixture
def accepted_process_provider(request):
    """Join a real loopback endpoint using the original HTTP/SSE conventions."""
    from Tests.private_profile import is_private_profile_child

    # The outer private-profile launcher never executes this test body.
    if not is_private_profile_child(request):
        yield None
        return
    from Tests.Chat.test_console_refused_stream_retry import (
        _Handler,
        _OpenAICompatibleServer,
    )

    class Handler(_Handler):
        def setup(self):
            super().setup()
            self.connection.settimeout(2)

        def do_GET(self):  # noqa: N802 - original http.server contract
            self.server.requests.append(("GET", self.path))
            super().do_GET()

        def do_POST(self):  # noqa: N802 - original http.server contract
            server = self.server
            server.post_received_at.append(time.monotonic())
            server.requests.append(("POST", self.path))
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            server.payloads.append(payload)
            server.stream_flags.append(payload.get("stream") is True)
            chunks = [
                {"choices": [{"index": 0, "delta": {"role": "assistant"}}]},
                {"choices": [{"index": 0, "delta": {"content": _ACCEPTED_REPLY}}]},
                {
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 21,
                        "completion_tokens": 5,
                        "total_tokens": 26,
                    },
                },
            ]
            body = (
                "".join(
                    "data: "
                    + json.dumps(
                        {
                            "id": "chatcmpl-task62-accepted",
                            "object": "chat.completion.chunk",
                            "created": 1,
                            "model": payload["model"],
                            **chunk,
                        }
                    )
                    + "\n\n"
                    for chunk in chunks
                )
                + "data: [DONE]\n\n"
            )
            self._send(200, "text/event-stream", body.encode())
            server.reply_sent_at.append(time.monotonic())

    class Server(_OpenAICompatibleServer):
        daemon_threads = False
        block_on_close = True

        def __init__(self):
            super().__init__()
            self.RequestHandlerClass = Handler
            self.requests = []
            self.post_received_at = []
            self.reply_sent_at = []

    server = Server()
    thread = threading.Thread(target=server.serve_forever, daemon=False)
    started = False
    try:
        thread.start()
        started = True
        yield server
    finally:
        try:
            if started and thread.is_alive():
                server.shutdown()
        finally:
            try:
                server.server_close()  # Joins the non-daemon request owners too.
            finally:
                if started:
                    thread.join(timeout=2)
        request.node.user_properties.append(
            ("accepted_provider_listener_joined", not thread.is_alive())
        )
        assert not thread.is_alive()


@private_profile_test
@pytest.mark.asyncio
async def test_original_backend_runs_in_one_contained_process_and_retires(request):
    from Tests.Chat.test_console_execution_profile import _settled_profile
    from tldw_chatbook.Chat.console_execution_process import (
        ConsoleExecutionClient,
        ConsoleProcessProfile,
    )

    path, directory, *_ = _settled_profile()
    client = ConsoleExecutionClient(None)
    tail = bytearray()
    from Tests.Chat.test_console_control_trace import _observe
    import pytest_timeout

    def created(frame):
        caller = frame.f_back
        if (
            caller.f_code is not ConsoleExecutionClient._launch.__code__
            or caller.f_locals.get("self") is not client
        ):
            return
        process = frame.f_locals["self"]
        assert type(process) is subprocess.Popen and process.stderr is not None
        process.stderr = _ObservedStderr(process.stderr, tail)

    closed = False
    try:
        with _observe(
            subprocess.Popen.__init__.__code__, sys.monitoring.events.PY_RETURN, created
        ):
            assert not pytest_timeout.is_debugging()
            assert float(request.config.getini("timeout")) == 300
            request.node.user_properties.append(("observer_watchdog", "300/nondebug"))
            await client.start(ConsoleProcessProfile(str(path), str(directory)))
        process = client._process
        assert process is not None and process.pid != os.getpid()
        assert client._ready.done() and client._ready.exception() is None
        assert client._tree is not None
        opened = client.submit(
            "OpenSession",
            {
                "title": "Contained original session",
                "workspace_id": None,
                "conversation_id": None,
            },
        )
        await asyncio.shield(opened.received)
        session = await asyncio.shield(opened.applied)
        assert session["session_id"] and session["incarnation_id"]
        assert await asyncio.shield(opened.retired) is True
        permission = client.submit(
            "PermissionAction",
            {
                "operation": {"kind": "ReadKillSwitch"},
            },
        )
        await asyncio.shield(permission.received)
        result = await asyncio.shield(permission.applied)
        assert result["kind"] == "ReadKillSwitch"
        assert type(result["result"]) is bool
        assert await asyncio.shield(permission.retired) is True
        assert client._process is process
        assert not any(
            name in vars(client)
            for name in (
                "_chat_store",
                "_chat_controller",
                "domain_services",
            )
        )
    except BaseException as error:
        request.node.user_properties.append(("first_body_error", type(error).__name__))
        raise
    finally:
        # Match the original contained-process maintenance exit allowance in
        # app_shutdown.py; the journey latency target is not a teardown bound.
        try:
            closed = await client.shutdown(deadline=time.monotonic() + 30)
        finally:
            text = tail.decode("utf-8", errors="replace")
            request.node.user_properties.append(("actual_backend_stderr_tail", text))
            if text and not closed:
                print(
                    "Actual backend diagnostic tail (bounded):", text, file=sys.stderr
                )
    assert closed is True
    assert client._disposed and client.physically_closed
    assert process.returncode == 0
    assert all(
        stream.closed for stream in (process.stdin, process.stdout, process.stderr)
    )
    assert not client._monitor.is_alive()
    assert all(not thread.is_alive() for thread in client._channel._started)
    request.node.user_properties.append(("actual_backend_pid", process.pid))
    request.node.user_properties.append(("actual_parent_pid", os.getpid()))
    request.node.user_properties.append(("actual_process_exit", process.returncode))


@private_profile_test
@pytest.mark.asyncio
async def test_actual_process_projection_reference_and_original_pre_ack_cancel(
    request, accepted_process_provider
):
    from Tests.Chat.test_console_execution_profile import _settled_profile
    from Tests.Chat.test_console_first_send_atomicity import _until
    from Tests.Chat.test_console_control_trace import _observe
    from tldw_chatbook import config
    from tldw_chatbook.Chat.console_chat_controller import (
        build_console_provider_selection_from_settings,
    )
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleMessageRole,
        ConsoleRunStatus,
        ConsoleWorkspaceContext,
    )
    from tldw_chatbook.Chat.console_execution_process import (
        ConsoleExecutionClient,
        ConsoleProcessProfile,
    )
    from tldw_chatbook.Chat.console_execution_projection_values import _settings
    from tldw_chatbook.Chat.console_session_settings import (
        default_console_session_settings,
    )
    from tldw_chatbook.Workspaces.models import DEFAULT_WORKSPACE_ID
    import pytest_timeout

    assert config.save_settings_to_cli_config(
        {
            "chat_defaults": {"provider": "openai", "model": "gpt-4o-mini"},
            "api_settings.openai": {
                "api_key": "task62-local-readiness-only",
                "api_base": accepted_process_provider.base_url,
            },
        }
    )
    path, directory, *_ = _settled_profile()
    configuration = config.load_settings()
    settings = default_console_session_settings(configuration)
    provider = build_console_provider_selection_from_settings(
        settings,
        app_config=configuration,
        workspace_context=ConsoleWorkspaceContext(
            active_workspace_id=DEFAULT_WORKSPACE_ID
        ),
    )
    assert provider.base_url == accepted_process_provider.base_url
    selected = {
        name: getattr(provider, name)
        for name in (
            "provider",
            "base_url",
            "base_url_is_pinned",
            "execution_provider",
            "configured_endpoint_fallback_allowed",
            "explicit_model",
            "configured_model",
            "temperature",
            "top_p",
            "min_p",
            "top_k",
            "max_tokens",
            "seed",
            "presence_penalty",
            "frequency_penalty",
            "reasoning_effort",
            "reasoning_summary",
            "verbosity",
            "thinking_effort",
            "thinking_budget_tokens",
            "streaming",
            "system_prompt",
        )
    }
    selected["endpoint_provenance"] = provider.endpoint_provenance.value
    selected["workspace_context"] = {
        "active_workspace_id": DEFAULT_WORKSPACE_ID,
        "staged_sources": [],
        "active_run_id": None,
        "handoff_id": None,
    }
    client = ConsoleExecutionClient(None)
    tail = bytearray()

    def created(frame):
        caller = frame.f_back
        if (
            caller.f_code is ConsoleExecutionClient._launch.__code__
            and caller.f_locals.get("self") is client
        ):
            process = frame.f_locals["self"]
            assert type(process) is subprocess.Popen and process.stderr is not None
            process.stderr = _ObservedStderr(process.stderr, tail)

    received = None
    accepted_fingerprint = None
    closed = False
    body_failed = False
    try:
        with _observe(
            subprocess.Popen.__init__.__code__, sys.monitoring.events.PY_RETURN, created
        ):
            assert not pytest_timeout.is_debugging()
            assert float(request.config.getini("timeout")) == 300
            await client.start(ConsoleProcessProfile(str(path), str(directory)))
        opened = client.submit(
            "OpenSession",
            {
                "title": "Original pre-ACK process turn",
                "workspace_id": None,
                "conversation_id": None,
            },
        )
        session = await asyncio.shield(opened.applied)
        assert await asyncio.shield(opened.retired) is True
        sid, incarnation = session["session_id"], session["incarnation_id"]
        first = await client.read_projection(sid, incarnation)
        second = await client.read_projection(sid, incarnation)
        assert second.messages is first.messages
        assert not client._requests
        request.node.user_properties.append(("projection_reference_reused", True))
        permission = client.submit(
            "PermissionAction", {"operation": {"kind": "SetKillSwitch", "value": False}}
        )
        await asyncio.shield(permission.applied)
        assert await asyncio.shield(permission.retired) is True
        settings_value = _settings(settings, False)
        updated = client.submit(
            "UpdateInputs",
            {
                "draft": "Original captured process input",
                "authored_token": [1, 1],
                "expected_settings_revision": 0,
                "expected_settings": settings_value,
                "expected_attachment_ids": [],
                "remove_attachment_ids": [],
                "prefill": None,
            },
            session_id=sid,
            incarnation_id=incarnation,
        )
        update = await asyncio.shield(updated.applied)
        assert await asyncio.shield(updated.retired) is True
        payload = {
            "turn_id": "process-pre-ack",
            "draft_revision": update["draft_revision"],
            "expected_settings_revision": 0,
            "expected_attachment_revision": 0,
            "expected_prefill_revision": 0,
            "selection": {
                "provider_selection": selected,
                "presentation_context": None,
                "rag_defaults": None,
                "tool_configuration": None,
                "skill_workspace_id": DEFAULT_WORKSPACE_ID,
                "project_bindings_eligible": False,
                "agent_runtime_enabled": True,
            },
            "pressed_inputs": {
                "draft": "Original captured process input",
                "authored_token": [1, 1],
                "settings": settings_value,
                "settings_revision": 0,
                "attachment_revision": 0,
                "attachment_ids": [],
                "attachments": [],
                "one_shot_prefill": None,
                "prefill_revision": 0,
                "session_id": sid,
                "incarnation_id": incarnation,
            },
        }
        started = time.monotonic()
        received = client.submit(
            "ReceiveTurn", payload, session_id=sid, incarnation_id=incarnation
        )
        await asyncio.shield(received.received)
        assert (await asyncio.shield(received.applied))["turn_id"] == "process-pre-ack"
        assert await _until(
            lambda: any(event.kind == "commit_planned" for event in received.events)
            or received.retired.done()
        )
        plans = [event for event in received.events if event.kind == "commit_planned"]
        request.node.user_properties.append(
            (
                "receive_to_commit_planned_seconds"
                if plans
                else "receive_to_terminal_failure_seconds",
                time.monotonic() - started,
            )
        )
        request.node.user_properties.append(
            ("received_event_kinds", [event.kind for event in received.events])
        )
        assert len(plans) == 1, [
            (event.kind, dict(event.payload)) for event in received.events
        ]
        plan = plans[0]
        assert plan.payload["received_request_id"] == received.identity.request_id
        assert (
            plan.payload["turn_id"] == "process-pre-ack"
            and plan.payload["ordinal"] == 1
        )
        assert not received.retired.done() and not received.potentially_committed
        assert not accepted_process_provider.requests
        cancelled = client.submit(
            "CancelTurn",
            {"turn_id": "process-pre-ack"},
            session_id=sid,
            incarnation_id=incarnation,
        )
        assert (await asyncio.shield(cancelled.applied))["applied"] is True
        assert await asyncio.shield(received.retired) is True
        assert await asyncio.shield(cancelled.retired) is True
        assert not client._plans and not client._requests
        request.node.user_properties.append(("actual_pre_ack_cancel_retired", True))
        assert not accepted_process_provider.requests

        # Integration prerequisite: the same retired projection publication
        # provides the original row plus detached current input sidecar.
        fresh = await client.read_projection(sid, incarnation)
        row = client.projection_view.session(sid)
        current_inputs = client.pressed_inputs
        assert fresh.session_id == row.id == current_inputs["session_id"] == sid
        assert (
            fresh.incarnation_id
            == row.incarnation_id
            == current_inputs["incarnation_id"]
            == incarnation
        )
        assert row.settings is not None
        assert current_inputs["settings"] == _settings(row.settings, False)
        assert current_inputs["settings_revision"] == row.settings_revision
        # This is a new parent-authored identity, not a guessed backend revision.
        authored_token = [2, 1]
        next_update = client.submit(
            "UpdateInputs",
            {
                "draft": _ACCEPTED_DRAFT,
                "authored_token": authored_token,
                "expected_settings_revision": row.settings_revision,
                "expected_settings": current_inputs["settings"],
                "expected_attachment_ids": current_inputs["attachment_ids"],
                "remove_attachment_ids": [],
                "prefill": None,
            },
            session_id=sid,
            incarnation_id=incarnation,
        )
        await asyncio.shield(next_update.received)
        next_draft = await asyncio.shield(next_update.applied)
        assert await asyncio.shield(next_update.retired) is True
        await client.read_projection(sid, incarnation)
        row = client.projection_view.session(sid)
        pressed = client.pressed_inputs
        assert row.draft_revision == next_draft["draft_revision"]
        assert row.draft == pressed["draft"] == _ACCEPTED_DRAFT
        assert pressed["authored_token"] == authored_token
        assert pressed["session_id"] == sid and pressed["incarnation_id"] == incarnation
        assert pressed["settings"] == _settings(row.settings, False)
        assert pressed["settings_revision"] == row.settings_revision
        next_provider = build_console_provider_selection_from_settings(
            row.settings,
            app_config=configuration,
            workspace_context=ConsoleWorkspaceContext(
                active_workspace_id=row.workspace_id
            ),
        )
        assert next_provider.base_url == accepted_process_provider.base_url
        next_selected = {
            name: getattr(next_provider, name)
            for name in selected
            if name not in {"endpoint_provenance", "workspace_context"}
        }
        next_selected["endpoint_provenance"] = next_provider.endpoint_provenance.value
        next_selected["workspace_context"] = {
            **selected["workspace_context"],
            "active_workspace_id": row.workspace_id,
        }
        accepted_started = time.monotonic()
        accepted = client.submit(
            "ReceiveTurn",
            {
                "turn_id": "process-accepted",
                "draft_revision": row.draft_revision,
                "expected_settings_revision": pressed["settings_revision"],
                "expected_attachment_revision": pressed["attachment_revision"],
                "expected_prefill_revision": pressed["prefill_revision"],
                "selection": {
                    **payload["selection"],
                    "provider_selection": next_selected,
                    "skill_workspace_id": row.workspace_id,
                },
                "pressed_inputs": pressed,
            },
            session_id=sid,
            incarnation_id=incarnation,
        )
        await asyncio.shield(accepted.received)
        assert (await asyncio.shield(accepted.applied))["turn_id"] == "process-accepted"
        assert await _until(
            lambda: any(event.kind == "commit_planned" for event in accepted.events)
            or accepted.retired.done()
        )
        accepted_plans = [
            event for event in accepted.events if event.kind == "commit_planned"
        ]
        request.node.user_properties.append(
            (
                "accepted_receive_to_commit_planned_seconds"
                if accepted_plans
                else "accepted_receive_to_terminal_failure_seconds",
                time.monotonic() - accepted_started,
            )
        )
        assert len(accepted_plans) == 1, [
            (event.kind, dict(event.payload)) for event in accepted.events
        ]
        accepted_plan = accepted_plans[0]
        assert (
            accepted_plan.payload["received_request_id"] == accepted.identity.request_id
        )
        assert accepted_plan.payload["turn_id"] == "process-accepted"
        assert not accepted.retired.done() and not accepted.potentially_committed
        assert not accepted_process_provider.requests
        from tldw_chatbook.Chat.console_execution_protocol import encode_message

        acknowledged_plan = encode_message(accepted_plan)["payload"]
        ack = client.submit(
            "CommitPlanAck",
            {
                name: acknowledged_plan[name]
                for name in ("received_request_id", "turn_id", "ordinal", "fingerprint")
            },
            session_id=sid,
            incarnation_id=incarnation,
        )
        assert accepted.potentially_committed
        await asyncio.shield(ack.received)
        assert (await asyncio.shield(ack.applied))["applied"] is True
        assert await asyncio.shield(ack.retired) is True
        assert await _until(accepted.retired.done)
        accepted_retired = await asyncio.shield(accepted.retired)
        accepted_finished = time.monotonic()
        if len(accepted_process_provider.post_received_at) == 1:
            request.node.user_properties.append(
                ("accepted_to_provider_post_seconds",
                 accepted_process_provider.post_received_at[0] - accepted_started)
            )
        if len(accepted_process_provider.reply_sent_at) == 1:
            request.node.user_properties.append(
                ("provider_reply_to_retirement_seconds",
                 accepted_finished - accepted_process_provider.reply_sent_at[0])
            )
        assert accepted_retired is True
        request.node.user_properties.append(
            (
                "accepted_receive_to_terminal_seconds",
                accepted_finished - accepted_started,
            )
        )
        assert not any(event.kind == "failed" for event in accepted.events), [
            (event.kind, dict(event.payload)) for event in accepted.events
        ]
        assert accepted_process_provider.requests == [("POST", "/v1/chat/completions")]
        assert accepted_process_provider.stream_flags == [True]
        assert len(accepted_process_provider.payloads) == 1
        sent = accepted_process_provider.payloads[0]
        assert any(
            message.get("role") == "user" and message.get("content") == _ACCEPTED_DRAFT
            for message in sent["messages"]
        )
        terminal = await client.read_projection(sid, incarnation)
        assert terminal.run_state.status is ConsoleRunStatus.COMPLETED
        accepted_fingerprint = dict(accepted_plan.payload["fingerprint"])
        messages = terminal.messages
        assert [
            (message.role, message.content, message.status) for message in messages
        ] == [
            (ConsoleMessageRole.USER, _ACCEPTED_DRAFT, "complete"),
            (ConsoleMessageRole.ASSISTANT, _ACCEPTED_REPLY, "complete"),
        ]
        assert messages[1].assistant_generation_state == "complete"
        assert [message.persisted_message_id for message in messages] == [
            accepted_fingerprint["user_message_id"],
            accepted_fingerprint["assistant_message_id"],
        ]
        saved_row = client.projection_view.session(sid)
        assert (
            saved_row.persisted_conversation_id
            == accepted_fingerprint["conversation_id"]
        )
        assert not client._plans and not client._requests
        request.node.user_properties.append(("actual_accepted_turn_retired", True))
    except BaseException as error:
        body_failed = True
        request.node.user_properties.append(("first_body_error", type(error).__name__))
        raise
    finally:
        try:
            closed = await client.shutdown(deadline=time.monotonic() + 30)
        finally:
            text = tail.decode("utf-8", errors="replace")
            request.node.user_properties.append(("actual_backend_stderr_tail", text))
            if text and (body_failed or not closed):
                print(
                    "Actual backend diagnostic tail (bounded):", text, file=sys.stderr
                )
    assert closed and client.physically_closed
    assert client._process.returncode == 0
    assert accepted_fingerprint is not None
    from tldw_chatbook.app_services import open_chachanotes_db

    # Reopen through the original repository after complete child retirement;
    # this thread owns and closes the actual verification connection.
    db = open_chachanotes_db(
        config.get_chachanotes_db_path(), client_id=config.CLI_APP_CLIENT_ID
    )
    persistence_failed = False
    try:
        conversation_id = accepted_fingerprint["conversation_id"]
        conversation = db.get_conversation_by_id(conversation_id)
        assert conversation is not None and conversation["id"] == conversation_id
        rows = db.get_messages_for_conversation(conversation_id, limit=3)
        assert len(rows) == 2
        by_id = {row["id"]: row for row in rows}
        user = by_id[accepted_fingerprint["user_message_id"]]
        assistant = by_id[accepted_fingerprint["assistant_message_id"]]
        assert (user["role"], user["content"]) == ("user", _ACCEPTED_DRAFT)
        assert (assistant["role"], assistant["content"]) == (
            "assistant",
            _ACCEPTED_REPLY,
        )
        assert assistant["parent_message_id"] == user["id"]
        assert assistant["assistant_generation_state"] == "complete"
        assert (
            user["conversation_id"] == assistant["conversation_id"] == conversation_id
        )
    except BaseException as error:
        persistence_failed = True
        request.node.user_properties.append(("first_body_error", type(error).__name__))
        raise
    finally:
        try:
            db.close()
        except BaseException as error:
            request.node.user_properties.append(
                ("persistence_cleanup_error", type(error).__name__)
            )
            if not persistence_failed:
                raise
    request.node.user_properties.append(("actual_accepted_turn_persisted", True))
