"""The actual contained backend, separate from parent UI qualification."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
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
        del self.tail[:-32768]
        return value

    def close(self):
        return self.original.close()

    @property
    def closed(self):
        return self.original.closed


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
async def test_actual_process_projection_reference_and_original_pre_ack_cancel(request):
    from Tests.Chat.test_console_execution_profile import _settled_profile
    from Tests.Chat.test_console_first_send_atomicity import _until
    from Tests.Chat.test_console_control_trace import _observe
    from tldw_chatbook import config
    from tldw_chatbook.Chat.console_chat_controller import (
        build_console_provider_selection_from_settings,
    )
    from tldw_chatbook.Chat.console_chat_models import ConsoleWorkspaceContext
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
                "api_base": "http://127.0.0.1:1/v1",
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
