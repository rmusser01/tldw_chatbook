"""Project context must not replace the saved owner of a captured Console send."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager, closing
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleEgressClass,
    ConsoleResolvedDestination,
)
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    ConsoleProviderResolution,
)
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_trace_native_reader import ConsoleTraceNativeReader
from tldw_chatbook.Chat.console_trace_runtime import ConsoleTraceBoundaryFactory
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService
from tldw_chatbook.Workspaces.models import WorkspaceRuntimeBinding

GUIDANCE = "PROJECT_TRACE_GUIDANCE_31976: run the project tests."
CREDENTIAL = "sk-" + "a" * 48
CONTACT = "elise@example.test"
RUNTIME_CREDENTIAL = "local-provider-password-31976"


@asynccontextmanager
async def _project_console(
    tmp_path,
    monkeypatch,
    *,
    source_name="AGENTS.md",
    capture_enabled=True,
    pii_redaction_enabled=False,
    response_mode="complete",
    tools=False,
):
    """Use real new-session/workspace defaults; replace only provider HTTP."""
    with (
        closing(CharactersRAGDB(tmp_path / "chat.sqlite", "project-trace")) as db,
        closing(
            AgentRunsDB(tmp_path / "runs.sqlite", client_id="project-trace")
        ) as runs,
        closing(
            WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="project-trace")
        ) as workspace_db,
    ):
        root = tmp_path / "project"
        root.mkdir()
        (root / source_name).write_text(
            f"{GUIDANCE}\nExample credential: {CREDENTIAL}\nContact: {CONTACT}"
            f"\nRuntime credential: {RUNTIME_CREDENTIAL}"
        )
        registry = LocalWorkspaceRegistryService(workspace_db)
        registry.create_workspace(workspace_id="project-trace", name="Project")
        registry.save_runtime_binding(
            WorkspaceRuntimeBinding(
                workspace_id="project-trace",
                binding_id="project-folder",
                binding_kind="local-filesystem",
                label="Project",
                locator=str(root),
                status="ready",
                metadata={"access": "rw"},
            )
        )
        factory = ConsoleTraceBoundaryFactory(db)
        reservation_errors = []
        http_payloads = []

        def boundary(request, resolved, route):
            try:
                return factory(request, resolved, route)
            except Exception as exc:
                reservation_errors.append(str(exc))
                raise

        def restart_factory():
            nonlocal factory
            factory = ConsoleTraceBoundaryFactory(db)

        def respond(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v1/chat/completions"
            payload = json.loads(request.content)
            http_payloads.append(payload)
            if payload.get("stream"):
                body = (
                    ""
                    if response_mode == "fallback"
                    else 'data: {"choices":[{"delta":{"content":"Fixture reply"}}]}\n\n'
                    "data: [DONE]\n\n"
                )
                return httpx.Response(
                    200, text=body, headers={"content-type": "text/event-stream"}
                )
            content = (
                '```tool_call\n{"name": "calculator", "arguments": {"expression": "6*7"}}\n```'
                if tools and len(http_payloads) == 1
                else "Fixture reply"
            )
            return httpx.Response(
                200, json={"choices": [{"message": {"content": content}}]}
            )

        resolution = ConsoleProviderResolution(
            ready=True,
            provider="llama_cpp",
            execution_key="llama_cpp",
            model="qwen3.7-27b",
            base_url="http://localhost:8080",
            api_key=RUNTIME_CREDENTIAL,
            streaming=response_mode != "complete",
            resolved_destination=ConsoleResolvedDestination(
                provider="llama_cpp",
                model="qwen3.7-27b",
                endpoint_identity="http://localhost:8080",
                egress_class=ConsoleEgressClass.ON_DEVICE,
            ),
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            gateway = ConsoleProviderGateway(
                http_client=client, trace_call_boundary_factory=boundary
            )

            async def resolve(_selection):
                return resolution

            monkeypatch.setattr(gateway, "resolve_for_send", resolve)
            store = ConsoleChatStore(
                persistence=ChatPersistenceService(db, workspace_registry=registry)
            )
            # Use the real new-session defaults: the old helper disabled AGENTS.
            session = store.create_session(
                workspace_id="project-trace",
                settings=ConsoleSessionSettings(
                    provider="llama_cpp", model="qwen3.7-27b"
                ),
            )
            bridge = ConsoleAgentBridge(
                agent_runs_db=runs, store=store, provider_gateway=gateway
            )
            controller = ConsoleChatController(
                store=store,
                provider_gateway=gateway,
                agent_runtime_enabled=True,
                agent_bridge=bridge,
                confirm_project_instruction_dispatch=lambda _notice: "proceed",
            )
            controller.app = SimpleNamespace(workspace_registry_service=registry)
            controller.set_next_trace_privacy(
                session.id,
                capture_enabled=capture_enabled,
                pii_redaction_enabled=pii_redaction_enabled,
                expected_policy_revision=controller.capture_policy_snapshot(
                    session.id
                ).policy_revision,
            )
            try:
                yield SimpleNamespace(
                    db=db,
                    store=store,
                    controller=controller,
                    session=session,
                    factory=factory,
                    http_payloads=http_payloads,
                    reservation_errors=reservation_errors,
                    project_file=root / source_name,
                    restart_factory=restart_factory,
                )
            finally:
                await gateway.aclose()


@pytest.mark.parametrize("source_name", ["AGENTS.md", "AGENTS.override.md"])
@pytest.mark.parametrize("capture_enabled", [False, True])
@pytest.mark.parametrize("pii_redaction_enabled", [False, True])
@pytest.mark.parametrize("response_mode", ["stream", "complete", "fallback"])
async def test_fresh_project_send_preserves_turn_and_captured_context(
    tmp_path,
    monkeypatch,
    source_name,
    capture_enabled,
    response_mode,
    pii_redaction_enabled,
):
    async with _project_console(
        tmp_path,
        monkeypatch,
        source_name=source_name,
        capture_enabled=capture_enabled,
        response_mode=response_mode,
        pii_redaction_enabled=pii_redaction_enabled,
    ) as app:
        result = await app.controller.submit_draft("Hello", session_id=app.session.id)

        assert result.accepted
        assert app.controller.run_state.status.value == "completed", (
            app.reservation_errors
        )
        assert len(app.http_payloads) == (2 if response_mode == "fallback" else 1)
        for payload in app.http_payloads:
            rows = payload["messages"]
            assert rows[-2]["content"] == "Hello"
            assert GUIDANCE in rows[-1]["content"]
            assert CREDENTIAL in rows[-1]["content"]
            assert RUNTIME_CREDENTIAL in rows[-1]["content"]
        messages = app.store.messages_for_session(app.session.id)
        assert any(message.content == "Fixture reply" for message in messages)
        assert all(GUIDANCE not in message.content for message in messages)
        user = next(message for message in messages if message.content == "Hello")
        captures = ConsoleTraceNativeReader(app.db).read_calls(
            user.persisted_message_id
        )
        with app.db.transaction() as cursor:
            calls = app.factory.repository.read_conversation_call_lineage(
                cursor, app.session.persisted_conversation_id
            )
        if capture_enabled:
            assert len(captures) == len(calls) == len(app.http_payloads)
            assert all(call.turn_id == user.persisted_message_id for call in calls)
            for call in captures:
                rows = call.capture.request["messages_payload"]
                assert rows[-2]["content"] == "Hello"
                assert GUIDANCE in rows[-1]["content"]
                assert RUNTIME_CREDENTIAL not in json.dumps(call.capture.request)
                assert CREDENTIAL not in json.dumps(call.capture.request)
                assert (CONTACT in rows[-1]["content"]) is not pii_redaction_enabled
        else:
            assert captures == ()
            assert not calls


@pytest.mark.parametrize("prior_response", ["ordinary", "tools", "fallback"])
@pytest.mark.parametrize("context_change", ["same", "changed", "disabled"])
@pytest.mark.parametrize("cold_factory", [False, True])
async def test_consecutive_project_sends_preserve_each_request(
    tmp_path,
    monkeypatch,
    prior_response,
    context_change,
    cold_factory,
):
    tools = prior_response == "tools"
    fallback = prior_response == "fallback"
    async with _project_console(
        tmp_path,
        monkeypatch,
        tools=tools,
        response_mode="fallback" if fallback else "complete",
    ) as app:
        first = await app.controller.submit_draft("Hello", session_id=app.session.id)
        assert first.accepted
        assert app.controller.run_state.status.value == "completed", (
            app.reservation_errors
        )
        first_user = next(
            row
            for row in app.store.messages_for_session(app.session.id)
            if row.content == "Hello"
        )
        reader = ConsoleTraceNativeReader(app.db)
        original = reader.read_calls(first_user.persisted_message_id)
        assert len(original) == (2 if tools or fallback else 1)
        for call in original:
            assert GUIDANCE in json.dumps(call.capture.request)
            assert CREDENTIAL not in json.dumps(call.capture.request)

        if context_change == "changed":
            app.project_file.write_text("UPDATED_PROJECT_GUIDANCE_31976")
        elif context_change == "disabled":
            app.store.set_session_project_instruction_state(
                app.session.id,
                replace(
                    app.session.project_instruction_state,
                    project_instructions_enabled=False,
                ),
            )

        for text in ("Next question", "One more question"):
            if cold_factory:
                app.restart_factory()
            before_http = len(app.http_payloads)
            result = await app.controller.submit_draft(text, session_id=app.session.id)
            assert result.accepted
            assert app.controller.run_state.status.value == "completed", (
                app.reservation_errors
            )
            assert len(app.http_payloads) == before_http + (2 if fallback else 1)
            user = next(
                row
                for row in app.store.messages_for_session(app.session.id)
                if row.content == text
            )
            captures = reader.read_calls(user.persisted_message_id)
            assert len(captures) == (2 if fallback else 1)
            rows = captures[0].capture.request["messages_payload"]
            wire_rows = app.http_payloads[-1]["messages"]
            if context_change == "disabled":
                assert rows[-1]["content"] == wire_rows[-1]["content"] == text
                assert GUIDANCE not in json.dumps(rows)
            else:
                assert rows[-2]["content"] == wire_rows[-2]["content"] == text
                expected = (
                    GUIDANCE
                    if context_change == "same"
                    else "UPDATED_PROJECT_GUIDANCE_31976"
                )
                assert expected in rows[-1]["content"]
                assert expected in wire_rows[-1]["content"]
                assert sum(expected in str(row.get("content")) for row in rows) == 1
            assert reader.read_calls(first_user.persisted_message_id) == original


@pytest.mark.parametrize("prior_response", ["ordinary", "tools", "fallback"])
@pytest.mark.parametrize("cold_factory", [False, True])
async def test_transformed_project_sends_preserve_source_and_context(
    tmp_path, monkeypatch, prior_response, cold_factory
):
    """The combined source/context replacement keeps both durable owners."""
    fallback = prior_response == "fallback"
    async with _project_console(
        tmp_path,
        monkeypatch,
        tools=prior_response == "tools",
        response_mode="fallback" if fallback else "complete",
    ) as app:
        assert app.store.persist_session_if_needed(app.session.id)
        app.controller._chat_dictionary_applier = lambda _conversation, text: (
            text.replace("alias", "expanded")
        )
        reader = ConsoleTraceNativeReader(app.db)
        originals = []
        for index, text in enumerate(
            ("alias first", "ordinary next", "alias third", "ordinary last")
        ):
            if index == 2:
                app.project_file.write_text("UPDATED_PROJECT_GUIDANCE_31976")
            elif index == 3:
                app.store.set_session_project_instruction_state(
                    app.session.id,
                    replace(
                        app.session.project_instruction_state,
                        project_instructions_enabled=False,
                    ),
                )
            if cold_factory:
                app.restart_factory()
            before_http = len(app.http_payloads)
            result = await app.controller.submit_draft(text, session_id=app.session.id)
            assert result.accepted and result.provider_started, app.reservation_errors
            assert app.controller.run_state.status.value == "completed", (
                app.reservation_errors
            )
            assert len(app.http_payloads) > before_http
            wire = app.http_payloads[-1]["messages"]
            expected = text.replace("alias", "expanded")
            assert any(row.get("content") == expected for row in wire)
            if index < 3:
                current_index = next(
                    i for i, row in enumerate(wire) if row.get("content") == expected
                )
                assert (
                    GUIDANCE if index < 2 else "UPDATED_PROJECT_GUIDANCE_31976"
                ) in wire[current_index + 1]["content"]
            else:
                assert wire[-1]["content"] == expected
            saved = app.store.get_message(result.user_message_id)
            assert saved.content == text
            trace = reader.read_calls(saved.persisted_message_id)
            assert trace
            assert all(call.capture.request is not None for call in trace)
            originals.append((saved.persisted_message_id, trace))
            for owner_id, original in originals:
                assert reader.read_calls(owner_id) == original
        with app.db.transaction() as cursor:
            pins = cursor.execute(
                "SELECT c.turn_id, r.source_message_id FROM console_trace_calls c "
                "JOIN console_trace_events e ON e.call_id = c.call_id "
                "AND e.event_type = 'call_boundary' "
                "JOIN console_trace_semantic_revisions r "
                "ON r.revision_id = e.semantic_revision_id"
            ).fetchall()
        assert pins and all(row[0] == row[1] for row in pins)
