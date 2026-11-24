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
from tldw_chatbook.Chat.console_scratch_space import ConsoleScratchSpaceManager
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

# The Console bridge reads the tool gates through config, which keeps the
# collection-time profile only (TASK-41508; RecoveryRequired otherwise).
pytestmark = pytest.mark.bootstrap_profile

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
    nested_tools=False,
    discover_tools=False,
    cold_after_request=None,
    tool_prefix="",
    first_failure=None,
    failures=1,
    hosted=False,
    structured_thinking=False,
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
        if nested_tools:
            for name in ("first", "second"):
                folder = root / name
                folder.mkdir()
                (folder / "AGENTS.md").write_text(
                    f"NESTED_{name}_GUIDANCE: check the test output."
                )
                (folder / "data.txt").write_text(f"{name} project data")
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
        attempts = []

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
            attempts.append(request)
            if first_failure is not None and len(attempts) <= failures:
                return first_failure(request)  # returns an error or raises
            payload = json.loads(request.content)
            http_payloads.append(payload)
            if len(http_payloads) == cold_after_request:
                restart_factory()
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
            content = "Fixture reply"
            if tools and len(http_payloads) <= tools:
                call = {
                    "name": "calculator",
                    "arguments": {"expression": f"{5 + len(http_payloads)}*7"},
                }
                if nested_tools:
                    folder = "first" if len(http_payloads) == 1 else "second"
                    call = {
                        "name": "fs_read",
                        "arguments": {"path": f"{folder}/data.txt"},
                    }
                if discover_tools:
                    call = (
                        {"name": "find_tools", "arguments": {"query": "calculator"}}
                        if len(http_payloads) == 1
                        else {
                            "name": "load_tools",
                            "arguments": {"ids": ["builtin:calculator"]},
                        }
                    )
                content = tool_prefix + "```tool_call\n" + json.dumps(call) + "\n```"
            return httpx.Response(
                200, json={"choices": [{"message": {"content": content}}]}
            )

        # hosted: OpenAI through the real chat_api_call dispatcher; the
        # test fakes its requests transport (TASK-42472).
        provider, model, endpoint = (
            ("openai", "gpt-4.1-mini", "https://api.openai.com/v1")
            if hosted
            else ("llama_cpp", "qwen3.7-27b", "http://localhost:8080")
        )
        resolution = ConsoleProviderResolution(
            ready=True,
            provider=provider,
            execution_key=provider,
            model=model,
            base_url=endpoint,
            api_key=RUNTIME_CREDENTIAL,
            streaming=response_mode != "complete",
            local_structured_thinking=structured_thinking,
            resolved_destination=ConsoleResolvedDestination(
                provider=provider,
                model=model,
                endpoint_identity=endpoint,
                egress_class=ConsoleEgressClass.PUBLIC_NETWORK
                if hosted
                else ConsoleEgressClass.ON_DEVICE,
            ),
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            if hosted:
                from tldw_chatbook.Chat.Chat_Functions import chat_api_call

                hosted_kwargs = {"chat_api_call_fn": chat_api_call, "environ": {}}
            else:
                hosted_kwargs = {}
            gateway = ConsoleProviderGateway(
                http_client=client,
                trace_call_boundary_factory=boundary,
                **hosted_kwargs,
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
                settings=ConsoleSessionSettings(provider=provider, model=model),
            )
            bridge = ConsoleAgentBridge(
                agent_runs_db=runs, store=store, provider_gateway=gateway
            )
            scratch_spaces = ConsoleScratchSpaceManager(temp_parent=tmp_path)
            controller = ConsoleChatController(
                store=store,
                provider_gateway=gateway,
                agent_runtime_enabled=True,
                agent_bridge=bridge,
                confirm_project_instruction_dispatch=lambda _notice: "proceed",
                scratch_spaces=scratch_spaces,
            )
            controller.app = SimpleNamespace(workspace_registry_service=registry)
            if nested_tools:
                from Tests.Chat.test_console_project_instruction_persistence_boundary import (
                    _InProcessWorkspaceExecutor,
                )
                from tldw_chatbook.Agents.local_tool_provider import (
                    LocalToolProvider,
                    _default_specs,
                )
                from tldw_chatbook.MCP.permission_store import EffectiveToolState

                local = LocalToolProvider(
                    workspace_root=root,
                    specs=[
                        spec
                        for spec in _default_specs(
                            root, workspace_executor=_InProcessWorkspaceExecutor(root)
                        )
                        if spec.name == "fs_read"
                    ],
                    resolve_state=lambda _tool: EffectiveToolState(
                        state="allow", origin="global_default"
                    ),
                )
                monkeypatch.setattr(
                    controller,
                    "_compose_local_provider",
                    lambda *_args, **_kwargs: (local, lambda _calls: {}),
                )
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
                    attempts=attempts,
                    reservation_errors=reservation_errors,
                    project_file=root / source_name,
                    restart_factory=restart_factory,
                    gateway=gateway,
                    runs=runs,
                    registry=registry,
                )
            finally:
                scratch_spaces.dispose()  # TASK-42112: the controller is never shut down
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


@pytest.mark.parametrize("tool_rounds", [2, 3])
async def test_repeated_project_tool_calls_preserve_the_request_history(
    tmp_path, monkeypatch, tool_rounds
):
    async with _project_console(tmp_path, monkeypatch, tools=tool_rounds) as app:
        result = await app.controller.submit_draft(
            "Calculate twice", session_id=app.session.id
        )
        assert result.accepted
        assert app.controller.run_state.status.value == "completed", (
            app.reservation_errors
        )
        assert len(app.http_payloads) == tool_rounds + 1


async def test_nested_project_context_during_tool_calls_preserves_history(
    tmp_path, monkeypatch
):
    async with _project_console(
        tmp_path, monkeypatch, tools=2, nested_tools=True
    ) as app:
        result = await app.controller.submit_draft(
            "Read both files", session_id=app.session.id
        )
        assert result.accepted
        assert app.controller.run_state.status.value == "completed", (
            app.reservation_errors
        )
        assert len(app.http_payloads) == 3
        assert "NESTED_first_GUIDANCE" in json.dumps(app.http_payloads[1])
        assert "NESTED_second_GUIDANCE" in json.dumps(app.http_payloads[2])


@pytest.mark.parametrize("capture_enabled", [False, True])
@pytest.mark.parametrize("cold_factory", [False, True])
@pytest.mark.parametrize("project_enabled", [False, True])
@pytest.mark.parametrize(
    "tool_prefix", ["", "Planning first.\n"], ids=["bare_fence", "planning_prefix"]
)
async def test_discovered_tool_schema_changes_preserve_the_captured_run(
    tmp_path, monkeypatch, capture_enabled, cold_factory, project_enabled, tool_prefix
):
    from tldw_chatbook.Agents import agent_service
    from tldw_chatbook.Chat.console_trace_redaction import CredentialSanitizer

    # Select discovery while leaving enough budget to load the chosen schema.
    monkeypatch.setattr(agent_service, "get_model_token_limit", lambda *_args: 100_000)
    monkeypatch.setattr(
        agent_service, "catalog_schema_tokens", lambda *_args, **_kwargs: 10_001
    )
    async with _project_console(
        tmp_path,
        monkeypatch,
        tools=2,
        discover_tools=True,
        capture_enabled=capture_enabled,
        cold_after_request=2 if cold_factory else None,
        tool_prefix=tool_prefix,
    ) as app:
        if not project_enabled:
            app.store.set_session_project_instruction_state(
                app.session.id,
                replace(
                    app.session.project_instruction_state,
                    project_instructions_enabled=False,
                ),
            )
        result = await app.controller.submit_draft(
            "Find and load the calculator", session_id=app.session.id
        )
        assert result.accepted
        assert app.controller.run_state.status.value == "completed", (
            app.reservation_errors
        )
        assert len(app.http_payloads) == 3

        assert (
            app.http_payloads[1]["messages"][0] != app.http_payloads[2]["messages"][0]
        ), app.http_payloads[2]["messages"]
        reader = ConsoleTraceNativeReader(app.db)
        user = app.store.get_message(result.user_message_id)
        original = reader.read_calls(user.persisted_message_id)
        assert len(original) == (3 if capture_enabled else 0)
        sanitizer = CredentialSanitizer(known_credentials=(RUNTIME_CREDENTIAL,))

        def wire_rows(captured):
            return [
                {
                    key: value
                    for key, value in row.items()
                    if key != "_chatbook_ephemeral_origin"
                }
                for row in captured.capture.request["messages_payload"]
            ]

        for captured, sent in zip(original, app.http_payloads, strict=capture_enabled):
            assert wire_rows(captured) == sanitizer.sanitize(sent["messages"]).value
            assert RUNTIME_CREDENTIAL not in json.dumps(captured.capture.request)
            assert CREDENTIAL not in json.dumps(captured.capture.request)
        for text in ("Next question", "And one more"):
            if cold_factory:
                app.restart_factory()
            if not capture_enabled:
                app.controller.set_next_trace_privacy(
                    app.session.id,
                    capture_enabled=False,
                    pii_redaction_enabled=False,
                    expected_policy_revision=app.controller.capture_policy_snapshot(
                        app.session.id
                    ).policy_revision,
                )
            followup = await app.controller.submit_draft(
                text, session_id=app.session.id
            )
            assert followup.accepted
            assert app.controller.run_state.status.value == "completed", (
                app.reservation_errors
            )
            assert reader.read_calls(user.persisted_message_id) == original
            current_user = app.store.get_message(followup.user_message_id)
            current = reader.read_calls(current_user.persisted_message_id)
            assert len(current) == (1 if capture_enabled else 0)
            if current:
                assert (
                    wire_rows(current[0])
                    == sanitizer.sanitize(app.http_payloads[-1]["messages"]).value
                )


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
        # The applier takes the turn's frozen inputs as a third argument;
        # a two-argument double raised and the send kept "alias" (TASK-41508).
        app.controller._chat_dictionary_applier = (
            lambda _conversation, text, _frozen_inputs=None: text.replace(
                "alias", "expanded"
            )
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


def _llama_failure(kind):
    def fail(request: httpx.Request) -> httpx.Response:
        if kind.startswith("http"):
            return httpx.Response(int(kind[4:]), json={"error": {"message": "fixture"}})
        error = {
            "connect": httpx.ConnectError,
            "timeout": httpx.ReadTimeout,
            "remote_protocol": httpx.RemoteProtocolError,
        }[kind]
        raise error("fixture", request=request)

    return fail


@pytest.mark.parametrize(
    "failure,retried",
    [
        ("http429", True),
        ("http503", True),
        ("connect", True),
        ("timeout", True),
        ("remote_protocol", True),
        ("http400", False),
        ("http401", False),
        ("http404", False),
    ],
)
async def test_llamacpp_agent_retries_a_transient_failure_in_place(
    tmp_path, monkeypatch, failure, retried
):
    """TASK-42472: a llama.cpp agent run retries a transient failure in place.

    The direct HTTP path raises raw httpx errors, which the agent's retry
    classifier did not count, so a 503 or a refused connection ended the run
    at once. The retried call is admitted to the captured trace as a
    TOOL_LOOP call against the failed first call; a 4xx still ends the run.
    """
    async with _project_console(
        tmp_path, monkeypatch, first_failure=_llama_failure(failure)
    ) as app:
        result = await app.controller.submit_draft("Hello", session_id=app.session.id)
        assert result.accepted
        with app.db.transaction() as cursor:
            calls = app.factory.repository.read_conversation_call_lineage(
                cursor, app.session.persisted_conversation_id
            )
        lineage = [(call.route_identity, call.state.value) for call in calls]
        assert app.reservation_errors == []
        if retried:
            assert app.controller.run_state.status.value == "completed"
            assert len(app.attempts) == 2
            assert lineage == [("agent_first", "error"), ("tool_loop", "complete")]
        else:
            assert app.controller.run_state.status.value == "failed"
            assert len(app.attempts) == 1
            assert lineage == [("agent_first", "error")]


_PARTIAL = "The capital of Fr"


def _sse(*contents):
    events = [{"choices": [{"index": 0, "delta": {"content": text}}]} for text in contents]
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events)


_FAILED_REASONING = "Reasoning the failed call streamed"


class _BrokenLlamaStream(httpx.AsyncByteStream):
    """Reasoning and one content event, then the connection breaks mid-body."""

    async def __aiter__(self):
        reasoning = {"choices": [{"index": 0, "delta": {"reasoning_content": _FAILED_REASONING}}]}
        yield f"data: {json.dumps(reasoning)}\n\n".encode()
        yield _sse(_PARTIAL).encode()
        raise httpx.RemoteProtocolError("fixture: peer closed mid-body")


def _hosted_post(failures, attempts):
    """A requests transport whose first ``failures`` streams break mid-body."""
    from unittest.mock import Mock

    import requests

    def post(*_args, **_kwargs):
        attempts.append(True)
        broken = len(attempts) <= failures
        body = _sse(_PARTIAL if broken else "Fixture reply").encode()

        def lines(*_a, **_k):
            yield body.strip()
            if broken:
                raise requests.exceptions.ChunkedEncodingError("fixture: mid-body")
            yield b"data: [DONE]"

        response = Mock(status_code=200, ok=True, headers={})
        response.iter_lines.side_effect = lines
        return response

    return post


@pytest.mark.parametrize("failures", [1, 3], ids=["retried", "exhausted"])
@pytest.mark.parametrize("hosted", [False, True], ids=["llama_cpp", "openai"])
async def test_a_retry_after_a_mid_stream_failure_replaces_the_partial_reply(
    tmp_path, monkeypatch, hosted, failures
):
    """TASK-42472: an in-place retry starts from the reply the failed call found.

    The failed stream's partial text stayed in the assistant message and the
    retry streamed after it ("The capital of FrFixture reply"). A run whose
    retries run out keeps only its last attempt's partial, as a run that
    fails without retrying does.
    """
    attempts = []
    if hosted:
        monkeypatch.setattr("requests.Session.post", _hosted_post(failures, attempts))
    async with _project_console(
        tmp_path,
        monkeypatch,
        response_mode="stream",
        first_failure=None
        if hosted
        else lambda _request: httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_BrokenLlamaStream(),
        ),
        failures=failures,
        hosted=hosted,
        structured_thinking=not hosted,
    ) as app:
        result = await app.controller.submit_draft("Hello", session_id=app.session.id)
        assert result.accepted
        (reply,) = [
            message
            for message in app.store.messages_for_session(app.session.id)
            if message.role.value == "assistant"
        ]
        saved = app.db.get_message_by_id(reply.persisted_message_id)
        assert len(attempts if hosted else app.attempts) == min(failures + 1, 3)
        if failures == 1:
            assert app.controller.run_state.status.value == "completed"
            assert (reply.status, reply.content) == ("complete", "Fixture reply")
        else:
            assert app.controller.run_state.status.value == "failed"
            # Only the last attempt's partial: the stream gate holds back a
            # short tail, so the visible partial is a prefix of _PARTIAL.
            assert reply.status == "failed"
            assert reply.content and _PARTIAL.startswith(reply.content)
        assert saved["content"] == reply.content
        if not hosted:
            # The failed call's reasoning goes with its text: none survives a
            # successful retry, and only the last attempt's when they run out.
            reasoning = [block.text for block in getattr(reply.thinking, "blocks", ())]
            assert reasoning == ([] if failures == 1 else [_FAILED_REASONING])
