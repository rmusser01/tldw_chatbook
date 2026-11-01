"""A new native user turn retains authenticated immutable archive selection."""

import asyncio
import json
import socket
import threading
from dataclasses import replace
from functools import partial
from types import SimpleNamespace

import httpx
import pytest

from Tests.Chat.test_archive_recovery_operations import (
    recovery_stores as recovery_stores,  # noqa: PLC0414 - pytest fixture export
)
from Tests.Chat.test_console_archive_recovery_dispatch import (
    RecoveryGateway,
    _DISPATCH_WAIT_SECONDS,
)
from Tests.Chat.test_console_prepared_request import _private_group
from Tests.Chat.test_console_runtime_lifetime import _View
from Tests.Chat.test_console_turn_library_authority import (
    _allowed_candidate,
    _execution_configuration,
)
from tldw_chatbook.Agents.library_rag_tool_provider import LibraryRagToolProvider
from tldw_chatbook.Agents.library_tool_provider import LibraryToolProvider
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_agent_bridge import (
    ConsoleAgentBridge,
    ConsoleAgentTraceRequestFactory,
)
from tldw_chatbook.Chat.console_chat_controller import (
    ConsoleChatController,
    _flatten_preflight_messages,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleEgressClass,
    ConsoleResolvedDestination,
)
from tldw_chatbook.Chat.console_library_policy import (
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
)
from tldw_chatbook.Chat.console_prepared_request import (
    CONTINUATION_OWNER_KEY,
    THINKING_OWNER_KEY,
    build_console_request,
    thaw_json,
)
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    ConsoleProviderResolution,
    ProviderToolCalls,
)
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime, _library_provider_for_app
from tldw_chatbook.Chat.console_thinking_history import (
    ResolvedThinkingBlock,
    ThinkingOwnerGroup,
)
from tldw_chatbook.Chat.console_trace_errors import TraceCallPersistenceError
from tldw_chatbook.Chat.console_trace_models import FrozenTracePolicy, new_opaque_id
from tldw_chatbook.Chat.console_trace_native_reader import ConsoleTraceNativeReader
from tldw_chatbook.Chat.console_trace_provenance import (
    ConsoleRequestRoute,
    DerivedTraceProvenance,
    ProviderArtifactTraceProvenance,
    SavedRevisionTraceProvenance,
    TraceProvenanceSource,
    TraceTransformKind,
)
from tldw_chatbook.Chat.console_trace_runtime import ConsoleTraceBoundaryFactory
from tldw_chatbook.Chat.local_reasoning import ReasoningReplayPolicy
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Library.local_library_tool_service import LocalLibraryToolService
from tldw_chatbook.UI.Console_Modules.wiring import _admit_console_turn_to_runtime


class FollowupGateway(RecoveryGateway):
    """Observe emitted provider context; select only its original second row."""

    def __init__(self):
        super().__init__()
        self.phase = "search"
        self.index = 0
        self.followup_context = None
        self.original_page = None
        self.destination_changes = {}
        self.response_reserve = None
        self.on_resolution = lambda: None

    async def resolve_for_send(self, selection):
        self.on_resolution()
        resolution = await super().resolve_for_send(selection)
        # Fake transport only, with a real immutable loopback destination identity.
        resolution.resolved_destination = replace(
            ConsoleResolvedDestination(
                provider=selection.provider,
                model=selection.explicit_model or selection.configured_model,
                endpoint_identity="http://127.0.0.1:9099",
                egress_class=ConsoleEgressClass.ON_DEVICE,
            ),
            **self.destination_changes,
        )
        resolution.max_tokens = self.response_reserve
        return resolution

    async def stream_chat(self, _resolution, messages, tools=None, **kwargs):
        self.messages_seen.append(messages)
        self.tools_seen.append(tools)
        self.stream_calls += 1
        index = self.index
        self.index += 1
        if self.phase == "search":
            if index == 0:
                name, arguments = (
                    "load_tools",
                    {"ids": ["archive_recovery:search_archived_conversations"]},
                )
            elif index == 1:
                name, arguments = "search_archived_conversations", {}
            else:
                self.original_page = json.loads(
                    messages[-1]["content"].removeprefix(
                        "Tool result for search_archived_conversations: "
                    )
                )
                yield "The first archived conversation is Original."
                return
        elif index == 0:
            # The original tool result is intentionally absent from prose.
            for message in messages:
                content = message.get("content")
                if not isinstance(content, str):
                    continue
                for line in content.splitlines():
                    try:
                        payload = json.loads(line)
                    except ValueError:
                        continue
                    if (
                        isinstance(payload, dict)
                        and "archive_recovery_results" in payload
                    ):
                        self.followup_context = payload
            if (
                self.phase == "observe"
                or not self.followup_context
                or not self.followup_context["archive_recovery_results"]
            ):
                yield "Which conversation did you mean?"
                return
            name, arguments = (
                "load_tools",
                {"ids": ["archive_recovery:request_conversation_restore"]},
            )
        elif index == 1:
            page = self.followup_context["archive_recovery_results"][0]
            name, arguments = (
                "request_conversation_restore",
                {
                    "result_ref": page["items"][1]["result_ref"],
                    "scope": "chat_only",
                },
            )
        else:
            yield "Recovery finished."
            return
        yield ProviderToolCalls(
            [
                {
                    "id": f"{self.phase}-{index}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ]
        )


@pytest.fixture
async def native_followup(recovery_stores, tmp_path):
    app, first, db = recovery_stores
    second = db.add_conversation({"title": "Second exact target"})
    db.set_conversations_archived(
        [second], archived=True, expected_versions={second: 1}
    )
    # Explicit chronology makes the native card order independent of clock granularity.
    with db.transaction() as cursor:
        cursor.execute(
            "UPDATE conversations SET archived_at = ? WHERE id = ?",
            ("2026-09-20T00:00:00+00:00", first),
        )
        cursor.execute(
            "UPDATE conversations SET archived_at = ? WHERE id = ?",
            ("2026-09-19T00:00:00+00:00", second),
        )
    with app.workspace_registry_service.db.transaction() as cursor:
        cursor.execute(
            "UPDATE workspace_records SET archived_at = ? WHERE workspace_id = ?",
            ("2026-09-18T00:00:00+00:00", "project"),
        )
    db.add_message(
        {
            "conversation_id": first,
            "sender": "user",
            "content": "ARCHIVE_BODY_NOT_FOR_FOLLOWUP",
        }
    )
    source_id = db.add_conversation({"title": "Source"})
    persistence = ChatPersistenceService(db)
    persistence.console_library_policy_repository.insert(
        source_id,
        replace(
            _allowed_candidate(allowed=True), auto_retrieve=ConsoleAutoRetrieve.NEVER
        ),
    )
    store = ConsoleChatStore(persistence=persistence)
    source = store.restore_persisted_session(
        title="Source",
        workspace_id=None,
        persisted_conversation_id=source_id,
        all_nodes=(),
    )
    await store.hydrate_session_library_policy(source.id)
    gateway = FollowupGateway()
    runs = AgentRunsDB(tmp_path / "followup-runs.sqlite", "test")
    bridge = ConsoleAgentBridge(
        agent_runs_db=runs,
        store=store,
        provider_gateway=gateway,
        native_tools_enabled=lambda: True,
    )
    owner = ConsoleRuntime(app)
    owner.set_chat_store(store)
    owner.set_agent_bridge(bridge)
    recovery = owner.ensure_archive_recovery()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        agent_runtime_enabled=True,
        agent_bridge=bridge,
        library_provider_factory=lambda context: (
            LibraryToolProvider(
                LocalLibraryToolService(
                    conversation_service=app.local_chat_conversation_service
                )
            )
            if context.library_authority.direct_library_tools
            else LibraryRagToolProvider(None)
        ),
    )
    owner.set_chat_controller(controller)
    controller._archive_recovery_runtime = recovery
    cards = []
    ready = asyncio.Event()

    def present(card):
        cards.append(card)
        ready.set()
        return True

    view = _View(
        {
            "present_confirmation": present,
            "is_answerable": lambda sid: sid == state.source.id,
        }
    )
    attached = owner.attach_view(view)
    owner.finish_view_reconciliation(view, attached)
    state = SimpleNamespace(
        app=app,
        db=db,
        first=first,
        second=second,
        store=store,
        source=source,
        gateway=gateway,
        recovery=recovery,
        controller=controller,
        persistence=persistence,
        cards=cards,
        ready=ready,
        owner=owner,
        runs=runs,
        direct=True,
        restriction=None,
    )

    def configuration(sid):
        captured = _execution_configuration(
            sid,
            direct=state.direct,
            native_tools=state.restriction != "non_native",
            agent_runtime=True,
            scratch_space=controller._scratch_spaces.snapshot(sid),
        )
        return replace(
            captured,
            provider_selection=replace(
                captured.provider_selection,
                explicit_model="gpt-4o",
                provider="unsupported"
                if state.restriction == "unsupported"
                else "openai",
            ),
            library_policy_maximum=(
                replace(
                    state.source.library_policy_holder.snapshot,
                    assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
                )
                if state.restriction == "frozen_blocked"
                else None
            ),
            persona_policy_rules=(
                {
                    "rule_kind": "mcp_tool",
                    "rule_name": "search_archived_conversations",
                    "allowed": False,
                },
            )
            if state.restriction == "persona"
            else (
                {
                    "rule_kind": "mcp_tool",
                    "rule_name": "search_archived_conversations",
                    "allowed": True,
                    "require_confirmation": True,
                },
            )
            if state.restriction == "ask"
            else (),
        )

    screen = SimpleNamespace(
        _ensure_console_chat_store=lambda: store,
        _session=SimpleNamespace(_build_console_turn_execution_context=configuration),
        _console_runtime=lambda: owner,
    )

    async def send(text):
        if state.restriction == "legacy":
            return await controller.submit_draft(
                text,
                session_id=state.source.id,
                configuration=configuration(state.source.id),
            )
        turn_id = _admit_console_turn_to_runtime(screen, text, state.source.id)
        return await owner.wait_for_turn(turn_id)

    state.send = send
    try:
        yield state
    finally:
        try:
            await owner.dispose()
        finally:
            runs.close()


@pytest.fixture
async def captured_followup(native_followup, monkeypatch, request):
    """Keep native custody and capture real; replace only external transport."""
    rig = native_followup
    rig.thinking_format = getattr(request, "param", False)
    rig.thinking_enabled = bool(rig.thinking_format)
    rig.fenced = rig.thinking_format == "fenced-structured"
    rig.controller._library_provider_factory = partial(
        _library_provider_for_app, rig.app
    )
    factory = ConsoleTraceBoundaryFactory(rig.db)
    rig.trace_failures = []
    rig.trace_requests = []
    rig.dns_attempts = []

    def deny_dns(*args, **_kwargs):
        rig.dns_attempts.append(args)
        raise OSError("Capture regression forbids DNS resolution")

    for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr"):
        monkeypatch.setattr(socket, name, deny_dns)
    # Tests/conftest.py additionally denies and records IPv4/IPv6 socket I/O.
    provider = "llama_cpp" if rig.thinking_enabled else "openai"
    model = (
        "unsloth/Qwen3.8-27B-GGUF:Q8_0"
        if rig.thinking_format in {"structured", "fenced-structured"}
        else "Gemma"
        if rig.thinking_enabled
        else "gpt-4o"
    )
    resolution = ConsoleProviderResolution(
        ready=True,
        provider=provider,
        execution_key=provider,
        model=model,
        thinking_stream_disposition="displayable"
        if rig.thinking_enabled
        else "ignored",
        thinking_round_trip_version=1 if rig.thinking_enabled else None,
        local_structured_thinking=rig.thinking_enabled,
        reasoning_replay=ReasoningReplayPolicy(
            mode="server_default" if rig.fenced else "current",
            source="Frozen transport capabilities",
            native_tools=not rig.fenced,
        )
        if rig.thinking_enabled
        else None,
        base_url="http://127.0.0.1:9099",
        streaming=False,
        resolved_destination=ConsoleResolvedDestination(
            provider=provider,
            model=model,
            endpoint_identity="http://127.0.0.1:9099",
            egress_class=ConsoleEgressClass.ON_DEVICE,
        ),
    )

    async def response(messages, tools=None):
        message = {
            "content": "<think>Find the exact archived selection.</think>"
            if rig.thinking_enabled
            and rig.thinking_format not in {"structured", "fenced-structured"}
            else ""
        }
        if rig.thinking_format in {"structured", "fenced-structured"}:
            message["reasoning_content"] = "Find the exact archived selection."
        async for chunk in rig.gateway.stream_chat(resolution, messages, tools=tools):
            if isinstance(chunk, ProviderToolCalls):
                if rig.fenced:
                    function = chunk.tool_calls[0]["function"]
                    message["content"] += (
                        "```tool_call\n"
                        + json.dumps(
                            {
                                "name": function["name"],
                                "arguments": json.loads(function["arguments"]),
                            }
                        )
                        + "\n```"
                    )
                else:
                    message["tool_calls"] = chunk.tool_calls
            else:
                message["content"] += chunk
        if rig.fenced:
            assert "tool_calls" not in message
        return {"choices": [{"message": message}]}

    def adapter(**kwargs):
        assert not rig.fenced, "Fenced llama.cpp uses the real HTTPX completion path"
        return asyncio.run(response(kwargs["messages_payload"], kwargs.get("tools")))

    async def transport(request):
        assert resolution.ready and resolution.provider == "llama_cpp"
        assert resolution.reasoning_replay.native_tools is False
        assert resolution.reasoning_replay.mode == "server_default"
        assert request.method == "POST"
        assert str(request.url) == "http://127.0.0.1:9099/v1/chat/completions"
        payload = json.loads(request.content)
        assert "tools" not in payload and "tool_choice" not in payload
        assert payload["model"] == "unsloth/Qwen3.8-27B-GGUF:Q8_0"
        assert payload["stream"] is False
        return httpx.Response(200, json=await response(payload["messages"]))

    client = (
        httpx.AsyncClient(transport=httpx.MockTransport(transport))
        if rig.fenced
        else None
    )

    def boundary(request, resolved, route):
        rig.trace_requests.append(request.semantic)
        try:
            return factory(request, resolved, route)
        except ValueError as exc:
            rig.trace_failures.append(str(exc))
            raise

    gateway = ConsoleProviderGateway(
        http_client=client,
        chat_api_call_fn=adapter,
        trace_call_boundary_factory=boundary,
    )

    async def resolve(_selection):
        return resolution

    monkeypatch.setattr(gateway, "resolve_for_send", resolve)
    monkeypatch.setattr(rig.controller, "provider_gateway", gateway)
    monkeypatch.setattr(rig.owner._agent_bridge, "_gateway", gateway)
    rig.controller.set_next_trace_privacy(
        rig.source.id,
        capture_enabled=True,
        pii_redaction_enabled=False,
        expected_policy_revision=rig.controller.capture_policy_snapshot(
            rig.source.id
        ).policy_revision,
    )
    try:
        yield rig
    finally:
        # A RED must settle its own paused accepted turn before native shutdown.
        preparation = rig.store.preparation_for_session(rig.source.id)
        if preparation is not None and preparation.state.value == "paused":
            cancelled = rig.controller.cancel_library_preparation(
                preparation.preparation_id
            )
            assert rig.store.preparation_for_session(rig.source.id) is None, cancelled
        recovery = rig.store.dispatch_recovery_for_session(rig.source.id)
        if recovery is not None:
            rig.store.mark_dispatch_recovery_needed(
                rig.source.id, recovery.assistant_message_id
            )
            discarded = await rig.controller.discard_dispatch_recovery(rig.source.id)
            assert discarded.accepted, discarded
        assert rig.store.dispatch_recovery_for_session(rig.source.id) is None
        await gateway.aclose()
        if client is not None:
            await client.aclose()
        assert not rig.dns_attempts


def _captured_turn_evidence(db, turn_id):
    reader = ConsoleTraceNativeReader(db)
    with db.transaction() as cursor:
        calls = tuple(
            tuple(row)
            for row in cursor.execute(
                "SELECT * FROM console_trace_calls WHERE turn_id = ? ORDER BY call_sequence",
                (turn_id,),
            )
        )
        records = tuple(reader.repository.get_call(cursor, row[0]) for row in calls)
        return (
            calls,
            tuple(reader._reconstruct_request(cursor, call) for call in records),
            tuple(
                (
                    reader.repository.get_request_header(
                        cursor, call.request_header_id
                    ),
                    reader.repository.get_response_link(cursor, call.call_id),
                    reader._reconstruct_response(cursor, call),
                )
                for call in records
            ),
        )


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
@pytest.mark.parametrize("interruption", ["refusal", "cancellation"])
async def test_native_archive_reseal_releases_exact_admitted_authority(
    captured_followup, monkeypatch, interruption
):
    rig = captured_followup
    await rig.send("Find my archived conversations.")
    first_user = next(
        m
        for m in rig.store.messages_for_session(rig.source.id)
        if m.role is ConsoleMessageRole.USER
    )
    previous = _captured_turn_evidence(rig.db, first_user.persisted_message_id)
    transport_count = rig.gateway.stream_calls
    boundary_count = len(rig.trace_requests)
    authorities = []
    releases = []
    errors = []
    started, unblock, drained = (threading.Event() for _ in range(3))
    admit = rig.recovery.admit
    release = rig.recovery.release
    reseal = rig.controller._reseal_archive_followup_trace_request

    def observe_admission(*args):
        provider = admit(*args)
        assert provider is not None
        authorities.append(provider.authority)
        return provider

    async def observe_release(authority):
        releases.append(authority)
        await release(authority)

    def interrupted_reseal(request, before, after, current_user_id):
        try:
            if interruption == "cancellation":
                started.set()
                assert unblock.wait(15), "Held reseal worker was not drained"
            else:
                request = replace(
                    request,
                    provenance=replace(
                        request.provenance,
                        active_request=(SavedRevisionTraceProvenance(new_opaque_id()),),
                    ),
                )
            return reseal(request, before, after, current_user_id)
        except TraceCallPersistenceError as exc:
            errors.append(exc)
            raise
        finally:
            drained.set()

    monkeypatch.setattr(rig.recovery, "admit", observe_admission)
    monkeypatch.setattr(rig.recovery, "release", observe_release)
    monkeypatch.setattr(
        rig.controller, "_reseal_archive_followup_trace_request", interrupted_reseal
    )
    rig.gateway.phase, rig.gateway.index = "restore", 0
    turn = asyncio.create_task(rig.send("Restore the second one"))
    try:
        if interruption == "cancellation":
            assert await asyncio.to_thread(started.wait, 10)
            assert len(authorities) == 1
            assert rig.recovery.authenticates(authorities[0])
            assert rig.controller.stop_active_run()
        (result,) = await asyncio.wait_for(
            asyncio.gather(turn, return_exceptions=True), 15
        )
        assert len(authorities) == 1
        authority = authorities[0]
        assert authority.closed and id(authority) not in rig.recovery._authorities
        assert releases == [authority]
        assert not rig.recovery.authenticates(authority)
        assert authority.run_id is None
        assert rig.gateway.stream_calls == transport_count
        assert len(rig.trace_requests) == boundary_count
        assert not rig.cards
        assert rig.db.get_conversation_by_id(rig.first)["archived"] == 1
        assert rig.db.get_conversation_by_id(rig.second)["archived"] == 1
        assert (
            _captured_turn_evidence(rig.db, first_user.persisted_message_id) == previous
        )
        assert not isinstance(result, BaseException), result
        if interruption == "cancellation":
            assert not drained.is_set()
            assert not errors
        if interruption == "refusal":
            assert result.terminal_status.value == "blocked"
            assert len(errors) == 1
            assert errors[0].reservation_status == "not_established"
            assert str(errors[0]) == "Provider call trace persistence failed."
    finally:
        unblock.set()
        assert await asyncio.to_thread(drained.wait, 15)
        await asyncio.gather(turn, return_exceptions=True)
        preparation = rig.store.preparation_for_session(rig.source.id)
        if preparation is not None and preparation.state.value == "paused":
            cancelled = rig.controller.cancel_library_preparation(
                preparation.preparation_id
            )
            assert rig.store.preparation_for_session(rig.source.id) is None, cancelled
        assert rig.store.dispatch_recovery_for_session(rig.source.id) is None


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
@pytest.mark.parametrize("direct", [True, False])
@pytest.mark.parametrize(
    "captured_followup",
    [False, True, "structured", "fenced-structured"],
    indirect=True,
    ids=["plain", "thinking", "structured", "fenced-structured"],
)
async def test_captured_native_turn_restores_original_second_row(
    captured_followup, direct, monkeypatch
):
    rig = captured_followup
    rig.direct = direct
    await rig.send("Find my archived conversations.")
    first_user = next(
        m
        for m in rig.store.messages_for_session(rig.source.id)
        if m.role is ConsoleMessageRole.USER
    )
    previous = _captured_turn_evidence(rig.db, first_user.persisted_message_id)
    assert len(previous[0]) == 3, [
        (m.role, m.content) for m in rig.store.messages_for_session(rig.source.id)
    ]
    historical = [
        rig.db.get_message_by_id(message.persisted_message_id)
        for message in rig.store.messages_for_session(rig.source.id)
        if message.role in {ConsoleMessageRole.USER, ConsoleMessageRole.ASSISTANT}
    ]
    assistant = next(
        message
        for message in rig.store.messages_for_session(rig.source.id)
        if message.role is ConsoleMessageRole.ASSISTANT
    )
    if rig.thinking_enabled:
        assert len(assistant.thinking.blocks) == 3
        assert all(block.status == "complete" for block in assistant.thinking.blocks)
        assert historical[-1]["thinking_blocks_json"] is not None
        expected_format = (
            "reasoning_content"
            if rig.thinking_format in {"structured", "fenced-structured"}
            else "start_anchored_think"
        )
        assert assistant.thinking.blocks[-1].source_format == expected_format
        header, link, response = previous[2][-1]
        assert link.link_kind == "revision"
        assert link.verification_outcome == "verified_equal"
        with rig.db.transaction() as cursor:
            revision = rig.persistence.console_trace_repository.get_semantic_revision(
                cursor, link.semantic_revision_id
            )
        assert revision.source_message_id == assistant.persisted_message_id
        assert revision.live_message_id == assistant.persisted_message_id
        assert revision.source_conversation_id == rig.source.persisted_conversation_id
        assert response["thinking"][0]["source_format"] == expected_format
        assert response["thinking"][0]["text"] == "Find the exact archived selection."
        assert header.adapter_defaults["response_projection"] == {
            "version": 2,
            "thinking_stream_disposition": "displayable",
            "thinking_round_trip_version": 1,
            "provider": "llama_cpp",
            "model": "unsloth/Qwen3.8-27B-GGUF:Q8_0"
            if rig.thinking_format in {"structured", "fenced-structured"}
            else "Gemma",
            "protocol": "chat_completions",
            "source_formats": ("start_anchored_think", "reasoning_content"),
        }
        with rig.db.transaction() as cursor:
            raw_defaults = cursor.execute(
                "SELECT adapter_defaults_json FROM console_trace_request_headers "
                "WHERE header_id = ?",
                (header.header_id,),
            ).fetchone()[0]
        assert json.loads(raw_defaults)["response_projection"]["source_formats"] == [
            "start_anchored_think",
            "reasoning_content",
        ]
    original = rig.gateway.original_page
    reseal = rig.controller._reseal_archive_followup_trace_request
    resealed_requests = []

    def observe_reseal(request, before, after, current_user_id):
        result = reseal(request, before, after, current_user_id)
        assert (
            replace(
                result,
                active_request=request.active_request,
                provenance=request.provenance,
            )
            == request
        )
        if rig.thinking_enabled and not resealed_requests:
            groups = [
                group for unit in result.compactable for group in unit.thinking_groups
            ]
            assert len(groups) == 1
            assert groups[0].owner_message_id == assistant.id
            assert [
                (block.source_format, block.text) for block in groups[0].blocks
            ] == [
                (block.source_format, block.text)
                for block in assistant.thinking.blocks
                if not block.source_format.endswith(":tool_call")
            ]
        resealed_requests.append(result)
        return result

    monkeypatch.setattr(
        rig.controller, "_reseal_archive_followup_trace_request", observe_reseal
    )
    offset = len(rig.gateway.messages_seen)
    rig.gateway.phase, rig.gateway.index = "restore", 0
    turn = asyncio.create_task(rig.send("Restore the second one"))
    ready = asyncio.create_task(rig.ready.wait())
    try:
        await asyncio.wait_for(
            asyncio.wait({turn, ready}, return_when=asyncio.FIRST_COMPLETED),
            _DISPATCH_WAIT_SECONDS,
        )
        assert rig.cards, (
            "Captured follow-up did not reach exact confirmation",
            rig.gateway.followup_context,
            rig.gateway.messages_seen[offset:],
            rig.trace_failures,
            rig.controller.run_state_for(rig.source.id),
        )
        assert not turn.done()
        if rig.thinking_enabled:
            bound = rig.trace_requests[offset]
            historical_index = next(
                index
                for index, row in enumerate(bound.flattened_messages())
                if row.get("role") == "assistant"
            )
            descriptor = bound.provenance.flattened_messages()[historical_index]
            assert type(descriptor) is SavedRevisionTraceProvenance
            with rig.db.transaction() as cursor:
                saved_owner = (
                    rig.persistence.console_trace_repository.get_semantic_revision(
                        cursor, descriptor.revision_id
                    )
                )
            assert saved_owner.source_message_id == assistant.persisted_message_id
            assert (
                saved_owner.source_conversation_id
                == rig.source.persisted_conversation_id
            )
            if rig.fenced:
                groups = [
                    group
                    for unit in bound.compactable
                    for group in unit.thinking_groups
                ]
                assert len(groups) == 1
                assert [
                    (block.source_format, block.text) for block in groups[0].blocks
                ] == [
                    (block.source_format, block.text)
                    for block in assistant.thinking.blocks
                    if not block.source_format.endswith(":tool_call")
                ]
            else:
                # Current-exchange replay legitimately omits this historical group.
                assert not any(unit.thinking_groups for unit in bound.compactable)
            assert not bound.active_thinking_groups
            assert all(
                "<think>" not in str(row.get("content", ""))
                for row in rig.gateway.messages_seen[offset]
            )
        assert (
            rig.gateway.followup_context["archive_recovery_results"][0]["items"][1][
                "result_ref"
            ]
            == original["items"][1]["result_ref"]
        )
        assert rig.cards[0].target.conversation_id == rig.second
        operation = rig.recovery.coordinator._operations[rig.cards[0].operation_id]
        call_key = operation.call_key
        assert call_key.session_id == rig.source.id
        assert call_key.call_id == ("turn-2-call-0" if rig.fenced else "restore-1")
        assert rig.db.get_conversation_by_id(rig.second)["archived"] == 1
        outcome = await rig.recovery.coordinator.decide(
            rig.source.id, rig.cards[0].request_id, allow=True, scope="chat_only"
        )
        assert outcome.chat_changed and not outcome.workspace_changed
        result = await asyncio.wait_for(turn, _DISPATCH_WAIT_SECONDS)
        assert result.terminal_status.value == "completed", rig.trace_failures
        restore_steps = [
            step
            for step in rig.runs.get_run_fresh(call_key.run_id)["steps"]
            if step["tool_name"] == "request_conversation_restore"
            and step["kind"] in {"tool_proposed", "tool_execution_started"}
        ]
        assert {step["kind"] for step in restore_steps} == {
            "tool_proposed",
            "tool_execution_started",
        }
        assert all(step["call_id"] == call_key.call_id for step in restore_steps)
        assert rig.db.get_conversation_by_id(rig.second)["archived"] == 0
        assert rig.db.get_conversation_by_id(rig.first)["archived"] == 1
        assert rig.app.workspace_registry_service.get_workspace("project").archived
        assert rig.store.active_session_id == rig.source.id
        assert (
            _captured_turn_evidence(rig.db, first_user.persisted_message_id) == previous
        )
        current = rig.store.get_message(result.user_message_id)
        assert current.content == "Restore the second one"
        captures = ConsoleTraceNativeReader(rig.db).read_calls(
            current.persisted_message_id
        )
        assert len(captures) == 3
        assert (
            captures[0].capture.request["messages_payload"]
            == rig.gateway.messages_seen[offset]
        )
        assert original["items"][1]["result_ref"] in str(captures[0].capture.request)
        with rig.db.transaction() as cursor:
            pins = cursor.execute(
                "SELECT e.semantic_revision_id, r.source_message_id FROM console_trace_events e "
                "JOIN console_trace_calls c ON c.call_id = e.call_id "
                "JOIN console_trace_semantic_revisions r ON r.revision_id = e.semantic_revision_id "
                "WHERE c.turn_id = ? AND e.event_type = 'call_boundary' ORDER BY c.call_sequence",
                (current.persisted_message_id,),
            ).fetchall()
        assert len(pins) == 3 and len({row[0] for row in pins}) == 1
        assert all(row[1] == current.persisted_message_id for row in pins)
        current_before = _captured_turn_evidence(rig.db, current.persisted_message_id)
        if rig.thinking_enabled:
            terminal_link = current_before[2][-1][1]
            assert terminal_link.link_kind == "revision"
            assert terminal_link.verification_outcome == "verified_equal"
            restored_answer = [
                m
                for m in rig.store.messages_for_session(rig.source.id)
                if m.role is ConsoleMessageRole.ASSISTANT
            ][-1]
            with rig.db.transaction() as cursor:
                terminal_revision = (
                    rig.persistence.console_trace_repository.get_semantic_revision(
                        cursor, terminal_link.semantic_revision_id
                    )
                )
            assert (
                terminal_revision.source_message_id
                == restored_answer.persisted_message_id
            )
            assert (
                terminal_revision.live_message_id
                == restored_answer.persisted_message_id
            )
        rig.gateway.phase, rig.gateway.index = "observe", 0
        ordinary = await rig.send("Thanks for restoring it.")
        assert ordinary.terminal_status.value == "completed", rig.trace_failures
        assert (
            rig.store.get_message(ordinary.user_message_id).content
            == "Thanks for restoring it."
        )
        assert all(
            "Previously model-observed archive selection metadata" not in m.content
            for m in rig.store.messages_for_session(rig.source.id)
            if m.role is not ConsoleMessageRole.TOOL
        )
        assert (
            _captured_turn_evidence(rig.db, first_user.persisted_message_id) == previous
        )
        assert (
            _captured_turn_evidence(rig.db, current.persisted_message_id)
            == current_before
        )
        assert not rig.trace_failures
        assert resealed_requests
        assert [rig.db.get_message_by_id(row["id"]) for row in historical] == historical
    finally:
        ready.cancel()
        if not turn.done():
            rig.controller.stop_active_run(record_user_stop=True)
        await asyncio.gather(turn, ready, return_exceptions=True)


@pytest.mark.parametrize(
    "change",
    [None, "unknown", "missing", "content", "image", "extra", "role", "conflicting"],
)
def test_agent_factory_thinking_alias_keeps_only_exact_saved_owner(change):
    owner = new_opaque_id()
    saved = SavedRevisionTraceProvenance(new_opaque_id())
    current = SavedRevisionTraceProvenance(new_opaque_id())
    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
    historical = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Exact saved answer"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,aW1hZ2U="},
            },
        ],
        THINKING_OWNER_KEY: owner,
    }
    admitted = build_console_request(
        [historical, {"role": "user", "content": "Current saved user"}],
        message_provenance=(saved, current),
        memory_provenance=(),
        mandatory_provenance=(),
        tool_provenance=(),
        capture_policy=policy,
        thinking_groups=(
            ThinkingOwnerGroup(
                owner,
                (ResolvedThinkingBlock(owner, "start_anchored_think", "private"),),
            ),
        ),
    )
    row = thaw_json(historical)
    key = "_tldw_call_thinking_owner"
    row[key] = row.pop(THINKING_OWNER_KEY)
    if change == "unknown":
        row[key] = new_opaque_id()
    elif change == "missing":
        row.pop(key)
    elif change == "content":
        row["content"][0]["text"] = "Changed historical answer"
    elif change == "image":
        row["content"][1]["image_url"]["url"] = "data:image/png;base64,b3RoZXI="
    elif change == "extra":
        row["unadmitted_private_field"] = True
    elif change == "role":
        row["role"] = "user"
    elif change == "conflicting":
        row[THINKING_OWNER_KEY] = new_opaque_id()
    factory = ConsoleAgentTraceRequestFactory(admitted)
    kwargs = {
        "tools": (),
        "route": ConsoleRequestRoute.AGENT_FIRST,
        "actor_id": new_opaque_id(),
        "chain_id": new_opaque_id(),
        "thinking_owner_key": key,
    }
    messages = [row, thaw_json(admitted.active_request[0])]
    if change == "conflicting":
        with pytest.raises(ValueError, match="thinking group must attach"):
            factory.build(messages, **kwargs)
        return
    result = factory.build(messages, **kwargs)
    descriptor = result.provenance.flattened_messages()[0]
    assert (descriptor == saved) is (change is None)
    if change is not None:
        assert type(descriptor) is ProviderArtifactTraceProvenance
    assert thaw_json(result.flattened_messages()[0]) == row
    assert result.provenance.active_request == (current,)


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "markers", ["continuation", "matching", "conflicting", "history"]
)
async def test_archive_reseal_preserves_canonical_history_owners(
    captured_followup, markers
):
    rig = captured_followup
    await rig.send("Find my archived conversations.")
    current = next(
        message
        for message in rig.store.messages_for_session(rig.source.id)
        if message.role is ConsoleMessageRole.USER
    )
    admitted = rig.trace_requests[0]
    owner = new_opaque_id()
    historical = {
        "role": "assistant",
        "content": "answer",
        CONTINUATION_OWNER_KEY: owner,
    }
    thinking = ()
    if markers in {"matching", "conflicting"}:
        thinking_owner = new_opaque_id() if markers == "conflicting" else owner
        historical[THINKING_OWNER_KEY] = thinking_owner
        thinking = (
            ThinkingOwnerGroup(
                thinking_owner,
                (
                    ResolvedThinkingBlock(
                        thinking_owner, "start_anchored_think", "private"
                    ),
                ),
            ),
        )
    request = build_console_request(
        thaw_json(admitted.system)
        + [historical, thaw_json(admitted.active_request[0])],
        message_provenance=admitted.provenance.system
        + (
            ProviderArtifactTraceProvenance(
                TraceProvenanceSource.ACTIVE_REQUEST, admitted.provenance.capture_policy
            ),
        )
        + admitted.provenance.active_request,
        memory_provenance=(),
        mandatory_provenance=(),
        tool_provenance=(),
        metadata_provenance=admitted.provenance.metadata,
        capture_policy=admitted.provenance.capture_policy,
        continuation_groups=(_private_group(owner, call_id="compatibility-call"),),
        thinking_groups=thinking,
    )
    before = [
        {
            "role": "assistant",
            "content": "changed" if markers == "history" else "answer",
            "_native_message_id": owner,
        },
        thaw_json(request.active_request[0]),
    ]
    after = [dict(row) for row in before]
    after[-1]["content"] = (
        "Evidence: approved selection\n\n---\n\n" + before[-1]["content"]
    )
    previous = _captured_turn_evidence(rig.db, current.persisted_message_id)
    if markers in {"conflicting", "history"}:
        if markers == "conflicting":
            with pytest.raises(ValueError, match="owner markers disagree"):
                _flatten_preflight_messages(request)
        with pytest.raises(TraceCallPersistenceError) as error:
            rig.controller._reseal_archive_followup_trace_request(
                request, before, after, current.id
            )
        assert error.value.reservation_status == "not_established"
    else:
        resealed = rig.controller._reseal_archive_followup_trace_request(
            request, before, after, current.id
        )
        assert (
            replace(
                resealed,
                active_request=request.active_request,
                provenance=request.provenance,
            )
            == request
        )
        descriptor = resealed.provenance.active_request[0]
        assert descriptor.transform is TraceTransformKind.CURRENT_TURN_TEXT
        assert descriptor.inputs == admitted.provenance.active_request
    assert current.content == "Find my archived conversations."
    assert _captured_turn_evidence(rig.db, current.persisted_message_id) == previous


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_archive_rider_composes_existing_transform_without_changing_its_source(
    captured_followup,
):
    rig = captured_followup
    await rig.send("Find my archived conversations.")
    current = next(
        m
        for m in rig.store.messages_for_session(rig.source.id)
        if m.role is ConsoleMessageRole.USER
    )
    admitted = rig.trace_requests[0]
    policy = admitted.provenance.capture_policy
    saved = admitted.provenance.active_request[0]
    assert type(saved) is SavedRevisionTraceProvenance
    expanded = {
        "role": "user",
        "content": "Existing text expansion: Find my archived conversations.",
    }
    admitted = replace(
        admitted,
        active_request=(expanded,),
        capture_durability="durable",
        provenance=replace(
            admitted.provenance,
            active_request=(
                DerivedTraceProvenance(
                    TraceTransformKind.CURRENT_TURN_TEXT,
                    inputs=(saved,),
                    artifact=ProviderArtifactTraceProvenance(
                        TraceProvenanceSource.ACTIVE_REQUEST, policy
                    ),
                ),
            ),
        ),
    )
    before = [expanded]
    after = [
        {
            "role": "user",
            "content": "Evidence: approved selection\n\n---\n\n" + expanded["content"],
        }
    ]
    resealed = rig.controller._reseal_archive_followup_trace_request(
        admitted, before, after, current.id
    )
    descriptor = resealed.provenance.active_request[0]
    assert descriptor.transform is TraceTransformKind.CURRENT_TURN_TEXT
    assert (
        descriptor.inputs == (saved,)
        and type(descriptor.inputs[0]) is SavedRevisionTraceProvenance
    )
    assert descriptor.artifact.policy == policy
    assert resealed.active_request == tuple(after)
    assert (
        replace(
            resealed,
            active_request=admitted.active_request,
            provenance=admitted.provenance,
        )
        == admitted
    )
    rebuilt = ConsoleAgentTraceRequestFactory(resealed).build(
        thaw_json(resealed.flattened_messages()),
        tools=thaw_json(resealed.tools),
        route=ConsoleRequestRoute.AGENT_FIRST,
        actor_id=new_opaque_id(),
        chain_id=new_opaque_id(),
    )
    assert rebuilt.provenance.active_request[0] == descriptor
    assert current.content == "Find my archived conversations."


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid",
    [
        "unowned",
        "unknown",
        "foreign",
        "history",
        "role",
        "attachment",
        "sidecar",
        "missing_current",
    ],
)
async def test_archive_rider_reseal_refuses_unowned_or_non_text_changes(
    captured_followup, invalid
):
    rig = captured_followup
    await rig.send("Find my archived conversations.")
    current = next(
        m
        for m in rig.store.messages_for_session(rig.source.id)
        if m.role is ConsoleMessageRole.USER
    )
    request = rig.trace_requests[0]
    if invalid == "history":
        request = build_console_request(
            thaw_json(request.system)
            + [
                {"role": "assistant", "content": "Keep exact historical row"},
                thaw_json(request.active_request[0]),
            ],
            message_provenance=request.provenance.system
            + (
                ProviderArtifactTraceProvenance(
                    TraceProvenanceSource.ACTIVE_REQUEST,
                    request.provenance.capture_policy,
                ),
            )
            + request.provenance.active_request,
            memory_provenance=(),
            mandatory_provenance=(),
            tool_provenance=(),
            metadata_provenance=request.provenance.metadata,
            capture_policy=request.provenance.capture_policy,
        )
    before = thaw_json(request.flattened_messages()[1:])
    after = [dict(row) for row in before]
    after[-1]["content"] = (
        "Evidence: approved selection\n\n---\n\n" + before[-1]["content"]
    )
    if invalid in {"unowned", "unknown", "foreign"}:
        descriptor = ProviderArtifactTraceProvenance(
            TraceProvenanceSource.ACTIVE_REQUEST, request.provenance.capture_policy
        )
        if invalid == "unknown":
            descriptor = SavedRevisionTraceProvenance(new_opaque_id())
        elif invalid == "foreign":
            with rig.db.transaction() as cursor:
                revision = cursor.execute(
                    "SELECT revision_id FROM console_trace_semantic_revisions WHERE source_conversation_id != ? LIMIT 1",
                    (rig.source.persisted_conversation_id,),
                ).fetchone()[0]
            descriptor = SavedRevisionTraceProvenance(revision)
        request = replace(
            request,
            provenance=replace(request.provenance, active_request=(descriptor,)),
        )
    elif invalid == "history":
        # Changed prior guidance cannot disappear into a current-user reseal.
        after[0]["content"] = "Changed historical row"
    elif invalid == "role":
        after[-1]["role"] = "assistant"
    elif invalid == "attachment":
        after[-1]["content"] = [
            {"type": "image_url", "image_url": {"url": "data:changed"}}
        ]
    elif invalid == "sidecar":
        after[-1]["provider_continuation"] = {"changed": True}
    before_trace = _captured_turn_evidence(rig.db, current.persisted_message_id)
    with pytest.raises(TraceCallPersistenceError) as error:
        rig.controller._reseal_archive_followup_trace_request(
            request, before, after, None if invalid == "missing_current" else current.id
        )
    assert error.value.reservation_status == "not_established"
    assert _captured_turn_evidence(rig.db, current.persisted_message_id) == before_trace
    assert current.content == "Find my archived conversations."


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
@pytest.mark.parametrize("direct", [True, False])
async def test_new_native_turn_restores_earlier_second_row_after_exact_confirmation(
    native_followup, direct
):
    rig = native_followup
    rig.direct = direct
    await asyncio.wait_for(
        rig.send("Find my archived conversations."), _DISPATCH_WAIT_SECONDS
    )
    original = rig.gateway.original_page
    assert [row["conversation_id"] for row in original["items"]] == [
        rig.first,
        rig.second,
    ]
    rig.gateway.phase, rig.gateway.index = "restore", 0
    turn = asyncio.create_task(rig.send("Restore the second one"))
    ready = asyncio.create_task(rig.ready.wait())
    try:
        await asyncio.wait_for(
            asyncio.wait({turn, ready}, return_when=asyncio.FIRST_COMPLETED),
            _DISPATCH_WAIT_SECONDS,
        )
        context = rig.gateway.followup_context
        assert context is not None, (
            "Fresh native follow-up lost the earlier immutable result rows/refs"
        )
        assert context["source_session_id"] == rig.source.id
        assert context["archive_recovery_results"] == [
            {
                "result_set_id": original["result_set_id"],
                "items": [
                    {
                        "ordinal": index,
                        "result_ref": row["result_ref"],
                        "title": row["title"],
                        "workspace_label": row["workspace"]["name"]
                        if row["workspace"]
                        else "Default",
                        "allowed_scopes": row["allowed_scopes"],
                    }
                    for index, row in enumerate(original["items"], 1)
                ],
            }
        ]
        assert context["omitted_pages"] == 0
        assert "ARCHIVE_BODY_NOT_FOR_FOLLOWUP" in original["items"][0]["excerpt"]
        assert "ARCHIVE_BODY_NOT_FOR_FOLLOWUP" not in str(context)
        assert rig.cards and not turn.done()
        assert rig.cards[0].target.conversation_id == rig.second
        assert rig.db.get_conversation_by_id(rig.second)["archived"] == 1
        assert rig.app.workspace_registry_service.get_workspace("project").archived
        outcome = await rig.recovery.coordinator.decide(
            rig.source.id, rig.cards[0].request_id, allow=True, scope="chat_only"
        )
        assert outcome.chat_changed and not outcome.workspace_changed
        await asyncio.wait_for(turn, _DISPATCH_WAIT_SECONDS)
        assert rig.db.get_conversation_by_id(rig.second)["archived"] == 0
        assert rig.db.get_conversation_by_id(rig.first)["archived"] == 1
        assert rig.store.active_session_id == rig.source.id
    finally:
        ready.cancel()
        if not turn.done():
            await rig.controller.stop_active_run(session_id=rig.source.id)
        await asyncio.gather(turn, ready, return_exceptions=True)


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "restriction",
    [
        "blocked",
        "frozen_blocked",
        "legacy",
        "persona",
        "ask",
        "kill",
        "kill_error",
        "non_native",
        "unsupported",
        "temporary",
        "remote",
    ],
)
async def test_fresh_turn_gates_exclude_previously_observed_selection(
    native_followup, restriction
):
    rig = native_followup
    await rig.send("Find my archived conversations.")
    original = rig.gateway.original_page
    rig.restriction = restriction
    if restriction in {"persona", "ask"}:
        # Mutating live choices during resolution cannot replace frozen rules.
        rig.gateway.on_resolution = lambda: setattr(rig, "restriction", None)
    if restriction == "blocked":
        result = rig.persistence.console_library_policy_repository.compare_and_swap(
            rig.source.persisted_conversation_id, 1, _allowed_candidate(allowed=False)
        )
        assert result.snapshot.assistant_access.value == "blocked"
    elif restriction in {"kill", "kill_error"}:

        def read_kill():
            if restriction == "kill_error":
                raise RuntimeError("injected unreadable switch")
            return True

        rig.controller._console_tool_kill_switch_reader = lambda: read_kill
    elif restriction == "temporary":
        rig.source.ephemeral = True
    elif restriction == "remote":
        rig.source.runtime_backend = "server"
    offset = len(rig.gateway.messages_seen)
    rig.gateway.phase, rig.gateway.index = "observe", 0
    await asyncio.wait_for(rig.send("Restore the second one"), _DISPATCH_WAIT_SECONDS)
    assert rig.gateway.followup_context is None
    outgoing = str(rig.gateway.messages_seen[offset:])
    assert all(row["result_ref"] not in outgoing for row in original["items"])
    assert not rig.cards
    assert rig.db.get_conversation_by_id(rig.second)["archived"] == 1


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", ["model", "provider", "endpoint", "egress", "unknown"]
)
async def test_fresh_destination_cannot_reuse_another_destinations_rows(
    native_followup, change
):
    rig = native_followup
    await rig.send("Find my archived conversations.")
    original = rig.gateway.original_page
    rig.gateway.destination_changes = {
        "model": {"model": "another-model"},
        "provider": {"provider": "another-provider"},
        "endpoint": {"endpoint_identity": "http://127.0.0.1:9100"},
        "egress": {"egress_class": ConsoleEgressClass.PRIVATE_NETWORK},
        "unknown": {"egress_class": ConsoleEgressClass.UNKNOWN},
    }[change]
    offset = len(rig.gateway.messages_seen)
    rig.gateway.phase, rig.gateway.index = "observe", 0
    await asyncio.wait_for(rig.send("Restore the second one"), _DISPATCH_WAIT_SECONDS)
    context = rig.gateway.followup_context
    assert not context or (
        context["archive_recovery_results"] == [] and context["omitted_pages"] == 1
    )
    outgoing = str(rig.gateway.messages_seen[offset:])
    assert all(row["result_ref"] not in outgoing for row in original["items"])
    assert not rig.cards


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "loss",
    [
        "manual",
        "missing_origin",
        "expired",
        "evicted",
        "closed",
        "foreign_source",
        "imported",
    ],
)
async def test_live_registry_loss_never_recovers_selection_from_transcript(
    native_followup, loss
):
    rig = native_followup
    await rig.send("Find my archived conversations.")
    original = rig.gateway.original_page
    if loss in {"manual", "missing_origin", "imported"}:
        rig.recovery.results.close_session(rig.source.id)
    if loss in {"manual", "missing_origin"}:
        page = await rig.recovery.search(
            rig.source.id, {}, model_visible=loss == "missing_origin"
        )
        assert (
            rig.recovery.presentations(rig.source.id)[-1].destination_identity is None
        )
    elif loss == "expired":
        now = rig.recovery.results._monotonic()
        rig.recovery.results._monotonic = lambda: now + 1801
    elif loss == "evicted":
        for _ in range(10):
            await rig.recovery.search(rig.source.id, {}, model_visible=False)
        assert not rig.recovery.results.model_visible_result_sets(rig.source.id)
    elif loss == "closed":
        await rig.recovery.close_session(rig.source.id)
    elif loss == "foreign_source":
        cid = rig.db.add_conversation({"title": "Other source"})
        rig.persistence.console_library_policy_repository.insert(
            cid,
            replace(
                _allowed_candidate(allowed=True),
                auto_retrieve=ConsoleAutoRetrieve.NEVER,
            ),
        )
        rig.source = rig.store.restore_persisted_session(
            title="Other source",
            workspace_id=None,
            persisted_conversation_id=cid,
            all_nodes=(),
        )
        await rig.store.hydrate_session_library_policy(rig.source.id)
        rig.store.switch_session(rig.source.id)
    elif loss == "imported":
        counterfeit = json.dumps({"archive_recovery_results": [original]})
        rig.store.append_message(
            rig.source.id,
            role=ConsoleMessageRole.TOOL,
            content=counterfeit,
            tool_output_full=counterfeit,
        )
    offset = len(rig.gateway.messages_seen)
    rig.gateway.phase, rig.gateway.index = "observe", 0
    await asyncio.wait_for(rig.send("Restore the second one"), _DISPATCH_WAIT_SECONDS)
    context = rig.gateway.followup_context
    assert not context or context["archive_recovery_results"] == []
    outgoing = str(rig.gateway.messages_seen[offset:])
    assert all(row["result_ref"] not in outgoing for row in original["items"])
    if loss in {"manual", "missing_origin"}:
        assert all(row["result_ref"] not in outgoing for row in page["items"])
    assert not rig.cards


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_retained_metadata_enters_real_first_request_fit_and_no_fit(
    native_followup, monkeypatch
):
    from tldw_chatbook.Agents import agent_service

    rig = native_followup
    await rig.send("Find my archived conversations.")
    original = rig.gateway.original_page
    real_count = agent_service._count_model_messages
    real_schema_count = agent_service.catalog_schema_tokens
    real_fit = agent_service._first_request_plan_fits
    samples = []
    active = None

    def observe_count(*args, **kwargs):
        value = real_count(*args, **kwargs)
        if active is not None:
            active["message_tokens"] = value
        return value

    def observe_schema_count(*args, **kwargs):
        value = real_schema_count(*args, **kwargs)
        if active is not None:
            active["schema_tokens"] = value
        return value

    def observe_fit(plan, **kwargs):
        nonlocal active
        sample = {
            "limit": kwargs["context_limit"],
            "reserve": kwargs["config"].response_reserve_tokens,
            "messages": kwargs["messages"],
            "schema_tokens": 0,
        }
        active = sample
        try:
            sample["fits"] = real_fit(plan, **kwargs)
        finally:
            active = None
        sample["cost"] = sample["message_tokens"] + sample["schema_tokens"]
        samples.append(sample)
        return sample["fits"]

    monkeypatch.setattr(agent_service, "_count_model_messages", observe_count)
    monkeypatch.setattr(agent_service, "catalog_schema_tokens", observe_schema_count)
    monkeypatch.setattr(agent_service, "_first_request_plan_fits", observe_fit)
    rig.gateway.phase, rig.gateway.index = "observe", 0
    await rig.send("Restore the second one")
    baseline = samples[-1]
    assert baseline["fits"]
    # Leave only 64 tokens beyond the actual measured full first-request cost.
    # This freezes an external resolution budget, without faking any estimator.
    rig.gateway.response_reserve = baseline["limit"] - baseline["cost"] - 64
    samples.clear()
    rig.gateway.index = 0
    await rig.send("Restore the second one")
    fitted = samples[-1]
    assert fitted["fits"]
    assert 0 <= fitted["limit"] - fitted["reserve"] - fitted["cost"] <= 64
    for sample in samples:
        assert original["items"][1]["result_ref"] in str(sample["messages"])
    assert (
        rig.gateway.followup_context["archive_recovery_results"][0]["result_set_id"]
        == original["result_set_id"]
    )
    samples.clear()
    before = rig.gateway.stream_calls
    rig.gateway.response_reserve = baseline["limit"]
    rig.gateway.index = 0
    await rig.send("Restore the second one")
    assert samples and not any(sample["fits"] for sample in samples)
    assert all(
        original["items"][1]["result_ref"] in str(sample["messages"])
        for sample in samples
    )
    assert rig.gateway.stream_calls == before
    assert any(
        "first request exceeds model context budget" in message.content
        for message in rig.store.messages_for_session(rig.source.id)
    )
    assert not rig.cards
    assert rig.db.get_conversation_by_id(rig.second)["archived"] == 1


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
@pytest.mark.parametrize("large", [False, True])
async def test_multiple_native_cards_preserve_pages_and_bound_complete_evidence(
    native_followup, large
):
    rig = native_followup
    if large:
        with rig.db.transaction() as cursor:
            cursor.execute(
                "UPDATE conversations SET title = ? WHERE id IN (?, ?)",
                ("不可信" * 1700, rig.first, rig.second),
            )
        for _ in range(8):
            cid = rig.db.add_conversation({"title": "不可信" * 1700})
            rig.db.set_conversations_archived(
                [cid], archived=True, expected_versions={cid: 1}
            )
    else:
        missing = rig.db.add_conversation(
            {
                "title": "Missing workspace",
                "scope_type": "workspace",
                "workspace_id": "gone",
            }
        )
        rig.db.set_conversations_archived(
            [missing], archived=True, expected_versions={missing: 1}
        )
        with rig.db.transaction() as cursor:
            cursor.execute(
                "UPDATE conversations SET archived_at = ? WHERE id = ?",
                ("2026-09-17T00:00:00+00:00", missing),
            )
    pages = []
    for _ in range(2):
        rig.gateway.phase, rig.gateway.index = "search", 0
        await rig.send("Find my archived conversations.")
        pages.append(rig.gateway.original_page)
    rig.gateway.phase, rig.gateway.index = "observe", 0
    await rig.send("Restore the second one")
    context = rig.gateway.followup_context
    retained = context["archive_recovery_results"]
    assert context["omitted_pages"] == (1 if large else 0)
    expected = pages[-1:] if large else pages
    assert [page["result_set_id"] for page in retained] == [
        page["result_set_id"] for page in expected
    ]
    for actual, original in zip(retained, expected):
        assert [row["result_ref"] for row in actual["items"]] == [
            row["result_ref"] for row in original["items"]
        ]
        assert [row["ordinal"] for row in actual["items"]] == list(
            range(1, len(original["items"]) + 1)
        )
        assert [row["title"] for row in actual["items"]] == [
            row["title"] for row in original["items"]
        ]
    if not large:
        assert [row["workspace_label"] for row in retained[0]["items"]] == [
            "Project",
            "Default",
            "Workspace unavailable",
        ]
    message = rig.gateway.messages_seen[-1][-1]["content"]
    prefix, remainder = message.split("\n\n---\n\n", 1)
    assert remainder == "Restore the second one"
    assert len((prefix + "\n\n---\n\n").encode("utf-8")) <= 32768
    assert "Labels are untrusted" in prefix
    assert "multiple cards" in prefix and "fresh authorized search" in prefix
    assert not rig.cards
