"""Immutable owning-session turn-context contracts for Console sends."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.attachment_core import PendingAttachment
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleProviderSelection,
    ConsoleStagedSource,
    ConsoleWorkspaceContext,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore as _ConsoleChatStore
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleEgressClass,
    ConsoleLibraryItemScopeSnapshot,
    ConsoleProviderIntent,
    ConsoleResolvedDestination,
    ConsoleTurnLibraryAuthority,
)
from tldw_chatbook.Chat.console_library_policy import (
    AUTOMATIC_LIBRARY_SOURCE_TYPES,
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicySnapshot,
)
from tldw_chatbook.Chat.console_scratch_space import ConsoleScratchSpaceManager
from tldw_chatbook.Chat.console_turn_context import (
    ConsoleTurnConfigurationSnapshot,
    ConsoleTurnExecutionContext,
)
from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController
from tldw_chatbook.Workspaces import SkippedReviewRoot


class ConsoleChatStore(_ConsoleChatStore):
    """Test store whose intentionally db-less sessions are explicitly ephemeral."""

    def create_session(self, **kwargs):
        kwargs.setdefault("ephemeral", self.persistence is None)
        return super().create_session(**kwargs)


class _PausedGateway:
    def __init__(self) -> None:
        self.resolve_started = asyncio.Event()
        self.release_resolve = asyncio.Event()
        self.selections: list[ConsoleProviderSelection] = []
        self.message_batches: list[list[dict[str, object]]] = []

    async def resolve_for_send(self, selection: ConsoleProviderSelection):
        self.selections.append(selection)
        self.resolve_started.set()
        await self.release_resolve.wait()
        model = selection.explicit_model or selection.configured_model or ""
        return SimpleNamespace(
            ready=True,
            provider=selection.provider,
            model=model,
            base_url=selection.base_url,
            max_tokens=selection.max_tokens,
            visible_copy="",
            resolved_destination=ConsoleResolvedDestination(
                provider=selection.provider,
                model=model,
                endpoint_identity="https://api.openai.com",
                egress_class=ConsoleEgressClass.PUBLIC_NETWORK,
            ),
        )

    async def stream_chat(self, resolution, messages, **_kwargs):
        self.message_batches.append(messages)
        yield "reply"


def _settings(
    provider: str,
    model: str,
    system_prompt: str,
    *,
    temperature: float = 0.25,
    max_tokens: int = 321,
) -> ConsoleSessionSettings:
    return ConsoleSessionSettings(
        provider=provider,
        model=model,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def test_capture_detaches_nested_mutable_configuration_sources():
    staged_sources = [
        ConsoleStagedSource(
            source_id="source-1",
            label="Source one",
            source_type="note",
            workspace_id="workspace-a",
        )
    ]
    workspace = ConsoleWorkspaceContext(
        active_workspace_id="workspace-a",
        staged_sources=tuple(staged_sources),
        active_run_id="run-1",
    )
    selection = ConsoleProviderSelection(
        provider="openai",
        explicit_model="gpt-context",
        system_prompt="system-a",
        workspace_context=workspace,
    )
    roots = ["C:/workspace/a"]
    review_aliases = ["folder-a"]
    capabilities = {"vision": True, "formats": ["image/png"]}
    rag_defaults = {"enabled": True, "scope": {"types": ["notes"]}}
    tool_configuration = {"local": {"enabled": True, "names": ["fs_read"]}}
    payload_settings = {"headers": {"x-mode": "one"}, "stops": ["END"]}

    context = ConsoleTurnConfigurationSnapshot.capture(
        session_id="session-a",
        provider_selection=selection,
        session_settings=_settings("openai", "gpt-context", "system-a"),
        workspace_roots=roots,
        change_review_root_aliases=review_aliases,
        capabilities=capabilities,
        rag_defaults=rag_defaults,
        tool_configuration=tool_configuration,
        provider_payload_settings=payload_settings,
    )

    roots.append("C:/workspace/leak")
    review_aliases.append("folder-leak")
    capabilities["formats"].append("image/jpeg")
    rag_defaults["scope"]["types"].append("media")
    tool_configuration["local"]["names"].append("fs_write")
    payload_settings["headers"]["x-mode"] = "two"
    staged_sources.clear()

    assert context.workspace_roots == ("C:/workspace/a",)
    assert context.change_review_root_aliases == ("folder-a",)
    assert context.capabilities["formats"] == ("image/png",)
    assert context.rag_defaults["scope"]["types"] == ("notes",)
    assert context.tool_configuration["local"]["names"] == ("fs_read",)
    assert context.provider_payload_settings["headers"]["x-mode"] == "one"
    assert context.provider_selection.workspace_context.staged_sources == (
        ConsoleStagedSource(
            source_id="source-1",
            label="Source one",
            source_type="note",
            workspace_id="workspace-a",
        ),
    )

    with pytest.raises(TypeError):
        context.rag_defaults["enabled"] = False

    for forbidden in (
        "credentials",
        "approval_grants",
        "skill_trust",
        "cancel_event",
    ):
        assert not hasattr(context, forbidden)


def test_direct_constructor_also_detaches_mutable_inputs():
    capabilities = {"formats": ["image/png"]}
    context = ConsoleTurnConfigurationSnapshot(
        session_id="session-a",
        provider_selection=ConsoleProviderSelection(provider="openai"),
        capabilities=capabilities,
    )

    capabilities["formats"].append("image/jpeg")

    assert context.capabilities["formats"] == ("image/png",)


def _authority() -> ConsoleTurnLibraryAuthority:
    return ConsoleTurnLibraryAuthority(
        policy=ConsoleLibraryPolicySnapshot(
            auto_retrieve=ConsoleAutoRetrieve.NEVER,
            assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
            policy_revision=4,
            source="durable",
        ),
        direct_library_tools=True,
        source_types=AUTOMATIC_LIBRARY_SOURCE_TYPES,
        scope_snapshot=ConsoleLibraryItemScopeSnapshot((), (), True),
        provider_intent=ConsoleProviderIntent("openai", "gpt-context", None),
        attempt_id="attempt-1",
    )


def _destination() -> ConsoleResolvedDestination:
    return ConsoleResolvedDestination(
        provider="openai",
        model="gpt-context",
        endpoint_identity="https://api.example.invalid/v1",
        egress_class=ConsoleEgressClass.UNKNOWN,
    )


def test_final_context_requires_complete_authority_and_destination():
    configuration = ConsoleTurnConfigurationSnapshot.capture(
        session_id="session-a",
        provider_selection=ConsoleProviderSelection(provider="openai"),
    )

    with pytest.raises(TypeError, match="library_authority"):
        ConsoleTurnExecutionContext(
            configuration=configuration,
            library_authority=None,
            resolved_destination=_destination(),
        )
    with pytest.raises(TypeError, match="resolved_destination"):
        ConsoleTurnExecutionContext(
            configuration=configuration,
            library_authority=_authority(),
            resolved_destination=None,
        )


def test_final_context_exposes_read_only_configuration_compatibility_properties():
    configuration = ConsoleTurnConfigurationSnapshot.capture(
        session_id="session-a",
        provider_selection=ConsoleProviderSelection(
            provider="openai", configured_model="gpt-context"
        ),
        capabilities={"vision": True},
        rag_defaults={"top_k": 5},
        tool_configuration={"direct_library_tools": True},
        provider_payload_settings={"temperature": 0.2},
    )
    context = ConsoleTurnExecutionContext(
        configuration=configuration,
        library_authority=_authority(),
        resolved_destination=_destination(),
    )

    assert context.session_id == "session-a"
    assert context.effective_model == "gpt-context"
    assert context.provider_selection.provider == "openai"
    assert context.capabilities == {"vision": True}
    assert context.rag_defaults == {"top_k": 5}
    assert context.tool_configuration == {"direct_library_tools": True}
    assert context.provider_payload_settings == {"temperature": 0.2}
    with pytest.raises(AttributeError):
        context.configuration = configuration


def test_live_tool_kill_switch_is_not_frozen_into_turn_context():
    store = ConsoleChatStore()
    session = store.create_session(workspace_id="workspace-a")
    context = ConsoleTurnConfigurationSnapshot.capture(
        session_id=session.id,
        provider_selection=ConsoleProviderSelection(provider="openai"),
        tool_configuration={"local_tools_enabled": True},
    )
    service = SimpleNamespace(get_kill_switch=lambda: True)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=_PausedGateway(),
    )
    controller.app = SimpleNamespace(unified_mcp_service=service)

    provider, review_hook = controller._compose_local_provider(
        session_id=session.id,
        turn_context=context,
    )

    assert provider is None
    assert review_hook is None


def test_legacy_session_without_settings_still_uses_own_workspace(tmp_path):
    store = ConsoleChatStore()
    first = store.create_session(workspace_id="workspace-a")
    store.create_session(workspace_id="workspace-b")
    store.set_workspace_context(
        ConsoleWorkspaceContext(active_workspace_id="workspace-b")
    )
    scratch_spaces = ConsoleScratchSpaceManager(temp_parent=tmp_path)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=_PausedGateway(),
        provider="anthropic",
        model="model-b",
        scratch_spaces=scratch_spaces,
    )

    context = controller.resolve_turn_execution_context(first.id)

    assert context.provider_selection.workspace_context.active_workspace_id == (
        "workspace-a"
    )
    assert scratch_spaces.dispose()


def test_turn_context_captures_frozen_scratch_snapshot(tmp_path):
    store = ConsoleChatStore()
    session = store.create_session(workspace_id="workspace-a")
    scratch_spaces = ConsoleScratchSpaceManager(temp_parent=tmp_path)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=_PausedGateway(),
        scratch_spaces=scratch_spaces,
    )

    context = controller.resolve_turn_execution_context(session.id)

    assert context.scratch_space == scratch_spaces.snapshot(session.id)
    assert context.scratch_space.root.is_dir()
    assert scratch_spaces.dispose()


def test_two_live_sessions_for_same_saved_conversation_get_distinct_scratch(
    tmp_path,
):
    store = ConsoleChatStore()
    first = store.create_session(workspace_id="workspace-a")
    second = store.create_session(workspace_id="workspace-a")
    first.persisted_conversation_id = "saved-conversation"
    second.persisted_conversation_id = "saved-conversation"
    scratch_spaces = ConsoleScratchSpaceManager(temp_parent=tmp_path)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=_PausedGateway(),
        scratch_spaces=scratch_spaces,
    )

    first_context = controller.resolve_turn_execution_context(first.id)
    second_context = controller.resolve_turn_execution_context(second.id)

    assert first_context.scratch_space.root != second_context.scratch_space.root
    assert first_context.scratch_space.token != second_context.scratch_space.token
    assert scratch_spaces.dispose()


def test_fallback_turn_context_does_not_capture_configured_workspace_root(
    monkeypatch,
    tmp_path,
):
    """The legacy tool confinement root is not Change Review consent."""
    store = ConsoleChatStore()
    session = store.create_session(workspace_id="workspace-a")
    scratch_spaces = ConsoleScratchSpaceManager(temp_parent=tmp_path)

    def setting(section, key, default=None):
        if section == "console" and key == "workspace_root":
            return "/configured/root"
        return default

    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_chat_controller.get_cli_setting",
        setting,
    )
    controller = ConsoleChatController(
        store=store,
        provider_gateway=_PausedGateway(),
        scratch_spaces=scratch_spaces,
    )

    context = controller.resolve_turn_execution_context(session.id)

    assert context.workspace_roots == ()
    assert "workspace_root" not in context.tool_configuration
    assert scratch_spaces.dispose()


@pytest.mark.asyncio
async def test_compatibility_controller_disposes_its_owned_scratch_space():
    store = ConsoleChatStore()
    session = store.create_session()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=_PausedGateway(),
    )
    snapshot = controller.resolve_turn_execution_context(session.id).scratch_space

    await controller.shutdown()

    assert snapshot is not None
    assert not snapshot.root.exists()


def test_session_builder_captures_roots_rag_tools_and_generation(
    monkeypatch,
    tmp_path,
):
    store = ConsoleChatStore()
    settings = _settings(
        "openai",
        "model-a",
        "system-a",
        temperature=0.4,
        max_tokens=777,
    )
    session = store.create_session(
        workspace_id="workspace-a",
        settings=settings,
    )
    selection = ConsoleProviderSelection(
        provider="openai",
        explicit_model="model-a",
        temperature=0.4,
        max_tokens=777,
        system_prompt="system-a",
        workspace_context=ConsoleWorkspaceContext(active_workspace_id="workspace-a"),
    )
    app_config = {
        "chat_defaults": {"rag_auto_retrieve_on_send": "true"},
        "console": {
            "agent_runtime": "true",
            "native_tool_calls": "false",
            "local_tools_enabled": "true",
            "workspace_root": "C:/configured-root",
            "direct_library_tools": "false",
        },
    }
    roots = [str(Path("C:/workspace/a"))]
    skipped = [
        SkippedReviewRoot(
            alias="folder-preparing",
            reason="Preparing change history",
        )
    ]
    admissions = 0

    class ConsentService:
        def admit_turn(self, workspace_id):
            nonlocal admissions
            admissions += 1
            assert workspace_id == "workspace-a"
            return SimpleNamespace(
                ready_roots=roots,
                ready_aliases=["folder-ready"],
                skipped_roots=skipped,
            )

    controller = ConsoleSessionController.__new__(ConsoleSessionController)
    controller.app_instance = SimpleNamespace(
        change_review_consent_service=ConsentService()
    )
    controller._provider_readiness_app_config_fn = lambda: app_config
    controller._build_provider_selection_fn = lambda _session_id: selection
    controller._current_chat_store_accessor = lambda: store
    controller._chat_store_accessor = lambda: store
    controller._rag_source_types_accessor = lambda: ["notes", "media"]
    controller._rag_top_k_accessor = lambda: 7
    scratch_spaces = ConsoleScratchSpaceManager(temp_parent=tmp_path)
    scratch_snapshot = scratch_spaces.snapshot(session.id)
    controller._scratch_snapshot_provider = lambda _session_id: scratch_snapshot

    context = controller._build_console_turn_execution_context(session.id)
    roots.append(str(Path("C:/workspace/leak")))
    skipped.append(SkippedReviewRoot(alias="folder-leak", reason="leak"))
    app_config["console"]["agent_runtime"] = "false"
    app_config["chat_defaults"]["rag_auto_retrieve_on_send"] = "false"

    assert context.workspace_roots == (str(Path("C:/workspace/a")),)
    assert context.change_review_root_aliases == ("folder-ready",)
    assert context.change_review_skipped_roots == (
        SkippedReviewRoot(
            alias="folder-preparing",
            reason="Preparing change history",
        ),
    )
    assert admissions == 1
    assert context.scratch_space is scratch_snapshot
    assert context.rag_defaults == {
        "source_types": ("notes", "media"),
        "top_k": 7,
    }
    assert context.tool_configuration["agent_runtime_enabled"] is True
    assert context.tool_configuration["native_tool_calls_enabled"] is False
    assert context.tool_configuration["local_tools_enabled"] is True
    assert "workspace_root" not in context.tool_configuration
    assert context.tool_configuration["direct_library_tools"] is False
    assert context.provider_payload_settings["temperature"] == 0.4
    assert context.provider_payload_settings["max_tokens"] == 777
    assert scratch_spaces.dispose()


def test_session_builder_without_consent_service_has_no_review_fallback(
    monkeypatch,
):
    """Turn capture never falls back to registry/CWD review roots."""
    store = ConsoleChatStore()
    session = store.create_session(workspace_id="workspace-a")
    selection = ConsoleProviderSelection(
        provider="openai",
        explicit_model="model-a",
        workspace_context=ConsoleWorkspaceContext(active_workspace_id="workspace-a"),
    )
    monkeypatch.setattr(
        "tldw_chatbook.Tools.workspace_file_roots.folder_binding_roots",
        lambda _workspace_id: pytest.fail("legacy root fallback was called"),
    )
    controller = ConsoleSessionController.__new__(ConsoleSessionController)
    controller.app_instance = SimpleNamespace()
    controller._provider_readiness_app_config_fn = lambda: {}
    controller._build_provider_selection_fn = lambda _session_id: selection
    controller._current_chat_store_accessor = lambda: store
    controller._chat_store_accessor = lambda: store
    controller._rag_source_types_accessor = lambda: []
    controller._rag_top_k_accessor = lambda: 4
    controller._scratch_snapshot_provider = lambda _session_id: None

    context = controller._build_console_turn_execution_context(session.id)

    assert context.workspace_roots == ()
    assert context.change_review_skipped_roots == ()


@pytest.mark.asyncio
async def test_submit_keeps_owning_session_selection_and_payload_across_switch():
    store = ConsoleChatStore()
    first = store.create_session(
        title="First",
        workspace_id="workspace-a",
        settings=_settings("openai", "model-a", "system-a"),
    )
    second = store.create_session(
        title="Second",
        workspace_id="workspace-b",
        settings=_settings("anthropic", "model-b", "system-b"),
    )
    gateway = _PausedGateway()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="anthropic",
        model="model-b",
        system_prompt="system-b",
        agent_runtime_enabled=False,
    )

    first_turn = asyncio.create_task(
        controller.submit_draft("question-a", session_id=first.id)
    )
    await gateway.resolve_started.wait()

    controller.switch_session(second.id)
    controller.update_provider_selection(
        ConsoleProviderSelection(
            provider="anthropic",
            explicit_model="model-b-new",
            system_prompt="system-b-new",
            workspace_context=ConsoleWorkspaceContext(
                active_workspace_id="workspace-b"
            ),
        )
    )
    gateway.release_resolve.set()

    result = await first_turn

    assert result.accepted is True
    assert gateway.selections[0].provider == "openai"
    assert gateway.selections[0].explicit_model == "model-a"
    assert gateway.selections[0].temperature == 0.25
    assert gateway.selections[0].max_tokens == 321
    assert gateway.selections[0].workspace_context.active_workspace_id == "workspace-a"
    assert gateway.message_batches[0][0] == {
        "role": "system",
        "content": "system-a",
    }
    assert gateway.message_batches[0][1]["content"] == "question-a"


@pytest.mark.asyncio
async def test_next_turn_observes_settings_replaced_after_prior_capture():
    store = ConsoleChatStore()
    session = store.create_session(
        title="Session",
        workspace_id="workspace-a",
        settings=_settings("openai", "model-a", "system-a"),
    )
    gateway = _PausedGateway()
    gateway.release_resolve.set()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="openai",
        model="model-a",
        system_prompt="system-a",
        agent_runtime_enabled=False,
    )

    first_result = await controller.submit_draft("first", session_id=session.id)
    assert first_result.accepted is True

    store.replace_session_settings(
        session.id,
        _settings(
            "anthropic",
            "model-b",
            "system-b",
            temperature=0.85,
            max_tokens=654,
        ),
    )
    second_result = await controller.submit_draft("second", session_id=session.id)

    assert second_result.accepted is True
    assert [selection.provider for selection in gateway.selections] == [
        "openai",
        "anthropic",
    ]
    assert gateway.message_batches[0][0]["content"] == "system-a"
    assert gateway.message_batches[1][0]["content"] == "system-b"
    assert gateway.selections[1].temperature == 0.85
    assert gateway.selections[1].max_tokens == 654


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action_name",
    ["retry", "continue", "regenerate", "edit-resend"],
)
async def test_message_actions_thread_one_captured_context(action_name: str):
    store = ConsoleChatStore()
    session = store.create_session(
        title="Actions",
        workspace_id="workspace-a",
        settings=_settings("openai", "model-a", "stored-system"),
    )
    user = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="question",
    )
    if action_name == "retry":
        assistant = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="",
        )
        store.append_stream_chunk(assistant.id, "failed answer")
        store.mark_message_failed(assistant.id)
    else:
        assistant = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
        )

    context = ConsoleTurnConfigurationSnapshot.capture(
        session_id=session.id,
        provider_selection=ConsoleProviderSelection(
            provider="openai",
            explicit_model="captured-model",
            system_prompt="captured-system",
            workspace_context=ConsoleWorkspaceContext(
                active_workspace_id="workspace-a"
            ),
        ),
        session_settings=store.session_settings(session.id),
        tool_configuration={"agent_runtime_enabled": False},
    )
    events: list[str] = []
    context_calls: list[str] = []

    def resolve_context(session_id: str) -> ConsoleTurnConfigurationSnapshot:
        events.append("configuration")
        context_calls.append(session_id)
        return context

    class UnavailableCoordinator:
        async def capture_for_execution(self, captured_session_id: str):
            assert captured_session_id == session.id
            events.append("policy")
            raise RuntimeError("durable policy unavailable")

    class ActionGateway(_PausedGateway):
        async def resolve_for_send(self, selection: ConsoleProviderSelection):
            events.append("gateway")
            resolution = await super().resolve_for_send(selection)
            resolution.resolved_destination = _destination()
            return resolution

    store.library_policy_coordinator = UnavailableCoordinator()
    gateway = ActionGateway()
    gateway.release_resolve.set()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="anthropic",
        model="mutable-model",
        system_prompt="mutable-system",
        agent_runtime_enabled=False,
        turn_context_provider=resolve_context,
    )
    observed_contexts: list[ConsoleTurnExecutionContext] = []
    real_inner = controller._stream_assistant_response_inner

    async def assert_complete_provider_boundary(**kwargs):
        events.append("provider-boundary")
        turn_context = kwargs["turn_context"]
        assert isinstance(turn_context, ConsoleTurnExecutionContext)
        observed_contexts.append(turn_context)
        return await real_inner(**kwargs)

    controller._stream_assistant_response_inner = assert_complete_provider_boundary

    if action_name == "retry":
        result = await controller.retry_message(assistant.id)
    elif action_name == "continue":
        result = await controller.continue_from_message(assistant.id)
    elif action_name == "regenerate":
        result = await controller.regenerate_message(assistant.id)
    else:
        result = await controller.edit_and_resend_message(user.id, "edited")

    assert result.accepted is True
    assert context_calls == [session.id]
    assert gateway.selections == [context.provider_selection]
    assert events == ["configuration", "policy", "gateway", "provider-boundary"]
    assert len(observed_contexts) == 1
    turn_context = observed_contexts[0]
    assert turn_context.resolved_destination == _destination()
    assert turn_context.library_authority.policy == ConsoleLibraryPolicySnapshot(
        auto_retrieve=ConsoleAutoRetrieve.NEVER,
        assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
        policy_revision=None,
        source="unavailable",
        error_code="policy_read_error",
    )
    assert gateway.message_batches[0][0] == {
        "role": "system",
        "content": "captured-system",
    }


@pytest.mark.asyncio
async def test_summarize_and_rag_capture_receive_the_owning_turn_context():
    store = ConsoleChatStore()
    session = store.create_session(
        title="Summary",
        workspace_id="workspace-a",
        settings=_settings("openai", "model-a", "stored-system"),
    )
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="first question",
    )
    store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="first answer",
    )
    boundary = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="second question",
    )
    context = ConsoleTurnConfigurationSnapshot.capture(
        session_id=session.id,
        provider_selection=ConsoleProviderSelection(
            provider="openai",
            explicit_model="captured-model",
            system_prompt="captured-system",
            workspace_context=ConsoleWorkspaceContext(
                active_workspace_id="workspace-a"
            ),
        ),
        session_settings=store.session_settings(session.id),
        rag_defaults={"auto_retrieve_on_send": False},
        tool_configuration={"agent_runtime_enabled": False},
    )
    gateway = _PausedGateway()
    gateway.release_resolve.set()
    rag_contexts: list[ConsoleTurnExecutionContext | None] = []

    async def capture_rag(
        _draft: str, turn_context: ConsoleTurnExecutionContext | None
    ):
        rag_contexts.append(turn_context)
        return SimpleNamespace(context=None)

    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        agent_runtime_enabled=False,
        turn_context_provider=lambda _session_id: context,
        rag_capture_provider=capture_rag,
    )

    summarize_result = await controller.summarize_up_to(boundary.id)
    submit_result = await controller.submit_draft("third", session_id=session.id)

    assert summarize_result.accepted is True
    assert submit_result.accepted is True
    assert gateway.selections == [
        context.provider_selection,
        context.provider_selection,
    ]
    assert len(rag_contexts) == 1
    assert rag_contexts[0] is not None
    assert rag_contexts[0].configuration == context


@pytest.mark.asyncio
async def test_attachment_gate_and_payload_use_captured_capabilities():
    store = ConsoleChatStore()
    session = store.create_session(
        title="Vision",
        workspace_id="workspace-a",
        settings=_settings("custom", "unknown-model", "system"),
    )
    store.add_pending_attachment(
        session.id,
        PendingAttachment(
            file_path="image.png",
            display_name="image.png",
            file_type="image",
            insert_mode="attachment",
            data=b"png",
            mime_type="image/png",
        ),
    )
    context = ConsoleTurnConfigurationSnapshot.capture(
        session_id=session.id,
        provider_selection=ConsoleProviderSelection(
            provider="custom",
            explicit_model="unknown-model",
            workspace_context=ConsoleWorkspaceContext(
                active_workspace_id="workspace-a"
            ),
        ),
        session_settings=store.session_settings(session.id),
        capabilities={"vision": True, "max_history_images": 1},
        tool_configuration={"agent_runtime_enabled": False},
    )
    gateway = _PausedGateway()
    gateway.release_resolve.set()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="custom",
        model="mutable-nonvision-model",
        agent_runtime_enabled=False,
        turn_context_provider=lambda _session_id: context,
    )

    result = await controller.submit_draft("describe", session_id=session.id)

    assert result.accepted is True
    content = gateway.message_batches[0][0]["content"]
    assert isinstance(content, list)
    assert [part["type"] for part in content] == ["text", "image_url"]


def test_screen_selection_builder_targets_session_without_switching_view():
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    store = ConsoleChatStore()
    first = store.create_session(
        title="First",
        workspace_id="workspace-a",
        settings=_settings("openai", "model-a", "system-a"),
    )
    store.create_session(
        title="Second",
        workspace_id="workspace-b",
        settings=_settings("anthropic", "model-b", "system-b"),
    )
    fake_screen = SimpleNamespace(
        # Real ChatScreen carries this as a CLASS-attribute default (None =
        # no derivation pass open); the derivation path reads it
        # unconditionally, and a SimpleNamespace double has no class default
        # to fall back on. Went red on dev when the memo landed without this
        # double being taught about it -- the stale-double class again.
        _console_derivation_memo=None,
        _provider_readiness_app_config=lambda: {
            "api_settings": {
                "openai": {"model": "configured-a"},
                "anthropic": {"model": "configured-b"},
            },
            "console": {},
        },
        _ensure_console_chat_store=lambda: store,
        _session=SimpleNamespace(
            _console_session_settings=lambda session_id: store.session_settings(
                session_id
            ),
            _ensure_active_console_session_settings=lambda: store.session_settings(
                store.active_session_id
            ),
        ),
        _effective_console_provider_model=lambda: ("anthropic", "model-b"),
        _config_section=lambda config, key: dict(config.get(key, {})),
        _workspace=SimpleNamespace(
            _current_console_workspace_context=lambda: ConsoleWorkspaceContext(
                active_workspace_id="workspace-b"
            )
        ),
        _normalize_llamacpp_base_url=lambda value: value,
    )
    # task-15452 split the builder into a memo wrapper plus
    # `_build_console_provider_selection_uncached`; the wrapper under test
    # delegates to the latter through `self`, so the double borrows the real
    # uncached half exactly as the memo-less path binds it in production.
    fake_screen._build_console_provider_selection_uncached = lambda session_id=None: (
        ChatScreen._build_console_provider_selection_uncached(fake_screen, session_id)
    )

    selection = ChatScreen._build_console_provider_selection(fake_screen, first.id)

    assert selection.provider == "openai"
    assert selection.explicit_model == "model-a"
    assert selection.configured_model == "configured-a"
    assert (selection.explicit_model or selection.configured_model) == "model-a"
    assert selection.system_prompt == "system-a"
    assert selection.workspace_context.active_workspace_id == "workspace-a"
    assert store.active_session_id != first.id
