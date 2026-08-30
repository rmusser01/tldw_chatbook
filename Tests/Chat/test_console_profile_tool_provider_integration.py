"""Console registry integration for governed Personal Context tools."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from tldw_chatbook.Agents.agent_models import (
    SPAWN_TOOL_NAME,
    ToolCatalogEntry,
    ToolResult,
    ToolSchema,
)
from tldw_chatbook.Agents.tool_catalog import (
    PROFILE_RESERVED_TOOL_NAMES,
    ToolCatalogRegistry,
)
from tldw_chatbook.Chat.console_agent_bridge import (
    _compose_run_registry_and_allowed,
    _non_colliding_skill_entries,
    build_console_first_request_plan,
)
from tldw_chatbook.Chat.console_chat_controller import (
    _compose_profile_tool_provider,
)
from tldw_chatbook.Chat.console_chat_models import (
    CONSOLE_GLOBAL_WORKSPACE_ID,
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Personal_Context.key_protector import (
    InMemoryProfileKeyProtector,
)
from tldw_chatbook.Personal_Context.context_service import ProfileContextSnapshot
from tldw_chatbook.Personal_Context.repository import PersonalContextRepository
from tldw_chatbook.Personal_Context.runtime_policy import AgentAuthority
from tldw_chatbook.Personal_Context.service import PersonalContextService


class _FakeProfileProvider:
    def __init__(self, *names: str) -> None:
        self._names = names
        self.invoke_calls: list[tuple[str, dict]] = []

    def list_catalog(self) -> list[ToolCatalogEntry]:
        return [
            ToolCatalogEntry(
                id=f"profile:{name}",
                name=name,
                one_line_description=name,
                source="profile",
            )
            for name in self._names
        ]

    def load_schema(self, tool_id: str) -> ToolSchema:
        name = tool_id.removeprefix("profile:")
        return ToolSchema(
            id=tool_id,
            name=name,
            description=name,
            parameters={"type": "object", "properties": {}},
        )

    def invoke(self, tool_id: str, args: dict) -> ToolResult:
        self.invoke_calls.append((tool_id, dict(args)))
        return ToolResult(ok=True, content="{}")


class _FakeMCPProvider:
    def __init__(self, *names: str) -> None:
        self._names = names
        self.invoke_calls: list[tuple[str, dict]] = []

    def list_catalog(self) -> list[ToolCatalogEntry]:
        return [
            ToolCatalogEntry(
                id=name,
                name=name,
                one_line_description=name,
                source="mcp",
            )
            for name in self._names
        ]

    def load_schema(self, tool_id: str) -> ToolSchema:
        return ToolSchema(
            id=tool_id,
            name=tool_id,
            description=tool_id,
            parameters={"type": "object", "properties": {}},
        )

    def invoke(self, tool_id: str, args: dict) -> ToolResult:
        self.invoke_calls.append((tool_id, dict(args)))
        return ToolResult(ok=True, content="{}")


def _skill(name: str) -> dict[str, object]:
    return {
        "name": name,
        "description": name,
        "argument_hint": "",
        "trust_blocked": False,
        "disable_model_invocation": False,
    }


def test_profile_tools_register_between_library_seam_and_skills() -> None:
    provider = _FakeProfileProvider("profile_search", "profile_get")
    context = {"available_skills": [_skill("ordinary_skill")]}

    registry, allowed, builtin_names, local_names = _compose_run_registry_and_allowed(
        context, profile_provider=provider
    )

    assert allowed == (
        *builtin_names,
        "profile_search",
        "profile_get",
        "ordinary_skill",
        SPAWN_TOOL_NAME,
    )
    assert local_names == ()
    assert [(entry.name, entry.source) for entry in registry.list_catalog()] == [
        *((name, "builtin") for name in builtin_names),
        ("profile_search", "profile"),
        ("profile_get", "profile"),
        ("ordinary_skill", "skill"),
    ]
    result = registry.invoke_by_name("profile_get", {"record_id": "r1"})
    assert result.ok is True
    assert provider.invoke_calls == [("profile:profile_get", {"record_id": "r1"})]


def test_profile_reserved_names_cannot_be_intercepted_by_skill_or_mcp() -> None:
    provider = _FakeProfileProvider("profile_update")
    mcp = _FakeMCPProvider("profile_update")
    context = {"available_skills": [_skill("profile_update")]}

    registry, allowed, _builtin_names, _local_names = _compose_run_registry_and_allowed(
        context,
        profile_provider=provider,
        mcp_provider=mcp,
    )

    assert allowed.count("profile_update") == 1
    assert [(entry.name, entry.source) for entry in registry.list_catalog()].count(
        ("profile_update", "profile")
    ) == 1
    assert ("profile_update", "skill") not in [
        (entry.name, entry.source) for entry in registry.list_catalog()
    ]
    assert ("profile_update", "mcp") not in [
        (entry.name, entry.source) for entry in registry.list_catalog()
    ]
    registry.invoke_by_name("profile_update", {})
    assert provider.invoke_calls == [("profile:profile_update", {})]
    assert mcp.invoke_calls == []


def test_profile_reserved_names_are_filtered_at_skill_runner_seam() -> None:
    context = {"available_skills": [_skill("profile_promote"), _skill("kept")]}

    eligible = _non_colliding_skill_entries(
        context,
        (),
        profile_names=PROFILE_RESERVED_TOOL_NAMES,
    )

    assert [entry["name"] for entry in eligible] == ["kept"]


def test_ephemeral_run_omits_profile_provider() -> None:
    provider = _FakeProfileProvider("profile_search")

    registry, allowed, _builtin_names, _local_names = _compose_run_registry_and_allowed(
        {},
        profile_provider=provider,
        ephemeral=True,
    )

    assert "profile_search" not in allowed
    assert all(entry.source != "profile" for entry in registry.list_catalog())


def _profile_service(tmp_path) -> PersonalContextService:
    service = PersonalContextService(
        PersonalContextRepository(
            tmp_path / "personal-context.db",
            key_protector=InMemoryProfileKeyProtector(),
        ),
        clock=lambda: datetime(2026, 8, 30, tzinfo=UTC),
    )
    service.create_profile()
    service.set_runtime_enabled(True)
    return service


def test_controller_composes_direct_write_provider_with_exact_user_evidence(
    tmp_path,
) -> None:
    service = _profile_service(tmp_path)
    global_scope = service.list_scopes()[0]
    service.set_scope_authority(global_scope.scope_id, AgentAuthority.DIRECT_WRITE)
    user_message = ConsoleChatMessage(
        id="message-1",
        role=ConsoleMessageRole.USER,
        content="I prefer concise replies.",
    )

    provider = _compose_profile_tool_provider(
        service,
        workspace_id=CONSOLE_GLOBAL_WORKSPACE_ID,
        ephemeral=False,
        run_id="assistant-1",
        session_id="session-1",
        current_user_message=user_message,
        kill_switch=lambda: False,
    )

    assert provider is not None
    assert "profile_update" in {entry.name for entry in provider.list_catalog()}


def test_controller_pins_durable_user_message_reference_when_available(
    tmp_path,
) -> None:
    service = _profile_service(tmp_path)
    global_scope = service.list_scopes()[0]
    service.set_scope_authority(global_scope.scope_id, AgentAuthority.DIRECT_WRITE)
    user_message = ConsoleChatMessage(
        id="native-message-1",
        persisted_message_id="durable-message-1",
        role=ConsoleMessageRole.USER,
        content="I prefer concise replies.",
    )

    provider = _compose_profile_tool_provider(
        service,
        workspace_id=CONSOLE_GLOBAL_WORKSPACE_ID,
        ephemeral=False,
        run_id="assistant-1",
        session_id="session-1",
        current_user_message=user_message,
        kill_switch=lambda: False,
    )

    assert provider is not None
    schema = provider.load_schema("personal-context:profile_update")
    assert schema.parameters["properties"]["current_user_message_id"]["const"] == (
        "durable-message-1"
    )


def test_controller_omits_direct_update_without_exact_user_evidence(tmp_path) -> None:
    service = _profile_service(tmp_path)
    global_scope = service.list_scopes()[0]
    service.set_scope_authority(global_scope.scope_id, AgentAuthority.DIRECT_WRITE)

    provider = _compose_profile_tool_provider(
        service,
        workspace_id=CONSOLE_GLOBAL_WORKSPACE_ID,
        ephemeral=False,
        run_id="assistant-1",
        session_id="session-1",
        current_user_message=None,
        kill_switch=lambda: False,
    )

    assert provider is not None
    assert "profile_update" not in {entry.name for entry in provider.list_catalog()}


def test_preview_reserves_direct_update_schema_without_granting_evidence(
    tmp_path,
) -> None:
    service = _profile_service(tmp_path)
    global_scope = service.list_scopes()[0]
    service.set_scope_authority(global_scope.scope_id, AgentAuthority.DIRECT_WRITE)
    preview = _compose_profile_tool_provider(
        service,
        workspace_id=CONSOLE_GLOBAL_WORKSPACE_ID,
        ephemeral=False,
        run_id="preview:session-1",
        session_id="session-1",
        current_user_message=None,
        kill_switch=lambda: False,
        reserve_direct_update_schema=True,
    )
    live = _compose_profile_tool_provider(
        service,
        workspace_id=CONSOLE_GLOBAL_WORKSPACE_ID,
        ephemeral=False,
        run_id="assistant-1",
        session_id="session-1",
        current_user_message=ConsoleChatMessage(
            id="native-message-1",
            persisted_message_id="durable-message-1",
            role=ConsoleMessageRole.USER,
            content="I prefer concise replies.",
        ),
        kill_switch=lambda: False,
    )

    assert preview is not None and live is not None
    assert "profile_update" in {entry.name for entry in preview.list_catalog()}
    assert (
        preview.invoke(
            "profile_update",
            {
                "record_id": "record-1",
                "base_version_id": "version-1",
                "current_user_message_id": "preview-only",
                "evidence_span": "I prefer concise replies.",
                "proposed_payload": {
                    "kind": "preference",
                    "subject": "response.detail",
                    "polarity": "like",
                    "value": "concise",
                },
            },
        ).error
        == "review_required"
    )

    class _Builder:
        def __init__(self) -> None:
            self.requests = []

        def build_snapshot(self, request):
            self.requests.append(request)
            return ProfileContextSnapshot.empty()

    def build_plan(provider, builder):
        return build_console_first_request_plan(
            shared_registry=ToolCatalogRegistry(),
            shared_allowed_tools=(),
            context={},
            skills_present=False,
            mcp_provider=None,
            builtin_gate=None,
            local_provider=None,
            library_provider=None,
            library_authority=None,
            profile_provider=provider,
            workspace_id=CONSOLE_GLOBAL_WORKSPACE_ID,
            ephemeral=False,
            diff_sink=None,
            scratch_root=None,
            scratch_lease=None,
            resolution=SimpleNamespace(
                model="gpt-4o-mini", execution_key="openai", max_tokens=2_048
            ),
            fallback_model="gpt-4o-mini",
            session_system_prompt="",
            native_tools=True,
            turn_skill_bindings=(),
            turn_bundle_block="",
            install_skill_enabled=False,
            run_skill_script_enabled=False,
            agent_messages=[{"role": "user", "content": "question"}],
            profile_context_service=builder,
        )

    preview_builder = _Builder()
    live_builder = _Builder()
    build_plan(preview, preview_builder)
    build_plan(live, live_builder)

    assert (
        preview_builder.requests[0].available_input_tokens
        <= live_builder.requests[0].available_input_tokens
    )


def test_controller_omits_profile_provider_for_ephemeral_session(tmp_path) -> None:
    service = _profile_service(tmp_path)

    assert (
        _compose_profile_tool_provider(
            service,
            workspace_id=CONSOLE_GLOBAL_WORKSPACE_ID,
            ephemeral=True,
            run_id="assistant-1",
            session_id="session-1",
            current_user_message=None,
            kill_switch=lambda: False,
        )
        is None
    )


def test_controller_binds_named_console_workspace_to_its_canonical_scope(
    tmp_path,
) -> None:
    service = _profile_service(tmp_path)
    workspace_scope = service.create_workspace_scope("workspace-1", "Project One")
    service.set_scope_authority(workspace_scope.scope_id, AgentAuthority.READ_ONLY)

    provider = _compose_profile_tool_provider(
        service,
        workspace_id="workspace-1",
        ephemeral=False,
        run_id="assistant-1",
        session_id="session-1",
        current_user_message=None,
        kill_switch=lambda: False,
    )

    assert provider is not None
    assert {entry.name for entry in provider.list_catalog()} == {
        "profile_search",
        "profile_get",
    }
