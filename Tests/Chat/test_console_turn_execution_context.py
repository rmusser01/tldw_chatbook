"""Immutable owning-session turn-context contracts for Console sends."""

from __future__ import annotations

import asyncio
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_chatbook.Agents.agent_models import RUN_DONE, RunBudget, RunOutcome
from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider
from tldw_chatbook.Chat.attachment_core import PendingAttachment
from tldw_chatbook.Chat.console_chat_controller import (
    ConsoleChatController,
    _capture_project_root_identity,
    capture_project_instruction_authority,
    capture_skill_context_maximum,
    project_instruction_authority_is_current,
)
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
from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch
from tldw_chatbook.Chat.console_project_instructions import (
    ProjectInstructionControlState,
    fingerprint_canonical_locator,
)
from tldw_chatbook.Chat.console_roleplay_identity import ConsolePresentationContext
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Chat.console_scratch_space import (
    ConsoleScratchSnapshot,
    ConsoleScratchSpaceManager,
)
from tldw_chatbook.Chat.console_turn_context import (
    ConsoleProjectBindingSnapshot,
    ConsoleTurnConfigurationSnapshot,
    ConsoleTurnCustodyRequest,
    ConsoleTurnExecutionContext,
)
from tldw_chatbook.Chat.console_turn_preparation import ConsoleTurnPreparation
from tldw_chatbook.Chat.console_turn_preparation import ConsoleTurnPreparationState
from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController
from tldw_chatbook.Workspaces import SkippedReviewRoot
from tldw_chatbook.Chat.message_metadata import MessageMetadata


class ConsoleChatStore(_ConsoleChatStore):
    """Test store whose intentionally db-less sessions are explicitly ephemeral."""

    def create_session(self, **kwargs):
        kwargs.setdefault("ephemeral", self.persistence is None)
        return super().create_session(**kwargs)


@pytest.mark.parametrize(
    ("captured", "live", "expected"),
    (
        (False, True, False),
        (True, False, False),
        (True, True, True),
    ),
)
def test_exchange_capture_authority_can_only_narrow_after_handoff(
    monkeypatch, captured, live, expected
):
    controller = ConsoleChatController(
        store=ConsoleChatStore(), provider_gateway=SimpleNamespace()
    )
    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_chat_controller.runtime_capture_policy",
        lambda: SimpleNamespace(enabled=live, legacy_writes_enabled=live),
    )

    signals = controller._new_run_stream_signals(maximum=captured)

    assert signals.exchange_capture_enabled is expected


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


class _CustodyController:
    def __init__(self, *, fail_before_acceptance: bool = False) -> None:
        self.fail_before_acceptance = fail_before_acceptance
        self.calls: list[dict[str, object]] = []

    async def run_prompt_chain(self, *, session_id, initial_turn):
        assert session_id
        return await initial_turn()

    async def submit_draft(self, draft, **kwargs):
        self.calls.append({"draft": draft, **kwargs})
        if self.fail_before_acceptance:
            raise RuntimeError("pre-durable failure")
        kwargs["custody_acceptance_hook"]()
        return SimpleNamespace(accepted=True)


class _EvidenceCustodyController(_CustodyController):
    """Minimal controller double that exercises the runtime-owned lease."""

    async def submit_draft(self, draft, **kwargs):
        self.calls.append({"draft": draft, **kwargs})
        if self.fail_before_acceptance:
            raise RuntimeError("pre-durable failure")
        launch = kwargs["staged_evidence_launch"]
        captured = await kwargs["staged_evidence_capture"](draft, None, launch)
        kwargs["staged_evidence_release"](launch, captured)
        kwargs["custody_acceptance_hook"]()
        return SimpleNamespace(accepted=True)


class _DurablyAcceptedBlockingController(_CustodyController):
    def __init__(self) -> None:
        super().__init__()
        self.accepted = asyncio.Event()

    async def submit_draft(self, draft, **kwargs):
        self.calls.append({"draft": draft, **kwargs})
        kwargs["custody_acceptance_hook"]()
        self.accepted.set()
        await asyncio.Event().wait()


def _custody_configuration(session_id: str) -> ConsoleTurnConfigurationSnapshot:
    return ConsoleTurnConfigurationSnapshot.capture(
        session_id=session_id,
        provider_selection=ConsoleProviderSelection(
            provider="openai", explicit_model="frozen-model"
        ),
        capabilities={"vision": True},
    )


@pytest.mark.asyncio
async def test_submit_diagnostics_preserve_all_frozen_custody_arguments(monkeypatch):
    """The diagnostics/lifecycle layers pass the complete admission unchanged."""
    from tldw_chatbook.Chat.console_chat_controller import ConsoleSubmitResult

    store = ConsoleChatStore()
    session = store.create_session(title="Wrapper custody", workspace_id="global")
    controller = ConsoleChatController(store=store, provider_gateway=SimpleNamespace())
    attachment = PendingAttachment("/a", "a.png", "image", "attachment", data=b"a")
    launch = ConsoleLiveWorkLaunch.from_values(
        source="rag", title="Frozen evidence", payload={"ids": [1]}
    )
    accepted = []

    async def capture(*args):
        return launch

    def release(*args):
        pass

    expected = {
        "configuration": _custody_configuration(session.id),
        "accepted_attachments": (attachment,),
        "captured_one_shot_prefill": "FROZEN_PREFILL_SENTINEL",
        "captured_one_shot_prefill_revision": 7,
        "staged_evidence_launch": launch,
        "staged_evidence_capture": capture,
        "staged_evidence_release": release,
        "custody_acceptance_hook": lambda: accepted.append(True),
    }

    async def inspect_admission(draft, **kwargs):
        assert draft == "frozen draft"
        assert kwargs["session_id"] == session.id
        for name, value in expected.items():
            assert kwargs[name] is value
        kwargs["custody_acceptance_hook"]()
        return ConsoleSubmitResult(True, True)

    monkeypatch.setattr(controller, "_submit_draft_inner", inspect_admission)
    result = await controller.submit_draft(
        "frozen draft", session_id=session.id, **expected
    )

    assert result.accepted
    assert accepted == [True]


@pytest.mark.asyncio
async def test_runtime_custody_transfers_exact_attachments_and_frozen_inputs():
    store = ConsoleChatStore()
    session = store.create_session(title="Custody", workspace_id="global")
    first = PendingAttachment("/a", "a.png", "image", "attachment", data=b"a")
    second = PendingAttachment("/b", "b.png", "image", "attachment", data=b"b")
    store.add_pending_attachment(session.id, first)
    store.add_pending_attachment(session.id, second)
    launch = ConsoleLiveWorkLaunch.from_values(
        source="rag", title="Frozen evidence", payload={"ids": [1]}
    )
    configuration = _custody_configuration(session.id)
    controller = _CustodyController()
    runtime = ConsoleRuntime(SimpleNamespace())
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    request = ConsoleTurnCustodyRequest(
        turn_id="turn-custody",
        session_id=session.id,
        draft="exact draft",
        configuration=configuration,
        attachment_ids=(first.attachment_id, second.attachment_id),
        staged_evidence_launch=launch,
        one_shot_prefill="FROZEN_PREFILL_SENTINEL",
        one_shot_prefill_revision=7,
    )

    turn_id = runtime.accept_turn(request)

    assert store.pending_attachments(session.id) == []
    await runtime.wait_for_turn(turn_id)
    call = controller.calls[0]
    assert call["configuration"] is configuration
    assert call["accepted_attachments"] == (first, second)
    assert call["accepted_attachments"][0] is first
    assert call["staged_evidence_launch"] is launch
    assert call["captured_one_shot_prefill"] == "FROZEN_PREFILL_SENTINEL"
    assert call["captured_one_shot_prefill_revision"] == 7
    assert "FROZEN_PREFILL_SENTINEL" not in repr(request)


@pytest.mark.asyncio
async def test_manual_custody_uses_captured_one_shot_and_preserves_newer_revision():
    store = ConsoleChatStore()
    session = store.create_session(title="Prefill custody", workspace_id="global")
    store.set_session_one_shot_prefill(session.id, "captured-old")
    old_prefill, old_revision = store.session_one_shot_prefill_snapshot(session.id)
    gateway = _PausedGateway()
    bridge_calls: list[dict[str, object]] = []
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        agent_bridge=SimpleNamespace(
            run_reply=lambda **kwargs: bridge_calls.append(kwargs)
        ),
        agent_runtime_enabled=True,
        provider="openai",
        model="frozen-model",
    )
    runtime = ConsoleRuntime(SimpleNamespace())
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    request = ConsoleTurnCustodyRequest(
        turn_id="manual-prefill-old",
        session_id=session.id,
        draft="first",
        configuration=controller.resolve_runtime_turn_configuration_snapshot(
            session.id
        ),
        one_shot_prefill=old_prefill,
        one_shot_prefill_revision=old_revision,
    )

    turn_id = runtime.accept_turn(request)
    await gateway.resolve_started.wait()
    store.set_session_one_shot_prefill(session.id, None)
    gateway.release_resolve.set()
    first = await runtime.wait_for_turn(turn_id)

    assert first.accepted is True
    assert bridge_calls == []
    assert gateway.message_batches[-1][-1] == {
        "role": "assistant",
        "content": "captured-old",
    }
    assert store.session_one_shot_prefill(session.id) is None

    store.set_session_one_shot_prefill(session.id, "staged-new")
    new_prefill, new_revision = store.session_one_shot_prefill_snapshot(session.id)
    later = ConsoleTurnCustodyRequest(
        turn_id="manual-prefill-new",
        session_id=session.id,
        draft="second",
        configuration=controller.resolve_runtime_turn_configuration_snapshot(
            session.id
        ),
        one_shot_prefill=new_prefill,
        one_shot_prefill_revision=new_revision,
    )
    second = await runtime.wait_for_turn(runtime.accept_turn(later))

    assert second.accepted is True
    assert gateway.message_batches[-1][-1] == {
        "role": "assistant",
        "content": "staged-new",
    }
    assert store.session_one_shot_prefill(session.id) is None


def test_queued_custody_freezes_prefill_at_enqueue_and_later_entry_gets_new_value():
    store = ConsoleChatStore()
    session = store.create_session(title="Queued prefill", workspace_id="global")
    controller = ConsoleChatController(store=store, provider_gateway=SimpleNamespace())
    initial = controller.prompt_queue_registry.snapshot(session.id)
    armed = controller.prompt_queue_registry.begin_chain(
        session.id,
        context_epoch=store.conversation_context_epoch(session.id),
        expected_revision=initial.revision,
    )
    store.set_session_one_shot_prefill(session.id, "queued-old")
    old_revision = store.session_one_shot_prefill_snapshot(session.id)[1]

    first = controller.queue_prompt(
        session.id,
        text="first queued turn",
        expected_revision=armed.snapshot.revision,
    )
    first_request = controller.prompt_queue_registry._states[
        session.id
    ].waiting[0].custody_request

    assert first.applied is True
    assert first_request.one_shot_prefill == "queued-old"
    assert first_request.one_shot_prefill_revision == old_revision

    store.set_session_one_shot_prefill(session.id, "queued-new")
    new_revision = store.session_one_shot_prefill_snapshot(session.id)[1]
    second = controller.queue_prompt(
        session.id,
        text="second queued turn",
        expected_revision=first.snapshot.revision,
    )
    requests = tuple(
        prompt.custody_request
        for prompt in controller.prompt_queue_registry._states[session.id].waiting
    )

    assert second.applied is True
    assert requests[0].one_shot_prefill == "queued-old"
    assert requests[0].one_shot_prefill_revision == old_revision
    assert requests[1].one_shot_prefill == "queued-new"
    assert requests[1].one_shot_prefill_revision == new_revision


@pytest.mark.asyncio
@pytest.mark.parametrize("capture_path", ["queue", "edit", "fleet_wake"])
@pytest.mark.parametrize("detached", [False, True])
async def test_runtime_created_controller_captures_owning_workspace_policy(
    monkeypatch, tmp_path, capture_path, detached
):
    from tldw_chatbook.Agents.persona_policy import (
        parse_persona_policy_from_rules,
        persona_floor_state,
    )
    from tldw_chatbook.MCP.permission_store import EffectiveToolState
    from tldw_chatbook.RAG_Search.simplified import active_config

    search_settings = SimpleNamespace(default_top_k=12)
    monkeypatch.delenv("RAG_TOP_K", raising=False)
    monkeypatch.setattr(
        active_config,
        "_resolved_active_profile",
        lambda: SimpleNamespace(rag_config=SimpleNamespace(search=search_settings)),
    )

    roots = [str(tmp_path / "approved")]
    aliases = ["approved-folder"]
    skipped = [
        SkippedReviewRoot(alias="pending-folder", reason="Preparing change history")
    ]
    rules = [
        {
            "rule_kind": "mcp_tool",
            "rule_name": "write",
            "allowed": True,
            "require_confirmation": True,
        }
    ]
    defaults = SimpleNamespace(tool_policy_profile_id="restricted")
    lookups = []

    def workspace(workspace_id):
        lookups.append(("workspace", workspace_id))
        return SimpleNamespace(assistant_defaults=defaults)

    def persona(persona_id):
        lookups.append(("persona", persona_id))
        return {"policy_rules": rules}

    def admit(workspace_id):
        lookups.append(("review", workspace_id))
        return SimpleNamespace(
            ready_roots=roots, ready_aliases=aliases, skipped_roots=skipped
        )

    app = SimpleNamespace(
        app_config={"console": {"native_tool_calls": "false"}},
        console_prompt_history_factory=lambda: SimpleNamespace(),
        workspace_registry_service=SimpleNamespace(get_workspace=workspace),
        local_character_persona_service=SimpleNamespace(get_persona_profile=persona),
        change_review_consent_service=SimpleNamespace(admit_turn=admit),
    )
    runtime = ConsoleRuntime(app)
    store = ConsoleChatStore()
    owner = store.create_session(
        workspace_id="workspace-a", assistant_kind="persona", assistant_id="persona-a"
    )
    other = store.create_session(workspace_id="workspace-b")
    runtime.set_chat_store(store)
    controller = runtime.ensure_chat_controller(
        store=store, provider_gateway=SimpleNamespace(), agent_runtime_enabled=False
    )
    assert controller._turn_context_provider is None
    if detached:
        view = SimpleNamespace(console_view_hooks=lambda: {})
        generation = runtime.attach_view(view)
        assert runtime.detach_view(view, generation)
    assert runtime.view is None
    assert store.active_session_id == other.id
    try:
        if capture_path == "fleet_wake":
            captured = []
            monkeypatch.setattr(
                runtime,
                "accept_turn",
                lambda request, **kwargs: captured.append(request) or request.turn_id,
            )
            runtime._submit_fleet_wake(
                "completed child",
                session_id=owner.id,
                wake_authorization=object(),
                on_terminal=lambda _: None,
            )
            configuration = captured[0].configuration
        else:
            initial = controller.prompt_queue_registry.snapshot(owner.id)
            armed = controller.prompt_queue_registry.begin_chain(
                owner.id,
                context_epoch=store.conversation_context_epoch(owner.id),
                expected_revision=initial.revision,
            )
            queued = controller.queue_prompt(
                owner.id, text="queued", expected_revision=armed.snapshot.revision
            )
            assert queued.applied
            if capture_path == "edit":
                # Edits must recapture the owner's latest policy, not reuse
                # admission or read the different active session.
                defaults.tool_policy_profile_id = "edited-restricted"
                search_settings.default_top_k = 17
                edited = controller.edit_queued_prompt(
                    owner.id,
                    entry_id=queued.snapshot.entries[0].entry_id,
                    text="edited",
                    expected_revision=queued.snapshot.revision,
                )
                assert edited.applied
            configuration = (
                controller.prompt_queue_registry._states[owner.id]
                .waiting[0]
                .custody_request.configuration
            )

        expected_profile = defaults.tool_policy_profile_id
        expected_top_k = search_settings.default_top_k
        search_settings.default_top_k = 2
        app.app_config["console"]["native_tool_calls"] = True
        roots.append(str(tmp_path / "later"))
        aliases.append("later-folder")
        skipped.clear()
        rules[0]["require_confirmation"] = False
        defaults.tool_policy_profile_id = "later-profile"
        store.switch_session(owner.id)
        assert configuration.session_id == owner.id
        assert configuration.workspace_roots == (str(tmp_path / "approved"),)
        assert configuration.change_review_root_aliases == ("approved-folder",)
        assert configuration.change_review_skipped_roots == (
            SkippedReviewRoot(
                alias="pending-folder", reason="Preparing change history"
            ),
        )
        assert configuration.tool_policy_profile_id == expected_profile
        assert configuration.rag_defaults["top_k"] == expected_top_k
        assert configuration.tool_configuration["native_tool_calls_enabled"] is False
        policy = parse_persona_policy_from_rules(configuration.persona_policy_rules)
        assert (
            persona_floor_state(
                EffectiveToolState(state="allow", origin="profile"), policy, "write"
            ).state
            == "ask"
        )
        assert ("workspace", "workspace-a") in lookups
        assert ("persona", "persona-a") in lookups
        assert ("review", "workspace-a") in lookups
        assert not any(value == "workspace-b" for _, value in lookups)

        # Exercise the actual automatic retrieval request from this queued /
        # wake snapshot after the live profile has changed.
        authority = _authority()
        authority = replace(
            authority,
            policy=replace(
                authority.policy, auto_retrieve=ConsoleAutoRetrieve.AUTOMATIC
            ),
        )
        context = ConsoleTurnExecutionContext(configuration, authority, _destination())
        preparation = ConsoleTurnPreparation(
            preparation_id="runtime-profile-rag",
            attempt_id=authority.attempt_id,
            session_id=owner.id,
            origin="manual",
            queue_entry_id=None,
            executed_draft="frozen query",
            execution_context=context,
            transient_user_message_id=None,
            attachment_ids=(),
            evidence_ids=(),
            prefill_id=None,
            queue_generation=None,
            pre_send_title="Chat",
            pre_send_conversation_id=None,
            state=ConsoleTurnPreparationState.PREPARING,
            pause_kind=None,
            one_shot_bypass=False,
            ephemeral=True,
        )
        requests = []

        async def search(query, source_types, mode, **kwargs):
            requests.append((query, source_types, mode, kwargs))
            return {"results": []}

        app.library_rag_search_service = SimpleNamespace(search=search)
        assert store.begin_preparation(preparation) is preparation
        outcome = await controller.prepare_library_for_turn(preparation.preparation_id)
        assert outcome.state is ConsoleTurnPreparationState.READY
        assert controller._frozen_rag_top_k(context) == expected_top_k
        assert requests[0][3]["top_k"] == expected_top_k
    finally:
        await runtime.dispose()


def test_project_instruction_byte_caps_freeze_per_later_turn(monkeypatch):
    from tldw_chatbook.Chat import console_chat_controller as controller_module

    store = ConsoleChatStore()
    session = store.create_session(title="Caps", workspace_id="workspace-a")
    values = {
        "project_instructions_startup_max_bytes": 2_048,
        "project_instructions_nested_max_bytes": 4_096,
    }

    def setting(_section, key, default=None):
        return values.get(key, default)

    monkeypatch.setattr(controller_module, "get_cli_setting", setting)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=_PausedGateway(),
        agent_runtime_enabled=True,
    )
    controller.app = SimpleNamespace(workspace_registry_service=None)

    old = controller.resolve_runtime_turn_configuration_snapshot(session.id)
    values.update(
        project_instructions_startup_max_bytes=8_192,
        project_instructions_nested_max_bytes=16_384,
    )
    new = controller.resolve_runtime_turn_configuration_snapshot(session.id)

    assert old.tool_configuration["project_instructions_startup_max_bytes"] == 2_048
    assert old.tool_configuration["project_instructions_nested_max_bytes"] == 4_096
    assert new.tool_configuration["project_instructions_startup_max_bytes"] == 8_192
    assert new.tool_configuration["project_instructions_nested_max_bytes"] == 16_384


@pytest.mark.asyncio
async def test_manual_runtime_custody_reaches_the_existing_agent_bridge_once(tmp_path):
    """Manual custody converges on the controller's one existing agent task."""
    store = ConsoleChatStore()
    session = store.create_session(title="Agent custody", workspace_id="workspace-a")
    attachment = PendingAttachment(
        "image.png",
        "private-name.png",
        "image",
        "attachment",
        data=b"png",
        mime_type="image/png",
    )
    store.add_pending_attachment(session.id, attachment)
    gateway = _PausedGateway()
    gateway.release_resolve.set()
    runtime = ConsoleRuntime(SimpleNamespace())
    bridge_calls: list[dict[str, object]] = []
    custody_at_bridge: list[tuple[str, ...]] = []

    def run_reply(**kwargs):
        custody_at_bridge.append(tuple(runtime._turn_custody))
        bridge_calls.append(kwargs)
        return "run-custody", RunOutcome(
            status=RUN_DONE,
            steps=[],
            final_text="agent reply",
        )

    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        agent_bridge=SimpleNamespace(run_reply=run_reply),
        agent_runtime_enabled=True,
        provider="openai",
        model="frozen-model",
    )
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()

    def binding(binding_id, root, access):
        return SimpleNamespace(
            binding_id=binding_id,
            workspace_id=session.workspace_id,
            display_name=binding_id,
            binding_kind=SimpleNamespace(value="local-filesystem"),
            status=SimpleNamespace(value="ready"),
            locator=str(root),
            metadata={"access": access},
        )

    current_bindings = [binding("binding-a", root_a, "ro")]

    class Registry:
        def list_runtime_bindings(self, _workspace_id):
            return tuple(current_bindings)

        def get_runtime_binding(self, binding_id):
            return next(
                item for item in current_bindings if item.binding_id == binding_id
            )

    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    controller.app = SimpleNamespace(workspace_registry_service=Registry())
    configuration = controller.resolve_runtime_turn_configuration_snapshot(
        session.id
    )
    assert configuration.tool_configuration["agent_runtime_enabled"] is True
    configuration = ConsoleTurnConfigurationSnapshot.capture(
        session_id=session.id,
        provider_selection=configuration.provider_selection,
        scratch_space=configuration.scratch_space,
        session_settings=configuration.session_settings,
        capabilities={"vision": True, "max_history_images": 1},
        tool_configuration={
            **configuration.tool_configuration,
            "agent_run_budget_maximum": RunBudget(
                max_steps=11,
                max_wall_seconds=12.0,
                max_subagents=1,
                max_total_tokens=13,
            ),
            "exchange_capture_enabled": False,
        },
        project_authority=configuration.project_authority,
    )
    current_bindings[:] = [binding("binding-b", root_b, "rw")]
    session.workspace_id = "workspace-b"
    request = ConsoleTurnCustodyRequest(
        turn_id="manual-agent-custody",
        session_id=session.id,
        draft="delegate this",
        configuration=configuration,
        attachment_ids=(attachment.attachment_id,),
    )

    turn_id = runtime.accept_turn(request)

    assert tuple(runtime._turn_custody) == (turn_id,)
    result = await runtime.wait_for_turn(turn_id)
    await asyncio.sleep(0)

    assert result.accepted is True
    assert len(bridge_calls) == 1
    assert custody_at_bridge == [(turn_id,)]
    assert runtime._turn_custody == {}
    assert bridge_calls[0]["workspace_id"] == "workspace-a"
    assert bridge_calls[0]["workspace_ephemeral"] is True
    assert bridge_calls[0]["workspace_read_binding_ids"] == ("binding-a",)
    assert bridge_calls[0]["workspace_write_binding_ids"] == ()
    assert bridge_calls[0]["run_budget"].max_steps == 11
    assert bridge_calls[0]["run_budget"].max_wall_seconds == 12.0
    assert bridge_calls[0]["run_budget"].max_subagents == 1
    assert bridge_calls[0]["run_budget"].max_total_tokens == 13
    assert (
        bridge_calls[0]["provider_stream_signals"].exchange_capture_enabled is False
    )
    agent_messages = bridge_calls[0]["agent_messages"]
    assert any(
        part.get("type") == "image_url"
        for row in agent_messages
        for part in (row.get("content") if isinstance(row.get("content"), list) else ())
        if isinstance(part, dict)
    )
    assert all(
        {"turn_id", "attention_id", "queue_entry_id", "terminal_receipt_id"}.isdisjoint(
            row
        )
        for row in agent_messages
    )


@pytest.mark.asyncio
async def test_durable_acceptance_survives_cancellation_for_terminal_callback():
    store = ConsoleChatStore()
    session = store.create_session(title="Cancellation", workspace_id="global")
    controller = _DurablyAcceptedBlockingController()
    runtime = ConsoleRuntime(SimpleNamespace())
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    request = ConsoleTurnCustodyRequest(
        turn_id="durably-accepted-cancelled",
        session_id=session.id,
        draft="accepted before cancellation",
        configuration=_custody_configuration(session.id),
    )
    terminal: list[tuple[bool, bool]] = []

    turn_id = runtime.accept_turn(
        request,
        terminal_callback=lambda accepted: terminal.append(
            (accepted, request.turn_id not in runtime._turn_custody)
        ),
        recover_before_acceptance=False,
    )
    await controller.accepted.wait()
    task = runtime._turn_custody[turn_id].task
    assert task is not None

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)

    assert terminal == [(True, True)]
    assert runtime._turn_custody == {}


@pytest.mark.asyncio
async def test_runtime_custody_releases_only_its_exact_durable_staged_evidence(
    monkeypatch: pytest.MonkeyPatch,
):
    """The runtime owns both sides of an admitted evidence lease."""
    store = ConsoleChatStore()
    session = store.create_session(title="Evidence", workspace_id="global")
    launch = ConsoleLiveWorkLaunch.from_values(
        source="rag", title="Evidence", payload={"ids": [1]}
    )
    controller = _EvidenceCustodyController()
    runtime = ConsoleRuntime(SimpleNamespace())
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    revision = runtime.stage_console_staged_evidence(launch)

    async def capture(_app, captured_launch, *, user_message):
        assert captured_launch is launch
        assert user_message == "use evidence"
        return SimpleNamespace(context="captured evidence")

    monkeypatch.setattr(
        "tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events."
        "capture_console_staged_evidence_for_chat",
        capture,
    )
    request = ConsoleTurnCustodyRequest(
        turn_id="turn-evidence",
        session_id=session.id,
        draft="use evidence",
        configuration=_custody_configuration(session.id),
        staged_evidence_launch=launch,
    )

    turn_id = runtime.accept_turn(request)
    await runtime.wait_for_turn(turn_id)

    call = controller.calls[0]
    assert call["staged_evidence_capture"].__self__ is runtime
    assert call["staged_evidence_release"].func.__self__ is runtime
    assert runtime.snapshot_console_staged_evidence()[0] is None
    assert runtime.snapshot_console_staged_evidence()[1] > revision


@pytest.mark.asyncio
async def test_runtime_custody_leaves_evidence_staged_before_durable_acceptance():
    """A retry keeps its sole staged-evidence owner and redacted recovery."""
    store = ConsoleChatStore()
    session = store.create_session(title="Evidence", workspace_id="global")
    attachment = PendingAttachment(
        "/evidence", "evidence.png", "image", "attachment", data=b"evidence"
    )
    store.add_pending_attachment(session.id, attachment)
    launch = ConsoleLiveWorkLaunch.from_values(
        source="rag", title="Evidence", payload={"ids": [1]}
    )
    runtime = ConsoleRuntime(SimpleNamespace())
    runtime.set_chat_store(store)
    runtime.set_chat_controller(_EvidenceCustodyController(fail_before_acceptance=True))
    runtime.stage_console_staged_evidence(launch)
    request = ConsoleTurnCustodyRequest(
        turn_id="turn-evidence-failure",
        session_id=session.id,
        draft="retry evidence",
        configuration=_custody_configuration(session.id),
        attachment_ids=(attachment.attachment_id,),
        staged_evidence_launch=launch,
    )

    turn_id = runtime.accept_turn(request)
    with pytest.raises(RuntimeError, match="pre-durable failure"):
        await runtime.wait_for_turn(turn_id)
    await asyncio.sleep(0)

    recovery = runtime.recoveries_for_session(session.id)[0]
    assert runtime.snapshot_console_staged_evidence()[0] is launch
    assert recovery.draft == "retry evidence"
    assert recovery.attachments == (attachment,)
    assert not hasattr(recovery, "staged_evidence_launch")


@pytest.mark.asyncio
async def test_runtime_custody_does_not_release_newer_evidence_staged_during_capture(
    monkeypatch: pytest.MonkeyPatch,
):
    """A blocked A capture cannot clear B staged before durable acceptance."""
    store = ConsoleChatStore()
    session = store.create_session(title="Evidence", workspace_id="global")
    launch_a = ConsoleLiveWorkLaunch.from_values(
        source="rag", title="A", payload={"ids": [1]}
    )
    launch_b = ConsoleLiveWorkLaunch.from_values(
        source="rag", title="B", payload={"ids": [2]}
    )
    controller = _EvidenceCustodyController()
    runtime = ConsoleRuntime(SimpleNamespace())
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    runtime.stage_console_staged_evidence(launch_a)
    capture_started = asyncio.Event()
    release_capture = asyncio.Event()

    async def capture(_app, captured_launch, *, user_message):
        assert captured_launch is launch_a
        assert user_message == "use A"
        capture_started.set()
        await release_capture.wait()
        return SimpleNamespace(context="captured A")

    monkeypatch.setattr(
        "tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events."
        "capture_console_staged_evidence_for_chat",
        capture,
    )
    request = ConsoleTurnCustodyRequest(
        turn_id="turn-evidence-race",
        session_id=session.id,
        draft="use A",
        configuration=_custody_configuration(session.id),
        staged_evidence_launch=launch_a,
    )

    turn_id = runtime.accept_turn(request)
    await capture_started.wait()
    runtime.stage_console_staged_evidence(launch_b)
    release_capture.set()
    await runtime.wait_for_turn(turn_id)

    assert runtime.snapshot_console_staged_evidence()[0] is launch_b


@pytest.mark.asyncio
async def test_pre_durable_custody_failures_keep_ordered_turn_keyed_recoveries():
    store = ConsoleChatStore()
    session = store.create_session(title="Recovery", workspace_id="global")
    controller = _CustodyController(fail_before_acceptance=True)
    runtime = ConsoleRuntime(SimpleNamespace())
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    attachments = [
        PendingAttachment(f"/{name}", name, "image", "attachment", data=name.encode())
        for name in ("old.png", "new.png")
    ]

    for index, attachment in enumerate(attachments):
        store.add_pending_attachment(session.id, attachment)
        request = ConsoleTurnCustodyRequest(
            turn_id=f"turn-{index}",
            session_id=session.id,
            draft=f"draft-{index}",
            configuration=_custody_configuration(session.id),
            attachment_ids=(attachment.attachment_id,),
        )
        turn_id = runtime.accept_turn(request)
        with pytest.raises(RuntimeError, match="pre-durable failure"):
            await runtime.wait_for_turn(turn_id)
        await asyncio.sleep(0)

    recoveries = runtime.recoveries_for_session(session.id)
    assert [entry.turn_id for entry in recoveries] == ["turn-0", "turn-1"]
    assert [entry.draft for entry in recoveries] == ["draft-0", "draft-1"]
    assert recoveries[0].attachments == (attachments[0],)
    assert recoveries[1].attachments == (attachments[1],)


@pytest.mark.asyncio
async def test_recovery_refuses_to_merge_an_ambiguous_live_draft_and_keeps_suffix():
    """Recovery is exact: it never concatenates an unrelated newer draft."""
    store = ConsoleChatStore()
    session = store.create_session(title="Recovery", workspace_id="global")
    transferred = PendingAttachment(
        "/old", "old.png", "image", "attachment", data=b"old"
    )
    newer = PendingAttachment(
        "/new", "new.png", "image", "attachment", data=b"new"
    )
    store.add_pending_attachment(session.id, transferred)
    controller = _CustodyController(fail_before_acceptance=True)
    runtime = ConsoleRuntime(SimpleNamespace())
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    request = ConsoleTurnCustodyRequest(
        turn_id="turn-recovery",
        session_id=session.id,
        draft="captured draft",
        configuration=_custody_configuration(session.id),
        attachment_ids=(transferred.attachment_id,),
    )

    turn_id = runtime.accept_turn(request)
    with pytest.raises(RuntimeError, match="pre-durable failure"):
        await runtime.wait_for_turn(turn_id)
    await asyncio.sleep(0)

    store.set_session_draft(session.id, "newer unrelated draft")
    store.add_pending_attachment(session.id, newer)

    with pytest.raises(RuntimeError, match="live draft changed"):
        runtime.restore_turn_recovery(turn_id)

    assert store.session_draft(session.id) == "newer unrelated draft"
    assert store.pending_attachments(session.id) == [newer]
    assert runtime.recoveries_for_session(session.id)[0].attachments == (transferred,)


@pytest.mark.asyncio
async def test_recovery_restores_exact_objects_before_later_attachment_suffix():
    """A compatible recovery restores its original attachment objects in order."""
    store = ConsoleChatStore()
    session = store.create_session(title="Recovery", workspace_id="global")
    transferred = PendingAttachment(
        "/old", "old.png", "image", "attachment", data=b"old"
    )
    newer = PendingAttachment(
        "/new", "new.png", "image", "attachment", data=b"new"
    )
    store.add_pending_attachment(session.id, transferred)
    controller = _CustodyController(fail_before_acceptance=True)
    runtime = ConsoleRuntime(SimpleNamespace())
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    request = ConsoleTurnCustodyRequest(
        turn_id="turn-recovery",
        session_id=session.id,
        draft="captured draft",
        configuration=_custody_configuration(session.id),
        attachment_ids=(transferred.attachment_id,),
    )

    turn_id = runtime.accept_turn(request)
    with pytest.raises(RuntimeError, match="pre-durable failure"):
        await runtime.wait_for_turn(turn_id)
    await asyncio.sleep(0)
    store.add_pending_attachment(session.id, newer)

    restored = runtime.restore_turn_recovery(turn_id)

    assert restored.attachments == (transferred,)
    assert store.session_draft(session.id) == "captured draft"
    assert store.pending_attachments(session.id) == [transferred, newer]
    assert store.pending_attachments(session.id)[0] is transferred
    assert runtime.recoveries_for_session(session.id) == ()


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


def test_capture_freezes_and_redacts_provider_presentation_identity():
    identity = ConsolePresentationContext(
        user_name="Private User",
        assistant_kind="character",
        character_name="Private Character",
        revision=7,
    )

    context = ConsoleTurnConfigurationSnapshot.capture(
        session_id="session-a",
        provider_selection=ConsoleProviderSelection(provider="openai"),
        presentation_context=identity,
    )

    assert context.presentation_context == identity
    assert "Private User" not in repr(context)
    assert "Private Character" not in repr(context)


def test_turn_configuration_authority_destination_and_scratch_repr_are_content_free():
    sentinels = {
        "SYSTEM_PROMPT_SENTINEL",
        "https://secret-endpoint.invalid/v1",
        "/private/scratch/sentinel",
        "SCRATCH_TOKEN_SENTINEL",
        "/private/workspace/sentinel",
        "TRANSFORM_BODY_SENTINEL",
        "SKILL_BODY_SENTINEL",
        "PAYLOAD_SECRET_SENTINEL",
    }
    scratch = ConsoleScratchSnapshot(
        root=Path("/private/scratch/sentinel"),
        token="SCRATCH_TOKEN_SENTINEL",
        identity=(7, 11),
    )
    configuration = ConsoleTurnConfigurationSnapshot.capture(
        session_id="session-a",
        provider_selection=ConsoleProviderSelection(
            provider="openai",
            base_url="https://secret-endpoint.invalid/v1",
            system_prompt="SYSTEM_PROMPT_SENTINEL",
        ),
        scratch_space=scratch,
        session_settings=_settings(
            "openai",
            "model-a",
            "SYSTEM_PROMPT_SENTINEL",
        ),
        workspace_roots=("/private/workspace/sentinel",),
        prompt_transform_inputs={"body": "TRANSFORM_BODY_SENTINEL"},
        skill_context_maximum={"body": "SKILL_BODY_SENTINEL"},
        provider_payload_settings={"header": "PAYLOAD_SECRET_SENTINEL"},
    )
    authority = ConsoleTurnLibraryAuthority(
        policy=ConsoleLibraryPolicySnapshot(
            auto_retrieve=ConsoleAutoRetrieve.NEVER,
            assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
            policy_revision=None,
            source="temporary",
        ),
        direct_library_tools=True,
        source_types=("notes",),
        scope_snapshot=ConsoleLibraryItemScopeSnapshot(
            note_ids=("private-note",),
            media_ids=(),
            conversations_allowed=False,
        ),
        provider_intent=ConsoleProviderIntent(
            provider="openai",
            model="model-a",
            endpoint="https://secret-endpoint.invalid/v1",
        ),
        attempt_id="private-attempt",
    )
    destination = ConsoleResolvedDestination(
        provider="openai",
        model="model-a",
        endpoint_identity="https://secret-endpoint.invalid/v1",
        egress_class=ConsoleEgressClass.PUBLIC_NETWORK,
    )
    execution_context = ConsoleTurnExecutionContext(
        configuration,
        authority,
        destination,
    )

    rendered = "\n".join(
        repr(value)
        for value in (
            scratch,
            configuration,
            authority,
            destination,
            execution_context,
        )
    )
    assert all(sentinel not in rendered for sentinel in sentinels)


@pytest.mark.asyncio
async def test_character_emote_authority_cannot_retarget_after_handoff():
    store = ConsoleChatStore()
    session = store.create_session(workspace_id="workspace-a")
    session.assistant_kind = "character"
    session.runtime_backend = "local"
    session.assistant_id = "7"
    session.assistant_authority_id = "character:7"
    session.character_id = 7
    session.identity_revision = 3
    controller = ConsoleChatController(store=store, provider_gateway=_PausedGateway())
    configuration = controller.resolve_runtime_turn_configuration_snapshot(session.id)
    context = ConsoleTurnExecutionContext(configuration, _authority(), _destination())

    session.assistant_id = "8"
    session.assistant_authority_id = "character:8"
    session.character_id = 8
    session.identity_revision = 4

    with pytest.raises(RuntimeError):
        await controller._character_emote_snapshot_for_run(session.id, context)

    later_configuration = controller.resolve_runtime_turn_configuration_snapshot(
        session.id
    )
    later_context = ConsoleTurnExecutionContext(
        later_configuration,
        _authority(),
        _destination(),
    )
    later = await controller._character_emote_snapshot_for_run(
        session.id,
        later_context,
    )

    assert later is not None
    assert later.actor_id == 8


@pytest.mark.asyncio
async def test_character_emote_pack_is_frozen_until_the_next_turn():
    class Repository:
        def __init__(self):
            self.graph = self.pack(11, 13, 17, "happy")

        @staticmethod
        def pack(pack_id, version_id, asset_id, expression_key):
            return {
                "pack": {"id": pack_id},
                "version": {"id": version_id},
                "assets": [
                    {"id": asset_id, "expression_key": expression_key}
                ],
            }

        def get_active_actor_pack(self, actor_kind, actor_id):
            assert (actor_kind, actor_id) == ("character", 7)
            return self.graph

    store = ConsoleChatStore()
    session = store.create_session(workspace_id="workspace-a")
    session.assistant_kind = "character"
    session.runtime_backend = "local"
    session.assistant_id = "7"
    session.assistant_authority_id = "character:7"
    session.character_id = 7
    session.identity_revision = 3
    repository = Repository()
    controller = ConsoleChatController(store=store, provider_gateway=_PausedGateway())
    controller._visual_identity_repository = repository
    old_configuration = controller.resolve_runtime_turn_configuration_snapshot(
        session.id
    )
    old_context = ConsoleTurnExecutionContext(
        old_configuration, _authority(), _destination()
    )

    repository.graph = repository.pack(21, 23, 27, "sad")

    old_snapshot = await controller._character_emote_snapshot_for_run(
        session.id, old_context
    )
    new_configuration = controller.resolve_runtime_turn_configuration_snapshot(
        session.id
    )
    new_snapshot = await controller._character_emote_snapshot_for_run(
        session.id,
        ConsoleTurnExecutionContext(new_configuration, _authority(), _destination()),
    )

    assert old_snapshot is not None
    assert (old_snapshot.pack_id, old_snapshot.pack_version_id) == (11, 13)
    assert old_snapshot.states == ("happy",)
    assert old_snapshot.assets[0].asset_id == 17
    assert new_snapshot is not None
    assert (new_snapshot.pack_id, new_snapshot.pack_version_id) == (21, 23)
    assert new_snapshot.states == ("sad",)
    assert new_snapshot.assets[0].asset_id == 27


def test_preparation_and_continuation_sensitive_fields_are_repr_hidden():
    from tldw_chatbook.Chat import console_chat_controller as controller_module

    expected = {
        ConsoleTurnPreparation: {"executed_draft", "execution_context"},
        controller_module._PreparedEvidenceLease: {
            "launch",
            "capture",
            "release",
            "capture_result",
        },
        controller_module._PreparedSendContinuation: {
            "attachments",
            "prefill",
            "staged_evidence",
        },
        controller_module._DurablePostcommitContinuation: {
            "clean_draft",
            "resolution",
            "provider_messages",
            "prefill",
            "skill_bindings",
            "skill_bundle_block",
            "citation_repair_session",
            "turn_context",
            "prepared",
        },
    }

    for owner, sensitive in expected.items():
        repr_fields = {item.name for item in fields(owner) if item.repr}
        assert sensitive.isdisjoint(repr_fields), owner.__name__


@pytest.mark.asyncio
async def test_library_policy_and_scope_can_only_narrow_after_handoff():
    store = ConsoleChatStore()
    session = store.create_session(workspace_id="workspace-a")
    frozen_policy = ConsoleLibraryPolicySnapshot(
        auto_retrieve=ConsoleAutoRetrieve.NEVER,
        assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
        policy_revision=1,
        source="durable",
    )
    live_policy = ConsoleLibraryPolicySnapshot(
        auto_retrieve=ConsoleAutoRetrieve.AUTOMATIC,
        assistant_access=ConsoleAssistantLibraryAccess.ALLOWED,
        policy_revision=2,
        source="durable",
    )
    session.library_policy_holder.snapshot = frozen_policy
    frozen_scope = ConsoleLibraryItemScopeSnapshot(("note-a",), (), False)
    configuration = ConsoleTurnConfigurationSnapshot.capture(
        session_id=session.id,
        provider_selection=ConsoleProviderSelection(provider="openai"),
        library_policy_maximum=frozen_policy,
        library_scope_maximum=frozen_scope,
    )

    class Coordinator:
        async def capture_for_execution(self, _session_id):
            return live_policy

    store.library_policy_coordinator = Coordinator()
    session.library_policy_holder.snapshot = live_policy
    from tldw_chatbook.Chat.rag_scope import RagScope, ScopeItem

    session.rag_scope_holder.set(
        RagScope((ScopeItem("note", "note-b"),), "later")
    )
    controller = ConsoleChatController(store=store, provider_gateway=_PausedGateway())

    first = await controller._capture_turn_library_authority(
        session.id,
        configuration,
    )
    later_configuration = ConsoleTurnConfigurationSnapshot.capture(
        session_id=session.id,
        provider_selection=configuration.provider_selection,
        library_policy_maximum=live_policy,
        library_scope_maximum=ConsoleLibraryItemScopeSnapshot(
            ("note-b",), (), False
        ),
    )
    later = await controller._capture_turn_library_authority(
        session.id,
        later_configuration,
    )

    assert first.policy.auto_retrieve is ConsoleAutoRetrieve.NEVER
    assert first.policy.assistant_access is ConsoleAssistantLibraryAccess.BLOCKED
    assert first.policy.policy_revision == 1
    assert first.scope_snapshot == ConsoleLibraryItemScopeSnapshot((), (), False)
    assert later.policy.assistant_access is ConsoleAssistantLibraryAccess.ALLOWED
    assert later.scope_snapshot.note_ids == ("note-b",)


def test_project_binding_authority_is_frozen_and_later_retarget_only_revokes(tmp_path):
    store = ConsoleChatStore()
    session = store.create_session(workspace_id="workspace-a")
    session.project_instruction_state = ProjectInstructionControlState(
        project_instructions_enabled=True,
        working_folder_binding_id=None,
        working_folder_locator_fingerprint=None,
        project_instruction_notice_key=None,
    )
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()

    def binding(binding_id, root, access):
        return SimpleNamespace(
            binding_id=binding_id,
            workspace_id="workspace-a",
            display_name=binding_id,
            binding_kind=SimpleNamespace(value="local-filesystem"),
            status=SimpleNamespace(value="ready"),
            locator=str(root),
            metadata={"access": access},
        )

    current = [binding("binding-a", root_a, "ro")]

    class Registry:
        def list_runtime_bindings(self, _workspace_id):
            return tuple(current)

        def get_runtime_binding(self, binding_id):
            return next(item for item in current if item.binding_id == binding_id)

    registry = Registry()
    first = capture_project_instruction_authority(session, registry)

    current[:] = [binding("binding-b", root_b, "rw")]

    assert first.selected.binding_id == "binding-a"
    assert first.selected.root == str(root_a)
    assert first.selected.allow_write is False
    assert not project_instruction_authority_is_current(
        store=store,
        session_id=session.id,
        registry=registry,
        expected_selection=SimpleNamespace(
            binding=first.selected,
            root=root_a,
            locator_fingerprint=first.selected.locator_fingerprint,
            allow_write=first.selected.allow_write,
            root_identity=first.selected.root_identity,
        ),
    )
    later = capture_project_instruction_authority(session, registry)
    assert later.selected.binding_id == "binding-b"
    assert later.selected.allow_write is True


def test_frozen_workspace_binding_maximum_excludes_later_roots(
    tmp_path,
    monkeypatch,
):
    from tldw_chatbook.Tools import workspace_file_roots as roots_module

    sandbox = tmp_path / "sandbox"
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    for root in (sandbox, root_a, root_b):
        root.mkdir()

    bindings = [
        SimpleNamespace(
            binding_id="binding-a",
            locator=str(root_a),
            metadata={"access": "ro"},
        )
    ]
    registry = SimpleNamespace(
        list_folder_bindings=lambda _workspace_id: tuple(bindings),
        get_workspace=lambda _workspace_id: SimpleNamespace(name="Workspace A"),
    )
    monkeypatch.setattr(roots_module, "_registry_factory", lambda: registry)
    frozen_authority = (
        ConsoleProjectBindingSnapshot(
            binding_id="binding-a",
            workspace_id="workspace-a",
            display_name="A",
            root=root_a,
            locator_fingerprint=fingerprint_canonical_locator(str(root_a)),
            allow_write=False,
            root_identity=_capture_project_root_identity(root_a),
        ),
    )

    with roots_module.run_workspace(
        "workspace-a",
        read_binding_ids=("binding-a",),
        write_binding_ids=(),
        binding_authority=frozen_authority,
    ):
        assert roots_module.allowed_file_roots(
            write=False,
            sandbox_root=sandbox,
        ) == (sandbox, root_a)
        assert roots_module.allowed_file_roots(
            write=True,
            sandbox_root=sandbox,
        ) == (sandbox,)
        initial_note = roots_module.workspace_context_note(
            "workspace-a",
            launch_cwd=tmp_path,
            registry=registry,
            binding_authority=frozen_authority,
        )
        assert "  - a (read-only)" in initial_note
        assert roots_module.frozen_workspace_roots(
            "workspace-a", frozen_authority, registry=registry
        ) == (root_a,)
        # A later binding is outside this turn's maximum.
        bindings.append(
            SimpleNamespace(
                binding_id="binding-b",
                locator=str(root_b),
                metadata={"access": "rw"},
            )
        )
        assert roots_module.allowed_file_roots(
            write=False,
            sandbox_root=sandbox,
        ) == (sandbox, root_a)
        assert roots_module.allowed_file_roots(
            write=True,
            sandbox_root=sandbox,
        ) == (sandbox,)
        assert "  - b" not in roots_module.workspace_context_note(
            "workspace-a",
            launch_cwd=tmp_path,
            registry=registry,
            binding_authority=frozen_authority,
        )
        assert roots_module.frozen_workspace_roots(
            "workspace-a", frozen_authority, registry=registry
        ) == (root_a,)

        # Reusing the captured ID for another locator revokes the old turn;
        # it never retargets that turn to the new folder.
        bindings[:] = [
            SimpleNamespace(
                binding_id="binding-a",
                locator=str(root_b),
                metadata={"access": "rw"},
            )
        ]
        assert roots_module.allowed_file_roots(
            write=False,
            sandbox_root=sandbox,
        ) == (sandbox,)
        assert roots_module.allowed_file_roots(
            write=True,
            sandbox_root=sandbox,
        ) == (sandbox,)
        assert roots_module.frozen_workspace_roots(
            "workspace-a", frozen_authority, registry=registry
        ) == ()

        # Restoring the exact locator may restore read access, but a live
        # RO->RW mutation cannot broaden the frozen turn's write authority.
        bindings[0].locator = str(root_a)
        assert roots_module.allowed_file_roots(
            write=False,
            sandbox_root=sandbox,
        ) == (sandbox, root_a)
        assert roots_module.allowed_file_roots(
            write=True,
            sandbox_root=sandbox,
        ) == (sandbox,)

        # Removal stays fail-closed, and adding a different ID cannot replace
        # the exact captured binding.
        bindings.clear()
        assert roots_module.allowed_file_roots(
            write=False,
            sandbox_root=sandbox,
        ) == (sandbox,)
        bindings.append(
            SimpleNamespace(
                binding_id="binding-b",
                locator=str(root_b),
                metadata={"access": "rw"},
            )
        )
        assert roots_module.allowed_file_roots(
            write=False,
            sandbox_root=sandbox,
        ) == (sandbox,)


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


def test_final_context_preserves_change_review_admission_snapshot():
    from tldw_chatbook.Chat.console_endpoint_provenance import ConsoleEndpointProvenance

    skipped = SkippedReviewRoot(alias="folder-busy", reason="Preparing history")
    persona_rules = [{"tool": "fs_write", "decision": "deny"}]
    configuration = ConsoleTurnConfigurationSnapshot.capture(
        session_id="session-a",
        provider_selection=ConsoleProviderSelection(
            provider="openai",
            configured_endpoint_fallback_allowed=False,
            endpoint_provenance=ConsoleEndpointProvenance.EPHEMERAL_SESSION,
        ),
        change_review_root_aliases=("folder-ready",),
        change_review_skipped_roots=(skipped,),
        persona_policy_rules=persona_rules,
        tool_policy_profile_id="restricted",
    )
    persona_rules[0]["decision"] = "allow"
    authority = _authority()
    authority = replace(authority, provider_intent=replace(
        authority.provider_intent,
        endpoint_provenance=ConsoleEndpointProvenance.EPHEMERAL_SESSION,
    ))

    context = ConsoleTurnExecutionContext(
        configuration=configuration,
        library_authority=authority,
        resolved_destination=replace(
            _destination(), endpoint_provenance=ConsoleEndpointProvenance.EPHEMERAL_SESSION
        ),
    )

    assert context.change_review_root_aliases == ("folder-ready",)
    assert context.change_review_skipped_roots == (skipped,)
    assert context.persona_policy_rules[0]["decision"] == "deny"
    assert context.tool_policy_profile_id == "restricted"
    assert context.provider_selection.configured_endpoint_fallback_allowed is False
    assert context.provider_selection.endpoint_provenance is ConsoleEndpointProvenance.EPHEMERAL_SESSION
    assert context.library_authority.provider_intent.endpoint_provenance is ConsoleEndpointProvenance.EPHEMERAL_SESSION
    assert context.resolved_destination.endpoint_provenance is ConsoleEndpointProvenance.EPHEMERAL_SESSION


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


@pytest.mark.asyncio
async def test_prompt_transforms_use_frozen_inputs_then_later_turn_uses_new_inputs():
    store = ConsoleChatStore()
    session = store.create_session(workspace_id="workspace-a")
    session.persisted_conversation_id = "conversation-new"
    gateway = _PausedGateway()
    gateway.release_resolve.set()

    def dictionary_applier(_conversation_id, text, frozen=None):
        return f"{frozen['dictionary_marker']}:{text}" if frozen else text

    def world_applier(_conversation_id, text, _history, frozen=None):
        return f"{frozen['world_marker']}:{text}" if frozen else text

    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        agent_runtime_enabled=False,
        chat_dictionary_applier=dictionary_applier,
        world_info_applier=world_applier,
    )
    old = ConsoleTurnConfigurationSnapshot.capture(
        session_id=session.id,
        provider_selection=ConsoleProviderSelection(
            provider="openai", explicit_model="model-a"
        ),
        prompt_transform_inputs={
            "conversation_id": "conversation-old",
            "dictionary_marker": "DICT-OLD",
            "world_marker": "WORLD-OLD",
        },
        tool_configuration={"agent_runtime_enabled": False},
    )
    new = ConsoleTurnConfigurationSnapshot.capture(
        session_id=session.id,
        provider_selection=old.provider_selection,
        prompt_transform_inputs={
            "conversation_id": "conversation-new",
            "dictionary_marker": "DICT-NEW",
            "world_marker": "WORLD-NEW",
        },
        tool_configuration={"agent_runtime_enabled": False},
    )

    assert (
        await controller.submit_draft(
            "first",
            session_id=session.id,
            configuration=old,
        )
    ).accepted
    assert (
        await controller.submit_draft(
            "second",
            session_id=session.id,
            configuration=new,
        )
    ).accepted

    assert "WORLD-OLD:DICT-OLD:first" in repr(gateway.message_batches[0])
    assert "WORLD-NEW:DICT-NEW:second" in repr(gateway.message_batches[1])


@pytest.mark.asyncio
async def test_skill_catalog_maximum_excludes_a_skill_added_after_handoff():
    class Skills:
        def __init__(self):
            self.get_context_calls = 0
            self.execute_calls = 0

        async def get_context(self, *, mode):
            assert mode == "local"
            self.get_context_calls += 1
            return {
                "available_skills": [{"name": "new", "user_invocable": True}],
                "blocked_skills": [],
            }

        async def execute_skill(self, name, *, mode, args):
            self.execute_calls += 1
            return {
                "skill_name": name,
                "execution_mode": "inline",
                "rendered_prompt": f"rendered:{name}:{args}",
            }

    store = ConsoleChatStore()
    session = store.create_session(workspace_id="workspace-a")
    skills = Skills()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=_PausedGateway(),
        skills_service=skills,
    )

    class FailingLocal:
        def _load_index(self):
            raise RuntimeError("capture unavailable")

    failed_capture = capture_skill_context_maximum(
        SimpleNamespace(
            skills_scope_service=SimpleNamespace(local_service=FailingLocal())
        )
    )

    def context_with(name):
        configuration = ConsoleTurnConfigurationSnapshot.capture(
            session_id=session.id,
            provider_selection=ConsoleProviderSelection(provider="openai"),
            skill_context_maximum={
                "backend": "local",
                "available_skills": (
                    ({"name": name, "user_invocable": True},) if name else ()
                ),
                "blocked_skills": (),
            },
        )
        return ConsoleTurnExecutionContext(configuration, _authority(), _destination())

    failed_configuration = ConsoleTurnConfigurationSnapshot.capture(
        session_id=session.id,
        provider_selection=ConsoleProviderSelection(provider="openai"),
        skill_context_maximum=failed_capture,
    )
    first = await controller._apply_skill_substitution(
        [{"role": "user", "content": "$new args"}],
        ConsoleTurnExecutionContext(
            failed_configuration, _authority(), _destination()
        ),
    )
    later = await controller._apply_skill_substitution(
        [{"role": "user", "content": "$new args"}],
        context_with("new"),
    )

    assert first[0][-1]["content"] == "$new args"
    assert later[0][-1]["content"] == "rendered:new:args"
    assert skills.get_context_calls == 0
    assert skills.execute_calls == 1


def test_skill_context_capture_failure_is_explicitly_local_and_empty():
    class FailingLocal:
        def _load_index(self):
            raise RuntimeError("capture unavailable")

    captured = capture_skill_context_maximum(
        SimpleNamespace(
            skills_scope_service=SimpleNamespace(local_service=FailingLocal())
        )
    )

    assert captured == {
        "backend": "local",
        "available_skills": (),
        "blocked_skills": (),
        "context_text": "",
    }


@pytest.mark.asyncio
async def test_direct_skill_substitution_rejects_same_name_definition_mutation():
    class Skills:
        def __init__(self):
            self.digest = "digest-a"
            self.body = "body-a"
            self.reference_files = [
                {"path": "references/a.md", "size": 1, "is_text": True}
            ]
            self.allowed_tools = ["calculator"]
            self.trust_service = self
            self.executions = []

        def current_fingerprint_digest(self, _name):
            return self.digest

        async def execute_skill(self, name, *, mode, args):
            self.executions.append((name, tuple(self.allowed_tools)))
            return {
                "skill_name": name,
                "execution_mode": "inline",
                "rendered_prompt": self.body,
                "allowed_tools": self.allowed_tools,
                "reference_files": self.reference_files,
            }

    store = ConsoleChatStore()
    session = store.create_session(workspace_id="workspace-a")
    skills = Skills()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=_PausedGateway(),
        skills_service=skills,
    )

    def context_with(digest):
        configuration = ConsoleTurnConfigurationSnapshot.capture(
            session_id=session.id,
            provider_selection=ConsoleProviderSelection(provider="openai"),
            skill_context_maximum={
                "backend": "local",
                "available_skills": (
                    {
                        "name": "review",
                        "user_invocable": True,
                        "definition_digest": digest,
                    },
                ),
                "blocked_skills": (),
            },
        )
        return ConsoleTurnExecutionContext(configuration, _authority(), _destination())

    old_context = context_with("digest-a")
    skills.digest = "digest-b"
    skills.body = "body-b"
    skills.reference_files = [
        {"path": "references/b.md", "size": 2, "is_text": True}
    ]
    skills.allowed_tools = ["datetime"]

    refused = await controller._apply_skill_substitution(
        [{"role": "user", "content": "$review args"}], old_context
    )
    accepted = await controller._apply_skill_substitution(
        [{"role": "user", "content": "$review args"}],
        context_with("digest-b"),
    )

    assert refused[1] is not None
    assert "skill_definition_changed" in refused[1]
    assert accepted[0][-1]["content"] == "body-b"
    assert "references/b.md (2 bytes)" in accepted[4]
    assert skills.executions == [("review", ("datetime",))]


@pytest.mark.asyncio
async def test_mcp_catalog_is_intersected_with_handoff_maximum():
    class Service:
        def __init__(self):
            self.name = "new"
            self.local_service = None

        def get_kill_switch(self):
            return False

        async def local_external_catalog(self):
            return [
                {
                    "profile_id": "server",
                    "is_connected": True,
                    "discovery_snapshot": {
                        "tools": [
                            {
                                "name": self.name,
                                "description": self.name,
                                "inputSchema": {"type": "object"},
                            }
                        ]
                    },
                }
            ]

        def effective_tool_states(self, tools):
            return {
                (tool.server_key, tool.name): SimpleNamespace(state="ask")
                for tool in tools
            }

    service = Service()
    first = MCPToolProvider(
        service=service,
        main_loop=asyncio.get_running_loop(),
        maximum_tool_ids=frozenset({"local:server::old"}),
    )
    await first.compose_catalog()
    later = MCPToolProvider(
        service=service,
        main_loop=asyncio.get_running_loop(),
        maximum_tool_ids=frozenset({"local:server::new"}),
    )
    await later.compose_catalog()

    assert first.list_catalog() == []
    assert [entry.name for entry in later.list_catalog()] == ["mcp__server__new"]


@pytest.mark.asyncio
async def test_automatic_library_preparation_uses_frozen_rag_defaults():
    calls = []

    class Rag:
        async def search(self, query, source_types, mode, **kwargs):
            calls.append((query, tuple(source_types), mode, kwargs))
            return {"results": []}

    store = ConsoleChatStore()
    session = store.create_session(workspace_id="workspace-a")
    configuration = ConsoleTurnConfigurationSnapshot.capture(
        session_id=session.id,
        provider_selection=ConsoleProviderSelection(provider="openai"),
        rag_defaults={"source_types": ("notes",), "top_k": 17},
    )
    authority = ConsoleTurnLibraryAuthority(
        policy=ConsoleLibraryPolicySnapshot(
            auto_retrieve=ConsoleAutoRetrieve.AUTOMATIC,
            assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
            policy_revision=1,
            source="durable",
        ),
        direct_library_tools=True,
        source_types=AUTOMATIC_LIBRARY_SOURCE_TYPES,
        scope_snapshot=ConsoleLibraryItemScopeSnapshot((), (), True),
        provider_intent=ConsoleProviderIntent("openai", None, None),
        attempt_id="attempt-rag",
    )
    context = ConsoleTurnExecutionContext(configuration, authority, _destination())
    preparation = ConsoleTurnPreparation(
        preparation_id="preparation-rag",
        attempt_id="attempt-rag",
        session_id=session.id,
        origin="manual",
        queue_entry_id=None,
        executed_draft="frozen query",
        execution_context=context,
        transient_user_message_id=None,
        attachment_ids=(),
        evidence_ids=(),
        prefill_id=None,
        queue_generation=None,
        pre_send_title="Chat",
        pre_send_conversation_id=None,
        state=ConsoleTurnPreparationState.PREPARING,
        pause_kind=None,
        one_shot_bypass=False,
        ephemeral=False,
    )
    assert store.begin_preparation(preparation) is preparation
    controller = ConsoleChatController(store=store, provider_gateway=_PausedGateway())
    controller.app = SimpleNamespace(library_rag_search_service=Rag())

    outcome = await controller.prepare_library_for_turn(preparation.preparation_id)

    assert outcome.state is ConsoleTurnPreparationState.READY
    assert calls == [
        (
            "frozen query",
            ("notes",),
            "rag",
            {"top_k": 17, "include_citations": True},
        )
    ]


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
        "chat_defaults": {
            "rag_auto_retrieve_on_send": "true",
            "user_display_name": "Frozen User",
        },
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
    app_config["chat_defaults"]["user_display_name"] = "Later User"

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
    assert context.presentation_context.user_name == "Frozen User"
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
async def test_summarize_and_rag_capture_receive_the_owning_turn_context(tmp_path):
    from Tests.Chat.test_console_rewind_summarize import SummaryGateway
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    # Current dev summaries commit durable branch memory; use its real
    # repository and prepared-request fake, not the source's old db-less path.
    class CapturedSummaryGateway(_PausedGateway, SummaryGateway):
        def __init__(self):
            _PausedGateway.__init__(self)
            SummaryGateway.__init__(self)

        async def resolve_for_send(self, selection):
            from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderResolution

            resolution = await _PausedGateway.resolve_for_send(self, selection)
            return ConsoleProviderResolution(**vars(resolution))

    db = CharactersRAGDB(tmp_path / "context-summary.sqlite", "context-summary")
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    session = store.create_session(
        title="Summary",
        settings=_settings("openai", "model-a", "stored-system"),
    )
    store.persist_session_if_needed(session.id)
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="first question " + "detail " * 30,
        persist=True,
    )
    store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="first answer " + "detail " * 30,
        persist=True,
    )
    boundary = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="second question",
        persist=True,
    )
    store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT,
        content="second answer", persist=True,
    )
    context = ConsoleTurnConfigurationSnapshot.capture(
        session_id=session.id,
        provider_selection=ConsoleProviderSelection(
            provider="openai",
            explicit_model="captured-model",
            system_prompt="captured-system",
        ),
        session_settings=store.session_settings(session.id),
        rag_defaults={"auto_retrieve_on_send": False},
        tool_configuration={"agent_runtime_enabled": False},
    )
    gateway = CapturedSummaryGateway()
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

    assert all(row.persisted_message_id for row in store.messages_for_session(session.id))
    assert len(controller._durable_context_snapshots(session.id)) == 4
    assert store.active_session_id == session.id

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

    nonvision = ConsoleTurnConfigurationSnapshot.capture(
        session_id=session.id,
        provider_selection=context.provider_selection,
        session_settings=store.session_settings(session.id),
        capabilities={"vision": False, "max_history_images": 0},
        tool_configuration={"agent_runtime_enabled": False},
    )
    later = await controller.submit_draft(
        "follow up",
        session_id=session.id,
        configuration=nonvision,
    )

    assert later.accepted is True
    later_payload = gateway.message_batches[1]
    assert later_payload[0]["content"] == "describe"
    assert "image_url" not in repr(later_payload)
    forbidden_provider_keys = {
        "_native_message_id",
        "turn_id",
        "attention_id",
        "queue_entry_id",
    }
    assert all(
        forbidden_provider_keys.isdisjoint(row)
        for row in gateway.message_batches[0] + later_payload
    )


@pytest.mark.asyncio
async def test_provider_payload_uses_handoff_identity_then_next_turn_uses_new_identity():
    store = ConsoleChatStore()
    session = store.create_session(
        title="Identity",
        workspace_id="workspace-a",
        settings=_settings("openai", "model-a", None),
        assistant_kind="character",
        character_name="Old Character",
    )
    store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Hello Old User from Old Character.",
        metadata=MessageMetadata(
            template_kind="character_greeting",
            template_source="Hello {{user}} from {{char}}.",
        ),
    )
    first_configuration = ConsoleTurnConfigurationSnapshot.capture(
        session_id=session.id,
        provider_selection=ConsoleProviderSelection(
            provider="openai",
            explicit_model="model-a",
            workspace_context=ConsoleWorkspaceContext(
                active_workspace_id="workspace-a"
            ),
        ),
        session_settings=store.session_settings(session.id),
        capabilities={"vision": False},
        tool_configuration={"agent_runtime_enabled": False},
        presentation_context=store.presentation_context(session.id, "Old User"),
    )
    gateway = _PausedGateway()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        agent_runtime_enabled=False,
        global_user_display_name=lambda: "New User",
    )

    first_turn = asyncio.create_task(
        controller.submit_draft(
            "first question",
            session_id=session.id,
            configuration=first_configuration,
        )
    )
    await gateway.resolve_started.wait()
    session.character_name = "New Character"
    gateway.release_resolve.set()

    first_result = await first_turn

    assert first_result.accepted is True
    first_system = gateway.message_batches[0][0]["content"]
    assert "Hello Old User from Old Character." in first_system
    assert "New User" not in first_system
    assert "New Character" not in first_system

    second_configuration = ConsoleTurnConfigurationSnapshot.capture(
        session_id=session.id,
        provider_selection=first_configuration.provider_selection,
        session_settings=store.session_settings(session.id),
        capabilities={"vision": False},
        tool_configuration={"agent_runtime_enabled": False},
        presentation_context=store.presentation_context(session.id, "New User"),
    )
    second_result = await controller.submit_draft(
        "second question",
        session_id=session.id,
        configuration=second_configuration,
    )

    assert second_result.accepted is True
    second_system = gateway.message_batches[1][0]["content"]
    assert "Hello New User from New Character." in second_system


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
    fake_screen._build_console_provider_selection_from_settings = (
        lambda *args, **kwargs: ChatScreen._build_console_provider_selection_from_settings(
            fake_screen, *args, **kwargs
        )
    )

    selection = ChatScreen._build_console_provider_selection(fake_screen, first.id)

    assert selection.provider == "openai"
    assert selection.explicit_model == "model-a"
    assert selection.configured_model == "configured-a"
    assert (selection.explicit_model or selection.configured_model) == "model-a"
    assert selection.system_prompt == "system-a"
    assert selection.workspace_context.active_workspace_id == "workspace-a"
    assert store.active_session_id != first.id
