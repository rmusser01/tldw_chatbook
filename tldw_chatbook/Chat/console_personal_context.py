"""Personal Context composition and disposable preview under current Console owners."""

from __future__ import annotations

import asyncio
import copy
import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tldw_chatbook.Agents.profile_tool_provider import ProfileToolProvider
    from tldw_chatbook.Personal_Context.context_service import (
        ProfileContextRequest,
        ProfileContextSelectionExplanation,
        ProfileContextService,
        ProfileContextSnapshot,
    )
    from tldw_chatbook.Personal_Context.service import PersonalContextService

    from .console_chat_models import ConsoleChatMessage, ConsoleChatSession
    from .console_provider_gateway import ConsoleProviderSelection


def capture_personal_context_preview_guard(self, session_id: str) -> Callable[[], bool]:
    """Fence disposable profile inspection across every later await boundary."""
    try:
        session = next(item for item in self.store.sessions() if item.id == session_id)
        temporary = session.ephemeral
        controller = self.store.canvas_turn_controller
        enabled = self._canvas_enabled_reader() is True
        scope = None
        if enabled and controller is not None:
            from tldw_chatbook.Canvas.models import CanvasScope

            scope = CanvasScope(
                session_id,
                self._agent_conversation_id(session_id),
                self.store.canvas_active_path_message_ids(session_id),
                None,
                None,
                f"preview:{session_id}",
            )
            owner = controller.capture_interactive_owner(scope, temporary=temporary)
    except Exception:  # noqa: BLE001 - an unavailable owner denies inspection.
        return lambda: False

    def current() -> bool:
        try:
            if (
                next(
                    (item for item in self.store.sessions() if item.id == session_id),
                    None,
                )
                is not session
                or session.ephemeral != temporary
                or self.store.canvas_turn_controller is not controller
                or (self._canvas_enabled_reader() is True) != enabled
            ):
                return False
            if scope is not None:
                controller.validate_interactive_owner(scope, owner, temporary=temporary)
                return (
                    self._agent_conversation_id(session_id) == scope.conversation_id
                    and self.store.canvas_active_path_message_ids(session_id)
                    == scope.active_message_ids
                )
            return True
        except Exception:  # noqa: BLE001 - stale inspection fails closed.
            return False

    return current


async def personal_context_selection_current(
    self,
    builder: ProfileContextService,
    request: ProfileContextRequest,
    explanation: ProfileContextSelectionExplanation,
    *,
    provider_selection: ConsoleProviderSelection,
) -> bool:
    """Recheck the exact preview builder and current app-owned profile."""

    try:
        resolution = await self.provider_gateway.resolve_for_send(provider_selection)
        if not getattr(resolution, "ready", True):
            return False
        model = str(
            getattr(resolution, "model", "")
            or provider_selection.explicit_model
            or provider_selection.configured_model
            or ""
        )
        provider = str(
            getattr(resolution, "execution_key", "")
            or getattr(resolution, "provider", "")
            or "agent"
        )
        if (model, provider) != (request.model, request.provider):
            return False
        current_service = await self._personal_context_service()
    except Exception:  # noqa: BLE001 - stale inspection fails closed.
        return False
    if current_service is None or current_service is not builder._service:
        return False
    if not await asyncio.to_thread(
        builder.explanation_is_current, explanation, request
    ):
        return False
    try:
        # The app may replace its service while the read-only validation
        # runs in a worker. Compare again at the final await boundary.
        return await self._personal_context_service() is builder._service
    except Exception:  # noqa: BLE001 - stale inspection fails closed.
        return False


async def _build_personal_context_snapshot(
    self,
    session: ConsoleChatSession | None,
    provider_messages: list[dict[str, Any]],
    *,
    provider_selection: ConsoleProviderSelection,
    turn_configuration: Any | None = None,
    turn_skill_bindings: tuple[str, ...] = (),
    turn_bundle_block: str = "",
    profile_selection_sink: Callable[
        [
            ProfileContextService,
            ProfileContextRequest,
            ProfileContextSelectionExplanation,
        ],
        None,
    ]
    | None = None,
) -> ProfileContextSnapshot:
    """Ask the agent planner for one fully reserved Next Send snapshot."""

    from . import console_chat_controller as owner

    personal_context_service = await self._personal_context_service()
    builder = await self._personal_context_builder(personal_context_service)
    bridge = self._agent_bridge
    build_preview = getattr(bridge, "build_personal_context_preview_snapshot", None)
    if builder is None or session is None or not callable(build_preview):
        return owner._empty_profile_context_snapshot()
    try:
        resolution = await self._resolve_for_send_bounded(provider_selection)
        if not getattr(resolution, "ready", True):
            return owner._empty_profile_context_snapshot()
        profile_provider = await asyncio.to_thread(
            owner._compose_profile_tool_provider,
            personal_context_service,
            workspace_id=session.workspace_id,
            ephemeral=session.ephemeral,
            run_id=f"preview:{session.id}",
            session_id=session.id,
            current_user_message=None,
            kill_switch=(self._console_tool_kill_switch_reader() or (lambda: False)),
            reserve_direct_update_schema=True,
        )
        configuration = (
            turn_configuration
            if turn_configuration is not None
            else self.resolve_turn_execution_context(session.id)
        )
        canvas_controller = getattr(self.store, "canvas_turn_controller", None)
        canvas_profile_snapshot = None
        try:
            canvas_enabled = self._canvas_enabled_reader() is True
        except Exception:  # noqa: BLE001 - match live fail-closed enablement
            canvas_enabled = False
        if canvas_enabled and canvas_controller is not None:
            canvas_profile_snapshot = canvas_controller.profile_snapshot
        library_provider: Any | None = None
        library_authority: Any | None = None
        if self._library_provider_factory is not None:
            try:
                captured_authority = await self._capture_turn_library_authority(
                    session.id,
                    configuration,
                )
                complete_context = self._finalize_turn_execution_context(
                    configuration,
                    captured_authority,
                    resolution,
                )
                library_selection = self._library_provider_for_context(complete_context)
                if library_selection is not None:
                    library_provider, library_authority = library_selection
            except Exception:  # noqa: BLE001 - match live fail-soft semantics
                owner.logger.opt(exception=True).warning(
                    "library_provider_factory failed; previewing without Library tools"
                )
        project_selection = None
        if session.project_instruction_state.project_instructions_enabled:
            try:
                registry = getattr(self.app, "workspace_registry_service", None)
                project_selection = owner.resolve_project_instruction_binding(
                    session,
                    registry,
                    status_cache=owner._app_remote_binding_status_cache(),
                )
            except Exception:  # noqa: BLE001 - preview cannot repair authority
                project_selection = None
        scratch_snapshot = configuration.scratch_space
        scratch_lease = (
            functools.partial(self._scratch_spaces.lease, scratch_snapshot)
            if scratch_snapshot is not None
            else None
        )
        (
            mcp_provider,
            builtin_gate,
            local_provider,
            _review_hook,
        ) = await self._compose_agent_request_providers(
            session_id=session.id,
            project_selection=project_selection,
            project_authority_guard=None,
            turn_context=configuration,
            publish_mcp_counts=False,
        )
        preview_admitted_roots = owner.capture_run_admitted_workspace_roots(
            session=session,
            registry=getattr(self.app, "workspace_registry_service", None),
            project_selection=project_selection,
        )
        virtual_cli_provider, _virtual_cli_review_hook = (
            self._compose_virtual_cli_provider(
                session_id=session.id,
                turn_context=configuration,
                project_root=(project_selection.root if project_selection else None),
                project_root_identity=(
                    project_selection.root_identity if project_selection else None
                ),
                admitted_roots=preview_admitted_roots,
            )
        )
        raw_shell_provider, _raw_shell_review_hook = self._compose_raw_shell_provider(
            session_id=session.id,
            turn_context=configuration,
            project_root=(project_selection.root if project_selection else None),
        )
        model = str(
            getattr(resolution, "model", "")
            or provider_selection.explicit_model
            or provider_selection.configured_model
            or ""
        )
        provider = str(
            getattr(resolution, "execution_key", "")
            or getattr(resolution, "provider", "")
            or provider_selection.provider
            or "agent"
        )
        bound = owner.bound_messages_to_window(
            copy.deepcopy(provider_messages),
            model=model,
            provider=provider,
            response_reservation=(
                getattr(resolution, "max_tokens", None)
                or owner.DEFAULT_RESPONSE_RESERVATION
            ),
        )
        agent_messages = list(bound.messages)
        session_system_prompt = ""
        if (
            agent_messages
            and agent_messages[0].get("role") == owner.ConsoleMessageRole.SYSTEM.value
        ):
            session_system_prompt = str(agent_messages[0].get("content", ""))
            agent_messages = agent_messages[1:]
    except Exception:  # noqa: BLE001 - uncertain budget fails closed
        return owner._empty_profile_context_snapshot()
    try:
        preview_args = dict(  # noqa: C408 - preserve the existing keyword call shape.
            session_id=session.id,
            workspace_id=session.workspace_id,
            ephemeral=session.ephemeral,
            resolution=resolution,
            fallback_model=(
                provider_selection.explicit_model
                or provider_selection.configured_model
                or ""
            ),
            session_system_prompt=session_system_prompt,
            agent_messages=agent_messages,
            mcp_provider=mcp_provider,
            builtin_gate=builtin_gate,
            local_provider=local_provider,
            virtual_cli_provider=virtual_cli_provider,
            raw_shell_provider=raw_shell_provider,
            library_provider=library_provider,
            library_authority=library_authority,
            profile_provider=profile_provider,
            canvas_profile_snapshot=canvas_profile_snapshot,
            scratch_root=(
                scratch_snapshot.root if scratch_snapshot is not None else None
            ),
            scratch_lease=scratch_lease,
            turn_skill_bindings=turn_skill_bindings,
            turn_bundle_block=turn_bundle_block,
            worktree_merge_enabled=self.worktree_confirmation_enabled,
            fork_chat_enabled=(
                self.set_pending_chat_create is not None
                and self.complete_agent_chat_create is not None
            ),
            new_chat_enabled=(
                self.set_pending_chat_create is not None
                and self.complete_agent_chat_create is not None
            ),
            request_skill_install_enabled=True,
            request_skill_script_enabled=(self.set_pending_skill_script is not None),
            profile_context_service=builder,
            persona_policy_rules=configuration.persona_policy_rules,
        )
        if profile_selection_sink is not None:
            preview_args["selection_sink"] = profile_selection_sink
        return await asyncio.to_thread(build_preview, **preview_args)
    except Exception:  # noqa: BLE001 - personalization never blocks preview
        return owner._empty_profile_context_snapshot()


def compose_profile_tool_provider(
    service: PersonalContextService,
    *,
    workspace_id: str | None,
    ephemeral: bool,
    run_id: str,
    session_id: str,
    current_user_message: ConsoleChatMessage | None,
    kill_switch: Callable[[], bool],
    reserve_direct_update_schema: bool = False,
) -> ProfileToolProvider | None:
    """Capture one stable, fail-closed profile authority for a Console run."""

    from .console_chat_models import CONSOLE_GLOBAL_WORKSPACE_ID, ConsoleMessageRole

    if ephemeral:
        return None
    from tldw_chatbook.Agents.profile_tool_provider import (
        ProfileToolProvider,
        ProfileToolRunScope,
    )

    active_workspace_id = (
        None if workspace_id == CONSOLE_GLOBAL_WORKSPACE_ID else workspace_id
    )
    try:
        with service.read_operation():
            first_view = service.authorized_context_view(
                active_workspace_id=active_workspace_id
            )
            scopes = service.list_scopes()
            scope_id = first_view.workspace_scope_id or next(
                scope.scope_id for scope in scopes if scope.kind.value == "global"
            )
            authority = service.get_scope_authority(scope_id)
            # Both views stay: the second fences the separately read scope and
            # authority. Reusing a connection must not turn this into a snapshot.
            stable_view = service.authorized_context_view(
                active_workspace_id=active_workspace_id
            )
            if (
                stable_view.generation != first_view.generation
                or stable_view.authority_revision != first_view.authority_revision
                or stable_view.workspace_scope_id != first_view.workspace_scope_id
            ):
                return None
            manifest = service.get_manifest()
    except Exception:  # noqa: BLE001 - optional profile tools fail soft
        return None

    trusted_message = (
        current_user_message
        if current_user_message is not None
        and current_user_message.role is ConsoleMessageRole.USER
        else None
    )
    return ProfileToolProvider(
        service,
        run_scope=ProfileToolRunScope(
            run_id=run_id,
            session_id=session_id,
            profile_id=manifest.profile_id,
            scope_id=scope_id,
            authority=authority,
            generation=stable_view.generation,
            authority_revision=stable_view.authority_revision,
            current_user_message_id=(
                trusted_message.persisted_message_id or trusted_message.id
                if trusted_message is not None
                else None
            ),
            current_user_text=(
                trusted_message.content if trusted_message is not None else None
            ),
        ),
        kill_switch=kill_switch,
        reserve_direct_update_schema=reserve_direct_update_schema,
    )
