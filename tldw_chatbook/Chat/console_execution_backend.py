"""The contained service's actual profile, graph and command ownership."""

from __future__ import annotations

import asyncio
from inspect import getattr_static
from types import FunctionType

from .console_execution_protocol import (
    ConsoleProtocolError,
    ConsoleWireMessage,
    WireIdentity,
)


class ConsoleExecutionBackend:
    """Own one original Console graph without constructing an App substitute."""

    def __init__(self, profile, *, profile_epoch: str, service_epoch: str) -> None:
        from .console_execution_profile import ConsoleBackendProfile

        if type(profile) is not ConsoleBackendProfile:
            raise TypeError("The original backend profile is required")
        WireIdentity(profile_epoch, service_epoch, 1, "backend-initialization")
        self.profile = profile
        self.profile_epoch = profile_epoch
        self.service_epoch = service_epoch
        self.runtime = None
        self.domain = None
        self.store = self.controller = self.controls = None
        self.presentation = None
        self.adapter = None
        self._worker_service = None
        self._closing = None
        self._shutdown_receipt = None
        self._initialization_started = False
        self._runtime_initialized = False
        self.closed = False
        self.ready = False
        self._notices = []
        self._sequence = 0
        self._loop = asyncio.get_running_loop()

    def start(self) -> None:
        """Construct once after the caller has retained this cleanup owner."""
        self._require_owner_loop()
        from tldw_chatbook import app_service_wiring, config

        # Original skill-source validation uses this defining metadata module;
        # importing it creates neither a widget nor the lazy trust service.
        from tldw_chatbook.Widgets import compact_model_bar  # noqa: F401
        from tldw_chatbook.DB.VisualIdentity_DB import VisualIdentityRepository
        from . import console_runtime as runtime_source
        from .console_backend_presentation import ConsoleBackendPresentation
        from .console_domain_services import ConsoleDomainServices, ConsoleDomainSources
        from .console_execution_worker import _BackendCommandAdapter
        from .prompt_history import PromptHistory
        from .console_scratch_space import ConsoleScratchSpaceManager
        from .console_settings_durability import ConsoleSettingsDurabilityOwner
        from .console_chat_controller import (
            build_console_provider_selection_from_settings,
        )
        from .console_chat_models import ConsoleWorkspaceContext

        if self._initialization_started or self._closing is not None:
            raise RuntimeError("Console backend initialization already entered")
        self._initialization_started = True
        profile = self.profile
        self.presentation = ConsoleBackendPresentation(self)
        sources = ConsoleDomainSources(
            configuration=profile.configuration,
            chat_database=profile.storage.chachanotes,
            prompt_database=profile.storage.prompts,
            workspace_registry=profile.workspace_registry,
            consent_service=profile.consent_service,
            skills_scope=profile.skills_scope,
            local_skills=profile.local_skills,
            visual_repository=VisualIdentityRepository(profile.storage.chachanotes),
            scratch=ConsoleScratchSpaceManager(),
            control_plane=profile.control_plane,
            prompt_scope=profile.prompt_scope,
            storage=profile.storage,
            citation_repository=profile.citation_repository,
            policy_enforcer=profile.policy_enforcer,
            data_root_selector=config.get_user_data_dir,
            conversation_marks=profile.conversation_marks,
            settings_durability_owner=ConsoleSettingsDurabilityOwner(),
            skill_wiring_source=app_service_wiring,
            conversation_scope_service=profile.conversation_scope_service,
            prompt_history=PromptHistory(),
        )
        self.domain = ConsoleDomainServices.__new__(ConsoleDomainServices)
        ConsoleDomainServices.__init__(
            self.domain,
            sources,
            runtime_source=runtime_source,
            presentation=self.presentation,
        )
        self.runtime = runtime_source.ConsoleRuntime.__new__(
            runtime_source.ConsoleRuntime
        )
        runtime_source.ConsoleRuntime.__init__(
            self.runtime, None, domain_services=self.domain
        )
        self._runtime_initialized = True
        active = profile.workspace_registry.get_active_workspace()
        self.store = self.runtime.ensure_chat_store(
            workspace_context=ConsoleWorkspaceContext(
                active_workspace_id=active.workspace_id if active is not None else None,
            )
        )
        # Every authored turn uses this coordinator. Compose its original hooks
        # on the backend loop before any parent visit can own the presentation.
        self.runtime.ensure_archive_recovery()
        defaults = self.domain.default_session_settings()
        selection = build_console_provider_selection_from_settings(
            defaults,
            app_config=profile.configuration,
            workspace_context=self.store.workspace_context,
            legacy_model=profile.configuration.get("chat_defaults", {}).get("model"),
            global_user_name=self.domain.global_user_display_name,
        )
        agent_runtime = profile.configuration.get("console", {}).get(
            "agent_runtime", True
        )
        self.controller = self.runtime.ensure_chat_controller(
            store=self.store,
            provider_gateway=self.runtime.ensure_provider_gateway(),
            provider=selection.provider,
            model=selection.explicit_model,
            configured_model=selection.configured_model,
            **{
                name: getattr(selection, name)
                for name in (
                    "base_url",
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
            },
            agent_bridge=self.runtime.ensure_agent_bridge(
                store_factory=self.runtime.ensure_chat_store,
                provider_gateway_factory=self.runtime.ensure_provider_gateway,
                skills_service=profile.skills_scope,
            ),
            skills_service=profile.skills_scope,
            agent_runtime_enabled=(
                bool(agent_runtime) if isinstance(agent_runtime, (bool, int)) else True
            ),
        )
        self.controls = self.runtime._execution_service.open_local_controls(
            profile_epoch=self.profile_epoch, service_epoch=self.service_epoch
        )
        self.adapter = _BackendCommandAdapter(self.controls)
        self.require_sources()
        self.ready = True

    def _require_owner_loop(self) -> None:
        if asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("Console backend owner loop changed")

    def require_sources(self) -> None:
        """Validate sources on the loop or an original qualified read worker.

        This grants no command/native authority. Original read handoff and
        source guards still qualify their actual worker; mutable command and
        lifecycle entry separately require the owning event loop.
        """
        from tldw_chatbook import config
        from .console_domain_services import _definition_current

        if (
            type(self) is not ConsoleExecutionBackend
            or not all(
                getattr_static(self, name, None) is record[0]
                and _definition_current(record)
                for name, record in _DEFINITIONS
            )
            or self.closed
            or config.get_cli_config_path() != self.profile.config_path
            or getattr_static(self.profile, "selection_is_current", None)
            is not _PROFILE_SELECTION[0]
            or not _definition_current(_PROFILE_SELECTION)
            or not self.profile.selection_is_current()
            or config.current_config_identity() != self.profile.configuration_identity
            or self.domain.sources.configuration is not self.profile.configuration
            or self.domain.sources.chat_database is not self.profile.storage.chachanotes
            or self.domain.sources.prompt_database is not self.profile.storage.prompts
            or self.domain.sources.control_plane is not self.profile.control_plane
            or self.domain.sources.prompt_scope is not self.profile.prompt_scope
        ):
            raise RuntimeError("Console backend source association changed")

    def is_ready(self) -> bool:
        """Expose only the actual service's completed initialization state."""
        return self.ready and not self.closed

    def retain_notice(self, message: str, *, severity: str) -> None:
        """Keep advisory presentation until the exact parent channel consumes it."""
        if type(message) is not str or severity not in {
            "information",
            "warning",
            "error",
        }:
            raise ConsoleProtocolError("invalid_backend_notice")
        self._notices.append((message, severity))

    def dispatch(self, message: ConsoleWireMessage):
        """Enter the existing local controls once for an exact channel command."""
        self._require_owner_loop()
        self.require_sources()
        if (
            type(message) is not ConsoleWireMessage
            or message.kind != "command"
            or message.identity.profile_epoch != self.profile_epoch
            or message.identity.service_epoch != self.service_epoch
            or message.identity.sequence != self._sequence + 1
        ):
            raise ConsoleProtocolError("backend_command_order_changed")
        self._sequence = message.identity.sequence
        command = self.adapter.command(message)
        checkpoint = self.presentation.admit_view_checkpoint(message)
        try:
            receipt = self.controls.dispatch(command)
        except BaseException:
            self.presentation.release_view_checkpoint(message.identity, checkpoint)
            raise
        if message.operation == "Shutdown":
            self._shutdown_receipt = receipt
        self.adapter.retain(message, receipt, command)
        if checkpoint is not None and receipt.work is not None:
            # Drop only this request's inert token registration, including
            # cancellation before coroutine entry. Resource retirement remains
            # the original control/effect owner's independent obligation.
            presentation = self.presentation
            receipt.work.add_done_callback(
                lambda _task: presentation.release_view_checkpoint(message.identity, checkpoint)
            )
        return receipt

    async def _close(self) -> None:
        self._require_owner_loop()
        from .console_control_dispatch import ControlIdentity, Shutdown

        error = None
        try:
            if self.presentation is not None:
                await self.presentation.drain_view_lifecycle()
        except BaseException as failure:
            error = failure
        try:
            self.ready = False
            if self.controls is not None:
                receipt = self._shutdown_receipt
                if receipt is None:
                    self._sequence += 1
                    receipt = self.controls.dispatch(
                        Shutdown(
                            ControlIdentity(
                                self.profile_epoch,
                                self.service_epoch,
                                self._sequence,
                                f"backend-close-{self._sequence}",
                                source_revision=self.controls.source_revision,
                            )
                        )
                    )
                    self._shutdown_receipt = receipt
                result = await asyncio.shield(receipt.applied)
                if receipt.work is not None:
                    await asyncio.shield(receipt.work)
                if result is not True or not self.runtime._disposed:
                    raise RuntimeError("Original Console backend shutdown is unproven")
            elif self.runtime is not None:
                # A partial object remains owned even if its original disposer
                # cannot finish. Do not manufacture completed initialization.
                await self.runtime.dispose()
                if not self._runtime_initialized:
                    raise RuntimeError("Partial Console runtime retirement is unproven")
        except BaseException as failure:
            if error is None:
                error = failure
        try:
            await self.profile.aclose()
        except BaseException as failure:
            if error is None:
                error = failure
        if error is not None:
            raise error
        self.closed = True

    async def aclose(self) -> None:
        """Retain original runtime and profile shutdown despite observer cancellation."""
        self._require_owner_loop()
        if self._closing is None:
            self._closing = self._loop.create_task(self._close())
        try:
            await asyncio.shield(self._closing)
        except asyncio.CancelledError:
            while not self._closing.done():
                try:
                    await asyncio.shield(self._closing)
                except asyncio.CancelledError:
                    continue
                except BaseException:
                    break
            if self._closing.done() and not self._closing.cancelled():
                self._closing.exception()
            raise


from .console_domain_services import _definition
from .console_execution_profile import ConsoleBackendProfile

_PROFILE_SELECTION = _definition(ConsoleBackendProfile.selection_is_current)

_DEFINITIONS = tuple(
    (name, _definition(value))
    for name, value in vars(ConsoleExecutionBackend).items()
    if type(value) is FunctionType
)
