"""Explicit in-process sources for the selected Console construction.

These are real collaborators, not an App substitute. No selection is enabled by
importing this module; legacy construction continues to use its original App.
"""

from __future__ import annotations

import inspect
import asyncio
import sys
from dataclasses import dataclass, field
from types import MethodType, ModuleType
from typing import Any, Callable

from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired


@dataclass(frozen=True, slots=True)
class ConsoleDomainSources:
    """Capture actual sources without constructing absent optional services."""

    configuration: Any = field(repr=False)
    chat_database: Any = field(default=None, repr=False)
    prompt_database: Any = field(default=None, repr=False)
    workspace_registry: Any = field(default=None, repr=False)
    consent_service: Any = field(default=None, repr=False)
    skills_scope: Any = field(default=None, repr=False)
    local_skills: Any = field(default=None, repr=False)
    skill_trust: Any = field(default=None, repr=False)
    plugin_service: Any = field(default=None, repr=False)
    visual_repository: Any = field(default=None, repr=False)
    scratch: Any = field(default=None, repr=False)
    control_plane: Any = field(default=None, repr=False)
    tool_pack_service: Any = field(default=None, repr=False)
    persona_service: Any = field(default=None, repr=False)
    prompt_scope: Any = field(default=None, repr=False)
    storage: Any = field(default=None, repr=False)
    citation_repository: Any = field(default=None, repr=False)
    policy_enforcer: Any = field(default=None, repr=False)
    data_root_selector: Callable[[], Any] | None = field(default=None, repr=False)
    plugin_mcp_mapping: Any = field(default=None, repr=False)
    plugin_model_mapping: Any = field(default=None, repr=False)
    raw_cli_runtime: Any = field(default=None, repr=False)
    conversation_marks: Any = field(default=None, repr=False)
    canvas_profile_snapshot: Any = field(default=None, repr=False)
    persona_buddy: Any = field(default=None, repr=False)
    settings_durability_owner: Any = field(default=None, repr=False)
    skill_wiring_source: ModuleType | None = field(default=None, repr=False)
    prompt_history: Any = field(default=None, repr=False)
    project_index_owner: Any = field(default=None, repr=False)
    conversation_scope_service: Any = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class DomainMcpAttachment:
    domain_services: Any = field(repr=False)
    sources: ConsoleDomainSources = field(repr=False)
    runtime: Any = field(repr=False)
    original_runtime: Any = field(repr=False)
    association: Any = field(repr=False)


class ConsoleDomainServices:
    """Own the finite extracted factories and live source readers."""

    def __init__(
        self,
        sources: ConsoleDomainSources,
        *,
        runtime_source: ModuleType,
        presentation: Any,
    ) -> None:
        if type(sources) is not ConsoleDomainSources:
            raise TypeError("sources must be ConsoleDomainSources")
        from .console_runtime import _RUNTIME_MAINTENANCE_SOURCE
        from .console_presentation_port import ConsolePresentationPort
        from .console_backend_presentation import ConsoleBackendPresentation

        if (
            runtime_source is not _RUNTIME_MAINTENANCE_SOURCE
            or type(presentation) not in (ConsolePresentationPort, ConsoleBackendPresentation)
        ):
            raise TypeError(
                "Selected Console construction requires the original source and concrete presentation port"
            )
        self.sources = sources
        self._construction_sources = sources
        self.runtime_source = runtime_source
        self._construction_runtime_source = runtime_source
        self.presentation = presentation
        self.runtime = None
        self._conversation_owner = None
        if sources.conversation_scope_service is not None:
            from .chat_conversation_scope_service import ChatConversationScopeService
            from .chat_conversation_service import ChatConversationService

            scope = sources.conversation_scope_service
            if (
                type(scope) is not ChatConversationScopeService
                or type(scope.local_service) is not ChatConversationService
                or scope.local_service.db is not sources.chat_database
            ):
                raise TypeError(
                    "Saved Console opening requires its original conversation owner"
                )
            self._conversation_owner = (
                scope,
                scope.local_service,
                scope.policy_enforcer,
            )
        from tldw_chatbook.Workspaces.project_index.runtime import ProjectIndexOwner

        self._project_index_owner = sources.project_index_owner
        if self._project_index_owner is None and sources.workspace_registry is not None:
            self._project_index_owner = ProjectIndexOwner(sources.workspace_registry)
        if self._project_index_owner is not None and (
            type(self._project_index_owner) is not ProjectIndexOwner
            or self._project_index_owner.registry is not sources.workspace_registry
        ):
            raise TypeError("Project index requires the original registry owner")
        self._plugin_inputs = _capture_plugin_constructor_inputs(sources)
        self._plugin = sources.plugin_service
        if self._plugin is not None:
            service = self._plugin
            self._plugin_construction = (
                service,
                service.profile_root,
                service.workspace_lookup,
                service._mcp_mapping_owner,
                service._model_mapping_owner,
            )
        self._trust = sources.skill_trust
        self._trust_lock = asyncio.Lock()
        from . import console_configuration_preparation, console_configuration_capture
        from . import (
            console_received_dispatch,
            console_turn_context,
            console_assistant_defaults,
            console_provider_composition,
            console_control_sessions,
            console_domain_display,
            console_display_phases,
            console_execution_retirement,
        )
        from tldw_chatbook.MCP import console_snapshot

        self._source_modules = (
            console_configuration_preparation,
            console_configuration_capture,
            console_received_dispatch,
            console_turn_context,
            console_assistant_defaults,
            console_provider_composition,
            console_control_sessions,
            console_domain_display,
            console_display_phases,
            console_execution_retirement,
            console_snapshot,
        )
        self._source_definitions = tuple(
            (module, name, record)
            for module in self._source_modules
            for name, record in module._DOMAIN_SOURCE_DEFINITIONS
        )
        local = sources.local_skills
        if local is not None:
            from tldw_chatbook.Skills_Interop.local_skills_service import (
                LocalSkillsService,
            )

            if type(local) is not LocalSkillsService:
                raise TypeError("Custom local skills require legacy construction")
            bindings = [("_builtin_disabled_loader", self.disabled_builtin_skills)]
            if sources.data_root_selector is not None:
                bindings.append(("_plugin_service_factory", self.build_plugin_service))
            if sources.skill_wiring_source is not None:
                bindings.append(("_trust_service_factory", self.read_skill_trust))
            if any(vars(local).get(name) is not None for name, _ in bindings):
                raise TypeError(
                    "Already bound skill factories require legacy construction"
                )
            for name, callback in bindings:
                setattr(local, name, callback)

    def build_skill_trust(self) -> Any:
        self.require_current()
        from .console_domain_skill_trust import build_domain_skill_trust

        return build_domain_skill_trust(self.sources.skill_wiring_source)

    def read_skill_trust(self) -> Any:
        """Keep the original lazy synchronous property semantics for local use."""
        self.require_current()
        if self._trust is None:
            candidate = self.build_skill_trust()
            if self._trust is None:
                self._trust = candidate
                from .console_domain_skill_trust import (
                    record_domain_skill_trust_candidate,
                )

                record_domain_skill_trust_candidate(self, candidate)
        return self._trust

    async def ensure_skill_trust(
        self, *, _source_current=None, _owner_current=None, _read_observers=()
    ) -> Any:
        """Retain the original lock, winner publication and native-read order."""
        if _owner_current is not None and _owner_current() is not True:
            raise RuntimeError("console_skill_trust_owner_changed")
        if self._trust is not None:
            return self._trust
        published_candidate = None
        async with self._trust_lock:
            if _owner_current is not None and _owner_current(self._trust) is not True:
                raise RuntimeError("console_skill_trust_owner_changed")
            if self._trust is None:
                from .console_preparation_reads import run_preparation_read

                def require_current():
                    if _source_current is not None and _source_current() is not True:
                        raise RuntimeError("console_skill_trust_source_changed")

                service = await run_preparation_read(
                    self.build_skill_trust,
                    creator=self,
                    session_id=None,
                    reads=set(),
                    observers=_read_observers,
                    require_current=require_current,
                )
                if (
                    _owner_current is not None
                    and _owner_current(self._trust) is not True
                ):
                    raise RuntimeError("console_skill_trust_owner_changed")
                if self._trust is None:
                    self._trust = service
                    published_candidate = service
            winner = self._trust
        if published_candidate is not None:
            from .console_domain_skill_trust import record_domain_skill_trust_candidate

            record_domain_skill_trust_candidate(self, published_candidate)
        return winner

    def require_current(self, sources: ConsoleDomainSources | None = None) -> None:
        """Reject replaced installed bodies and the actual captured source set."""
        if (
            type(self) is not ConsoleDomainServices
            or self.sources is not self._construction_sources
            or (sources is not None and self.sources is not sources)
            or self.runtime_source is not self._construction_runtime_source
            or sys.modules.get(self.runtime_source.__name__) is not self.runtime_source
            or not _domain_definitions_current()
            or not _plugin_constructor_inputs_current(self.sources, self._plugin_inputs)
            or not all(
                globals().get(name) is record[0] and _definition_current(record)
                for name, record in _DOMAIN_HELPER_DEFINITIONS
            )
            or not self.runtime_source._domain_callback_definitions_current()
            or any(name in vars(self) for _, name, _ in _DOMAIN_DEFINITIONS)
            or not self.presentation.definitions_current()
            or any(
                sys.modules.get(module.__name__) is not module
                for module in self._source_modules
            )
            or any(
                vars(module).get(name) is not record[0]
                or not _definition_current(record)
                for module, name, record in self._source_definitions
            )
        ):
            raise RecoveryRequired("console_snapshot_owner_changed")

    def provider_config(self) -> Any:
        self.require_current()
        return self.runtime_source._provider_config_for_domain(
            self.sources.configuration
        )

    def conversation_scope_service(self):
        """Retain the original local conversation reader and its exact database."""
        from .console_conversation_hydration import ConversationServiceUnavailable

        self.require_current()
        if self._conversation_owner is None:
            raise ConversationServiceUnavailable(
                "Saved conversation resume is unavailable in this build."
            )
        scope, local, policy = self._conversation_owner
        if (
            self.sources.conversation_scope_service is not scope
            or scope.local_service is not local
            or scope.policy_enforcer is not policy
            or local.db is not self.sources.chat_database
        ):
            raise RecoveryRequired("console_snapshot_owner_changed")
        from .console_control_sessions import require_conversation_definitions

        require_conversation_definitions(scope, local)
        return scope

    def project_index_owner(self):
        self.require_current()
        owner = self._project_index_owner
        if owner is None or owner.registry is not self.sources.workspace_registry:
            raise RecoveryRequired("console_snapshot_owner_changed")
        owner.require_current()
        return owner

    def apply_configuration_publication(self, publication):
        """Advance only the named configuration result from this runtime's fence."""
        from dataclasses import fields
        from .console_control_publication import ConsoleConfigurationPublication

        if type(publication) is not ConsoleConfigurationPublication:
            raise TypeError("An original configuration publication is required")
        controls = self.runtime._execution_service._local_controls
        if controls is None or controls._domain is not self:
            raise RecoveryRequired("console_snapshot_owner_changed")
        controls._require_publication(publication)
        self.require_current(publication.previous_sources)
        successor = publication.sources
        if type(successor) is not ConsoleDomainSources or any(
            field.name != "configuration"
            and getattr(successor, field.name) is not getattr(self.sources, field.name)
            for field in fields(ConsoleDomainSources)
        ):
            raise RecoveryRequired("console_snapshot_owner_changed")
        display = controls._service._display_owner
        if display is not None:
            from .console_domain_display import ConsoleDomainDisplayOwner, _require_definitions

            if type(display) is not ConsoleDomainDisplayOwner or display.domain is not self:
                raise RecoveryRequired("console_snapshot_owner_changed")
            _require_definitions(display)
            display.apply_configuration_publication(publication)
        self.sources = successor
        self._construction_sources = successor
        return successor

    def native_tools_enabled(self) -> bool:
        self.require_current()
        return self.runtime_source._native_tools_enabled_for_domain(
            self.sources.configuration
        )

    def global_user_display_name(self) -> str:
        self.require_current()
        return self.runtime_source._global_user_display_name_for_domain(
            self.sources.configuration
        )

    def default_session_settings(self) -> Any:
        self.require_current()
        return self.runtime_source._default_session_settings_for_domain(
            self.sources.configuration
        )

    def library_policy_defaults(self) -> Any:
        self.require_current()
        return self.runtime_source._library_policy_defaults_for_domain(
            self.sources.configuration
        )

    def thinking_history_policy_default(self) -> str:
        self.require_current()
        return self.runtime_source._thinking_history_policy_default_for_domain(
            self.sources.configuration
        )

    def apply_chat_dictionaries(self, conversation_id, text, frozen_inputs=None):
        self.require_current()
        return self.runtime_source._apply_chat_dictionaries_for_domain(
            self.sources.chat_database, conversation_id, text, frozen_inputs
        )

    def apply_world_info(self, conversation_id, text, history, frozen_inputs=None):
        self.require_current()
        return self.runtime_source._apply_world_info_for_domain(
            self.sources.chat_database, conversation_id, text, history, frozen_inputs
        )

    def storage(self) -> Any:
        self.require_current()
        return self.sources.storage

    def build_plugin_service(self) -> Any:
        """Build the original IO-free facade only on explicit first use."""
        self.require_current()
        service = self.published_plugin_service()
        if service is None:
            from tldw_chatbook.Plugins.service import PluginService

            source = self.sources
            if self._plugin_inputs is None:
                raise RecoveryRequired("console_capture_source_not_ready")
            lookup = self._plugin_inputs[-1]
            root = source.data_root_selector()
            self.require_current(source)
            service = PluginService(
                root,
                workspace_lookup=lookup,
                mcp_mapping_owner=source.plugin_mcp_mapping,
                model_mapping_owner=source.plugin_model_mapping,
            )
            self.require_current(source)
            self._plugin = service
            self._plugin_construction = (
                service,
                service.profile_root,
                service.workspace_lookup,
                service._mcp_mapping_owner,
                service._model_mapping_owner,
            )
        return service

    def published_plugin_service(self) -> Any:
        """Read only the resident slot; this never creates the lazy facade."""
        self.require_current()
        construction = getattr(self, "_plugin_construction", None)
        if construction is not None:
            service, root, lookup, mcp, models = construction
            if (
                self._plugin is not service
                or service.profile_root is not root
                or service.workspace_lookup is not lookup
                or service._mcp_mapping_owner is not mcp
                or service._model_mapping_owner is not models
                or type(lookup) is not MethodType
                or lookup.__self__ is not self.sources.workspace_registry
                or self._plugin_inputs is None
                or lookup.__func__ is not self._plugin_inputs[-1].__func__
                or mcp is not self.sources.plugin_mcp_mapping
                or models is not self.sources.plugin_model_mapping
            ):
                raise RecoveryRequired("console_snapshot_owner_changed")
        return self._plugin

    def disabled_builtin_skills(self) -> frozenset[str]:
        self.require_current()
        from tldw_chatbook.Skills_Interop.builtin_skills import (
            disabled_builtins_from_config,
        )

        return disabled_builtins_from_config(self.sources.configuration)

    def configuration_sources(self, store, creator, session, *, unused_trust=None):
        """Read the same finite source set from the explicit domain graph."""
        from .console_configuration_preparation import (
            _ConfigurationSources,
            _ready_attribute,
        )

        self.require_current()
        if creator.domain_services is not self:
            raise RecoveryRequired("console_snapshot_owner_changed")
        source = self.sources
        registry, consent, scope = (
            source.workspace_registry,
            source.consent_service,
            source.skills_scope,
        )
        local = (
            _ready_attribute(scope, "local_service") if scope is not None else None
        ) or source.local_skills
        if unused_trust is not None:
            from .console_domain_skill_trust import DomainSkillTrustSource
            from .console_configuration_preparation import (
                _supported_published_skill_trust,
            )

            if (
                type(unused_trust) is not DomainSkillTrustSource
                or unused_trust.domain_services is not self
                or unused_trust.local is not local
                or unused_trust.scope is not scope
                or not unused_trust.source_current()
                or not _supported_published_skill_trust(unused_trust)
            ):
                raise RecoveryRequired("console_snapshot_owner_changed")
        trust = (
            _ready_attribute(
                local, "trust_service", "_trust_service", "_trust_service_factory"
            )
            if unused_trust is None
            else None
        )
        plugin = _ready_domain_published_plugin(local, self)
        persistence = getattr(store, "persistence", None)
        visual = getattr(creator, "_visual_identity_repository", None)
        persona = (
            source.persona_service
            if session is not None and session.assistant_kind == "persona"
            else None
        )
        return _ConfigurationSources(
            source.configuration,
            registry,
            getattr(registry, "db", None),
            consent,
            scope,
            local,
            trust,
            plugin,
            persistence,
            getattr(persistence, "db", None),
            # This legacy configuration participant is the main chat DB, as in
            # the original App projection; PromptScope owns the separate DB.
            source.chat_database,
            visual,
            getattr(visual, "db", None),
            getattr(creator, "_scratch_spaces", None),
            source.control_plane,
            getattr(creator, "_provider_config", None),
            persona,
            getattr(persona, "db", getattr(persona, "_db", None)),
            getattr(consent, "_registry", None),
            tuple(
                inspect.getattr_static(owner, name, None) if owner is not None else None
                for owner, name in (
                    (consent, "_capability_reader"),
                    (consent, "admit_turn"),
                    (local, "_builtin_disabled_loader"),
                    (local, "_plugin_service_factory"),
                    (local, "_visible_records"),
                    (local, "_summary_for_record"),
                    (trust, "current_fingerprint_digest"),
                    (plugin, "capture_maximum"),
                    (persona, "get_persona_profile"),
                )
            )
            + (self.published_plugin_service(),),
        )


def _recorded_selector_current(row):
    owner, name, original, records = row
    if owner.get(name) is not original:
        return False
    for (
        function,
        code,
        namespace,
        defaults,
        keywords,
        entries,
        closure,
        cells,
        wrapped,
    ) in records:
        if not (
            function.__code__ is code
            and function.__globals__ is namespace
            and function.__defaults__ is defaults
            and function.__kwdefaults__ is keywords
            and len(keywords or {}) == len(entries)
            and all((keywords or {}).get(key) is value for key, value in entries)
            and function.__closure__ is closure
            and all(cell.cell_contents is value for cell, value in cells)
            and vars(function).get("__wrapped__") is wrapped
        ):
            return False
    return bool(records)


def _capture_plugin_constructor_inputs(sources):
    if sources.data_root_selector is None:
        if sources.plugin_service is not None:
            raise TypeError(
                "Published plugin construction without original sources requires legacy construction"
            )
        return None
    from tldw_chatbook import config
    from tldw_chatbook.Workspaces import registry_service as registry_source

    selector = next(
        row
        for row in config._COLLECTIONS_SETUP_SOURCE[1]
        if row[1] == "get_user_data_dir"
    )
    registry_record = registry_source._CONSOLE_PLUGIN_LOOKUP_SOURCE
    registry = sources.workspace_registry
    if (
        type(registry) is not registry_record[1]
        or sources.data_root_selector is not selector[2]
    ):
        raise TypeError(
            "Custom plugin constructor callbacks require legacy construction"
        )
    lookup = registry.get_workspace
    captured = (config, selector, registry_source, registry_record, registry, lookup)
    if not _plugin_constructor_inputs_current(sources, captured):
        raise TypeError(
            "Changed plugin constructor callbacks require legacy construction"
        )
    return captured


def _plugin_constructor_inputs_current(sources, captured):
    if captured is None:
        return sources.data_root_selector is None
    from tldw_chatbook.Utils.definition_records import definitions_current

    config, selector, registry_source, registry_record, registry, lookup = captured
    namespace, original_class, defining_globals, definitions = registry_record
    current = getattr(registry, "get_workspace", None)
    return (
        sys.modules.get(config.__name__) is config
        and sources.data_root_selector is selector[2]
        and selector[0] is vars(config)
        and _recorded_selector_current(selector)
        and sys.modules.get(registry_source.__name__) is registry_source
        and vars(registry_source).get("_CONSOLE_PLUGIN_LOOKUP_SOURCE")
        is registry_record
        and namespace is vars(registry_source)
        and vars(registry_source).get("LocalWorkspaceRegistryService") is original_class
        and type(registry) is original_class
        and sources.workspace_registry is registry
        and definitions_current(definitions)
        and type(current) is MethodType
        and type(lookup) is MethodType
        and lookup.__func__.__globals__ is defining_globals
        and current.__self__ is registry
        and lookup.__self__ is registry
        and current.__func__ is lookup.__func__ is definitions[0][2]
        and "get_workspace" not in vars(registry)
    )


def _ready_domain_published_plugin(local, domain: ConsoleDomainServices):
    """Qualify the selected local property, factory and publication receiver."""
    domain.require_current()
    if local is None:
        return None
    from tldw_chatbook.Skills_Interop.local_skills_service import (
        _STOCK_PLUGIN_SERVICE_PROPERTY,
        LocalSkillsService,
    )

    original_property, getter, code = _STOCK_PLUGIN_SERVICE_PROPERTY
    descriptor = inspect.getattr_static(local, "plugin_service", None)
    if (
        type(local) is not LocalSkillsService
        or descriptor is not original_property
        or original_property.fget is not getter
        or getter.__code__ is not code
    ):
        raise RecoveryRequired("console_capture_source_not_ready")
    published = domain.published_plugin_service()
    existing = vars(local).get("_plugin_service")
    if existing is not None:
        return existing
    factory = vars(local).get("_plugin_service_factory")
    if factory is None:
        return None
    if (
        type(factory) is not MethodType
        or factory.__self__ is not domain
        or factory.__func__ is not ConsoleDomainServices.build_plugin_service
        or local is not domain.sources.local_skills
    ):
        raise RecoveryRequired("console_capture_source_not_ready")
    return published


def _definition(function):
    wrapped = vars(function).get("__wrapped__")
    return (
        function,
        function.__code__,
        function.__globals__,
        function.__defaults__,
        function.__kwdefaults__,
        tuple((function.__kwdefaults__ or {}).items()),
        function.__closure__,
        tuple((cell, cell.cell_contents) for cell in function.__closure__ or ()),
        None if wrapped is None else _definition(wrapped),
    )


def _definition_current(record):
    (
        function,
        code,
        namespace,
        defaults,
        kwdefaults,
        entries,
        closure,
        cells,
        wrapped,
    ) = record
    return (
        function.__code__ is code
        and function.__globals__ is namespace
        and function.__defaults__ is defaults
        and function.__kwdefaults__ is kwdefaults
        and len(function.__kwdefaults__ or {}) == len(entries)
        and (
            not entries
            or all(
                (function.__kwdefaults__ or {}).get(key) is value
                for key, value in entries
            )
        )
        and function.__closure__ is closure
        and (not cells or all(cell.cell_contents is value for cell, value in cells))
        and (
            vars(function).get("__wrapped__") is None
            if wrapped is None
            else vars(function).get("__wrapped__") is wrapped[0]
            and _definition_current(wrapped)
        )
    )


def _domain_definitions_current():
    return all(
        inspect.getattr_static(owner, name, None) is record[0]
        and _definition_current(record)
        for owner, name, record in _DOMAIN_DEFINITIONS
    )


_DOMAIN_DEFINITIONS = tuple(
    (ConsoleDomainServices, name, _definition(getattr(ConsoleDomainServices, name)))
    for name in (
        "__init__",
        "require_current",
        "provider_config",
        "conversation_scope_service",
        "project_index_owner",
        "apply_configuration_publication",
        "native_tools_enabled",
        "global_user_display_name",
        "default_session_settings",
        "library_policy_defaults",
        "thinking_history_policy_default",
        "apply_chat_dictionaries",
        "apply_world_info",
        "storage",
        "build_plugin_service",
        "published_plugin_service",
        "disabled_builtin_skills",
        "configuration_sources",
        "build_skill_trust",
        "read_skill_trust",
        "ensure_skill_trust",
    )
)

_DOMAIN_HELPER_DEFINITIONS = tuple(
    (name, _definition(globals()[name]))
    for name in (
        "_ready_domain_published_plugin",
        "_recorded_selector_current",
        "_capture_plugin_constructor_inputs",
        "_plugin_constructor_inputs_current",
    )
)
