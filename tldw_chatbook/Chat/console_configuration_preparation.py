"""Selected loop values and retained stock Console configuration capture."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import inspect
import sys
from functools import partial
from types import MethodType
from typing import Any, Callable, Mapping, NamedTuple

from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from .console_chat_models import ConsoleProviderSelection
from .console_configuration_capture import (
    CONSOLE_MCP_BUILTIN_RAW_NAME_EXCLUSIONS,
    capture_console_turn_configuration,
)
from .console_preparation_reads import run_preparation_read
from .console_turn_context import (
    ConsoleTurnConfigurationSnapshot,
    _detached_selection,
    _freeze,
    resolve_turn_persona_policy_rules,
    resolve_turn_tool_policy_profile_id,
)


@dataclass(frozen=True, slots=True)
class ConsoleTurnCaptureSelection:
    """Detached selected values; None maps request existing runtime defaults."""

    provider_selection: ConsoleProviderSelection
    presentation_context: Any
    rag_defaults: Mapping[str, Any] | None
    tool_configuration: Mapping[str, Any] | None
    skill_workspace_id: str | None
    project_bindings_eligible: bool
    agent_runtime_enabled: bool

    def __post_init__(self):
        if not isinstance(self.provider_selection, ConsoleProviderSelection):
            raise TypeError("provider_selection must be ConsoleProviderSelection")
        if not isinstance(self.project_bindings_eligible, bool) or not isinstance(
            self.agent_runtime_enabled, bool
        ):
            raise TypeError("capture eligibility flags must be bool")
        object.__setattr__(
            self, "provider_selection", _detached_selection(self.provider_selection)
        )
        object.__setattr__(
            self, "presentation_context", _freeze(self.presentation_context)
        )
        for name in ("rag_defaults", "tool_configuration"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Mapping):
                raise TypeError(f"{name} must be Mapping or None")
            object.__setattr__(self, name, None if value is None else _freeze(value))


def _ready_attribute(owner, name, backing=None, factory=None):
    """Inspect existing slots without invoking a native lazy service getter."""
    if owner is None:
        return None
    values = vars(owner)
    descriptor = inspect.getattr_static(owner, name, None)
    if isinstance(descriptor, property):
        if backing is None or backing not in values:
            raise RecoveryRequired("console_capture_source_not_ready")
        value = values[backing]
        if value is None and (factory is None or values.get(factory) is not None):
            raise RecoveryRequired("console_capture_source_not_ready")
        return value
    return values.get(name, descriptor)


class _ConfigurationSources(NamedTuple):
    app_config: Any
    registry: Any
    workspace_database: Any
    consent: Any
    scope: Any
    local: Any
    trust: Any
    plugin: Any
    persistence: Any
    chat_database: Any
    prompt_database: Any
    visual: Any
    visual_database: Any
    scratch: Any
    mcp: Any
    provider_config: Any
    persona: Any
    persona_database: Any
    consent_registry: Any
    callbacks: tuple[Any, ...]


def _ready_published_plugin(local, app):
    if local is None:
        return None
    from tldw_chatbook.Skills_Interop.local_skills_service import (
        LocalSkillsService,
        _STOCK_PLUGIN_SERVICE_PROPERTY,
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
    existing = vars(local).get("_plugin_service")
    if existing is not None:
        return existing
    factory = vars(local).get("_plugin_service_factory")
    if factory is None:
        return None
    wiring = sys.modules.get("tldw_chatbook.app_service_wiring")
    namespace = vars(wiring) if wiring is not None else {}
    original = namespace.get("_STOCK_PLUGIN_SERVICE_FACTORY")
    reader = namespace.get("_STOCK_PUBLISHED_PLUGIN_READER")
    if (
        type(factory) is not MethodType
        or type(original) is not tuple
        or len(original) != 2
        or factory.__self__ is not app
        or factory.__func__ is not original[0]
        or original[0].__code__ is not original[1]
        or type(reader) is not tuple
        or len(reader) != 2
        or namespace.get("_read_app_published_plugin_service") is not reader[0]
        or reader[0].__code__ is not reader[1]
    ):
        raise RecoveryRequired("console_capture_source_not_ready")
    return reader[0](app)


def _source_references(app, store, creator, session):
    registry = _ready_attribute(
        app, "workspace_registry_service", "_workspace_registry_service"
    )
    consent = _ready_attribute(
        app, "change_review_consent_service", "_change_review_consent_service"
    )
    scope = _ready_attribute(app, "skills_scope_service", "_skills_scope_service")
    local = (
        _ready_attribute(scope, "local_service") if scope is not None else None
    ) or _ready_attribute(app, "local_skills_service", "_local_skills_service")
    trust = _ready_attribute(
        local, "trust_service", "_trust_service", "_trust_service_factory"
    )
    plugin = _ready_published_plugin(local, app)
    persistence = getattr(store, "persistence", None)
    visual = getattr(creator, "_visual_identity_repository", None)
    persona = (
        _ready_attribute(
            app, "local_character_persona_service", "_local_character_persona_service"
        )
        if session is not None and session.assistant_kind == "persona"
        else None
    )
    return _ConfigurationSources(
        _ready_attribute(app, "app_config"),
        registry,
        getattr(registry, "db", None),
        consent,
        scope,
        local,
        trust,
        plugin,
        persistence,
        getattr(persistence, "db", None),
        _ready_attribute(app, "chachanotes_db"),
        visual,
        getattr(visual, "db", None),
        getattr(creator, "_scratch_spaces", None),
        _ready_attribute(app, "unified_mcp_service"),
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
        ),
    )


def standard_console_configuration_sources(
    app, store, creator, *, session_id: str
) -> bool:
    """Allow ready stock owners only; cold/custom/memory affinity stays unchanged."""
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.DB.VisualIdentity_DB import VisualIdentityRepository
    from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
    from tldw_chatbook.Skills_Interop.skills_scope_service import SkillsScopeService
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService
    from .console_scratch_space import ConsoleScratchSpaceManager
    from tldw_chatbook.Workspaces.change_review_consent import (
        ChangeReviewConsentService,
        _default_capability_reader,
    )
    from tldw_chatbook.Skills_Interop.skill_trust_service import SkillTrustService
    from tldw_chatbook.Plugins.service import PluginService
    from tldw_chatbook.Character_Chat.local_character_persona_service import (
        LocalCharacterPersonaService,
    )
    from tldw_chatbook.MCP.console_snapshot import standard_console_sources

    try:
        session = next(
            (item for item in store.sessions() if item.id == session_id), None
        )
        if session is None:
            return False
        refs = _source_references(app, store, creator, session)
        registry, local, visual, scratch = (
            refs.registry,
            refs.local,
            refs.visual,
            refs.scratch,
        )
        if registry is not None and type(registry) is not LocalWorkspaceRegistryService:
            return False
        if refs.scope is not None and type(refs.scope) is not SkillsScopeService:
            return False
        if local is not None and type(local) is not LocalSkillsService:
            return False
        loader = getattr(local, "_builtin_disabled_loader", None)
        if loader is not None:
            # The app wiring module is already loaded by a stock app. Do not
            # import its boot/UI dependency graph merely to check this reader.
            wiring = sys.modules.get("tldw_chatbook.app_service_wiring")
            namespace = vars(wiring) if wiring is not None else {}
            original = namespace.get("_STOCK_DISABLED_BUILTINS_READER")
            if (
                type(loader) is not partial
                or type(original) is not tuple
                or len(original) != 2
                or loader.func is not original[0]
                or namespace.get("_read_app_disabled_builtin_skills") is not original[0]
                or original[0].__code__ is not original[1]
                or len(loader.args) != 1
                or loader.args[0] is not app
                or loader.keywords
            ):
                return False
        if visual is not None and type(visual) is not VisualIdentityRepository:
            return False
        if type(scratch) is not ConsoleScratchSpaceManager or "snapshot" in vars(
            scratch
        ):
            return False
        for owner, expected, callbacks in (
            (refs.consent, ChangeReviewConsentService, ("admit_turn",)),
            (refs.trust, SkillTrustService, ("current_fingerprint_digest",)),
            (refs.plugin, PluginService, ("capture_maximum",)),
            (refs.persona, LocalCharacterPersonaService, ("get_persona_profile",)),
        ):
            if owner is not None and (
                type(owner) is not expected
                or any(name in vars(owner) for name in callbacks)
            ):
                return False
        if refs.consent is not None and (
            refs.consent._capability_reader is not _default_capability_reader
            or refs.consent_registry is not refs.registry
        ):
            return False
        if local is not None and any(
            name in vars(local) for name in ("_visible_records", "_summary_for_record")
        ):
            return False
        if refs.mcp is not None and not standard_console_sources(refs.mcp):
            return False
        for database in (
            refs.workspace_database,
            refs.chat_database,
            refs.prompt_database,
            refs.visual_database,
            refs.persona_database,
        ):
            if database is not None and (
                type(database) not in (CharactersRAGDB, WorkspaceDB)
                or database.is_memory_db
            ):
                return False
        return True
    except (AttributeError, TypeError, RecoveryRequired):
        return False


def _runtime_tool_configuration(raw, selection):
    from tldw_chatbook import config
    from .console_agent_bridge import console_run_budget
    from .console_chat_controller import (
        coerce_bool_setting,
        coerce_int_setting,
        DEFAULT_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
        MIN_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
        MAX_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
    )

    section = raw.get("console", {}) if isinstance(raw, Mapping) else {}
    if not isinstance(section, Mapping):
        section = {}
    result = {
        "agent_runtime_enabled": selection.agent_runtime_enabled,
        "native_tool_calls_enabled": coerce_bool_setting(
            section.get("native_tool_calls", True), True
        ),
        "local_tools_enabled": coerce_bool_setting(
            config.get_cli_setting("console", "local_tools_enabled", False), False
        ),
        "direct_library_tools": coerce_bool_setting(
            config.get_cli_setting("console", "direct_library_tools", True), True
        ),
        "exchange_capture_enabled": coerce_bool_setting(
            config.get_cli_setting("console", "exchange_capture", True), True
        ),
        "agent_run_budget_maximum": console_run_budget(),
    }
    for name in (
        "project_instructions_startup_max_bytes",
        "project_instructions_nested_max_bytes",
    ):
        result[name] = coerce_int_setting(
            config.get_cli_setting(
                "console", name, DEFAULT_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES
            ),
            DEFAULT_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
            minimum=MIN_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
            maximum=MAX_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
        )
    return result


async def capture_console_turn_configuration_owned(
    app,
    store,
    session_id,
    *,
    selection: ConsoleTurnCaptureSelection,
    creator,
    reads,
    observers=(),
    require_current: Callable[[], None],
) -> ConsoleTurnConfigurationSnapshot:
    """Capture eligible native sources through retained finite operations.

    Args:
        app: Original application owner.
        store: Original session store.
        session_id: Exact selected session.
        selection: Detached loop-selected values, never a screen callback.
        creator: Original controller or runtime observing physical lifetime.
        reads: Existing creator physical-read set.
        observers: Additional exact teardown observer sets.
        require_current: Caller-loop checks; never invoked by a native worker.

    Returns:
        The existing complete snapshot from the shared synchronous producer.

    Raises:
        RecoveryRequired: Readiness or original source identity changed.
        asyncio.CancelledError: Cancellation after issued native work retires.
    """
    from tldw_chatbook import config
    from tldw_chatbook.MCP.console_snapshot import capture_console_definition_maximum
    from tldw_chatbook.DB.base_db import operation_owned_connection
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB

    require_current()
    if not isinstance(selection, ConsoleTurnCaptureSelection):
        raise TypeError("selection must be ConsoleTurnCaptureSelection")
    if not standard_console_configuration_sources(
        app, store, creator, session_id=session_id
    ):
        raise RecoveryRequired("console_capture_source_not_ready")
    session = next(item for item in store.sessions() if item.id == session_id)
    refs = _source_references(app, store, creator, session)
    identity = (
        session.incarnation_id,
        session.conversation_binding_revision,
        session.workspace_id,
        session.ephemeral,
        session.identity_revision,
        session.generation_settings_revision,
        session.settings,
    )
    config_identity = config.current_config_identity()
    runtime = getattr(creator, "_hooks_v2_runtime", None)
    runtime_store = getattr(runtime, "_chat_store", None)
    runtime_controller = getattr(runtime, "_chat_controller", None)
    raw_config = refs.app_config
    # Existing configurable selection callbacks keep their caller-loop affinity.
    if selection.tool_configuration is None:
        raw_config = (
            _freeze(refs.provider_config()) if callable(refs.provider_config) else {}
        )
    else:
        raw_config = _freeze(raw_config)
    require_current()

    def current():
        actual = next(
            (item for item in store.sessions() if item.id == session_id), None
        )
        if (
            actual is not session
            or getattr(creator, "_disposed", False)
            or getattr(creator, "store", store) is not store
            or getattr(creator, "app", app) is not app
            or config.current_config_identity() != config_identity
            or (
                runtime is not None
                and (
                    getattr(creator, "_hooks_v2_runtime", None) is not runtime
                    or runtime._disposed
                    or runtime._chat_store is not runtime_store
                    or runtime._chat_controller is not runtime_controller
                    or session_id in runtime._admission_fenced_sessions
                )
            )
            or (
                session.incarnation_id,
                session.conversation_binding_revision,
                session.workspace_id,
                session.ephemeral,
                session.identity_revision,
                session.generation_settings_revision,
                session.settings,
            )
            != identity
        ):
            raise RecoveryRequired("console_snapshot_owner_changed")
        fresh = _source_references(app, store, creator, session)
        if any(now is not before for now, before in zip(fresh[:-1], refs[:-1])) or any(
            now is not before for now, before in zip(fresh.callbacks, refs.callbacks)
        ):
            raise RecoveryRequired("console_snapshot_owner_changed")

    async def run_native(callback):
        require_current()
        result = await run_preparation_read(
            callback,
            creator=creator,
            session_id=session_id,
            reads=reads,
            observers=observers,
            require_current=current,
        )
        require_current()
        return result

    maximum = {}
    if refs.mcp is not None:
        maximum = await capture_console_definition_maximum(
            refs.mcp,
            CONSOLE_MCP_BUILTIN_RAW_NAME_EXCLUSIONS,
            _run_native=run_native,
        )
    current()
    require_current()

    def capture():
        from .console_chat_controller import capture_project_instruction_authority
        from .console_agent_bridge import console_run_budget
        from tldw_chatbook.Library.library_rag_state import library_rag_profile_top_k

        current()
        with ExitStack() as scope:
            databases = []
            for database in (
                refs.chat_database,
                refs.prompt_database,
                refs.visual_database,
                refs.workspace_database,
                refs.persona_database,
            ):
                if database is not None and not any(
                    database is prior for prior in databases
                ):
                    databases.append(database)
                    scope.enter_context(operation_owned_connection(database))
                    if type(database) is WorkspaceDB:
                        scope.enter_context(database.connection())
            current()
            scratch = refs.scratch.snapshot(session_id)
            current()
            project = capture_project_instruction_authority(
                session,
                refs.registry,
                include_bindings=selection.project_bindings_eligible,
            )
            current()
            profile = resolve_turn_tool_policy_profile_id(app, session.workspace_id)
            current()
            rules = resolve_turn_persona_policy_rules(app, session)
            current()
            tools = (
                _runtime_tool_configuration(raw_config, selection)
                if selection.tool_configuration is None
                else dict(selection.tool_configuration)
            )
            tools["session_ephemeral"] = bool(session.ephemeral)
            if "agent_run_budget_maximum" not in tools:
                tools["agent_run_budget_maximum"] = console_run_budget()
            rag = {} if selection.rag_defaults is None else dict(selection.rag_defaults)
            if "top_k" not in rag:
                rag["top_k"] = library_rag_profile_top_k()
            current()
            result = capture_console_turn_configuration(
                app,
                store,
                session_id,
                provider_selection=selection.provider_selection,
                scratch_space=scratch,
                presentation_context=selection.presentation_context,
                rag_defaults=rag,
                tool_configuration=tools,
                project_authority=project,
                skill_workspace_id=selection.skill_workspace_id,
                character_repository=refs.visual,
                tool_policy_profile_id=profile,
                persona_policy_rules=rules,
                mcp_definition_maximum=maximum,
                _require_current=current,
                _plugin_service=refs.plugin,
            )
            current()
            return result

    result = await run_native(capture)
    current()
    require_current()
    return result
