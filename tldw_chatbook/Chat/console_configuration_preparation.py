"""Selected loop values and retained stock Console configuration capture."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
import inspect
import sys
from functools import partial
from types import FunctionType, MethodType, ModuleType
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


def _source_references(app, store, creator, session, *, _unused_trust=None):
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
    if _unused_trust is None:
        trust = _ready_attribute(
            local, "trust_service", "_trust_service", "_trust_service_factory"
        )
    else:
        if (
            _unused_trust.app is not app
            or _unused_trust.local is not local
            or _unused_trust.scope is not scope
            or not _unused_trust.source_current()
            or not _supported_published_skill_trust(_unused_trust)
        ):
            raise RecoveryRequired("console_snapshot_owner_changed")
        # The finite catalog operation has no trust consumer. An independent
        # initializer's ready publication is not a changed data dependency.
        trust = None
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
    try:
        session = next(
            (item for item in store.sessions() if item.id == session_id), None
        )
        if session is None:
            return False
        return _standard_configuration_references(
            app, _source_references(app, store, creator, session)
        )
    except (AttributeError, TypeError, RecoveryRequired):
        return False


def _standard_configuration_references(app, refs) -> bool:
    """Share structural checks; source readiness stays with each owning route."""
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

    # Absent optional owners need no defining feature graph before receipt.
    SkillTrustService = PluginService = LocalCharacterPersonaService = None
    if refs.trust is not None:
        from tldw_chatbook.Skills_Interop.skill_trust_service import SkillTrustService
    if refs.plugin is not None:
        from tldw_chatbook.Plugins.service import PluginService
    if refs.persona is not None:
        from tldw_chatbook.Character_Chat.local_character_persona_service import (
            LocalCharacterPersonaService,
        )

    try:
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
        if refs.mcp is not None:
            from tldw_chatbook.MCP.console_snapshot import standard_console_sources

            if not standard_console_sources(refs.mcp):
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


@dataclass(frozen=True, slots=True)
class ConsoleReceivedConfigurationPreparation:
    """Receipt-bound source identities; no catalog or execution authority yet."""

    app: Any = field(repr=False)
    store: Any = field(repr=False)
    creator: Any = field(repr=False)
    session: Any = field(repr=False)
    session_identity: tuple = field(repr=False)
    config: Any = field(repr=False)
    config_identity: Any = field(repr=False)
    refs: Any = field(repr=False)
    trust_source: Any = field(default=None, repr=False)


def _configuration_session_identity(session):
    return (
        session.incarnation_id,
        session.conversation_binding_revision,
        session.workspace_id,
        session.ephemeral,
        session.identity_revision,
        session.generation_settings_revision,
        session.settings,
    )


def _resident_skill_wiring():
    """Validate existing defining records before calling any proof helper."""
    wiring = sys.modules.get("tldw_chatbook.app_service_wiring")
    metadata = sys.modules.get("tldw_chatbook.Widgets.compact_model_bar")
    if type(wiring) is not ModuleType or type(metadata) is not ModuleType:
        return None
    values = vars(wiring)
    source = values.get("_CONSOLE_SKILL_WIRING_SOURCE")
    names = {
        "_capture_console_skill_trust_source",
        "_console_skill_metadata_current",
        "_console_skill_source_current",
    }
    if type(source) is not tuple or len(source) != 6 or type(source[5]) is not tuple:
        return None
    rows = [
        row
        for row in source[5]
        if type(row) is tuple
        and len(row) == 11
        and type(row[1]) is str  # noqa: E721 -- exact source metadata
        and row[1] in names
    ]
    if len(rows) != len(names) or any(
        type(row[2]) is not FunctionType
        or values.get(row[1]) is not row[2]
        or row[2].__code__ is not row[3]
        or row[2].__globals__ is not row[4]
        or row[2].__defaults__ is not None
        or row[2].__kwdefaults__ is not None
        or row[2].__closure__ is not None
        or vars(row[2]).get("__wrapped__") is not row[10]
        for row in rows
    ):
        return None
    if not wiring._console_skill_metadata_current(
        metadata
    ) or not wiring._console_skill_source_current(metadata, source):
        return None
    return wiring, metadata


def _supported_published_skill_trust(source):
    """Preserve existing eligible-winner rules without invoking a lazy getter."""
    module = sys.modules.get("tldw_chatbook.Skills_Interop.skill_trust_service")
    expected = (
        vars(module).get("SkillTrustService") if type(module) is ModuleType else None
    )
    return all(
        value is None
        or (
            expected is not None
            and type(value) is expected
            and "current_fingerprint_digest" not in vars(value)  # noqa: E721 -- original exact winner contract
        )
        for value in (
            vars(source.app).get("_local_skill_trust_service"),
            vars(source.local).get("_trust_service"),
        )
    )


def _skill_catalog_sources_current(preparation):
    checked = _resident_skill_wiring()
    if checked is None:
        return False
    wiring, metadata = checked
    source = preparation.trust_source
    if (
        source is None
        or not source.source_current()
        or not _supported_published_skill_trust(source)
    ):
        return False
    local_module = sys.modules.get("tldw_chatbook.Skills_Interop.local_skills_service")
    capture_module = sys.modules.get("tldw_chatbook.Chat.console_configuration_capture")
    if type(local_module) is not ModuleType or type(capture_module) is not ModuleType:
        return False
    if any(
        name in vars(source.local)
        for name in vars(local_module).get("_CONSOLE_SKILL_CATALOG_METHODS", ())
    ):
        return False
    return all(
        wiring._console_skill_source_current(metadata, record)
        for record in (
            vars(local_module).get("_CONSOLE_SKILL_CATALOG_SOURCE"),
            vars(capture_module).get("_CONSOLE_SKILL_CAPTURE_SOURCE"),
        )
    )


def _same_configuration_references(left, right):
    return (
        all(now is before for now, before in zip(left[:-1], right[:-1]))
        and len(left.callbacks) == len(right.callbacks)
        and all(now is before for now, before in zip(left.callbacks, right.callbacks))
    )


def capture_console_received_configuration_preparation(
    app, store, creator, *, session_id: str
) -> ConsoleReceivedConfigurationPreparation | None:
    """Classify resident stock readiness without constructing or reading skills."""
    from tldw_chatbook import config

    try:
        session = next(
            (item for item in store.sessions() if item.id == session_id), None
        )
        if session is None:
            return None
        trust_source = None
        if not standard_console_configuration_sources(
            app, store, creator, session_id=session_id
        ):
            checked = _resident_skill_wiring()
            if checked is None:
                return None
            wiring, _metadata = checked
            trust_source = wiring._capture_console_skill_trust_source(
                app, vars(app).get("_skills_scope_service")
            )
            if (
                trust_source is None
                or vars(trust_source.local).get("_trust_service") is not None
            ):
                return None
        refs = _source_references(
            app, store, creator, session, _unused_trust=trust_source
        )
        if not _standard_configuration_references(app, refs):
            return None
        preparation = ConsoleReceivedConfigurationPreparation(
            app,
            store,
            creator,
            session,
            _configuration_session_identity(session),
            config,
            config.current_config_identity(),
            refs,
            trust_source,
        )
        if trust_source is not None and not _skill_catalog_sources_current(preparation):
            return None
        return preparation
    except (AttributeError, TypeError, ValueError, RecoveryRequired):
        return None


def require_received_configuration_preparation(
    preparation, app, store, creator, *, session_id: str, on_loop: bool = True
):
    """Revalidate the same source/session without consuming trust publication."""
    if (
        type(preparation) is not ConsoleReceivedConfigurationPreparation
        or preparation.app is not app
        or preparation.store is not store
        or preparation.creator is not creator
        or preparation.session.id != session_id
        or next((item for item in store.sessions() if item.id == session_id), None)
        is not preparation.session
        or _configuration_session_identity(preparation.session)
        != preparation.session_identity
        or sys.modules.get("tldw_chatbook.config") is not preparation.config
        or preparation.config.current_config_identity() != preparation.config_identity
    ):
        raise RecoveryRequired("console_snapshot_owner_changed")
    source = preparation.trust_source
    if source is not None and (
        not _skill_catalog_sources_current(preparation)
        or (on_loop and not source.current())
    ):
        raise RecoveryRequired("console_snapshot_owner_changed")
    fresh = _source_references(
        app, store, creator, preparation.session, _unused_trust=source
    )
    if not _standard_configuration_references(
        app, fresh
    ) or not _same_configuration_references(fresh, preparation.refs):
        raise RecoveryRequired("console_snapshot_owner_changed")


@dataclass(frozen=True, slots=True)
class ConsoleSkillCatalogRead:
    """One finite catalog observation, confined to its received attempt."""

    preparation: ConsoleReceivedConfigurationPreparation = field(repr=False)
    session_id: str
    turn_id: str
    skill_workspace_id: str | None
    records: Any = field(repr=False)
    maximum: Any = field(repr=False)
    unavailable: bool
    require_current: Any = field(repr=False)
    reads: set = field(repr=False)
    observers: tuple = field(repr=False)


@dataclass(frozen=True, slots=True)
class ConsolePreparedSkillContext:
    """Immutable maximum plus attempt-local sources; never execution authority."""

    catalog: ConsoleSkillCatalogRead = field(repr=False)
    maximum: Mapping[str, Any] = field(repr=False)
    trust_refs: Any = field(default=None, repr=False)


def _require_skill_catalog_current(catalog, *, on_loop=True):
    if type(catalog) is not ConsoleSkillCatalogRead:
        raise RecoveryRequired("console_snapshot_owner_changed")
    preparation = catalog.preparation
    if (
        type(preparation) is not ConsoleReceivedConfigurationPreparation
        or preparation.trust_source is None
    ):
        raise RecoveryRequired("console_snapshot_owner_changed")
    require_received_configuration_preparation(
        preparation,
        preparation.app,
        preparation.store,
        preparation.creator,
        session_id=catalog.session_id,
        on_loop=on_loop,
    )
    if preparation.creator._preparation_reads is not catalog.reads:
        raise RecoveryRequired("console_snapshot_owner_changed")
    if not any(
        values is preparation.trust_source.runtime_reads for values in catalog.observers
    ):
        raise RecoveryRequired("console_snapshot_owner_changed")
    if on_loop:
        catalog.require_current()


async def capture_console_skill_catalog_owned(
    app,
    store,
    creator,
    *,
    session_id,
    turn_id,
    skill_workspace_id,
    preparation,
    reads,
    observers=(),
    require_current,
) -> ConsoleSkillCatalogRead:
    """Read original records once; only builtin projection can occur while cold."""
    from dataclasses import replace
    from .console_configuration_capture import (
        _capture_skill_context_from_records,
        _empty_local_skill_context,
    )

    catalog = ConsoleSkillCatalogRead(
        preparation,
        session_id,
        turn_id,
        skill_workspace_id,
        None,
        None,
        False,
        require_current,
        reads,
        tuple(observers),
    )
    if (
        preparation.app is not app
        or preparation.store is not store
        or preparation.creator is not creator
    ):
        raise RecoveryRequired("console_snapshot_owner_changed")
    _require_skill_catalog_current(catalog)

    def capture():
        _require_skill_catalog_current(catalog, on_loop=False)
        try:
            records = _freeze(preparation.trust_source.local._visible_records())
            if any(record.get("source") != "builtin" for record in records.values()):
                result = replace(catalog, records=records)
            else:
                maximum = _capture_skill_context_from_records(
                    preparation.trust_source.local,
                    records,
                    skill_workspace_id,
                    _plugin_service=preparation.refs.plugin,
                )
                result = replace(catalog, records=records, maximum=_freeze(maximum))
        except RecoveryRequired:
            raise
        except Exception:
            result = replace(
                catalog, unavailable=True, maximum=_freeze(_empty_local_skill_context())
            )
        _require_skill_catalog_current(catalog, on_loop=False)
        return result

    result = await run_preparation_read(
        capture,
        creator=creator,
        session_id=session_id,
        reads=reads,
        observers=observers,
        require_current=lambda: _require_skill_catalog_current(catalog, on_loop=False),
    )
    _require_skill_catalog_current(result)
    return result


async def finish_console_skill_catalog_owned(
    catalog, *, reads, observers=(), require_current
) -> ConsolePreparedSkillContext:
    """Resolve actual managed demand, then project the same captured records."""
    from .console_configuration_capture import (
        _capture_skill_context_from_records,
        _empty_local_skill_context,
    )

    if (
        type(catalog) is not ConsoleSkillCatalogRead
        or reads is not catalog.reads
        or len(observers) != len(catalog.observers)
        or any(now is not before for now, before in zip(observers, catalog.observers))
    ):
        raise RecoveryRequired("console_snapshot_owner_changed")
    _require_skill_catalog_current(catalog)
    require_current()
    if catalog.maximum is not None:
        return ConsolePreparedSkillContext(catalog, catalog.maximum)
    preparation = catalog.preparation
    source = preparation.trust_source
    initial_winner = vars(source.app).get("_local_skill_trust_service")

    def owner_current(expected=initial_winner):
        _require_skill_catalog_current(catalog)
        require_current()
        return vars(source.app).get("_local_skill_trust_service") is expected

    def native_current():
        _require_skill_catalog_current(catalog, on_loop=False)
        return True

    # No await separates winner selection from original initializer entry.
    winner = await source.app.ensure_local_skill_trust_service(
        _source_current=native_current,
        _owner_current=owner_current,
        _read_observers=(reads, *observers),
    )
    _require_skill_catalog_current(catalog)
    require_current()
    if (
        winner is None
        or vars(source.app).get("_local_skill_trust_service") is not winner
    ):
        raise RecoveryRequired("console_snapshot_owner_changed")
    # Both original getters are already proven; the ready app winner prevents
    # construction here. Preserve an independently installed local winner.
    local_winner = source.local.trust_service
    if local_winner is None:
        raise RecoveryRequired("console_capture_source_not_ready")
    _require_skill_catalog_current(catalog)
    if not standard_console_configuration_sources(
        preparation.app,
        preparation.store,
        preparation.creator,
        session_id=catalog.session_id,
    ):
        raise RecoveryRequired("console_capture_source_not_ready")
    refs = _source_references(
        preparation.app, preparation.store, preparation.creator, preparation.session
    )

    def current():
        _require_skill_catalog_current(catalog, on_loop=False)
        fresh = _source_references(
            preparation.app, preparation.store, preparation.creator, preparation.session
        )
        if not _same_configuration_references(fresh, refs):
            raise RecoveryRequired("console_snapshot_owner_changed")

    def project():
        current()
        try:
            maximum = _capture_skill_context_from_records(
                source.local,
                catalog.records,
                catalog.skill_workspace_id,
                _plugin_service=refs.plugin,
            )
        except RecoveryRequired:
            raise
        except Exception:
            maximum = _empty_local_skill_context()
        current()
        return _freeze(maximum)

    maximum = await run_preparation_read(
        project,
        creator=preparation.creator,
        session_id=catalog.session_id,
        reads=reads,
        observers=observers,
        require_current=current,
    )
    _require_skill_catalog_current(catalog)
    require_current()
    current()
    return ConsolePreparedSkillContext(catalog, maximum, refs)


def require_prepared_skill_context(
    prepared, app, store, creator, *, session_id, selection, on_loop=True
):
    """Bind the private maximum to its one live receipt and selected workspace."""
    if type(prepared) is not ConsolePreparedSkillContext:
        raise RecoveryRequired("console_snapshot_owner_changed")
    catalog = prepared.catalog
    if type(catalog) is not ConsoleSkillCatalogRead:
        raise RecoveryRequired("console_snapshot_owner_changed")
    preparation = catalog.preparation
    runtime = getattr(creator, "_hooks_v2_runtime", None)
    record = getattr(runtime, "_turn_custody", {}).get(catalog.turn_id)
    if (
        type(preparation) is not ConsoleReceivedConfigurationPreparation
        or preparation.app is not app
        or preparation.store is not store
        or preparation.creator is not creator
        or catalog.session_id != session_id
        or catalog.skill_workspace_id != selection.skill_workspace_id
        or record is None
        or record.session_id != session_id
        or record.received_intent is None
        or record.received_intent.selection is not selection
        or record.received_intent.turn_id != catalog.turn_id
        or record.store is not store
        or (prepared.trust_refs is None and prepared.maximum is not catalog.maximum)
    ):
        raise RecoveryRequired("console_snapshot_owner_changed")
    _require_skill_catalog_current(catalog, on_loop=on_loop)
    if prepared.trust_refs is not None:
        refs = _source_references(app, store, creator, preparation.session)
        if not _same_configuration_references(refs, prepared.trust_refs):
            raise RecoveryRequired("console_snapshot_owner_changed")
    return preparation.trust_source if prepared.trust_refs is None else None


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


def _composition_mcp_capture_current(creator, service):
    """Qualify the existing stock composition owner without reading tool sources."""
    from . import console_chat_controller as controller_source
    from tldw_chatbook.Agents import mcp_tool_provider as provider_source
    from tldw_chatbook.MCP import console_tool_preparation as preparation_source
    from tldw_chatbook.MCP.console_snapshot import standard_console_catalog_sources

    checker, code, namespace = controller_source._CONSOLE_TOOL_COMPOSITION_CHECKER
    return (
        type(creator) is controller_source.ConsoleChatController
        and controller_source._stock_console_tool_composition_current is checker
        and checker.__code__ is code
        and checker.__globals__ is namespace
        and namespace is vars(controller_source)
        and checker(creator)
        and provider_source._controller_factory_current(
            controller_source.MCPToolProvider
        )
        and provider_source._console_preparation_pipeline_current(preparation_source)
        and standard_console_catalog_sources(service)
    )


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
    _prepared_skills: ConsolePreparedSkillContext | None = None,
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
    from dataclasses import replace

    from tldw_chatbook import config
    from tldw_chatbook.MCP.console_snapshot import capture_console_definition_maximum
    from tldw_chatbook.DB.base_db import operation_owned_connection
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB

    require_current()
    if not isinstance(selection, ConsoleTurnCaptureSelection):
        raise TypeError("selection must be ConsoleTurnCaptureSelection")
    unused_trust = None
    if _prepared_skills is not None:
        unused_trust = require_prepared_skill_context(
            _prepared_skills,
            app,
            store,
            creator,
            session_id=session_id,
            selection=selection,
        )
    if unused_trust is None and not standard_console_configuration_sources(
        app, store, creator, session_id=session_id
    ):
        raise RecoveryRequired("console_capture_source_not_ready")
    session = next(item for item in store.sessions() if item.id == session_id)
    refs = _source_references(app, store, creator, session, _unused_trust=unused_trust)
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
        if _prepared_skills is not None:
            require_prepared_skill_context(
                _prepared_skills,
                app,
                store,
                creator,
                session_id=session_id,
                selection=selection,
                on_loop=False,
            )
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
        fresh = _source_references(
            app, store, creator, session, _unused_trust=unused_trust
        )
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

    stock_mcp = refs.mcp is not None and _composition_mcp_capture_current(
        creator, refs.mcp
    )
    composition_mcp = stock_mcp and not session.ephemeral
    maximum = {}
    if refs.mcp is not None and not stock_mcp:
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
                **(
                    {}
                    if _prepared_skills is None
                    else {
                        "_skill_context_maximum": _prepared_skills.maximum,
                    }
                ),
            )
            current()
            if composition_mcp:
                result = replace(
                    result,
                    mcp_tool_maximum=None,
                    mcp_definition_capture="composition",
                )
            return result

    result = await run_native(capture)
    current()
    if stock_mcp and not _composition_mcp_capture_current(creator, refs.mcp):
        raise RecoveryRequired("console_snapshot_owner_changed")
    require_current()
    return result
