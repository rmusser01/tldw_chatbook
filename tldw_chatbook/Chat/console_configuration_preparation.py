"""Selected loop values and retained stock Console configuration capture."""

from __future__ import annotations

import asyncio
import inspect
import os
import sys
from collections.abc import Callable, Mapping
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import partial
from types import FunctionType, MethodType, ModuleType
from typing import Any, NamedTuple

from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

from .console_chat_models import ConsoleProviderSelection
from .console_configuration_capture import (
    CONSOLE_MCP_BUILTIN_RAW_NAME_EXCLUSIONS,
    _has_managed_console_runtime,
    _mcp_attachment_source,
    _require_mcp_attachment_source,
    capture_console_turn_configuration,
)
from .console_ordinary_tree import _function_record, _function_unchanged
from .console_preparation_reads import run_preparation_read
from .console_turn_context import (
    ConsoleTurnConfigurationSnapshot,
    _detached_selection,
    _freeze,
    resolve_turn_persona_policy_rules,
    resolve_turn_tool_policy_profile_id,
)

_mcp_capture_witness = ContextVar("console_mcp_capture_witness", default=None)


@dataclass(frozen=True, slots=True)
class _QueuedCaptureTransfer:
    witness: tuple = field(repr=False)
    configuration: Any = field(repr=False)
    preparation: Any = field(repr=False)
    app_trust: Any = field(repr=False)
    local_trust: Any = field(repr=False)
    unused_trust: bool


@dataclass(frozen=True, slots=True)
class _ClaimedCaptureTransfer:
    transfer: _QueuedCaptureTransfer = field(repr=False)
    prompt: Any = field(repr=False)
    authorization: Any = field(repr=False)
    witness: tuple = field(repr=False)


@dataclass(frozen=True, slots=True)
class _DeferredSkillCapture:
    preparation: Any = field(repr=False)
    selection: Any = field(repr=False)


@dataclass(frozen=True, slots=True)
class _NativeSkillTrustPublication:
    source: Any = field(repr=False)
    initial_app: Any = field(repr=False)
    initial_local: Any = field(repr=False)
    app_winner: Any = field(repr=False)
    local_winner: Any = field(repr=False)
    source_functions: tuple = field(repr=False)
    guard: Any = field(repr=False)


@dataclass(frozen=True, slots=True)
class _FirstLibraryPublication:
    controller: Any = field(repr=False)
    store: Any = field(repr=False)
    session: Any = field(repr=False)
    holder: Any = field(repr=False)
    commit: Any = field(repr=False)
    continuation: Any = field(repr=False)
    fingerprint: Any = field(repr=False)
    result: Any = field(repr=False)
    source: Any = field(repr=False)
    binding_revision: int
    verified: bool = False


@dataclass(slots=True)
class _FirstLibraryTransport:
    controller: Any = field(repr=False)
    store: Any = field(repr=False)
    session: Any = field(repr=False)
    commit: Any = field(repr=False)
    continuation: Any = field(repr=False)
    fingerprint: Any = field(repr=False)
    source: Any = field(repr=False)
    results: list = field(default_factory=list, repr=False)


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
    except (AttributeError, TypeError, ValueError, RecoveryRequired):
        return False


def _standard_configuration_references(app, refs) -> bool:
    """Share structural checks; source readiness stays with each owning route."""
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.VisualIdentity_DB import VisualIdentityRepository
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
    from tldw_chatbook.Skills_Interop.skills_scope_service import SkillsScopeService
    from tldw_chatbook.Workspaces.change_review_consent import (
        ChangeReviewConsentService,
    )
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

    from .console_scratch_space import ConsoleScratchSpaceManager

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
        if refs.consent is not None and refs.consent_registry is not refs.registry:
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
    trust_initial: tuple = field(default=(), repr=False)
    trust_functions: tuple = field(default=(), repr=False)


def _configuration_session_identity(session):
    return (
        session.incarnation_id,
        session.conversation_binding_revision,
        session.workspace_id,
        session.ephemeral,
        session.identity_revision,
        session.generation_settings_revision,
        session.prompt_mixin_preferences.preferences.revision
        if session.prompt_mixin_preferences.preferences is not None
        else None,
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
        and type(row[1]) is str  # exact source metadata
        and row[1] in names
    ]
    if len(rows) != len(names) or any(
        type(row[2]) is not FunctionType
        or values.get(row[1]) is not row[2]
        or row[2].__code__ is not row[3]
        or row[2].__globals__ is not row[4]
        or row[2].__globals__.get("_source_wiring") is not wiring
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
            and "current_fingerprint_digest" not in vars(value)  # original exact winner contract
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
        all(now is before for now, before in zip(left[:-1], right[:-1], strict=False))
        and len(left.callbacks) == len(right.callbacks)
        and all(
            now is before
            for now, before in zip(left.callbacks, right.callbacks, strict=False)
        )
    )


def _first_library_source_current(source):
    """Check the original native Q1 publisher/repository without invoking them."""
    (
        store,
        persistence,
        repository,
        database,
        path,
        store_module,
        repository_module,
        store_record,
        repo_record,
        persistence_module,
        persistence_type,
    ) = source
    if (
        type(store_record) is not tuple
        or len(store_record) != 6
        or type(repo_record) is not tuple
        or len(repo_record) != 12
        or type(store) is not store_record[0]
        or type(repository) is not repo_record[0]
        or type(persistence) is not persistence_type
        or sys.modules.get("tldw_chatbook.Chat.console_chat_store") is not store_module
        or sys.modules.get("tldw_chatbook.Chat.console_library_policy_repository")
        is not repository_module
        or sys.modules.get("tldw_chatbook.Chat.chat_persistence_service")
        is not persistence_module
        or vars(persistence_module).get("ChatPersistenceService")
        is not persistence_type
        or vars(store_module).get("_NATIVE_FIRST_LIBRARY_STORE") is not store_record
        or vars(repository_module).get("_NATIVE_FIRST_LIBRARY_REPOSITORY")
        is not repo_record
        or vars(store).get("persistence") is not persistence
        or vars(persistence).get("console_library_policy_repository") is not repository
        or vars(persistence).get("db") is not database
        or vars(repository).get("db") is not database
        or type(database) is not repo_record[6]
        or vars(database).get("db_path") is not path
        or any(
            name in vars(store)
            for name in (
                "publish_durable_turn_identity",
                "publish_committed_identity",
                "_session_or_raise",
            )
        )
        or any(name in vars(repository) for name in ("read", "_result_from_row"))
    ):
        return False
    (
        store_type,
        publish,
        implementation,
        identity,
        identity_implementation,
        session_lookup,
    ) = store_record
    (
        repo_type,
        read,
        row,
        normalize,
        valid,
        sql,
        db_type,
        policy_type,
        result_type,
        snapshot_type,
        retrieve_type,
        access_type,
    ) = repo_record
    policy_module = vars(repository_module).get("_ordinary_policy_source")
    row_descriptor = vars(repo_type).get("_result_from_row")
    return bool(
        vars(store_module).get("ConsoleChatStore") is store_type
        and inspect.getattr_static(store_type, "publish_durable_turn_identity", None)
        is publish[0]
        and vars(publish[0]).get("__wrapped__") is implementation[0]
        and inspect.getattr_static(store_type, "publish_committed_identity", None)
        is identity[0]
        and vars(identity[0]).get("__wrapped__") is identity_implementation[0]
        and inspect.getattr_static(store_type, "_session_or_raise", None)
        is session_lookup[0]
        and vars(repository_module).get("ConsoleLibraryPolicyRepository") is repo_type
        and vars(repository_module).get("CharactersRAGDB") is db_type
        and vars(repository_module).get("ConsoleConversationLibraryPolicy")
        is policy_type
        and vars(repository_module).get("ConsoleLibraryPolicyReadResult") is result_type
        and vars(repository_module).get("ConsoleAutoRetrieve") is retrieve_type
        and vars(repository_module).get("ConsoleAssistantLibraryAccess") is access_type
        and inspect.getattr_static(repo_type, "read", None) is read[0]
        and type(row_descriptor) is staticmethod
        and row_descriptor.__func__ is row[0]
        and vars(repository_module).get("normalize_policy_read") is normalize[0]
        and vars(repository_module).get("_POLICY_SELECT") is sql
        and type(policy_module) is ModuleType
        and vars(policy_module).get("_is_valid_durable_policy") is valid[0]
        and vars(policy_module).get("ConsoleLibraryPolicySnapshot") is snapshot_type
        and vars(policy_module).get("ConsoleConversationLibraryPolicy") is policy_type
        and vars(policy_module).get("ConsoleLibraryPolicyReadResult") is result_type
        and vars(policy_module).get("ConsoleAutoRetrieve") is retrieve_type
        and vars(policy_module).get("ConsoleAssistantLibraryAccess") is access_type
        and all(
            _function_unchanged(record)
            for record in (*store_record[1:], *repo_record[1:5])
        )
    )


def begin_first_library_publication(
    controller, session_id, preparation_id, fingerprint, commit, continuation
):
    """Prepare one private transport for the original first committed policy."""
    from .console_ordinary_tree import current_ordinary_tree

    if not commit.first_persist:
        return None
    store = controller.store
    session = store._sessions.get(session_id)
    tree = current_ordinary_tree(controller)
    store_module = sys.modules.get("tldw_chatbook.Chat.console_chat_store")
    repository_module = sys.modules.get(
        "tldw_chatbook.Chat.console_library_policy_repository"
    )
    persistence_module = sys.modules.get("tldw_chatbook.Chat.chat_persistence_service")
    if (
        session is None
        or tree is None
        or tree.store is not store
        or tree.session is not session
    ):
        return None
    if any(
        type(module) is not ModuleType
        for module in (store_module, repository_module, persistence_module)
    ):
        return None
    persistence = store.persistence
    repository = getattr(persistence, "console_library_policy_repository", None)
    database = getattr(persistence, "db", None)
    source = (
        store,
        persistence,
        repository,
        database,
        getattr(database, "db_path", None),
        store_module,
        repository_module,
        vars(store_module).get("_NATIVE_FIRST_LIBRARY_STORE"),
        vars(repository_module).get("_NATIVE_FIRST_LIBRARY_REPOSITORY"),
        persistence_module,
        vars(persistence_module).get("ChatPersistenceService"),
    )
    try:
        tree.check()
        if (
            not _first_library_source_current(source)
            or controller._durable_postcommit_continuations.get(preparation_id)
            is not continuation
            or continuation.commit is not commit
            or store.durable_turn_commit_for(preparation_id, fingerprint=fingerprint)
            is not commit
        ):
            return None
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return None
    return _FirstLibraryTransport(
        controller, store, session, commit, continuation, fingerprint, source
    )


def stage_first_library_publication(transport):
    """Keep only the result returned by this original store invocation."""
    from dataclasses import replace

    from .console_library_policy import (
        ConsoleConversationLibraryPolicy,
        ConsoleLibraryPolicyReadResult,
        ConsoleLibraryPolicySnapshot,
    )

    if type(transport) is not _FirstLibraryTransport or len(transport.results) != 1:
        return
    controller, store, session, commit = (
        transport.controller,
        transport.store,
        transport.session,
        transport.commit,
    )
    observed_session, holder, result = transport.results[0]
    old = commit.checkpoint.frozen_authority.policy
    if (
        observed_session is not session
        or session.library_policy_holder is not holder
        or type(result) is not ConsoleLibraryPolicyReadResult
        or type(result.durable_policy) is not ConsoleConversationLibraryPolicy
        or type(result.snapshot) is not ConsoleLibraryPolicySnapshot
        or type(old) is not ConsoleLibraryPolicySnapshot
        or old.source != "new_session"
        or old.policy_revision is not None
        or result.snapshot.source != "durable"
        or result.snapshot.policy_revision != 1
        or result.durable_policy.conversation_id != commit.identity.conversation_id
        or result.durable_policy.policy_revision != 1
        or replace(
            result.snapshot, source=old.source, policy_revision=old.policy_revision
        )
        != old
        or holder.snapshot is not result.snapshot
        or session.persisted_conversation_id != commit.identity.conversation_id
        or not _first_library_source_current(transport.source)
    ):
        return
    receipt = _FirstLibraryPublication(
        controller,
        store,
        session,
        holder,
        commit,
        transport.continuation,
        transport.fingerprint,
        result,
        transport.source,
        session.conversation_binding_revision,
    )
    original_fingerprint = store._durable_fingerprint_by_preparation.get(
        commit.checkpoint.preparation_id
    )
    if original_fingerprint != transport.fingerprint:
        return
    with store._preparation_lock:
        if (
            store._sessions.get(session.id) is not session
            or not _first_library_source_current(transport.source)
            or controller._durable_postcommit_continuations.get(
                commit.checkpoint.preparation_id
            )
            is not transport.continuation
            or store._durable_commit_by_preparation.get(
                commit.checkpoint.preparation_id
            )
            is not commit
            or store._durable_fingerprint_by_preparation.get(
                commit.checkpoint.preparation_id
            )
            is not original_fingerprint
            or (commit.checkpoint.preparation_id, "identity_publication")
            not in store._durable_effects_in_flight
            or session.library_policy_holder is not holder
            or holder.snapshot is not result.snapshot
        ):
            return
        vars(session)["_console_first_library_publication"] = receipt


def verify_first_library_publication(
    controller, session_id, preparation_id, fingerprint, commit, continuation
):
    """Verify the same provisional result after the original ledger completed."""
    from dataclasses import replace

    store = controller.store
    session = store._sessions.get(session_id)
    receipt = (
        vars(session).get("_console_first_library_publication")
        if session is not None
        else None
    )
    if (
        type(receipt) is not _FirstLibraryPublication
        or receipt.controller is not controller
        or receipt.store is not store
        or receipt.session is not session
        or receipt.commit is not commit
        or receipt.continuation is not continuation
        or receipt.fingerprint != fingerprint
        or not _first_library_source_current(receipt.source)
    ):
        return
    try:
        effects = store.durable_postcommit_effects_for(
            preparation_id, fingerprint=fingerprint
        )
        original_commit = store.durable_turn_commit_for(
            preparation_id, fingerprint=fingerprint
        )
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return
    if (
        effects is None
        or "identity_publication" not in effects.completed
        or original_commit is not commit
    ):
        return
    original_fingerprint = store._durable_fingerprint_by_preparation.get(preparation_id)
    if original_fingerprint != fingerprint:
        return
    verified = replace(receipt, verified=True)
    with store._preparation_lock:
        if (
            store._sessions.get(session_id) is not session
            or not _first_library_source_current(receipt.source)
            or vars(session).get("_console_first_library_publication") is not receipt
            or controller._durable_postcommit_continuations.get(preparation_id)
            is not continuation
            or store._durable_commit_by_preparation.get(preparation_id) is not commit
            or store._durable_fingerprint_by_preparation.get(preparation_id)
            is not original_fingerprint
            or store._durable_effects_by_preparation.get(preparation_id) is not effects
            or session.library_policy_holder is not receipt.holder
            or receipt.holder.snapshot is not receipt.result.snapshot
            or session.persisted_conversation_id != commit.identity.conversation_id
            or session.conversation_binding_revision != receipt.binding_revision
        ):
            return
        vars(session)["_console_first_library_publication"] = verified


def _first_library_coordinator_current(
    coordinator, repository, registered, session, record
):
    module = sys.modules.get("tldw_chatbook.Chat.console_library_policy_coordinator")
    if (
        type(module) is not ModuleType
        or vars(module).get("_NATIVE_FIRST_LIBRARY_COORDINATOR") is not record
        or type(record) is not tuple
        or len(record) != 2
        or type(coordinator) is not record[0]
        or vars(module).get("ConsoleLibraryPolicyCoordinator") is not record[0]
        or vars(coordinator).get("repository") is not repository
        or vars(coordinator).get("_holders", {}).get(session.id) is not registered
        or registered.holder is not session.library_policy_holder
        or registered.conversation_id != session.persisted_conversation_id
    ):
        return False
    for name, function in record[1]:
        if (
            name in vars(coordinator)
            or inspect.getattr_static(type(coordinator), name, None) is not function[0]
            or not _function_unchanged(function)
        ):
            return False
    return True


def _first_library_witness_current(controller, session, tree, configuration, receipt):
    state = _mcp_capture_witness.get()
    if type(state) is _ClaimedCaptureTransfer:
        original = state.transfer.configuration
    elif type(state) is tuple:
        original = state[-4]
    else:
        return False
    initial = tree.original_configuration
    if (
        original is None
        or initial.library_policy_maximum != original.library_policy_maximum
        or configuration.library_policy_maximum != original.library_policy_maximum
        or original.library_policy_maximum
        != receipt.commit.checkpoint.frozen_authority.policy
        or not original_mcp_capture_matches(
            controller,
            session,
            initial,
            pending=initial.mcp_capture_pending,
            require_complete=False,
        )
    ):
        return False
    task = asyncio.current_task()
    records = [
        record
        for record in tree.runtime._turn_custody.values()
        if record.task is task
        and record.request is not None
        and record.request.configuration is original
    ]
    return len(records) == 1


def begin_first_library_execution_read(controller, session, configuration, coordinator):
    """Prepare a no-read witness for the existing live policy capture."""
    from .console_ordinary_tree import current_ordinary_tree

    tree = current_ordinary_tree(controller)
    receipt = vars(session).get("_console_first_library_publication")
    module = sys.modules.get("tldw_chatbook.Chat.console_library_policy_coordinator")
    record = (
        vars(module).get("_NATIVE_FIRST_LIBRARY_COORDINATOR")
        if type(module) is ModuleType
        else None
    )
    registered = (
        vars(coordinator).get("_holders", {}).get(session.id)
        if coordinator is not None
        else None
    )
    if (
        tree is None
        or tree.session is not session
        or tree.store is not controller.store
        or type(receipt) is not _FirstLibraryPublication
        or not receipt.verified
        or receipt.controller is not controller
        or receipt.store is not controller.store
        or receipt.session is not session
        or receipt.holder is not session.library_policy_holder
        or receipt.binding_revision != session.conversation_binding_revision
        or session.persisted_conversation_id != receipt.commit.identity.conversation_id
        or not _first_library_source_current(receipt.source)
        or not _first_library_witness_current(
            controller, session, tree, configuration, receipt
        )
        or not _first_library_coordinator_current(
            coordinator, receipt.source[2], registered, session, record
        )
    ):
        return None
    return receipt, tree, coordinator, registered, record


def finish_first_library_execution_read(
    controller, session, configuration, policy, token
):
    """Accept only the original live native read of the same Q1 policy."""
    from .console_library_policy import ConsoleLibraryPolicySnapshot

    if type(token) is not tuple or len(token) != 5:
        return False
    receipt, tree, coordinator, registered, record = token
    return bool(
        vars(session).get("_console_first_library_publication") is receipt
        and receipt.verified
        and receipt.session is session
        and receipt.store is controller.store
        and receipt.holder is session.library_policy_holder
        and receipt.holder.snapshot is policy
        and session.conversation_binding_revision == receipt.binding_revision
        and session.persisted_conversation_id == receipt.commit.identity.conversation_id
        and type(policy) is ConsoleLibraryPolicySnapshot
        and policy == receipt.result.snapshot
        and _first_library_source_current(receipt.source)
        and _first_library_coordinator_current(
            coordinator, receipt.source[2], registered, session, record
        )
        and _first_library_witness_current(
            controller, session, tree, configuration, receipt
        )
    )


def _same_trust_slots(source, expected):
    return (
        type(expected) is tuple
        and len(expected) == 2
        and vars(source.app).get("_local_skill_trust_service") is expected[0]
        and vars(source.local).get("_trust_service") is expected[1]
    )


def _trust_source_functions_current(preparation):
    return _trust_source_records_current(
        preparation.trust_source, preparation.trust_functions
    )


def _trust_source_records_current(source, records):
    return bool(
        source is not None
        and len(records) == 2
        and source.source_current is records[0][0]
        and source.current is records[1][0]
        and all(_function_unchanged(record) for record in records)
    )


def _skill_publisher_guard(preparation):
    """Prepare exact plain identities before the publication fence."""
    source = preparation.trust_source
    app, local = source.app, source.local
    bindings = []
    functions = []
    for obj, name in (
        (app, "local_skill_trust_service"),
        (app, "_build_local_skill_trust_service"),
        (app, "ensure_local_skill_trust_service"),
        (local, "trust_service"),
    ):
        for cls in type(obj).__mro__:
            namespace = vars(cls)
            if name in namespace:
                value = namespace[name]
                bindings.append((namespace, name, value))
                function = value.fget if type(value) is property else value
                if type(function) is not FunctionType:
                    return None
                functions.append(_function_record(function))
                break
        else:
            return None
    app_fields, local_fields = vars(app), vars(local)
    fields = tuple(
        (container, name, container.get(name))
        for container, names in (
            (
                app_fields,
                (
                    "_local_skill_trust_service_build_lock",
                    "_local_skills_service",
                    "_skills_scope_service",
                    "_local_skills_stack_inputs",
                    "app_config",
                ),
            ),
            (
                local_fields,
                (
                    "_trust_service_factory",
                    "store_dir",
                    "skills_dir",
                    "policy_enforcer",
                ),
            ),
        )
        for name in names
    )
    factory = local_fields.get("_trust_service_factory")
    if type(factory) is not FunctionType:
        return None
    functions.append(_function_record(factory))
    environment = os.environ
    if type(environment) is not os._Environ:
        return None
    environment_data = vars(environment).get("_data")
    if type(environment_data) is not dict:
        return None
    selector = tuple(
        (environment.encodekey(key), environment_data.get(environment.encodekey(key)))
        for key in (
            "TLDW_CONFIG_PATH",
            "HOME",
            "USERPROFILE",
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
        )
    )
    if any(
        type(key) not in (str, bytes) or type(value) not in (str, bytes, type(None))
        for key, value in selector
    ):
        return None
    return (
        tuple(bindings),
        fields,
        tuple(functions),
        type(app),
        type(local),
        environment,
        environment_data,
        selector,
    )


def _skill_publisher_guard_current(source, guard):
    (
        bindings,
        fields,
        functions,
        app_type,
        local_type,
        environment,
        environment_data,
        selector,
    ) = guard
    return bool(
        type(source.app) is app_type
        and type(source.local) is local_type
        and os.environ is environment
        and vars(environment).get("_data") is environment_data
        and all(environment_data.get(key) == value for key, value in selector)
        and all(namespace.get(name) is value for namespace, name, value in bindings)
        and all(container.get(name) is value for container, name, value in fields)
        and all(_function_unchanged(record) for record in functions)
    )


@contextmanager
def native_skill_trust_publication(preparation):
    """Track only this original native producer invocation; never grant execution."""
    from tldw_chatbook.app_skill_wiring import (
        _SkillTrustPublisher,
        _skill_publisher_identity,
        _skill_trust_publisher,
    )

    marker = None
    if (
        type(preparation) is ConsoleReceivedConfigurationPreparation
        and preparation.trust_source is not None
        and _same_trust_slots(preparation.trust_source, preparation.trust_initial)
    ):
        try:
            require_received_configuration_preparation(
                preparation,
                preparation.app,
                preparation.store,
                preparation.creator,
                session_id=preparation.session.id,
                on_loop=False,
            )
        except (AttributeError, TypeError, ValueError, RecoveryRequired):
            pass
        else:
            guard = _skill_publisher_guard(preparation)
            if guard is not None:
                thread_id, task = _skill_publisher_identity()
                marker = _SkillTrustPublisher(preparation, thread_id, task, guard)
    token = _skill_trust_publisher.set(marker)
    try:
        yield marker
    finally:
        if marker is not None:
            marker.active = False
        _skill_trust_publisher.reset(token)


def finish_native_skill_trust_publication(marker, winner):
    """Publish a source-bound receipt after the original local getter returns."""
    from tldw_chatbook.app_skill_wiring import (
        _SkillTrustPublisher,
        _skill_publisher_identity,
    )

    if (
        type(marker) is not _SkillTrustPublisher
        or not marker.active
        or _skill_publisher_identity() != (marker.thread_id, marker.task)
    ):
        return
    preparation = marker.preparation
    source = preparation.trust_source
    initial_app, initial_local = preparation.trust_initial
    expected = marker.candidate if initial_app is None else initial_app
    if (
        initial_local is not None
        or expected is None
        or winner is not expected
        or not _same_trust_slots(source, (expected, winner))
    ):
        return
    try:
        require_received_configuration_preparation(
            preparation,
            preparation.app,
            preparation.store,
            preparation.creator,
            session_id=preparation.session.id,
            on_loop=False,
        )
    except (AttributeError, TypeError, ValueError, RecoveryRequired):
        return
    runtime = preparation.creator._hooks_v2_runtime
    receipt = _NativeSkillTrustPublication(
        source,
        initial_app,
        initial_local,
        expected,
        winner,
        preparation.trust_functions,
        marker.guard,
    )
    # Only plain native identity checks and assignment occur under this fence.
    with runtime._execution_capacity_lock:
        if (
            runtime._app is not source.app
            or runtime._disposed
            or vars(source.app).get("_local_skill_trust_service") is not expected
            or vars(source.local).get("_trust_service") is not winner
            or not _skill_publisher_guard_current(source, marker.guard)
            or not _trust_source_functions_current(preparation)
            or vars(preparation.config).get("_CONFIG_GENERATION")
            != preparation.config_identity[0]
        ):
            return
        vars(source.local)["_console_native_skill_trust_publication"] = receipt


def _original_skill_trust_slots_current(preparation, initial):
    source = preparation.trust_source
    if source is None or type(initial) is not tuple or len(initial) != 2:
        return False
    if _same_trust_slots(source, initial):
        return True
    receipt = vars(source.local).get("_console_native_skill_trust_publication")
    return bool(
        type(receipt) is _NativeSkillTrustPublication
        and receipt.source.app is source.app
        and receipt.source.local is source.local
        and receipt.source.scope is source.scope
        and receipt.initial_app is initial[0]
        and receipt.initial_local is initial[1]
        and _trust_source_functions_current(preparation)
        and _trust_source_records_current(receipt.source, receipt.source_functions)
        and _skill_publisher_guard_current(receipt.source, receipt.guard)
        and receipt.source.source_current()
        and source.source_current()
        and _same_trust_slots(source, (receipt.app_winner, receipt.local_winner))
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
        trust_initial = ()
        trust_functions = ()
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
            trust_initial = (
                vars(trust_source.app).get("_local_skill_trust_service"),
                vars(trust_source.local).get("_trust_service"),
            )
            if (
                type(trust_source.source_current) is not FunctionType
                or type(trust_source.current) is not FunctionType
            ):
                return None
            trust_functions = (
                _function_record(trust_source.source_current),
                _function_record(trust_source.current),
            )
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
            trust_initial,
            trust_functions,
        )
        if trust_source is not None and (
            not _trust_source_functions_current(preparation)
            or not _skill_catalog_sources_current(preparation)
            or not _same_trust_slots(trust_source, trust_initial)
        ):
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
        not _trust_source_functions_current(preparation)
        or not _skill_catalog_sources_current(preparation)
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
        except Exception:  # noqa: BLE001 - preserve optional catalog unavailable semantics after source checks.
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
        or any(
            now is not before
            for now, before in zip(observers, catalog.observers, strict=False)
        )
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

    with native_skill_trust_publication(preparation) as publication:
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
        require_current()
        finish_native_skill_trust_publication(publication, local_winner)
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
        except Exception:  # noqa: BLE001 - preserve optional catalog unavailable semantics after source checks.
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
    from .console_preparation_reads import current_workflow_preparation

    workflow = current_workflow_preparation(creator)
    workflow_member = workflow is not None and workflow.accepts_skill_context(
        prepared, selection
    )
    if (
        type(preparation) is not ConsoleReceivedConfigurationPreparation
        or preparation.app is not app
        or preparation.store is not store
        or preparation.creator is not creator
        or catalog.session_id != session_id
        or catalog.skill_workspace_id != selection.skill_workspace_id
        or not workflow_member
        and (
            record is None
            or record.session_id != session_id
            or record.received_intent is None
            or record.received_intent.selection is not selection
            or record.received_intent.turn_id != catalog.turn_id
            or record.store is not store
        )
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
        DEFAULT_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
        MAX_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
        MIN_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
        coerce_bool_setting,
        coerce_int_setting,
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
        and provider_source.controller_factory_current(
            controller_source.MCPToolProvider
        )
        and provider_source.console_preparation_pipeline_current(preparation_source)
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
    _deferred_skills: _DeferredSkillCapture | None = None,
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
    from tldw_chatbook.DB.base_db import operation_owned_connection
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.MCP.console_snapshot import _capture_console_definition_selection

    from .console_preparation_reads import current_planning_preparation

    planning = current_planning_preparation(creator)
    attachment = _mcp_attachment_source(app, creator)
    require_current()
    if not isinstance(selection, ConsoleTurnCaptureSelection):
        raise TypeError("selection must be ConsoleTurnCaptureSelection")
    unused_trust = None
    if planning is not None:
        require_received_configuration_preparation(
            planning._skill_preparation, app, store, creator, session_id=session_id
        )
        unused_trust = planning._skill_preparation.trust_source
    if _deferred_skills is not None:
        if (
            _prepared_skills is not None
            or type(_deferred_skills) is not _DeferredSkillCapture
            or _deferred_skills.selection is not selection
            or _deferred_skills.preparation.trust_source is None
        ):
            raise RecoveryRequired("console_snapshot_owner_changed")
        require_received_configuration_preparation(
            _deferred_skills.preparation,
            app,
            store,
            creator,
            session_id=session_id,
        )
        unused_trust = _deferred_skills.preparation.trust_source
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
    identity = _configuration_session_identity(session)
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
        _require_mcp_attachment_source(app, creator, attachment)
        if planning is not None:
            require_received_configuration_preparation(
                planning._skill_preparation,
                app,
                store,
                creator,
                session_id=session_id,
                on_loop=False,
            )
        if _deferred_skills is not None:
            require_received_configuration_preparation(
                _deferred_skills.preparation,
                app,
                store,
                creator,
                session_id=session_id,
                on_loop=False,
            )
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
            or _configuration_session_identity(session) != identity
        ):
            raise RecoveryRequired("console_snapshot_owner_changed")
        fresh = _source_references(
            app, store, creator, session, _unused_trust=unused_trust
        )
        if any(
            now is not before for now, before in zip(fresh[:-1], refs[:-1], strict=False)
        ) or any(
            now is not before
            for now, before in zip(fresh.callbacks, refs.callbacks, strict=False)
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
    profile, rules = await run_native(
        lambda: (
            resolve_turn_tool_policy_profile_id(app, session.workspace_id),
            resolve_turn_persona_policy_rules(app, session),
        )
    )
    maximum, mcp_sources, mcp_complete, mcp_pending = {}, (), True, False
    if planning is None and refs.mcp is not None:
        mcp_selection = await _capture_console_definition_selection(
            refs.mcp,
            CONSOLE_MCP_BUILTIN_RAW_NAME_EXCLUSIONS,
            _run_native=run_native,
            profile_id=profile,
            persona_policy_rules=rules,
            _admitted=not _has_managed_console_runtime(app, creator),
        )
        maximum, mcp_sources, mcp_complete, mcp_pending = (
            mcp_selection.definitions,
            mcp_selection.sources,
            mcp_selection.complete,
            mcp_selection.pending,
        )
    current()
    require_current()

    def capture():
        from tldw_chatbook.Library.library_rag_state import library_rag_profile_top_k

        from .console_agent_bridge import console_run_budget
        from .console_chat_controller import capture_project_instruction_authority

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
            tools = (
                _runtime_tool_configuration(raw_config, selection)
                if selection.tool_configuration is None
                else dict(selection.tool_configuration)
            )
            tools["session_ephemeral"] = bool(session.ephemeral)
            if "agent_run_budget_maximum" not in tools:
                tools["agent_run_budget_maximum"] = console_run_budget()
            rag = {} if selection.rag_defaults is None else dict(selection.rag_defaults)
            if planning is None and "top_k" not in rag:
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
                _mcp_definition_sources=mcp_sources,
                _mcp_definition_complete=mcp_complete,
                _mcp_capture_pending=mcp_pending,
                _skill_capture_pending=_deferred_skills is not None,
                _require_current=current,
                _plugin_service=None if planning is not None else refs.plugin,
                **(
                    {
                        "_skill_context_maximum": {},
                        "_captured_character_authority": None,
                        "_captured_prompt_transform_inputs": {},
                    }
                    if planning is not None
                    else {}
                    if _prepared_skills is None and _deferred_skills is None
                    else {
                        "_skill_context_maximum": (
                            _prepared_skills.maximum
                            if _prepared_skills is not None
                            else {"available_skills": (), "blocked_skills": ()}
                        ),
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
    if (
        _prepared_skills is not None
        or _deferred_skills is not None
        or _standard_configuration_references(app, refs)
    ):
        _mcp_capture_witness.set(
            (
                creator,
                app,
                store,
                session,
                _configuration_session_identity(session),
                config.current_config_identity(),
                refs,
                result.mcp_definition_maximum,
                result.mcp_definition_sources,
                result.tool_policy_profile_id,
                result.persona_policy_rules,
                result.mcp_definition_complete,
                result.mcp_capture_pending,
                result.workspace_roots,
                result.change_review_root_aliases,
                result.change_review_skipped_roots,
                result.change_review_capture_pending,
                result.skill_capture_pending,
                _deferred_skills,
                result,
                _prepared_skills,
                _consent_source_identity(refs.consent),
                _mcp_source_identity(refs.mcp),
            )
        )
    else:
        _mcp_capture_witness.set(None)
    return result


def _consent_source_identity(service):
    from .console_ordinary_tree import _function_record

    values = (
        tuple(
            inspect.getattr_static(service, name, None)
            for name in (
                "_registry",
                "_capability_reader",
                "_initialize_root",
                "admit_turn",
            )
        )
        if service is not None
        else ()
    )
    return tuple(
        (value, _function_record(value) if type(value) is FunctionType else None)
        for value in values
    )


def _mcp_source_identity(service):
    """Record selected MCP receivers and callbacks without invoking them."""
    if service is None:
        return ()
    local = inspect.getattr_static(service, "local_service", None)
    permission = inspect.getattr_static(service, "_permission_store", None)
    store = inspect.getattr_static(local, "store", None) if local is not None else None
    policy = (
        inspect.getattr_static(local, "policy_enforcer", None)
        if local is not None
        else None
    )
    values = (
        service,
        local,
        store,
        permission,
        inspect.getattr_static(local, "manifest_provider", None)
        if local is not None
        else None,
        policy,
        inspect.getattr_static(service, "get_kill_switch", None),
        inspect.getattr_static(service, "effective_tool_states", None),
        inspect.getattr_static(local, "get_inventory", None)
        if local is not None
        else None,
        inspect.getattr_static(policy, "require_allowed", None)
        if policy is not None
        else None,
        inspect.getattr_static(permission, "load", None)
        if permission is not None
        else None,
    )
    from .console_ordinary_tree import _function_record

    return (
        tuple(
            (value, _function_record(value) if type(value) is FunctionType else None)
            for value in values
        ),
        inspect.getattr_static(store, "path", None) if store is not None else None,
        inspect.getattr_static(permission, "path", None)
        if permission is not None
        else None,
    )


def _same_mcp_source_entry(original, current):
    from .console_ordinary_tree import _function_unchanged

    value, record = original
    return value is current[0] and (record is None or _function_unchanged(record))


def _same_mcp_source(left, right):
    if not left or not right:
        return left == right
    return left[1:] == right[1:] and all(
        _same_mcp_source_entry(original, current)
        for original, current in zip(left[0], right[0], strict=True)
    )


def _claimed_capture_current(claimed, controller, session, configuration):
    transfer, prompt, authorization = (
        claimed.transfer,
        claimed.prompt,
        claimed.authorization,
    )
    coordinator = controller.prompt_queue_coordinator
    request = prompt.custody_request
    if (
        prompt.mcp_capture_witness is not transfer
        or request is None
        or request.configuration is not transfer.configuration
        or request.session_id != session.id
        or authorization.entry_id != prompt.entry_id
        or not coordinator.authorizes(authorization, session.id)
        or not coordinator.registry.owns_claimed_prompt(session.id, prompt)
    ):
        return False
    expected = dict(transfer.configuration.skill_context_maximum)
    actual = dict(configuration.skill_context_maximum)
    for key in ("plugin_turn_id", "plugin_run_id"):
        expected.pop(key, None)
        actual.pop(key, None)
    return expected == actual


def original_mcp_capture_matches(
    controller, session, configuration, *, pending=False, require_complete=True
):
    """Check only the same original capture; detached fields grant nothing."""
    from tldw_chatbook import config

    state = _mcp_capture_witness.get()
    claimed = state if type(state) is _ClaimedCaptureTransfer else None
    witness = claimed.witness if claimed is not None else state
    if (
        witness is None
        or type(witness) is not tuple
        or configuration.mcp_capture_pending is not pending
    ):
        return False
    if require_complete and not pending and not configuration.mcp_definition_complete:
        return False
    (
        creator,
        app,
        store,
        captured_session,
        identity,
        config_identity,
        refs,
        *selected,
        deferred,
        _original_configuration,
        prepared,
        consent_source,
        mcp_source,
    ) = witness
    if (
        creator is not controller
        or app is not controller.app
        or store is not controller.store
        or captured_session is not session
        or identity != _configuration_session_identity(session)
        or config_identity != config.current_config_identity()
    ):
        return False
    try:
        unused_trust = None
        if claimed is not None:
            if not _claimed_capture_current(
                claimed, controller, session, configuration
            ):
                return False
            transfer = claimed.transfer
            if transfer.preparation is not None:
                source = transfer.preparation.trust_source
                initial = (transfer.app_trust, transfer.local_trust)
                if not (
                    _original_skill_trust_slots_current(transfer.preparation, initial)
                    if transfer.unused_trust
                    else _same_trust_slots(source, initial)
                ):
                    return False
                require_received_configuration_preparation(
                    transfer.preparation,
                    app,
                    store,
                    controller,
                    session_id=session.id,
                )
                unused_trust = source if transfer.unused_trust else None
        elif deferred is not None:
            if (
                type(deferred) is not _DeferredSkillCapture
                or not configuration.skill_capture_pending
            ):
                return False
            require_received_configuration_preparation(
                deferred.preparation,
                app,
                store,
                controller,
                session_id=session.id,
            )
            unused_trust = deferred.preparation.trust_source
        elif prepared is not None:
            record = controller._hooks_v2_runtime._turn_custody.get(
                prepared.catalog.turn_id
            )
            from .console_preparation_reads import current_workflow_preparation

            workflow = current_workflow_preparation(controller)
            if workflow is not None and workflow.accepts_skill_context(
                prepared, workflow._configuration_selection
            ):
                selection = workflow._configuration_selection
            else:
                if record is None or record.received_intent is None:
                    return False
                selection = record.received_intent.selection
            unused_trust = require_prepared_skill_context(
                prepared,
                app,
                store,
                controller,
                session_id=session.id,
                selection=selection,
            )
        current_refs = _source_references(
            app, store, controller, session, _unused_trust=unused_trust
        )
        matches = (
            _same_configuration_references(refs, current_refs)
            and _same_mcp_source(mcp_source, _mcp_source_identity(current_refs.mcp))
            and all(
                _same_mcp_source_entry(a, b)
                for a, b in zip(
                    consent_source,
                    _consent_source_identity(current_refs.consent),
                    strict=True,
                )
            )
            and tuple(selected)
            == (
                configuration.mcp_definition_maximum,
                configuration.mcp_definition_sources,
                configuration.tool_policy_profile_id,
                configuration.persona_policy_rules,
                configuration.mcp_definition_complete,
                pending,
                configuration.workspace_roots,
                configuration.change_review_root_aliases,
                configuration.change_review_skipped_roots,
                configuration.change_review_capture_pending,
                configuration.skill_capture_pending,
            )
        )
        return matches and (
            claimed is None
            or _claimed_capture_current(claimed, controller, session, configuration)
        )
    except (AttributeError, TypeError, ValueError, RecoveryRequired):
        return False


def consume_native_mcp_permission_publication(source):
    """Accept only the original native publisher's exact retained candidate."""
    state = _mcp_capture_witness.get()
    claimed = state if type(state) is _ClaimedCaptureTransfer else None
    witness = claimed.witness if claimed is not None else state
    publication = source.native_permission_publication
    if witness is None or publication is None:
        return
    service, store, candidate, binding, path = publication
    original = witness[-1]
    current = _mcp_source_identity(service)
    from tldw_chatbook.MCP.permission_store import MCPPermissionStore

    if (
        not original
        or not current
        or original[0][3][0] is not None
        or original[0][10][0] is not None
        or current[0][0][0] is not service
        or current[0][2][0] is not store
        or current[0][3][0] is not candidate
        or source.permission is not candidate
        or type(candidate) is not MCPPermissionStore
        or source.permission_path != path
        or current[2] != path
        or original[1] != current[1]
        or original[2] is not None
        or any(
            not _same_mcp_source_entry(a, b)
            for index, (a, b) in enumerate(zip(original[0], current[0], strict=True))
            if index not in (3, 10)
        )
    ):
        raise RuntimeError("Original MCP permission publication changed.")
    from tldw_chatbook.Backup_Recovery import mcp_source_participants

    if mcp_source_participants._BINDINGS.get(candidate) is not binding:
        raise RuntimeError("Original MCP permission binding changed.")
    updated = (*witness[:-1], current)
    if claimed is None:
        _mcp_capture_witness.set(updated)
    else:
        from dataclasses import replace

        _mcp_capture_witness.set(replace(claimed, witness=updated))
    source.native_permission_publication = None


def remember_pending_mcp_capture(controller, session, configuration):
    """Retain the original synchronous pending source outside the snapshot."""
    if not (
        configuration.mcp_capture_pending
        or configuration.change_review_capture_pending
        or configuration.skill_capture_pending
    ):
        return
    from tldw_chatbook import config

    app, store = controller.app, controller.store
    refs = _source_references(app, store, controller, session)
    _mcp_capture_witness.set(
        (
            controller,
            app,
            store,
            session,
            _configuration_session_identity(session),
            config.current_config_identity(),
            refs,
            configuration.mcp_definition_maximum,
            configuration.mcp_definition_sources,
            configuration.tool_policy_profile_id,
            configuration.persona_policy_rules,
            configuration.mcp_definition_complete,
            configuration.mcp_capture_pending,
            configuration.workspace_roots,
            configuration.change_review_root_aliases,
            configuration.change_review_skipped_roots,
            configuration.change_review_capture_pending,
            configuration.skill_capture_pending,
            None,
            configuration,
            None,
            _consent_source_identity(refs.consent),
            _mcp_source_identity(refs.mcp),
        )
    )


def pending_mcp_witness(configuration):
    """Return only the pending witness for these exact detached MCP values."""
    state = _mcp_capture_witness.get()
    transfer = state if type(state) is _QueuedCaptureTransfer else None
    witness = transfer.witness if transfer is not None else state
    if witness is None or not (
        configuration.mcp_capture_pending
        or configuration.change_review_capture_pending
        or configuration.skill_capture_pending
    ):
        return None
    if (
        type(witness) is not tuple
        or transfer is not None
        and transfer.configuration is not configuration
    ):
        return None
    if tuple(witness[7:13]) != (
        configuration.mcp_definition_maximum,
        configuration.mcp_definition_sources,
        configuration.tool_policy_profile_id,
        configuration.persona_policy_rules,
        configuration.mcp_definition_complete,
        configuration.mcp_capture_pending,
    ):
        return None
    if tuple(witness[13:17]) != (
        configuration.workspace_roots,
        configuration.change_review_root_aliases,
        configuration.change_review_skipped_roots,
        configuration.change_review_capture_pending,
    ):
        return None
    if witness[17] != configuration.skill_capture_pending:
        return None
    return transfer if transfer is not None else witness


def pending_skill_capture(configuration):
    """Return the original private preparation, never one from detached fields."""
    state = _mcp_capture_witness.get()
    witness = state.witness if type(state) is _ClaimedCaptureTransfer else state
    if type(witness) is not tuple or not configuration.skill_capture_pending:
        return None
    deferred, original = witness[-5], witness[-4]
    if type(deferred) is not _DeferredSkillCapture:
        return None
    return deferred, original


def original_skill_capture_preparation():
    """Return the preparation carried by the already-validated private witness."""
    state = _mcp_capture_witness.get()
    if type(state) is _ClaimedCaptureTransfer:
        return state.transfer.preparation
    if type(state) is not tuple:
        return None
    deferred, prepared = state[-5], state[-3]
    if type(deferred) is _DeferredSkillCapture:
        return deferred.preparation
    if type(prepared) is ConsolePreparedSkillContext:
        return prepared.catalog.preparation
    return None


@contextmanager
def seal_pending_queue_capture(controller, session, configuration):
    """Seal an original live received result before the queue's own lock."""
    if not (
        configuration.mcp_capture_pending
        or configuration.change_review_capture_pending
        or configuration.skill_capture_pending
    ):
        yield
        return
    witness = pending_mcp_witness(configuration)
    if type(witness) is not tuple or not original_mcp_capture_matches(
        controller,
        session,
        configuration,
        pending=configuration.mcp_capture_pending,
        require_complete=False,
    ):
        raise RuntimeError("Original queued capture source is unavailable; send again.")
    prepared, deferred = witness[-3], witness[-5]
    preparation = (
        prepared.catalog.preparation
        if prepared is not None
        else (deferred.preparation if deferred is not None else None)
    )
    source = preparation.trust_source if preparation is not None else None
    if prepared is not None and (
        source is None
        or dict(prepared.maximum) != dict(configuration.skill_context_maximum)
    ):
        raise RuntimeError("Original queued skill capture changed; send again.")
    if deferred is not None and (
        witness[-4] is not configuration or not configuration.skill_capture_pending
    ):
        raise RuntimeError("Original queued skill preparation changed; send again.")
    unused_trust = (
        deferred is not None or prepared is not None and prepared.trust_refs is None
    )
    if source is not None and unused_trust:
        if not _original_skill_trust_slots_current(
            preparation, preparation.trust_initial
        ):
            raise RuntimeError(
                "Original queued trust publication is unavailable; send again."
            )
        app_trust, local_trust = preparation.trust_initial
    else:
        app_trust = (
            vars(source.app).get("_local_skill_trust_service")
            if source is not None
            else None
        )
        local_trust = (
            vars(source.local).get("_trust_service") if source is not None else None
        )
    sealed = _QueuedCaptureTransfer(
        (*witness[:-3], None, *witness[-2:]),
        configuration,
        preparation,
        app_trust,
        local_trust,
        unused_trust,
    )
    token = _mcp_capture_witness.set(sealed)
    try:
        yield
    finally:
        _mcp_capture_witness.reset(token)


@contextmanager
def use_pending_mcp_witness(witness, *, prompt=None, authorization=None):
    if type(witness) is _QueuedCaptureTransfer:
        if prompt is None or authorization is None:
            raise RuntimeError("Original queued capture claim is unavailable.")
        witness = _ClaimedCaptureTransfer(
            witness, prompt, authorization, witness.witness
        )
    token = _mcp_capture_witness.set(witness)
    try:
        yield
    finally:
        _mcp_capture_witness.reset(token)


async def capture_runtime_configuration_owned(
    controller,
    session_id,
    *,
    require_current,
    selection=None,
    prepared_skills=None,
    deferred_skills=None,
):
    """Use the existing runtime's selected values and physical read observers."""
    store = controller.store
    session = next(row for row in store.sessions() if row.id == session_id)
    runtime = getattr(controller, "_hooks_v2_runtime", None)

    def current():
        require_current()
        if (
            controller._disposed
            or controller._shutdown_requested.is_set()
            or getattr(controller, "_hooks_v2_runtime", None) is not runtime
            or (
                runtime is not None
                and (
                    runtime._disposed
                    or runtime._chat_controller is not controller
                    or runtime._chat_store is not store
                    or session_id in runtime._admission_fenced_sessions
                )
            )
        ):
            raise RecoveryRequired("console_snapshot_owner_changed")

    current()
    if selection is None:
        selection = ConsoleTurnCaptureSelection(
            provider_selection=controller._provider_selection_for_session(session_id),
            presentation_context=controller._presentation_context_for(session_id),
            rag_defaults=None,
            tool_configuration=None,
            skill_workspace_id=store.session_workspace_id(session_id),
            project_bindings_eligible=controller._agent_dispatch_is_eligible(
                session, prefill=store.session_one_shot_prefill(session_id)
            ),
            agent_runtime_enabled=controller._agent_runtime_enabled,
        )
    result = await capture_console_turn_configuration_owned(
        controller.app,
        store,
        session_id,
        selection=selection,
        creator=controller,
        reads=controller._preparation_reads,
        observers=() if runtime is None else (runtime._preparation_reads,),
        require_current=current,
        **({} if prepared_skills is None else {"_prepared_skills": prepared_skills}),
        **({} if deferred_skills is None else {"_deferred_skills": deferred_skills}),
    )
    current()
    if not isinstance(result, ConsoleTurnConfigurationSnapshot):
        raise TypeError("Console capture must return ConsoleTurnConfigurationSnapshot.")
    if result.session_id != session_id:
        raise ValueError("Console capture returned a different session.")
    return result


async def capture_mounted_configuration_owned(
    controller, session_id, owner, provider, *, require_current, _with_current=False
):
    """Prepare a qualified mounted selection; custom producers retain fallback."""
    from tldw_chatbook.UI.Console_Modules import turn_admission

    app, store = controller.app, controller.store
    cold_preparation = None
    if not standard_console_configuration_sources(app, store, controller, session_id=session_id):
        cold_preparation = capture_console_received_configuration_preparation(
            app,
            store,
            controller,
            session_id=session_id,
        )
        if cold_preparation is None or cold_preparation.trust_source is None:
            return None
    # The UI owns the answer to "is this the stock capture seam"; Chat only asks.
    stock = turn_admission.stock_mounted_capture(owner)
    if stock is None:
        return None
    selector, stock_still_current = stock
    selection = selector(session_id, scratch_owner=controller._scratch_spaces)
    if selection is None:
        return None
    adapter = owner._scratch_snapshot_provider
    screen = adapter.args[0]
    runtime_accessor = screen._console_runtime

    def same(left, right):
        return left is right or (
            type(left) is MethodType and type(right) is MethodType
            and left.__self__ is right.__self__ and left.__func__ is right.__func__
        )

    def current():
        require_current()
        if (
            not same(owner._build_console_turn_execution_context, provider)
            or not same(owner._build_console_turn_capture_selection, selector)
            or owner._scratch_snapshot_provider is not adapter
            or not stock_still_current(adapter)
            or adapter.args != (screen,)
            or adapter.keywords
            or getattr(screen, "_session", None) is not owner
            or screen.app_instance is not app
            or not same(screen._console_runtime, runtime_accessor)
            or not turn_admission.policy_producers_current()
        ):
            raise RecoveryRequired("console_snapshot_owner_changed")

    current()
    result = await capture_runtime_configuration_owned(
        controller,
        session_id,
        require_current=current,
        selection=selection,
        deferred_skills=(
            _DeferredSkillCapture(cold_preparation, selection)
            if cold_preparation is not None
            else None
        ),
    )
    return (result, current) if _with_current else result


async def capture_received_configuration_owned(
    controller, session_id, *, selection, prepared_skills=None
):
    """Capture a detached receipt selection through the existing domain owner."""
    import sys

    from tldw_chatbook import config
    from tldw_chatbook.MCP.console_snapshot import standard_console_sources

    if selection is None:
        raise RecoveryRequired("console_snapshot_owner_changed")
    if not isinstance(selection, ConsoleTurnCaptureSelection):
        raise TypeError("selection must be ConsoleTurnCaptureSelection")
    app, store = controller.app, controller.store
    session = next((row for row in store.sessions() if row.id == session_id), None)
    if session is None:
        raise KeyError(session_id)
    service = getattr(app, "unified_mcp_service", None)
    if service is not None and not standard_console_sources(service):
        raise RecoveryRequired("console_snapshot_owner_changed")
    app_config = getattr(app, "app_config", None)
    scratch, provider_config = controller._scratch_spaces, controller._provider_config
    identity = _configuration_session_identity(session)
    config_identity = config.current_config_identity()

    def current():
        if (
            controller.app is not app
            or controller.store is not store
            or next((row for row in store.sessions() if row.id == session_id), None)
            is not session
            or _configuration_session_identity(session) != identity
            or controller._scratch_spaces is not scratch
            or controller._provider_config is not provider_config
            or getattr(app, "app_config", None) is not app_config
            or getattr(app, "unified_mcp_service", None) is not service
            or sys.modules.get("tldw_chatbook.config") is not config
            or config.current_config_identity() != config_identity
        ):
            raise RecoveryRequired("console_snapshot_owner_changed")
        if prepared_skills is not None:
            require_prepared_skill_context(
                prepared_skills, app, store, controller,
                session_id=session_id, selection=selection,
            )

    current()
    if prepared_skills is None and not standard_console_configuration_sources(
        app, store, controller, session_id=session_id
    ):
        raise RecoveryRequired("console_snapshot_owner_changed")
    return await capture_runtime_configuration_owned(
        controller, session_id, selection=selection,
        prepared_skills=prepared_skills, require_current=current,
    )


async def _controller_capture_turn_configuration_snapshot(
    self,
    session_id: str,
    *,
    context_provider: Callable[..., ConsoleTurnConfigurationSnapshot] | None = None,
    selection: Any = None,
    _prepared_skills: Any = None,
) -> ConsoleTurnConfigurationSnapshot:
    """Prepare fresh native MCP inputs before composing on the owning loop."""
    import sys
    from types import MethodType

    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.MCP.console_snapshot import (
        _capture_console_definition_selection,
        standard_console_sources,
    )

    from . import console_chat_controller as _source_controller
    if selection is not None or _prepared_skills is not None:
        from .console_configuration_preparation import (
            capture_received_configuration_owned,
        )
        return await capture_received_configuration_owned(
            self, session_id, selection=selection, prepared_skills=_prepared_skills
        )
    original_provider = self._turn_context_provider
    provider = (
        context_provider if context_provider is not None else original_provider
    )
    def synchronous_capture() -> ConsoleTurnConfigurationSnapshot:
        if context_provider is None:
            return self.resolve_turn_configuration_snapshot(session_id)
        context = provider(session_id)
        if not isinstance(context, _source_controller.ConsoleTurnConfigurationSnapshot):
            raise TypeError(
                "Console turn-context provider must return ConsoleTurnConfigurationSnapshot."
            )
        if context.session_id != session_id:
            raise ValueError(
                "Console turn-context provider returned a different session."
            )
        return context
    screen_owner = getattr(provider, "__self__", None)
    standard_screen = False
    if isinstance(provider, MethodType):
        # UI compatibility is needed only for an explicitly supplied bound
        # provider. Default app-owned capture has no Console UI dependency.
        from tldw_chatbook.UI.Console_Modules.turn_admission import (
            is_stock_session_provider,
        )
        standard_screen = is_stock_session_provider(provider)
    if provider is not None and not standard_screen:
        return synchronous_capture()
    app = self.app
    attachment = _mcp_attachment_source(app, self)
    app_config = getattr(app, "app_config", None)
    service = getattr(app, "unified_mcp_service", None)
    stock_mcp = standard_console_sources(service)
    if not stock_mcp and not standard_screen:
        return synchronous_capture()
    store = self.store
    session = next(item for item in store.sessions() if item.id == session_id)
    runtime = getattr(self, "_hooks_v2_runtime", None)
    scratch = self._scratch_spaces
    provider_config = self._provider_config
    configuration_identity = config.current_config_identity()
    revision = store.session_settings_revision(session_id)
    workspace_id = store.session_workspace_id(session_id)
    selection = self._provider_selection_for_session(session_id)
    if standard_screen and (
        screen_owner.app_instance is not app
        or screen_owner._ensure_console_chat_store() is not store
    ):
        raise RecoveryRequired("console_snapshot_owner_changed")
    screen_callbacks = (
        tuple(
            getattr(screen_owner, name, None)
            for name in (
                "_current_chat_store_accessor",
                "_chat_store_accessor",
                "_provider_readiness_app_config_fn",
                "_build_provider_selection_fn",
                "_scratch_snapshot_provider",
                "_rag_source_types_accessor",
                "_rag_top_k_accessor",
            )
        )
        if standard_screen
        else ()
    )
    def same_callback(current: Any, original: Any) -> bool:
        return current is original or (
            type(current) is MethodType
            and type(original) is MethodType
            and current.__func__ is original.__func__
            and current.__self__ is original.__self__
        )
    def require_resident_current() -> None:
        _require_mcp_attachment_source(app, self, attachment)
        current_session = next(
            (item for item in store.sessions() if item.id == session_id), None
        )
        if (
            self._disposed
            or self._shutdown_requested.is_set()
            or getattr(self, "_hooks_v2_runtime", None) is not runtime
            or (runtime is not None and (runtime._disposed
                or runtime._chat_controller is not self
                or runtime._chat_store is not store
                or session_id in runtime._admission_fenced_sessions))
            or self.app is not app
            or self.store is not store
            or current_session is not session
            or self._turn_context_provider is not original_provider
            or getattr(app, "app_config", None) is not app_config
            or self._scratch_spaces is not scratch
            or self._provider_config is not provider_config
            or getattr(app, "unified_mcp_service", None) is not service
            or sys.modules.get("tldw_chatbook.config") is not config
            or config.current_config_identity() != configuration_identity
            or store.session_settings_revision(session_id) != revision
            or store.session_workspace_id(session_id) != workspace_id
        ):
            raise RecoveryRequired("console_snapshot_owner_changed")
        if standard_screen and (
            screen_owner.app_instance is not app
            or not all(
                same_callback(getattr(screen_owner, name, None), original)
                for name, original in zip(
                    (
                        "_current_chat_store_accessor",
                        "_chat_store_accessor",
                        "_provider_readiness_app_config_fn",
                        "_build_provider_selection_fn",
                        "_scratch_snapshot_provider",
                        "_rag_source_types_accessor",
                        "_rag_top_k_accessor",
                    ),
                    screen_callbacks,
                )
            )
        ):
            raise RecoveryRequired("console_snapshot_owner_changed")
    def require_current() -> None:
        require_resident_current()
        if (self._provider_selection_for_session(session_id) != selection
            or (standard_screen and screen_owner._ensure_console_chat_store() is not store)
            or (stock_mcp and not standard_console_sources(service))):
            raise RecoveryRequired("console_snapshot_owner_changed")

    if standard_screen:
        from .console_configuration_preparation import (
            capture_mounted_configuration_owned,
        )
        captured = await capture_mounted_configuration_owned(
            self, session_id, screen_owner, provider, require_current=require_current
        ) if stock_mcp else None
        if captured is not None:
            return captured
    else:
        from .console_configuration_preparation import (
            capture_runtime_configuration_owned,
            standard_console_configuration_sources,
        )
        if standard_console_configuration_sources(
            app, store, self, session_id=session_id
        ):
            return await capture_runtime_configuration_owned(
                self, session_id, require_current=require_current
            )
    if standard_screen:
        from tldw_chatbook.Backup_Recovery.participants import run_finite_local_worker
        from tldw_chatbook.UI.Console_Modules.turn_admission import (
            applied, authority_inputs, capture_authority,
        )
        from .console_preparation_reads import run_preparation_read

        inputs = authority_inputs(screen_owner, session_id)
        require_current()
        authority = await run_preparation_read(
            lambda: run_finite_local_worker(capture_authority, screen_owner, inputs),
            creator=self,
            session_id=session_id,
            reads=self._preparation_reads,
            observers=() if runtime is None else (runtime._preparation_reads,),
            require_current=require_resident_current,
        )
        require_current()
        if authority_inputs(screen_owner, session_id).key != authority.key:
            raise RecoveryRequired("console_snapshot_owner_changed")
        with applied(authority):
            context = synchronous_capture()
        require_current()
        return context
    profile = _source_controller.resolve_turn_tool_policy_profile_id(app, workspace_id)
    rules = _source_controller.resolve_turn_persona_policy_rules(app, session)
    mcp_selection = await _capture_console_definition_selection(
        service,
        _source_controller.CONSOLE_MCP_BUILTIN_RAW_NAME_EXCLUSIONS,
        profile_id=profile,
        persona_policy_rules=rules,
        _admitted=not _has_managed_console_runtime(app, self),
    )
    maximum = mcp_selection.definitions
    require_current()
    context = (
        provider(session_id, mcp_definition_maximum=maximum)
        if standard_screen
        else self.resolve_runtime_turn_configuration_snapshot(
            session_id, mcp_definition_maximum=maximum
        )
    )
    require_current()
    if not isinstance(context, _source_controller.ConsoleTurnConfigurationSnapshot):
        raise TypeError(
            "Console turn-context provider must return ConsoleTurnConfigurationSnapshot."
        )
    if context.session_id != session_id:
        raise ValueError(
            "Console turn-context provider returned a different session."
        )
    if context.tool_policy_profile_id == profile and dict(
        context.mcp_definition_maximum
    ) == dict(maximum):
        from dataclasses import replace

        context = replace(
            context,
            mcp_definition_sources=mcp_selection.sources,
            mcp_definition_complete=mcp_selection.complete,
            mcp_capture_pending=mcp_selection.pending,
        )
    return context
