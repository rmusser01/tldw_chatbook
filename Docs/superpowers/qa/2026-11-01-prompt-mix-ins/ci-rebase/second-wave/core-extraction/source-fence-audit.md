# Core extraction source-fence audit

Baseline ba1cafc55dd5ac9b6b1f066c936206e652ed6b65. This records the concrete stock predicates audited, not a claim that every future dependency is known. None of the 16 moved execution/settings/fork methods is captured by these preparation stock tables. The execution/settings delegates are invoked at their existing execution/persistence sites, not used as stock qualifications for FULL preparation skipping. Existing source readers, guards and revision checks remain in their old owners. Pure banner/trace exports bind the actual moved function/class objects, not wrappers hiding their implementation. Live host module lookup preserves the original callback dependencies at use time.

Exact moved-name matches in preparation owners: `{"console_configuration_preparation.py": [], "console_attention_preparation.py": [], "console_turn_preparation.py": [], "console_received_dispatch.py": []}`.

Actual current stock table and configuration predicate source:

```python
_CONSOLE_TOOL_COMPOSITION_METHODS = tuple(
    (name, method, method.__code__, method.__globals__, method.__defaults__, method.__kwdefaults__,
     tuple((method.__kwdefaults__ or {}).items()), method.__closure__)
    for name in ("_compose_mcp_provider", "_compose_local_provider", "_publish_mcp_inspector_counts")
    for method in (vars(ConsoleChatController)[name],)
)

_CONTEXT_DISPLAY_CALLBACKS = tuple(
    (name, function, function.__code__, function.__defaults__, function.__kwdefaults__,
     isinstance(vars(ConsoleChatController)[name], staticmethod))
    for name in ("context_control_inputs", "_global_context_policy_overrides",
                 "_durable_context_snapshots", "_applicable_branch_memory_state",
                 "_held_send_echo_id")
    for function in (getattr(ConsoleChatController, name),)
)

_CONSOLE_PENDING_FACTS_READERS = tuple(
    (
        name,
        method,
        method.__code__,
        method.__globals__,
        method.__defaults__,
        method.__kwdefaults__,
        tuple((method.__kwdefaults__ or {}).items()),
        method.__closure__,
        tuple((cell, cell.cell_contents) for cell in method.__closure__ or ()),
    )
    for name in (
        "pending_round_count",
        "pending_round_kinds",
        "has_pending_approval_round",
    )
    for method in (getattr(ConsoleChatController, name),)
)

_CONSOLE_PREPARING_CONTROLLER_READERS = tuple(
    (name, method, method.__code__, method.__defaults__, method.__kwdefaults__,
     tuple((method.__kwdefaults__ or {}).items()))
    for name in ('_provider_selection', 'in_flight_run_count', '_live_busy_session_ids', 'activity_for')
    for method in (getattr(ConsoleChatController, name),)
)

_CONSOLE_CONTEXT_PRESENTATION_READERS = tuple((name, function, function.__code__) for name in ("context_control_presentation_inputs", "_held_send_echo_id") for function in (getattr(ConsoleChatController, name),))

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
    from tldw_chatbook.DB.VisualIdentity_DB import VisualIdentityRepository
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
    from tldw_chatbook.Skills_Interop.skills_scope_service import SkillsScopeService
    from tldw_chatbook.Workspaces.change_review_consent import (
        ChangeReviewConsentService,
        _default_capability_reader,
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
```
