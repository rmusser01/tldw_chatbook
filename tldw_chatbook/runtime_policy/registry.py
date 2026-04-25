from __future__ import annotations

from dataclasses import dataclass

from .types import (
    ActionKind,
    AuthorityOwner,
    CapabilityEntry,
    OfflinePolicy,
    PolicyDeniedError,
    RequiredSource,
)


def _entry(
    action_id: str,
    *,
    capability_id: str,
    domain_id: str,
    action_kind: ActionKind,
    required_source: RequiredSource,
    authority_owner: AuthorityOwner,
    offline_policy: OfflinePolicy = "available",
    enabled: bool = True,
    default_deny_reason: str = "authority_denied",
    display_name: str | None = None,
) -> CapabilityEntry:
    return CapabilityEntry(
        action_id=action_id,
        capability_id=capability_id,
        domain_id=domain_id,
        action_kind=action_kind,
        required_source=required_source,
        authority_owner=authority_owner,
        offline_policy=offline_policy,
        enabled=enabled,
        default_deny_reason=default_deny_reason,
        display_name=display_name,
    )


@dataclass(frozen=True, slots=True)
class CapabilitySourceSeed:
    action_id_suffix: str
    required_source: RequiredSource
    authority_owner: AuthorityOwner
    offline_policy: OfflinePolicy = "available"


@dataclass(frozen=True, slots=True)
class CapabilityActionSeed:
    action_id_suffix: str
    action_kind: ActionKind
    enabled: bool = True
    default_deny_reason: str = "authority_denied"
    display_name: str | None = None


@dataclass(frozen=True, slots=True)
class CapabilityResourceSeed:
    action_namespace: str
    actions: tuple[CapabilityActionSeed, ...]
    domain_id: str | None = None
    sources: tuple[CapabilitySourceSeed, ...] = ()


@dataclass(frozen=True, slots=True)
class CapabilitySeed:
    capability_id: str
    display_name: str
    domain_id: str
    sources: tuple[CapabilitySourceSeed, ...]
    resources: tuple[CapabilityResourceSeed, ...]


def _action(
    action_id_suffix: str,
    action_kind: ActionKind,
    *,
    display_name: str | None = None,
) -> CapabilityActionSeed:
    return CapabilityActionSeed(
        action_id_suffix=action_id_suffix,
        action_kind=action_kind,
        display_name=display_name,
    )


def _resource(
    action_namespace: str,
    *,
    actions: tuple[CapabilityActionSeed, ...],
    domain_id: str | None = None,
    sources: tuple[CapabilitySourceSeed, ...] = (),
) -> CapabilityResourceSeed:
    return CapabilityResourceSeed(
        action_namespace=action_namespace,
        actions=actions,
        domain_id=domain_id,
        sources=sources,
    )


def _capability(
    capability_id: str,
    display_name: str,
    domain_id: str,
    *,
    sources: tuple[CapabilitySourceSeed, ...],
    resources: tuple[CapabilityResourceSeed, ...],
) -> CapabilitySeed:
    return CapabilitySeed(
        capability_id=capability_id,
        display_name=display_name,
        domain_id=domain_id,
        sources=sources,
        resources=resources,
    )


def _combine_action_sets(*action_sets: tuple[CapabilityActionSeed, ...]) -> tuple[CapabilityActionSeed, ...]:
    combined_actions: list[CapabilityActionSeed] = []
    seen_suffixes: set[str] = set()
    for action_set in action_sets:
        for action in action_set:
            if action.action_id_suffix in seen_suffixes:
                continue
            combined_actions.append(action)
            seen_suffixes.add(action.action_id_suffix)
    return tuple(combined_actions)


LIST = _action("list", "browse")
DETAIL = _action("detail", "detail")
PREVIEW = _action("preview", "detail")
CREATE = _action("create", "create")
UPDATE = _action("update", "update")
DELETE = _action("delete", "delete")
CANCEL = _action("cancel", "delete")
LAUNCH = _action("launch", "launch")
OBSERVE = _action("observe", "observe")
CONFIGURE = _action("configure", "update")
TRIGGER = _action("trigger", "launch")
PROCESS = _action("process", "launch")
IMPORT = _action("import", "launch")
EXPORT = _action("export", "launch")
APPROVE = _action("approve", "update")
INSPECT = _action("inspect", "detail")
REVOKE = _action("revoke", "delete")
CAPTURE = _action("capture", "launch")
STATUS = _action("status", "detail")
STRUCTURE = _action("structure", "detail")
RESTORE = _action("restore", "update")
REORDER = _action("reorder", "update")
BULK_UPDATE = _action("bulk_update", "update")
ARCHIVE = _action("archive", "create")
SUMMARIZE = _action("summarize", "launch")
TTS = _action("tts", "launch")
REATTACH = _action("reattach", "update")

CRUD_ACTIONS = (LIST, DETAIL, CREATE, UPDATE, DELETE)
DISCOVER_TRIGGER_OBSERVE_ACTIONS = (LIST, LAUNCH, OBSERVE)
DISCOVER_CONFIGURE_TRIGGER_OBSERVE_ACTIONS = (LIST, CONFIGURE, LAUNCH, OBSERVE)

LOCAL_SOURCE = CapabilitySourceSeed("local", "local", "local")
SERVER_SOURCE = CapabilitySourceSeed("server", "server", "server")
WORKSPACE_SOURCE = CapabilitySourceSeed("workspace", "server", "shared")
REMOTE_ONLY_SOURCE = CapabilitySourceSeed("server", "server", "server", "unavailable")

SEPARATED_SOURCES = (LOCAL_SOURCE, SERVER_SOURCE)
LOCAL_ONLY_SOURCES = (LOCAL_SOURCE,)
REMOTE_ONLY_SOURCES = (REMOTE_ONLY_SOURCE,)

FULL_AUDITED_CAPABILITY_IDS = frozenset(
    {
        "chat",
        "characters_personas_ccp",
        "notes_workspaces",
        "media_reading_ingestion_sources",
        "prompts_chatbooks",
        "study_core",
        "study_packs",
        "study_suggestions",
        "collections_reading_list",
        "collections_outputs_templates_artifacts",
        "watchlists",
        "writing_suite",
        "research_sessions_runs",
        "research_search_provider_surfaces",
        "client_notifications",
        "server_reminders_notification_feeds",
        "workflows",
        "scheduler_workflows",
        "chat_workflows",
        "local_mcp_runtime",
        "remote_mcp_control_plane_governance",
        "sharing",
        "web_clipper",
        "evaluations",
        "rag_embeddings_chunking_admin",
        "cross_cutting_runtime_policy",
    }
)

AUDITED_CAPABILITY_SEEDS = (
    _capability(
        "chat",
        "Chat",
        "chat",
        sources=SEPARATED_SOURCES,
        resources=(
            _resource(
                "chat",
                actions=_combine_action_sets(CRUD_ACTIONS, (LAUNCH,)),
            ),
        ),
    ),
    _capability(
        "characters_personas_ccp",
        "Characters / Personas / CCP",
        "characters",
        sources=SEPARATED_SOURCES,
        resources=(
            _resource("character.persona", actions=CRUD_ACTIONS),
            _resource(
                "character.sessions",
                actions=_combine_action_sets(CRUD_ACTIONS, (LAUNCH, EXPORT, OBSERVE, RESTORE)),
            ),
            _resource(
                "chat.dictionaries",
                actions=_combine_action_sets(CRUD_ACTIONS, (PROCESS, IMPORT, EXPORT)),
                domain_id="characters",
                sources=(SERVER_SOURCE,),
            ),
            _resource(
                "chat.dictionary.entries",
                actions=(LIST, CREATE, UPDATE, DELETE, REORDER),
                domain_id="characters",
                sources=(SERVER_SOURCE,),
            ),
            _resource(
                "chat.dictionary.activity",
                actions=(LIST,),
                domain_id="characters",
                sources=(SERVER_SOURCE,),
            ),
            _resource(
                "chat.dictionary.versions",
                actions=(LIST, DETAIL, RESTORE),
                domain_id="characters",
                sources=(SERVER_SOURCE,),
            ),
            _resource(
                "chat.dictionary.statistics",
                actions=(DETAIL,),
                domain_id="characters",
                sources=(SERVER_SOURCE,),
            ),
        ),
    ),
    _capability(
        "notes_workspaces",
        "Notes / Workspaces",
        "notes",
        sources=SEPARATED_SOURCES,
        resources=(
            _resource(
                "notes",
                actions=CRUD_ACTIONS,
                sources=(LOCAL_SOURCE, SERVER_SOURCE, WORKSPACE_SOURCE),
            ),
            _resource("notes.graph", actions=(LIST, DETAIL, CREATE, DELETE), sources=(SERVER_SOURCE,)),
            _resource("notes.workspace", actions=CRUD_ACTIONS),
        ),
    ),
    _capability(
        "media_reading_ingestion_sources",
        "Media / Reading / Ingestion Sources",
        "media",
        sources=SEPARATED_SOURCES,
        resources=(
            _resource(
                "media.reading",
                actions=_combine_action_sets(CRUD_ACTIONS, (BULK_UPDATE, ARCHIVE, SUMMARIZE, IMPORT, EXPORT, TTS)),
            ),
            _resource("media.reading_import_jobs", actions=(LIST, DETAIL)),
            _resource("media.reading.saved_searches", actions=CRUD_ACTIONS),
            _resource("media.reading.note_links", actions=(LIST, CREATE, DELETE)),
            _resource("media.reading_progress", actions=(DETAIL, UPDATE)),
            _resource("media.ingestion_sources", actions=CRUD_ACTIONS),
            _resource("media.ingestion_source_items", actions=(REATTACH,)),
            _resource("media.ingestion_jobs", actions=(LIST, DETAIL, LAUNCH, OBSERVE, CANCEL)),
        ),
    ),
    _capability(
        "prompts_chatbooks",
        "Prompts / Chatbooks",
        "prompts",
        sources=SEPARATED_SOURCES,
        resources=(
            _resource(
                "prompts",
                actions=(LIST, PREVIEW, CREATE, UPDATE, DELETE),
                domain_id="prompts",
            ),
            _resource(
                "prompts.versions",
                actions=(LIST, RESTORE),
                domain_id="prompts",
                sources=(SERVER_SOURCE,),
            ),
            _resource(
                "chatbooks",
                actions=_combine_action_sets(CRUD_ACTIONS, (IMPORT, EXPORT)),
                domain_id="chatbooks",
            ),
        ),
    ),
    _capability(
        "study_core",
        "Study Core",
        "study",
        sources=SEPARATED_SOURCES,
        resources=(
            _resource("study.deck", actions=CRUD_ACTIONS, domain_id="study"),
            _resource("quiz", actions=CRUD_ACTIONS, domain_id="quiz"),
            _resource("quiz.question", actions=(LIST, DETAIL), domain_id="quiz"),
            _resource("quiz.attempt", actions=(CREATE, OBSERVE), domain_id="quiz"),
            _resource("study.guides", actions=(LAUNCH, OBSERVE), domain_id="study"),
        ),
    ),
    _capability(
        "study_packs",
        "Study Packs",
        "study_packs",
        sources=REMOTE_ONLY_SOURCES,
        resources=(
            _resource("study.packs.jobs", actions=DISCOVER_TRIGGER_OBSERVE_ACTIONS),
        ),
    ),
    _capability(
        "study_suggestions",
        "Study Suggestions",
        "study_suggestions",
        sources=REMOTE_ONLY_SOURCES,
        resources=(
            _resource("study.suggestions", actions=DISCOVER_CONFIGURE_TRIGGER_OBSERVE_ACTIONS),
        ),
    ),
    _capability(
        "collections_reading_list",
        "Collections: Reading List / Read-it-later",
        "collections_reading",
        sources=SEPARATED_SOURCES,
        resources=(
            _resource("collections.reading_list", actions=CRUD_ACTIONS),
        ),
    ),
    _capability(
        "collections_outputs_templates_artifacts",
        "Collections: Outputs / Templates / Artifacts",
        "outputs",
        sources=SEPARATED_SOURCES,
        resources=(
            _resource("outputs.templates", actions=CRUD_ACTIONS),
            _resource("outputs.artifacts", actions=CRUD_ACTIONS),
            _resource("outputs.render_jobs", actions=(LIST, DETAIL, LAUNCH, OBSERVE)),
        ),
    ),
    _capability(
        "watchlists",
        "Watchlists",
        "watchlists",
        sources=SEPARATED_SOURCES,
        resources=(
            _resource("watchlists", actions=CRUD_ACTIONS),
            _resource("watchlists.alert_rules", actions=CRUD_ACTIONS),
            _resource("watchlists.runs", actions=(LIST, DETAIL, LAUNCH, OBSERVE)),
        ),
    ),
    _capability(
        "writing_suite",
        "Writing Suite",
        "writing",
        sources=SEPARATED_SOURCES,
        resources=(
            _resource("writing.projects", actions=_combine_action_sets(CRUD_ACTIONS, (STRUCTURE,))),
            _resource("writing.manuscripts", actions=CRUD_ACTIONS),
            _resource("writing.chapters", actions=CRUD_ACTIONS),
            _resource("writing.scenes", actions=CRUD_ACTIONS),
            _resource("writing.versions", actions=(LIST, DETAIL, CREATE, RESTORE)),
            _resource("writing.trash", actions=(LIST, RESTORE)),
            _resource("writing.outline", actions=(REORDER,)),
        ),
    ),
    _capability(
        "research_sessions_runs",
        "Research Sessions / Runs",
        "research",
        sources=SEPARATED_SOURCES,
        resources=(
            _resource("research.sessions", actions=CRUD_ACTIONS),
            _resource("research.runs", actions=_combine_action_sets(CRUD_ACTIONS, (LAUNCH, OBSERVE))),
        ),
    ),
    _capability(
        "research_search_provider_surfaces",
        "Research Search / Provider Surfaces",
        "research_search",
        sources=SEPARATED_SOURCES,
        resources=(
            _resource(
                "research.search.providers",
                actions=DISCOVER_CONFIGURE_TRIGGER_OBSERVE_ACTIONS,
            ),
        ),
    ),
    _capability(
        "client_notifications",
        "Client Notifications",
        "notifications",
        sources=LOCAL_ONLY_SOURCES,
        resources=(
            _resource("notifications.queue", actions=(LIST, UPDATE, OBSERVE)),
            _resource("notifications.settings", actions=(LIST, UPDATE)),
            _resource("notifications.dispatch", actions=(LAUNCH,)),
        ),
    ),
    _capability(
        "server_reminders_notification_feeds",
        "Server Reminders / Notification Feeds",
        "notifications_server",
        sources=REMOTE_ONLY_SOURCES,
        resources=(
            _resource("notifications.reminders", actions=DISCOVER_CONFIGURE_TRIGGER_OBSERVE_ACTIONS),
            _resource("notifications.feed", actions=(LIST, UPDATE, OBSERVE)),
        ),
    ),
    _capability(
        "workflows",
        "Workflows",
        "workflows",
        sources=REMOTE_ONLY_SOURCES,
        resources=(
            _resource("workflows", actions=DISCOVER_TRIGGER_OBSERVE_ACTIONS),
        ),
    ),
    _capability(
        "scheduler_workflows",
        "Scheduler Workflows",
        "scheduler",
        sources=REMOTE_ONLY_SOURCES,
        resources=(
            _resource("scheduler.workflows", actions=DISCOVER_CONFIGURE_TRIGGER_OBSERVE_ACTIONS),
        ),
    ),
    _capability(
        "chat_workflows",
        "Chat Workflows",
        "chat_workflows",
        sources=REMOTE_ONLY_SOURCES,
        resources=(
            _resource("chat.workflows", actions=DISCOVER_TRIGGER_OBSERVE_ACTIONS),
        ),
    ),
    _capability(
        "local_mcp_runtime",
        "Local MCP Runtime",
        "mcp_runtime",
        sources=LOCAL_ONLY_SOURCES,
        resources=(
            _resource("mcp.runtime", actions=(LIST, CONFIGURE, LAUNCH, TRIGGER, OBSERVE)),
        ),
    ),
    _capability(
        "remote_mcp_control_plane_governance",
        "Remote MCP Control Plane / Governance",
        "mcp_governance",
        sources=REMOTE_ONLY_SOURCES,
        resources=(
            _resource("mcp.governance", actions=(LIST, CONFIGURE, LAUNCH, APPROVE, OBSERVE)),
        ),
    ),
    _capability(
        "sharing",
        "Sharing",
        "sharing",
        sources=REMOTE_ONLY_SOURCES,
        resources=(
            _resource("sharing.links", actions=(LIST, CREATE, LAUNCH, INSPECT, REVOKE, OBSERVE)),
            _resource("sharing.permissions", actions=(CONFIGURE,)),
        ),
    ),
    _capability(
        "web_clipper",
        "Web Clipper",
        "web_clipper",
        sources=REMOTE_ONLY_SOURCES,
        resources=(
            _resource("web_clipper", actions=(LIST, CAPTURE, OBSERVE, STATUS)),
        ),
    ),
    _capability(
        "evaluations",
        "Evaluations",
        "evaluations",
        sources=SEPARATED_SOURCES,
        resources=(
            _resource("evaluations.dataset", actions=CRUD_ACTIONS),
            _resource("evaluations.run", actions=_combine_action_sets(CRUD_ACTIONS, (LAUNCH, OBSERVE))),
            _resource(
                "evaluations.rag_pipeline",
                actions=_combine_action_sets(CRUD_ACTIONS, (LAUNCH,)),
                sources=(SERVER_SOURCE,),
            ),
        ),
    ),
    _capability(
        "rag_embeddings_chunking_admin",
        "RAG / Embeddings / Chunking Admin",
        "rag",
        sources=SEPARATED_SOURCES,
        resources=(
            _resource("rag.admin", actions=DISCOVER_CONFIGURE_TRIGGER_OBSERVE_ACTIONS),
            _resource("rag.template", actions=(LIST, DETAIL, CREATE, UPDATE, DELETE)),
        ),
    ),
    _capability(
        "cross_cutting_runtime_policy",
        "Cross-cutting Runtime Policy",
        "runtime_policy",
        sources=SEPARATED_SOURCES,
        resources=(
            _resource("runtime.policy", actions=DISCOVER_CONFIGURE_TRIGGER_OBSERVE_ACTIONS),
        ),
    ),
)


def _iter_capability_entries(capability_seed: CapabilitySeed):
    for resource in capability_seed.resources:
        resource_domain_id = resource.domain_id or capability_seed.domain_id
        resource_sources = resource.sources or capability_seed.sources
        for action in resource.actions:
            for source in resource_sources:
                action_id = (
                    f"{resource.action_namespace}.{action.action_id_suffix}.{source.action_id_suffix}"
                )
                yield _entry(
                    action_id,
                    capability_id=capability_seed.capability_id,
                    domain_id=resource_domain_id,
                    action_kind=action.action_kind,
                    required_source=source.required_source,
                    authority_owner=source.authority_owner,
                    offline_policy=source.offline_policy,
                    enabled=action.enabled,
                    default_deny_reason=action.default_deny_reason,
                    display_name=action.display_name,
                )


def _build_capability_registry(
    capability_seeds: tuple[CapabilitySeed, ...],
) -> tuple[tuple[CapabilityEntry, ...], dict[str, CapabilityEntry]]:
    capability_rows: list[CapabilityEntry] = []
    capability_registry: dict[str, CapabilityEntry] = {}
    duplicate_action_ids: set[str] = set()

    for capability_seed in capability_seeds:
        for entry in _iter_capability_entries(capability_seed):
            capability_rows.append(entry)
            if entry.action_id in capability_registry:
                duplicate_action_ids.add(entry.action_id)
            capability_registry[entry.action_id] = entry

    if duplicate_action_ids:
        raise ValueError(
            "runtime policy registry has duplicate action_ids: "
            f"{sorted(duplicate_action_ids)}"
        )

    return tuple(capability_rows), capability_registry


_CAPABILITY_ROWS, CAPABILITY_REGISTRY = _build_capability_registry(AUDITED_CAPABILITY_SEEDS)

PHASE_ONE_REQUIRED_ACTION_IDS = frozenset(
    {
        "notes.create.local",
        "notes.create.server",
        "notes.create.workspace",
        "media.ingestion_sources.list.server",
        "study.deck.create.server",
        "quiz.create.server",
        "rag.template.create.local",
        "character.persona.list.server",
        "workflows.launch.server",
    }
)


def validate_registry_completeness() -> None:
    seeded_capability_ids = {seed.capability_id for seed in AUDITED_CAPABILITY_SEEDS}
    if seeded_capability_ids != FULL_AUDITED_CAPABILITY_IDS:
        raise ValueError(
            "runtime policy audited capability seeds do not match the full audited set: "
            f"{sorted(FULL_AUDITED_CAPABILITY_IDS.difference(seeded_capability_ids))}"
        )

    registry_capability_ids = {entry.capability_id for entry in CAPABILITY_REGISTRY.values()}
    if registry_capability_ids != FULL_AUDITED_CAPABILITY_IDS:
        raise ValueError(
            "runtime policy registry capability_ids do not expose the full audited set: "
            f"{sorted(FULL_AUDITED_CAPABILITY_IDS.difference(registry_capability_ids))}"
        )

    missing = PHASE_ONE_REQUIRED_ACTION_IDS.difference(CAPABILITY_REGISTRY)
    if missing:
        raise ValueError(f"runtime policy registry is missing required rows: {sorted(missing)}")


def get_capability_entry(action_id: str) -> CapabilityEntry:
    try:
        return CAPABILITY_REGISTRY[action_id]
    except KeyError as exc:
        raise PolicyDeniedError(
            action_id=action_id,
            reason_code="authority_denied",
            user_message=f"Unknown runtime-policy action_id: {action_id}",
            effective_source="unknown",
            authority_owner="unknown",
        ) from exc


validate_registry_completeness()
