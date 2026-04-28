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
PURGE = _action("purge", "delete")
CANCEL = _action("cancel", "delete")
LAUNCH = _action("launch", "launch")
OBSERVE = _action("observe", "observe")
OBSERVE_LIST = _action("list", "observe")
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
SEARCH = _action("search", "browse")
INITIALIZE = _action("initialize", "launch")
SHUTDOWN = _action("shutdown", "delete")
STRUCTURE = _action("structure", "detail")
RESTORE = _action("restore", "update")
REORDER = _action("reorder", "update")
BULK_UPDATE = _action("bulk_update", "update")
ARCHIVE = _action("archive", "create")
SUMMARIZE = _action("summarize", "launch")
TTS = _action("tts", "launch")
REATTACH = _action("reattach", "update")
VALIDATE = _action("validate", "launch")

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
        "companion_personalization",
        "notes_workspaces",
        "media_reading_ingestion_sources",
        "prompts_chatbooks",
        "study_core",
        "study_packs",
        "study_suggestions",
        "collections_reading_list",
        "collections_feed_subscriptions",
        "collections_outputs_templates_artifacts",
        "watchlists",
        "writing_suite",
        "research_sessions_runs",
        "research_search_provider_surfaces",
        "chat_grammars",
        "explicit_feedback",
        "claims_notifications_alerts",
        "meetings",
        "prompt_studio",
        "kanban_boards_tasks",
        "translation_utility",
        "client_notifications",
        "server_runtime_config_discovery",
        "llm_provider_model_catalog",
        "audio_speech_services",
        "voice_assistant",
        "auth_profile_sessions",
        "server_reminders_notification_feeds",
        "external_connectors",
        "server_skills",
        "server_tools",
        "text2sql_query",
        "sync_transport",
        "user_governance",
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
            _resource("chat.analytics", actions=(OBSERVE,)),
            _resource("chat.commands", actions=(LIST,)),
            _resource("chat.knowledge", actions=(CREATE,)),
            _resource("chat.loop", actions=(LAUNCH, OBSERVE, APPROVE, CANCEL)),
            _resource("chat.share_links", actions=(LIST, DETAIL, CREATE, REVOKE)),
        ),
    ),
    _capability(
        "characters_personas_ccp",
        "Characters / Personas / CCP",
        "characters",
        sources=SEPARATED_SOURCES,
        resources=(
            _resource("character.archetypes", actions=(LIST, DETAIL, PREVIEW)),
            _resource("character.persona", actions=CRUD_ACTIONS),
            _resource(
                "character.sessions",
                actions=_combine_action_sets(CRUD_ACTIONS, (LAUNCH, EXPORT, OBSERVE, RESTORE)),
            ),
            _resource(
                "character.messages",
                actions=CRUD_ACTIONS,
                domain_id="characters",
                sources=(SERVER_SOURCE,),
            ),
            _resource(
                "chat.dictionaries",
                actions=_combine_action_sets(CRUD_ACTIONS, (PROCESS, IMPORT, EXPORT)),
                domain_id="characters",
                sources=SEPARATED_SOURCES,
            ),
            _resource(
                "chat.dictionary.entries",
                actions=(LIST, CREATE, UPDATE, DELETE, REORDER),
                domain_id="characters",
                sources=SEPARATED_SOURCES,
            ),
            _resource(
                "chat.dictionary.activity",
                actions=(LIST,),
                domain_id="characters",
                sources=SEPARATED_SOURCES,
            ),
            _resource(
                "chat.dictionary.versions",
                actions=(LIST, DETAIL, RESTORE),
                domain_id="characters",
                sources=SEPARATED_SOURCES,
            ),
            _resource(
                "chat.dictionary.statistics",
                actions=(DETAIL,),
                domain_id="characters",
                sources=SEPARATED_SOURCES,
            ),
        ),
    ),
    _capability(
        "companion_personalization",
        "Companion / Personalization",
        "companion",
        sources=REMOTE_ONLY_SOURCES,
        resources=(
            _resource("companion.activity", actions=(LIST, DETAIL, CREATE)),
            _resource("companion.checkins", actions=(CREATE,)),
            _resource("companion.knowledge", actions=(LIST, DETAIL)),
            _resource("companion.reflections", actions=(DETAIL,)),
            _resource("companion.conversation_prompts", actions=(LIST,)),
            _resource("companion.goals", actions=(LIST, CREATE, UPDATE)),
            _resource("companion.lifecycle", actions=(PURGE, LAUNCH)),
            _resource("personalization.profile", actions=(DETAIL,)),
            _resource("personalization.opt_in", actions=(UPDATE,)),
            _resource("personalization.preferences", actions=(UPDATE,)),
            _resource("personalization.lifecycle", actions=(PURGE,)),
            _resource(
                "personalization.memories",
                actions=(LIST, DETAIL, CREATE, UPDATE, DELETE, IMPORT, EXPORT, VALIDATE),
            ),
            _resource("personalization.explanations", actions=(LIST,)),
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
            _resource("media.add", actions=(CREATE,)),
            _resource("media.file_artifacts", actions=(DETAIL, CREATE, DELETE, EXPORT, PURGE)),
            _resource("media.reference_images", actions=(LIST,)),
            _resource("media.reading_import_jobs", actions=(LIST, DETAIL)),
            _resource("media.reading.digest_schedules", actions=CRUD_ACTIONS),
            _resource("media.reading.digest_scheduler", actions=(TRIGGER,), sources=(LOCAL_SOURCE,)),
            _resource("media.reading.digest_outputs", actions=(LIST,)),
            _resource("media.web_content_ingest", actions=(LAUNCH,), sources=(SERVER_SOURCE,)),
            _resource("media.items", actions=(LIST, DETAIL, UPDATE, DELETE, RESTORE)),
            _resource("media.items.trash", actions=(LIST, DELETE)),
            _resource("media.items.keywords", actions=(LIST, UPDATE)),
            _resource("media.items.permanent", actions=(DELETE,)),
            _resource("media.items.metadata_search", actions=(LIST,)),
            _resource("media.items.identifier_lookup", actions=(DETAIL,)),
            _resource("media.items.file", actions=(DETAIL,)),
            _resource("media.processing.video", actions=(PROCESS,)),
            _resource("media.processing.audio", actions=(PROCESS,)),
            _resource("media.processing.pdf", actions=(PROCESS,)),
            _resource("media.processing.ebook", actions=(PROCESS,)),
            _resource("media.processing.document", actions=(PROCESS,)),
            _resource("media.processing.plaintext", actions=(PROCESS,)),
            _resource("media.processing.code", actions=(PROCESS,)),
            _resource("media.processing.emails", actions=(PROCESS,)),
            _resource("media.processing.web_scraping", actions=(PROCESS,)),
            _resource("media.web_scraping", actions=(STATUS, DETAIL, CANCEL, OBSERVE, INSPECT), sources=(SERVER_SOURCE,)),
            _resource("media.web_scraping.cookies", actions=(DETAIL, UPDATE), sources=(SERVER_SOURCE,)),
            _resource("media.web_scraping.service", actions=(INITIALIZE, SHUTDOWN), sources=(SERVER_SOURCE,)),
            _resource("media.processing.mediawiki", actions=(PROCESS, IMPORT)),
            _resource("media.transcription_models", actions=(LIST,), sources=(SERVER_SOURCE,)),
            _resource("media.reading.saved_searches", actions=CRUD_ACTIONS),
            _resource("media.reading.note_links", actions=(LIST, CREATE, DELETE)),
            _resource("media.reading_progress", actions=(DETAIL, UPDATE)),
            _resource("media.navigation", actions=(DETAIL,)),
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
            ),
            _resource("prompts.health", actions=(DETAIL,), sources=(SERVER_SOURCE,), domain_id="prompts"),
            _resource("prompts.sync_log", actions=(LIST,), sources=(SERVER_SOURCE,), domain_id="prompts"),
            _resource("prompts.search", actions=(LIST,), sources=(SERVER_SOURCE,), domain_id="prompts"),
            _resource("prompts.keywords", actions=(LIST, CREATE, DELETE, EXPORT), sources=(SERVER_SOURCE,), domain_id="prompts"),
            _resource("prompts.transfer", actions=(IMPORT, EXPORT), sources=(SERVER_SOURCE,), domain_id="prompts"),
            _resource("prompts.templates", actions=(PROCESS,), sources=(SERVER_SOURCE,), domain_id="prompts"),
            _resource("prompts.bulk", actions=(UPDATE, DELETE), sources=(SERVER_SOURCE,), domain_id="prompts"),
            _resource("prompts.usage", actions=(UPDATE,), sources=(SERVER_SOURCE,), domain_id="prompts"),
            _resource("prompts.collections", actions=(LIST, DETAIL, CREATE, UPDATE), sources=(SERVER_SOURCE,), domain_id="prompts"),
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
            _resource("study.flashcard", actions=CRUD_ACTIONS, domain_id="study"),
            _resource("study.flashcard.tags", actions=(LIST, UPDATE), domain_id="study"),
            _resource("study.flashcard.analytics", actions=(OBSERVE,), domain_id="study"),
            _resource("study.flashcard.review_sessions", actions=(LIST, OBSERVE), domain_id="study"),
            _resource("study.flashcard.assistant", actions=(DETAIL, LAUNCH), domain_id="study"),
            _resource("study.flashcard.generation", actions=(LAUNCH,), domain_id="study"),
            _resource("study.flashcard.assets", actions=(CREATE, DETAIL), domain_id="study"),
            _resource("study.flashcard.bulk", actions=(CREATE, UPDATE), domain_id="study"),
            _resource("study.flashcard.import", actions=(PREVIEW, IMPORT), domain_id="study"),
            _resource("study.flashcard.export", actions=(EXPORT,), domain_id="study"),
            _resource("study.flashcard.templates", actions=CRUD_ACTIONS, domain_id="study"),
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
        "collections_feed_subscriptions",
        "Collections: Feed Subscriptions",
        "collections_feeds",
        sources=SEPARATED_SOURCES,
        resources=(
            _resource("collections.feeds", actions=CRUD_ACTIONS),
            _resource("collections.feeds.websub", actions=(DETAIL, LAUNCH, DELETE), sources=(SERVER_SOURCE,)),
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
            _resource("writing.characters", actions=CRUD_ACTIONS),
            _resource("writing.relationships", actions=(LIST, CREATE, DELETE)),
            _resource("writing.world_info", actions=CRUD_ACTIONS),
            _resource("writing.plot_lines", actions=(LIST, CREATE, UPDATE, DELETE)),
            _resource("writing.plot_events", actions=(LIST, CREATE, UPDATE, DELETE)),
            _resource("writing.plot_holes", actions=(LIST, CREATE, UPDATE, DELETE)),
            _resource("writing.scene_characters", actions=(LIST, CREATE, DELETE)),
            _resource("writing.scene_world_info", actions=(LIST, CREATE, DELETE)),
            _resource("writing.citations", actions=(LIST, CREATE, DELETE)),
            _resource("writing.research", actions=(LAUNCH,)),
            _resource("writing.analysis", actions=(LIST, LAUNCH)),
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
        "chat_grammars",
        "Chat Grammars",
        "chat_grammars",
        sources=SEPARATED_SOURCES,
        resources=(
            _resource("chat.grammars", actions=CRUD_ACTIONS),
        ),
    ),
    _capability(
        "explicit_feedback",
        "Explicit Feedback",
        "feedback",
        sources=SEPARATED_SOURCES,
        resources=(
            _resource("feedback", actions=CRUD_ACTIONS),
        ),
    ),
    _capability(
        "claims_notifications_alerts",
        "Claims Notifications / Alerts",
        "claims",
        sources=REMOTE_ONLY_SOURCES,
        resources=(
            _resource("claims.status", actions=(DETAIL,)),
            _resource("claims.items", actions=(LIST, DETAIL, UPDATE, LAUNCH)),
            _resource("claims.search", actions=(LIST,)),
            _resource("claims.settings", actions=(LIST, UPDATE)),
            _resource("claims.monitoring", actions=(LIST, UPDATE)),
            _resource("claims.notifications", actions=(LIST, UPDATE, LAUNCH)),
            _resource("claims.alerts", actions=(LIST, CREATE, UPDATE, DELETE, LAUNCH)),
            _resource("claims.rebuild", actions=(DETAIL, LAUNCH)),
            _resource("claims.review", actions=(LIST, UPDATE, LAUNCH)),
            _resource("claims.review_rules", actions=(LIST, CREATE, UPDATE, DELETE)),
            _resource("claims.extractors", actions=(LIST,)),
            _resource("claims.analytics", actions=(LIST, DETAIL, EXPORT)),
            _resource("claims.clusters", actions=(LIST, DETAIL, LAUNCH)),
            _resource("claims.cluster_links", actions=(LIST, CREATE, DELETE)),
            _resource("claims.cluster_members", actions=(LIST,)),
            _resource("claims.cluster_timeline", actions=(LIST,)),
            _resource("claims.cluster_evidence", actions=(LIST,)),
            _resource("claims.fva", actions=(LIST, LAUNCH)),
        ),
    ),
    _capability(
        "meetings",
        "Meetings",
        "meetings",
        sources=REMOTE_ONLY_SOURCES,
        resources=(
            _resource("meetings.health", actions=(DETAIL,)),
            _resource("meetings.sessions", actions=(LIST, DETAIL, CREATE, UPDATE, LAUNCH)),
            _resource("meetings.templates", actions=(LIST, DETAIL, CREATE)),
            _resource("meetings.artifacts", actions=(LIST, CREATE)),
            _resource("meetings.share", actions=(LAUNCH,)),
            _resource("meetings.events", actions=(OBSERVE,)),
        ),
    ),
    _capability(
        "prompt_studio",
        "Prompt Studio",
        "prompt_studio",
        sources=REMOTE_ONLY_SOURCES,
        resources=(
            _resource("prompt_studio.projects", actions=(LIST, DETAIL, CREATE, UPDATE, DELETE, ARCHIVE, RESTORE)),
            _resource("prompt_studio.project_stats", actions=(DETAIL,)),
            _resource("prompt_studio.prompts", actions=(LIST, DETAIL, CREATE, UPDATE, RESTORE, PREVIEW, PROCESS, LAUNCH)),
            _resource("prompt_studio.prompt_versions", actions=(LIST,)),
            _resource("prompt_studio.test_cases", actions=(LIST, DETAIL, CREATE, UPDATE, DELETE, IMPORT, EXPORT, LAUNCH)),
            _resource("prompt_studio.evaluations", actions=(LIST, DETAIL, CREATE, DELETE)),
            _resource("prompt_studio.optimizations", actions=(LIST, DETAIL, CREATE, CANCEL, LAUNCH)),
            _resource("prompt_studio.optimization_strategies", actions=(LIST, LAUNCH)),
            _resource("prompt_studio.optimization_iterations", actions=(LIST, CREATE)),
            _resource("prompt_studio.status", actions=(DETAIL,)),
            _resource("prompt_studio.events", actions=(OBSERVE,)),
        ),
    ),
    _capability(
        "kanban_boards_tasks",
        "Kanban Boards / Tasks",
        "kanban",
        sources=REMOTE_ONLY_SOURCES,
        resources=(
            _resource("kanban.boards", actions=(LIST, DETAIL, CREATE, UPDATE, DELETE, ARCHIVE, RESTORE, IMPORT, EXPORT)),
            _resource("kanban.lists", actions=(LIST, DETAIL, CREATE, UPDATE, DELETE, ARCHIVE, RESTORE, REORDER)),
            _resource("kanban.cards", actions=(LIST, DETAIL, CREATE, UPDATE, DELETE, ARCHIVE, RESTORE, REORDER, LAUNCH)),
            _resource("kanban.activities", actions=(OBSERVE_LIST,)),
            _resource("kanban.labels", actions=CRUD_ACTIONS),
            _resource("kanban.card_labels", actions=(LIST, CREATE, DELETE)),
            _resource("kanban.checklists", actions=(LIST, DETAIL, CREATE, UPDATE, DELETE, REORDER)),
            _resource("kanban.checklist_items", actions=(LIST, DETAIL, CREATE, UPDATE, DELETE, REORDER)),
            _resource("kanban.comments", actions=CRUD_ACTIONS),
            _resource("kanban.search", actions=(LIST, DETAIL)),
            _resource("kanban.card_links", actions=(LIST, DETAIL, CREATE, DELETE)),
        ),
    ),
    _capability(
        "translation_utility",
        "Translation Utility",
        "translation",
        sources=REMOTE_ONLY_SOURCES,
        resources=(
            _resource("translation.text", actions=(LAUNCH,)),
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
        "server_runtime_config_discovery",
        "Server Runtime / Config Discovery",
        "server_runtime",
        sources=REMOTE_ONLY_SOURCES,
        resources=(
            _resource("server.runtime.health", actions=(LIST, OBSERVE)),
            _resource("server.runtime.config", actions=(LIST, UPDATE)),
            _resource("server.runtime.providers", actions=(LIST, VALIDATE)),
        ),
    ),
    _capability(
        "llm_provider_model_catalog",
        "LLM Provider / Model Catalog",
        "llm_catalog",
        sources=SEPARATED_SOURCES,
        resources=(
            _resource("llm.catalog.health", actions=(OBSERVE,)),
            _resource("llm.catalog.providers", actions=(LIST, DETAIL, CONFIGURE)),
            _resource("llm.catalog.models", actions=(LIST, DETAIL)),
        ),
    ),
    _capability(
        "audio_speech_services",
        "Audio / Speech Services",
        "audio",
        sources=SEPARATED_SOURCES,
        resources=(
            _resource("audio.health", actions=(OBSERVE,)),
            _resource("audio.providers", actions=(LIST,)),
            _resource("audio.voices", actions=(LIST, DETAIL, CREATE, DELETE, PREVIEW, LAUNCH)),
            _resource("audio.speech", actions=(LAUNCH,)),
            _resource("audio.speech_chat", actions=(LAUNCH,), sources=(SERVER_SOURCE,)),
            _resource("audio.streaming", actions=(STATUS, DETAIL, LAUNCH), sources=(SERVER_SOURCE,)),
            _resource("audio.speech_jobs", actions=(DETAIL,)),
            _resource("audio.jobs", actions=(CREATE, DETAIL, OBSERVE)),
            _resource("audio.history", actions=(LIST, DETAIL, UPDATE, DELETE)),
            _resource("audio.transcriptions", actions=(LAUNCH,)),
            _resource("audio.translations", actions=(LAUNCH,)),
            _resource("audio.tokenizer", actions=(LAUNCH,)),
            _resource("audiobooks.parse", actions=(LAUNCH,), domain_id="audiobooks"),
            _resource("audiobooks.jobs", actions=(CREATE, DETAIL, OBSERVE), domain_id="audiobooks"),
            _resource("audiobooks.projects", actions=(LIST, DETAIL), domain_id="audiobooks"),
            _resource("audiobooks.chapters", actions=(LIST,), domain_id="audiobooks"),
            _resource("audiobooks.artifacts", actions=(LIST,), domain_id="audiobooks"),
            _resource("audiobooks.voice_profiles", actions=(LIST, CREATE, DELETE), domain_id="audiobooks"),
            _resource("audiobooks.subtitles", actions=(EXPORT,), domain_id="audiobooks"),
        ),
    ),
    _capability(
        "voice_assistant",
        "Voice Assistant",
        "voice_assistant",
        sources=REMOTE_ONLY_SOURCES,
        resources=(
            _resource("voice_assistant.commands", actions=(LIST, DETAIL, CREATE, UPDATE, DELETE, LAUNCH, PREVIEW, OBSERVE)),
            _resource("voice_assistant.sessions", actions=(LIST, DETAIL, DELETE)),
            _resource("voice_assistant.analytics", actions=(OBSERVE,)),
        ),
    ),
    _capability(
        "auth_profile_sessions",
        "Auth / Profile / Sessions",
        "auth",
        sources=REMOTE_ONLY_SOURCES,
        resources=(
            _resource("auth.identity", actions=(LAUNCH, UPDATE, DELETE)),
            _resource("auth.sessions", actions=(LIST, DELETE)),
            _resource("auth.profile", actions=(LIST, DETAIL, UPDATE)),
            _resource("auth.registration", actions=(CREATE,)),
            _resource("auth.security", actions=(LAUNCH, UPDATE)),
            _resource("auth.api_keys", actions=(LIST, CREATE, UPDATE, DELETE)),
            _resource("auth.provider_keys", actions=(LIST, DETAIL, CREATE, UPDATE, DELETE, VALIDATE)),
            _resource("auth.storage", actions=_combine_action_sets(CRUD_ACTIONS, (EXPORT,))),
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
        "external_connectors",
        "External Connectors",
        "connectors",
        sources=REMOTE_ONLY_SOURCES,
        resources=(
            _resource("connectors.providers", actions=(LIST, LAUNCH)),
            _resource("connectors.accounts", actions=(LIST, DELETE)),
            _resource("connectors.sources", actions=(LIST, CREATE, UPDATE, LAUNCH, OBSERVE)),
            _resource("connectors.jobs", actions=(OBSERVE,)),
        ),
    ),
    _capability(
        "server_skills",
        "Server Skills",
        "skills",
        sources=REMOTE_ONLY_SOURCES,
        resources=(
            _resource("skills", actions=CRUD_ACTIONS),
            _resource("skills.context", actions=(LIST,)),
            _resource("skills.import", actions=(LAUNCH,)),
            _resource("skills.export", actions=(LAUNCH,)),
            _resource("skills.execute", actions=(LAUNCH,)),
            _resource("skills.seed", actions=(LAUNCH,)),
        ),
    ),
    _capability(
        "server_tools",
        "Server Tools",
        "tools",
        sources=REMOTE_ONLY_SOURCES,
        resources=(
            _resource("tools.catalog", actions=(LIST,)),
            _resource("tools.execution", actions=(LAUNCH,)),
        ),
    ),
    _capability(
        "text2sql_query",
        "Text2SQL Query",
        "text2sql",
        sources=REMOTE_ONLY_SOURCES,
        resources=(
            _resource("text2sql.targets", actions=(LIST,)),
            _resource("text2sql.query", actions=(LAUNCH,)),
        ),
    ),
    _capability(
        "sync_transport",
        "Sync Transport",
        "sync",
        sources=REMOTE_ONLY_SOURCES,
        resources=(
            _resource("sync.changes", actions=(LAUNCH, OBSERVE)),
        ),
    ),
    _capability(
        "user_governance",
        "User Governance / Consent",
        "user_governance",
        sources=REMOTE_ONLY_SOURCES,
        resources=(
            _resource("user_governance.consent", actions=(LIST, UPDATE)),
            _resource("user_governance.privileges", actions=(LIST, DETAIL)),
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
            _resource("mcp.inventory", actions=(LIST, OBSERVE)),
            _resource("mcp.external_profiles", actions=(LIST, CONFIGURE, LAUNCH, TRIGGER, OBSERVE)),
            _resource("mcp.governance", actions=(LIST, CONFIGURE, OBSERVE, APPROVE)),
        ),
    ),
    _capability(
        "remote_mcp_control_plane_governance",
        "Remote MCP Control Plane / Governance",
        "mcp_governance",
        sources=REMOTE_ONLY_SOURCES,
        resources=(
            _resource("mcp.runtime", actions=(OBSERVE,)),
            _resource("mcp.inventory", actions=(LIST,)),
            _resource("mcp.catalogs", actions=(LIST, CONFIGURE)),
            _resource("mcp.external_servers", actions=(LIST, CONFIGURE)),
            _resource("mcp.credentials", actions=(LIST, CONFIGURE)),
            _resource("mcp.governance", actions=(LIST, CONFIGURE, OBSERVE, APPROVE)),
            _resource("mcp.effective_access", actions=(OBSERVE,)),
            _resource("mcp.advanced", actions=(LIST, CONFIGURE, TRIGGER, OBSERVE)),
            _resource("mcp.governance.tool_registry", actions=(LIST, DETAIL)),
            _resource("mcp.governance.capability_mappings", actions=(LIST, PREVIEW, CREATE, UPDATE, DELETE)),
            _resource("mcp.governance.external_servers", actions=(LIST, CREATE, UPDATE, DELETE)),
            _resource("mcp.governance.external_servers.secrets", actions=(UPDATE,)),
            _resource("mcp.governance.permission_profiles", actions=(LIST, CREATE, UPDATE, DELETE)),
            _resource("mcp.governance.policy_assignments", actions=(LIST, CREATE, UPDATE, DELETE)),
            _resource("mcp.governance.approval_policies", actions=(LIST, CREATE, UPDATE, DELETE)),
            _resource("mcp.governance.approval_decisions", actions=(APPROVE,)),
            _resource("mcp.governance.effective_policy", actions=(DETAIL,)),
            _resource("mcp.governance.catalogs", actions=(LIST, CREATE, DELETE)),
            _resource("mcp.governance.catalog_entries", actions=(CREATE, DELETE)),
            _resource("mcp.governance.events", actions=(OBSERVE,)),
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
            _resource(
                "evaluations.embeddings_abtest",
                actions=(CREATE, LAUNCH, DETAIL, OBSERVE, EXPORT, DELETE),
                sources=(SERVER_SOURCE,),
            ),
            _resource(
                "evaluations.synthetic",
                actions=(LIST, CREATE, UPDATE, LAUNCH),
                sources=(SERVER_SOURCE,),
            ),
            _resource(
                "evaluations.benchmarks",
                actions=(LIST, DETAIL, LAUNCH),
                sources=(SERVER_SOURCE,),
            ),
            _resource(
                "evaluations.webhooks",
                actions=(LIST, CREATE, DELETE, LAUNCH),
                sources=(SERVER_SOURCE,),
            ),
            _resource(
                "evaluations.recipes",
                actions=(LIST, DETAIL, LAUNCH, OBSERVE),
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
            _resource(
                "rag.media_embeddings",
                actions=(STATUS, CREATE, SEARCH, DELETE),
                sources=(SERVER_SOURCE,),
            ),
            _resource(
                "rag.media_embedding_jobs",
                actions=(LIST, DETAIL),
                sources=(SERVER_SOURCE,),
            ),
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
