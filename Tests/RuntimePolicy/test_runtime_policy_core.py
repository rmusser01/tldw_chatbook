from collections import defaultdict
from datetime import datetime, timedelta, timezone

from tldw_chatbook.runtime_policy.engine import PolicyEngine
import pytest

from tldw_chatbook.runtime_policy.enforcement import ServicePolicyEnforcer, classify_backend_exception
from tldw_chatbook.runtime_policy.registry import CAPABILITY_ACTION_MATRIX, CAPABILITY_REGISTRY
from tldw_chatbook.runtime_policy.types import PolicyDeniedError, RuntimeSourceState


def _backend_exception_type(name: str, *, module_name: str):
    class _BackendError(Exception):
        pass

    _BackendError.__name__ = name
    _BackendError.__qualname__ = name
    _BackendError.__module__ = module_name
    return _BackendError


def _backend_auth_error(
    *,
    module_name: str = "tldw_chatbook.tldw_api.exceptions",
    message: str = "bad credentials",
    response_data=None,
):
    error_type = _backend_exception_type("AuthenticationError", module_name=module_name)
    error = error_type(message)
    error.response_data = response_data or {}
    return error


def _backend_response_error(
    status_code: int,
    message: str,
    *,
    module_name: str = "tldw_chatbook.tldw_api.exceptions",
    response_data=None,
):
    error_type = _backend_exception_type("APIResponseError", module_name=module_name)
    error = error_type(message)
    error.status_code = status_code
    error.response_data = response_data or {}
    return error


def _action_kinds(*kinds: str) -> frozenset[str]:
    return frozenset(kinds)


def _action_ids(block: str) -> frozenset[str]:
    return frozenset(line.strip() for line in block.splitlines() if line.strip())


FULL_CRUD = _action_kinds("browse", "detail", "create", "update", "delete")
FULL_CRUD_AND_LAUNCH = _action_kinds("browse", "detail", "create", "update", "delete", "launch")
FULL_CRUD_AND_LAUNCH_AND_OBSERVE = _action_kinds(
    "browse", "detail", "create", "update", "delete", "launch", "observe"
)
DISCOVER_TRIGGER_OBSERVE = _action_kinds("browse", "launch", "observe")
DISCOVER_CONFIGURE_TRIGGER_OBSERVE = _action_kinds("browse", "update", "launch", "observe")


EXPECTED_AUDITED_CAPABILITIES = {
    "chat": {
        "expected_domain_ids": {"chat"},
        "expected_action_kinds_by_source": {
            "local": FULL_CRUD_AND_LAUNCH,
            "server": FULL_CRUD_AND_LAUNCH,
        },
    },
    "characters_personas_ccp": {
        "expected_domain_ids": {"characters"},
        "expected_action_kinds_by_source": {
            "local": FULL_CRUD_AND_LAUNCH,
            "server": FULL_CRUD_AND_LAUNCH,
        },
    },
    "notes_workspaces": {
        "expected_domain_ids": {"notes"},
        "expected_action_kinds_by_source": {
            "local": FULL_CRUD,
            "server": FULL_CRUD,
        },
    },
    "media_reading_ingestion_sources": {
        "expected_domain_ids": {"media"},
        "expected_action_kinds_by_source": {
            "local": FULL_CRUD_AND_LAUNCH_AND_OBSERVE,
            "server": FULL_CRUD_AND_LAUNCH_AND_OBSERVE,
        },
    },
    "prompts_chatbooks": {
        "expected_domain_ids": {"prompts", "chatbooks"},
        "expected_action_kinds_by_source": {
            "local": FULL_CRUD_AND_LAUNCH,
            "server": FULL_CRUD_AND_LAUNCH,
        },
    },
    "study_core": {
        "expected_domain_ids": {"study", "quiz"},
        "expected_action_kinds_by_source": {
            "local": FULL_CRUD_AND_LAUNCH_AND_OBSERVE,
            "server": FULL_CRUD_AND_LAUNCH_AND_OBSERVE,
        },
    },
    "study_packs": {
        "expected_domain_ids": {"study_packs"},
        "expected_action_kinds_by_source": {
            "server": DISCOVER_TRIGGER_OBSERVE,
        },
    },
    "study_suggestions": {
        "expected_domain_ids": {"study_suggestions"},
        "expected_action_kinds_by_source": {
            "server": DISCOVER_CONFIGURE_TRIGGER_OBSERVE,
        },
    },
    "collections_reading_list": {
        "expected_domain_ids": {"collections_reading"},
        "expected_action_kinds_by_source": {
            "local": FULL_CRUD,
            "server": FULL_CRUD,
        },
    },
    "collections_outputs_templates_artifacts": {
        "expected_domain_ids": {"outputs"},
        "expected_action_kinds_by_source": {
            "server": FULL_CRUD_AND_LAUNCH_AND_OBSERVE,
        },
    },
    "watchlists": {
        "expected_domain_ids": {"watchlists"},
        "expected_action_kinds_by_source": {
            "local": FULL_CRUD_AND_LAUNCH_AND_OBSERVE,
            "server": FULL_CRUD_AND_LAUNCH_AND_OBSERVE,
        },
    },
    "writing_suite": {
        "expected_domain_ids": {"writing"},
        "expected_action_kinds_by_source": {
            "local": FULL_CRUD,
            "server": FULL_CRUD,
        },
    },
    "research_sessions_runs": {
        "expected_domain_ids": {"research"},
        "expected_action_kinds_by_source": {
            "local": FULL_CRUD_AND_LAUNCH_AND_OBSERVE,
            "server": FULL_CRUD_AND_LAUNCH_AND_OBSERVE,
        },
    },
    "research_search_provider_surfaces": {
        "expected_domain_ids": {"research_search"},
        "expected_action_kinds_by_source": {
            "server": DISCOVER_CONFIGURE_TRIGGER_OBSERVE,
        },
    },
    "client_notifications": {
        "expected_domain_ids": {"notifications"},
        "expected_action_kinds_by_source": {
            "local": DISCOVER_CONFIGURE_TRIGGER_OBSERVE,
        },
    },
    "server_reminders_notification_feeds": {
        "expected_domain_ids": {"notifications_server"},
        "expected_action_kinds_by_source": {
            "server": DISCOVER_CONFIGURE_TRIGGER_OBSERVE,
        },
    },
    "workflows": {
        "expected_domain_ids": {"workflows"},
        "expected_action_kinds_by_source": {
            "server": DISCOVER_TRIGGER_OBSERVE,
        },
    },
    "scheduler_workflows": {
        "expected_domain_ids": {"scheduler"},
        "expected_action_kinds_by_source": {
            "server": DISCOVER_CONFIGURE_TRIGGER_OBSERVE,
        },
    },
    "chat_workflows": {
        "expected_domain_ids": {"chat_workflows"},
        "expected_action_kinds_by_source": {
            "server": DISCOVER_TRIGGER_OBSERVE,
        },
    },
    "local_mcp_runtime": {
        "expected_domain_ids": {"mcp_governance", "mcp_runtime"},
        "expected_action_kinds_by_source": {
            "local": DISCOVER_CONFIGURE_TRIGGER_OBSERVE,
        },
    },
    "remote_mcp_control_plane_governance": {
        "expected_domain_ids": {"mcp_governance", "mcp_runtime"},
        "expected_action_kinds_by_source": {
            "server": DISCOVER_CONFIGURE_TRIGGER_OBSERVE,
        },
    },
    "sharing": {
        "expected_domain_ids": {"sharing"},
        "expected_action_kinds_by_source": {
            "server": DISCOVER_CONFIGURE_TRIGGER_OBSERVE,
        },
    },
    "web_clipper": {
        "expected_domain_ids": {"web_clipper"},
        "expected_action_kinds_by_source": {
            "server": DISCOVER_TRIGGER_OBSERVE,
        },
    },
    "evaluations": {
        "expected_domain_ids": {"evaluations"},
        "expected_action_kinds_by_source": {
            "local": FULL_CRUD_AND_LAUNCH_AND_OBSERVE,
            "server": FULL_CRUD_AND_LAUNCH_AND_OBSERVE,
        },
    },
    "rag_embeddings_chunking_admin": {
        "expected_domain_ids": {"rag"},
        "expected_action_kinds_by_source": {
            "local": DISCOVER_CONFIGURE_TRIGGER_OBSERVE,
            "server": DISCOVER_CONFIGURE_TRIGGER_OBSERVE,
        },
    },
    "cross_cutting_runtime_policy": {
        "expected_domain_ids": {"runtime_policy"},
        "expected_action_kinds_by_source": {
            "local": DISCOVER_CONFIGURE_TRIGGER_OBSERVE,
            "server": DISCOVER_CONFIGURE_TRIGGER_OBSERVE,
        },
    },
}

EXPECTED_ACTION_IDS_BY_CAPABILITY = {
    "characters_personas_ccp": _action_ids("""
        character.persona.create.local
        character.persona.create.server
        character.persona.delete.local
        character.persona.delete.server
        character.persona.detail.local
        character.persona.detail.server
        character.persona.list.local
        character.persona.list.server
        character.persona.update.local
        character.persona.update.server
        character.memory.create.local
        character.memory.create.server
        character.memory.delete.local
        character.memory.delete.server
        character.memory.detail.local
        character.memory.detail.server
        character.memory.launch.local
        character.memory.launch.server
        character.memory.list.local
        character.memory.list.server
        character.memory.update.local
        character.memory.update.server
        character.greetings.list.server
        character.greetings.update.server
        character.messages.create.local
        character.messages.create.server
        character.messages.delete.local
        character.messages.delete.server
        character.messages.detail.local
        character.messages.detail.server
        character.messages.list.local
        character.messages.list.server
        character.messages.update.local
        character.messages.update.server
        character.presets.create.server
        character.presets.delete.server
        character.presets.list.server
        character.presets.update.server
        character.sessions.create.local
        character.sessions.create.server
        character.sessions.delete.local
        character.sessions.delete.server
        character.sessions.detail.local
        character.sessions.detail.server
        character.sessions.launch.local
        character.sessions.launch.server
        character.sessions.list.local
        character.sessions.list.server
        character.sessions.update.local
        character.sessions.update.server
        character.world_book_entries.create.local
        character.world_book_entries.delete.local
        character.world_book_entries.detail.local
        character.world_book_entries.list.local
        character.world_book_entries.update.local
        character.world_books.create.local
        character.world_books.delete.local
        character.world_books.detail.local
        character.world_books.list.local
        character.world_books.update.local
    """),
    "chat": _action_ids("""
        chat.create.local
        chat.create.server
        chat.delete.local
        chat.delete.server
        chat.detail.local
        chat.detail.server
        chat.launch.local
        chat.launch.server
        chat.list.local
        chat.list.server
        chat.update.local
        chat.update.server
    """),
    "chat_workflows": _action_ids("""
        chat.workflows.launch.server
        chat.workflows.list.server
        chat.workflows.observe.server
    """),
    "client_notifications": _action_ids("""
        notifications.dispatch.launch.local
        notifications.preferences.configure.local
        notifications.preferences.list.local
        notifications.queue.list.local
        notifications.queue.update.local
        notifications.queue.observe.local
    """),
    "collections_outputs_templates_artifacts": _action_ids("""
        outputs.artifacts.create.server
        outputs.artifacts.delete.server
        outputs.artifacts.detail.server
        outputs.artifacts.list.server
        outputs.artifacts.update.server
        outputs.render_jobs.detail.server
        outputs.render_jobs.launch.server
        outputs.render_jobs.list.server
        outputs.render_jobs.observe.server
        outputs.templates.create.server
        outputs.templates.delete.server
        outputs.templates.detail.server
        outputs.templates.list.server
        outputs.templates.update.server
    """),
    "collections_reading_list": _action_ids("""
        collections.reading_list.create.local
        collections.reading_list.create.server
        collections.reading_list.delete.local
        collections.reading_list.delete.server
        collections.reading_list.detail.local
        collections.reading_list.detail.server
        collections.reading_list.list.local
        collections.reading_list.list.server
        collections.reading_list.update.local
        collections.reading_list.update.server
    """),
    "cross_cutting_runtime_policy": _action_ids("""
        runtime.policy.configure.local
        runtime.policy.configure.server
        runtime.policy.launch.local
        runtime.policy.launch.server
        runtime.policy.list.local
        runtime.policy.list.server
        runtime.policy.observe.local
        runtime.policy.observe.server
    """),
    "evaluations": _action_ids("""
        evaluations.dataset.create.local
        evaluations.dataset.create.server
        evaluations.dataset.delete.local
        evaluations.dataset.delete.server
        evaluations.dataset.detail.local
        evaluations.dataset.detail.server
        evaluations.dataset.list.local
        evaluations.dataset.list.server
        evaluations.dataset.update.local
        evaluations.run.create.local
        evaluations.run.create.server
        evaluations.run.delete.local
        evaluations.run.delete.server
        evaluations.run.detail.local
        evaluations.run.detail.server
        evaluations.run.launch.local
        evaluations.run.launch.server
        evaluations.run.list.local
        evaluations.run.list.server
        evaluations.run.observe.local
        evaluations.run.observe.server
        evaluations.run.update.local
        evaluations.run.update.server
    """),
    "local_mcp_runtime": _action_ids("""
        mcp.external_profiles.configure.local
        mcp.external_profiles.launch.local
        mcp.external_profiles.list.local
        mcp.external_profiles.observe.local
        mcp.external_profiles.trigger.local
        mcp.governance.approve.local
        mcp.governance.configure.local
        mcp.governance.list.local
        mcp.governance.observe.local
        mcp.inventory.list.local
        mcp.inventory.observe.local
        mcp.runtime.configure.local
        mcp.runtime.launch.local
        mcp.runtime.list.local
        mcp.runtime.observe.local
        mcp.runtime.trigger.local
    """),
    "media_reading_ingestion_sources": _action_ids("""
        media.ingestion_jobs.detail.local
        media.ingestion_jobs.detail.server
        media.ingestion_jobs.launch.local
        media.ingestion_jobs.launch.server
        media.ingestion_jobs.list.local
        media.ingestion_jobs.list.server
        media.ingestion_jobs.observe.local
        media.ingestion_jobs.observe.server
        media.ingestion_sources.create.local
        media.ingestion_sources.create.server
        media.ingestion_sources.delete.local
        media.ingestion_sources.detail.local
        media.ingestion_sources.detail.server
        media.ingestion_sources.list.local
        media.ingestion_sources.list.server
        media.ingestion_sources.update.local
        media.ingestion_sources.update.server
        media.items.reprocess.launch.server
        media.processing.launch.server
        media.processing_models.list.server
        media.document_annotations.create.local
        media.document_annotations.create.server
        media.document_annotations.delete.local
        media.document_annotations.delete.server
        media.document_annotations.list.local
        media.document_annotations.list.server
        media.document_annotations.update.local
        media.document_annotations.update.server
        media.document_figures.list.local
        media.document_figures.list.server
        media.document_insights.create.local
        media.document_insights.create.server
        media.document_navigation.detail.local
        media.document_navigation.detail.server
        media.document_navigation_content.detail.local
        media.document_navigation_content.detail.server
        media.document_outline.detail.local
        media.document_outline.detail.server
        media.document_references.list.local
        media.document_references.list.server
        media.items.create.server
        media.items.delete.server
        media.items.detail.server
        media.items.file.detail.server
        media.items.identifier_lookup.detail.server
        media.items.keywords.update.server
        media.items.keywords.list.server
        media.items.list.server
        media.items.metadata_search.list.server
        media.items.permanent_delete.server
        media.items.restore.server
        media.items.trash.delete.server
        media.items.trash.list.server
        media.items.update.server
        media.unified_items.delete.server
        media.unified_items.detail.server
        media.unified_items.list.server
        media.unified_items.update.server
        media.reading.create.local
        media.reading.create.server
        media.reading.delete.local
        media.reading.delete.server
        media.reading.detail.local
        media.reading.detail.server
        media.reading_highlights.create.local
        media.reading_highlights.create.server
        media.reading_highlights.delete.local
        media.reading_highlights.delete.server
        media.reading_highlights.detail.local
        media.reading_highlights.detail.server
        media.reading_highlights.list.local
        media.reading_highlights.list.server
        media.reading_highlights.update.local
        media.reading_highlights.update.server
        media.reading.list.local
        media.reading.list.server
        media.reading.update.local
        media.reading.update.server
        media.reading_archives.create.local
        media.reading_archives.create.server
        media.reading_digest_outputs.list.local
        media.reading_digest_outputs.list.server
        media.reading_digest_schedules.create.local
        media.reading_digest_schedules.create.server
        media.reading_digest_schedules.delete.local
        media.reading_digest_schedules.delete.server
        media.reading_digest_schedules.detail.local
        media.reading_digest_schedules.detail.server
        media.reading_digest_schedules.list.local
        media.reading_digest_schedules.list.server
        media.reading_digest_schedules.update.local
        media.reading_digest_schedules.update.server
        media.reading_export.export.local
        media.reading_export.export.server
        media.reading_import.detail.local
        media.reading_import.detail.server
        media.reading_import.launch.local
        media.reading_import.launch.server
        media.reading_import.list.local
        media.reading_import.list.server
        media.reading_note_links.create.local
        media.reading_note_links.create.server
        media.reading_note_links.delete.local
        media.reading_note_links.delete.server
        media.reading_note_links.list.local
        media.reading_note_links.list.server
        media.reading_progress.detail.local
        media.reading_progress.detail.server
        media.reading_progress.update.local
        media.reading_progress.update.server
        media.reading_saved_searches.create.local
        media.reading_saved_searches.create.server
        media.reading_saved_searches.delete.local
        media.reading_saved_searches.delete.server
        media.reading_saved_searches.list.local
        media.reading_saved_searches.list.server
        media.reading_saved_searches.update.local
        media.reading_saved_searches.update.server
        media.reading_summaries.create.local
        media.reading_summaries.create.server
        media.reading_tts.launch.local
        media.reading_tts.launch.server
        media.web_content_ingest.launch.local
        media.web_content_ingest.launch.server
    """),
    "notes_workspaces": _action_ids("""
        notes.create.local
        notes.create.server
        notes.create.workspace
        notes.delete.local
        notes.delete.server
        notes.delete.workspace
        notes.detail.local
        notes.detail.server
        notes.detail.workspace
        notes.graph.create.local
        notes.graph.create.server
        notes.graph.delete.local
        notes.graph.delete.server
        notes.graph.detail.local
        notes.graph.detail.server
        notes.graph.list.local
        notes.graph.list.server
        notes.list.local
        notes.list.server
        notes.list.workspace
        notes.update.local
        notes.update.server
        notes.update.workspace
        notes.workspace.create.local
        notes.workspace.create.server
        notes.workspace.delete.local
        notes.workspace.delete.server
        notes.workspace.detail.local
        notes.workspace.detail.server
        notes.workspace.list.local
        notes.workspace.list.server
        notes.workspace.update.local
        notes.workspace.update.server
    """),
    "prompts_chatbooks": _action_ids("""
        chatbooks.create.local
        chatbooks.create.server
        chatbooks.delete.local
        chatbooks.delete.server
        chatbooks.detail.local
        chatbooks.detail.server
        chatbooks.export.local
        chatbooks.export.server
        chatbooks.export_jobs.delete.server
        chatbooks.export_jobs.detail.server
        chatbooks.export_jobs.export.server
        chatbooks.export_jobs.list.server
        chatbooks.export_jobs.update.server
        chatbooks.import.local
        chatbooks.import.server
        chatbooks.import_jobs.delete.server
        chatbooks.import_jobs.detail.server
        chatbooks.import_jobs.list.server
        chatbooks.import_jobs.update.server
        chatbooks.list.local
        chatbooks.list.server
        chatbooks.update.local
        chatbooks.update.server
        prompts.collections.create.local
        prompts.collections.create.server
        prompts.collections.detail.local
        prompts.collections.detail.server
        prompts.collections.list.local
        prompts.collections.list.server
        prompts.collections.update.local
        prompts.collections.update.server
        prompts.create.local
        prompts.create.server
        prompts.delete.local
        prompts.delete.server
        prompts.list.local
        prompts.list.server
        prompts.preview.local
        prompts.preview.server
        prompts.restore_version.server
        prompts.update.local
        prompts.update.server
        prompts.use.local
        prompts.use.server
        prompts.versions.server
    """),
    "rag_embeddings_chunking_admin": _action_ids("""
        rag.admin.configure.local
        rag.admin.configure.server
        rag.admin.launch.local
        rag.admin.launch.server
        rag.admin.list.local
        rag.admin.list.server
        rag.admin.observe.local
        rag.admin.observe.server
        rag.template.create.local
        rag.template.create.server
        rag.template.delete.local
        rag.template.delete.server
        rag.template.detail.local
        rag.template.detail.server
        rag.template.list.local
        rag.template.list.server
        rag.template.update.local
        rag.template.update.server
    """),
    "remote_mcp_control_plane_governance": _action_ids("""
        mcp.advanced.configure.server
        mcp.advanced.list.server
        mcp.advanced.observe.server
        mcp.advanced.trigger.server
        mcp.catalogs.configure.server
        mcp.catalogs.list.server
        mcp.catalogs.observe.server
        mcp.catalogs.trigger.server
        mcp.credentials.configure.server
        mcp.credentials.list.server
        mcp.credentials.observe.server
        mcp.effective_access.observe.server
        mcp.external_servers.configure.server
        mcp.external_servers.list.server
        mcp.external_servers.observe.server
        mcp.external_servers.trigger.server
        mcp.governance.approve.server
        mcp.governance.configure.server
        mcp.governance.launch.server
        mcp.governance.list.server
        mcp.governance.observe.server
        mcp.inventory.list.server
        mcp.inventory.observe.server
        mcp.runtime.observe.server
        mcp.tools.trigger.server
    """),
    "research_search_provider_surfaces": _action_ids("""
        research.search.providers.configure.server
        research.search.providers.launch.server
        research.search.providers.list.server
        research.search.providers.observe.server
    """),
    "research_sessions_runs": _action_ids("""
        research.runs.create.local
        research.runs.create.server
        research.runs.delete.local
        research.runs.delete.server
        research.runs.detail.local
        research.runs.detail.server
        research.runs.launch.local
        research.runs.launch.server
        research.runs.list.local
        research.runs.list.server
        research.runs.observe.local
        research.runs.observe.server
        research.runs.update.local
        research.runs.update.server
        research.sessions.create.local
        research.sessions.create.server
        research.sessions.delete.local
        research.sessions.delete.server
        research.sessions.detail.local
        research.sessions.detail.server
        research.sessions.list.local
        research.sessions.list.server
        research.sessions.update.local
        research.sessions.update.server
    """),
    "scheduler_workflows": _action_ids("""
        scheduler.workflows.configure.server
        scheduler.workflows.launch.server
        scheduler.workflows.list.server
        scheduler.workflows.observe.server
    """),
    "server_reminders_notification_feeds": _action_ids("""
        notifications.feed.configure.server
        notifications.feed.list.server
        notifications.feed.observe.server
        notifications.reminders.configure.server
        notifications.reminders.launch.server
        notifications.reminders.list.server
        notifications.reminders.observe.server
    """),
    "sharing": _action_ids("""
        sharing.links.create.server
        sharing.links.inspect.server
        sharing.links.launch.server
        sharing.links.list.server
        sharing.links.observe.server
        sharing.links.revoke.server
        sharing.permissions.configure.server
    """),
    "study_core": _action_ids("""
        quiz.attempt.create.local
        quiz.attempt.create.server
        quiz.attempt.observe.local
        quiz.attempt.observe.server
        quiz.create.local
        quiz.create.server
        quiz.delete.local
        quiz.delete.server
        quiz.detail.local
        quiz.detail.server
        quiz.list.local
        quiz.list.server
        quiz.question.detail.local
        quiz.question.detail.server
        quiz.question.list.local
        quiz.question.list.server
        quiz.update.local
        quiz.update.server
        study.deck.create.local
        study.deck.create.server
        study.deck.delete.local
        study.deck.delete.server
        study.deck.detail.local
        study.deck.detail.server
        study.deck.list.local
        study.deck.list.server
        study.deck.update.local
        study.deck.update.server
        study.flashcard.assets.create.local
        study.flashcard.bulk.create.local
        study.flashcard.assets.create.server
        study.flashcard.assets.detail.local
        study.flashcard.assets.detail.server
        study.flashcard.assistant.create.server
        study.flashcard.assistant.detail.server
        study.flashcard.bulk.create.server
        study.flashcard.bulk.update.local
        study.flashcard.bulk.update.server
        study.flashcard.export.export.local
        study.flashcard.export.export.server
        study.flashcard.import.import.local
        study.flashcard.import.import.server
        study.flashcard.import.preview.local
        study.flashcard.import.preview.server
        study.flashcard.tags.list.local
        study.flashcard.tags.list.server
        study.flashcard.template.create.local
        study.flashcard.template.create.server
        study.flashcard.template.delete.local
        study.flashcard.template.delete.server
        study.flashcard.template.detail.local
        study.flashcard.template.detail.server
        study.flashcard.template.list.local
        study.flashcard.template.list.server
        study.flashcard.template.update.local
        study.flashcard.template.update.server
        study.guides.launch.local
        study.guides.launch.server
        study.guides.observe.local
        study.guides.observe.server
    """),
    "study_packs": _action_ids("""
        study.packs.jobs.launch.server
        study.packs.jobs.list.server
        study.packs.jobs.observe.server
    """),
    "study_suggestions": _action_ids("""
        study.suggestions.configure.server
        study.suggestions.launch.server
        study.suggestions.list.server
        study.suggestions.observe.server
    """),
    "watchlists": _action_ids("""
        watchlists.create.local
        watchlists.create.server
        watchlists.delete.local
        watchlists.delete.server
        watchlists.detail.local
        watchlists.detail.server
        watchlists.list.local
        watchlists.list.server
        watchlists.runs.detail.local
        watchlists.runs.detail.server
        watchlists.runs.launch.local
        watchlists.runs.launch.server
        watchlists.runs.list.local
        watchlists.runs.list.server
        watchlists.runs.observe.local
        watchlists.runs.observe.server
        watchlists.update.local
        watchlists.update.server
    """),
    "web_clipper": _action_ids("""
        web_clipper.capture.server
        web_clipper.list.server
        web_clipper.observe.server
        web_clipper.status.server
    """),
    "workflows": _action_ids("""
        workflows.launch.server
        workflows.list.server
        workflows.observe.server
    """),
    "writing_suite": _action_ids("""
        writing.chapters.create.local
        writing.chapters.create.server
        writing.chapters.delete.local
        writing.chapters.delete.server
        writing.chapters.detail.local
        writing.chapters.detail.server
        writing.chapters.list.local
        writing.chapters.list.server
        writing.chapters.update.local
        writing.chapters.update.server
        writing.manuscripts.create.local
        writing.manuscripts.create.server
        writing.manuscripts.delete.local
        writing.manuscripts.delete.server
        writing.manuscripts.detail.local
        writing.manuscripts.detail.server
        writing.manuscripts.list.local
        writing.manuscripts.list.server
        writing.manuscripts.update.local
        writing.manuscripts.update.server
        writing.projects.create.local
        writing.projects.create.server
        writing.projects.delete.local
        writing.projects.delete.server
        writing.projects.detail.local
        writing.projects.detail.server
        writing.projects.list.local
        writing.projects.list.server
        writing.projects.update.local
        writing.projects.update.server
        writing.scenes.create.local
        writing.scenes.create.server
        writing.scenes.delete.local
        writing.scenes.delete.server
        writing.scenes.detail.local
        writing.scenes.detail.server
        writing.scenes.list.local
        writing.scenes.list.server
        writing.scenes.update.local
        writing.scenes.update.server
    """),
}
def test_client_notifications_capability_exposes_queue_update_local():
    action_ids = CAPABILITY_ACTION_MATRIX["client_notifications"]
    assert "notifications.queue.update.local" in action_ids


def test_runtime_policy_registers_canonical_unified_mcp_action_ids():
    assert "mcp.runtime.list.local" in CAPABILITY_REGISTRY
    assert "mcp.inventory.list.local" in CAPABILITY_REGISTRY
    assert "mcp.catalogs.configure.server" in CAPABILITY_REGISTRY
    assert "mcp.external_servers.observe.server" in CAPABILITY_REGISTRY


def test_runtime_policy_preserves_coarse_mcp_aliases_during_rollout():
    assert "mcp.runtime.trigger.local" in CAPABILITY_REGISTRY
    assert "mcp.governance.launch.server" in CAPABILITY_REGISTRY


def _collect_registry_action_kinds_by_source(entries) -> dict[str, frozenset[str]]:
    action_kinds_by_source: dict[str, set[str]] = {}
    for entry in entries:
        kinds = action_kinds_by_source.setdefault(entry.required_source, set())
        kinds.add(entry.action_kind)

    return {source: frozenset(kinds) for source, kinds in action_kinds_by_source.items()}


def _group_registry_entries_by_capability():
    grouped_entries = defaultdict(list)
    for entry in CAPABILITY_REGISTRY.values():
        grouped_entries[entry.capability_id].append(entry)
    return grouped_entries


def test_runtime_source_state_downgrades_stale_server_signals_to_unknown():
    state = RuntimeSourceState(
        active_source="server",
        active_server_id="primary",
        server_configured=True,
        server_reachability="reachable",
        server_reachability_checked_at=datetime.now(timezone.utc) - timedelta(minutes=30),
        server_auth_state="authenticated",
        server_auth_checked_at=datetime.now(timezone.utc) - timedelta(minutes=30),
    )

    normalized = state.normalized_for_policy(
        now=datetime.now(timezone.utc),
        freshness_window=timedelta(minutes=5),
    )

    assert normalized.server_reachability == "unknown"
    assert normalized.server_auth_state == "unknown"


def test_runtime_source_state_from_dict_coerces_invalid_persisted_enum_values_to_safe_defaults():
    state = RuntimeSourceState.from_dict(
        {
            "active_source": "remote",
            "active_server_id": "primary",
            "server_configured": True,
            "server_reachability": "maybe",
            "server_auth_state": "logged_in",
            "last_known_server_label": "Primary",
        }
    )

    assert state.active_source == "local"
    assert state.server_reachability == "unknown"
    assert state.server_auth_state == "unknown"
    assert state.active_server_id == "primary"
    assert state.server_configured is True
    assert state.last_known_server_label == "Primary"


def test_runtime_source_state_from_dict_drops_malformed_timestamps():
    state = RuntimeSourceState.from_dict(
        {
            "server_reachability_checked_at": "not-a-timestamp",
            "server_auth_checked_at": "2026-13-99T25:61:00Z",
        }
    )

    assert state.server_reachability_checked_at is None
    assert state.server_auth_checked_at is None


def test_policy_engine_denies_remote_only_action_in_local_mode():
    engine = PolicyEngine(CAPABILITY_REGISTRY)
    decision = engine.evaluate(
        action_id="workflows.launch.server",
        state=RuntimeSourceState(active_source="local"),
    )

    assert decision.allowed is False
    assert decision.reason_code == "wrong_source"


def test_policy_engine_can_evaluate_against_explicit_runtime_state_override():
    engine = PolicyEngine(CAPABILITY_REGISTRY)
    authoritative_state = RuntimeSourceState(
        active_source="local",
        active_server_id="https://server.example.com/api",
        server_configured=True,
        server_reachability="reachable",
        server_auth_state="authenticated",
    )
    selected_destination_state = RuntimeSourceState(
        active_source="server",
        active_server_id="https://server.example.com/api",
        server_configured=True,
        server_reachability="reachable",
        server_auth_state="authenticated",
    )

    decision = engine.evaluate(
        action_id="mcp.inventory.list.server",
        state=authoritative_state,
        runtime_state_override=selected_destination_state,
    )

    assert decision.allowed is True
    assert decision.effective_source == "server"
    assert authoritative_state.active_source == "local"


def test_policy_engine_falls_back_to_explicit_runtime_state_override_when_state_is_omitted():
    engine = PolicyEngine(CAPABILITY_REGISTRY)
    decision = engine.evaluate(
        action_id="mcp.inventory.list.server",
        runtime_state_override=RuntimeSourceState(active_source="local"),
    )

    assert decision.allowed is False
    assert decision.reason_code == "wrong_source"
    assert decision.effective_source == "local"


def test_policy_engine_denies_unknown_action_ids_without_raising():
    engine = PolicyEngine(CAPABILITY_REGISTRY)

    decision = engine.evaluate(
        action_id="runtime.policy.unknown.server",
        state=RuntimeSourceState(active_source="server"),
    )

    assert decision.allowed is False
    assert decision.reason_code == "authority_denied"
    assert decision.authority_owner == "unknown"


def test_runtime_policy_registry_contains_full_audited_rows():
    expected_capability_ids = set(EXPECTED_AUDITED_CAPABILITIES)
    actual_entries_by_capability = _group_registry_entries_by_capability()

    assert set(actual_entries_by_capability) == expected_capability_ids, (
        "Audited capability seed coverage must match the full 26-row parity matrix."
    )

    for capability_id, expected in EXPECTED_AUDITED_CAPABILITIES.items():
        entries = actual_entries_by_capability[capability_id]
        assert {entry.domain_id for entry in entries} == expected["expected_domain_ids"]
        assert {entry.action_id for entry in entries} == EXPECTED_ACTION_IDS_BY_CAPABILITY[capability_id]

        actual_action_kinds_by_source = _collect_registry_action_kinds_by_source(entries)
        assert set(actual_action_kinds_by_source) == set(expected["expected_action_kinds_by_source"])

        for source, expected_action_kinds in expected["expected_action_kinds_by_source"].items():
            assert expected_action_kinds.issubset(actual_action_kinds_by_source[source]), (
                f"{capability_id} is missing audited action kinds for {source}: "
                f"{sorted(expected_action_kinds.difference(actual_action_kinds_by_source[source]))}"
            )

def test_backend_exception_classifier_prefers_loaded_backend_exception_classes(monkeypatch):
    from tldw_chatbook.runtime_policy import enforcement

    backend_auth_error_type = _backend_exception_type(
        "AuthenticationError",
        module_name="unexpected.module.name",
    )
    monkeypatch.setattr(
        enforcement,
        "_load_backend_exception_classes",
        lambda module_importer=None: {"AuthenticationError": backend_auth_error_type},
    )

    error = backend_auth_error_type("bad credentials")
    error.response_data = {}

    assert classify_backend_exception(error) == "server_auth_required"


def test_service_policy_enforcer_fails_closed_when_runtime_state_is_missing():
    enforcer = ServicePolicyEnforcer(state_provider=lambda: None)

    with pytest.raises(PolicyDeniedError) as exc:
        enforcer.require_allowed(action_id="notes.create.local")

    assert exc.value.reason_code == "authority_denied"
    assert exc.value.effective_source == "unknown"


def test_backend_exception_classifier_handles_session_invalid_authentication_errors():
    error = _backend_auth_error(
        message="session expired",
        response_data={"code": "session_invalid", "detail": "Session expired"},
    )

    assert classify_backend_exception(error) == "server_session_invalid"


def test_backend_exception_classifier_ignores_same_named_exceptions_from_unrelated_modules():
    error = _backend_auth_error(
        module_name="different.vendor.exceptions",
        message="bad credentials",
    )

    assert classify_backend_exception(error) is None


def test_backend_exception_classifier_handles_session_invalid_401s():
    error = _backend_response_error(
        401,
        "Unauthorized",
        response_data={"detail": "Session invalid. Please sign in again."},
    )

    assert classify_backend_exception(error) == "server_session_invalid"
