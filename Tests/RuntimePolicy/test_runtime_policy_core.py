from collections import defaultdict
from datetime import datetime, timedelta, timezone

from tldw_chatbook.runtime_policy.engine import PolicyEngine
import pytest

from tldw_chatbook.runtime_policy.enforcement import ServicePolicyEnforcer, classify_backend_exception
from tldw_chatbook.runtime_policy.registry import CAPABILITY_REGISTRY
from tldw_chatbook.runtime_policy.types import PolicyDeniedError, RuntimeSourceState
from tldw_chatbook.tldw_api.exceptions import APIResponseError, AuthenticationError


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
CONNECTOR_ACTIONS = _action_kinds("browse", "create", "update", "delete", "launch", "observe")


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
            "local": FULL_CRUD_AND_LAUNCH_AND_OBSERVE,
            "server": FULL_CRUD_AND_LAUNCH_AND_OBSERVE,
        },
    },
    "companion_personalization": {
        "expected_domain_ids": {"companion"},
        "expected_action_kinds_by_source": {
            "server": _action_kinds("browse", "detail", "create", "update", "delete", "launch"),
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
    "collections_feed_subscriptions": {
        "expected_domain_ids": {"collections_feeds"},
        "expected_action_kinds_by_source": {
            "local": FULL_CRUD,
            "server": FULL_CRUD,
        },
    },
    "collections_outputs_templates_artifacts": {
        "expected_domain_ids": {"outputs"},
        "expected_action_kinds_by_source": {
            "local": FULL_CRUD_AND_LAUNCH_AND_OBSERVE,
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
            "local": DISCOVER_CONFIGURE_TRIGGER_OBSERVE,
            "server": DISCOVER_CONFIGURE_TRIGGER_OBSERVE,
        },
    },
    "chat_grammars": {
        "expected_domain_ids": {"chat_grammars"},
        "expected_action_kinds_by_source": {
            "local": FULL_CRUD,
            "server": FULL_CRUD,
        },
    },
    "explicit_feedback": {
        "expected_domain_ids": {"feedback"},
        "expected_action_kinds_by_source": {
            "local": FULL_CRUD,
            "server": FULL_CRUD,
        },
    },
    "claims_notifications_alerts": {
        "expected_domain_ids": {"claims"},
        "expected_action_kinds_by_source": {
            "server": FULL_CRUD_AND_LAUNCH,
        },
    },
    "meetings": {
        "expected_domain_ids": {"meetings"},
        "expected_action_kinds_by_source": {
            "server": _action_kinds("browse", "detail", "create", "update", "launch", "observe"),
        },
    },
    "prompt_studio": {
        "expected_domain_ids": {"prompt_studio"},
        "expected_action_kinds_by_source": {
            "server": _action_kinds("browse", "detail", "create", "update", "delete", "launch", "observe"),
        },
    },
    "kanban_boards_tasks": {
        "expected_domain_ids": {"kanban"},
        "expected_action_kinds_by_source": {
            "server": _action_kinds("browse", "detail", "create", "update", "delete", "launch", "observe"),
        },
    },
    "translation_utility": {
        "expected_domain_ids": {"translation"},
        "expected_action_kinds_by_source": {
            "server": _action_kinds("launch"),
        },
    },
    "client_notifications": {
        "expected_domain_ids": {"notifications"},
        "expected_action_kinds_by_source": {
            "local": DISCOVER_CONFIGURE_TRIGGER_OBSERVE,
        },
    },
    "server_runtime_config_discovery": {
        "expected_domain_ids": {"server_runtime"},
        "expected_action_kinds_by_source": {
            "server": _action_kinds("browse", "update", "launch", "observe"),
        },
    },
    "llm_provider_model_catalog": {
        "expected_domain_ids": {"llm_catalog"},
        "expected_action_kinds_by_source": {
            "local": _action_kinds("browse", "detail", "update", "observe"),
            "server": _action_kinds("browse", "detail", "update", "observe"),
        },
    },
    "audio_speech_services": {
        "expected_domain_ids": {"audio", "audiobooks"},
        "expected_action_kinds_by_source": {
            "local": FULL_CRUD_AND_LAUNCH_AND_OBSERVE,
            "server": FULL_CRUD_AND_LAUNCH_AND_OBSERVE,
        },
    },
    "voice_assistant": {
        "expected_domain_ids": {"voice_assistant"},
        "expected_action_kinds_by_source": {
            "server": _action_kinds("browse", "detail", "create", "update", "delete", "launch", "observe"),
        },
    },
    "auth_profile_sessions": {
        "expected_domain_ids": {"auth"},
        "expected_action_kinds_by_source": {
            "server": FULL_CRUD_AND_LAUNCH,
        },
    },
    "server_reminders_notification_feeds": {
        "expected_domain_ids": {"notifications_server"},
        "expected_action_kinds_by_source": {
            "server": DISCOVER_CONFIGURE_TRIGGER_OBSERVE,
        },
    },
    "external_connectors": {
        "expected_domain_ids": {"connectors"},
        "expected_action_kinds_by_source": {
            "server": CONNECTOR_ACTIONS,
        },
    },
    "server_skills": {
        "expected_domain_ids": {"skills"},
        "expected_action_kinds_by_source": {
            "server": FULL_CRUD_AND_LAUNCH,
        },
    },
    "server_tools": {
        "expected_domain_ids": {"tools"},
        "expected_action_kinds_by_source": {
            "server": _action_kinds("browse", "launch"),
        },
    },
    "text2sql_query": {
        "expected_domain_ids": {"text2sql"},
        "expected_action_kinds_by_source": {
            "server": _action_kinds("launch"),
        },
    },
    "sync_transport": {
        "expected_domain_ids": {"sync"},
        "expected_action_kinds_by_source": {
            "server": _action_kinds("launch", "observe"),
        },
    },
    "user_governance": {
        "expected_domain_ids": {"user_governance"},
        "expected_action_kinds_by_source": {
            "server": _action_kinds("browse", "detail", "update"),
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
        "expected_domain_ids": {"mcp_runtime"},
        "expected_action_kinds_by_source": {
            "local": DISCOVER_CONFIGURE_TRIGGER_OBSERVE,
        },
    },
    "remote_mcp_control_plane_governance": {
        "expected_domain_ids": {"mcp_governance"},
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
        character.archetypes.detail.local
        character.archetypes.detail.server
        character.archetypes.list.local
        character.archetypes.list.server
        character.archetypes.preview.local
        character.archetypes.preview.server
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
        character.sessions.create.local
        character.sessions.create.server
        character.sessions.delete.local
        character.sessions.delete.server
        character.sessions.detail.local
        character.sessions.detail.server
        character.sessions.export.local
        character.sessions.export.server
        character.sessions.launch.local
        character.sessions.launch.server
        character.sessions.list.local
        character.sessions.list.server
        character.sessions.observe.local
        character.sessions.observe.server
        character.sessions.restore.local
        character.sessions.restore.server
        character.sessions.update.local
        character.sessions.update.server
        character.messages.create.server
        character.messages.delete.server
        character.messages.detail.server
        character.messages.list.server
        character.messages.update.server
        chat.dictionaries.create.local
        chat.dictionaries.create.server
        chat.dictionaries.delete.local
        chat.dictionaries.delete.server
        chat.dictionaries.detail.local
        chat.dictionaries.detail.server
        chat.dictionaries.export.local
        chat.dictionaries.export.server
        chat.dictionaries.import.local
        chat.dictionaries.import.server
        chat.dictionaries.list.local
        chat.dictionaries.list.server
        chat.dictionaries.process.local
        chat.dictionaries.process.server
        chat.dictionaries.update.local
        chat.dictionaries.update.server
        chat.dictionary.activity.list.local
        chat.dictionary.activity.list.server
        chat.dictionary.entries.create.local
        chat.dictionary.entries.create.server
        chat.dictionary.entries.delete.local
        chat.dictionary.entries.delete.server
        chat.dictionary.entries.list.local
        chat.dictionary.entries.list.server
        chat.dictionary.entries.reorder.local
        chat.dictionary.entries.reorder.server
        chat.dictionary.entries.update.local
        chat.dictionary.entries.update.server
        chat.dictionary.statistics.detail.local
        chat.dictionary.statistics.detail.server
        chat.dictionary.versions.detail.local
        chat.dictionary.versions.detail.server
        chat.dictionary.versions.list.local
        chat.dictionary.versions.list.server
        chat.dictionary.versions.restore.local
        chat.dictionary.versions.restore.server
    """),
    "companion_personalization": _action_ids("""
        companion.activity.create.server
        companion.activity.detail.server
        companion.activity.list.server
        companion.checkins.create.server
        companion.conversation_prompts.list.server
        companion.goals.create.server
        companion.goals.list.server
        companion.goals.update.server
        companion.knowledge.detail.server
        companion.knowledge.list.server
        companion.lifecycle.launch.server
        companion.lifecycle.purge.server
        companion.reflections.detail.server
    """),
    "chat": _action_ids("""
        chat.create.local
        chat.create.server
        chat.delete.local
        chat.delete.server
        chat.analytics.observe.local
        chat.analytics.observe.server
        chat.commands.list.local
        chat.commands.list.server
        chat.detail.local
        chat.detail.server
        chat.knowledge.create.local
        chat.knowledge.create.server
        chat.launch.local
        chat.launch.server
        chat.list.local
        chat.list.server
        chat.loop.approve.local
        chat.loop.approve.server
        chat.loop.cancel.local
        chat.loop.cancel.server
        chat.loop.launch.local
        chat.loop.launch.server
        chat.loop.observe.local
        chat.loop.observe.server
        chat.share_links.create.local
        chat.share_links.create.server
        chat.share_links.detail.local
        chat.share_links.detail.server
        chat.share_links.list.local
        chat.share_links.list.server
        chat.share_links.revoke.local
        chat.share_links.revoke.server
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
        notifications.queue.list.local
        notifications.queue.observe.local
        notifications.queue.update.local
        notifications.settings.list.local
        notifications.settings.update.local
    """),
    "collections_outputs_templates_artifacts": _action_ids("""
        outputs.artifacts.create.local
        outputs.artifacts.create.server
        outputs.artifacts.delete.local
        outputs.artifacts.delete.server
        outputs.artifacts.detail.local
        outputs.artifacts.detail.server
        outputs.artifacts.list.local
        outputs.artifacts.list.server
        outputs.artifacts.update.local
        outputs.artifacts.update.server
        outputs.render_jobs.detail.local
        outputs.render_jobs.detail.server
        outputs.render_jobs.launch.local
        outputs.render_jobs.launch.server
        outputs.render_jobs.list.local
        outputs.render_jobs.list.server
        outputs.render_jobs.observe.local
        outputs.render_jobs.observe.server
        outputs.templates.create.local
        outputs.templates.create.server
        outputs.templates.delete.local
        outputs.templates.delete.server
        outputs.templates.detail.local
        outputs.templates.detail.server
        outputs.templates.list.local
        outputs.templates.list.server
        outputs.templates.update.local
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
    "collections_feed_subscriptions": _action_ids("""
        collections.feeds.create.local
        collections.feeds.create.server
        collections.feeds.delete.local
        collections.feeds.delete.server
        collections.feeds.detail.local
        collections.feeds.detail.server
        collections.feeds.list.local
        collections.feeds.list.server
        collections.feeds.update.local
        collections.feeds.update.server
        collections.feeds.websub.delete.server
        collections.feeds.websub.detail.server
        collections.feeds.websub.launch.server
    """),
    "external_connectors": _action_ids("""
        connectors.accounts.delete.server
        connectors.accounts.list.server
        connectors.jobs.observe.server
        connectors.providers.launch.server
        connectors.providers.list.server
        connectors.sources.create.server
        connectors.sources.launch.server
        connectors.sources.list.server
        connectors.sources.observe.server
        connectors.sources.update.server
    """),
    "chat_grammars": _action_ids("""
        chat.grammars.create.local
        chat.grammars.create.server
        chat.grammars.delete.local
        chat.grammars.delete.server
        chat.grammars.detail.local
        chat.grammars.detail.server
        chat.grammars.list.local
        chat.grammars.list.server
        chat.grammars.update.local
        chat.grammars.update.server
    """),
    "explicit_feedback": _action_ids("""
        feedback.create.local
        feedback.create.server
        feedback.delete.local
        feedback.delete.server
        feedback.detail.local
        feedback.detail.server
        feedback.list.local
        feedback.list.server
        feedback.update.local
        feedback.update.server
    """),
    "claims_notifications_alerts": _action_ids("""
        claims.alerts.create.server
        claims.alerts.delete.server
        claims.alerts.launch.server
        claims.alerts.list.server
        claims.alerts.update.server
        claims.analytics.detail.server
        claims.analytics.export.server
        claims.analytics.list.server
        claims.cluster_evidence.list.server
        claims.cluster_links.create.server
        claims.cluster_links.delete.server
        claims.cluster_links.list.server
        claims.cluster_members.list.server
        claims.cluster_timeline.list.server
        claims.clusters.detail.server
        claims.clusters.launch.server
        claims.clusters.list.server
        claims.extractors.list.server
        claims.fva.launch.server
        claims.fva.list.server
        claims.items.detail.server
        claims.items.launch.server
        claims.items.list.server
        claims.items.update.server
        claims.monitoring.list.server
        claims.monitoring.update.server
        claims.notifications.launch.server
        claims.notifications.list.server
        claims.notifications.update.server
        claims.rebuild.detail.server
        claims.rebuild.launch.server
        claims.review.launch.server
        claims.review.list.server
        claims.review.update.server
        claims.review_rules.create.server
        claims.review_rules.delete.server
        claims.review_rules.list.server
        claims.review_rules.update.server
        claims.search.list.server
        claims.settings.list.server
        claims.settings.update.server
        claims.status.detail.server
    """),
    "meetings": _action_ids("""
        meetings.artifacts.create.server
        meetings.artifacts.list.server
        meetings.events.observe.server
        meetings.health.detail.server
        meetings.sessions.create.server
        meetings.sessions.detail.server
        meetings.sessions.launch.server
        meetings.sessions.list.server
        meetings.sessions.update.server
        meetings.share.launch.server
        meetings.templates.create.server
        meetings.templates.detail.server
        meetings.templates.list.server
    """),
    "prompt_studio": _action_ids("""
        prompt_studio.events.observe.server
        prompt_studio.evaluations.create.server
        prompt_studio.evaluations.delete.server
        prompt_studio.evaluations.detail.server
        prompt_studio.evaluations.list.server
        prompt_studio.optimization_iterations.create.server
        prompt_studio.optimization_iterations.list.server
        prompt_studio.optimization_strategies.launch.server
        prompt_studio.optimization_strategies.list.server
        prompt_studio.optimizations.cancel.server
        prompt_studio.optimizations.create.server
        prompt_studio.optimizations.detail.server
        prompt_studio.optimizations.launch.server
        prompt_studio.optimizations.list.server
        prompt_studio.project_stats.detail.server
        prompt_studio.projects.archive.server
        prompt_studio.projects.create.server
        prompt_studio.projects.delete.server
        prompt_studio.projects.detail.server
        prompt_studio.projects.list.server
        prompt_studio.projects.restore.server
        prompt_studio.projects.update.server
        prompt_studio.prompt_versions.list.server
        prompt_studio.prompts.create.server
        prompt_studio.prompts.detail.server
        prompt_studio.prompts.launch.server
        prompt_studio.prompts.list.server
        prompt_studio.prompts.preview.server
        prompt_studio.prompts.process.server
        prompt_studio.prompts.restore.server
        prompt_studio.prompts.update.server
        prompt_studio.status.detail.server
        prompt_studio.test_cases.create.server
        prompt_studio.test_cases.delete.server
        prompt_studio.test_cases.detail.server
        prompt_studio.test_cases.export.server
        prompt_studio.test_cases.import.server
        prompt_studio.test_cases.launch.server
        prompt_studio.test_cases.list.server
        prompt_studio.test_cases.update.server
    """),
    "kanban_boards_tasks": _action_ids("""
        kanban.activities.list.server
        kanban.boards.archive.server
        kanban.boards.create.server
        kanban.boards.delete.server
        kanban.boards.detail.server
        kanban.boards.export.server
        kanban.boards.import.server
        kanban.boards.list.server
        kanban.boards.restore.server
        kanban.boards.update.server
        kanban.card_labels.create.server
        kanban.card_labels.delete.server
        kanban.card_labels.list.server
        kanban.card_links.create.server
        kanban.card_links.delete.server
        kanban.card_links.detail.server
        kanban.card_links.list.server
        kanban.cards.archive.server
        kanban.cards.create.server
        kanban.cards.delete.server
        kanban.cards.detail.server
        kanban.cards.launch.server
        kanban.cards.list.server
        kanban.cards.reorder.server
        kanban.cards.restore.server
        kanban.cards.update.server
        kanban.checklist_items.create.server
        kanban.checklist_items.delete.server
        kanban.checklist_items.detail.server
        kanban.checklist_items.list.server
        kanban.checklist_items.reorder.server
        kanban.checklist_items.update.server
        kanban.checklists.create.server
        kanban.checklists.delete.server
        kanban.checklists.detail.server
        kanban.checklists.list.server
        kanban.checklists.reorder.server
        kanban.checklists.update.server
        kanban.comments.create.server
        kanban.comments.delete.server
        kanban.comments.detail.server
        kanban.comments.list.server
        kanban.comments.update.server
        kanban.labels.create.server
        kanban.labels.delete.server
        kanban.labels.detail.server
        kanban.labels.list.server
        kanban.labels.update.server
        kanban.lists.archive.server
        kanban.lists.create.server
        kanban.lists.delete.server
        kanban.lists.detail.server
        kanban.lists.list.server
        kanban.lists.reorder.server
        kanban.lists.restore.server
        kanban.lists.update.server
        kanban.search.detail.server
        kanban.search.list.server
    """),
    "translation_utility": _action_ids("""
        translation.text.launch.server
    """),
    "server_skills": _action_ids("""
        skills.context.list.server
        skills.create.server
        skills.delete.server
        skills.detail.server
        skills.execute.launch.server
        skills.export.launch.server
        skills.import.launch.server
        skills.list.server
        skills.seed.launch.server
        skills.update.server
    """),
    "server_tools": _action_ids("""
        tools.catalog.list.server
        tools.execution.launch.server
    """),
    "text2sql_query": _action_ids("""
        text2sql.query.launch.server
    """),
    "sync_transport": _action_ids("""
        sync.changes.launch.server
        sync.changes.observe.server
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
        evaluations.dataset.update.server
        evaluations.benchmarks.detail.server
        evaluations.benchmarks.launch.server
        evaluations.benchmarks.list.server
        evaluations.embeddings_abtest.create.server
        evaluations.embeddings_abtest.delete.server
        evaluations.embeddings_abtest.detail.server
        evaluations.embeddings_abtest.export.server
        evaluations.embeddings_abtest.launch.server
        evaluations.embeddings_abtest.observe.server
        evaluations.webhooks.create.server
        evaluations.webhooks.delete.server
        evaluations.webhooks.launch.server
        evaluations.webhooks.list.server
        evaluations.recipes.detail.server
        evaluations.recipes.launch.server
        evaluations.recipes.list.server
        evaluations.recipes.observe.server
        evaluations.rag_pipeline.create.server
        evaluations.rag_pipeline.delete.server
        evaluations.rag_pipeline.detail.server
        evaluations.rag_pipeline.launch.server
        evaluations.rag_pipeline.list.server
        evaluations.rag_pipeline.update.server
        evaluations.synthetic.create.server
        evaluations.synthetic.launch.server
        evaluations.synthetic.list.server
        evaluations.synthetic.update.server
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
        mcp.runtime.configure.local
        mcp.runtime.launch.local
        mcp.runtime.list.local
        mcp.runtime.observe.local
        mcp.runtime.trigger.local
    """),
    "media_reading_ingestion_sources": _action_ids("""
        media.ingestion_jobs.cancel.local
        media.ingestion_jobs.cancel.server
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
        media.ingestion_sources.delete.server
        media.ingestion_sources.detail.local
        media.ingestion_sources.detail.server
        media.ingestion_sources.list.local
        media.ingestion_sources.list.server
        media.ingestion_sources.update.local
        media.ingestion_sources.update.server
        media.ingestion_source_items.reattach.local
        media.ingestion_source_items.reattach.server
        media.add.create.local
        media.add.create.server
        media.file_artifacts.create.local
        media.file_artifacts.create.server
        media.file_artifacts.delete.local
        media.file_artifacts.delete.server
        media.file_artifacts.detail.local
        media.file_artifacts.detail.server
        media.file_artifacts.export.local
        media.file_artifacts.export.server
        media.file_artifacts.purge.local
        media.file_artifacts.purge.server
        media.items.delete.local
        media.items.delete.server
        media.items.detail.local
        media.items.detail.server
        media.items.file.detail.local
        media.items.file.detail.server
        media.items.identifier_lookup.detail.local
        media.items.identifier_lookup.detail.server
        media.items.keywords.list.local
        media.items.keywords.list.server
        media.items.keywords.update.local
        media.items.keywords.update.server
        media.items.list.local
        media.items.list.server
        media.items.metadata_search.list.local
        media.items.metadata_search.list.server
        media.items.permanent.delete.local
        media.items.permanent.delete.server
        media.items.restore.local
        media.items.restore.server
        media.items.trash.delete.local
        media.items.trash.delete.server
        media.items.trash.list.local
        media.items.trash.list.server
        media.items.update.local
        media.items.update.server
        media.navigation.detail.local
        media.navigation.detail.server
        media.processing.audio.process.local
        media.processing.audio.process.server
        media.processing.code.process.local
        media.processing.code.process.server
        media.processing.document.process.local
        media.processing.document.process.server
        media.processing.ebook.process.local
        media.processing.ebook.process.server
        media.processing.emails.process.local
        media.processing.emails.process.server
        media.processing.mediawiki.import.local
        media.processing.mediawiki.process.local
        media.processing.mediawiki.import.server
        media.processing.mediawiki.process.server
        media.processing.pdf.process.local
        media.processing.pdf.process.server
        media.processing.plaintext.process.local
        media.processing.plaintext.process.server
        media.processing.video.process.local
        media.processing.video.process.server
        media.processing.web_scraping.process.local
        media.processing.web_scraping.process.server
        media.transcription_models.list.server
        media.web_scraping.cancel.server
        media.web_scraping.cookies.detail.server
        media.web_scraping.cookies.update.server
        media.web_scraping.detail.server
        media.web_scraping.inspect.server
        media.web_scraping.observe.server
        media.web_scraping.service.initialize.server
        media.web_scraping.service.shutdown.server
        media.web_scraping.status.server
        media.web_content_ingest.launch.server
        media.reading.digest_outputs.list.local
        media.reading.digest_outputs.list.server
        media.reading.digest_scheduler.trigger.local
        media.reading.digest_schedules.create.local
        media.reading.digest_schedules.create.server
        media.reading.digest_schedules.delete.local
        media.reading.digest_schedules.delete.server
        media.reading.digest_schedules.detail.local
        media.reading.digest_schedules.detail.server
        media.reading.digest_schedules.list.local
        media.reading.digest_schedules.list.server
        media.reading.digest_schedules.update.local
        media.reading.digest_schedules.update.server
        media.reading.create.local
        media.reading.create.server
        media.reading.archive.local
        media.reading.archive.server
        media.reading.bulk_update.local
        media.reading.bulk_update.server
        media.reading.delete.local
        media.reading.delete.server
        media.reading.detail.local
        media.reading.detail.server
        media.reading.export.local
        media.reading.export.server
        media.reading.import.local
        media.reading.import.server
        media.reading.list.local
        media.reading.list.server
        media.reading.summarize.local
        media.reading.summarize.server
        media.reading.tts.local
        media.reading.tts.server
        media.reading_import_jobs.detail.local
        media.reading_import_jobs.detail.server
        media.reading_import_jobs.list.local
        media.reading_import_jobs.list.server
        media.reading.note_links.create.local
        media.reading.note_links.create.server
        media.reading.note_links.delete.local
        media.reading.note_links.delete.server
        media.reading.note_links.list.local
        media.reading.note_links.list.server
        media.reading.saved_searches.create.local
        media.reading.saved_searches.create.server
        media.reading.saved_searches.delete.local
        media.reading.saved_searches.delete.server
        media.reading.saved_searches.detail.local
        media.reading.saved_searches.detail.server
        media.reading.saved_searches.list.local
        media.reading.saved_searches.list.server
        media.reading.saved_searches.update.local
        media.reading.saved_searches.update.server
        media.reading.update.local
        media.reading.update.server
        media.reading_progress.detail.local
        media.reading_progress.detail.server
        media.reading_progress.update.local
        media.reading_progress.update.server
        media.reference_images.list.local
        media.reference_images.list.server
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
        notes.graph.create.server
        notes.graph.delete.server
        notes.graph.detail.server
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
        chatbooks.import.local
        chatbooks.import.server
        chatbooks.list.local
        chatbooks.list.server
        chatbooks.update.local
        chatbooks.update.server
        prompts.create.local
        prompts.create.server
        prompts.bulk.delete.server
        prompts.bulk.update.server
        prompts.collections.create.server
        prompts.collections.detail.server
        prompts.collections.list.server
        prompts.collections.update.server
        prompts.delete.local
        prompts.delete.server
        prompts.health.detail.server
        prompts.keywords.create.server
        prompts.keywords.delete.server
        prompts.keywords.export.server
        prompts.keywords.list.server
        prompts.list.local
        prompts.list.server
        prompts.preview.local
        prompts.preview.server
        prompts.search.list.server
        prompts.sync_log.list.server
        prompts.templates.process.server
        prompts.transfer.export.server
        prompts.transfer.import.server
        prompts.update.local
        prompts.update.server
        prompts.usage.update.server
        prompts.versions.list.local
        prompts.versions.list.server
        prompts.versions.restore.local
        prompts.versions.restore.server
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
        rag.media_embedding_jobs.detail.server
        rag.media_embedding_jobs.list.server
        rag.media_embeddings.create.server
        rag.media_embeddings.delete.server
        rag.media_embeddings.search.server
        rag.media_embeddings.status.server
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
        mcp.governance.approve.server
        mcp.governance.configure.server
        mcp.governance.launch.server
        mcp.governance.list.server
        mcp.governance.observe.server
    """),
    "research_search_provider_surfaces": _action_ids("""
        research.search.providers.configure.local
        research.search.providers.configure.server
        research.search.providers.launch.local
        research.search.providers.launch.server
        research.search.providers.list.local
        research.search.providers.list.server
        research.search.providers.observe.local
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
        notifications.feed.list.server
        notifications.feed.observe.server
        notifications.feed.update.server
        notifications.reminders.configure.server
        notifications.reminders.launch.server
        notifications.reminders.list.server
        notifications.reminders.observe.server
    """),
    "server_runtime_config_discovery": _action_ids("""
        server.runtime.config.list.server
        server.runtime.config.update.server
        server.runtime.health.list.server
        server.runtime.health.observe.server
        server.runtime.providers.list.server
        server.runtime.providers.validate.server
    """),
    "llm_provider_model_catalog": _action_ids("""
        llm.catalog.health.observe.local
        llm.catalog.health.observe.server
        llm.catalog.models.detail.local
        llm.catalog.models.detail.server
        llm.catalog.models.list.local
        llm.catalog.models.list.server
        llm.catalog.providers.configure.local
        llm.catalog.providers.configure.server
        llm.catalog.providers.detail.local
        llm.catalog.providers.detail.server
        llm.catalog.providers.list.local
        llm.catalog.providers.list.server
    """),
    "audio_speech_services": _action_ids("""
        audio.health.observe.local
        audio.health.observe.server
        audio.history.delete.local
        audio.history.delete.server
        audio.history.detail.local
        audio.history.detail.server
        audio.history.list.local
        audio.history.list.server
        audio.history.update.local
        audio.history.update.server
        audio.jobs.create.local
        audio.jobs.create.server
        audio.jobs.detail.local
        audio.jobs.detail.server
        audio.jobs.observe.local
        audio.jobs.observe.server
        audio.providers.list.local
        audio.providers.list.server
        audio.speech.launch.local
        audio.speech.launch.server
        audio.speech_chat.launch.server
        audio.speech_jobs.detail.local
        audio.speech_jobs.detail.server
        audio.streaming.detail.server
        audio.streaming.launch.server
        audio.streaming.status.server
        audio.tokenizer.launch.local
        audio.tokenizer.launch.server
        audio.transcriptions.launch.local
        audio.transcriptions.launch.server
        audio.translations.launch.local
        audio.translations.launch.server
        audio.voices.create.local
        audio.voices.create.server
        audio.voices.delete.local
        audio.voices.delete.server
        audio.voices.detail.local
        audio.voices.detail.server
        audio.voices.launch.local
        audio.voices.launch.server
        audio.voices.list.local
        audio.voices.list.server
        audio.voices.preview.local
        audio.voices.preview.server
        audiobooks.artifacts.list.local
        audiobooks.artifacts.list.server
        audiobooks.chapters.list.local
        audiobooks.chapters.list.server
        audiobooks.jobs.create.local
        audiobooks.jobs.create.server
        audiobooks.jobs.detail.local
        audiobooks.jobs.detail.server
        audiobooks.jobs.observe.local
        audiobooks.jobs.observe.server
        audiobooks.parse.launch.local
        audiobooks.parse.launch.server
        audiobooks.projects.detail.local
        audiobooks.projects.detail.server
        audiobooks.projects.list.local
        audiobooks.projects.list.server
        audiobooks.subtitles.export.local
        audiobooks.subtitles.export.server
        audiobooks.voice_profiles.create.local
        audiobooks.voice_profiles.create.server
        audiobooks.voice_profiles.delete.local
        audiobooks.voice_profiles.delete.server
        audiobooks.voice_profiles.list.local
        audiobooks.voice_profiles.list.server
    """),
    "voice_assistant": _action_ids("""
        voice_assistant.analytics.observe.server
        voice_assistant.commands.create.server
        voice_assistant.commands.delete.server
        voice_assistant.commands.detail.server
        voice_assistant.commands.launch.server
        voice_assistant.commands.list.server
        voice_assistant.commands.observe.server
        voice_assistant.commands.preview.server
        voice_assistant.commands.update.server
        voice_assistant.sessions.delete.server
        voice_assistant.sessions.detail.server
        voice_assistant.sessions.list.server
    """),
    "auth_profile_sessions": _action_ids("""
        auth.api_keys.create.server
        auth.api_keys.delete.server
        auth.api_keys.list.server
        auth.api_keys.update.server
        auth.identity.delete.server
        auth.identity.launch.server
        auth.identity.update.server
        auth.profile.detail.server
        auth.profile.list.server
        auth.profile.update.server
        auth.provider_keys.create.server
        auth.provider_keys.delete.server
        auth.provider_keys.detail.server
        auth.provider_keys.list.server
        auth.provider_keys.update.server
        auth.provider_keys.validate.server
        auth.registration.create.server
        auth.security.launch.server
        auth.security.update.server
        auth.sessions.delete.server
        auth.sessions.list.server
        auth.storage.create.server
        auth.storage.delete.server
        auth.storage.detail.server
        auth.storage.export.server
        auth.storage.list.server
        auth.storage.update.server
    """),
    "user_governance": _action_ids("""
        user_governance.consent.list.server
        user_governance.consent.update.server
        user_governance.privileges.detail.server
        user_governance.privileges.list.server
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
        study.flashcard.analytics.observe.local
        study.flashcard.analytics.observe.server
        study.flashcard.assistant.detail.local
        study.flashcard.assistant.detail.server
        study.flashcard.assistant.launch.local
        study.flashcard.assistant.launch.server
        study.flashcard.assets.create.local
        study.flashcard.assets.create.server
        study.flashcard.assets.detail.local
        study.flashcard.assets.detail.server
        study.flashcard.bulk.create.local
        study.flashcard.bulk.create.server
        study.flashcard.bulk.update.local
        study.flashcard.bulk.update.server
        study.flashcard.create.local
        study.flashcard.create.server
        study.flashcard.delete.local
        study.flashcard.delete.server
        study.flashcard.detail.local
        study.flashcard.detail.server
        study.flashcard.generation.launch.local
        study.flashcard.generation.launch.server
        study.flashcard.export.export.local
        study.flashcard.export.export.server
        study.flashcard.import.import.local
        study.flashcard.import.import.server
        study.flashcard.import.preview.local
        study.flashcard.import.preview.server
        study.flashcard.list.local
        study.flashcard.list.server
        study.flashcard.review_sessions.list.local
        study.flashcard.review_sessions.list.server
        study.flashcard.review_sessions.observe.local
        study.flashcard.review_sessions.observe.server
        study.flashcard.tags.list.local
        study.flashcard.tags.list.server
        study.flashcard.tags.update.local
        study.flashcard.tags.update.server
        study.flashcard.templates.create.local
        study.flashcard.templates.create.server
        study.flashcard.templates.delete.local
        study.flashcard.templates.delete.server
        study.flashcard.templates.detail.local
        study.flashcard.templates.detail.server
        study.flashcard.templates.list.local
        study.flashcard.templates.list.server
        study.flashcard.templates.update.local
        study.flashcard.templates.update.server
        study.flashcard.update.local
        study.flashcard.update.server
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
        watchlists.alert_rules.create.local
        watchlists.alert_rules.create.server
        watchlists.alert_rules.delete.local
        watchlists.alert_rules.delete.server
        watchlists.alert_rules.detail.local
        watchlists.alert_rules.detail.server
        watchlists.alert_rules.list.local
        watchlists.alert_rules.list.server
        watchlists.alert_rules.update.local
        watchlists.alert_rules.update.server
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
        writing.analysis.launch.local
        writing.analysis.launch.server
        writing.analysis.list.local
        writing.analysis.list.server
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
        writing.characters.create.local
        writing.characters.create.server
        writing.characters.delete.local
        writing.characters.delete.server
        writing.characters.detail.local
        writing.characters.detail.server
        writing.characters.list.local
        writing.characters.list.server
        writing.characters.update.local
        writing.characters.update.server
        writing.citations.create.local
        writing.citations.create.server
        writing.citations.delete.local
        writing.citations.delete.server
        writing.citations.list.local
        writing.citations.list.server
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
        writing.outline.reorder.local
        writing.outline.reorder.server
        writing.plot_events.create.local
        writing.plot_events.create.server
        writing.plot_events.delete.local
        writing.plot_events.delete.server
        writing.plot_events.list.local
        writing.plot_events.list.server
        writing.plot_events.update.local
        writing.plot_events.update.server
        writing.plot_holes.create.local
        writing.plot_holes.create.server
        writing.plot_holes.delete.local
        writing.plot_holes.delete.server
        writing.plot_holes.list.local
        writing.plot_holes.list.server
        writing.plot_holes.update.local
        writing.plot_holes.update.server
        writing.plot_lines.create.local
        writing.plot_lines.create.server
        writing.plot_lines.delete.local
        writing.plot_lines.delete.server
        writing.plot_lines.list.local
        writing.plot_lines.list.server
        writing.plot_lines.update.local
        writing.plot_lines.update.server
        writing.projects.create.local
        writing.projects.create.server
        writing.projects.delete.local
        writing.projects.delete.server
        writing.projects.detail.local
        writing.projects.detail.server
        writing.projects.list.local
        writing.projects.list.server
        writing.projects.structure.local
        writing.projects.structure.server
        writing.projects.update.local
        writing.projects.update.server
        writing.relationships.create.local
        writing.relationships.create.server
        writing.relationships.delete.local
        writing.relationships.delete.server
        writing.relationships.list.local
        writing.relationships.list.server
        writing.research.launch.local
        writing.research.launch.server
        writing.scene_characters.create.local
        writing.scene_characters.create.server
        writing.scene_characters.delete.local
        writing.scene_characters.delete.server
        writing.scene_characters.list.local
        writing.scene_characters.list.server
        writing.scene_world_info.create.local
        writing.scene_world_info.create.server
        writing.scene_world_info.delete.local
        writing.scene_world_info.delete.server
        writing.scene_world_info.list.local
        writing.scene_world_info.list.server
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
        writing.trash.list.local
        writing.trash.list.server
        writing.trash.restore.local
        writing.trash.restore.server
        writing.versions.create.local
        writing.versions.create.server
        writing.versions.detail.local
        writing.versions.detail.server
        writing.versions.list.local
        writing.versions.list.server
        writing.versions.restore.local
        writing.versions.restore.server
        writing.world_info.create.local
        writing.world_info.create.server
        writing.world_info.delete.local
        writing.world_info.delete.server
        writing.world_info.detail.local
        writing.world_info.detail.server
        writing.world_info.list.local
        writing.world_info.list.server
        writing.world_info.update.local
        writing.world_info.update.server
    """),
}


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
        "Audited capability seed coverage must match the full parity matrix."
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

def test_backend_exception_classifier_handles_authentication_errors():
    assert classify_backend_exception(AuthenticationError("bad credentials")) == "server_auth_required"


def test_service_policy_enforcer_fails_closed_when_runtime_state_is_missing():
    enforcer = ServicePolicyEnforcer(state_provider=lambda: None)

    with pytest.raises(PolicyDeniedError) as exc:
        enforcer.require_allowed(action_id="notes.create.local")

    assert exc.value.reason_code == "authority_denied"
    assert exc.value.effective_source == "unknown"


def test_backend_exception_classifier_handles_session_invalid_authentication_errors():
    error = AuthenticationError(
        "session expired",
        response_data={"code": "session_invalid", "detail": "Session expired"},
    )

    assert classify_backend_exception(error) == "server_session_invalid"


def test_backend_exception_classifier_handles_session_invalid_401s():
    error = APIResponseError(
        401,
        "Unauthorized",
        response_data={"detail": "Session invalid. Please sign in again."},
    )

    assert classify_backend_exception(error) == "server_session_invalid"
