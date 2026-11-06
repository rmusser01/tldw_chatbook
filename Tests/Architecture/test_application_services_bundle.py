"""The application services bundle and its app-attribute aliases (ADR-250, TASK-36947.2).

``TldwCli.services`` is built once, after composition. The app attributes that
composition assigns stay in place as aliases of the bundle's fields until their
last reader migrates (ADR-250 section 4). Three things are pinned here:

* every alias still reaches the very object the bundle holds;
* every service name production code reads from an app-like receiver (the
  definition in ``test_app_string_lookup_census.py``) is a bundle field, reached
  through its alias, or is on the exclusion list below;
* the module imports only the standard library at module level (ADR-097).

The alias list and the exclusion list only shrink. A service added later goes
into the bundle with no alias; one built after composition may join the
exclusion list (ADR-250 section 2, Q3).
"""

from __future__ import annotations

import ast
import inspect
import re
import sys
from pathlib import Path

import pytest

from Tests.private_profile import private_profile_test

PACKAGE = Path(__file__).resolve().parents[2] / "tldw_chatbook"
_RECEIVERS = {"app", "app_instance", "tldw_app", "_app", "_app_instance"}
_SERVICE_NAME = re.compile(
    r"_(service|db|provider|repository|store|owner|coordinator|controller"
    r"|manager|runtime|registry|enforcer)$"
)

# (app attribute, dotted path into ``app.services``). Shrink only.
ALIASES: tuple[tuple[str, str], ...] = (
    ("server_context_provider", "core.server_context_provider"),
    ("service_policy_enforcer", "core.service_policy_enforcer"),
    ("unified_mcp_target_store", "core.unified_mcp_target_store"),
    ("chachanotes_db", "storage.chachanotes"),
    ("media_db", "storage.media"),
    ("prompts_db", "storage.prompts"),
    ("local_workspace_db", "storage.workspaces"),
    ("local_chat_conversation_service", "chat.conversation.local"),
    ("server_chat_conversation_service", "chat.conversation.server"),
    ("chat_conversation_scope_service", "chat.conversation.scope"),
    ("local_chat_dictionary_service", "chat.dictionary.local"),
    ("server_chat_dictionary_service", "chat.dictionary.server"),
    ("chat_dictionary_scope_service", "chat.dictionary.scope"),
    ("local_chat_grammars_service", "chat.grammars.local"),
    ("server_chat_grammars_service", "chat.grammars.server"),
    ("chat_grammars_scope_service", "chat.grammars.scope"),
    ("conversation_local_marks_service", "chat.conversation_local_marks_service"),
    ("citation_trace_repository", "chat.citation_trace_repository"),
    ("citation_artifact_ownership_coordinator", "chat.citation_artifact_ownership_coordinator"),
    ("citation_legacy_migration_service", "chat.citation_legacy_migration_service"),
    ("notes_service", "notes.local"),
    ("server_notes_workspace_service", "notes.server"),
    ("notes_scope_service", "notes.scope"),
    ("file_notes_session_owner", "notes.file_notes_session_owner"),
    ("local_media_reading_service", "media.local"),
    ("server_media_reading_service", "media.server"),
    ("media_reading_scope_service", "media.scope"),
    ("workspace_registry_service", "workspace.workspace_registry_service"),
    ("change_review_consent_service", "workspace.change_review_consent_service"),
    ("unified_mcp_service", "mcp.unified_mcp_service"),
    ("server_unified_mcp_service", "mcp.server_unified_mcp_service"),
    ("local_mcp_control_service", "mcp.local_mcp_control_service"),
    ("local_mcp_store", "mcp.local_mcp_store"),
    ("unified_mcp_context_store", "mcp.unified_mcp_context_store"),
    ("server_mcp_governance_service", "mcp.governance.server"),
    ("mcp_governance_scope_service", "mcp.governance.scope"),
    ("server_sync_service", "sync.server_sync_service"),
    ("sync_scope_service", "sync.sync_scope_service"),
    ("local_first_sync_service", "sync.local_first_sync_service"),
    ("manual_sync_control_service", "sync.manual_sync_control_service"),
    ("sync_state_repository", "sync.sync_state_repository"),
    ("sync_restore_service", "sync.sync_restore_service"),
    ("local_watchlists_service", "watchlists.local"),
    ("server_watchlists_service", "watchlists.server"),
    ("watchlist_scope_service", "watchlists.scope"),
    ("subscriptions_db", "watchlists.subscriptions_db"),
    ("watchlist_bundle_service", "watchlists.watchlist_bundle_service"),
    ("local_quiz_service", "quiz.local"),
    ("server_quiz_service", "quiz.server"),
    ("study_quiz_scope_service", "quiz.scope"),
    ("local_chatbook_service", "chatbook.local"),
    ("server_chatbook_service", "chatbook.server"),
    ("prompt_chatbook_scope_service", "chatbook.scope"),
    ("server_runtime_service", "runtime.server"),
    ("server_runtime_scope_service", "runtime.scope"),
    ("active_server_capability_service", "runtime.active_server_capability_service"),
    ("local_research_service", "research.local"),
    ("server_research_service", "research.server"),
    ("research_scope_service", "research.scope"),
    ("research_paste_staging_store", "research.research_paste_staging_store"),
    ("research_source_association_coordinator", "research.research_source_association_coordinator"),
    ("research_source_operation_store", "research.research_source_operation_store"),
    ("research_source_readiness_coordinator", "research.research_source_readiness_coordinator"),
    ("server_notifications_service", "notifications.server"),
    ("notifications_scope_service", "notifications.scope"),
    ("client_notifications_db", "notifications.client_notifications_db"),
    ("client_notifications_service", "notifications.client_notifications_service"),
    ("notification_dispatch_service", "notifications.notification_dispatch_service"),
    ("local_library_collections_service", "library.local_library_collections_service"),
    ("library_collections_service", "library.library_collections_service"),
    ("library_rag_search_service", "library.library_rag_search_service"),
    ("actor_pack_repository", "actor_packs.actor_pack_repository"),
    ("actor_pack_creation_service", "actor_packs.actor_pack_creation_service"),
    ("actor_pack_export_service", "actor_packs.actor_pack_export_service"),
    ("actor_pack_export_controller", "actor_packs.actor_pack_export_controller"),
    ("actor_pack_import_service", "actor_packs.actor_pack_import_service"),
    ("actor_pack_activation_service", "actor_packs.actor_pack_activation_service"),
    ("actor_pack_import_controller", "actor_packs.actor_pack_import_controller"),
    ("persona_actor_pack_coordinator", "actor_packs.persona_actor_pack_coordinator"),
    ("acp_runtime_process_manager", "platform.acp_runtime_process_manager"),
    ("audio_cpp_model_install_owner", "platform.audio_cpp_model_install_owner"),
    ("tts_service", "platform.tts_service"),
    ("generated_video_store", "platform.generated_video_store"),
    ("model_catalog_disk_store", "platform.model_catalog_disk_store"),
    ("event_state_repository", "platform.event_state_repository"),
    ("scheduling_service", "platform.scheduling_service"),
    ("local_audio_services_service", "audio_services.local"),
    ("server_audio_services_service", "audio_services.server"),
    ("audio_services_scope_service", "audio_services.scope"),
    ("server_auth_account_service", "auth_account.server"),
    ("auth_account_scope_service", "auth_account.scope"),
    ("local_character_persona_service", "character_persona.local"),
    ("server_character_persona_service", "character_persona.server"),
    ("character_persona_scope_service", "character_persona.scope"),
    ("server_claims_service", "claims.server"),
    ("claims_scope_service", "claims.scope"),
    ("server_collections_feeds_service", "collections_feeds.server"),
    ("collections_feeds_scope_service", "collections_feeds.scope"),
    ("server_companion_service", "companion.server"),
    ("companion_scope_service", "companion.scope"),
    ("server_connectors_service", "connectors.server"),
    ("connectors_scope_service", "connectors.scope"),
    ("local_evaluation_service", "evaluation.local"),
    ("server_evaluation_service", "evaluation.server"),
    ("evaluation_scope_service", "evaluation.scope"),
    ("local_feedback_service", "feedback.local"),
    ("server_feedback_service", "feedback.server"),
    ("feedback_scope_service", "feedback.scope"),
    ("local_llm_provider_catalog_service", "llm_provider_catalog.local"),
    ("server_llm_provider_catalog_service", "llm_provider_catalog.server"),
    ("llm_provider_catalog_scope_service", "llm_provider_catalog.scope"),
    ("server_meetings_service", "meetings.server"),
    ("meetings_scope_service", "meetings.scope"),
    ("server_outputs_service", "outputs.server"),
    ("outputs_scope_service", "outputs.scope"),
    ("server_personalization_service", "personalization.server"),
    ("personalization_scope_service", "personalization.scope"),
    ("local_prompt_service", "prompts.local"),
    ("server_prompt_service", "prompts.server"),
    ("prompt_scope_service", "prompts.scope"),
    ("server_prompt_studio_service", "prompt_studio.server"),
    ("prompt_studio_scope_service", "prompt_studio.scope"),
    ("local_research_search_service", "research_search.local"),
    ("server_research_search_service", "research_search.server"),
    ("research_search_scope_service", "research_search.scope"),
    ("server_sharing_service", "sharing.server"),
    ("sharing_scope_service", "sharing.scope"),
    ("local_study_service", "study.local"),
    ("server_study_service", "study.server"),
    ("study_scope_service", "study.scope"),
    ("server_text2sql_service", "text2sql.server"),
    ("text2sql_scope_service", "text2sql.scope"),
    ("server_tools_service", "tools.server"),
    ("tools_scope_service", "tools.scope"),
    ("server_translation_service", "translation.server"),
    ("translation_scope_service", "translation.scope"),
    ("server_user_governance_service", "user_governance.server"),
    ("user_governance_scope_service", "user_governance.scope"),
    ("server_voice_assistant_service", "voice_assistant.server"),
    ("voice_assistant_scope_service", "voice_assistant.scope"),
    ("server_web_clipper_service", "web_clipper.server"),
    ("web_clipper_scope_service", "web_clipper.scope"),
    ("server_web_scraping_service", "web_scraping.server"),
    ("web_scraping_scope_service", "web_scraping.scope"),
    ("local_writing_service", "writing.local"),
    ("server_writing_service", "writing.server"),
    ("writing_scope_service", "writing.scope"),
)

# Service names read from the app that are NOT in the bundle, and why
# (ADR-250 section 2, rules 4 and 5). Shrinks as consumers migrate.
EXCLUDED: dict[str, str] = {
    **dict.fromkeys(
        (
            "console_runtime",
            "db_status_manager",
            "raw_cli_runtime",
            "screen_state_store",
            "console_settings_durability_owner",
            "console_chat_store",
            "_registry",
            "buddy_conversation_coordinator",
            "buddy_speech_coordinator",
            "library_skill_import_coordinator",
            "_rag_service",
            "_project_index_service",
            "_llamacpp_connection_owner",
            "_vllm_connection_owner",
        ),
        "rule 4: the app's own runtime, session state, or attached by a feature module after boot",
    ),
    **dict.fromkeys(
        (
            "artifact_share_controller",
            "daily_report_demo_service",
            "llamacpp_snapshot_service",
            "meeting_session_owner",
            "notes_sync_runtime_owner",
            "persona_buddy_controller",
            "plugin_source_service",
            "rag_admin_scope_service",
            "terminal_session_manager",
            "_terminal_session_manager",
            "_audio_cpp_artifact_lease_coordinator",
            "_local_stt_dispatch_coordinator",
            "_notes_sync_runtime_owner",
            "_parakeet_source_service",
            "_personal_context_service",
            "_plugin_service",
            "_tts_profile_repository",
            "_tts_profile_service",
            "_tts_voice_bundle_service",
        ),
        "rule 5: built on first access",
    ),
    **dict.fromkeys(
        ("tool_pack_service", "watchlists_command_service", "watchlists_operation_coordinator"),
        "rule 5: composed in on_mount or later",
    ),
    **dict.fromkeys(
        ("local_skill_trust_service", "local_skills_service", "skills_scope_service"),
        "rule 5: pinned in place (off-loop trust proof checks these exact slots)",
    ),
    **dict.fromkeys(
        ("dreams_db", "guardian_db"),
        "rule 5: opened on first use",
    ),
    **dict.fromkeys(
        (
            "collections_capture_scope_service",
            "collections_legacy_recovery_service",
            "notes_organization_repository",
            "notes_organization_sync_service",
            "local_library_collections_db",
        ),
        "rule 5: reset and rebuilt (collections capture reassigns the Library "
        "collections database when it is not a LibraryCollectionsDB; Notes-sync wiring re-runs)",
    ),
    **dict.fromkeys(
        ("server_outputs_scope_service", "server_sharing_scope_service"),
        "legacy fallback names no app attribute carries (getattr default None)",
    ),
}


def _receiver(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _service_names_read_from_the_app() -> set[str]:
    names: set[str] = set()
    for path in PACKAGE.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            name = None
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.ctx, ast.Load)
                and _receiver(node.value) in _RECEIVERS
            ):
                name = node.attr
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in ("getattr", "hasattr")
                and len(node.args) > 1
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
                and _receiver(node.args[0]) in _RECEIVERS
            ):
                name = node.args[1].value
            if name and _SERVICE_NAME.search(name):
                names.add(name)
    return names


def _is_method(cls: type, name: str) -> bool:
    return inspect.isfunction(inspect.getattr_static(cls, name, None))


@pytest.mark.unit
def test_the_bundle_exposes_every_service_that_consumers_look_up() -> None:
    from tldw_chatbook.app import TldwCli

    read = {n for n in _service_names_read_from_the_app() if not _is_method(TldwCli, n)}
    aliased = {attr for attr, _ in ALIASES}
    missing = sorted(read - aliased - set(EXCLUDED))
    assert not missing, (
        f"Service(s) {missing} are read from the app but are neither a bundle field "
        "nor excluded. Add a field in app_services.py and an ALIASES row, or "
        "add the name to EXCLUDED with the ADR-250 rule that keeps it out."
    )
    stale = sorted(set(EXCLUDED) - read)
    assert not stale, f"{stale} are no longer read from the app; remove them from EXCLUDED."
    both = sorted(aliased & set(EXCLUDED))
    assert not both, f"{both} are both a bundle field and excluded."


@pytest.mark.unit
def test_app_services_imports_only_the_standard_library_at_module_level() -> None:
    tree = ast.parse((PACKAGE / "app_services.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"relative import at module level: line {node.lineno}"
            roots = [(node.module or "").split(".")[0]]
        else:
            continue
        for root in roots:
            assert root == "__future__" or root in sys.stdlib_module_names, (
                f"app_services.py imports {root!r} at module level (ADR-097)"
            )


pytestmark = [pytest.mark.bootstrap_profile]


@private_profile_test
def test_every_alias_is_the_bundle_field_after_construction(request) -> None:
    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.app_services import ApplicationServices

    app = _build_test_app()
    assert isinstance(app.services, ApplicationServices)
    wrong = []
    for attr, path in ALIASES:
        field = app.services
        for part in path.split("."):
            field = getattr(field, part)
        if getattr(app, attr) is not field:
            wrong.append(f"app.{attr} is not services.{path}")
    assert not wrong, wrong
