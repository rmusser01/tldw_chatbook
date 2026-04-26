import pytest

from tldw_chatbook.Character_Chat.character_persona_scope_service import CharacterPersonaScopeService
from tldw_chatbook.Chat.chat_conversation_scope_service import ChatConversationScopeService
from tldw_chatbook.Chat_Grammars_Interop.chat_grammars_scope_service import ChatGrammarsScopeService
from tldw_chatbook.Claims_Interop.claims_scope_service import ClaimsScopeService
from tldw_chatbook.Auth_Account_Interop.auth_account_scope_service import AuthAccountScopeService
from tldw_chatbook.Audio_Services_Interop.audio_services_scope_service import AudioServicesScopeService
from tldw_chatbook.Collections_Interop.collections_feeds_scope_service import CollectionsFeedsScopeService
from tldw_chatbook.Evaluations_Interop.evaluation_scope_service import EvaluationScopeService
from tldw_chatbook.External_Connectors_Interop.connectors_scope_service import ConnectorsScopeService
from tldw_chatbook.Feedback_Interop.feedback_scope_service import FeedbackScopeService
from tldw_chatbook.Kanban_Interop.kanban_scope_service import KanbanScopeService
from tldw_chatbook.LLM_Provider_Catalog.llm_provider_catalog_scope_service import LLMProviderCatalogScopeService
from tldw_chatbook.Media.media_reading_scope_service import MediaReadingScopeService
from tldw_chatbook.Meetings_Interop.meetings_scope_service import MeetingsScopeService
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService, ScopeType
from tldw_chatbook.Notifications.notifications_scope_service import NotificationsScopeService
from tldw_chatbook.Outputs_Interop.outputs_scope_service import OutputsScopeService
from tldw_chatbook.Prompt_Management.prompt_chatbook_scope_service import PromptChatbookScopeService
from tldw_chatbook.Prompt_Studio_Interop.prompt_studio_scope_service import PromptStudioScopeService
from tldw_chatbook.RAG_Admin.rag_admin_scope_service import RAGAdminScopeService
from tldw_chatbook.Research_Interop.research_scope_service import ResearchScopeService
from tldw_chatbook.Research_Interop.research_search_scope_service import ResearchSearchScopeService
from tldw_chatbook.Server_Runtime_Interop.server_runtime_scope_service import ServerRuntimeScopeService
from tldw_chatbook.Sharing_Interop.sharing_scope_service import SharingScopeService
from tldw_chatbook.Skills_Interop.skills_scope_service import SkillsScopeService
from tldw_chatbook.Study_Interop.quiz_scope_service import QuizScopeService
from tldw_chatbook.Study_Interop.study_scope_service import StudyScopeService
from tldw_chatbook.Subscriptions.watchlist_scope_service import WatchlistScopeService
from tldw_chatbook.Sync_Interop.sync_scope_service import SyncScopeService
from tldw_chatbook.Text2SQL_Interop.text2sql_scope_service import Text2SQLScopeService
from tldw_chatbook.Tools_Interop.tools_scope_service import ToolsScopeService
from tldw_chatbook.Translation_Interop.translation_scope_service import TranslationScopeService
from tldw_chatbook.User_Governance_Interop.user_governance_scope_service import UserGovernanceScopeService
from tldw_chatbook.Voice_Assistant_Interop.voice_assistant_scope_service import VoiceAssistantScopeService
from tldw_chatbook.Web_Clipper_Interop.web_clipper_scope_service import WebClipperScopeService
from tldw_chatbook.Web_Scraping_Interop.web_scraping_scope_service import WebScrapingScopeService
from tldw_chatbook.Writing_Interop.writing_scope_service import WritingScopeService
from tldw_chatbook.runtime_policy.unsupported_capabilities import (
    UnsupportedCapabilityReportError,
    collect_unsupported_capability_reports,
    validate_unsupported_capability_report,
)


def test_validate_unsupported_capability_report_returns_isolated_normalized_copies():
    raw_report = [
        {
            "operation_id": "chat.remote_create.server",
            "source": "server",
            "supported": False,
            "reason_code": "server_contract_missing",
            "user_message": "Server first-class conversation creation is not available.",
            "affected_action_ids": ["chat.create.server"],
        }
    ]

    normalized = validate_unsupported_capability_report(raw_report)

    assert normalized == raw_report
    assert normalized is not raw_report
    assert normalized[0] is not raw_report[0]
    raw_report[0]["affected_action_ids"].append("chat.delete.server")
    assert normalized[0]["affected_action_ids"] == ["chat.create.server"]


def test_validate_unsupported_capability_report_rejects_unknown_action_ids():
    report = [
        {
            "operation_id": "chat.remote_create.server",
            "source": "server",
            "supported": False,
            "reason_code": "server_contract_missing",
            "user_message": "Server first-class conversation creation is not available.",
            "affected_action_ids": ["chat.create.server", "chat.typo.server"],
        }
    ]

    with pytest.raises(UnsupportedCapabilityReportError, match="chat.typo.server"):
        validate_unsupported_capability_report(report)


def test_collect_unsupported_capability_reports_labels_each_report_scope():
    reports_by_scope = {
        "chat:server": [
            {
                "operation_id": "chat.remote_create.server",
                "source": "server",
                "supported": False,
                "reason_code": "server_contract_missing",
                "user_message": "Server first-class conversation creation is not available.",
                "affected_action_ids": ["chat.create.server"],
            }
        ],
        "notes:server": [],
    }

    collected = collect_unsupported_capability_reports(reports_by_scope)

    assert collected == [
        {
            "operation_id": "chat.remote_create.server",
            "source": "server",
            "supported": False,
            "reason_code": "server_contract_missing",
            "user_message": "Server first-class conversation creation is not available.",
            "affected_action_ids": ["chat.create.server"],
            "report_scope": "chat:server",
        }
    ]


@pytest.mark.parametrize(
    ("service", "call_kwargs"),
    [
        (ChatConversationScopeService(local_service=None, server_service=None), [{"mode": "local"}, {"mode": "server"}]),
        (CharacterPersonaScopeService(local_service=None, server_service=None), [{"mode": "local"}, {"mode": "server"}]),
        (
            NotesScopeService(local_notes_service=None, server_service=None),
            [
                {"scope": ScopeType.LOCAL_NOTE},
                {"scope": ScopeType.WORKSPACE},
                {"scope": ScopeType.SERVER_NOTE},
            ],
        ),
        (MediaReadingScopeService(local_service=None, server_service=None), [{"mode": "local"}, {"mode": "server"}]),
        (
            PromptChatbookScopeService(local_prompt_service=None, server_prompt_service=None),
            [{"mode": "local"}, {"mode": "server"}],
        ),
        (StudyScopeService(local_service=None, server_service=None), [{"mode": "local"}, {"mode": "server"}]),
        (QuizScopeService(local_service=None, server_service=None), [{"mode": "local"}, {"mode": "server"}]),
        (OutputsScopeService(local_service=None, server_service=None), [{"mode": "local"}, {"mode": "server"}]),
        (
            WatchlistScopeService(local_service=None, server_service=None),
            [{"runtime_backend": "local"}, {"runtime_backend": "server"}],
        ),
        (WritingScopeService(local_service=None, server_service=None), [{"mode": "local"}, {"mode": "server"}]),
        (ResearchScopeService(local_service=None, server_service=None), [{"mode": "local"}, {"mode": "server"}]),
        (ResearchSearchScopeService(local_service=None, server_service=None), [{"mode": "local"}, {"mode": "server"}]),
        (ChatGrammarsScopeService(server_service=None), [{"mode": "local"}, {"mode": "server"}]),
        (FeedbackScopeService(server_service=None), [{"mode": "local"}, {"mode": "server"}]),
        (ClaimsScopeService(server_service=None), [{"mode": "local"}, {"mode": "server"}]),
        (MeetingsScopeService(server_service=None), [{"mode": "local"}, {"mode": "server"}]),
        (PromptStudioScopeService(server_service=None), [{"mode": "local"}, {"mode": "server"}]),
        (KanbanScopeService(server_service=None), [{"mode": "local"}, {"mode": "server"}]),
        (TranslationScopeService(server_service=None), [{"mode": "local"}, {"mode": "server"}]),
        (NotificationsScopeService(server_service=None), [{"mode": "local"}, {"mode": "server"}]),
        (ServerRuntimeScopeService(server_service=None), [{"mode": "local"}, {"mode": "server"}]),
        (LLMProviderCatalogScopeService(local_service=None, server_service=None), [{"mode": "local"}, {"mode": "server"}]),
        (AuthAccountScopeService(server_service=None), [{"mode": "local"}, {"mode": "server"}]),
        (AudioServicesScopeService(local_service=None, server_service=None), [{"mode": "local"}, {"mode": "server"}]),
        (CollectionsFeedsScopeService(server_service=None), [{"mode": "local"}, {"mode": "server"}]),
        (ConnectorsScopeService(server_service=None), [{"mode": "local"}, {"mode": "server"}]),
        (SkillsScopeService(server_service=None), [{"mode": "local"}, {"mode": "server"}]),
        (ToolsScopeService(server_service=None), [{"mode": "local"}, {"mode": "server"}]),
        (Text2SQLScopeService(server_service=None), [{"mode": "local"}, {"mode": "server"}]),
        (SyncScopeService(server_service=None), [{"mode": "local"}, {"mode": "server"}]),
        (UserGovernanceScopeService(server_service=None), [{"mode": "local"}, {"mode": "server"}]),
        (VoiceAssistantScopeService(server_service=None), [{"mode": "local"}, {"mode": "server"}]),
        (SharingScopeService(server_service=None), [{"mode": "local"}, {"mode": "server"}]),
        (WebClipperScopeService(server_service=None), [{"mode": "local"}, {"mode": "server"}]),
        (WebScrapingScopeService(server_service=None), [{"mode": "local"}, {"mode": "server"}]),
        (EvaluationScopeService(local_service=None, server_service=None), [{"mode": "local"}, {"mode": "server"}]),
        (RAGAdminScopeService(local_service=None, server_service=None), [{"mode": "local"}, {"mode": "server"}]),
    ],
)
def test_scope_service_unsupported_reports_match_the_shared_contract(service, call_kwargs):
    for kwargs in call_kwargs:
        report = service.list_unsupported_capabilities(**kwargs)

        assert validate_unsupported_capability_report(report) == report
