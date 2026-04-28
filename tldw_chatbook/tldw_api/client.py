# tldw_chatbook/tldw_api/client.py
#
#
from __future__ import annotations

# Imports
import json # For MediaWiki streaming
from pathlib import Path # For utils.prepare_files_for_httpx
from typing import Optional, Dict, Any, List, AsyncGenerator, Union, Literal
from urllib.parse import quote
#
# 3rd-party Libraries
import httpx
from pydantic import BaseModel
#
# Local Imports
from .schemas import (
    AddMediaRequest,
    ProcessVideoRequest, ProcessAudioRequest, ProcessPDFRequest,
    ProcessEbookRequest, ProcessDocumentRequest, ProcessXMLRequest, ProcessMediaWikiRequest,
    ProcessPlaintextRequest, ProcessCodeRequest, ProcessEmailsRequest, ProcessWebScrapingRequest,
    BatchMediaProcessResponse, MediaItemProcessResult,
    BatchProcessXMLResponse, ProcessedMediaWikiPage,
    ProcessXMLResponseItem,  # Add specific XML/MediaWiki later if needed
)
from .notes_workspace_schemas import (
    MediaListResponse,
    MediaSearchRequest,
    NoteGraphRequest,
    NoteLinkCreate,
    NoteCreateRequest,
    NoteGraphRequest,
    NoteLinkCreateRequest,
    NoteListResponse,
    NoteResponse,
    NoteUpdateRequest,
    WorkspaceArtifactCreateRequest,
    WorkspaceArtifactResponse,
    WorkspaceArtifactUpdateRequest,
    WorkspaceCreateRequest,
    WorkspaceListResponse,
    WorkspaceNoteCreateRequest,
    WorkspaceNoteResponse,
    WorkspaceNoteUpdateRequest,
    WorkspaceResponse,
    WorkspaceSourceCreateRequest,
    WorkspaceSourceResponse,
    WorkspaceSourceUpdateRequest,
    WorkspaceUpdateRequest,
)
from .media_reading_schemas import (
    AsyncMode,
    DocumentAnnotationCreateRequest,
    DocumentAnnotationCreate,
    DocumentAnnotationListResponse,
    DocumentAnnotationResponse,
    DocumentAnnotationSyncRequest,
    DocumentAnnotationSyncResponse,
    DocumentAnnotationUpdateRequest,
    DocumentAnnotationUpdate,
    DocumentVersionAdvancedUpsertRequest,
    DocumentVersionCreateRequest,
    DocumentVersionDetailResponse,
    DocumentVersionMetadataPatchRequest,
    DocumentVersionRollbackRequest,
    DocumentFiguresResponse,
    DocumentInsightsRequest,
    DocumentInsightsResponse,
    DocumentOutlineResponse,
    DocumentReferencesResponse,
    ExportMode,
    FileArtifactsPurgeRequest,
    FileCreateRequest,
    AddMediaRequest,
    IngestionSourceCreateRequest,
    IngestionSourceItemListResponse,
    IngestionSourceItemResponse,
    IngestionSourceListResponse,
    IngestionSourcePatchRequest,
    IngestionSourceResponse,
    IngestionSourceSyncTriggerResponse,
    IngestWebContentRequest,
    IngestWebContentResponse,
    ItemsBulkRequest,
    ItemsBulkResponse,
    CancelMediaIngestBatchResponse,
    CancelMediaIngestJobResponse,
    MediaIngestBatchCancelResponse,
    MediaIngestJobCancelResponse,
    MediaIngestJobListResponse,
    MediaIngestJobStreamEvent,
    MediaIngestJobSubmitRequest,
    MediaIngestJobStatus,
    MediaIngestSubmitRequest,
    MediaIngestSubmitResponse,
    SubmitMediaIngestJobsResponse,
    MediaAdvancedVersionUpsertRequest,
    MediaFileAvailabilityResponse,
    MediaDetailResponse,
    MediaIdentifierLookupResponse,
    MediaItemUpdateRequest,
    MediaKeywordListResponse,
    MediaKeywordsResponse,
    MediaKeywordsUpdateRequest,
    MediaMetadataSearchResponse,
    MediaMetadataPatchRequest,
    MediaTranscriptionModelsResponse,
    MediaTrashEmptyResponse,
    MediaUpdateRequest,
    MediaNavigationContentResponse,
    MediaNavigationResponse,
    MediaVersionCreateRequest,
    MediaVersionDetail,
    MediaVersionRollbackRequest,
    ReadingDigestOutputsListResponse,
    ReadingDigestScheduleCreateRequest,
    ReadingDigestScheduleResponse,
    ReadingDigestScheduleUpdateRequest,
    ReadingExportResponse,
    ReadingExportRequest,
    ReadingHighlight,
    ReadingHighlightCreateRequest,
    ReadingHighlightDeleteResponse,
    ReadingHighlightUpdateRequest,
    ReadingImportJobResponse,
    ReadingImportJobStatus,
    ReadingImportJobsListResponse,
    ReadingItem,
    ReadingNoteLinkCreateRequest,
    ReadingNoteLinkResponse,
    ReadingNoteLinksListResponse,
    ReadingArchiveCreateRequest,
    ReadingArchiveResponse,
    ReadingProgressUpdate,
    ReadingSaveRequest,
    ReadingSavedSearchCreateRequest,
    ReadingSavedSearchListResponse,
    ReadingSavedSearchResponse,
    ReadingSavedSearchUpdateRequest,
    ReadingSummarizeRequest,
    ReadingSummaryResponse,
    ReadingTTSRequest,
    ReadingTTSResponse,
    ReadingUpdateRequest,
    ReprocessMediaRequest,
    ReprocessMediaResponse,
    ServerMediaListResponse,
    UnifiedItem,
    UnifiedItemsListResponse,
)
from .prompt_chatbook_schemas import (
    ChatbookContinueExportRequest,
    ChatbookExportRequest,
    ChatbookImportRequest,
    ChatbookImportJobListResponse,
    ChatbookImportJobResponse,
    ChatbookJobMutationResponse,
    PaginatedPromptsResponse,
    PromptCollectionCreateRequest,
    PromptCollectionCreateResponse,
    PromptCollectionListResponse,
    PromptCollectionResponse,
    PromptCollectionUpdateRequest,
    PromptCreateRequest,
    PromptPreviewRequest,
    PromptResponse,
    PromptVersionResponse,
)
from .translation_schemas import (
    TranslateRequest,
    TranslateResponse,
)
from .companion_schemas import (
    CompanionActivityCreate,
    CompanionActivityItem,
    CompanionActivityListResponse,
    CompanionCheckInCreate,
    CompanionConversationPromptsResponse,
    CompanionGoal,
    CompanionGoalCreate,
    CompanionGoalListResponse,
    CompanionGoalUpdate,
    CompanionKnowledgeDetail,
    CompanionKnowledgeListResponse,
    CompanionLifecycleResponse,
    CompanionPurgeRequest,
    CompanionRebuildRequest,
    CompanionReflectionDetail,
)
from .personalization_schemas import (
    PersonalizationDetailResponse,
    PersonalizationExplanationListResponse,
    PersonalizationMemoryCreate,
    PersonalizationMemoryExportResponse,
    PersonalizationMemoryImportRequest,
    PersonalizationMemoryItem,
    PersonalizationMemoryListResponse,
    PersonalizationMemoryUpdate,
    PersonalizationMemoryValidateRequest,
    PersonalizationOptInRequest,
    PersonalizationPreferencesUpdate,
    PersonalizationProfile,
    PersonalizationPurgeResponse,
)
from .ocr_vlm_schemas import (
    OCRBackendsResponse,
    OCRPointsPreloadResponse,
    VLMBackendsResponse,
)
from .llm_provider_schemas import (
    LLMHealthResponse,
    LLMModelMetadataResponse,
    LLMProviderDetail,
    LLMProviderListResponse,
)
from .voice_assistant_schemas import (
    VoiceAnalyticsSummary,
    VoiceCommandDefinition,
    VoiceCommandDryRunRequest,
    VoiceCommandDryRunResponse,
    VoiceCommandInfo,
    VoiceCommandListResponse,
    VoiceCommandRequest,
    VoiceCommandResponse,
    VoiceCommandToggleRequest,
    VoiceCommandUsage,
    VoiceCommandValidationResponse,
    VoiceSessionInfo,
    VoiceSessionListResponse,
)
from .server_runtime_schemas import (
    FlashcardsImportLimitsResponse,
    JobsConfigResponse,
    ProviderValidateRequest,
    ProviderValidateResponse,
    ProvidersStatusResponse,
    ServerDocsInfoResponse,
    ServerHealthResponse,
    ServerLivenessResponse,
    ServerMetricsResponse,
    ServerReadinessResponse,
    ServerSecurityHealthResponse,
    TokenizerConfigResponse,
    TokenizerUpdateRequest,
)
from .auth_user_schemas import (
    AuthTokenResponse,
    LogoutRequest,
    MessageResponse,
    MFAChallengeResponse,
    RefreshTokenRequest,
    RegisterRequest,
    RegistrationResponse,
    SessionResponse,
    UserProfileCatalogResponse,
    UserProfileResponse,
    UserProfileUpdateRequest,
    UserProfileUpdateResponse,
)
from .account_security_schemas import (
    APIKeyCreateRequest,
    APIKeyCreateResponse,
    APIKeyMetadata,
    APIKeyRotateRequest,
    MFASetupResponse,
    PasswordChangeRequest,
    PasswordResetConfirm,
    PasswordResetRequest,
    StorageQuotaResponse,
    VirtualAPIKeyCreateRequest,
)
from .user_keys_schemas import (
    OpenAICredentialSourceSwitchRequest,
    OpenAICredentialSourceSwitchResponse,
    OpenAIOAuthAuthorizeRequest,
    OpenAIOAuthAuthorizeResponse,
    OpenAIOAuthCallbackResponse,
    OpenAIOAuthRefreshResponse,
    OpenAIOAuthStatusResponse,
    ProviderKeyTestRequest,
    ProviderKeyTestResponse,
    UserProviderKeyResponse,
    UserProviderKeysResponse,
    UserProviderKeyUpsertRequest,
)
from .storage_schemas import (
    BulkDeleteRequest,
    BulkDeleteResponse,
    BulkMoveRequest,
    BulkMoveResponse,
    FileCategory,
    FolderListResponse,
    GeneratedFileResponse,
    GeneratedFilesListResponse,
    GeneratedFileUpdate,
    RestoreResponse,
    SourceFeature,
    StorageUsageResponse,
    TrashListResponse,
    UsageBreakdownResponse,
)
from .user_governance_schemas import (
    ConsentPreferencesResponse,
    ConsentRecordResponse,
    PrivilegeDetailResponse,
    PrivilegeSelfResponse,
)
from .connectors_schemas import (
    AuthorizeURLResponse,
    ConnectorAccount,
    ConnectorBrowseResponse,
    ConnectorImportJob,
    ConnectorProvider,
    ConnectorSource,
    ConnectorSourceCreateRequest,
    ConnectorSourcePatchRequest,
    ConnectorSourceSyncStatus,
    ConnectorSourceSyncTriggerResponse,
)
from .chat_grammar_schemas import (
    ChatGrammarCreate,
    ChatGrammarListResponse,
    ChatGrammarResponse,
    ChatGrammarUpdate,
)
from .feedback_schemas import (
    ExplicitFeedbackRequest,
    ExplicitFeedbackResponse,
    FeedbackDeleteResponse,
    FeedbackListResponse,
    FeedbackUpdateRequest,
)
from .collections_feeds_schemas import (
    CollectionsFeed,
    CollectionsFeedCreateRequest,
    CollectionsFeedsListResponse,
    CollectionsFeedUpdateRequest,
    CollectionsWebSubSubscribeRequest,
    CollectionsWebSubSubscriptionResponse,
)
from .claims_schemas import (
    ClaimReviewBulkRequest,
    ClaimReviewRequest,
    ClaimReviewRuleCreate,
    ClaimReviewRuleUpdate,
    ClaimNotificationsAckRequest,
    ClaimNotificationsDigestResponse,
    ClaimNotificationResponse,
    ClaimUpdateRequest,
    ClaimsAlertConfigCreate,
    ClaimsAlertConfigResponse,
    ClaimsAlertConfigUpdate,
    ClaimsAnalyticsDashboardResponse,
    ClaimsAnalyticsExportListResponse,
    ClaimsAnalyticsExportRequest,
    ClaimsAnalyticsExportResponse,
    ClaimsClusterLinkCreate,
    ClaimsClusterLinkResponse,
    ClaimsExtractorCatalogResponse,
    ClaimsMonitoringSettingsResponse,
    ClaimsMonitoringSettingsUpdate,
    ClaimsReviewExtractorMetricsResponse,
    ClaimsSearchResponse,
    ClaimsSettingsResponse,
    ClaimsSettingsUpdate,
    FVASettingsResponse,
    FVAVerifyRequest,
    FVAVerifyResponse,
)
from .skills_schemas import (
    SkillContextPayload,
    SkillCreate,
    SkillExecuteRequest,
    SkillExecutionResult,
    SkillImportRequest,
    SkillResponse,
    SkillsListResponse,
    SkillUpdate,
)
from .tools_schemas import ExecuteToolRequest, ExecuteToolResult, ToolListResponse
from .mcp_governance_schemas import (
    MCPApprovalDecisionCreate,
    MCPApprovalPolicyCreate,
    MCPApprovalPolicyUpdate,
    MCPCapabilityMappingCreate,
    MCPCapabilityMappingUpdate,
    MCPCatalogCreate,
    MCPCatalogEntryCreate,
    MCPEffectivePolicyResponse,
    MCPExternalServerCreate,
    MCPExternalServerUpdate,
    MCPGovernanceObject,
    MCPGovernanceSummary,
    MCPPermissionProfileCreate,
    MCPPermissionProfileUpdate,
    MCPPolicyAssignmentCreate,
    MCPPolicyAssignmentUpdate,
    MCPSecretSetRequest,
)
from .text2sql_schemas import Text2SQLRequest, Text2SQLResponse
from .sync_schemas import ClientChangesPayload, ServerChangesResponse
from .prompt_studio_schemas import (
    PromptStudioCompareStrategiesRequest,
    PromptStudioDeleteMessage,
    PromptStudioEvaluationCreate,
    PromptStudioEvaluationListResponse,
    PromptStudioEvaluationResponse,
    PromptStudioListResponse,
    PromptStudioOptimizationCreate,
    PromptStudioOptimizationIterationCreate,
    PromptStudioOptimizationSimpleCreateRequest,
    PromptStudioPromptConvertRequest,
    PromptStudioPromptCreate,
    PromptStudioPromptExecuteRequest,
    PromptStudioPromptExecutionResponse,
    PromptStudioPromptPreviewRequest,
    PromptStudioPromptUpdate,
    PromptStudioProjectCreate,
    PromptStudioProjectUpdate,
    PromptStudioRunTestCasesRequest,
    PromptStudioRunTestCasesResponse,
    PromptStudioSimpleJobResponse,
    PromptStudioStandardResponse,
    PromptStudioStatusResponse,
    PromptStudioTestCaseBulkCreate,
    PromptStudioTestCaseExportRequest,
    PromptStudioTestCaseGenerateRequest,
    PromptStudioTestCaseImportRequest,
    PromptStudioTestCaseCreate,
    PromptStudioTestCaseUpdate,
)
from .data_tables_schemas import (
    DataTableContentUpdateRequest,
    DataTableDeleteResponse,
    DataTableDetailResponse,
    DataTableExportFormat,
    DataTableExportResponse,
    DataTableGenerateRequest,
    DataTableGenerateResponse,
    DataTableJobCancelResponse,
    DataTableJobStatus,
    DataTableRegenerateRequest,
    DataTablesListResponse,
    DataTableSummary,
    DataTableUpdateRequest,
)
from .kanban_schemas import (
    KanbanActivitiesListResponse,
    KanbanBoardCreate,
    KanbanBoardExportRequest,
    KanbanBoardExportResponse,
    KanbanBoardImportRequest,
    KanbanBoardImportResponse,
    KanbanBoardListResponse,
    KanbanBoardResponse,
    KanbanBoardUpdate,
    KanbanBoardWithListsResponse,
    KanbanBulkArchiveCardsRequest,
    KanbanBulkArchiveCardsResponse,
    KanbanBulkCardLinksAddResponse,
    KanbanBulkCardLinksRemoveResponse,
    KanbanBulkCardLinksRequest,
    KanbanBulkDeleteCardsRequest,
    KanbanBulkDeleteCardsResponse,
    KanbanBulkLabelCardsRequest,
    KanbanBulkLabelCardsResponse,
    KanbanBulkMoveCardsRequest,
    KanbanBulkMoveCardsResponse,
    KanbanBulkUnarchiveCardsResponse,
    KanbanCardCopyRequest,
    KanbanCardCopyWithChecklistsRequest,
    KanbanCardCreate,
    KanbanCardLinkCountsResponse,
    KanbanCardLinkCreate,
    KanbanCardLinkResponse,
    KanbanCardLinksListResponse,
    KanbanCardMoveRequest,
    KanbanCardResponse,
    KanbanCardSearchRequest,
    KanbanCardSearchResponse,
    KanbanCardsListResponse,
    KanbanCardUpdate,
    KanbanCardWithDetailsResponse,
    KanbanChecklistCreate,
    KanbanChecklistItemCreate,
    KanbanChecklistItemReorderRequest,
    KanbanChecklistItemResponse,
    KanbanChecklistItemsListResponse,
    KanbanChecklistItemUpdate,
    KanbanChecklistReorderRequest,
    KanbanChecklistResponse,
    KanbanChecklistsListResponse,
    KanbanChecklistUpdate,
    KanbanChecklistWithItemsResponse,
    KanbanCommentCreate,
    KanbanCommentResponse,
    KanbanCommentsListResponse,
    KanbanCommentUpdate,
    KanbanDetailResponse,
    KanbanFilteredCardsResponse,
    KanbanLabelCreate,
    KanbanLabelResponse,
    KanbanLabelsListResponse,
    KanbanLabelUpdate,
    KanbanListCreate,
    KanbanListResponse,
    KanbanListsListResponse,
    KanbanListUpdate,
    KanbanLinkedCardsListResponse,
    KanbanReorderRequest,
    KanbanReorderResponse,
    KanbanSearchRequest,
    KanbanSearchResponse,
    KanbanToggleAllChecklistItemsRequest,
)
from .meetings_schemas import (
    MeetingArtifactCreate,
    MeetingArtifactResponse,
    MeetingFinalizeRequest,
    MeetingFinalizeResponse,
    MeetingHealthResponse,
    MeetingSessionCreate,
    MeetingSessionResponse,
    MeetingSessionStatus,
    MeetingSessionStatusUpdate,
    MeetingShareRequest,
    MeetingShareResponse,
    MeetingTemplateCreate,
    MeetingTemplateResponse,
    MeetingTemplateScope,
)
from .slides_schemas import (
    GenerateFromChatRequest,
    GenerateFromMediaRequest,
    GenerateFromNotesRequest,
    GenerateFromPromptRequest,
    GenerateFromRagRequest,
    PresentationCreateRequest,
    PresentationListResponse,
    PresentationPatchRequest,
    PresentationRenderArtifactListResponse,
    PresentationRenderJobResponse,
    PresentationRenderJobStatusResponse,
    PresentationRenderRequest,
    PresentationReorderRequest,
    PresentationResponse,
    PresentationSearchResponse,
    PresentationUpdateRequest,
    PresentationVersionListResponse,
    SlidesExportFormat,
    SlidesHealthResponse,
    SlidesTemplateListResponse,
    SlidesTemplateResponse,
    VisualStyleCreateRequest,
    VisualStyleListResponse,
    VisualStylePatchRequest,
    VisualStyleResponse,
)
from .audio_schemas import (
    AudioJobResponse,
    AudioSpeechJobArtifactsResponse,
    AudioSpeechJobCreateResponse,
    AudioTokenizerDecodeRequest,
    AudioTokenizerEncodeRequest,
    AudioTokenizerEncodeResponse,
    AudioTranscriptionRequest,
    AudioTranscriptionResponse,
    AudioTranslationRequest,
    CustomVoiceDeleteResponse,
    CustomVoiceListResponse,
    CustomVoiceResponse,
    OpenAISpeechRequest,
    SpeechChatRequest,
    SpeechChatResponse,
    StreamingLimitsResponse,
    StreamingStatusResponse,
    StreamingTestResponse,
    SubmitAudioJobRequest,
    SubmitAudioJobResponse,
    TTSHealthResponse,
    TTSHistoryDetailResponse,
    TTSHistoryFavoriteUpdate,
    TTSHistoryListResponse,
    TTSProvidersResponse,
    TTSVoicesResponse,
    VoiceEncodeRequest,
    VoiceEncodeResponse,
)
from .audiobook_schemas import (
    AudiobookArtifactsResponse,
    AudiobookChapterListResponse,
    AudiobookJobCreateResponse,
    AudiobookJobRequest,
    AudiobookJobStatusResponse,
    AudiobookParseRequest,
    AudiobookParseResponse,
    AudiobookProjectListResponse,
    AudiobookProjectResponse,
    SubtitleExportRequest,
    VoiceProfileCreateRequest,
    VoiceProfileDeleteResponse,
    VoiceProfileListResponse,
    VoiceProfileResponse,
)
from .rag_admin_schemas import (
    BatchMediaEmbeddingsRequest,
    BatchMediaEmbeddingsResponse,
    ChunkingTemplateApplyRequest,
    ChunkingTemplateApplyResponse,
    ChunkingTemplateCreateRequest,
    ChunkingTemplateDiagnosticsResponse,
    ChunkingTemplateLearnRequest,
    ChunkingTemplateLearnResponse,
    ChunkingTemplateListResponse,
    ChunkingTemplateMatchResponse,
    ChunkingTemplateResponse,
    ChunkingTemplateUpdateRequest,
    ChunkingTemplateValidationResponse,
    EmbeddingCollectionCreateRequest,
    EmbeddingCollectionListResponse,
    EmbeddingCollectionResponse,
    EmbeddingCollectionStatsResponse,
    MediaEmbeddingJobListResponse,
    MediaEmbeddingJobResponse,
    MediaEmbeddingsBatchRequest,
    MediaEmbeddingsBatchResponse,
    MediaEmbeddingsGenerateRequest,
    MediaEmbeddingsGenerateResponse,
    MediaEmbeddingsSearchRequest,
    MediaEmbeddingsSearchResponse,
    MediaEmbeddingsStatusResponse,
)
from .evaluations_schemas import (
    BatchEvaluationRequest,
    BatchEvaluationResponse,
    CreateEvaluationRequest,
    EmbeddingsABTestCreateRequest,
    EmbeddingsABTestCreateResponse,
    EmbeddingsABTestResultSummary,
    EmbeddingsABTestResultsResponse,
    EmbeddingsABTestRunRequest,
    EmbeddingsABTestStatusResponse,
    EvaluationBenchmarkListResponse,
    EvaluationBenchmarkRunRequest,
    EvaluationBenchmarkRunResponse,
    EvaluationDatasetCreateRequest,
    EvaluationDatasetListResponse,
    EvaluationDatasetResponse,
    EvaluationHistoryRequest,
    EvaluationHistoryResponse,
    EvaluationListResponse,
    EvaluationRecipeDatasetValidationRequest,
    EvaluationRecipeDatasetValidationResponse,
    EvaluationRecipeLaunchReadiness,
    EvaluationRecipeManifest,
    EvaluationRecipeRunCreateRequest,
    EvaluationRecipeRunRecord,
    EvaluationRecipeRunReport,
    EvaluationResponse,
    EvaluationWebhookRegistrationRequest,
    EvaluationWebhookRegistrationResponse,
    EvaluationWebhookStatusResponse,
    EvaluationWebhookTestRequest,
    EvaluationWebhookTestResponse,
    GEvalRequest,
    GEvalResponse,
    OCREvaluationRequest,
    OCREvaluationResponse,
    PipelineCleanupResponse,
    PipelinePresetCreate,
    PipelinePresetListResponse,
    PipelinePresetResponse,
    PropositionEvaluationRequest,
    PropositionEvaluationResponse,
    RAGEvaluationRequest,
    RAGEvaluationResponse,
    ResponseQualityRequest,
    ResponseQualityResponse,
    WebhookEventType,
    EvaluationRunCreateRequest,
    EvaluationRunListResponse,
    EvaluationRunResponse,
    RecipeDatasetValidationRequest,
    RecipeDatasetValidationResponse,
    RecipeLaunchReadiness,
    RecipeManifest,
    RecipeRunCreateRequest,
    RecipeRunRecord,
    SyntheticEvalGenerationRequest,
    SyntheticEvalGenerationResponse,
    SyntheticEvalPromotionRequest,
    SyntheticEvalPromotionResponse,
    SyntheticEvalQueueResponse,
    SyntheticEvalReviewActionRecord,
    SyntheticEvalReviewRequest,
    UpdateEvaluationRequest,
    WebhookRegistrationRequest,
    WebhookRegistrationResponse,
    WebhookStatusResponse,
    WebhookTestRequest,
    WebhookTestResponse,
)
from .flashcards_schemas import (
    FlashcardAnalyticsSummaryResponse,
    FlashcardAssetMetadata,
    FlashcardBulkUpdateItem,
    FlashcardBulkUpdateResponse,
    FlashcardCreateRequest,
    FlashcardDeckCreateRequest,
    FlashcardDeckResponse,
    FlashcardDeckUpdateRequest,
    FlashcardGenerateRequest,
    FlashcardGenerateResponse,
    FlashcardListResponse,
    FlashcardNextReviewResponse,
    FlashcardResetSchedulingRequest,
    FlashcardResponse,
    FlashcardResetSchedulingRequest,
    FlashcardReviewRequest,
    FlashcardReviewResponse,
    FlashcardReviewSessionEndRequest,
    FlashcardReviewSessionSummary,
    FlashcardTagSuggestionsResponse,
    FlashcardTagsUpdate,
    FlashcardTemplateCreateRequest,
    FlashcardTemplateListResponse,
    FlashcardTemplateResponse,
    FlashcardTemplateUpdateRequest,
    FlashcardUpdateRequest,
    FlashcardsImportRequest,
    StructuredQaImportPreviewRequest,
    StructuredQaImportPreviewResponse,
    StudyAssistantContextResponse,
    StudyAssistantRespondRequest,
    StudyAssistantRespondResponse,
    StudyPackCreateJobRequest,
    StudyPackJobAcceptedResponse,
    StudyPackJobStatusResponse,
    StudyPackSummaryResponse,
)
from .study_suggestions_schemas import (
    SuggestionActionRequest,
    SuggestionActionResponse,
    SuggestionJobAcceptedResponse,
    SuggestionRefreshRequest,
    SuggestionSnapshotResponse,
    SuggestionStatusResponse,
)
from .writing_manuscript_schemas import (
    ManuscriptAnalysisListResponse,
    ManuscriptAnalysisRequest,
    ManuscriptAnalysisResponse,
    ManuscriptCharacterCreate,
    ManuscriptCharacterResponse,
    ManuscriptCharacterUpdate,
    ManuscriptChapterCreate,
    ManuscriptChapterResponse,
    ManuscriptChapterUpdate,
    ManuscriptCitationCreate,
    ManuscriptCitationResponse,
    ManuscriptPartCreate,
    ManuscriptPartResponse,
    ManuscriptPartUpdate,
    ManuscriptPlotEventCreate,
    ManuscriptPlotEventResponse,
    ManuscriptPlotEventUpdate,
    ManuscriptPlotHoleCreate,
    ManuscriptPlotHoleResponse,
    ManuscriptPlotHoleUpdate,
    ManuscriptPlotLineCreate,
    ManuscriptPlotLineResponse,
    ManuscriptPlotLineUpdate,
    ManuscriptProjectCreate,
    ManuscriptProjectListResponse,
    ManuscriptProjectResponse,
    ManuscriptProjectUpdate,
    ManuscriptRelationshipCreate,
    ManuscriptRelationshipResponse,
    ManuscriptResearchRequest,
    ManuscriptResearchResponse,
    ManuscriptSceneCreate,
    ManuscriptSceneResponse,
    ManuscriptSceneUpdate,
    ManuscriptStructureResponse,
    ManuscriptWorldInfoCreate,
    ManuscriptWorldInfoResponse,
    ManuscriptWorldInfoUpdate,
    ReorderRequest,
    SceneCharacterLink,
    SceneCharacterLinkResponse,
    SceneWorldInfoLink,
    SceneWorldInfoLinkResponse,
)
from .quizzes_schemas import (
    QuizAttemptListResponse,
    QuizAttemptResponse,
    QuizAttemptSubmitRequest,
    QuizCreateRequest,
    QuizListResponse,
    QuizQuestionCreateRequest,
    QuizQuestionListResponse,
    QuizQuestionResponse,
    QuizQuestionUpdateRequest,
    QuizResponse,
    QuizUpdateRequest,
)
from .research_runs_schemas import (
    ResearchArtifactResponse,
    ResearchCheckpointPatchApproveRequest,
    ResearchRunCreateRequest,
    ResearchRunListItemResponse,
    ResearchRunResponse,
    ResearchRunStreamEvent,
)
from .research_search_schemas import (
    PaperSearchDetailRequest,
    PaperSearchDetailResponse,
    PaperSearchIngestRequest,
    PaperSearchListResponse,
    PaperSearchOperationResponse,
    PaperSearchRequest,
    WebSearchAggregateResponse,
    WebSearchRawResponse,
    WebSearchRequest,
)
from .writing_manuscript_schemas import (
    ManuscriptChapterCreateRequest,
    ManuscriptChapterResponse,
    ManuscriptChapterUpdateRequest,
    ManuscriptPartCreateRequest,
    ManuscriptPartResponse,
    ManuscriptPartUpdateRequest,
    ManuscriptProjectCreateRequest,
    ManuscriptProjectListResponse,
    ManuscriptProjectResponse,
    ManuscriptProjectUpdateRequest,
    ManuscriptSceneCreateRequest,
    ManuscriptSceneResponse,
    ManuscriptSceneUpdateRequest,
    ManuscriptSearchResponse,
    ManuscriptStructureResponse,
    ReorderRequest,
)
from .chat_conversation_schemas import (
    ChatAnalyticsResponse,
    ChatCommandsListResponse,
    ChatKnowledgeSaveRequest,
    ChatKnowledgeSaveResponse,
    ConversationShareLinkCreateRequest,
    ConversationShareLinkResponse,
    ConversationShareLinkRevokeResponse,
    ConversationShareLinksResponse,
    ConversationScopeParams,
    SharedConversationResolveResponse,
    ConversationUpdateRequest,
    ValidateDictionaryRequest,
    ValidateDictionaryResponse,
    normalize_conversation_state,
)
from .chat_loop_schemas import (
    ChatLoopActionResponse,
    ChatLoopApprovalDecisionRequest,
    ChatLoopEventsResponse,
    ChatLoopStartRequest,
    ChatLoopStartResponse,
)
from .chat_documents_schemas import (
    AsyncGenerationResponse,
    BulkGenerateRequest,
    BulkGenerateResponse,
    DocumentListResponse,
    DocumentType,
    GenerateDocumentRequest,
    GenerateDocumentResponse,
    GeneratedDocument,
    GenerationStatistics,
    JobStatusResponse,
    PromptConfigResponse,
    SavePromptConfigRequest,
)
from .character_persona_schemas import (
    ArchetypePreviewResponse,
    ArchetypeSummary,
    ArchetypeTemplate,
    CharacterChatSessionCreate,
    CharacterChatSessionUpdate,
    CharacterCreateRequest,
    CharacterMessageCreate,
    CharacterMessageListResponse,
    CharacterMessageResponse,
    CharacterMessageUpdate,
    CharacterExemplarCreate,
    CharacterExemplarSearchRequest,
    CharacterExemplarSelectionDebugRequest,
    CharacterExemplarUpdate,
    CharacterMemoryArchiveRequest,
    CharacterMemoryCreate,
    CharacterMemoryExtractRequest,
    CharacterMemoryUpdate,
    CharacterQueryRequest,
    CharacterResponse,
    CharacterUpdateRequest,
    ChatSettingsUpdate,
    GreetingListResponse,
    GreetingSelectRequest,
    PersonaExemplarResponse,
    PersonaExemplarCreate,
    PersonaExemplarImportRequest,
    PersonaExemplarReviewRequest,
    PersonaExemplarUpdate,
    PersonaInfo,
    PersonaProfileCreate,
    PersonaProfileResponse,
    PersonaProfileUpdate,
    PresetListResponse,
    PresetCreate,
    PresetUpdate,
)
from .chat_dictionary_schemas import (
    BulkDictionaryEntryOperationRequest,
    ChatDictionaryCreateRequest,
    ChatDictionaryUpdateRequest,
    DictionaryEntryCreateRequest,
    DictionaryEntryReorderRequest,
    DictionaryEntryUpdateRequest,
    ImportDictionaryJSONRequest,
    ImportDictionaryMarkdownRequest,
    ProcessChatDictionariesRequest,
    ValidateDictionaryRequest,
)
from .watchlists_schemas import (
    SourceCreateRequest,
    SourceDeleteResponse,
    SourceListResponse,
    SourceResponse,
    SourceUpdateRequest,
    WatchlistFiltersPayload,
    WatchlistGroupCreateRequest,
    WatchlistGroupListResponse,
    WatchlistGroupResponse,
    WatchlistGroupUpdateRequest,
    WatchlistJobCreateRequest,
    WatchlistJobDeleteResponse,
    WatchlistJobListResponse,
    WatchlistJobResponse,
    WatchlistJobUpdateRequest,
    WatchlistAlertRuleCreateRequest,
    WatchlistAlertRuleDeleteResponse,
    WatchlistAlertRuleListResponse,
    WatchlistAlertRuleResponse,
    WatchlistAlertRuleUpdateRequest,
    WatchlistOutputCreateRequest,
    WatchlistOutputListResponse,
    WatchlistOutputResponse,
    WatchlistPreviewResponse,
    WatchlistScrapedItemListResponse,
    WatchlistScrapedItemResponse,
    WatchlistScrapedItemSmartCountsResponse,
    WatchlistScrapedItemUpdateRequest,
    WatchlistSourceBulkCreateRequest,
    WatchlistSourceBulkCreateResponse,
    WatchlistSourceCheckNowRequest,
    WatchlistSourceCheckNowResponse,
    WatchlistSourceImportResponse,
    WatchlistSourceSeenResetResponse,
    WatchlistSourceSeenStatsResponse,
    WatchlistSourceTestRequest,
    WatchlistTagListResponse,
    WatchlistTemplateComposerFlowCheckRequest,
    WatchlistTemplateComposerFlowCheckResponse,
    WatchlistTemplateComposerSectionRequest,
    WatchlistTemplateComposerSectionResponse,
    WatchlistTemplateCreateRequest,
    WatchlistTemplateDetailResponse,
    WatchlistTemplateListResponse,
    WatchlistTemplatePreviewRequest,
    WatchlistTemplatePreviewResponse,
    WatchlistTemplateValidationRequest,
    WatchlistTemplateValidationResponse,
    WatchlistTemplateVersionsResponse,
    WatchlistRunCancelResponse,
    WatchlistRunDetailResponse,
    WatchlistRunListResponse,
    WatchlistRunResponse,
)
from .notifications_reminders_schemas import (
    NotificationCancelSnoozeResponse,
    NotificationDismissResponse,
    NotificationPreferencesResponse,
    NotificationPreferencesUpdateRequest,
    NotificationSnoozeRequest,
    NotificationSnoozeResponse,
    NotificationStreamEvent,
    NotificationsListResponse,
    NotificationsMarkReadRequest,
    NotificationsMarkReadResponse,
    NotificationsUnreadCountResponse,
    ReminderTaskCreateRequest,
    ReminderTaskDeleteResponse,
    ReminderTaskListResponse,
    ReminderTaskResponse,
    ReminderTaskUpdateRequest,
)
from .server_notifications_schemas import (
    ServerNotificationStreamEvent,
)
from .outputs_schemas import (
    OutputArtifact,
    OutputCreateRequest,
    OutputDeleteResponse,
    OutputListResponse,
    OutputTemplate,
    OutputTemplateCreate,
    OutputTemplateList,
    OutputTemplateUpdate,
    OutputUpdateRequest,
    OutputsPurgeRequest,
    OutputsPurgeResponse,
    TemplatePreviewRequest,
    TemplatePreviewResponse,
)
from .research_runs_schemas import (
    ResearchArtifactResponse,
    ResearchCheckpointPatchApproveRequest,
    ResearchRunCreateRequest,
    ResearchRunListItemResponse,
    ResearchRunResponse,
    ResearchRunStreamEvent,
)
from .research_search_schemas import (
    ArxivPaper,
    ArxivSearchResponse,
    BioRxivPaper,
    BioRxivSearchResponse,
    PubMedPaper,
    PubMedSearchResponse,
    SemanticScholarPaper,
    SemanticScholarSearchResponse,
    WebSearchAggregateResponse,
    WebSearchRawResponse,
    WebSearchRequest,
)
from .sharing_schemas import (
    CloneWorkspaceRequest,
    CloneWorkspaceResponse,
    CreateTokenRequest,
    PublicShareImportResponse,
    PublicSharePreview,
    ShareListResponse,
    ShareResponse,
    ShareWorkspaceRequest,
    SharedChatRequest,
    SharedMediaResponse,
    SharedWithMeResponse,
    SharedWorkspaceResponse,
    SharedWorkspaceSourceResponse,
    TokenListResponse,
    TokenResponse,
    UpdateShareRequest,
    VerifyPasswordRequest,
    VerifyPasswordResponse,
)
from .web_clipper_schemas import (
    WebClipperEnrichmentPayload,
    WebClipperEnrichmentResponse,
    WebClipperSaveRequest,
    WebClipperSaveResponse,
    WebClipperStatusResponse,
)
from .exceptions import APIConnectionError, APIRequestError, APIResponseError, AuthenticationError
from .utils import model_to_form_data, prepare_files_for_httpx, cleanup_file_objects
#
########################################################################################################################
#
# Functions:

class TLDWAPIClient:
    def __init__(self, base_url: str, token: Optional[str] = None, timeout: float = 300.0):
        self.base_url = base_url.rstrip('/')
        self.token = token
        self.bearer_token = None
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers = {}
            if self.bearer_token:
                # Bearer Auth
                headers["Authorization"] = f"Bearer {self.bearer_token}"
            if self.token:
                # Token Auth
                headers["X-API-KEY"] = self.token
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout,
                follow_redirects=True
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    def set_bearer_token(self, token: str | None) -> None:
        self.bearer_token = token
        if self._client is None or self._client.is_closed:
            return
        if token:
            self._client.headers["Authorization"] = f"Bearer {token}"
        elif "Authorization" in self._client.headers:
            del self._client.headers["Authorization"]

    def _normalize_conversation_scope_params(
        self,
        scope_type: Optional[Literal["global", "workspace"]] = None,
        workspace_id: Optional[str] = None,
    ) -> Optional[ConversationScopeParams]:
        if scope_type is None and workspace_id is None:
            return None
        effective_scope_type = scope_type or ("workspace" if workspace_id else "global")
        return ConversationScopeParams(scope_type=effective_scope_type, workspace_id=workspace_id)

    @staticmethod
    def _dump_request_payload(
        request_data: Any,
        *,
        exclude_none: bool = True,
        exclude_unset: bool = False,
    ) -> Dict[str, Any]:
        if hasattr(request_data, "model_dump"):
            return request_data.model_dump(
                exclude_none=exclude_none,
                exclude_unset=exclude_unset,
                mode="json",
            )
        return dict(request_data)

    @staticmethod
    def _if_match_header(if_match: str | None) -> Optional[Dict[str, str]]:
        if if_match is None:
            return None
        return {"If-Match": if_match}

    @staticmethod
    def _expected_version_header(expected_version: int | None) -> Optional[Dict[str, str]]:
        if expected_version is None:
            return None
        return {"X-Expected-Version": str(expected_version)}

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None, # Changed from BaseModel to Dict
        files: Optional[List[tuple]] = None, # For httpx files format
        json_data: Optional[Union[Dict[str, Any], List[Any]]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        client = await self._get_client()
        url = f"{self.base_url}{endpoint}" # Ensure base_url doesn't make double slash

        try:
            # httpx expects 'data' for form-encoded and 'files' for multipart
            response = await client.request(
                method,
                endpoint,
                data=data,
                files=files,
                json=json_data,
                params=params,
                headers=headers,
            ) # Pass endpoint directly
            response.raise_for_status()  # Raises HTTPStatusError for 4xx/5xx
            if response.status_code in {204, 205}:
                return {}
            return response.json()
        except httpx.HTTPStatusError as e:
            # Try to get more details from response if available
            error_detail = str(e)
            response_data = None
            try:
                response_data = e.response.json()
                if isinstance(response_data, dict) and "detail" in response_data:
                    if isinstance(response_data["detail"], list) and response_data["detail"]:
                        # Pydantic validation error format
                        error_detail = f"Validation Error: {response_data['detail'][0].get('msg', '')} for field '{'.'.join(map(str, response_data['detail'][0].get('loc', [])))}'"
                    elif isinstance(response_data["detail"], str):
                        error_detail = response_data["detail"]
            except Exception:
                pass # Ignore if response is not JSON or detail not found

            if e.response.status_code == 401:
                raise AuthenticationError(
                    f"Authentication failed: {error_detail}",
                    response_data=response_data,
                )
            elif e.response.status_code == 422: # Unprocessable Entity (Pydantic validation error)
                raise APIRequestError(f"Validation Error: {error_detail}", response_data=response_data)
            raise APIResponseError(e.response.status_code, error_detail, response_data=response_data)
        except httpx.RequestError as e: # Covers ConnectError, TimeoutException, etc.
            raise APIConnectionError(f"Connection error to {url}: {e}")
        except json.JSONDecodeError:
            raise APIResponseError(response.status_code, "Failed to decode JSON response", response_data={"raw_text": response.text})

    @staticmethod
    def _filename_from_content_disposition(content_disposition: str | None) -> str | None:
        if not content_disposition:
            return None
        for part in content_disposition.split(";"):
            key, _, raw_value = part.strip().partition("=")
            if key.lower() != "filename" or not raw_value:
                continue
            return raw_value.strip().strip('"')
        return None

    async def _binary_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[List[tuple]] = None,
        json_data: Optional[Union[Dict[str, Any], List[Any]]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> ReadingExportResponse:
        client = await self._get_client()
        url = f"{self.base_url}{endpoint}"

        try:
            response = await client.request(
                method,
                endpoint,
                data=data,
                files=files,
                json=json_data,
                params=params,
                headers=headers,
            )
            response.raise_for_status()
            content_disposition = response.headers.get("content-disposition")
            return ReadingExportResponse(
                content=response.content,
                content_type=response.headers.get("content-type"),
                content_disposition=content_disposition,
                filename=self._filename_from_content_disposition(content_disposition),
            )
        except httpx.HTTPStatusError as e:
            error_detail = str(e)
            response_data = None
            try:
                response_data = e.response.json()
                if isinstance(response_data, dict) and isinstance(response_data.get("detail"), str):
                    error_detail = response_data["detail"]
            except Exception:
                response_data = {"raw_text": e.response.text}
            if e.response.status_code == 401:
                raise AuthenticationError(
                    f"Authentication failed: {error_detail}",
                    response_data=response_data,
                )
            elif e.response.status_code == 422:
                raise APIRequestError(f"Validation Error: {error_detail}", response_data=response_data)
            raise APIResponseError(e.response.status_code, error_detail, response_data=response_data)
        except httpx.RequestError as e:
            raise APIConnectionError(f"Connection error to {url}: {e}")

    async def _request_bytes(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[List[tuple]] = None,
        json_data: Optional[Union[Dict[str, Any], List[Any]]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> bytes:
        response = await self._binary_request(
            method,
            endpoint,
            data=data,
            files=files,
            json_data=json_data,
            params=params,
            headers=headers,
        )
        return response.content

    async def _headers_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        client = await self._get_client()
        url = f"{self.base_url}{endpoint}"

        try:
            response = await client.request(method, endpoint, params=params, headers=headers)
            response.raise_for_status()
            return {str(key).lower(): str(value) for key, value in response.headers.items()}
        except httpx.HTTPStatusError as e:
            error_detail = str(e)
            response_data = None
            try:
                response_data = e.response.json()
                if isinstance(response_data, dict) and isinstance(response_data.get("detail"), str):
                    error_detail = response_data["detail"]
            except Exception:
                response_data = {"raw_text": e.response.text}
            if e.response.status_code == 401:
                raise AuthenticationError(
                    f"Authentication failed: {error_detail}",
                    response_data=response_data,
                )
            elif e.response.status_code == 422:
                raise APIRequestError(f"Validation Error: {error_detail}", response_data=response_data)
            raise APIResponseError(e.response.status_code, error_detail, response_data=response_data)
        except httpx.RequestError as e:
            raise APIConnectionError(f"Connection error to {url}: {e}")

    async def _stream_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[List[tuple]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        client = await self._get_client()
        url = f"{self.base_url}{endpoint}"

        try:
            async with client.stream(method, endpoint, data=data, files=files) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            # Log or handle malformed JSON lines if necessary
                            print(f"Warning: Could not decode JSON line: {line}")
        except httpx.HTTPStatusError as e:
            error_detail = str(e)
            # Stream errors are harder to parse nicely, attempt if possible
            response_text = ""
            try:
                response_text = await e.response.aread() # read the body
                response_data = json.loads(response_text)
                if isinstance(response_data, dict) and "detail" in response_data:
                     error_detail = response_data["detail"]
            except Exception:
                response_data = {"raw_text": response_text}
                pass
            if e.response.status_code == 401:
                raise AuthenticationError(
                    f"Authentication failed: {error_detail}",
                    response_data=response_data if isinstance(response_data, dict) else None,
                )
            raise APIResponseError(e.response.status_code, error_detail, response_data={"raw_text": response_text})
        except httpx.RequestError as e:
            raise APIConnectionError(f"Connection error to {url}: {e}")

    async def _sse_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        client = await self._get_client()
        url = f"{self.base_url}{endpoint}"
        event_name = "message"
        event_id: str | None = None
        data_lines: list[str] = []

        async def _flush_event() -> Dict[str, Any] | None:
            nonlocal event_name, event_id, data_lines
            if not data_lines:
                event_name = "message"
                event_id = None
                return None
            raw_data = "\n".join(data_lines)
            try:
                payload = json.loads(raw_data)
            except json.JSONDecodeError:
                payload = {"raw": raw_data}
            event = {"event": event_name, "data": payload, "event_id": event_id}
            event_name = "message"
            event_id = None
            data_lines = []
            return event

        try:
            async with client.stream(
                method,
                endpoint,
                params=params,
                headers=headers,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line == "":
                        event = await _flush_event()
                        if event is not None:
                            yield event
                        continue
                    if line.startswith(":"):
                        continue
                    field, _, value = line.partition(":")
                    if value.startswith(" "):
                        value = value[1:]
                    if field == "event":
                        event_name = value or "message"
                    elif field == "id":
                        event_id = value
                    elif field == "data":
                        data_lines.append(value)
                event = await _flush_event()
                if event is not None:
                    yield event
        except httpx.HTTPStatusError as e:
            error_detail = str(e)
            response_data = None
            try:
                response_data = e.response.json()
                if isinstance(response_data, dict) and isinstance(response_data.get("detail"), str):
                    error_detail = response_data["detail"]
            except Exception:
                pass
            if e.response.status_code == 401:
                raise AuthenticationError(
                    f"Authentication failed: {error_detail}",
                    response_data=response_data,
                )
            raise APIResponseError(e.response.status_code, error_detail, response_data=response_data)
        except httpx.RequestError as e:
            raise APIConnectionError(f"Connection error to {url}: {e}")

    async def _stream_sse_request(
        self,
        endpoint: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        event_model: type[BaseModel] | None = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> AsyncGenerator[Any, None]:
        async for event in self._sse_request("GET", endpoint, params=params, headers=headers):
            yield event_model.model_validate(event) if event_model is not None else event

    async def list_server_notes(self, limit: int = 100, offset: int = 0, include_keywords: bool = True) -> Dict[str, Any]:
        return await self._request(
            "GET",
            "/api/v1/notes/",
            params={
                "limit": limit,
                "offset": offset,
                "include_keywords": str(include_keywords).lower(),
            },
        )

    async def search_server_notes(
        self,
        query: Optional[str] = None,
        tokens: Optional[List[str]] = None,
        limit: int = 10,
        offset: int = 0,
        include_keywords: bool = False,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"limit": limit, "offset": offset, "include_keywords": str(include_keywords).lower()}
        if query is not None:
            params["query"] = query
        if tokens is not None:
            params["tokens"] = tokens
        return await self._request("GET", "/api/v1/notes/search", params=params)

    async def get_server_note(self, note_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/api/v1/notes/{note_id}")

    async def create_server_note(self, request_data: NoteCreateRequest) -> Dict[str, Any]:
        return await self._request(
            "POST",
            "/api/v1/notes/",
            json_data=request_data.model_dump(exclude_none=True, exclude_defaults=True, mode="json"),
        )

    async def update_server_note(
        self,
        note_id: str,
        request_data: NoteUpdateRequest,
        expected_version: int,
    ) -> Dict[str, Any]:
        return await self._request(
            "PUT",
            f"/api/v1/notes/{note_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
            headers={"expected-version": str(expected_version)},
        )

    async def delete_server_note(self, note_id: str, expected_version: int) -> Dict[str, Any]:
        return await self._request(
            "DELETE",
            f"/api/v1/notes/{note_id}",
            headers={"expected-version": str(expected_version)},
        )

    @staticmethod
    def _normalize_api_namespace_endpoint(namespace: str, endpoint: str) -> str:
        namespace = str(namespace or "").strip("/")
        if not namespace:
            raise ValueError("API namespace is required.")
        namespace_path = f"/api/v1/{namespace}"
        raw_endpoint = str(endpoint or "").strip()
        if raw_endpoint.startswith(f"{namespace_path}/"):
            raw_endpoint = raw_endpoint.removeprefix(f"{namespace_path}/")
        elif raw_endpoint in {namespace_path, namespace_path.lstrip("/")}:
            raw_endpoint = ""
        elif raw_endpoint.startswith("/") or raw_endpoint.startswith("api/"):
            raise ValueError(f"{namespace} gateway endpoints must stay inside the {namespace} namespace.")
        trailing_slash = raw_endpoint.endswith("/")
        normalized = raw_endpoint.strip("/")
        if ".." in normalized.split("/"):
            raise ValueError(f"unsafe {namespace} gateway endpoint.")
        if not normalized:
            return f"{namespace_path}/"
        suffix = "/" if trailing_slash else ""
        return f"{namespace_path}/{normalized}{suffix}"

    @staticmethod
    def _normalize_notes_namespace_endpoint(endpoint: str) -> str:
        return TLDWAPIClient._normalize_api_namespace_endpoint("notes", endpoint)

    async def _call_server_api_namespace_endpoint(
        self,
        namespace: str,
        method: str,
        endpoint: str,
        *,
        params: Dict[str, Any] | None = None,
        payload: Dict[str, Any] | list[Any] | None = None,
        data: Dict[str, Any] | None = None,
        files: list[tuple] | None = None,
        headers: Dict[str, str] | None = None,
    ) -> Any:
        normalized_method = str(method or "").upper()
        if normalized_method not in {"GET", "POST", "PATCH", "PUT", "DELETE"}:
            raise ValueError(f"Unsupported {namespace} gateway method: {method}")
        return await self._request(
            normalized_method,
            self._normalize_api_namespace_endpoint(namespace, endpoint),
            params=params,
            json_data=payload,
            data=data,
            files=files,
            headers=headers,
        )

    async def call_server_notes_endpoint(
        self,
        method: str,
        endpoint: str,
        *,
        params: Dict[str, Any] | None = None,
        payload: Dict[str, Any] | list[Any] | None = None,
        data: Dict[str, Any] | None = None,
        files: list[tuple] | None = None,
        headers: Dict[str, str] | None = None,
    ) -> Any:
        return await self._call_server_api_namespace_endpoint(
            "notes",
            method,
            endpoint,
            params=params,
            payload=payload,
            data=data,
            files=files,
            headers=headers,
        )

    async def call_server_acp_endpoint(
        self,
        method: str,
        endpoint: str,
        *,
        params: Dict[str, Any] | None = None,
        payload: Dict[str, Any] | list[Any] | None = None,
        data: Dict[str, Any] | None = None,
        files: list[tuple] | None = None,
        headers: Dict[str, str] | None = None,
    ) -> Any:
        return await self._call_server_api_namespace_endpoint(
            "acp",
            method,
            endpoint,
            params=params,
            payload=payload,
            data=data,
            files=files,
            headers=headers,
        )

    def _notes_graph_query_params(self, request_data: NoteGraphRequest | Dict[str, Any]) -> Dict[str, Any]:
        payload = (
            request_data.model_dump(exclude_none=True, exclude_defaults=True, mode="json")
            if hasattr(request_data, "model_dump")
            else dict(request_data)
        )
        params: Dict[str, Any] = {}
        for key, value in payload.items():
            if key == "edge_types" and isinstance(value, list):
                params[key] = ",".join(str(item) for item in value)
                continue
            if key == "time_range" and isinstance(value, dict):
                if value.get("start") is not None:
                    params["time_range.start"] = value["start"]
                if value.get("end") is not None:
                    params["time_range.end"] = value["end"]
                continue
            if isinstance(value, bool):
                params[key] = str(value).lower()
                continue
            params[key] = value
        return params

    async def get_notes_graph(
        self,
        request_data: NoteGraphRequest | Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        request = request_data if request_data is not None else NoteGraphRequest(**kwargs)
        return await self._request(
            "GET",
            "/api/v1/notes/graph",
            params=self._notes_graph_query_params(request),
        )

    async def get_note_neighbors(
        self,
        note_id: str,
        request_data: NoteGraphRequest | Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if request_data is None:
            request = NoteGraphRequest(**kwargs)
        elif hasattr(request_data, "model_dump"):
            payload = request_data.model_dump(exclude_none=True, mode="json")
            payload.update(kwargs)
            request = NoteGraphRequest(**payload)
        else:
            payload = dict(request_data)
            payload.update(kwargs)
            request = NoteGraphRequest(**payload)
        return await self._request(
            "GET",
            f"/api/v1/notes/{note_id}/neighbors",
            params=self._notes_graph_query_params(request),
        )

    async def create_note_link(self, note_id: str, request_data: NoteLinkCreate | Dict[str, Any]) -> Dict[str, Any]:
        payload = (
            request_data.model_dump(exclude_none=True, mode="json")
            if hasattr(request_data, "model_dump")
            else dict(request_data)
        )
        return await self._request(
            "POST",
            f"/api/v1/notes/{note_id}/links",
            json_data=payload,
        )

    async def delete_note_link(self, edge_id: str) -> Dict[str, Any]:
        return await self._request("DELETE", f"/api/v1/notes/links/{edge_id}")

    async def list_workspaces(self) -> Dict[str, Any]:
        return await self._request("GET", "/api/v1/workspaces/")

    async def get_workspace(self, workspace_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/api/v1/workspaces/{workspace_id}")

    async def create_workspace(self, workspace_id: str, request_data: WorkspaceCreateRequest) -> Dict[str, Any]:
        return await self._request(
            "PUT",
            f"/api/v1/workspaces/{workspace_id}",
            json_data=request_data.model_dump(mode="json"),
        )

    async def update_workspace(self, workspace_id: str, request_data: WorkspaceUpdateRequest) -> Dict[str, Any]:
        return await self._request(
            "PATCH",
            f"/api/v1/workspaces/{workspace_id}",
            json_data=request_data.model_dump(exclude_unset=True, mode="json"),
        )

    async def delete_workspace(self, workspace_id: str) -> Dict[str, Any]:
        return await self._request("DELETE", f"/api/v1/workspaces/{workspace_id}")

    async def list_workspace_notes(self, workspace_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/api/v1/workspaces/{workspace_id}/notes")

    async def create_workspace_note(self, workspace_id: str, request_data: WorkspaceNoteCreateRequest) -> Dict[str, Any]:
        return await self._request(
            "POST",
            f"/api/v1/workspaces/{workspace_id}/notes",
            json_data=request_data.model_dump(mode="json"),
        )

    async def update_workspace_note(
        self,
        workspace_id: str,
        note_id: int,
        request_data: WorkspaceNoteUpdateRequest,
    ) -> Dict[str, Any]:
        return await self._request(
            "PUT",
            f"/api/v1/workspaces/{workspace_id}/notes/{note_id}",
            json_data=request_data.model_dump(exclude_unset=True, mode="json"),
        )

    async def delete_workspace_note(self, workspace_id: str, note_id: int) -> Dict[str, Any]:
        return await self._request("DELETE", f"/api/v1/workspaces/{workspace_id}/notes/{note_id}")

    async def list_workspace_sources(self, workspace_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/api/v1/workspaces/{workspace_id}/sources")

    async def create_workspace_source(
        self,
        workspace_id: str,
        request_data: WorkspaceSourceCreateRequest,
    ) -> Dict[str, Any]:
        return await self._request(
            "POST",
            f"/api/v1/workspaces/{workspace_id}/sources",
            json_data=request_data.model_dump(mode="json"),
        )

    async def update_workspace_source(
        self,
        workspace_id: str,
        source_id: str,
        request_data: WorkspaceSourceUpdateRequest,
    ) -> Dict[str, Any]:
        return await self._request(
            "PUT",
            f"/api/v1/workspaces/{workspace_id}/sources/{source_id}",
            json_data=request_data.model_dump(exclude_unset=True, mode="json"),
        )

    async def delete_workspace_source(self, workspace_id: str, source_id: str) -> Dict[str, Any]:
        return await self._request("DELETE", f"/api/v1/workspaces/{workspace_id}/sources/{source_id}")

    async def list_workspace_artifacts(self, workspace_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/api/v1/workspaces/{workspace_id}/artifacts")

    async def create_workspace_artifact(
        self,
        workspace_id: str,
        request_data: WorkspaceArtifactCreateRequest,
    ) -> Dict[str, Any]:
        return await self._request(
            "POST",
            f"/api/v1/workspaces/{workspace_id}/artifacts",
            json_data=request_data.model_dump(mode="json"),
        )

    async def update_workspace_artifact(
        self,
        workspace_id: str,
        artifact_id: str,
        request_data: WorkspaceArtifactUpdateRequest,
    ) -> Dict[str, Any]:
        return await self._request(
            "PUT",
            f"/api/v1/workspaces/{workspace_id}/artifacts/{artifact_id}",
            json_data=request_data.model_dump(exclude_unset=True, mode="json"),
        )

    async def delete_workspace_artifact(self, workspace_id: str, artifact_id: str) -> Dict[str, Any]:
        return await self._request("DELETE", f"/api/v1/workspaces/{workspace_id}/artifacts/{artifact_id}")

    async def list_media_items(
        self,
        page: int = 1,
        results_per_page: int = 10,
        include_keywords: bool = False,
    ) -> ServerMediaListResponse:
        response = await self._request(
            "GET",
            "/api/v1/media/",
            params={
                "page": page,
                "results_per_page": results_per_page,
                "include_keywords": str(include_keywords).lower(),
            },
        )
        return ServerMediaListResponse.model_validate(response)

    async def list_media_keywords(
        self,
        *,
        query: str | None = None,
        limit: int = 100,
    ) -> MediaKeywordListResponse:
        params = {"query": query, "limit": limit}
        response = await self._request(
            "GET",
            "/api/v1/media/keywords",
            params={key: value for key, value in params.items() if value is not None},
        )
        return MediaKeywordListResponse.model_validate(response)

    async def list_media_trash(
        self,
        page: int = 1,
        results_per_page: int = 10,
        include_keywords: bool = False,
    ) -> ServerMediaListResponse:
        response = await self._request(
            "GET",
            "/api/v1/media/trash",
            params={
                "page": page,
                "results_per_page": results_per_page,
                "include_keywords": str(include_keywords).lower(),
            },
        )
        return ServerMediaListResponse.model_validate(response)

    async def empty_media_trash(self) -> MediaTrashEmptyResponse:
        response = await self._request("POST", "/api/v1/media/trash/empty")
        return MediaTrashEmptyResponse.model_validate(response)

    async def search_media_metadata(
        self,
        *,
        filters: list[dict[str, Any]] | None = None,
        field: str | None = None,
        op: str | None = None,
        value: str | None = None,
        match_mode: str = "all",
        group_by_media: bool = True,
        page: int = 1,
        per_page: int = 20,
        q: str | None = None,
        media_types: list[str] | str | None = None,
        must_have: list[str] | str | None = None,
        must_not_have: list[str] | str | None = None,
        date_start: str | None = None,
        date_end: str | None = None,
        sort_by: str | None = None,
    ) -> MediaMetadataSearchResponse:
        def _csv(value: list[str] | str | None) -> str | None:
            if value is None:
                return None
            if isinstance(value, str):
                return value
            return ",".join(str(entry) for entry in value)

        params: Dict[str, Any] = {
            "filters": json.dumps(filters) if filters else None,
            "field": field,
            "op": op,
            "value": value,
            "match_mode": match_mode,
            "group_by_media": str(group_by_media).lower(),
            "page": page,
            "per_page": per_page,
            "q": q,
            "media_types": _csv(media_types),
            "must_have": _csv(must_have),
            "must_not_have": _csv(must_not_have),
            "date_start": date_start,
            "date_end": date_end,
            "sort_by": sort_by,
        }
        response = await self._request(
            "GET",
            "/api/v1/media/metadata-search",
            params={key: value for key, value in params.items() if value is not None},
        )
        return MediaMetadataSearchResponse.model_validate(response)

    async def get_media_by_identifier(
        self,
        *,
        doi: str | None = None,
        pmid: str | None = None,
        pmcid: str | None = None,
        arxiv_id: str | None = None,
        s2_paper_id: str | None = None,
        group_by_media: bool = True,
    ) -> MediaIdentifierLookupResponse:
        params: Dict[str, Any] = {
            "doi": doi,
            "pmid": pmid,
            "pmcid": pmcid,
            "arxiv_id": arxiv_id,
            "s2_paper_id": s2_paper_id,
            "group_by_media": str(group_by_media).lower(),
        }
        response = await self._request(
            "GET",
            "/api/v1/media/by-identifier",
            params={key: value for key, value in params.items() if value is not None},
        )
        return MediaIdentifierLookupResponse.model_validate(response)

    async def get_media_transcription_models(self) -> MediaTranscriptionModelsResponse:
        response = await self._request("GET", "/api/v1/media/transcription-models")
        return MediaTranscriptionModelsResponse.model_validate(response)

    async def get_media_item(
        self,
        media_id: int,
        *,
        include_content: bool = True,
        include_versions: bool = True,
        include_version_content: bool = False,
    ) -> MediaDetailResponse:
        response = await self._request(
            "GET",
            f"/api/v1/media/{media_id}",
            params={
                "include_content": str(include_content).lower(),
                "include_versions": str(include_versions).lower(),
                "include_version_content": str(include_version_content).lower(),
            },
        )
        return MediaDetailResponse.model_validate(response)

    async def update_media_item(
        self,
        media_id: int,
        request_data: MediaUpdateRequest,
    ) -> MediaDetailResponse:
        response = await self._request(
            "PUT",
            f"/api/v1/media/{media_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return MediaDetailResponse.model_validate(response)

    async def trash_media_item(self, media_id: int) -> Dict[str, Any]:
        response = await self._request("DELETE", f"/api/v1/media/{media_id}")
        return {"deleted": True, **response}

    async def restore_media_item(
        self,
        media_id: int,
        *,
        include_content: bool = True,
        include_versions: bool = True,
        include_version_content: bool = False,
    ) -> MediaDetailResponse:
        response = await self._request(
            "POST",
            f"/api/v1/media/{media_id}/restore",
            params={
                "include_content": str(include_content).lower(),
                "include_versions": str(include_versions).lower(),
                "include_version_content": str(include_version_content).lower(),
            },
        )
        return MediaDetailResponse.model_validate(response)

    async def permanently_delete_media_item(self, media_id: int) -> Dict[str, Any]:
        response = await self._request("DELETE", f"/api/v1/media/{media_id}/permanent")
        return {"deleted": True, **response}

    async def update_media_keywords(
        self,
        media_id: int,
        request_data: MediaKeywordsUpdateRequest,
    ) -> MediaKeywordsResponse:
        response = await self._request(
            "PATCH",
            f"/api/v1/media/{media_id}/keywords",
            json_data=request_data.model_dump(mode="json"),
        )
        return MediaKeywordsResponse.model_validate(response)

    async def download_media_file(self, media_id: int, *, file_type: str = "original") -> ReadingExportResponse | bytes:
        request_bytes_override = self.__dict__.get("_request_bytes")
        if request_bytes_override is not None:
            return await request_bytes_override(
                "GET",
                f"/api/v1/media/{media_id}/file",
                params={"file_type": file_type},
            )
        return await self._binary_request(
            "GET",
            f"/api/v1/media/{media_id}/file",
            params={"file_type": file_type},
        )

    async def get_media_navigation(
        self,
        media_id: int,
        *,
        include_generated_fallback: bool = False,
        max_depth: int = 4,
        max_nodes: int = 500,
        parent_id: str | None = None,
    ) -> MediaNavigationResponse:
        params: Dict[str, Any] = {
            "include_generated_fallback": str(include_generated_fallback).lower(),
            "max_depth": max_depth,
            "max_nodes": max_nodes,
            "parent_id": parent_id,
        }
        response = await self._request(
            "GET",
            f"/api/v1/media/{media_id}/navigation",
            params={key: value for key, value in params.items() if value is not None},
        )
        return MediaNavigationResponse.model_validate(response)

    async def get_media_navigation_content(
        self,
        media_id: int,
        node_id: str,
        *,
        content_format: str = "auto",
        include_alternates: bool = False,
    ) -> MediaNavigationContentResponse:
        response = await self._request(
            "GET",
            f"/api/v1/media/{media_id}/navigation/{node_id}/content",
            params={
                "format": content_format,
                "include_alternates": str(include_alternates).lower(),
            },
        )
        return MediaNavigationContentResponse.model_validate(response)

    async def add_media(
        self,
        request_data: AddMediaRequest,
        file_paths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        form_data = model_to_form_data(request_data)
        httpx_files = prepare_files_for_httpx(file_paths, upload_field_name="files")
        try:
            return await self._request("POST", "/api/v1/media/add", data=form_data, files=httpx_files)
        finally:
            cleanup_file_objects(httpx_files)

    async def list_media_keywords(self, *, query: str | None = None, limit: int = 100) -> MediaKeywordListResponse:
        params: Dict[str, Any] = {"limit": limit}
        if query is not None:
            params["query"] = query
        response = await self._request("GET", "/api/v1/media/keywords", params=params)
        return MediaKeywordListResponse.model_validate(response)

    async def list_media_trash(
        self,
        *,
        page: int = 1,
        results_per_page: int = 10,
        include_keywords: bool = False,
    ) -> ServerMediaListResponse:
        response = await self._request(
            "GET",
            "/api/v1/media/trash",
            params={
                "page": page,
                "results_per_page": results_per_page,
                "include_keywords": str(include_keywords).lower(),
            },
        )
        return ServerMediaListResponse.model_validate(response)

    async def empty_media_trash(self) -> MediaTrashEmptyResponse:
        response = await self._request("POST", "/api/v1/media/trash/empty")
        return MediaTrashEmptyResponse.model_validate(response)

    async def get_media_item(
        self,
        media_id: int,
        *,
        include_content: bool = True,
        include_versions: bool = True,
        include_version_content: bool = False,
    ) -> MediaDetailResponse:
        response = await self._request(
            "GET",
            f"/api/v1/media/{media_id}",
            params={
                "include_content": str(include_content).lower(),
                "include_versions": str(include_versions).lower(),
                "include_version_content": str(include_version_content).lower(),
            },
        )
        return MediaDetailResponse.model_validate(response)

    async def update_media_item(self, media_id: int, request_data: MediaItemUpdateRequest) -> MediaDetailResponse:
        response = await self._request(
            "PUT",
            f"/api/v1/media/{media_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return MediaDetailResponse.model_validate(response)

    async def delete_media_item(self, media_id: int) -> Dict[str, Any]:
        return await self._request("DELETE", f"/api/v1/media/{media_id}")

    async def restore_media_item(
        self,
        media_id: int,
        *,
        include_content: bool = True,
        include_versions: bool = True,
        include_version_content: bool = False,
    ) -> MediaDetailResponse:
        response = await self._request(
            "POST",
            f"/api/v1/media/{media_id}/restore",
            params={
                "include_content": str(include_content).lower(),
                "include_versions": str(include_versions).lower(),
                "include_version_content": str(include_version_content).lower(),
            },
        )
        return MediaDetailResponse.model_validate(response)

    async def permanently_delete_media_item(self, media_id: int) -> Dict[str, Any]:
        response = await self._request("DELETE", f"/api/v1/media/{media_id}/permanent")
        return {"deleted": True, **response}

    async def update_media_keywords(
        self,
        media_id: int,
        request_data: MediaKeywordsUpdateRequest,
    ) -> MediaKeywordsResponse:
        response = await self._request(
            "PATCH",
            f"/api/v1/media/{media_id}/keywords",
            json_data=request_data.model_dump(mode="json"),
        )
        return MediaKeywordsResponse.model_validate(response)

    async def search_media_metadata(
        self,
        *,
        filters: list[dict[str, Any]] | None = None,
        field: str | None = None,
        op: str | None = None,
        value: str | None = None,
        match_mode: str = "all",
        group_by_media: bool = True,
        page: int = 1,
        per_page: int = 20,
        q: str | None = None,
        media_types: list[str] | None = None,
        must_have: list[str] | None = None,
        must_not_have: list[str] | None = None,
        date_start: str | None = None,
        date_end: str | None = None,
        sort_by: str | None = None,
    ) -> MediaMetadataSearchResponse:
        params: Dict[str, Any] = {
            "match_mode": match_mode,
            "group_by_media": str(group_by_media).lower(),
            "page": page,
            "per_page": per_page,
        }
        if filters is not None:
            params["filters"] = json.dumps(filters)
        if field is not None:
            params["field"] = field
            if op is None and filters is None:
                params["op"] = "icontains"
        if op is not None:
            params["op"] = op
        if value is not None:
            params["value"] = value
        if q is not None:
            params["q"] = q
        if media_types is not None:
            params["media_types"] = ",".join(media_types)
        if must_have is not None:
            params["must_have"] = ",".join(must_have)
        if must_not_have is not None:
            params["must_not_have"] = ",".join(must_not_have)
        if date_start is not None:
            params["date_start"] = date_start
        if date_end is not None:
            params["date_end"] = date_end
        if sort_by is not None:
            params["sort_by"] = sort_by
        response = await self._request("GET", "/api/v1/media/metadata-search", params=params)
        return MediaMetadataSearchResponse.model_validate(response)

    async def get_media_by_identifier(
        self,
        *,
        doi: str | None = None,
        pmid: str | None = None,
        pmcid: str | None = None,
        arxiv_id: str | None = None,
        s2_paper_id: str | None = None,
        group_by_media: bool = True,
    ) -> MediaIdentifierLookupResponse:
        params: Dict[str, Any] = {"group_by_media": str(group_by_media).lower()}
        for key, value in {
            "doi": doi,
            "pmid": pmid,
            "pmcid": pmcid,
            "arxiv_id": arxiv_id,
            "s2_paper_id": s2_paper_id,
        }.items():
            if value is not None:
                params[key] = value
        response = await self._request("GET", "/api/v1/media/by-identifier", params=params)
        return MediaIdentifierLookupResponse.model_validate(response)

    async def download_media_file(self, media_id: int, *, file_type: str = "original") -> ReadingExportResponse | bytes:
        request_bytes_override = self.__dict__.get("_request_bytes")
        if request_bytes_override is not None:
            return await request_bytes_override(
                "GET",
                f"/api/v1/media/{media_id}/file",
                params={"file_type": file_type},
            )
        return await self._binary_request(
            "GET",
            f"/api/v1/media/{media_id}/file",
            params={"file_type": file_type},
        )

    async def check_media_file(self, media_id: int, *, file_type: str = "original") -> MediaFileAvailabilityResponse:
        headers = await self._headers_request(
            "HEAD",
            f"/api/v1/media/{media_id}/file",
            params={"file_type": file_type},
        )
        content_length: int | None = None
        raw_length = headers.get("content-length")
        if raw_length is not None:
            try:
                content_length = int(raw_length)
            except (TypeError, ValueError):
                content_length = None
        content_disposition = headers.get("content-disposition")
        return MediaFileAvailabilityResponse(
            available=True,
            content_type=headers.get("content-type"),
            content_length=content_length,
            content_disposition=content_disposition,
            filename=self._filename_from_content_disposition(content_disposition),
            etag=headers.get("etag"),
            accept_ranges=headers.get("accept-ranges"),
            cache_control=headers.get("cache-control"),
            headers=headers,
        )

    async def create_file_artifact(self, request_data: FileCreateRequest) -> Dict[str, Any]:
        return await self._request(
            "POST",
            "/api/v1/files/create",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )

    async def list_reference_images(self) -> Dict[str, Any]:
        return await self._request("GET", "/api/v1/files/reference-images")

    async def get_file_artifact(self, file_id: int) -> Dict[str, Any]:
        return await self._request("GET", f"/api/v1/files/{file_id}")

    async def export_file_artifact(self, file_id: int, *, format: str) -> ReadingExportResponse:
        return await self._binary_request(
            "GET",
            f"/api/v1/files/{file_id}/export",
            params={"format": format},
        )

    async def delete_file_artifact(self, file_id: int, hard: bool = False, delete_file: bool = False) -> Dict[str, Any]:
        return await self._request(
            "DELETE",
            f"/api/v1/files/{file_id}",
            params={"hard": str(hard).lower(), "delete_file": str(delete_file).lower()},
        )

    async def purge_file_artifacts(
        self,
        request_data: FileArtifactsPurgeRequest | None = None,
    ) -> Dict[str, Any]:
        payload = request_data or FileArtifactsPurgeRequest()
        return await self._request(
            "POST",
            "/api/v1/files/purge",
            json_data=payload.model_dump(exclude_none=True, mode="json"),
        )

    async def translate_text(self, request_data: TranslateRequest) -> TranslateResponse:
        response = await self._request(
            "POST",
            "/api/v1/translate",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return TranslateResponse.model_validate(response)

    async def create_companion_activity(self, request_data: CompanionActivityCreate) -> CompanionActivityItem:
        response = await self._request(
            "POST",
            "/api/v1/companion/activity",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return CompanionActivityItem.model_validate(response)

    async def create_companion_check_in(self, request_data: CompanionCheckInCreate) -> CompanionActivityItem:
        response = await self._request(
            "POST",
            "/api/v1/companion/check-ins",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return CompanionActivityItem.model_validate(response)

    async def list_companion_activity(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> CompanionActivityListResponse:
        response = await self._request(
            "GET",
            "/api/v1/companion/activity",
            params={"limit": limit, "offset": offset},
        )
        return CompanionActivityListResponse.model_validate(response)

    async def get_companion_activity(self, event_id: str) -> CompanionActivityItem:
        response = await self._request("GET", f"/api/v1/companion/activity/{quote(event_id, safe='')}")
        return CompanionActivityItem.model_validate(response)

    async def list_companion_knowledge(self, *, status: str | None = "active") -> CompanionKnowledgeListResponse:
        params: Dict[str, Any] = {}
        if status is not None:
            params["status"] = status
        response = await self._request("GET", "/api/v1/companion/knowledge", params=params)
        return CompanionKnowledgeListResponse.model_validate(response)

    async def get_companion_knowledge(self, card_id: str) -> CompanionKnowledgeDetail:
        response = await self._request("GET", f"/api/v1/companion/knowledge/{quote(card_id, safe='')}")
        return CompanionKnowledgeDetail.model_validate(response)

    async def get_companion_reflection(self, reflection_id: str) -> CompanionReflectionDetail:
        response = await self._request("GET", f"/api/v1/companion/reflections/{quote(reflection_id, safe='')}")
        return CompanionReflectionDetail.model_validate(response)

    async def get_companion_conversation_prompts(self, *, query: str) -> CompanionConversationPromptsResponse:
        response = await self._request(
            "GET",
            "/api/v1/companion/conversation-prompts",
            params={"query": query},
        )
        return CompanionConversationPromptsResponse.model_validate(response)

    async def list_companion_goals(self, *, status: str | None = None) -> CompanionGoalListResponse:
        params: Dict[str, Any] = {}
        if status is not None:
            params["status"] = status
        response = await self._request("GET", "/api/v1/companion/goals", params=params)
        return CompanionGoalListResponse.model_validate(response)

    async def create_companion_goal(self, request_data: CompanionGoalCreate) -> CompanionGoal:
        response = await self._request(
            "POST",
            "/api/v1/companion/goals",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return CompanionGoal.model_validate(response)

    async def update_companion_goal(self, goal_id: str, request_data: CompanionGoalUpdate) -> CompanionGoal:
        response = await self._request(
            "PATCH",
            f"/api/v1/companion/goals/{quote(goal_id, safe='')}",
            json_data=request_data.model_dump(exclude_unset=True, exclude_none=True, mode="json"),
        )
        return CompanionGoal.model_validate(response)

    async def purge_companion_data(self, request_data: CompanionPurgeRequest) -> CompanionLifecycleResponse:
        response = await self._request(
            "POST",
            "/api/v1/companion/purge",
            json_data=request_data.model_dump(mode="json"),
        )
        return CompanionLifecycleResponse.model_validate(response)

    async def rebuild_companion_data(self, request_data: CompanionRebuildRequest) -> CompanionLifecycleResponse:
        response = await self._request(
            "POST",
            "/api/v1/companion/rebuild",
            json_data=request_data.model_dump(mode="json"),
        )
        return CompanionLifecycleResponse.model_validate(response)

    async def get_personalization_profile(self) -> PersonalizationProfile:
        response = await self._request("GET", "/api/v1/personalization/profile")
        return PersonalizationProfile.model_validate(response)

    async def set_personalization_opt_in(
        self,
        request_data: PersonalizationOptInRequest,
    ) -> PersonalizationProfile:
        response = await self._request(
            "POST",
            "/api/v1/personalization/opt-in",
            json_data=request_data.model_dump(mode="json"),
        )
        return PersonalizationProfile.model_validate(response)

    async def update_personalization_preferences(
        self,
        request_data: PersonalizationPreferencesUpdate,
    ) -> PersonalizationProfile:
        response = await self._request(
            "POST",
            "/api/v1/personalization/preferences",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return PersonalizationProfile.model_validate(response)

    async def purge_personalization_data(self) -> PersonalizationPurgeResponse:
        response = await self._request("POST", "/api/v1/personalization/purge")
        return PersonalizationPurgeResponse.model_validate(response)

    async def list_personalization_memories(
        self,
        *,
        memory_type: str | None = None,
        q: str | None = None,
        page: int = 1,
        size: int = 50,
        include_hidden: bool = False,
    ) -> PersonalizationMemoryListResponse:
        params: Dict[str, Any] = {"page": page, "size": size, "include_hidden": include_hidden}
        if memory_type is not None:
            params["type"] = memory_type
        if q is not None:
            params["q"] = q
        response = await self._request("GET", "/api/v1/personalization/memories", params=params)
        return PersonalizationMemoryListResponse.model_validate(response)

    async def export_personalization_memories(self) -> PersonalizationMemoryExportResponse:
        response = await self._request("GET", "/api/v1/personalization/memories/export")
        return PersonalizationMemoryExportResponse.model_validate(response)

    async def get_personalization_memory(self, memory_id: str) -> PersonalizationMemoryItem:
        response = await self._request(
            "GET",
            f"/api/v1/personalization/memories/{quote(memory_id, safe='')}",
        )
        return PersonalizationMemoryItem.model_validate(response)

    async def create_personalization_memory(
        self,
        request_data: PersonalizationMemoryCreate,
    ) -> PersonalizationMemoryItem:
        response = await self._request(
            "POST",
            "/api/v1/personalization/memories",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return PersonalizationMemoryItem.model_validate(response)

    async def update_personalization_memory(
        self,
        memory_id: str,
        request_data: PersonalizationMemoryUpdate,
    ) -> PersonalizationMemoryItem:
        response = await self._request(
            "PATCH",
            f"/api/v1/personalization/memories/{quote(memory_id, safe='')}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return PersonalizationMemoryItem.model_validate(response)

    async def delete_personalization_memory(self, memory_id: str) -> PersonalizationDetailResponse:
        response = await self._request(
            "DELETE",
            f"/api/v1/personalization/memories/{quote(memory_id, safe='')}",
        )
        return PersonalizationDetailResponse.model_validate(response)

    async def validate_personalization_memories(
        self,
        request_data: PersonalizationMemoryValidateRequest,
    ) -> PersonalizationDetailResponse:
        response = await self._request(
            "POST",
            "/api/v1/personalization/memories/validate",
            json_data=request_data.model_dump(mode="json"),
        )
        return PersonalizationDetailResponse.model_validate(response)

    async def import_personalization_memories(
        self,
        request_data: PersonalizationMemoryImportRequest,
    ) -> PersonalizationDetailResponse:
        response = await self._request(
            "POST",
            "/api/v1/personalization/memories/import",
            json_data=request_data.model_dump(mode="json"),
        )
        return PersonalizationDetailResponse.model_validate(response)

    async def list_personalization_explanations(
        self,
        *,
        limit: int = 10,
    ) -> PersonalizationExplanationListResponse:
        response = await self._request(
            "GET",
            "/api/v1/personalization/explanations",
            params={"limit": limit},
        )
        return PersonalizationExplanationListResponse.model_validate(response)

    async def process_voice_command(self, request_data: VoiceCommandRequest) -> VoiceCommandResponse:
        response = await self._request(
            "POST",
            "/api/v1/voice/command",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return VoiceCommandResponse.model_validate(response)

    async def list_voice_commands(
        self,
        *,
        include_system: bool = True,
        include_disabled: bool = False,
        persona_id: str | None = None,
    ) -> VoiceCommandListResponse:
        response = await self._request(
            "GET",
            "/api/v1/voice/commands",
            params={
                key: value
                for key, value in {
                    "include_system": str(include_system).lower(),
                    "include_disabled": str(include_disabled).lower(),
                    "persona_id": persona_id,
                }.items()
                if value is not None
            },
        )
        return VoiceCommandListResponse.model_validate(response)

    async def create_voice_command(self, request_data: VoiceCommandDefinition) -> VoiceCommandInfo:
        response = await self._request(
            "POST",
            "/api/v1/voice/commands",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return VoiceCommandInfo.model_validate(response)

    async def get_voice_command(
        self,
        command_id: str,
        *,
        persona_id: str | None = None,
    ) -> VoiceCommandInfo:
        response = await self._request(
            "GET",
            f"/api/v1/voice/commands/{command_id}",
            params={key: value for key, value in {"persona_id": persona_id}.items() if value is not None},
        )
        return VoiceCommandInfo.model_validate(response)

    async def update_voice_command(
        self,
        command_id: str,
        request_data: VoiceCommandDefinition,
        *,
        persona_id: str | None = None,
    ) -> VoiceCommandInfo:
        response = await self._request(
            "PUT",
            f"/api/v1/voice/commands/{command_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
            params={key: value for key, value in {"persona_id": persona_id}.items() if value is not None},
        )
        return VoiceCommandInfo.model_validate(response)

    async def toggle_voice_command(
        self,
        command_id: str,
        request_data: VoiceCommandToggleRequest,
        *,
        persona_id: str | None = None,
    ) -> VoiceCommandInfo:
        response = await self._request(
            "POST",
            f"/api/v1/voice/commands/{command_id}/toggle",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
            params={key: value for key, value in {"persona_id": persona_id}.items() if value is not None},
        )
        return VoiceCommandInfo.model_validate(response)

    async def validate_voice_command(
        self,
        command_id: str,
        *,
        persona_id: str | None = None,
    ) -> VoiceCommandValidationResponse:
        response = await self._request(
            "POST",
            f"/api/v1/voice/commands/{command_id}/validate",
            params={key: value for key, value in {"persona_id": persona_id}.items() if value is not None},
        )
        return VoiceCommandValidationResponse.model_validate(response)

    async def get_voice_command_usage(self, command_id: str, *, days: int = 30) -> VoiceCommandUsage:
        response = await self._request(
            "GET",
            f"/api/v1/voice/commands/{command_id}/usage",
            params={"days": days},
        )
        return VoiceCommandUsage.model_validate(response)

    async def delete_voice_command(self, command_id: str, *, persona_id: str | None = None) -> Dict[str, Any]:
        return await self._request(
            "DELETE",
            f"/api/v1/voice/commands/{command_id}",
            params={key: value for key, value in {"persona_id": persona_id}.items() if value is not None},
        )

    async def list_voice_sessions(
        self,
        *,
        active_only: bool = True,
        limit: int = 100,
    ) -> VoiceSessionListResponse:
        response = await self._request(
            "GET",
            "/api/v1/voice/sessions",
            params={"active_only": str(active_only).lower(), "limit": limit},
        )
        return VoiceSessionListResponse.model_validate(response)

    async def get_voice_session(self, session_id: str) -> VoiceSessionInfo:
        response = await self._request("GET", f"/api/v1/voice/sessions/{session_id}")
        return VoiceSessionInfo.model_validate(response)

    async def delete_voice_session(self, session_id: str) -> Dict[str, Any]:
        return await self._request("DELETE", f"/api/v1/voice/sessions/{session_id}")

    async def get_voice_analytics(self, *, days: int = 7) -> VoiceAnalyticsSummary:
        response = await self._request("GET", "/api/v1/voice/analytics", params={"days": days})
        return VoiceAnalyticsSummary.model_validate(response)

    async def dry_run_voice_command(
        self,
        request_data: VoiceCommandDryRunRequest,
    ) -> VoiceCommandDryRunResponse:
        response = await self._request(
            "POST",
            "/api/v1/voice/voice/commands/dry-run",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return VoiceCommandDryRunResponse.model_validate(response)

    async def list_ocr_backends(self) -> OCRBackendsResponse:
        response = await self._request("GET", "/api/v1/ocr/backends")
        return OCRBackendsResponse.model_validate(response)

    async def preload_ocr_points_transformers(self) -> OCRPointsPreloadResponse:
        response = await self._request("POST", "/api/v1/ocr/points/preload")
        return OCRPointsPreloadResponse.model_validate(response)

    async def list_vlm_backends(self) -> VLMBackendsResponse:
        response = await self._request("GET", "/api/v1/vlm/backends")
        return VLMBackendsResponse.model_validate(response)

    async def login(
        self,
        username: str,
        password: str,
        *,
        set_bearer_token: bool = True,
    ) -> AuthTokenResponse | MFAChallengeResponse:
        response = await self._request(
            "POST",
            "/api/v1/auth/login",
            data={"username": username, "password": password, "grant_type": "password"},
        )
        if isinstance(response, dict) and response.get("mfa_required"):
            return MFAChallengeResponse.model_validate(response)
        token_response = AuthTokenResponse.model_validate(response)
        if set_bearer_token:
            self.set_bearer_token(token_response.access_token)
        return token_response

    async def refresh_auth_token(
        self,
        request_data: RefreshTokenRequest,
        *,
        set_bearer_token: bool = True,
    ) -> AuthTokenResponse:
        response = await self._request(
            "POST",
            "/api/v1/auth/refresh",
            json_data=request_data.model_dump(mode="json"),
        )
        token_response = AuthTokenResponse.model_validate(response)
        if set_bearer_token:
            self.set_bearer_token(token_response.access_token)
        return token_response

    async def logout(self, *, all_devices: bool = False, clear_bearer_token: bool = True) -> MessageResponse:
        response = await self._request(
            "POST",
            "/api/v1/auth/logout",
            json_data=LogoutRequest(all_devices=all_devices).model_dump(mode="json"),
        )
        message = MessageResponse.model_validate(response)
        if clear_bearer_token:
            self.set_bearer_token(None)
        return message

    async def list_auth_sessions(self) -> list[SessionResponse]:
        response = await self._request("GET", "/api/v1/auth/sessions")
        return [SessionResponse.model_validate(item) for item in response]

    async def revoke_auth_session(self, session_id: int) -> MessageResponse:
        response = await self._request("DELETE", f"/api/v1/auth/sessions/{session_id}")
        return MessageResponse.model_validate(response)

    async def revoke_all_auth_sessions(self) -> MessageResponse:
        response = await self._request("POST", "/api/v1/auth/sessions/revoke-all")
        return MessageResponse.model_validate(response)

    async def get_user_profile_catalog(self, *, if_none_match: str | None = None) -> UserProfileCatalogResponse:
        response = await self._request(
            "GET",
            "/api/v1/users/profile/catalog",
            headers={"If-None-Match": if_none_match} if if_none_match is not None else None,
        )
        return UserProfileCatalogResponse.model_validate(response)

    async def get_current_user_profile(
        self,
        *,
        sections: str | list[str] | None = None,
        include_sources: bool = False,
    ) -> UserProfileResponse:
        section_param = ",".join(sections) if isinstance(sections, list) else sections
        params = {
            "sections": section_param,
            "include_sources": str(include_sources).lower() if include_sources else None,
        }
        response = await self._request(
            "GET",
            "/api/v1/users/me/profile",
            params={key: value for key, value in params.items() if value is not None},
        )
        return UserProfileResponse.model_validate(response)

    async def update_current_user_profile(
        self,
        request_data: UserProfileUpdateRequest,
    ) -> UserProfileUpdateResponse:
        payload = request_data.model_dump(mode="json")
        if request_data.profile_version is None:
            payload.pop("profile_version", None)
        response = await self._request(
            "PATCH",
            "/api/v1/users/me/profile",
            json_data=payload,
        )
        return UserProfileUpdateResponse.model_validate(response)

    async def register_user(self, request_data: RegisterRequest) -> RegistrationResponse:
        response = await self._request(
            "POST",
            "/api/v1/auth/register",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return RegistrationResponse.model_validate(response)

    async def change_password(self, request_data: PasswordChangeRequest) -> MessageResponse:
        response = await self._request(
            "POST",
            "/api/v1/users/change-password",
            json_data=request_data.model_dump(mode="json"),
        )
        return MessageResponse.model_validate(response)

    async def request_password_reset(self, request_data: PasswordResetRequest) -> MessageResponse:
        response = await self._request(
            "POST",
            "/api/v1/auth/forgot-password",
            json_data=request_data.model_dump(mode="json"),
        )
        return MessageResponse.model_validate(response)

    async def reset_password(self, request_data: PasswordResetConfirm) -> MessageResponse:
        response = await self._request(
            "POST",
            "/api/v1/auth/reset-password",
            json_data=request_data.model_dump(mode="json"),
        )
        return MessageResponse.model_validate(response)

    async def verify_email(self, token: str) -> MessageResponse:
        response = await self._request(
            "GET",
            "/api/v1/auth/verify-email",
            params={"token": token},
        )
        return MessageResponse.model_validate(response)

    async def resend_verification(self, email: str) -> MessageResponse:
        response = await self._request(
            "POST",
            "/api/v1/auth/resend-verification",
            json_data={"email": email},
        )
        return MessageResponse.model_validate(response)

    async def request_magic_link(self, email: str) -> MessageResponse:
        response = await self._request(
            "POST",
            "/api/v1/auth/magic-link/request",
            json_data={"email": email},
        )
        return MessageResponse.model_validate(response)

    async def verify_magic_link(self, token: str, *, set_bearer_token: bool = True) -> AuthTokenResponse:
        response = await self._request(
            "POST",
            "/api/v1/auth/magic-link/verify",
            json_data={"token": token},
        )
        token_response = AuthTokenResponse.model_validate(response)
        if set_bearer_token:
            self.set_bearer_token(token_response.access_token)
        return token_response

    async def setup_mfa(self) -> MFASetupResponse:
        response = await self._request("POST", "/api/v1/auth/mfa/setup")
        return MFASetupResponse.model_validate(response)

    async def verify_mfa_setup(self, token: str) -> MessageResponse:
        response = await self._request(
            "POST",
            "/api/v1/auth/mfa/verify",
            json_data={"token": token},
        )
        if isinstance(response, dict) and "backup_codes" in response and "details" not in response:
            response = dict(response)
            response["details"] = {"backup_codes": response.pop("backup_codes")}
        return MessageResponse.model_validate(response)

    async def disable_mfa(self, password: str) -> MessageResponse:
        response = await self._request(
            "POST",
            "/api/v1/auth/mfa/disable",
            data={"password": password},
        )
        return MessageResponse.model_validate(response)

    async def complete_mfa_login(
        self,
        *,
        session_token: str,
        mfa_token: str,
        set_bearer_token: bool = True,
    ) -> AuthTokenResponse:
        response = await self._request(
            "POST",
            "/api/v1/auth/mfa/login",
            json_data={"session_token": session_token, "mfa_token": mfa_token},
        )
        token_response = AuthTokenResponse.model_validate(response)
        if set_bearer_token:
            self.set_bearer_token(token_response.access_token)
        return token_response

    async def list_user_api_keys(self) -> list[APIKeyMetadata]:
        response = await self._request("GET", "/api/v1/users/api-keys")
        return [APIKeyMetadata.model_validate(item) for item in response]

    async def create_user_api_key(self, request_data: APIKeyCreateRequest) -> APIKeyCreateResponse:
        response = await self._request(
            "POST",
            "/api/v1/users/api-keys",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return APIKeyCreateResponse.model_validate(response)

    async def create_virtual_api_key(
        self,
        request_data: VirtualAPIKeyCreateRequest | None = None,
        **kwargs: Any,
    ) -> APIKeyCreateResponse:
        payload = request_data or VirtualAPIKeyCreateRequest(**kwargs)
        response = await self._request(
            "POST",
            "/api/v1/users/api-keys/virtual",
            json_data=payload.model_dump(exclude_none=True, mode="json"),
        )
        return APIKeyCreateResponse.model_validate(response)

    async def rotate_user_api_key(
        self,
        key_id: int,
        request_data: APIKeyRotateRequest | None = None,
    ) -> APIKeyCreateResponse:
        payload = request_data or APIKeyRotateRequest()
        response = await self._request(
            "POST",
            f"/api/v1/users/api-keys/{key_id}/rotate",
            json_data=payload.model_dump(exclude_none=True, mode="json"),
        )
        return APIKeyCreateResponse.model_validate(response)

    async def revoke_user_api_key(self, key_id: int) -> MessageResponse:
        response = await self._request("DELETE", f"/api/v1/users/api-keys/{key_id}")
        return MessageResponse.model_validate(response)

    async def get_user_storage_quota(self) -> StorageQuotaResponse:
        response = await self._request("GET", "/api/v1/users/storage")
        return StorageQuotaResponse.model_validate(response)

    async def recalculate_user_storage_quota(self) -> StorageQuotaResponse:
        response = await self._request("POST", "/api/v1/users/storage/recalculate")
        return StorageQuotaResponse.model_validate(response)

    async def get_consent_preferences(self) -> ConsentPreferencesResponse:
        response = await self._request("GET", "/api/v1/consent/preferences")
        return ConsentPreferencesResponse.model_validate(response)

    async def grant_consent(self, purpose: str) -> ConsentRecordResponse:
        response = await self._request("POST", f"/api/v1/consent/preferences/{purpose}")
        return ConsentRecordResponse.model_validate(response)

    async def withdraw_consent(self, purpose: str) -> ConsentRecordResponse:
        response = await self._request("DELETE", f"/api/v1/consent/preferences/{purpose}")
        return ConsentRecordResponse.model_validate(response)

    async def get_self_privilege_map(self, *, resource: str | None = None) -> PrivilegeSelfResponse:
        response = await self._request(
            "GET",
            "/api/v1/privileges/self",
            params={key: value for key, value in {"resource": resource}.items() if value is not None},
        )
        return PrivilegeSelfResponse.model_validate(response)

    async def get_user_privilege_map(
        self,
        user_id: str,
        *,
        page: int = 1,
        page_size: int = 100,
        resource: str | None = None,
    ) -> PrivilegeDetailResponse:
        response = await self._request(
            "GET",
            f"/api/v1/privileges/users/{user_id}",
            params={
                key: value
                for key, value in {
                    "page": page,
                    "page_size": page_size,
                    "resource": resource,
                }.items()
                if value is not None
            },
        )
        return PrivilegeDetailResponse.model_validate(response)

    async def upsert_user_provider_key(
        self,
        request_data: UserProviderKeyUpsertRequest,
    ) -> UserProviderKeyResponse:
        response = await self._request(
            "POST",
            "/api/v1/users/keys",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return UserProviderKeyResponse.model_validate(response)

    async def list_user_provider_keys(self) -> UserProviderKeysResponse:
        response = await self._request("GET", "/api/v1/users/keys")
        return UserProviderKeysResponse.model_validate(response)

    async def test_user_provider_key(self, request_data: ProviderKeyTestRequest) -> ProviderKeyTestResponse:
        response = await self._request(
            "POST",
            "/api/v1/users/keys/test",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return ProviderKeyTestResponse.model_validate(response)

    async def authorize_openai_oauth(
        self,
        request_data: OpenAIOAuthAuthorizeRequest | None = None,
    ) -> OpenAIOAuthAuthorizeResponse:
        payload = request_data or OpenAIOAuthAuthorizeRequest()
        response = await self._request(
            "POST",
            "/api/v1/users/keys/openai/oauth/authorize",
            json_data=payload.model_dump(exclude_none=True, mode="json"),
        )
        return OpenAIOAuthAuthorizeResponse.model_validate(response)

    async def complete_openai_oauth_callback(
        self,
        *,
        code: str,
        state: str,
        redirect: bool = False,
    ) -> OpenAIOAuthCallbackResponse:
        response = await self._request(
            "GET",
            "/api/v1/users/keys/openai/oauth/callback",
            params={"code": code, "state": state, "redirect": str(redirect).lower()},
        )
        return OpenAIOAuthCallbackResponse.model_validate(response)

    async def get_openai_oauth_status(self) -> OpenAIOAuthStatusResponse:
        response = await self._request("GET", "/api/v1/users/keys/openai/oauth/status")
        return OpenAIOAuthStatusResponse.model_validate(response)

    async def refresh_openai_oauth(self) -> OpenAIOAuthRefreshResponse:
        response = await self._request("POST", "/api/v1/users/keys/openai/oauth/refresh")
        return OpenAIOAuthRefreshResponse.model_validate(response)

    async def disconnect_openai_oauth(self) -> bool:
        await self._request("DELETE", "/api/v1/users/keys/openai/oauth")
        return True

    async def switch_openai_credential_source(
        self,
        request_data: OpenAICredentialSourceSwitchRequest,
    ) -> OpenAICredentialSourceSwitchResponse:
        response = await self._request(
            "POST",
            "/api/v1/users/keys/openai/source",
            json_data=request_data.model_dump(mode="json"),
        )
        return OpenAICredentialSourceSwitchResponse.model_validate(response)

    async def delete_user_provider_key(self, provider: str) -> bool:
        await self._request("DELETE", f"/api/v1/users/keys/{provider}")
        return True

    async def list_storage_files(
        self,
        *,
        offset: int = 0,
        limit: int = 50,
        file_category: FileCategory | None = None,
        source_feature: SourceFeature | None = None,
        folder_tag: str | None = None,
        search: str | None = None,
        include_deleted: bool = False,
    ) -> GeneratedFilesListResponse:
        params: Dict[str, Any] = {
            "offset": offset,
            "limit": limit,
            "file_category": file_category,
            "source_feature": source_feature,
            "folder_tag": folder_tag,
            "search": search,
            "include_deleted": str(include_deleted).lower(),
        }
        response = await self._request(
            "GET",
            "/api/v1/storage/files",
            params={key: value for key, value in params.items() if value is not None},
        )
        return GeneratedFilesListResponse.model_validate(response)

    async def get_storage_file(self, file_id: int) -> GeneratedFileResponse:
        response = await self._request("GET", f"/api/v1/storage/files/{file_id}")
        return GeneratedFileResponse.model_validate(response)

    async def download_storage_file(self, file_id: int) -> ReadingExportResponse:
        return await self._binary_request("GET", f"/api/v1/storage/files/{file_id}/download")

    async def update_storage_file(
        self,
        file_id: int,
        request_data: GeneratedFileUpdate,
    ) -> GeneratedFileResponse:
        response = await self._request(
            "PATCH",
            f"/api/v1/storage/files/{file_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return GeneratedFileResponse.model_validate(response)

    async def delete_storage_file(self, file_id: int, *, hard_delete: bool = False) -> Dict[str, Any]:
        return await self._request(
            "DELETE",
            f"/api/v1/storage/files/{file_id}",
            params={"hard_delete": str(hard_delete).lower()},
        )

    async def bulk_delete_storage_files(self, request_data: BulkDeleteRequest) -> BulkDeleteResponse:
        response = await self._request(
            "POST",
            "/api/v1/storage/files/bulk-delete",
            json_data=request_data.model_dump(mode="json"),
        )
        return BulkDeleteResponse.model_validate(response)

    async def bulk_move_storage_files(self, request_data: BulkMoveRequest) -> BulkMoveResponse:
        response = await self._request(
            "POST",
            "/api/v1/storage/files/bulk-move",
            json_data=request_data.model_dump(mode="json"),
        )
        return BulkMoveResponse.model_validate(response)

    async def list_storage_folders(self) -> FolderListResponse:
        response = await self._request("GET", "/api/v1/storage/folders")
        return FolderListResponse.model_validate(response)

    async def create_storage_folder(self, name: str) -> Dict[str, Any]:
        return await self._request(
            "POST",
            "/api/v1/storage/folders",
            json_data={"name": name},
        )

    async def list_least_accessed_storage_files(self, *, limit: int = 20) -> GeneratedFilesListResponse:
        response = await self._request(
            "GET",
            "/api/v1/storage/files/least-accessed",
            params={"limit": limit},
        )
        return GeneratedFilesListResponse.model_validate(response)

    async def get_storage_usage(self) -> StorageUsageResponse:
        response = await self._request("GET", "/api/v1/storage/usage")
        return StorageUsageResponse.model_validate(response)

    async def get_storage_usage_breakdown(self) -> UsageBreakdownResponse:
        response = await self._request("GET", "/api/v1/storage/usage/breakdown")
        return UsageBreakdownResponse.model_validate(response)

    async def list_storage_trash(self, *, offset: int = 0, limit: int = 50) -> TrashListResponse:
        response = await self._request(
            "GET",
            "/api/v1/storage/trash",
            params={"offset": offset, "limit": limit},
        )
        return TrashListResponse.model_validate(response)

    async def restore_storage_file(self, file_id: int) -> RestoreResponse:
        response = await self._request("POST", f"/api/v1/storage/trash/restore/{file_id}")
        return RestoreResponse.model_validate(response)

    async def permanently_delete_storage_file(self, file_id: int) -> Dict[str, Any]:
        return await self._request("DELETE", f"/api/v1/storage/trash/{file_id}")

    async def get_server_health(self) -> ServerHealthResponse:
        response = await self._request("GET", "/api/v1/health")
        return ServerHealthResponse.model_validate(response)

    async def get_server_liveness(self) -> ServerLivenessResponse:
        response = await self._request("GET", "/api/v1/health/live")
        return ServerLivenessResponse.model_validate(response)

    async def get_server_readiness(self) -> ServerReadinessResponse:
        response = await self._request("GET", "/api/v1/health/ready")
        return ServerReadinessResponse.model_validate(response)

    async def get_server_metrics(self) -> ServerMetricsResponse:
        response = await self._request("GET", "/api/v1/health/metrics")
        return ServerMetricsResponse.model_validate(response)

    async def get_server_security_health(self) -> ServerSecurityHealthResponse:
        response = await self._request("GET", "/api/v1/health/security")
        return ServerSecurityHealthResponse.model_validate(response)

    async def get_server_docs_info(self) -> ServerDocsInfoResponse:
        response = await self._request("GET", "/api/v1/config/docs-info")
        return ServerDocsInfoResponse.model_validate(response)

    async def get_flashcards_import_limits(self) -> FlashcardsImportLimitsResponse:
        response = await self._request("GET", "/api/v1/config/flashcards-import-limits")
        return FlashcardsImportLimitsResponse.model_validate(response)

    async def get_tokenizer_config(self) -> TokenizerConfigResponse:
        response = await self._request("GET", "/api/v1/config/tokenizer")
        return TokenizerConfigResponse.model_validate(response)

    async def update_tokenizer_config(self, request_data: TokenizerUpdateRequest) -> TokenizerConfigResponse:
        response = await self._request(
            "PUT",
            "/api/v1/config/tokenizer",
            json_data=request_data.model_dump(mode="json"),
        )
        return TokenizerConfigResponse.model_validate(response)

    async def get_jobs_config(self) -> JobsConfigResponse:
        response = await self._request("GET", "/api/v1/config/jobs")
        return JobsConfigResponse.model_validate(response)

    async def list_config_providers(self) -> ProvidersStatusResponse:
        response = await self._request("GET", "/api/v1/config/providers")
        return ProvidersStatusResponse.model_validate(response)

    async def validate_provider_key(self, request_data: ProviderValidateRequest) -> ProviderValidateResponse:
        response = await self._request(
            "POST",
            "/api/v1/config/validate-provider",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return ProviderValidateResponse.model_validate(response)

    async def get_llm_health(self) -> LLMHealthResponse:
        response = await self._request("GET", "/api/v1/llm/health")
        return LLMHealthResponse.model_validate(response)

    async def list_llm_providers(
        self,
        *,
        include_deprecated: bool = False,
    ) -> LLMProviderListResponse:
        response = await self._request(
            "GET",
            "/api/v1/llm/providers",
            params={"include_deprecated": str(include_deprecated).lower()},
        )
        return LLMProviderListResponse.model_validate(response)

    async def get_llm_provider(
        self,
        provider_name: str,
        *,
        include_deprecated: bool = False,
    ) -> LLMProviderDetail:
        response = await self._request(
            "GET",
            f"/api/v1/llm/providers/{provider_name}",
            params={"include_deprecated": str(include_deprecated).lower()},
        )
        return LLMProviderDetail.model_validate(response)

    async def get_llm_models_metadata(
        self,
        *,
        include_deprecated: bool = False,
        refresh_openrouter: bool = False,
        model_type: str | list[str] | None = None,
        input_modality: str | list[str] | None = None,
        output_modality: str | list[str] | None = None,
    ) -> LLMModelMetadataResponse:
        params = {
            "include_deprecated": str(include_deprecated).lower(),
            "refresh_openrouter": str(refresh_openrouter).lower(),
            "type": model_type,
            "input_modality": input_modality,
            "output_modality": output_modality,
        }
        response = await self._request(
            "GET",
            "/api/v1/llm/models/metadata",
            params={key: value for key, value in params.items() if value is not None},
        )
        return LLMModelMetadataResponse.model_validate(response)

    async def list_llm_models(
        self,
        *,
        include_deprecated: bool = False,
        model_type: str | list[str] | None = None,
        input_modality: str | list[str] | None = None,
        output_modality: str | list[str] | None = None,
    ) -> list[str]:
        params = {
            "include_deprecated": str(include_deprecated).lower(),
            "type": model_type,
            "input_modality": input_modality,
            "output_modality": output_modality,
        }
        response = await self._request(
            "GET",
            "/api/v1/llm/models",
            params={key: value for key, value in params.items() if value is not None},
        )
        return [str(model) for model in response]

    async def create_prompt_studio_project(
        self,
        request_data: PromptStudioProjectCreate,
        *,
        idempotency_key: str | None = None,
    ) -> PromptStudioStandardResponse:
        response = await self._request(
            "POST",
            "/api/v1/prompt-studio/projects/",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
            headers={"Idempotency-Key": idempotency_key} if idempotency_key else None,
        )
        return PromptStudioStandardResponse.model_validate(response)

    async def list_prompt_studio_projects(
        self,
        *,
        page: int = 1,
        per_page: int = 20,
        status: str | None = None,
        include_deleted: bool = False,
        search: str | None = None,
    ) -> PromptStudioListResponse:
        params = {
            "page": page,
            "per_page": per_page,
            "status": status,
            "include_deleted": str(include_deleted).lower(),
            "search": search,
        }
        response = await self._request(
            "GET",
            "/api/v1/prompt-studio/projects/",
            params={key: value for key, value in params.items() if value is not None},
        )
        return PromptStudioListResponse.model_validate(response)

    async def get_prompt_studio_project(self, project_id: int) -> PromptStudioStandardResponse:
        response = await self._request("GET", f"/api/v1/prompt-studio/projects/get/{project_id}")
        return PromptStudioStandardResponse.model_validate(response)

    async def update_prompt_studio_project(
        self,
        project_id: int,
        request_data: PromptStudioProjectUpdate,
    ) -> PromptStudioStandardResponse:
        response = await self._request(
            "PUT",
            f"/api/v1/prompt-studio/projects/update/{project_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return PromptStudioStandardResponse.model_validate(response)

    async def delete_prompt_studio_project(
        self,
        project_id: int,
        *,
        permanent: bool = False,
    ) -> PromptStudioStandardResponse:
        response = await self._request(
            "DELETE",
            f"/api/v1/prompt-studio/projects/delete/{project_id}",
            params={"permanent": str(permanent).lower()},
        )
        return PromptStudioStandardResponse.model_validate(response)

    async def archive_prompt_studio_project(self, project_id: int) -> PromptStudioStandardResponse:
        response = await self._request("POST", f"/api/v1/prompt-studio/projects/archive/{project_id}")
        return PromptStudioStandardResponse.model_validate(response)

    async def unarchive_prompt_studio_project(self, project_id: int) -> PromptStudioStandardResponse:
        response = await self._request("POST", f"/api/v1/prompt-studio/projects/unarchive/{project_id}")
        return PromptStudioStandardResponse.model_validate(response)

    async def get_prompt_studio_project_stats(self, project_id: int) -> PromptStudioStandardResponse:
        response = await self._request("GET", f"/api/v1/prompt-studio/projects/stats/{project_id}")
        return PromptStudioStandardResponse.model_validate(response)

    async def create_prompt_studio_prompt(
        self,
        request_data: PromptStudioPromptCreate,
        *,
        idempotency_key: str | None = None,
    ) -> PromptStudioStandardResponse:
        response = await self._request(
            "POST",
            "/api/v1/prompt-studio/prompts/create",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
            headers={"Idempotency-Key": idempotency_key} if idempotency_key else None,
        )
        return PromptStudioStandardResponse.model_validate(response)

    async def list_prompt_studio_prompts(
        self,
        project_id: int,
        *,
        page: int = 1,
        per_page: int = 20,
        include_deleted: bool = False,
    ) -> PromptStudioListResponse:
        response = await self._request(
            "GET",
            f"/api/v1/prompt-studio/prompts/list/{project_id}",
            params={
                "page": page,
                "per_page": per_page,
                "include_deleted": str(include_deleted).lower(),
            },
        )
        return PromptStudioListResponse.model_validate(response)

    async def get_prompt_studio_prompt(self, prompt_id: int) -> PromptStudioStandardResponse:
        response = await self._request("GET", f"/api/v1/prompt-studio/prompts/get/{prompt_id}")
        return PromptStudioStandardResponse.model_validate(response)

    async def update_prompt_studio_prompt(
        self,
        prompt_id: int,
        request_data: PromptStudioPromptUpdate,
    ) -> PromptStudioStandardResponse:
        response = await self._request(
            "PUT",
            f"/api/v1/prompt-studio/prompts/update/{prompt_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return PromptStudioStandardResponse.model_validate(response)

    async def get_prompt_studio_prompt_history(self, prompt_id: int) -> PromptStudioStandardResponse:
        response = await self._request("GET", f"/api/v1/prompt-studio/prompts/history/{prompt_id}")
        return PromptStudioStandardResponse.model_validate(response)

    async def revert_prompt_studio_prompt(self, prompt_id: int, version: int) -> PromptStudioStandardResponse:
        response = await self._request("POST", f"/api/v1/prompt-studio/prompts/revert/{prompt_id}/{version}")
        return PromptStudioStandardResponse.model_validate(response)

    async def preview_prompt_studio_prompt(
        self,
        request_data: PromptStudioPromptPreviewRequest,
    ) -> PromptStudioStandardResponse:
        response = await self._request(
            "POST",
            "/api/v1/prompt-studio/prompts/preview",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return PromptStudioStandardResponse.model_validate(response)

    async def convert_prompt_studio_prompt(
        self,
        request_data: PromptStudioPromptConvertRequest,
    ) -> PromptStudioStandardResponse:
        response = await self._request(
            "POST",
            "/api/v1/prompt-studio/prompts/convert",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return PromptStudioStandardResponse.model_validate(response)

    async def execute_prompt_studio_prompt(
        self,
        request_data: PromptStudioPromptExecuteRequest,
    ) -> PromptStudioPromptExecutionResponse:
        response = await self._request(
            "POST",
            "/api/v1/prompt-studio/prompts/execute",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return PromptStudioPromptExecutionResponse.model_validate(response)

    async def create_prompt_studio_test_case(
        self,
        request_data: PromptStudioTestCaseCreate,
    ) -> PromptStudioStandardResponse:
        response = await self._request(
            "POST",
            "/api/v1/prompt-studio/test-cases/create",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return PromptStudioStandardResponse.model_validate(response)

    async def create_prompt_studio_test_cases_bulk(
        self,
        request_data: PromptStudioTestCaseBulkCreate,
    ) -> PromptStudioStandardResponse:
        response = await self._request(
            "POST",
            "/api/v1/prompt-studio/test-cases/bulk",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return PromptStudioStandardResponse.model_validate(response)

    async def list_prompt_studio_test_cases(
        self,
        project_id: int,
        *,
        page: int = 1,
        per_page: int = 20,
        is_golden: bool | None = None,
        tags: list[str] | str | None = None,
        search: str | None = None,
        signature_id: int | None = None,
    ) -> PromptStudioListResponse:
        tags_param = ",".join(tags) if isinstance(tags, list) else tags
        params = {
            "page": page,
            "per_page": per_page,
            "is_golden": str(is_golden).lower() if is_golden is not None else None,
            "tags": tags_param,
            "search": search,
            "signature_id": signature_id,
        }
        response = await self._request(
            "GET",
            f"/api/v1/prompt-studio/test-cases/list/{project_id}",
            params={key: value for key, value in params.items() if value is not None},
        )
        return PromptStudioListResponse.model_validate(response)

    async def get_prompt_studio_test_case(self, test_case_id: int) -> PromptStudioStandardResponse:
        response = await self._request("GET", f"/api/v1/prompt-studio/test-cases/get/{test_case_id}")
        return PromptStudioStandardResponse.model_validate(response)

    async def update_prompt_studio_test_case(
        self,
        test_case_id: int,
        request_data: PromptStudioTestCaseUpdate,
    ) -> PromptStudioStandardResponse:
        response = await self._request(
            "PUT",
            f"/api/v1/prompt-studio/test-cases/update/{test_case_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return PromptStudioStandardResponse.model_validate(response)

    async def delete_prompt_studio_test_case(
        self,
        test_case_id: int,
        *,
        permanent: bool = False,
    ) -> PromptStudioStandardResponse:
        response = await self._request(
            "DELETE",
            f"/api/v1/prompt-studio/test-cases/delete/{test_case_id}",
            params={"permanent": str(permanent).lower()},
        )
        return PromptStudioStandardResponse.model_validate(response)

    async def import_prompt_studio_test_cases(
        self,
        request_data: PromptStudioTestCaseImportRequest,
    ) -> PromptStudioStandardResponse:
        response = await self._request(
            "POST",
            "/api/v1/prompt-studio/test-cases/import",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return PromptStudioStandardResponse.model_validate(response)

    async def import_prompt_studio_test_cases_csv_upload(
        self,
        *,
        project_id: int,
        csv_content: str | bytes,
        filename: str = "prompt_studio_test_cases.csv",
        signature_id: int | None = None,
        auto_generate_names: bool = True,
    ) -> PromptStudioStandardResponse:
        content = csv_content.encode("utf-8") if isinstance(csv_content, str) else csv_content
        form_data = {
            "project_id": str(project_id),
            "auto_generate_names": str(auto_generate_names).lower(),
        }
        if signature_id is not None:
            form_data["signature_id"] = str(signature_id)
        response = await self._request(
            "POST",
            "/api/v1/prompt-studio/test-cases/import/csv-upload",
            data=form_data,
            files=[("file", (filename, content, "text/csv"))],
        )
        return PromptStudioStandardResponse.model_validate(response)

    async def get_prompt_studio_test_cases_csv_template(
        self,
        *,
        signature_id: int | None = None,
    ) -> ReadingExportResponse:
        params = {"signature_id": signature_id} if signature_id is not None else None
        return await self._binary_request(
            "GET",
            "/api/v1/prompt-studio/test-cases/import/template",
            params=params,
        )

    async def export_prompt_studio_test_cases(
        self,
        project_id: int,
        request_data: PromptStudioTestCaseExportRequest,
    ) -> PromptStudioStandardResponse:
        response = await self._request(
            "POST",
            f"/api/v1/prompt-studio/test-cases/export/{project_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return PromptStudioStandardResponse.model_validate(response)

    async def generate_prompt_studio_test_cases(
        self,
        request_data: PromptStudioTestCaseGenerateRequest | None = None,
        *,
        project_id: int | None = None,
        prompt_id: int | None = None,
        signature_id: int | None = None,
        num_cases: int = 5,
        generation_strategy: str = "diverse",
        base_on_description: str | None = None,
    ) -> PromptStudioStandardResponse:
        if request_data is None:
            if project_id is None:
                raise ValueError("project_id is required when request_data is not provided")
            request_data = PromptStudioTestCaseGenerateRequest(
                project_id=project_id,
                prompt_id=prompt_id,
                signature_id=signature_id,
                num_cases=num_cases,
                generation_strategy=generation_strategy,
                base_on_description=base_on_description,
            )
        response = await self._request(
            "POST",
            "/api/v1/prompt-studio/test-cases/generate",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return PromptStudioStandardResponse.model_validate(response)

    async def run_prompt_studio_test_cases(
        self,
        request_data: PromptStudioRunTestCasesRequest,
    ) -> PromptStudioRunTestCasesResponse:
        response = await self._request(
            "POST",
            "/api/v1/prompt-studio/test-cases/run",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return PromptStudioRunTestCasesResponse.model_validate(response)

    async def create_prompt_studio_evaluation(
        self,
        request_data: PromptStudioEvaluationCreate,
    ) -> PromptStudioEvaluationResponse:
        response = await self._request(
            "POST",
            "/api/v1/prompt-studio/evaluations",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return PromptStudioEvaluationResponse.model_validate(response)

    async def list_prompt_studio_evaluations(
        self,
        *,
        project_id: int,
        prompt_id: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> PromptStudioEvaluationListResponse:
        params = {
            "project_id": project_id,
            "prompt_id": prompt_id,
            "limit": limit,
            "offset": offset,
        }
        response = await self._request(
            "GET",
            "/api/v1/prompt-studio/evaluations",
            params={key: value for key, value in params.items() if value is not None},
        )
        return PromptStudioEvaluationListResponse.model_validate(response)

    async def get_prompt_studio_evaluation(self, evaluation_id: int) -> PromptStudioEvaluationResponse:
        response = await self._request("GET", f"/api/v1/prompt-studio/evaluations/{evaluation_id}")
        return PromptStudioEvaluationResponse.model_validate(response)

    async def delete_prompt_studio_evaluation(self, evaluation_id: int) -> PromptStudioDeleteMessage:
        response = await self._request("DELETE", f"/api/v1/prompt-studio/evaluations/{evaluation_id}")
        return PromptStudioDeleteMessage.model_validate(response)

    async def create_prompt_studio_optimization(
        self,
        request_data: PromptStudioOptimizationCreate,
        *,
        idempotency_key: str | None = None,
    ) -> PromptStudioStandardResponse:
        response = await self._request(
            "POST",
            "/api/v1/prompt-studio/optimizations/create",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
            headers={"Idempotency-Key": idempotency_key} if idempotency_key else None,
        )
        return PromptStudioStandardResponse.model_validate(response)

    async def create_prompt_studio_optimization_simple(
        self,
        request_data: PromptStudioOptimizationSimpleCreateRequest,
    ) -> PromptStudioSimpleJobResponse:
        response = await self._request(
            "POST",
            "/api/v1/prompt-studio/optimizations",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return PromptStudioSimpleJobResponse.model_validate(response)

    async def list_prompt_studio_optimizations(
        self,
        project_id: int,
        *,
        page: int = 1,
        per_page: int = 20,
        status: str | None = None,
    ) -> PromptStudioListResponse:
        params = {"page": page, "per_page": per_page, "status": status}
        response = await self._request(
            "GET",
            f"/api/v1/prompt-studio/optimizations/list/{project_id}",
            params={key: value for key, value in params.items() if value is not None},
        )
        return PromptStudioListResponse.model_validate(response)

    async def get_prompt_studio_optimization(self, optimization_id: int) -> PromptStudioStandardResponse:
        response = await self._request("GET", f"/api/v1/prompt-studio/optimizations/get/{optimization_id}")
        return PromptStudioStandardResponse.model_validate(response)

    async def get_prompt_studio_optimization_job_status(self, job_id: str) -> dict[str, Any]:
        response = await self._request("GET", f"/api/v1/prompt-studio/optimizations/{job_id}")
        return dict(response)

    async def cancel_prompt_studio_optimization(
        self,
        optimization_id: int,
        *,
        reason: str | None = None,
    ) -> PromptStudioStandardResponse:
        response = await self._request(
            "POST",
            f"/api/v1/prompt-studio/optimizations/cancel/{optimization_id}",
            json_data=reason,
        )
        return PromptStudioStandardResponse.model_validate(response)

    async def get_prompt_studio_optimization_strategies(self) -> PromptStudioStandardResponse:
        response = await self._request("GET", "/api/v1/prompt-studio/optimizations/strategies")
        return PromptStudioStandardResponse.model_validate(response)

    async def get_prompt_studio_optimization_history(self, optimization_id: int) -> PromptStudioStandardResponse:
        response = await self._request("GET", f"/api/v1/prompt-studio/optimizations/history/{optimization_id}")
        return PromptStudioStandardResponse.model_validate(response)

    async def add_prompt_studio_optimization_iteration(
        self,
        optimization_id: int,
        request_data: PromptStudioOptimizationIterationCreate,
    ) -> PromptStudioStandardResponse:
        response = await self._request(
            "POST",
            f"/api/v1/prompt-studio/optimizations/iterations/{optimization_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return PromptStudioStandardResponse.model_validate(response)

    async def list_prompt_studio_optimization_iterations(
        self,
        optimization_id: int,
        *,
        page: int = 1,
        per_page: int = 50,
    ) -> PromptStudioStandardResponse:
        response = await self._request(
            "GET",
            f"/api/v1/prompt-studio/optimizations/iterations/{optimization_id}",
            params={"page": page, "per_page": per_page},
        )
        return PromptStudioStandardResponse.model_validate(response)

    async def compare_prompt_studio_optimization_strategies(
        self,
        request_data: PromptStudioCompareStrategiesRequest,
    ) -> PromptStudioStandardResponse:
        response = await self._request(
            "POST",
            "/api/v1/prompt-studio/optimizations/compare",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return PromptStudioStandardResponse.model_validate(response)

    async def get_prompt_studio_status(self, *, warn_seconds: int = 30) -> PromptStudioStatusResponse:
        response = await self._request(
            "GET",
            "/api/v1/prompt-studio/status",
            params={"warn_seconds": warn_seconds},
        )
        return PromptStudioStatusResponse.model_validate(response)

    async def stream_prompt_studio_events(
        self,
        *,
        client_id: str,
        project_id: int | None = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        params: Dict[str, Any] = {"client_id": client_id}
        if project_id is not None:
            params["project_id"] = project_id
        async for event in self._sse_request(
            "GET",
            "/api/v1/prompt-studio/ws",
            params=params,
            headers={"Accept": "text/event-stream"},
        ):
            yield event

    async def generate_data_table(
        self,
        request_data: DataTableGenerateRequest,
        *,
        wait_for_completion: bool = False,
        wait_timeout_seconds: int = 300,
    ) -> DataTableGenerateResponse | DataTableDetailResponse:
        response = await self._request(
            "POST",
            "/api/v1/data-tables/generate",
            params={
                "wait_for_completion": str(wait_for_completion).lower(),
                "wait_timeout_seconds": wait_timeout_seconds,
            },
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        if isinstance(response, dict) and "job_id" in response:
            return DataTableGenerateResponse.model_validate(response)
        return DataTableDetailResponse.model_validate(response)

    async def list_data_tables(
        self,
        *,
        status_filter: str | None = None,
        search: str | None = None,
        workspace_tag: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> DataTablesListResponse:
        params = {
            "status": status_filter,
            "search": search,
            "workspace_tag": workspace_tag,
            "limit": limit,
            "offset": offset,
        }
        response = await self._request(
            "GET",
            "/api/v1/data-tables",
            params={key: value for key, value in params.items() if value is not None},
        )
        return DataTablesListResponse.model_validate(response)

    async def get_data_table(
        self,
        table_uuid: str,
        *,
        rows_limit: int = 200,
        rows_offset: int = 0,
        include_rows: bool = True,
        include_sources: bool = True,
    ) -> DataTableDetailResponse:
        response = await self._request(
            "GET",
            f"/api/v1/data-tables/{table_uuid}",
            params={
                "rows_limit": rows_limit,
                "rows_offset": rows_offset,
                "include_rows": str(include_rows).lower(),
                "include_sources": str(include_sources).lower(),
            },
        )
        return DataTableDetailResponse.model_validate(response)

    async def export_data_table(
        self,
        table_uuid: str,
        *,
        format: DataTableExportFormat,
        async_mode: AsyncMode = "auto",
        mode: ExportMode = "url",
        download: bool = False,
    ) -> DataTableExportResponse:
        response = await self._request(
            "GET",
            f"/api/v1/data-tables/{table_uuid}/export",
            params={
                "format": format,
                "async_mode": async_mode,
                "mode": mode,
                "download": str(download).lower(),
            },
        )
        return DataTableExportResponse.model_validate(response)

    async def update_data_table_content(
        self,
        table_uuid: str,
        request_data: DataTableContentUpdateRequest,
    ) -> DataTableDetailResponse:
        response = await self._request(
            "PUT",
            f"/api/v1/data-tables/{table_uuid}/content",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return DataTableDetailResponse.model_validate(response)

    async def update_data_table(
        self,
        table_uuid: str,
        request_data: DataTableUpdateRequest,
    ) -> DataTableSummary:
        response = await self._request(
            "PATCH",
            f"/api/v1/data-tables/{table_uuid}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return DataTableSummary.model_validate(response)

    async def delete_data_table(self, table_uuid: str) -> DataTableDeleteResponse:
        response = await self._request("DELETE", f"/api/v1/data-tables/{table_uuid}")
        return DataTableDeleteResponse.model_validate(response)

    async def regenerate_data_table(
        self,
        table_uuid: str,
        request_data: DataTableRegenerateRequest,
        *,
        wait_for_completion: bool = False,
        wait_timeout_seconds: int = 300,
    ) -> DataTableGenerateResponse | DataTableDetailResponse:
        response = await self._request(
            "POST",
            f"/api/v1/data-tables/{table_uuid}/regenerate",
            params={
                "wait_for_completion": str(wait_for_completion).lower(),
                "wait_timeout_seconds": wait_timeout_seconds,
            },
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        if isinstance(response, dict) and "job_id" in response:
            return DataTableGenerateResponse.model_validate(response)
        return DataTableDetailResponse.model_validate(response)

    async def get_data_table_job(self, job_id: int) -> DataTableJobStatus:
        response = await self._request("GET", f"/api/v1/data-tables/jobs/{job_id}")
        return DataTableJobStatus.model_validate(response)

    async def cancel_data_table_job(
        self,
        job_id: int,
        *,
        reason: str | None = None,
    ) -> DataTableJobCancelResponse:
        response = await self._request(
            "DELETE",
            f"/api/v1/data-tables/jobs/{job_id}",
            params={key: value for key, value in {"reason": reason}.items() if value is not None},
        )
        return DataTableJobCancelResponse.model_validate(response)

    async def get_meetings_health(self) -> MeetingHealthResponse:
        response = await self._request("GET", "/api/v1/meetings/health")
        return MeetingHealthResponse.model_validate(response)

    async def create_meeting_session(self, request_data: MeetingSessionCreate) -> MeetingSessionResponse:
        response = await self._request(
            "POST",
            "/api/v1/meetings/sessions",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return MeetingSessionResponse.model_validate(response)

    async def list_meeting_sessions(
        self,
        *,
        status_filter: MeetingSessionStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[MeetingSessionResponse]:
        params = {"status": status_filter, "limit": limit, "offset": offset}
        response = await self._request(
            "GET",
            "/api/v1/meetings/sessions",
            params={key: value for key, value in params.items() if value is not None},
        )
        return [MeetingSessionResponse.model_validate(item) for item in response]

    async def get_meeting_session(self, session_id: str) -> MeetingSessionResponse:
        response = await self._request("GET", f"/api/v1/meetings/sessions/{session_id}")
        return MeetingSessionResponse.model_validate(response)

    async def update_meeting_session_status(
        self,
        session_id: str,
        request_data: MeetingSessionStatusUpdate,
    ) -> MeetingSessionResponse:
        response = await self._request(
            "POST",
            f"/api/v1/meetings/sessions/{session_id}/status",
            json_data=request_data.model_dump(mode="json"),
        )
        return MeetingSessionResponse.model_validate(response)

    async def create_meeting_template(self, request_data: MeetingTemplateCreate) -> MeetingTemplateResponse:
        response = await self._request(
            "POST",
            "/api/v1/meetings/templates",
            json_data=request_data.model_dump(mode="json", by_alias=True),
        )
        return MeetingTemplateResponse.model_validate(response)

    async def list_meeting_templates(
        self,
        *,
        scope: MeetingTemplateScope | None = None,
        include_disabled: bool = False,
    ) -> list[MeetingTemplateResponse]:
        params = {"scope": scope, "include_disabled": str(include_disabled).lower()}
        response = await self._request(
            "GET",
            "/api/v1/meetings/templates",
            params={key: value for key, value in params.items() if value is not None},
        )
        return [MeetingTemplateResponse.model_validate(item) for item in response]

    async def get_meeting_template(self, template_id: str) -> MeetingTemplateResponse:
        response = await self._request("GET", f"/api/v1/meetings/templates/{template_id}")
        return MeetingTemplateResponse.model_validate(response)

    async def create_meeting_artifact(
        self,
        session_id: str,
        request_data: MeetingArtifactCreate,
    ) -> MeetingArtifactResponse:
        response = await self._request(
            "POST",
            f"/api/v1/meetings/sessions/{session_id}/artifacts",
            json_data=request_data.model_dump(mode="json"),
        )
        return MeetingArtifactResponse.model_validate(response)

    async def list_meeting_artifacts(self, session_id: str) -> list[MeetingArtifactResponse]:
        response = await self._request("GET", f"/api/v1/meetings/sessions/{session_id}/artifacts")
        return [MeetingArtifactResponse.model_validate(item) for item in response]

    async def finalize_meeting_session(
        self,
        session_id: str,
        request_data: MeetingFinalizeRequest,
    ) -> MeetingFinalizeResponse:
        response = await self._request(
            "POST",
            f"/api/v1/meetings/sessions/{session_id}/commit",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return MeetingFinalizeResponse.model_validate(response)

    async def share_meeting_session_to_slack(
        self,
        session_id: str,
        request_data: MeetingShareRequest,
    ) -> MeetingShareResponse:
        response = await self._request(
            "POST",
            f"/api/v1/meetings/sessions/{session_id}/share/slack",
            json_data=request_data.model_dump(mode="json"),
        )
        return MeetingShareResponse.model_validate(response)

    async def share_meeting_session_to_webhook(
        self,
        session_id: str,
        request_data: MeetingShareRequest,
    ) -> MeetingShareResponse:
        response = await self._request(
            "POST",
            f"/api/v1/meetings/sessions/{session_id}/share/webhook",
            json_data=request_data.model_dump(mode="json"),
        )
        return MeetingShareResponse.model_validate(response)

    async def stream_meeting_session_events(
        self,
        session_id: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        async for event in self._sse_request(
            "GET",
            f"/api/v1/meetings/sessions/{session_id}/events",
        ):
            yield event

    async def get_slides_health(self) -> SlidesHealthResponse:
        response = await self._request("GET", "/api/v1/slides/health")
        return SlidesHealthResponse.model_validate(response)

    async def create_presentation(self, request_data: PresentationCreateRequest) -> PresentationResponse:
        response = await self._request(
            "POST",
            "/api/v1/slides/presentations",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return PresentationResponse.model_validate(response)

    async def list_presentations(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        sort: str | None = None,
        include_deleted: bool = False,
    ) -> PresentationListResponse:
        params = {
            "limit": limit,
            "offset": offset,
            "sort": sort,
            "include_deleted": str(include_deleted).lower(),
        }
        response = await self._request(
            "GET",
            "/api/v1/slides/presentations",
            params={key: value for key, value in params.items() if value is not None},
        )
        return PresentationListResponse.model_validate(response)

    async def search_presentations(
        self,
        query: str,
        *,
        limit: int = 50,
        offset: int = 0,
        include_deleted: bool = False,
    ) -> PresentationSearchResponse:
        response = await self._request(
            "GET",
            "/api/v1/slides/presentations/search",
            params={
                "q": query,
                "limit": limit,
                "offset": offset,
                "include_deleted": str(include_deleted).lower(),
            },
        )
        return PresentationSearchResponse.model_validate(response)

    async def get_presentation(
        self,
        presentation_id: str,
        *,
        include_deleted: bool = False,
    ) -> PresentationResponse:
        response = await self._request(
            "GET",
            f"/api/v1/slides/presentations/{presentation_id}",
            params={"include_deleted": str(include_deleted).lower()},
        )
        return PresentationResponse.model_validate(response)

    async def update_presentation(
        self,
        presentation_id: str,
        request_data: PresentationUpdateRequest,
        *,
        if_match: str | None = None,
    ) -> PresentationResponse:
        response = await self._request(
            "PUT",
            f"/api/v1/slides/presentations/{presentation_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
            headers=self._if_match_header(if_match),
        )
        return PresentationResponse.model_validate(response)

    async def patch_presentation(
        self,
        presentation_id: str,
        request_data: PresentationPatchRequest,
        *,
        if_match: str | None = None,
    ) -> PresentationResponse:
        response = await self._request(
            "PATCH",
            f"/api/v1/slides/presentations/{presentation_id}",
            json_data=request_data.model_dump(exclude_none=False, exclude_unset=True, mode="json"),
            headers=self._if_match_header(if_match),
        )
        return PresentationResponse.model_validate(response)

    async def reorder_presentation(
        self,
        presentation_id: str,
        order: list[int] | PresentationReorderRequest,
        *,
        if_match: str | None = None,
    ) -> PresentationResponse:
        request_data = order if isinstance(order, PresentationReorderRequest) else PresentationReorderRequest(order=order)
        response = await self._request(
            "POST",
            f"/api/v1/slides/presentations/{presentation_id}/reorder",
            json_data=request_data.model_dump(mode="json"),
            headers=self._if_match_header(if_match),
        )
        return PresentationResponse.model_validate(response)

    async def delete_presentation(
        self,
        presentation_id: str,
        *,
        if_match: str | None = None,
    ) -> PresentationResponse:
        response = await self._request(
            "DELETE",
            f"/api/v1/slides/presentations/{presentation_id}",
            headers=self._if_match_header(if_match),
        )
        return PresentationResponse.model_validate(response)

    async def restore_presentation(
        self,
        presentation_id: str,
        *,
        if_match: str | None = None,
    ) -> PresentationResponse:
        response = await self._request(
            "POST",
            f"/api/v1/slides/presentations/{presentation_id}/restore",
            headers=self._if_match_header(if_match),
        )
        return PresentationResponse.model_validate(response)

    async def list_slide_templates(self) -> SlidesTemplateListResponse:
        response = await self._request("GET", "/api/v1/slides/templates")
        return SlidesTemplateListResponse.model_validate(response)

    async def get_slide_template(self, template_id: str) -> SlidesTemplateResponse:
        response = await self._request("GET", f"/api/v1/slides/templates/{template_id}")
        return SlidesTemplateResponse.model_validate(response)

    async def list_visual_styles(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> VisualStyleListResponse:
        response = await self._request(
            "GET",
            "/api/v1/slides/styles",
            params={"limit": limit, "offset": offset},
        )
        return VisualStyleListResponse.model_validate(response)

    async def get_visual_style(self, style_id: str) -> VisualStyleResponse:
        response = await self._request("GET", f"/api/v1/slides/styles/{style_id}")
        return VisualStyleResponse.model_validate(response)

    async def create_visual_style(self, request_data: VisualStyleCreateRequest) -> VisualStyleResponse:
        response = await self._request(
            "POST",
            "/api/v1/slides/styles",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return VisualStyleResponse.model_validate(response)

    async def patch_visual_style(
        self,
        style_id: str,
        request_data: VisualStylePatchRequest,
    ) -> VisualStyleResponse:
        response = await self._request(
            "PATCH",
            f"/api/v1/slides/styles/{style_id}",
            json_data=request_data.model_dump(exclude_none=False, exclude_unset=True, mode="json"),
        )
        return VisualStyleResponse.model_validate(response)

    async def delete_visual_style(self, style_id: str) -> Dict[str, Any]:
        return await self._request("DELETE", f"/api/v1/slides/styles/{style_id}")

    async def list_presentation_versions(
        self,
        presentation_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> PresentationVersionListResponse:
        response = await self._request(
            "GET",
            f"/api/v1/slides/presentations/{presentation_id}/versions",
            params={"limit": limit, "offset": offset},
        )
        return PresentationVersionListResponse.model_validate(response)

    async def get_presentation_version(self, presentation_id: str, version: int) -> PresentationResponse:
        response = await self._request(
            "GET",
            f"/api/v1/slides/presentations/{presentation_id}/versions/{version}",
        )
        return PresentationResponse.model_validate(response)

    async def restore_presentation_version(
        self,
        presentation_id: str,
        version: int,
        *,
        if_match: str | None = None,
    ) -> PresentationResponse:
        response = await self._request(
            "POST",
            f"/api/v1/slides/presentations/{presentation_id}/versions/{version}/restore",
            headers=self._if_match_header(if_match),
        )
        return PresentationResponse.model_validate(response)

    async def submit_presentation_render_job(
        self,
        presentation_id: str,
        request_data: PresentationRenderRequest,
        *,
        if_match: str | None = None,
    ) -> PresentationRenderJobResponse:
        response = await self._request(
            "POST",
            f"/api/v1/slides/presentations/{presentation_id}/render-jobs",
            json_data=request_data.model_dump(mode="json"),
            headers=self._if_match_header(if_match),
        )
        return PresentationRenderJobResponse.model_validate(response)

    async def get_presentation_render_job_status(self, job_id: int) -> PresentationRenderJobStatusResponse:
        response = await self._request("GET", f"/api/v1/slides/render-jobs/{job_id}")
        return PresentationRenderJobStatusResponse.model_validate(response)

    async def list_presentation_render_artifacts(
        self,
        presentation_id: str,
    ) -> PresentationRenderArtifactListResponse:
        response = await self._request(
            "GET",
            f"/api/v1/slides/presentations/{presentation_id}/render-artifacts",
        )
        return PresentationRenderArtifactListResponse.model_validate(response)

    async def generate_presentation_from_prompt(
        self,
        request_data: GenerateFromPromptRequest,
    ) -> PresentationResponse:
        response = await self._request(
            "POST",
            "/api/v1/slides/generate",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return PresentationResponse.model_validate(response)

    async def generate_presentation_from_chat(self, request_data: GenerateFromChatRequest) -> PresentationResponse:
        response = await self._request(
            "POST",
            "/api/v1/slides/generate/from-chat",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return PresentationResponse.model_validate(response)

    async def generate_presentation_from_notes(self, request_data: GenerateFromNotesRequest) -> PresentationResponse:
        response = await self._request(
            "POST",
            "/api/v1/slides/generate/from-notes",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return PresentationResponse.model_validate(response)

    async def generate_presentation_from_media(self, request_data: GenerateFromMediaRequest) -> PresentationResponse:
        response = await self._request(
            "POST",
            "/api/v1/slides/generate/from-media",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return PresentationResponse.model_validate(response)

    async def generate_presentation_from_rag(self, request_data: GenerateFromRagRequest) -> PresentationResponse:
        response = await self._request(
            "POST",
            "/api/v1/slides/generate/from-rag",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return PresentationResponse.model_validate(response)

    async def export_presentation(
        self,
        presentation_id: str,
        *,
        format: SlidesExportFormat = "revealjs",
        pdf_format: str | None = None,
        pdf_width: str | None = None,
        pdf_height: str | None = None,
        pdf_landscape: bool | None = None,
        pdf_margin_top: str | None = None,
        pdf_margin_bottom: str | None = None,
        pdf_margin_left: str | None = None,
        pdf_margin_right: str | None = None,
    ) -> ReadingExportResponse:
        params = {
            "format": format,
            "pdf_format": pdf_format,
            "pdf_width": pdf_width,
            "pdf_height": pdf_height,
            "pdf_landscape": str(pdf_landscape).lower() if pdf_landscape is not None else None,
            "pdf_margin_top": pdf_margin_top,
            "pdf_margin_bottom": pdf_margin_bottom,
            "pdf_margin_left": pdf_margin_left,
            "pdf_margin_right": pdf_margin_right,
        }
        return await self._binary_request(
            "GET",
            f"/api/v1/slides/presentations/{presentation_id}/export",
            params={key: value for key, value in params.items() if value is not None},
        )

    async def get_tts_health(self) -> TTSHealthResponse:
        response = await self._request("GET", "/api/v1/audio/health")
        return TTSHealthResponse.model_validate(response)

    async def get_stt_health(
        self,
        *,
        model: str | None = None,
        warm: bool = False,
    ) -> TTSHealthResponse:
        response = await self._request(
            "GET",
            "/api/v1/audio/transcriptions/health",
            params={key: value for key, value in {"model": model, "warm": str(warm).lower()}.items() if value is not None},
        )
        return TTSHealthResponse.model_validate(response)

    async def list_tts_providers(self) -> TTSProvidersResponse:
        response = await self._request("GET", "/api/v1/audio/providers")
        return TTSProvidersResponse.model_validate(response)

    async def list_tts_voices(self, *, provider: str | None = None) -> TTSVoicesResponse:
        response = await self._request(
            "GET",
            "/api/v1/audio/voices/catalog",
            params={key: value for key, value in {"provider": provider}.items() if value is not None},
        )
        return TTSVoicesResponse.model_validate(response)

    async def get_audio_streaming_status(self) -> StreamingStatusResponse:
        response = await self._request("GET", "/api/v1/audio/stream/status")
        return StreamingStatusResponse.model_validate(response)

    async def get_audio_streaming_limits(self) -> StreamingLimitsResponse:
        response = await self._request("GET", "/api/v1/audio/stream/limits")
        return StreamingLimitsResponse.model_validate(response)

    async def test_audio_streaming(self) -> StreamingTestResponse:
        response = await self._request("POST", "/api/v1/audio/stream/test")
        return StreamingTestResponse.model_validate(response)

    async def create_speech_chat(self, request_data: SpeechChatRequest) -> SpeechChatResponse:
        response = await self._request(
            "POST",
            "/api/v1/audio/chat",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return SpeechChatResponse.model_validate(response)

    async def create_audio_speech(self, request_data: OpenAISpeechRequest) -> ReadingExportResponse:
        return await self._binary_request(
            "POST",
            "/api/v1/audio/speech",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )

    async def create_audio_speech_job(self, request_data: OpenAISpeechRequest) -> AudioSpeechJobCreateResponse:
        response = await self._request(
            "POST",
            "/api/v1/audio/speech/jobs",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return AudioSpeechJobCreateResponse.model_validate(response)

    async def list_audio_speech_job_artifacts(self, job_id: int) -> AudioSpeechJobArtifactsResponse:
        response = await self._request("GET", f"/api/v1/audio/speech/jobs/{job_id}/artifacts")
        return AudioSpeechJobArtifactsResponse.model_validate(response)

    async def submit_audio_job(self, request_data: SubmitAudioJobRequest) -> SubmitAudioJobResponse:
        response = await self._request(
            "POST",
            "/api/v1/audio/jobs/submit",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return SubmitAudioJobResponse.model_validate(response)

    async def get_audio_job(self, job_id: int) -> AudioJobResponse:
        response = await self._request("GET", f"/api/v1/audio/jobs/{job_id}")
        return AudioJobResponse.model_validate(response)

    async def stream_audio_job_progress(
        self,
        job_id: int,
        *,
        after_id: int = 0,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        async for event in self._sse_request(
            "GET",
            f"/api/v1/audio/jobs/{job_id}/progress/stream",
            params={"after_id": after_id},
        ):
            yield event

    async def encode_audio_tokenizer(
        self,
        request_data: AudioTokenizerEncodeRequest,
    ) -> AudioTokenizerEncodeResponse:
        response = await self._request(
            "POST",
            "/api/v1/audio/tokenizer/encode",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return AudioTokenizerEncodeResponse.model_validate(response)

    async def encode_audio_tokenizer_file(
        self,
        file_path: str,
        *,
        tokenizer_model: str | None = None,
        token_format: Literal["list", "base64"] = "list",
        sample_rate: int | None = None,
    ) -> AudioTokenizerEncodeResponse:
        httpx_files = prepare_files_for_httpx([file_path], upload_field_name="file")
        form_data = {
            key: value
            for key, value in {
                "tokenizer_model": tokenizer_model,
                "token_format": token_format,
                "sample_rate": str(sample_rate) if sample_rate is not None else None,
            }.items()
            if value is not None
        }
        try:
            response = await self._request(
                "POST",
                "/api/v1/audio/tokenizer/encode",
                data=form_data,
                files=httpx_files,
            )
            return AudioTokenizerEncodeResponse.model_validate(response)
        finally:
            cleanup_file_objects(httpx_files)

    async def decode_audio_tokenizer(
        self,
        request_data: AudioTokenizerDecodeRequest,
    ) -> ReadingExportResponse:
        return await self._binary_request(
            "POST",
            "/api/v1/audio/tokenizer/decode",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )

    async def upload_custom_voice(
        self,
        file_path: str,
        *,
        name: str,
        description: str | None = None,
        provider: str = "vibevoice",
        reference_text: str | None = None,
    ) -> CustomVoiceResponse:
        httpx_files = prepare_files_for_httpx([file_path], upload_field_name="file")
        form_data = {
            key: value
            for key, value in {
                "name": name,
                "description": description,
                "provider": provider,
                "reference_text": reference_text,
            }.items()
            if value is not None
        }
        try:
            response = await self._request(
                "POST",
                "/api/v1/audio/voices/upload",
                data=form_data,
                files=httpx_files,
            )
            return CustomVoiceResponse.model_validate(response)
        finally:
            cleanup_file_objects(httpx_files)

    async def encode_custom_voice_reference(
        self,
        request_data: VoiceEncodeRequest,
    ) -> VoiceEncodeResponse:
        response = await self._request(
            "POST",
            "/api/v1/audio/voices/encode",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return VoiceEncodeResponse.model_validate(response)

    async def list_custom_voices(self) -> CustomVoiceListResponse:
        response = await self._request("GET", "/api/v1/audio/voices")
        return CustomVoiceListResponse.model_validate(response)

    async def get_custom_voice(self, voice_id: str) -> CustomVoiceResponse:
        response = await self._request("GET", f"/api/v1/audio/voices/{voice_id}")
        return CustomVoiceResponse.model_validate(response)

    async def delete_custom_voice(self, voice_id: str) -> CustomVoiceDeleteResponse:
        response = await self._request("DELETE", f"/api/v1/audio/voices/{voice_id}")
        return CustomVoiceDeleteResponse.model_validate(response)

    async def preview_custom_voice(
        self,
        voice_id: str,
        *,
        text: str = "Hello, this is a preview of your custom voice.",
    ) -> ReadingExportResponse:
        return await self._binary_request(
            "POST",
            f"/api/v1/audio/voices/{voice_id}/preview",
            data={"text": text},
        )

    async def parse_audiobook_source(self, request_data: AudiobookParseRequest) -> AudiobookParseResponse:
        response = await self._request(
            "POST",
            "/api/v1/audiobooks/parse",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return AudiobookParseResponse.model_validate(response)

    async def create_audiobook_job(self, request_data: AudiobookJobRequest) -> AudiobookJobCreateResponse:
        response = await self._request(
            "POST",
            "/api/v1/audiobooks/jobs",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return AudiobookJobCreateResponse.model_validate(response)

    async def get_audiobook_job_status(self, job_id: int) -> AudiobookJobStatusResponse:
        response = await self._request("GET", f"/api/v1/audiobooks/jobs/{job_id}")
        return AudiobookJobStatusResponse.model_validate(response)

    async def list_audiobook_job_artifacts(self, job_id: int) -> AudiobookArtifactsResponse:
        response = await self._request("GET", f"/api/v1/audiobooks/jobs/{job_id}/artifacts")
        return AudiobookArtifactsResponse.model_validate(response)

    async def list_audiobook_projects(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> AudiobookProjectListResponse:
        response = await self._request(
            "GET",
            "/api/v1/audiobooks/projects",
            params={"limit": limit, "offset": offset},
        )
        return AudiobookProjectListResponse.model_validate(response)

    async def get_audiobook_project(self, project_ref: str) -> AudiobookProjectResponse:
        response = await self._request("GET", f"/api/v1/audiobooks/projects/{project_ref}")
        return AudiobookProjectResponse.model_validate(response)

    async def list_audiobook_project_chapters(
        self,
        project_ref: str,
        *,
        limit: int = 200,
        offset: int = 0,
    ) -> AudiobookChapterListResponse:
        response = await self._request(
            "GET",
            f"/api/v1/audiobooks/projects/{project_ref}/chapters",
            params={"limit": limit, "offset": offset},
        )
        return AudiobookChapterListResponse.model_validate(response)

    async def list_audiobook_project_artifacts(
        self,
        project_ref: str,
        *,
        limit: int = 200,
        offset: int = 0,
    ) -> AudiobookArtifactsResponse:
        response = await self._request(
            "GET",
            f"/api/v1/audiobooks/projects/{project_ref}/artifacts",
            params={"limit": limit, "offset": offset},
        )
        return AudiobookArtifactsResponse.model_validate(response)

    async def create_audiobook_voice_profile(
        self,
        request_data: VoiceProfileCreateRequest,
    ) -> VoiceProfileResponse:
        response = await self._request(
            "POST",
            "/api/v1/audiobooks/voices/profiles",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return VoiceProfileResponse.model_validate(response)

    async def list_audiobook_voice_profiles(self) -> VoiceProfileListResponse:
        response = await self._request("GET", "/api/v1/audiobooks/voices/profiles")
        return VoiceProfileListResponse.model_validate(response)

    async def delete_audiobook_voice_profile(self, profile_id: str) -> VoiceProfileDeleteResponse:
        response = await self._request("DELETE", f"/api/v1/audiobooks/voices/profiles/{profile_id}")
        return VoiceProfileDeleteResponse.model_validate(response)

    async def export_audiobook_subtitles(self, request_data: SubtitleExportRequest) -> ReadingExportResponse:
        return await self._binary_request(
            "POST",
            "/api/v1/audiobooks/subtitles",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )

    async def list_tts_history(
        self,
        *,
        q: str | None = None,
        text_exact: str | None = None,
        favorite: bool | None = None,
        provider: str | None = None,
        model: str | None = None,
        voice_id: str | None = None,
        voice_name: str | None = None,
        limit: int = 50,
        offset: int = 0,
        cursor: str | None = None,
        include_total: bool = False,
        from_: str | None = None,
        to: str | None = None,
    ) -> TTSHistoryListResponse:
        params = {
            "q": q,
            "text_exact": text_exact,
            "favorite": str(favorite).lower() if favorite is not None else None,
            "provider": provider,
            "model": model,
            "voice_id": voice_id,
            "voice_name": voice_name,
            "limit": limit,
            "offset": offset,
            "cursor": cursor,
            "include_total": str(include_total).lower(),
            "from": from_,
            "to": to,
        }
        response = await self._request(
            "GET",
            "/api/v1/audio/history",
            params={key: value for key, value in params.items() if value is not None},
        )
        return TTSHistoryListResponse.model_validate(response)

    async def get_tts_history_entry(self, history_id: int) -> TTSHistoryDetailResponse:
        response = await self._request("GET", f"/api/v1/audio/history/{history_id}")
        return TTSHistoryDetailResponse.model_validate(response)

    async def update_tts_history_favorite(
        self,
        history_id: int,
        request_data: TTSHistoryFavoriteUpdate,
    ) -> Dict[str, Any]:
        return await self._request(
            "PATCH",
            f"/api/v1/audio/history/{history_id}",
            json_data=request_data.model_dump(mode="json"),
        )

    async def delete_tts_history_entry(self, history_id: int) -> Dict[str, Any]:
        return await self._request("DELETE", f"/api/v1/audio/history/{history_id}")

    async def create_audio_transcription(
        self,
        file_path: str,
        request_data: AudioTranscriptionRequest | None = None,
    ) -> AudioTranscriptionResponse:
        payload = request_data or AudioTranscriptionRequest()
        httpx_files = prepare_files_for_httpx([file_path], upload_field_name="file")
        try:
            response = await self._request(
                "POST",
                "/api/v1/audio/transcriptions",
                data=model_to_form_data(payload),
                files=httpx_files,
            )
            return AudioTranscriptionResponse.model_validate(response)
        finally:
            cleanup_file_objects(httpx_files)

    async def create_audio_translation(
        self,
        file_path: str,
        request_data: AudioTranslationRequest | None = None,
    ) -> AudioTranscriptionResponse:
        payload = request_data or AudioTranslationRequest()
        httpx_files = prepare_files_for_httpx([file_path], upload_field_name="file")
        try:
            response = await self._request(
                "POST",
                "/api/v1/audio/translations",
                data=model_to_form_data(payload),
                files=httpx_files,
            )
            return AudioTranscriptionResponse.model_validate(response)
        finally:
            cleanup_file_objects(httpx_files)

    async def submit_media_ingest_jobs(
        self,
        request_data: MediaIngestSubmitRequest | MediaIngestJobSubmitRequest,
        file_paths: Optional[List[str]] = None,
    ) -> SubmitMediaIngestJobsResponse:
        httpx_files = prepare_files_for_httpx(file_paths or []) if file_paths else None
        form_data = (
            model_to_form_data(request_data)
            if httpx_files or isinstance(request_data, MediaIngestJobSubmitRequest)
            else request_data.model_dump(exclude_none=True, mode="json")
        )
        try:
            response = await self._request(
                "POST",
                "/api/v1/media/ingest/jobs",
                data=form_data,
                files=httpx_files,
            )
            return SubmitMediaIngestJobsResponse.model_validate(response)
        finally:
            if httpx_files:
                cleanup_file_objects(httpx_files)

    async def ingest_web_content(self, request_data: IngestWebContentRequest) -> IngestWebContentResponse:
        response = await self._request(
            "POST",
            "/api/v1/media/ingest-web-content",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return IngestWebContentResponse.model_validate(response)

    async def get_media_ingest_job(self, job_id: int) -> MediaIngestJobStatus:
        response = await self._request("GET", f"/api/v1/media/ingest/jobs/{job_id}")
        return MediaIngestJobStatus.model_validate(response)

    async def list_media_ingest_jobs(self, batch_id: str, *, limit: int = 100) -> MediaIngestJobListResponse:
        response = await self._request(
            "GET",
            "/api/v1/media/ingest/jobs",
            params={"batch_id": batch_id, "limit": limit},
        )
        return MediaIngestJobListResponse.model_validate(response)

    async def stream_media_ingest_job_events(
        self,
        *,
        batch_id: str | None = None,
        after_id: int = 0,
    ) -> AsyncGenerator[MediaIngestJobStreamEvent, None]:
        params = {"batch_id": batch_id, "after_id": after_id}
        async for event in self._sse_request(
            "GET",
            "/api/v1/media/ingest/jobs/events/stream",
            params={key: value for key, value in params.items() if value is not None},
            headers={"Accept": "text/event-stream"},
        ):
            if "event_id" in event and "id" not in event:
                event["id"] = event.pop("event_id")
            yield MediaIngestJobStreamEvent.model_validate(event)

    async def cancel_media_ingest_job(self, job_id: int, *, reason: str | None = None) -> CancelMediaIngestJobResponse:
        response = await self._request(
            "DELETE",
            f"/api/v1/media/ingest/jobs/{job_id}",
            params={key: value for key, value in {"reason": reason}.items() if value is not None},
        )
        return CancelMediaIngestJobResponse.model_validate(response)

    async def cancel_media_ingest_batch(
        self,
        *,
        batch_id: str | None = None,
        session_id: str | None = None,
        reason: str | None = None,
    ) -> CancelMediaIngestBatchResponse:
        response = await self._request(
            "POST",
            "/api/v1/media/ingest/jobs/cancel",
            params={
                key: value
                for key, value in {"batch_id": batch_id, "session_id": session_id, "reason": reason}.items()
                if value is not None
            },
        )
        return CancelMediaIngestBatchResponse.model_validate(response)

    async def cancel_media_ingest_jobs_batch(
        self,
        *,
        batch_id: str | None = None,
        session_id: str | None = None,
        reason: str | None = None,
    ) -> CancelMediaIngestBatchResponse:
        return await self.cancel_media_ingest_batch(
            batch_id=batch_id,
            session_id=session_id,
            reason=reason,
        )

    async def reprocess_media(self, media_id: int, request_data: ReprocessMediaRequest) -> ReprocessMediaResponse:
        response = await self._request(
            "POST",
            f"/api/v1/media/{media_id}/reprocess",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return ReprocessMediaResponse.model_validate(response)

    async def list_media_versions(
        self,
        media_id: int,
        *,
        include_content: bool = False,
        limit: int = 10,
        page: int = 1,
    ) -> list[MediaVersionDetail]:
        response = await self._request(
            "GET",
            f"/api/v1/media/{media_id}/versions",
            params={
                "include_content": str(include_content).lower(),
                "limit": limit,
                "page": page,
            },
        )
        return [MediaVersionDetail.model_validate(item) for item in response]

    async def get_media_version(
        self,
        media_id: int,
        version_number: int,
        *,
        include_content: bool = True,
    ) -> MediaVersionDetail:
        response = await self._request(
            "GET",
            f"/api/v1/media/{media_id}/versions/{version_number}",
            params={"include_content": str(include_content).lower()},
        )
        return MediaVersionDetail.model_validate(response)

    async def create_media_version(
        self,
        media_id: int,
        request_data: MediaVersionCreateRequest,
    ) -> Dict[str, Any]:
        return await self._request(
            "POST",
            f"/api/v1/media/{media_id}/versions",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )

    async def delete_media_version(self, media_id: int, version_number: int) -> Dict[str, Any]:
        return await self._request("DELETE", f"/api/v1/media/{media_id}/versions/{version_number}")

    async def rollback_media_version(
        self,
        media_id: int,
        request_data: MediaVersionRollbackRequest,
    ) -> Dict[str, Any]:
        return await self._request(
            "POST",
            f"/api/v1/media/{media_id}/versions/rollback",
            json_data=request_data.model_dump(mode="json"),
        )

    async def patch_media_metadata(
        self,
        media_id: int,
        request_data: MediaMetadataPatchRequest,
    ) -> Dict[str, Any]:
        return await self._request(
            "PATCH",
            f"/api/v1/media/{media_id}/metadata",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )

    async def put_media_version_metadata(
        self,
        media_id: int,
        version_number: int,
        request_data: MediaMetadataPatchRequest,
    ) -> Dict[str, Any]:
        return await self._request(
            "PUT",
            f"/api/v1/media/{media_id}/versions/{version_number}/metadata",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )

    async def upsert_media_version_advanced(
        self,
        media_id: int,
        request_data: MediaAdvancedVersionUpsertRequest,
    ) -> Dict[str, Any]:
        return await self._request(
            "POST",
            f"/api/v1/media/{media_id}/versions/advanced",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )

    async def list_media_document_versions(
        self,
        media_id: int,
        *,
        include_content: bool = False,
        limit: int = 10,
        page: int = 1,
    ) -> list[DocumentVersionDetailResponse]:
        response = await self._request(
            "GET",
            f"/api/v1/media/{media_id}/versions",
            params={
                "include_content": str(include_content).lower(),
                "limit": limit,
                "page": page,
            },
        )
        return [DocumentVersionDetailResponse.model_validate(item) for item in response]

    async def get_media_document_version(
        self,
        media_id: int,
        version_number: int,
        *,
        include_content: bool = True,
    ) -> DocumentVersionDetailResponse:
        response = await self._request(
            "GET",
            f"/api/v1/media/{media_id}/versions/{version_number}",
            params={"include_content": str(include_content).lower()},
        )
        return DocumentVersionDetailResponse.model_validate(response)

    async def create_media_document_version(
        self,
        media_id: int,
        request_data: DocumentVersionCreateRequest,
    ) -> Dict[str, Any]:
        return await self._request(
            "POST",
            f"/api/v1/media/{media_id}/versions",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )

    async def delete_media_document_version(self, media_id: int, version_number: int) -> Dict[str, Any]:
        response = await self._request("DELETE", f"/api/v1/media/{media_id}/versions/{version_number}")
        return {"deleted": True, **response}

    async def rollback_media_document_version(
        self,
        media_id: int,
        request_data: DocumentVersionRollbackRequest,
    ) -> Dict[str, Any]:
        return await self._request(
            "POST",
            f"/api/v1/media/{media_id}/versions/rollback",
            json_data=request_data.model_dump(mode="json"),
        )

    async def patch_media_document_metadata(
        self,
        media_id: int,
        request_data: DocumentVersionMetadataPatchRequest,
    ) -> Dict[str, Any]:
        return await self._request(
            "PATCH",
            f"/api/v1/media/{media_id}/metadata",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )

    async def update_media_document_version_metadata(
        self,
        media_id: int,
        version_number: int,
        request_data: DocumentVersionMetadataPatchRequest,
    ) -> Dict[str, Any]:
        return await self._request(
            "PUT",
            f"/api/v1/media/{media_id}/versions/{version_number}/metadata",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )

    async def upsert_media_document_version(
        self,
        media_id: int,
        request_data: DocumentVersionAdvancedUpsertRequest,
    ) -> Dict[str, Any]:
        return await self._request(
            "POST",
            f"/api/v1/media/{media_id}/versions/advanced",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )

    async def create_ingestion_source(self, request_data: IngestionSourceCreateRequest) -> IngestionSourceResponse:
        response = await self._request(
            "POST",
            "/api/v1/ingestion-sources/",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return IngestionSourceResponse.model_validate(response)

    async def list_ingestion_sources(self) -> IngestionSourceListResponse:
        response = await self._request("GET", "/api/v1/ingestion-sources/")
        return [IngestionSourceResponse.model_validate(item) for item in response]

    async def get_ingestion_source(self, source_id: int) -> IngestionSourceResponse:
        response = await self._request("GET", f"/api/v1/ingestion-sources/{source_id}")
        return IngestionSourceResponse.model_validate(response)

    async def patch_ingestion_source(
        self,
        source_id: int,
        request_data: IngestionSourcePatchRequest,
    ) -> IngestionSourceResponse:
        response = await self._request(
            "PATCH",
            f"/api/v1/ingestion-sources/{source_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return IngestionSourceResponse.model_validate(response)

    async def list_ingestion_source_items(self, source_id: int) -> IngestionSourceItemListResponse:
        response = await self._request("GET", f"/api/v1/ingestion-sources/{source_id}/items")
        return [IngestionSourceItemResponse.model_validate(item) for item in response]

    async def reattach_ingestion_source_item(self, source_id: int, item_id: int) -> IngestionSourceItemResponse:
        response = await self._request(
            "POST",
            f"/api/v1/ingestion-sources/{source_id}/items/{item_id}/reattach",
        )
        return IngestionSourceItemResponse.model_validate(response)

    async def trigger_ingestion_source_sync(self, source_id: int) -> IngestionSourceSyncTriggerResponse:
        response = await self._request("POST", f"/api/v1/ingestion-sources/{source_id}/sync")
        return IngestionSourceSyncTriggerResponse.model_validate(response)

    async def upload_ingestion_source_archive(self, source_id: int, archive_path: str) -> IngestionSourceSyncTriggerResponse:
        httpx_files = prepare_files_for_httpx([archive_path], upload_field_name="archive")
        try:
            response = await self._request(
                "POST",
                f"/api/v1/ingestion-sources/{source_id}/archive",
                files=httpx_files,
            )
            return IngestionSourceSyncTriggerResponse.model_validate(response)
        finally:
            cleanup_file_objects(httpx_files)

    async def reattach_ingestion_source_item(self, source_id: int, item_id: int) -> IngestionSourceItemResponse:
        response = await self._request(
            "POST",
            f"/api/v1/ingestion-sources/{source_id}/items/{item_id}/reattach",
        )
        return IngestionSourceItemResponse.model_validate(response)

    async def save_reading_item(self, request_data: ReadingSaveRequest) -> ReadingItem:
        response = await self._request(
            "POST",
            "/api/v1/reading/save",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return ReadingItem.model_validate(response)

    async def list_reading_items(
        self,
        *,
        status: list[str] | None = None,
        tags: list[str] | None = None,
        q: str | None = None,
        domain: str | None = None,
        favorite: bool | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        page: int = 1,
        size: int = 20,
        offset: int | None = None,
        limit: int | None = None,
        sort: str | None = None,
    ) -> Dict[str, Any]:
        params = {
            "status": status,
            "tags": tags,
            "q": q,
            "domain": domain,
            "favorite": favorite,
            "date_from": date_from,
            "date_to": date_to,
            "sort": sort,
        }
        if offset is not None or limit is not None:
            params["offset"] = offset
            params["limit"] = limit
        else:
            params["page"] = page
            params["size"] = size
        return await self._request(
            "GET",
            "/api/v1/reading/items",
            params={key: value for key, value in params.items() if value is not None},
        )

    async def get_reading_item(self, item_id: int) -> Dict[str, Any]:
        return await self._request("GET", f"/api/v1/reading/items/{item_id}")

    async def update_reading_item(self, item_id: int, request_data: ReadingUpdateRequest) -> Dict[str, Any]:
        return await self._request(
            "PATCH",
            f"/api/v1/reading/items/{item_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )

    async def delete_reading_item(self, item_id: int, hard: bool = False) -> Dict[str, Any]:
        return await self._request(
            "DELETE",
            f"/api/v1/reading/items/{item_id}",
            params={"hard": str(hard).lower()},
        )

    async def create_reading_saved_search(
        self,
        request_data: ReadingSavedSearchCreateRequest,
    ) -> ReadingSavedSearchResponse:
        response = await self._request(
            "POST",
            "/api/v1/reading/saved-searches",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return ReadingSavedSearchResponse.model_validate(response)

    async def list_reading_saved_searches(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> ReadingSavedSearchListResponse:
        response = await self._request(
            "GET",
            "/api/v1/reading/saved-searches",
            params={"limit": limit, "offset": offset},
        )
        return ReadingSavedSearchListResponse.model_validate(response)

    async def update_reading_saved_search(
        self,
        search_id: int,
        request_data: ReadingSavedSearchUpdateRequest,
    ) -> ReadingSavedSearchResponse:
        response = await self._request(
            "PATCH",
            f"/api/v1/reading/saved-searches/{search_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return ReadingSavedSearchResponse.model_validate(response)

    async def delete_reading_saved_search(self, search_id: int) -> Dict[str, Any]:
        return await self._request("DELETE", f"/api/v1/reading/saved-searches/{search_id}")

    async def link_note_to_reading_item(
        self,
        item_id: int,
        note_id: str,
    ) -> ReadingNoteLinkResponse:
        request_data = ReadingNoteLinkCreateRequest(note_id=note_id)
        response = await self._request(
            "POST",
            f"/api/v1/reading/items/{item_id}/links/note",
            json_data=request_data.model_dump(mode="json"),
        )
        return ReadingNoteLinkResponse.model_validate(response)

    async def list_reading_item_note_links(self, item_id: int) -> ReadingNoteLinksListResponse:
        response = await self._request("GET", f"/api/v1/reading/items/{item_id}/links")
        return ReadingNoteLinksListResponse.model_validate(response)

    async def unlink_note_from_reading_item(self, item_id: int, note_id: str) -> Dict[str, Any]:
        return await self._request("DELETE", f"/api/v1/reading/items/{item_id}/links/note/{note_id}")

    async def link_reading_item_note(
        self,
        item_id: int,
        request_data: ReadingNoteLinkCreateRequest,
    ) -> ReadingNoteLinkResponse:
        response = await self._request(
            "POST",
            f"/api/v1/reading/items/{item_id}/links/note",
            json_data=request_data.model_dump(mode="json"),
        )
        return ReadingNoteLinkResponse.model_validate(response)

    async def unlink_reading_item_note(self, item_id: int, note_id: str) -> Dict[str, Any]:
        return await self._request("DELETE", f"/api/v1/reading/items/{item_id}/links/note/{note_id}")

    async def bulk_update_reading_items(self, request_data: ItemsBulkRequest) -> ItemsBulkResponse:
        response = await self._request(
            "POST",
            "/api/v1/reading/items/bulk",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return ItemsBulkResponse.model_validate(response)

    async def list_unified_items(
        self,
        *,
        q: str | None = None,
        tags: list[str] | None = None,
        status_filter: list[str] | None = None,
        origin: str | None = None,
        page: int = 1,
        size: int = 50,
    ) -> UnifiedItemsListResponse:
        params = {
            "q": q,
            "tags": tags,
            "status_filter": status_filter,
            "origin": origin,
            "page": page,
            "size": size,
        }
        response = await self._request(
            "GET",
            "/api/v1/items",
            params={key: value for key, value in params.items() if value is not None},
        )
        return UnifiedItemsListResponse.model_validate(response)

    async def get_unified_item(self, item_id: int) -> UnifiedItem:
        response = await self._request("GET", f"/api/v1/items/{item_id}")
        return UnifiedItem.model_validate(response)

    async def bulk_update_unified_items(self, request_data: ItemsBulkRequest) -> ItemsBulkResponse:
        response = await self._request(
            "POST",
            "/api/v1/items/bulk",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return ItemsBulkResponse.model_validate(response)

    async def create_reading_archive(
        self,
        item_id: int,
        request_data: ReadingArchiveCreateRequest | None = None,
    ) -> ReadingArchiveResponse:
        payload = (request_data or ReadingArchiveCreateRequest()).model_dump(exclude_none=True, mode="json")
        response = await self._request(
            "POST",
            f"/api/v1/reading/items/{item_id}/archive",
            json_data=payload,
        )
        return ReadingArchiveResponse.model_validate(response)

    async def summarize_reading_item(
        self,
        item_id: int,
        request_data: ReadingSummarizeRequest | None = None,
    ) -> ReadingSummaryResponse:
        payload = (request_data or ReadingSummarizeRequest()).model_dump(exclude_none=True, mode="json")
        response = await self._request(
            "POST",
            f"/api/v1/reading/items/{item_id}/summarize",
            json_data=payload,
        )
        return ReadingSummaryResponse.model_validate(response)

    async def tts_reading_item(
        self,
        item_id: int,
        request_data: ReadingTTSRequest,
    ) -> ReadingTTSResponse | bytes:
        payload = request_data.model_dump(exclude_none=True, mode="json")
        request_bytes_override = self.__dict__.get("_request_bytes")
        if request_bytes_override is not None:
            return await request_bytes_override(
                "POST",
                f"/api/v1/reading/items/{item_id}/tts",
                json_data=payload,
            )
        response = await self._binary_request(
            "POST",
            f"/api/v1/reading/items/{item_id}/tts",
            json_data=payload,
        )
        return ReadingTTSResponse(
            item_id=item_id,
            content=response.content,
            content_type=response.content_type,
            content_disposition=response.content_disposition,
            filename=response.filename,
        )

    async def import_reading_items(
        self,
        import_path: str,
        *,
        source: str = "auto",
        merge_tags: bool = True,
    ) -> ReadingImportJobResponse:
        httpx_files = prepare_files_for_httpx([import_path], upload_field_name="file")
        try:
            response = await self._request(
                "POST",
                "/api/v1/reading/import",
                data={"source": source, "merge_tags": str(merge_tags).lower()},
                files=httpx_files,
            )
            return ReadingImportJobResponse.model_validate(response)
        finally:
            cleanup_file_objects(httpx_files)

    async def export_reading_items(
        self,
        request_data: ReadingExportRequest | None = None,
        *,
        status: list[str] | None = None,
        tags: list[str] | None = None,
        favorite: bool | None = None,
        q: str | None = None,
        domain: str | None = None,
        page: int = 1,
        size: int = 1000,
        include_metadata: bool = True,
        include_clean_html: bool = False,
        include_text: bool = False,
        include_highlights: bool = False,
        include_notes: bool = True,
        format: str = "jsonl",
    ) -> ReadingExportResponse | bytes:
        if request_data is not None:
            return await self._request_bytes(
                "GET",
                "/api/v1/reading/export",
                params=request_data.model_dump(exclude_none=True, mode="json"),
            )
        params: dict[str, Any] = {
            "status": status,
            "tags": tags,
            "favorite": str(favorite).lower() if favorite is not None else None,
            "q": q,
            "domain": domain,
            "page": page,
            "size": size,
            "include_metadata": str(include_metadata).lower(),
            "include_clean_html": str(include_clean_html).lower(),
            "include_text": str(include_text).lower(),
            "include_highlights": str(include_highlights).lower(),
            "include_notes": str(include_notes).lower(),
            "format": format,
        }
        return await self._binary_request(
            "GET",
            "/api/v1/reading/export",
            params={key: value for key, value in params.items() if value is not None},
        )

    async def list_reading_import_jobs(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> ReadingImportJobsListResponse:
        response = await self._request(
            "GET",
            "/api/v1/reading/import/jobs",
            params={key: value for key, value in {"status": status, "limit": limit, "offset": offset}.items() if value is not None},
        )
        return ReadingImportJobsListResponse.model_validate(response)

    async def get_reading_import_job(self, job_id: int) -> ReadingImportJobStatus:
        response = await self._request("GET", f"/api/v1/reading/import/jobs/{job_id}")
        return ReadingImportJobStatus.model_validate(response)

    async def create_reading_digest_schedule(
        self,
        request_data: ReadingDigestScheduleCreateRequest,
    ) -> Dict[str, str]:
        return await self._request(
            "POST",
            "/api/v1/reading/digests/schedules",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )

    async def list_reading_digest_schedules(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ReadingDigestScheduleResponse]:
        response = await self._request(
            "GET",
            "/api/v1/reading/digests/schedules",
            params={"limit": limit, "offset": offset},
        )
        return [ReadingDigestScheduleResponse.model_validate(item) for item in response]

    async def get_reading_digest_schedule(self, schedule_id: str) -> ReadingDigestScheduleResponse:
        response = await self._request("GET", f"/api/v1/reading/digests/schedules/{schedule_id}")
        return ReadingDigestScheduleResponse.model_validate(response)

    async def update_reading_digest_schedule(
        self,
        schedule_id: str,
        request_data: ReadingDigestScheduleUpdateRequest,
    ) -> ReadingDigestScheduleResponse:
        response = await self._request(
            "PATCH",
            f"/api/v1/reading/digests/schedules/{schedule_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return ReadingDigestScheduleResponse.model_validate(response)

    async def delete_reading_digest_schedule(self, schedule_id: str) -> Dict[str, bool]:
        return await self._request("DELETE", f"/api/v1/reading/digests/schedules/{schedule_id}")

    async def list_reading_digest_outputs(
        self,
        *,
        schedule_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> ReadingDigestOutputsListResponse:
        response = await self._request(
            "GET",
            "/api/v1/reading/digests/outputs",
            params={
                key: value
                for key, value in {"schedule_id": schedule_id, "limit": limit, "offset": offset}.items()
                if value is not None
            },
        )
        return ReadingDigestOutputsListResponse.model_validate(response)

    async def get_reading_progress(self, media_id: int) -> Dict[str, Any]:
        return await self._request("GET", f"/api/v1/media/{media_id}/progress")

    async def update_reading_progress(self, media_id: int, request_data: ReadingProgressUpdate) -> Dict[str, Any]:
        return await self._request(
            "PUT",
            f"/api/v1/media/{media_id}/progress",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )

    async def delete_reading_progress(self, media_id: int) -> Dict[str, Any]:
        return await self._request("DELETE", f"/api/v1/media/{media_id}/progress")

    async def create_reading_highlight(
        self,
        item_id: int,
        request_data: ReadingHighlightCreateRequest,
    ) -> ReadingHighlight:
        response = await self._request(
            "POST",
            f"/api/v1/reading/items/{item_id}/highlight",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return ReadingHighlight.model_validate(response)

    async def list_reading_highlights(self, item_id: int) -> list[ReadingHighlight]:
        response = await self._request("GET", f"/api/v1/reading/items/{item_id}/highlights")
        return [ReadingHighlight.model_validate(item) for item in response]

    async def update_reading_highlight(
        self,
        highlight_id: int,
        request_data: ReadingHighlightUpdateRequest,
    ) -> ReadingHighlight:
        response = await self._request(
            "PATCH",
            f"/api/v1/reading/highlights/{highlight_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return ReadingHighlight.model_validate(response)

    async def delete_reading_highlight(self, highlight_id: int) -> ReadingHighlightDeleteResponse:
        response = await self._request("DELETE", f"/api/v1/reading/highlights/{highlight_id}")
        return ReadingHighlightDeleteResponse.model_validate(response)

    async def list_document_annotations(self, media_id: int) -> DocumentAnnotationListResponse:
        response = await self._request("GET", f"/api/v1/media/{media_id}/annotations")
        return DocumentAnnotationListResponse.model_validate(response)

    async def create_document_annotation(
        self,
        media_id: int,
        request_data: DocumentAnnotationCreate,
    ) -> DocumentAnnotationResponse:
        response = await self._request(
            "POST",
            f"/api/v1/media/{media_id}/annotations",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return DocumentAnnotationResponse.model_validate(response)

    async def update_document_annotation(
        self,
        media_id: int,
        annotation_id: str,
        request_data: DocumentAnnotationUpdate,
    ) -> DocumentAnnotationResponse:
        response = await self._request(
            "PUT",
            f"/api/v1/media/{media_id}/annotations/{annotation_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return DocumentAnnotationResponse.model_validate(response)

    async def delete_document_annotation(self, media_id: int, annotation_id: str) -> Dict[str, Any]:
        return await self._request("DELETE", f"/api/v1/media/{media_id}/annotations/{annotation_id}")

    async def sync_document_annotations(
        self,
        media_id: int,
        request_data: DocumentAnnotationSyncRequest,
    ) -> DocumentAnnotationSyncResponse:
        response = await self._request(
            "POST",
            f"/api/v1/media/{media_id}/annotations/sync",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return DocumentAnnotationSyncResponse.model_validate(response)

    async def get_document_outline(self, media_id: int) -> DocumentOutlineResponse:
        response = await self._request("GET", f"/api/v1/media/{media_id}/outline")
        return DocumentOutlineResponse.model_validate(response)

    async def get_document_figures(self, media_id: int, *, min_size: int = 50) -> DocumentFiguresResponse:
        response = await self._request(
            "GET",
            f"/api/v1/media/{media_id}/figures",
            params={"min_size": min_size},
        )
        return DocumentFiguresResponse.model_validate(response)

    async def get_document_references(
        self,
        media_id: int,
        *,
        enrich: bool = False,
        reference_index: int | None = None,
        offset: int = 0,
        limit: int = 50,
        parse_cap: int | None = None,
        search: str | None = None,
    ) -> DocumentReferencesResponse:
        params = {
            "enrich": str(enrich).lower(),
            "reference_index": reference_index,
            "offset": offset,
            "limit": limit,
            "parse_cap": parse_cap,
            "search": search,
        }
        response = await self._request(
            "GET",
            f"/api/v1/media/{media_id}/references",
            params={key: value for key, value in params.items() if value is not None},
        )
        return DocumentReferencesResponse.model_validate(response)

    async def get_media_navigation(
        self,
        media_id: int,
        *,
        include_generated_fallback: bool = False,
        max_depth: int = 4,
        max_nodes: int = 500,
        parent_id: str | None = None,
    ) -> MediaNavigationResponse:
        params = {
            "include_generated_fallback": "true" if include_generated_fallback else False,
            "max_depth": max_depth,
            "max_nodes": max_nodes,
            "parent_id": parent_id,
        }
        response = await self._request(
            "GET",
            f"/api/v1/media/{media_id}/navigation",
            params={key: value for key, value in params.items() if value is not None},
        )
        return MediaNavigationResponse.model_validate(response)

    async def get_media_navigation_content(
        self,
        media_id: int,
        node_id: str,
        *,
        format: str = "auto",
        content_format: str | None = None,
        include_alternates: bool = False,
    ) -> MediaNavigationContentResponse:
        selected_format = content_format or format
        include_alternates_param: bool | str = (
            str(include_alternates).lower() if content_format is not None else include_alternates
        )
        response = await self._request(
            "GET",
            f"/api/v1/media/{media_id}/navigation/{node_id}/content",
            params={"format": selected_format, "include_alternates": include_alternates_param},
        )
        return MediaNavigationContentResponse.model_validate(response)

    async def generate_document_insights(
        self,
        media_id: int,
        request_data: DocumentInsightsRequest | None = None,
    ) -> DocumentInsightsResponse:
        payload = (request_data or DocumentInsightsRequest()).model_dump(exclude_none=True, mode="json")
        response = await self._request(
            "POST",
            f"/api/v1/media/{media_id}/insights",
            json_data=payload,
        )
        return DocumentInsightsResponse.model_validate(response)

    async def list_watchlist_sources(
        self,
        *,
        q: str | None = None,
        tags: list[str] | None = None,
        source_type: str | None = None,
        active: bool | None = None,
        page: int = 1,
        size: int = 50,
        offset: int | None = None,
        limit: int | None = None,
    ) -> SourceListResponse:
        params: dict[str, Any] = {
            "q": q,
            "tags": tags,
            "source_type": source_type,
            "active": active,
        }
        if offset is not None or limit is not None:
            params["offset"] = offset
            params["limit"] = limit
        else:
            params["page"] = page
            params["size"] = size
        response = await self._request(
            "GET",
            "/api/v1/watchlists/sources",
            params={key: value for key, value in params.items() if value is not None},
        )
        if isinstance(response, list):
            response = {"items": response, "total": len(response), "page": page, "size": size}
        return SourceListResponse.model_validate(response)

    async def get_watchlist_source(self, source_id: int) -> SourceResponse:
        response = await self._request("GET", f"/api/v1/watchlists/sources/{source_id}")
        return SourceResponse.model_validate(response)

    async def create_watchlist_source(self, request_data: SourceCreateRequest) -> SourceResponse:
        response = await self._request(
            "POST",
            "/api/v1/watchlists/sources",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return SourceResponse.model_validate(response)

    async def update_watchlist_source(
        self,
        source_id: int,
        request_data: SourceUpdateRequest,
    ) -> SourceResponse:
        response = await self._request(
            "PATCH",
            f"/api/v1/watchlists/sources/{source_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return SourceResponse.model_validate(response)

    async def delete_watchlist_source(self, source_id: int) -> SourceDeleteResponse:
        response = await self._request("DELETE", f"/api/v1/watchlists/sources/{source_id}")
        if not response:
            response = {"success": True, "source_id": source_id}
        elif isinstance(response, dict) and "source_id" not in response:
            response = {**response, "source_id": source_id}
        return SourceDeleteResponse.model_validate(response)

    async def export_watchlist_sources(
        self,
        *,
        tag: list[str] | None = None,
        group: list[int] | None = None,
        source_type: str | None = None,
        target_user_id: int | None = None,
    ) -> bytes:
        response = await self._request_bytes(
            "GET",
            "/api/v1/watchlists/sources/export",
            params={
                key: value
                for key, value in {
                    "tag": tag,
                    "group": group,
                    "type": source_type,
                    "target_user_id": target_user_id,
                }.items()
                if value is not None
            },
        )
        return response

    async def import_watchlist_sources(
        self,
        content: bytes,
        *,
        filename: str = "sources.opml",
        active: bool = True,
        tags: list[str] | None = None,
        group_id: int | None = None,
    ) -> WatchlistSourceImportResponse:
        response = await self._request(
            "POST",
            "/api/v1/watchlists/sources/import",
            data={key: value for key, value in {"active": active, "tags": tags, "group_id": group_id}.items() if value is not None},
            files=[("file", (filename, content, "application/xml"))],
        )
        return WatchlistSourceImportResponse.model_validate(response)

    async def bulk_create_watchlist_sources(
        self,
        request_data: WatchlistSourceBulkCreateRequest,
    ) -> WatchlistSourceBulkCreateResponse:
        response = await self._request(
            "POST",
            "/api/v1/watchlists/sources/bulk",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return WatchlistSourceBulkCreateResponse.model_validate(response)

    async def check_watchlist_sources_now(
        self,
        request_data: WatchlistSourceCheckNowRequest,
    ) -> WatchlistSourceCheckNowResponse:
        response = await self._request(
            "POST",
            "/api/v1/watchlists/sources/check-now",
            json_data=request_data.model_dump(mode="json"),
        )
        return WatchlistSourceCheckNowResponse.model_validate(response)

    async def get_watchlist_source_seen_stats(
        self,
        source_id: int,
        *,
        target_user_id: int | None = None,
        keys_limit: int = 0,
    ) -> WatchlistSourceSeenStatsResponse:
        response = await self._request(
            "GET",
            f"/api/v1/watchlists/sources/{source_id}/seen",
            params={
                key: value
                for key, value in {"target_user_id": target_user_id, "keys_limit": keys_limit}.items()
                if value is not None
            },
        )
        return WatchlistSourceSeenStatsResponse.model_validate(response)

    async def clear_watchlist_source_seen_state(
        self,
        source_id: int,
        *,
        target_user_id: int | None = None,
        clear_backoff: bool = True,
    ) -> WatchlistSourceSeenResetResponse:
        response = await self._request(
            "DELETE",
            f"/api/v1/watchlists/sources/{source_id}/seen",
            params={
                key: value
                for key, value in {"target_user_id": target_user_id, "clear_backoff": clear_backoff}.items()
                if value is not None
            },
        )
        return WatchlistSourceSeenResetResponse.model_validate(response)

    async def test_watchlist_source_draft(
        self,
        request_data: WatchlistSourceTestRequest,
        *,
        limit: int = 20,
    ) -> WatchlistPreviewResponse:
        response = await self._request(
            "POST",
            "/api/v1/watchlists/sources/test",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
            params={"limit": limit},
        )
        return WatchlistPreviewResponse.model_validate(response)

    async def test_watchlist_source(
        self,
        source_id: int,
        *,
        limit: int = 20,
    ) -> WatchlistPreviewResponse:
        response = await self._request(
            "POST",
            f"/api/v1/watchlists/sources/{source_id}/test",
            params={"limit": limit},
        )
        return WatchlistPreviewResponse.model_validate(response)

    async def restore_watchlist_source(self, source_id: int) -> SourceResponse:
        response = await self._request("POST", f"/api/v1/watchlists/sources/{source_id}/restore")
        return SourceResponse.model_validate(response)

    async def list_watchlist_tags(
        self,
        *,
        q: str | None = None,
        page: int = 1,
        size: int = 50,
    ) -> WatchlistTagListResponse:
        response = await self._request(
            "GET",
            "/api/v1/watchlists/tags",
            params={key: value for key, value in {"q": q, "page": page, "size": size}.items() if value is not None},
        )
        return WatchlistTagListResponse.model_validate(response)

    async def list_watchlist_groups(
        self,
        *,
        q: str | None = None,
        page: int = 1,
        size: int = 50,
    ) -> WatchlistGroupListResponse:
        response = await self._request(
            "GET",
            "/api/v1/watchlists/groups",
            params={key: value for key, value in {"q": q, "page": page, "size": size}.items() if value is not None},
        )
        return WatchlistGroupListResponse.model_validate(response)

    async def create_watchlist_group(self, request_data: WatchlistGroupCreateRequest) -> WatchlistGroupResponse:
        response = await self._request(
            "POST",
            "/api/v1/watchlists/groups",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return WatchlistGroupResponse.model_validate(response)

    async def update_watchlist_group(
        self,
        group_id: int,
        request_data: WatchlistGroupUpdateRequest,
    ) -> WatchlistGroupResponse:
        response = await self._request(
            "PATCH",
            f"/api/v1/watchlists/groups/{group_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return WatchlistGroupResponse.model_validate(response)

    async def delete_watchlist_group(self, group_id: int) -> dict[str, Any]:
        response = await self._request("DELETE", f"/api/v1/watchlists/groups/{group_id}")
        return dict(response or {"success": True})

    async def get_watchlist_settings(self) -> dict[str, Any]:
        response = await self._request("GET", "/api/v1/watchlists/settings")
        return dict(response or {})

    async def record_watchlist_onboarding_telemetry(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = await self._request(
            "POST",
            "/api/v1/watchlists/telemetry/onboarding",
            json_data=dict(payload),
        )
        return dict(response or {})

    async def get_watchlist_onboarding_telemetry_summary(
        self,
        *,
        since: str | None = None,
        until: str | None = None,
    ) -> dict[str, Any]:
        response = await self._request(
            "GET",
            "/api/v1/watchlists/telemetry/onboarding/summary",
            params={key: value for key, value in {"since": since, "until": until}.items() if value is not None},
        )
        return dict(response or {})

    async def record_watchlist_ia_experiment_telemetry(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = await self._request(
            "POST",
            "/api/v1/watchlists/telemetry/ia-experiment",
            json_data=dict(payload),
        )
        return dict(response or {})

    async def get_watchlist_ia_experiment_telemetry_summary(
        self,
        *,
        since: str | None = None,
        until: str | None = None,
    ) -> dict[str, Any]:
        response = await self._request(
            "GET",
            "/api/v1/watchlists/telemetry/ia-experiment/summary",
            params={key: value for key, value in {"since": since, "until": until}.items() if value is not None},
        )
        return dict(response or {})

    async def get_watchlist_rc_telemetry_summary(
        self,
        *,
        since: str | None = None,
        until: str | None = None,
    ) -> dict[str, Any]:
        response = await self._request(
            "GET",
            "/api/v1/watchlists/telemetry/rc-summary",
            params={key: value for key, value in {"since": since, "until": until}.items() if value is not None},
        )
        return dict(response or {})

    async def create_watchlist_job(self, request_data: WatchlistJobCreateRequest) -> WatchlistJobResponse:
        response = await self._request(
            "POST",
            "/api/v1/watchlists/jobs",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return WatchlistJobResponse.model_validate(response)

    async def preview_watchlist_job(
        self,
        job_id: int,
        *,
        limit: int = 20,
        per_source: int = 10,
        include_content: bool = False,
    ) -> WatchlistPreviewResponse:
        response = await self._request(
            "POST",
            f"/api/v1/watchlists/jobs/{job_id}/preview",
            params={"limit": limit, "per_source": per_source, "include_content": include_content},
        )
        return WatchlistPreviewResponse.model_validate(response)

    async def list_watchlist_jobs(
        self,
        *,
        q: str | None = None,
        target_user_id: int | None = None,
        page: int = 1,
        size: int = 50,
    ) -> WatchlistJobListResponse:
        response = await self._request(
            "GET",
            "/api/v1/watchlists/jobs",
            params={
                key: value
                for key, value in {
                    "q": q,
                    "target_user_id": target_user_id,
                    "page": page,
                    "size": size,
                }.items()
                if value is not None
            },
        )
        return WatchlistJobListResponse.model_validate(response)

    async def get_watchlist_job(
        self,
        job_id: int,
        *,
        include_internal: bool = False,
        target_user_id: int | None = None,
    ) -> WatchlistJobResponse:
        response = await self._request(
            "GET",
            f"/api/v1/watchlists/jobs/{job_id}",
            params={
                key: value
                for key, value in {
                    "include_internal": include_internal,
                    "target_user_id": target_user_id,
                }.items()
                if value is not None
            },
        )
        return WatchlistJobResponse.model_validate(response)

    async def update_watchlist_job(
        self,
        job_id: int,
        request_data: WatchlistJobUpdateRequest,
    ) -> WatchlistJobResponse:
        response = await self._request(
            "PATCH",
            f"/api/v1/watchlists/jobs/{job_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return WatchlistJobResponse.model_validate(response)

    async def delete_watchlist_job(self, job_id: int) -> WatchlistJobDeleteResponse:
        response = await self._request("DELETE", f"/api/v1/watchlists/jobs/{job_id}")
        if isinstance(response, dict) and "job_id" not in response:
            response = {**response, "job_id": job_id}
        return WatchlistJobDeleteResponse.model_validate(response or {"success": True, "job_id": job_id})

    async def restore_watchlist_job(self, job_id: int) -> WatchlistJobResponse:
        response = await self._request("POST", f"/api/v1/watchlists/jobs/{job_id}/restore")
        return WatchlistJobResponse.model_validate(response)

    async def replace_watchlist_job_filters(
        self,
        job_id: int,
        request_data: WatchlistFiltersPayload | dict[str, Any],
    ) -> WatchlistFiltersPayload:
        payload = WatchlistFiltersPayload.model_validate(request_data)
        response = await self._request(
            "PATCH",
            f"/api/v1/watchlists/jobs/{job_id}/filters",
            json_data=payload.model_dump(mode="json"),
        )
        return WatchlistFiltersPayload.model_validate(response)

    async def append_watchlist_job_filters(
        self,
        job_id: int,
        request_data: WatchlistFiltersPayload | dict[str, Any],
    ) -> WatchlistFiltersPayload:
        payload = WatchlistFiltersPayload.model_validate(request_data)
        response = await self._request(
            "POST",
            f"/api/v1/watchlists/jobs/{job_id}/filters:add",
            json_data=payload.model_dump(mode="json"),
        )
        return WatchlistFiltersPayload.model_validate(response)

    async def trigger_watchlist_run(self, job_id: int) -> WatchlistRunResponse:
        response = await self._request("POST", f"/api/v1/watchlists/jobs/{job_id}/run")
        return WatchlistRunResponse.model_validate(response)

    async def list_watchlist_runs(
        self,
        *,
        job_id: int | None = None,
        q: str | None = None,
        page: int = 1,
        size: int = 50,
        target_user_id: int | None = None,
    ) -> WatchlistRunListResponse:
        path = f"/api/v1/watchlists/jobs/{job_id}/runs" if job_id is not None else "/api/v1/watchlists/runs"
        response = await self._request(
            "GET",
            path,
            params={
                key: value
                for key, value in {
                    "q": q,
                    "page": page,
                    "size": size,
                    "target_user_id": target_user_id,
                }.items()
                if value is not None
            },
        )
        if isinstance(response, list):
            response = {"items": response, "total": len(response), "has_more": False}
        return WatchlistRunListResponse.model_validate(response)

    async def list_watchlist_runs_for_job(
        self,
        job_id: int,
        *,
        target_user_id: int | None = None,
        page: int = 1,
        size: int = 50,
    ) -> WatchlistRunListResponse:
        response = await self._request(
            "GET",
            f"/api/v1/watchlists/jobs/{job_id}/runs",
            params={
                key: value
                for key, value in {
                    "target_user_id": target_user_id,
                    "page": page,
                    "size": size,
                }.items()
                if value is not None
            },
        )
        return WatchlistRunListResponse.model_validate(response)

    async def list_watchlist_runs_global(
        self,
        *,
        q: str | None = None,
        target_user_id: int | None = None,
        page: int = 1,
        size: int = 50,
    ) -> WatchlistRunListResponse:
        response = await self._request(
            "GET",
            "/api/v1/watchlists/runs",
            params={
                key: value
                for key, value in {
                    "q": q,
                    "target_user_id": target_user_id,
                    "page": page,
                    "size": size,
                }.items()
                if value is not None
            },
        )
        return WatchlistRunListResponse.model_validate(response)

    async def get_watchlist_run(self, run_id: int) -> WatchlistRunResponse:
        response = await self._request("GET", f"/api/v1/watchlists/runs/{run_id}")
        return WatchlistRunResponse.model_validate(response)

    async def get_watchlist_run_details(
        self,
        run_id: int,
        *,
        include_tallies: bool = False,
        filtered_sample_max: int = 5,
    ) -> WatchlistRunDetailResponse:
        response = await self._request(
            "GET",
            f"/api/v1/watchlists/runs/{run_id}/details",
            params={
                "include_tallies": str(include_tallies).lower(),
                "filtered_sample_max": filtered_sample_max,
            },
        )
        return WatchlistRunDetailResponse.model_validate(response)

    async def export_watchlist_runs_csv(
        self,
        *,
        scope: str = "global",
        job_id: int | None = None,
        target_user_id: int | None = None,
        q: str | None = None,
        page: int = 1,
        size: int = 200,
        include_tallies: bool = False,
        tallies_mode: str | None = None,
    ) -> bytes:
        return await self._request_bytes(
            "GET",
            "/api/v1/watchlists/runs/export.csv",
            params={
                key: value
                for key, value in {
                    "scope": scope,
                    "job_id": job_id,
                    "target_user_id": target_user_id,
                    "q": q,
                    "page": page if page != 1 else None,
                    "size": size if size != 200 else None,
                    "include_tallies": include_tallies,
                    "tallies_mode": tallies_mode,
                }.items()
                if value is not None
            },
        )

    async def cancel_watchlist_run(
        self,
        run_id: int,
        *,
        target_user_id: int | None = None,
    ) -> WatchlistRunCancelResponse:
        response = await self._request(
            "POST",
            f"/api/v1/watchlists/runs/{run_id}/cancel",
            params={key: value for key, value in {"target_user_id": target_user_id}.items() if value is not None},
        )
        return WatchlistRunCancelResponse.model_validate(response)

    async def get_watchlist_run_audio(
        self,
        run_id: int,
        *,
        target_user_id: int | None = None,
    ) -> dict[str, Any]:
        response = await self._request(
            "GET",
            f"/api/v1/watchlists/runs/{run_id}/audio",
            params={key: value for key, value in {"target_user_id": target_user_id}.items() if value is not None},
        )
        return dict(response or {})

    async def export_watchlist_run_tallies_csv(
        self,
        run_id: int,
        *,
        target_user_id: int | None = None,
    ) -> bytes:
        return await self._request_bytes(
            "GET",
            f"/api/v1/watchlists/runs/{run_id}/tallies.csv",
            params={key: value for key, value in {"target_user_id": target_user_id}.items() if value is not None},
        )

    async def get_watchlist_item_smart_counts(
        self,
        *,
        run_id: int | None = None,
        job_id: int | None = None,
        source_id: int | None = None,
        status: str | None = None,
        target_user_id: int | None = None,
        q: str | None = None,
        since: str | None = None,
        until: str | None = None,
        queue_run_id: int | None = None,
    ) -> WatchlistScrapedItemSmartCountsResponse:
        response = await self._request(
            "GET",
            "/api/v1/watchlists/items/smart-counts",
            params={
                key: value
                for key, value in {
                    "run_id": run_id,
                    "job_id": job_id,
                    "source_id": source_id,
                    "status": status,
                    "target_user_id": target_user_id,
                    "q": q,
                    "since": since,
                    "until": until,
                    "queue_run_id": queue_run_id,
                }.items()
                if value is not None
            },
        )
        return WatchlistScrapedItemSmartCountsResponse.model_validate(response)

    async def list_watchlist_items(
        self,
        *,
        run_id: int | None = None,
        job_id: int | None = None,
        source_id: int | None = None,
        status: str | None = None,
        reviewed: bool | None = None,
        queued_for_briefing: bool | None = None,
        target_user_id: int | None = None,
        q: str | None = None,
        since: str | None = None,
        until: str | None = None,
        page: int = 1,
        size: int = 50,
    ) -> WatchlistScrapedItemListResponse:
        response = await self._request(
            "GET",
            "/api/v1/watchlists/items",
            params={
                key: value
                for key, value in {
                    "run_id": run_id,
                    "job_id": job_id,
                    "source_id": source_id,
                    "status": status,
                    "reviewed": reviewed,
                    "queued_for_briefing": queued_for_briefing,
                    "target_user_id": target_user_id,
                    "q": q,
                    "since": since,
                    "until": until,
                    "page": page,
                    "size": size,
                }.items()
                if value is not None
            },
        )
        return WatchlistScrapedItemListResponse.model_validate(response)

    async def get_watchlist_item(
        self,
        item_id: int,
        *,
        target_user_id: int | None = None,
    ) -> WatchlistScrapedItemResponse:
        response = await self._request(
            "GET",
            f"/api/v1/watchlists/items/{item_id}",
            params={key: value for key, value in {"target_user_id": target_user_id}.items() if value is not None},
        )
        return WatchlistScrapedItemResponse.model_validate(response)

    async def update_watchlist_item(
        self,
        item_id: int,
        request_data: WatchlistScrapedItemUpdateRequest,
    ) -> WatchlistScrapedItemResponse:
        response = await self._request(
            "PATCH",
            f"/api/v1/watchlists/items/{item_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return WatchlistScrapedItemResponse.model_validate(response)

    async def create_watchlist_output(
        self,
        request_data: WatchlistOutputCreateRequest,
    ) -> WatchlistOutputResponse:
        response = await self._request(
            "POST",
            "/api/v1/watchlists/outputs",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return WatchlistOutputResponse.model_validate(response)

    async def list_watchlist_outputs(
        self,
        *,
        run_id: int | None = None,
        job_id: int | None = None,
        page: int = 1,
        size: int = 50,
    ) -> WatchlistOutputListResponse:
        response = await self._request(
            "GET",
            "/api/v1/watchlists/outputs",
            params={
                key: value
                for key, value in {"run_id": run_id, "job_id": job_id, "page": page, "size": size}.items()
                if value is not None
            },
        )
        return WatchlistOutputListResponse.model_validate(response)

    async def get_watchlist_output(self, output_id: int) -> WatchlistOutputResponse:
        response = await self._request("GET", f"/api/v1/watchlists/outputs/{output_id}")
        return WatchlistOutputResponse.model_validate(response)

    async def download_watchlist_output(self, output_id: int) -> bytes:
        return await self._request_bytes("GET", f"/api/v1/watchlists/outputs/{output_id}/download")

    async def list_watchlist_templates(self) -> WatchlistTemplateListResponse:
        response = await self._request("GET", "/api/v1/watchlists/templates")
        return WatchlistTemplateListResponse.model_validate(response)

    async def create_watchlist_template(
        self,
        request_data: WatchlistTemplateCreateRequest,
    ) -> WatchlistTemplateDetailResponse:
        response = await self._request(
            "POST",
            "/api/v1/watchlists/templates",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return WatchlistTemplateDetailResponse.model_validate(response)

    async def validate_watchlist_template(
        self,
        request_data: WatchlistTemplateValidationRequest,
    ) -> WatchlistTemplateValidationResponse:
        response = await self._request(
            "POST",
            "/api/v1/watchlists/templates/validate",
            json_data=request_data.model_dump(mode="json"),
        )
        return WatchlistTemplateValidationResponse.model_validate(response)

    async def preview_watchlist_template(
        self,
        request_data: WatchlistTemplatePreviewRequest,
    ) -> WatchlistTemplatePreviewResponse:
        response = await self._request(
            "POST",
            "/api/v1/watchlists/templates/preview",
            json_data=request_data.model_dump(mode="json"),
        )
        return WatchlistTemplatePreviewResponse.model_validate(response)

    async def compose_watchlist_template_section(
        self,
        request_data: WatchlistTemplateComposerSectionRequest,
    ) -> WatchlistTemplateComposerSectionResponse:
        response = await self._request(
            "POST",
            "/api/v1/watchlists/templates/compose/section",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return WatchlistTemplateComposerSectionResponse.model_validate(response)

    async def check_watchlist_template_flow(
        self,
        request_data: WatchlistTemplateComposerFlowCheckRequest,
    ) -> WatchlistTemplateComposerFlowCheckResponse:
        response = await self._request(
            "POST",
            "/api/v1/watchlists/templates/compose/flow-check",
            json_data=request_data.model_dump(mode="json"),
        )
        return WatchlistTemplateComposerFlowCheckResponse.model_validate(response)

    async def list_watchlist_template_versions(self, template_name: str) -> WatchlistTemplateVersionsResponse:
        response = await self._request("GET", f"/api/v1/watchlists/templates/{template_name}/versions")
        return WatchlistTemplateVersionsResponse.model_validate(response)

    async def get_watchlist_template(
        self,
        template_name: str,
        *,
        version: int | None = None,
    ) -> WatchlistTemplateDetailResponse:
        response = await self._request(
            "GET",
            f"/api/v1/watchlists/templates/{template_name}",
            params={key: value for key, value in {"version": version}.items() if value is not None},
        )
        return WatchlistTemplateDetailResponse.model_validate(response)

    async def delete_watchlist_template(self, template_name: str) -> dict[str, Any]:
        response = await self._request("DELETE", f"/api/v1/watchlists/templates/{template_name}")
        return dict(response or {"deleted": True})

    async def list_watchlist_clusters(self, watchlist_id: int) -> dict[str, Any]:
        response = await self._request("GET", f"/api/v1/watchlists/{watchlist_id}/clusters")
        return dict(response or {})

    async def add_watchlist_cluster(self, watchlist_id: int, cluster_id: int) -> dict[str, Any]:
        response = await self._request(
            "POST",
            f"/api/v1/watchlists/{watchlist_id}/clusters",
            json_data={"cluster_id": cluster_id},
        )
        return dict(response or {})

    async def remove_watchlist_cluster(self, watchlist_id: int, cluster_id: int) -> dict[str, Any]:
        response = await self._request("DELETE", f"/api/v1/watchlists/{watchlist_id}/clusters/{cluster_id}")
        return dict(response or {})

    async def list_watchlist_alert_rules(self, *, job_id: int | None = None) -> WatchlistAlertRuleListResponse:
        response = await self._request(
            "GET",
            "/api/v1/watchlists/alert-rules",
            params={key: value for key, value in {"job_id": job_id}.items() if value is not None},
        )
        if isinstance(response, list):
            response = {"items": response}
        return WatchlistAlertRuleListResponse.model_validate(response)

    async def create_watchlist_alert_rule(
        self,
        request_data: WatchlistAlertRuleCreateRequest,
    ) -> WatchlistAlertRuleResponse:
        response = await self._request(
            "POST",
            "/api/v1/watchlists/alert-rules",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return WatchlistAlertRuleResponse.model_validate(response)

    async def update_watchlist_alert_rule(
        self,
        rule_id: int,
        request_data: WatchlistAlertRuleUpdateRequest,
    ) -> WatchlistAlertRuleResponse:
        response = await self._request(
            "PATCH",
            f"/api/v1/watchlists/alert-rules/{rule_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return WatchlistAlertRuleResponse.model_validate(response)

    async def delete_watchlist_alert_rule(self, rule_id: int) -> WatchlistAlertRuleDeleteResponse:
        response = await self._request("DELETE", f"/api/v1/watchlists/alert-rules/{rule_id}")
        if not response:
            response = {"deleted": True, "rule_id": rule_id}
        elif isinstance(response, dict) and "rule_id" not in response:
            response = {**response, "rule_id": rule_id}
        return WatchlistAlertRuleDeleteResponse.model_validate(response)

    async def list_notifications(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        include_archived: bool = False,
        only_snoozed: bool = False,
    ) -> NotificationsListResponse:
        response = await self._request(
            "GET",
            "/api/v1/notifications",
            params={
                "limit": limit,
                "offset": offset,
                "include_archived": include_archived,
                "only_snoozed": only_snoozed,
            },
        )
        return NotificationsListResponse.model_validate(response)

    async def get_notifications_unread_count(self) -> NotificationsUnreadCountResponse:
        response = await self._request("GET", "/api/v1/notifications/unread-count")
        return NotificationsUnreadCountResponse.model_validate(response)

    async def mark_notifications_read(self, ids: list[int]) -> NotificationsMarkReadResponse:
        request = NotificationsMarkReadRequest(ids=ids)
        response = await self._request(
            "POST",
            "/api/v1/notifications/mark-read",
            json_data=request.model_dump(mode="json"),
        )
        return NotificationsMarkReadResponse.model_validate(response)

    async def dismiss_notification(self, notification_id: int) -> NotificationDismissResponse:
        response = await self._request("POST", f"/api/v1/notifications/{notification_id}/dismiss")
        return NotificationDismissResponse.model_validate(response)

    async def snooze_notification(
        self,
        notification_id: int,
        request_data: NotificationSnoozeRequest,
    ) -> NotificationSnoozeResponse:
        response = await self._request(
            "POST",
            f"/api/v1/notifications/{notification_id}/snooze",
            json_data=request_data.model_dump(mode="json"),
        )
        return NotificationSnoozeResponse.model_validate(response)

    async def cancel_notification_snooze(self, notification_id: int) -> NotificationCancelSnoozeResponse:
        response = await self._request("DELETE", f"/api/v1/notifications/{notification_id}/snooze")
        return NotificationCancelSnoozeResponse.model_validate(response)

    async def get_notification_preferences(self) -> NotificationPreferencesResponse:
        response = await self._request("GET", "/api/v1/notifications/preferences")
        return NotificationPreferencesResponse.model_validate(response)

    async def update_notification_preferences(
        self,
        request_data: NotificationPreferencesUpdateRequest,
    ) -> NotificationPreferencesResponse:
        response = await self._request(
            "PATCH",
            "/api/v1/notifications/preferences",
            json_data=request_data.model_dump(exclude_unset=True, mode="json"),
        )
        return NotificationPreferencesResponse.model_validate(response)

    async def stream_notification_events(
        self,
        *,
        after: int = 0,
        last_event_id: str | None = None,
    ) -> AsyncGenerator[NotificationStreamEvent, None]:
        headers = {"Last-Event-ID": last_event_id} if last_event_id else None
        async for event in self._sse_request(
            "GET",
            "/api/v1/notifications/stream",
            params={"after": after},
            headers=headers,
        ):
            yield NotificationStreamEvent.model_validate(event)

    async def list_server_notifications(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        include_archived: bool = False,
        only_snoozed: bool = False,
    ) -> NotificationsListResponse:
        return await self.list_notifications(
            limit=limit,
            offset=offset,
            include_archived=include_archived,
            only_snoozed=only_snoozed,
        )

    async def get_server_notifications_unread_count(self) -> NotificationsUnreadCountResponse:
        return await self.get_notifications_unread_count()

    async def mark_server_notifications_read(self, ids: list[int]) -> NotificationsMarkReadResponse:
        return await self.mark_notifications_read(ids)

    async def dismiss_server_notification(self, notification_id: int) -> NotificationDismissResponse:
        return await self.dismiss_notification(notification_id)

    async def snooze_server_notification(
        self,
        notification_id: int,
        request_data: NotificationSnoozeRequest,
    ) -> NotificationSnoozeResponse:
        return await self.snooze_notification(notification_id, request_data)

    async def cancel_server_notification_snooze(self, notification_id: int) -> NotificationCancelSnoozeResponse:
        return await self.cancel_notification_snooze(notification_id)

    async def get_server_notification_preferences(self) -> NotificationPreferencesResponse:
        return await self.get_notification_preferences()

    async def update_server_notification_preferences(
        self,
        request_data: NotificationPreferencesUpdateRequest,
    ) -> NotificationPreferencesResponse:
        return await self.update_notification_preferences(request_data)

    async def stream_server_notifications(
        self,
        *,
        after: int = 0,
    ) -> AsyncGenerator[ServerNotificationStreamEvent, None]:
        async for event in self._stream_sse_request(
            "/api/v1/notifications/stream",
            params={"after": after},
            event_model=ServerNotificationStreamEvent,
        ):
            yield event

    async def create_reminder_task(self, request_data: ReminderTaskCreateRequest) -> ReminderTaskResponse:
        response = await self._request(
            "POST",
            "/api/v1/tasks",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return ReminderTaskResponse.model_validate(response)

    async def list_reminder_tasks(self) -> ReminderTaskListResponse:
        response = await self._request("GET", "/api/v1/tasks")
        return ReminderTaskListResponse.model_validate(response)

    async def get_reminder_task(self, task_id: str) -> ReminderTaskResponse:
        response = await self._request("GET", f"/api/v1/tasks/{task_id}")
        return ReminderTaskResponse.model_validate(response)

    async def update_reminder_task(
        self,
        task_id: str,
        request_data: ReminderTaskUpdateRequest,
    ) -> ReminderTaskResponse:
        response = await self._request(
            "PATCH",
            f"/api/v1/tasks/{task_id}",
            json_data=request_data.model_dump(exclude_unset=True, mode="json"),
        )
        return ReminderTaskResponse.model_validate(response)

    async def delete_reminder_task(self, task_id: str) -> ReminderTaskDeleteResponse:
        response = await self._request("DELETE", f"/api/v1/tasks/{task_id}")
        return ReminderTaskDeleteResponse.model_validate(response)

    async def list_output_templates(
        self,
        *,
        q: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> OutputTemplateList:
        response = await self._request(
            "GET",
            "/api/v1/outputs/templates",
            params={key: value for key, value in {"q": q, "limit": limit, "offset": offset}.items() if value is not None},
        )
        return OutputTemplateList.model_validate(response)

    async def create_output_template(self, request_data: OutputTemplateCreate) -> OutputTemplate:
        response = await self._request(
            "POST",
            "/api/v1/outputs/templates",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return OutputTemplate.model_validate(response)

    async def get_output_template(self, template_id: int) -> OutputTemplate:
        response = await self._request("GET", f"/api/v1/outputs/templates/{template_id}")
        return OutputTemplate.model_validate(response)

    async def update_output_template(
        self,
        template_id: int,
        request_data: OutputTemplateUpdate,
    ) -> OutputTemplate:
        response = await self._request(
            "PATCH",
            f"/api/v1/outputs/templates/{template_id}",
            json_data=request_data.model_dump(exclude_unset=True, mode="json"),
        )
        return OutputTemplate.model_validate(response)

    async def delete_output_template(self, template_id: int) -> Dict[str, Any]:
        return await self._request("DELETE", f"/api/v1/outputs/templates/{template_id}")

    async def preview_output_template(
        self,
        template_id: int,
        request_data: TemplatePreviewRequest,
    ) -> TemplatePreviewResponse:
        response = await self._request(
            "POST",
            f"/api/v1/outputs/templates/{template_id}/preview",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return TemplatePreviewResponse.model_validate(response)

    async def list_outputs(
        self,
        *,
        page: int = 1,
        size: int = 50,
        job_id: int | None = None,
        run_id: int | None = None,
        type: str | None = None,
        workspace_tag: str | None = None,
        include_deleted: bool | None = None,
    ) -> OutputListResponse:
        params = {
            "page": page,
            "size": size,
            "job_id": job_id,
            "run_id": run_id,
            "type": type,
            "workspace_tag": workspace_tag,
            "include_deleted": include_deleted,
        }
        response = await self._request(
            "GET",
            "/api/v1/outputs",
            params={key: value for key, value in params.items() if value is not None},
        )
        return OutputListResponse.model_validate(response)

    async def list_deleted_outputs(self, *, page: int = 1, size: int = 50) -> OutputListResponse:
        response = await self._request(
            "GET",
            "/api/v1/outputs/deleted",
            params={"page": page, "size": size},
        )
        return OutputListResponse.model_validate(response)

    async def create_output(self, request_data: OutputCreateRequest) -> OutputArtifact:
        response = await self._request(
            "POST",
            "/api/v1/outputs",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return OutputArtifact.model_validate(response)

    async def get_output(self, output_id: int) -> OutputArtifact:
        response = await self._request("GET", f"/api/v1/outputs/{output_id}")
        return OutputArtifact.model_validate(response)

    async def update_output(
        self,
        output_id: int,
        request_data: OutputUpdateRequest,
    ) -> OutputArtifact:
        response = await self._request(
            "PATCH",
            f"/api/v1/outputs/{output_id}",
            json_data=request_data.model_dump(exclude_unset=True, mode="json"),
        )
        return OutputArtifact.model_validate(response)

    async def delete_output(
        self,
        output_id: int,
        *,
        hard: bool = False,
        delete_file: bool = False,
    ) -> OutputDeleteResponse:
        response = await self._request(
            "DELETE",
            f"/api/v1/outputs/{output_id}",
            params={"hard": hard, "delete_file": delete_file},
        )
        return OutputDeleteResponse.model_validate(response)

    async def purge_outputs(
        self,
        *,
        delete_files: bool = False,
        soft_deleted_grace_days: int = 30,
        include_retention: bool = True,
    ) -> OutputsPurgeResponse:
        request = OutputsPurgeRequest(
            delete_files=delete_files,
            soft_deleted_grace_days=soft_deleted_grace_days,
            include_retention=include_retention,
        )
        response = await self._request(
            "POST",
            "/api/v1/outputs/purge",
            json_data=request.model_dump(mode="json"),
        )
        return OutputsPurgeResponse.model_validate(response)

    async def create_research_run(self, request_data: ResearchRunCreateRequest) -> ResearchRunResponse:
        response = await self._request(
            "POST",
            "/api/v1/research/runs",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return ResearchRunResponse.model_validate(response)

    async def list_research_runs(self, *, limit: int = 25) -> list[ResearchRunListItemResponse]:
        response = await self._request(
            "GET",
            "/api/v1/research/runs",
            params={"limit": limit},
        )
        return [ResearchRunListItemResponse.model_validate(item) for item in response]

    async def get_research_run(self, session_id: str) -> ResearchRunResponse:
        response = await self._request("GET", f"/api/v1/research/runs/{session_id}")
        return ResearchRunResponse.model_validate(response)

    async def stream_research_run_events(
        self,
        session_id: str,
        *,
        after_id: int = 0,
    ) -> AsyncGenerator[ResearchRunStreamEvent, None]:
        async for event in self._sse_request(
            "GET",
            f"/api/v1/research/runs/{session_id}/events/stream",
            params={"after_id": after_id},
        ):
            yield ResearchRunStreamEvent.model_validate(event)

    async def pause_research_run(self, session_id: str) -> ResearchRunResponse:
        response = await self._request("POST", f"/api/v1/research/runs/{session_id}/pause")
        return ResearchRunResponse.model_validate(response)

    async def resume_research_run(self, session_id: str) -> ResearchRunResponse:
        response = await self._request("POST", f"/api/v1/research/runs/{session_id}/resume")
        return ResearchRunResponse.model_validate(response)

    async def cancel_research_run(self, session_id: str) -> ResearchRunResponse:
        response = await self._request("POST", f"/api/v1/research/runs/{session_id}/cancel")
        return ResearchRunResponse.model_validate(response)

    async def get_research_bundle(self, session_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/api/v1/research/runs/{session_id}/bundle")

    async def get_research_artifact(
        self,
        session_id: str,
        artifact_name: str,
    ) -> ResearchArtifactResponse:
        response = await self._request(
            "GET",
            f"/api/v1/research/runs/{session_id}/artifacts/{artifact_name}",
        )
        return ResearchArtifactResponse.model_validate(response)

    async def patch_and_approve_research_checkpoint(
        self,
        session_id: str,
        checkpoint_id: str,
        request_data: ResearchCheckpointPatchApproveRequest,
    ) -> ResearchRunResponse:
        response = await self._request(
            "POST",
            f"/api/v1/research/runs/{session_id}/checkpoints/{checkpoint_id}/patch-and-approve",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return ResearchRunResponse.model_validate(response)

    async def research_websearch(
        self,
        request_data: WebSearchRequest,
    ) -> WebSearchRawResponse | WebSearchAggregateResponse:
        response = await self._request(
            "POST",
            "/api/v1/research/websearch",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        if isinstance(response, dict) and "final_answer" in response:
            return WebSearchAggregateResponse.model_validate(response)
        return WebSearchRawResponse.model_validate(response)

    async def paper_search(self, request_data: PaperSearchRequest) -> PaperSearchListResponse:
        response = await self._request(
            "GET",
            f"/api/v1/paper-search/{request_data.endpoint}",
            params=dict(request_data.params or {}),
        )
        return PaperSearchListResponse.model_validate(response)

    async def paper_search_detail(self, request_data: PaperSearchDetailRequest) -> PaperSearchDetailResponse:
        response = await self._request(
            "GET",
            f"/api/v1/paper-search/{request_data.endpoint}",
            params=dict(request_data.params or {}),
        )
        return PaperSearchDetailResponse.model_validate(response)

    async def paper_search_ingest(self, request_data: PaperSearchIngestRequest) -> PaperSearchOperationResponse:
        response = await self._request(
            "POST",
            f"/api/v1/paper-search/{request_data.endpoint}",
            params=dict(request_data.params or {}),
            json_data=request_data.payload or {},
        )
        return PaperSearchOperationResponse.model_validate(response)

    async def search_arxiv_papers(
        self,
        *,
        query: str | None = None,
        author: str | None = None,
        year: str | None = None,
        page: int = 1,
        results_per_page: int = 10,
    ) -> ArxivSearchResponse:
        params = {
            "query": query,
            "author": author,
            "year": year,
            "page": page,
            "results_per_page": results_per_page,
        }
        response = await self._request(
            "GET",
            "/api/v1/paper-search/arxiv",
            params={key: value for key, value in params.items() if value is not None},
        )
        return ArxivSearchResponse.model_validate(response)

    async def get_arxiv_paper_by_id(self, *, id: str) -> ArxivPaper:
        response = await self._request(
            "GET",
            "/api/v1/paper-search/arxiv/by-id",
            params={"id": id},
        )
        return ArxivPaper.model_validate(response)

    async def search_semantic_scholar_papers(
        self,
        *,
        query: str,
        fields_of_study: list[str] | str | None = None,
        publication_types: list[str] | str | None = None,
        year_range: str | None = None,
        venue: list[str] | str | None = None,
        min_citations: int | None = None,
        page: int = 1,
        results_per_page: int = 10,
    ) -> SemanticScholarSearchResponse:
        params = {
            "query": query,
            "fields_of_study": self._csv_param(fields_of_study),
            "publication_types": self._csv_param(publication_types),
            "year_range": year_range,
            "venue": self._csv_param(venue),
            "min_citations": min_citations,
            "page": page,
            "results_per_page": results_per_page,
        }
        response = await self._request(
            "GET",
            "/api/v1/paper-search/semantic-scholar",
            params={key: value for key, value in params.items() if value is not None},
        )
        return SemanticScholarSearchResponse.model_validate(response)

    async def get_semantic_scholar_paper_by_id(self, *, paper_id: str) -> SemanticScholarPaper:
        response = await self._request(
            "GET",
            "/api/v1/paper-search/semantic-scholar/by-id",
            params={"paper_id": paper_id},
        )
        return SemanticScholarPaper.model_validate(response)

    async def search_biorxiv_papers(
        self,
        *,
        q: str | None = None,
        server: str = "biorxiv",
        from_date: str | None = None,
        to_date: str | None = None,
        category: str | None = None,
        recent_days: int | None = None,
        recent_count: int | None = None,
        page: int = 1,
        results_per_page: int = 10,
    ) -> BioRxivSearchResponse:
        params = {
            "q": q,
            "server": server,
            "from_date": from_date,
            "to_date": to_date,
            "category": category,
            "recent_days": recent_days,
            "recent_count": recent_count,
            "page": page,
            "results_per_page": results_per_page,
        }
        response = await self._request(
            "GET",
            "/api/v1/paper-search/biorxiv",
            params={key: value for key, value in params.items() if value is not None},
        )
        return BioRxivSearchResponse.model_validate(response)

    async def get_biorxiv_paper_by_doi(
        self,
        *,
        doi: str,
        server: str = "biorxiv",
    ) -> BioRxivPaper:
        response = await self._request(
            "GET",
            "/api/v1/paper-search/biorxiv/by-doi",
            params={"doi": doi, "server": server},
        )
        return BioRxivPaper.model_validate(response)

    async def search_medrxiv_papers(
        self,
        *,
        q: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        category: str | None = None,
        recent_days: int | None = None,
        recent_count: int | None = None,
        page: int = 1,
        results_per_page: int = 10,
    ) -> BioRxivSearchResponse:
        params = {
            "q": q,
            "from_date": from_date,
            "to_date": to_date,
            "category": category,
            "recent_days": recent_days,
            "recent_count": recent_count,
            "page": page,
            "results_per_page": results_per_page,
        }
        response = await self._request(
            "GET",
            "/api/v1/paper-search/medrxiv",
            params={key: value for key, value in params.items() if value is not None},
        )
        return BioRxivSearchResponse.model_validate(response)

    async def get_medrxiv_paper_by_doi(self, *, doi: str) -> BioRxivPaper:
        response = await self._request(
            "GET",
            "/api/v1/paper-search/medrxiv/by-doi",
            params={"doi": doi},
        )
        return BioRxivPaper.model_validate(response)

    async def search_pubmed_papers(
        self,
        *,
        q: str,
        from_year: int | None = None,
        to_year: int | None = None,
        free_full_text: bool = False,
        page: int = 1,
        results_per_page: int = 10,
    ) -> PubMedSearchResponse:
        params = {
            "q": q,
            "from_year": from_year,
            "to_year": to_year,
            "free_full_text": free_full_text,
            "page": page,
            "results_per_page": results_per_page,
        }
        response = await self._request(
            "GET",
            "/api/v1/paper-search/pubmed",
            params={key: value for key, value in params.items() if value is not None},
        )
        return PubMedSearchResponse.model_validate(response)

    async def get_pubmed_paper_by_id(self, *, pmid: str) -> PubMedPaper:
        response = await self._request(
            "GET",
            "/api/v1/paper-search/pubmed/by-id",
            params={"pmid": pmid},
        )
        return PubMedPaper.model_validate(response)

    async def share_workspace(
        self,
        workspace_id: str,
        request_data: ShareWorkspaceRequest,
    ) -> ShareResponse:
        response = await self._request(
            "POST",
            f"/api/v1/sharing/workspaces/{workspace_id}/share",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return ShareResponse.model_validate(response)

    async def list_workspace_shares(
        self,
        workspace_id: str,
        *,
        include_revoked: bool = False,
    ) -> ShareListResponse:
        response = await self._request(
            "GET",
            f"/api/v1/sharing/workspaces/{workspace_id}/shares",
            params={"include_revoked": include_revoked},
        )
        return ShareListResponse.model_validate(response)

    async def update_share(
        self,
        share_id: int,
        request_data: UpdateShareRequest,
    ) -> ShareResponse:
        response = await self._request(
            "PATCH",
            f"/api/v1/sharing/shares/{share_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return ShareResponse.model_validate(response)

    async def revoke_share(self, share_id: int) -> Dict[str, Any]:
        return await self._request("DELETE", f"/api/v1/sharing/shares/{share_id}")

    async def list_shared_with_me(self) -> SharedWithMeResponse:
        response = await self._request("GET", "/api/v1/sharing/shared-with-me")
        return SharedWithMeResponse.model_validate(response)

    async def get_shared_workspace(self, share_id: int) -> SharedWorkspaceResponse:
        response = await self._request("GET", f"/api/v1/sharing/shared-with-me/{share_id}/workspace")
        return SharedWorkspaceResponse.model_validate(response)

    async def clone_shared_workspace(
        self,
        share_id: int,
        request_data: CloneWorkspaceRequest,
    ) -> CloneWorkspaceResponse:
        response = await self._request(
            "POST",
            f"/api/v1/sharing/shared-with-me/{share_id}/clone",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return CloneWorkspaceResponse.model_validate(response)

    async def list_shared_workspace_sources(self, share_id: int) -> list[SharedWorkspaceSourceResponse]:
        response = await self._request("GET", f"/api/v1/sharing/shared-with-me/{share_id}/sources")
        return [SharedWorkspaceSourceResponse.model_validate(item) for item in response]

    async def get_shared_workspace_media(self, share_id: int, media_id: int) -> SharedMediaResponse:
        response = await self._request("GET", f"/api/v1/sharing/shared-with-me/{share_id}/media/{media_id}")
        return SharedMediaResponse.model_validate(response)

    async def chat_with_shared_workspace(
        self,
        share_id: int,
        request_data: SharedChatRequest,
    ) -> Dict[str, Any]:
        return await self._request(
            "POST",
            f"/api/v1/sharing/shared-with-me/{share_id}/chat",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )

    async def create_share_token(self, request_data: CreateTokenRequest) -> TokenResponse:
        response = await self._request(
            "POST",
            "/api/v1/sharing/tokens",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return TokenResponse.model_validate(response)

    async def list_share_tokens(self) -> TokenListResponse:
        response = await self._request("GET", "/api/v1/sharing/tokens")
        return TokenListResponse.model_validate(response)

    async def revoke_share_token(self, token_id: int) -> Dict[str, Any]:
        return await self._request("DELETE", f"/api/v1/sharing/tokens/{token_id}")

    async def preview_public_share(self, token: str) -> PublicSharePreview:
        response = await self._request("GET", f"/api/v1/sharing/public/{token}")
        return PublicSharePreview.model_validate(response)

    async def verify_public_share_password(
        self,
        token: str,
        request_data: VerifyPasswordRequest,
    ) -> VerifyPasswordResponse:
        response = await self._request(
            "POST",
            f"/api/v1/sharing/public/{token}/verify",
            json_data=request_data.model_dump(mode="json"),
        )
        return VerifyPasswordResponse.model_validate(response)

    async def import_public_share(self, token: str) -> PublicShareImportResponse:
        response = await self._request("POST", f"/api/v1/sharing/public/{token}/import")
        return PublicShareImportResponse.model_validate(response)

    async def save_web_clip(self, request_data: WebClipperSaveRequest) -> WebClipperSaveResponse:
        response = await self._request(
            "POST",
            "/api/v1/web-clipper/save",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return WebClipperSaveResponse.model_validate(response)

    async def get_web_clip_status(self, clip_id: str) -> WebClipperStatusResponse:
        response = await self._request("GET", f"/api/v1/web-clipper/{clip_id}")
        return WebClipperStatusResponse.model_validate(response)

    async def persist_web_clip_enrichment(
        self,
        clip_id: str,
        request_data: WebClipperEnrichmentPayload,
    ) -> WebClipperEnrichmentResponse:
        response = await self._request(
            "POST",
            f"/api/v1/web-clipper/{clip_id}/enrichments",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return WebClipperEnrichmentResponse.model_validate(response)

    @staticmethod
    def _csv_param(value: list[str] | str | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return ",".join(str(item).strip() for item in value if str(item).strip())

    async def list_chunking_templates(
        self,
        *,
        include_builtin: bool = True,
        include_custom: bool = True,
        tags: Optional[List[str]] = None,
        user_id: Optional[str] = None,
    ) -> ChunkingTemplateListResponse:
        response = await self._request(
            "GET",
            "/api/v1/chunking/templates",
            params={
                key: value
                for key, value in {
                    "include_builtin": include_builtin,
                    "include_custom": include_custom,
                    "tags": tags,
                    "user_id": user_id,
                }.items()
                if value is not None
            },
        )
        return ChunkingTemplateListResponse.model_validate(response)

    async def get_chunking_template(self, template_name: str) -> ChunkingTemplateResponse:
        response = await self._request("GET", f"/api/v1/chunking/templates/{template_name}")
        return ChunkingTemplateResponse.model_validate(response)

    async def create_chunking_template(
        self,
        request_data: ChunkingTemplateCreateRequest,
    ) -> ChunkingTemplateResponse:
        response = await self._request(
            "POST",
            "/api/v1/chunking/templates",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return ChunkingTemplateResponse.model_validate(response)

    async def update_chunking_template(
        self,
        template_name: str,
        request_data: ChunkingTemplateUpdateRequest,
    ) -> ChunkingTemplateResponse:
        response = await self._request(
            "PUT",
            f"/api/v1/chunking/templates/{template_name}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return ChunkingTemplateResponse.model_validate(response)

    async def delete_chunking_template(self, template_name: str, hard_delete: bool = False) -> None:
        await self._request(
            "DELETE",
            f"/api/v1/chunking/templates/{template_name}",
            params={"hard_delete": hard_delete},
        )

    async def apply_chunking_template(
        self,
        request_data: ChunkingTemplateApplyRequest,
        *,
        include_metadata: bool = False,
    ) -> ChunkingTemplateApplyResponse:
        response = await self._request(
            "POST",
            "/api/v1/chunking/templates/apply",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
            params={"include_metadata": include_metadata},
        )
        return ChunkingTemplateApplyResponse.model_validate(response)

    async def get_chunking_template_diagnostics(self) -> ChunkingTemplateDiagnosticsResponse:
        response = await self._request("GET", "/api/v1/chunking/templates/diagnostics")
        return ChunkingTemplateDiagnosticsResponse.model_validate(response)

    async def validate_chunking_template(
        self,
        template_config: dict[str, Any],
    ) -> ChunkingTemplateValidationResponse:
        response = await self._request(
            "POST",
            "/api/v1/chunking/templates/validate",
            json_data=dict(template_config),
        )
        return ChunkingTemplateValidationResponse.model_validate(response)

    async def match_chunking_templates(
        self,
        *,
        media_type: Optional[str] = None,
        title: Optional[str] = None,
        url: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> ChunkingTemplateMatchResponse:
        params = {
            key: value
            for key, value in {
                "media_type": media_type,
                "title": title,
                "url": url,
                "filename": filename,
            }.items()
            if value is not None
        }
        response = await self._request(
            "POST",
            "/api/v1/chunking/templates/match",
            params=params,
        )
        return ChunkingTemplateMatchResponse.model_validate(response)

    async def learn_chunking_template(
        self,
        request_data: ChunkingTemplateLearnRequest,
    ) -> ChunkingTemplateLearnResponse:
        response = await self._request(
            "POST",
            "/api/v1/chunking/templates/learn",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return ChunkingTemplateLearnResponse.model_validate(response)

    async def list_embedding_collections(self) -> EmbeddingCollectionListResponse:
        response = await self._request("GET", "/api/v1/embeddings/collections")
        return [EmbeddingCollectionResponse.model_validate(item) for item in response]

    async def create_embedding_collection(
        self,
        request_data: EmbeddingCollectionCreateRequest,
    ) -> EmbeddingCollectionResponse:
        response = await self._request(
            "POST",
            "/api/v1/embeddings/collections",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return EmbeddingCollectionResponse.model_validate(response)

    async def delete_embedding_collection(self, collection_name: str) -> None:
        await self._request("DELETE", f"/api/v1/embeddings/collections/{collection_name}")

    async def get_embedding_collection_stats(
        self,
        collection_name: str,
    ) -> EmbeddingCollectionStatsResponse:
        response = await self._request("GET", f"/api/v1/embeddings/collections/{collection_name}/stats")
        return EmbeddingCollectionStatsResponse.model_validate(response)

    async def get_media_embeddings_status(self, media_id: int) -> MediaEmbeddingsStatusResponse:
        response = await self._request("GET", f"/api/v1/media/{media_id}/embeddings/status")
        return MediaEmbeddingsStatusResponse.model_validate(response)

    async def generate_media_embeddings(
        self,
        media_id: int,
        request_data: MediaEmbeddingsGenerateRequest,
    ) -> MediaEmbeddingsGenerateResponse:
        response = await self._request(
            "POST",
            f"/api/v1/media/{media_id}/embeddings",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return MediaEmbeddingsGenerateResponse.model_validate(response)

    async def generate_media_embeddings_batch(
        self,
        request_data: MediaEmbeddingsBatchRequest,
    ) -> MediaEmbeddingsBatchResponse:
        response = await self._request(
            "POST",
            "/api/v1/media/embeddings/batch",
            json_data=request_data.model_dump(exclude_none=True, mode="json", by_alias=True),
        )
        return MediaEmbeddingsBatchResponse.model_validate(response)

    async def search_media_embeddings(
        self,
        request_data: MediaEmbeddingsSearchRequest,
    ) -> MediaEmbeddingsSearchResponse:
        response = await self._request(
            "POST",
            "/api/v1/media/embeddings/search",
            json_data=request_data.model_dump(exclude_none=True, mode="json", by_alias=True),
        )
        return MediaEmbeddingsSearchResponse.model_validate(response)

    async def delete_media_embeddings(self, media_id: int) -> Dict[str, Any]:
        return await self._request("DELETE", f"/api/v1/media/{media_id}/embeddings")

    async def get_media_embedding_job(self, job_id: str) -> MediaEmbeddingJobResponse:
        response = await self._request("GET", f"/api/v1/media/embeddings/jobs/{job_id}")
        return MediaEmbeddingJobResponse.model_validate(response)

    async def list_media_embedding_jobs(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> MediaEmbeddingJobListResponse:
        response = await self._request(
            "GET",
            "/api/v1/media/embeddings/jobs",
            params={"status": status, "limit": limit, "offset": offset},
        )
        return MediaEmbeddingJobListResponse.model_validate(response)

    async def create_evaluation_dataset(
        self,
        request_data: EvaluationDatasetCreateRequest,
    ) -> EvaluationDatasetResponse:
        response = await self._request(
            "POST",
            "/api/v1/evaluations/datasets",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return EvaluationDatasetResponse.model_validate(response)

    async def list_evaluation_datasets(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> EvaluationDatasetListResponse:
        response = await self._request(
            "GET",
            "/api/v1/evaluations/datasets",
            params={"limit": limit, "offset": offset},
        )
        return EvaluationDatasetListResponse.model_validate(response)

    async def get_evaluation_dataset(
        self,
        dataset_id: str,
        *,
        include_samples: bool = True,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> EvaluationDatasetResponse:
        response = await self._request(
            "GET",
            f"/api/v1/evaluations/datasets/{dataset_id}",
            params={
                key: value
                for key, value in {
                    "include_samples": include_samples,
                    "limit": limit,
                    "offset": offset,
                }.items()
                if value is not None
            },
        )
        return EvaluationDatasetResponse.model_validate(response)

    async def delete_evaluation_dataset(self, dataset_id: str) -> None:
        await self._request("DELETE", f"/api/v1/evaluations/datasets/{dataset_id}")

    async def create_evaluation(
        self,
        request_data: CreateEvaluationRequest,
    ) -> EvaluationResponse:
        response = await self._request(
            "POST",
            "/api/v1/evaluations",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return EvaluationResponse.model_validate(response)

    async def list_evaluations(
        self,
        *,
        limit: int = 20,
        after: Optional[str] = None,
        eval_type: Optional[str] = None,
    ) -> EvaluationListResponse:
        response = await self._request(
            "GET",
            "/api/v1/evaluations",
            params={
                key: value
                for key, value in {
                    "limit": limit,
                    "after": after,
                    "eval_type": eval_type,
                }.items()
                if value is not None
            },
        )
        return EvaluationListResponse.model_validate(response)

    async def get_evaluation(self, eval_id: str) -> EvaluationResponse:
        response = await self._request("GET", f"/api/v1/evaluations/{eval_id}")
        return EvaluationResponse.model_validate(response)

    async def update_evaluation(
        self,
        eval_id: str,
        request_data: UpdateEvaluationRequest,
    ) -> EvaluationResponse:
        response = await self._request(
            "PATCH",
            f"/api/v1/evaluations/{eval_id}",
            json_data=request_data.model_dump(exclude_none=True, exclude_unset=True, mode="json"),
        )
        return EvaluationResponse.model_validate(response)

    async def delete_evaluation(self, eval_id: str) -> None:
        await self._request("DELETE", f"/api/v1/evaluations/{eval_id}")

    async def create_evaluation_run(
        self,
        eval_id: str,
        request_data: EvaluationRunCreateRequest,
    ) -> EvaluationRunResponse:
        response = await self._request(
            "POST",
            f"/api/v1/evaluations/{eval_id}/runs",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return EvaluationRunResponse.model_validate(response)

    async def list_evaluation_runs(
        self,
        eval_id: str,
        *,
        limit: int = 20,
        after: Optional[str] = None,
        status: Optional[str] = None,
    ) -> EvaluationRunListResponse:
        response = await self._request(
            "GET",
            f"/api/v1/evaluations/{eval_id}/runs",
            params={
                key: value
                for key, value in {
                    "limit": limit,
                    "after": after,
                    "status": status,
                }.items()
                if value is not None
            },
        )
        return EvaluationRunListResponse.model_validate(response)

    async def get_evaluation_run(self, run_id: str) -> EvaluationRunResponse:
        response = await self._request("GET", f"/api/v1/evaluations/runs/{run_id}")
        return EvaluationRunResponse.model_validate(response)

    async def cancel_evaluation_run(self, run_id: str) -> Dict[str, Any]:
        return await self._request("POST", f"/api/v1/evaluations/runs/{run_id}/cancel")

    async def create_or_update_evaluation_rag_pipeline_preset(
        self,
        *,
        name: str,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        return await self._request(
            "POST",
            "/api/v1/evaluations/rag/pipeline/presets",
            json_data={"name": name, "config": config},
        )

    async def list_evaluation_rag_pipeline_presets(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        return await self._request(
            "GET",
            "/api/v1/evaluations/rag/pipeline/presets",
            params={"limit": limit, "offset": offset},
        )

    async def get_evaluation_rag_pipeline_preset(self, name: str) -> Dict[str, Any]:
        return await self._request("GET", f"/api/v1/evaluations/rag/pipeline/presets/{name}")

    async def delete_evaluation_rag_pipeline_preset(self, name: str) -> None:
        await self._request("DELETE", f"/api/v1/evaluations/rag/pipeline/presets/{name}")

    async def cleanup_evaluation_rag_pipeline(self) -> Dict[str, Any]:
        return await self._request("POST", "/api/v1/evaluations/rag/pipeline/cleanup")

    async def create_evaluation_embeddings_abtest(
        self,
        *,
        name: str,
        config: Dict[str, Any],
        run_immediately: bool | None = False,
    ) -> Dict[str, Any]:
        return await self._request(
            "POST",
            "/api/v1/evaluations/embeddings/abtest",
            json_data={
                "name": name,
                "config": config,
                "run_immediately": run_immediately,
            },
        )

    async def run_evaluation_embeddings_abtest(
        self,
        test_id: str,
        *,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        return await self._request(
            "POST",
            f"/api/v1/evaluations/embeddings/abtest/{test_id}/run",
            json_data={"config": config},
        )

    async def get_evaluation_embeddings_abtest_status(self, test_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/api/v1/evaluations/embeddings/abtest/{test_id}")

    async def get_evaluation_embeddings_abtest_results(
        self,
        test_id: str,
        *,
        page: int = 1,
        page_size: int = 50,
    ) -> Dict[str, Any]:
        return await self._request(
            "GET",
            f"/api/v1/evaluations/embeddings/abtest/{test_id}/results",
            params={"page": page, "page_size": page_size},
        )

    async def get_evaluation_embeddings_abtest_significance(
        self,
        test_id: str,
        *,
        metric: str = "ndcg",
    ) -> Dict[str, Any]:
        return await self._request(
            "GET",
            f"/api/v1/evaluations/embeddings/abtest/{test_id}/significance",
            params={"metric": metric},
        )

    async def export_evaluation_embeddings_abtest(
        self,
        test_id: str,
        *,
        format: Literal["json", "csv"] = "json",
    ) -> Any:
        return await self._request(
            "GET",
            f"/api/v1/evaluations/embeddings/abtest/{test_id}/export",
            params={"format": format},
        )

    async def delete_evaluation_embeddings_abtest(self, test_id: str) -> Dict[str, Any]:
        return await self._request("DELETE", f"/api/v1/evaluations/embeddings/abtest/{test_id}")

    async def generate_synthetic_evaluation_drafts(
        self,
        request_data: SyntheticEvalGenerationRequest,
    ) -> SyntheticEvalGenerationResponse:
        response = await self._request(
            "POST",
            "/api/v1/evaluations/synthetic/drafts/generate",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return SyntheticEvalGenerationResponse.model_validate(response)

    async def list_synthetic_evaluation_queue(
        self,
        *,
        recipe_kind: str | None = None,
        review_state: str | None = None,
        source_kind: str | None = None,
        generation_batch_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> SyntheticEvalQueueResponse:
        response = await self._request(
            "GET",
            "/api/v1/evaluations/synthetic/queue",
            params={
                key: value
                for key, value in {
                    "recipe_kind": recipe_kind,
                    "review_state": review_state,
                    "source_kind": source_kind,
                    "generation_batch_id": generation_batch_id,
                    "limit": limit,
                    "offset": offset,
                }.items()
                if value is not None
            },
        )
        return SyntheticEvalQueueResponse.model_validate(response)

    async def review_synthetic_evaluation_sample(
        self,
        sample_id: str,
        request_data: SyntheticEvalReviewRequest,
    ) -> SyntheticEvalReviewActionRecord:
        response = await self._request(
            "POST",
            f"/api/v1/evaluations/synthetic/queue/{sample_id}/review",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return SyntheticEvalReviewActionRecord.model_validate(response)

    async def promote_synthetic_evaluation_samples(
        self,
        request_data: SyntheticEvalPromotionRequest,
    ) -> SyntheticEvalPromotionResponse:
        response = await self._request(
            "POST",
            "/api/v1/evaluations/synthetic/promotions",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return SyntheticEvalPromotionResponse.model_validate(response)

    async def list_evaluation_benchmarks(self) -> Dict[str, Any]:
        return await self._request("GET", "/api/v1/evaluations/benchmarks")

    async def get_evaluation_benchmark(self, benchmark_name: str) -> Dict[str, Any]:
        return await self._request("GET", f"/api/v1/evaluations/benchmarks/{benchmark_name}")

    async def run_evaluation_benchmark(
        self,
        benchmark_name: str,
        *,
        limit: int | None = None,
        api_name: str = "openai",
        parallel: int = 4,
        save_results: bool = True,
        filter_categories: list[str] | None = None,
    ) -> Dict[str, Any]:
        return await self._request(
            "POST",
            f"/api/v1/evaluations/benchmarks/{benchmark_name}/run",
            json_data={
                key: value
                for key, value in {
                    "limit": limit,
                    "api_name": api_name,
                    "parallel": parallel,
                    "save_results": save_results,
                    "filter_categories": filter_categories,
                }.items()
                if value is not None
            },
        )

    async def register_evaluation_webhook(
        self,
        request_data: WebhookRegistrationRequest,
    ) -> WebhookRegistrationResponse:
        response = await self._request(
            "POST",
            "/api/v1/evaluations/webhooks",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return WebhookRegistrationResponse.model_validate(response)

    async def list_evaluation_webhooks(self) -> list[WebhookStatusResponse]:
        response = await self._request("GET", "/api/v1/evaluations/webhooks")
        return [WebhookStatusResponse.model_validate(item) for item in list(response or [])]

    async def unregister_evaluation_webhook(self, url: str) -> Dict[str, Any]:
        return await self._request(
            "DELETE",
            "/api/v1/evaluations/webhooks",
            params={"url": url},
        )

    async def test_evaluation_webhook(
        self,
        request_data: WebhookTestRequest,
    ) -> WebhookTestResponse:
        response = await self._request(
            "POST",
            "/api/v1/evaluations/webhooks/test",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return WebhookTestResponse.model_validate(response)

    async def list_evaluation_recipe_manifests(self) -> list[RecipeManifest]:
        response = await self._request("GET", "/api/v1/evaluations/recipes")
        return [RecipeManifest.model_validate(item) for item in list(response or [])]

    async def get_evaluation_recipe_manifest(self, recipe_id: str) -> RecipeManifest:
        response = await self._request("GET", f"/api/v1/evaluations/recipes/{recipe_id}")
        return RecipeManifest.model_validate(response)

    async def get_evaluation_recipe_launch_readiness(
        self,
        recipe_id: str,
    ) -> RecipeLaunchReadiness:
        response = await self._request(
            "GET",
            f"/api/v1/evaluations/recipes/{recipe_id}/launch-readiness",
        )
        return RecipeLaunchReadiness.model_validate(response)

    async def validate_evaluation_recipe_dataset(
        self,
        recipe_id: str,
        request_data: RecipeDatasetValidationRequest,
    ) -> RecipeDatasetValidationResponse:
        response = await self._request(
            "POST",
            f"/api/v1/evaluations/recipes/{recipe_id}/validate-dataset",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return RecipeDatasetValidationResponse.model_validate(response)

    async def create_evaluation_recipe_run(
        self,
        recipe_id: str,
        request_data: RecipeRunCreateRequest,
    ) -> RecipeRunRecord:
        response = await self._request(
            "POST",
            f"/api/v1/evaluations/recipes/{recipe_id}/runs",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return RecipeRunRecord.model_validate(response)

    async def get_evaluation_recipe_run(self, run_id: str) -> RecipeRunRecord:
        response = await self._request("GET", f"/api/v1/evaluations/recipe-runs/{run_id}")
        return RecipeRunRecord.model_validate(response)

    async def get_evaluation_recipe_run_report(self, run_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/api/v1/evaluations/recipe-runs/{run_id}/report")

    async def create_flashcard_deck(
        self,
        request_data: FlashcardDeckCreateRequest,
    ) -> FlashcardDeckResponse:
        response = await self._request(
            "POST",
            "/api/v1/flashcards/decks",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return FlashcardDeckResponse.model_validate(response)

    async def list_flashcard_decks(
        self,
        *,
        include_deleted: bool | None = None,
        workspace_id: str | None = None,
        include_workspace_items: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[FlashcardDeckResponse]:
        response = await self._request(
            "GET",
            "/api/v1/flashcards/decks",
            params={
                key: value
                for key, value in {
                    "include_deleted": include_deleted,
                    "workspace_id": workspace_id,
                    "include_workspace_items": include_workspace_items,
                    "limit": limit,
                    "offset": offset,
                }.items()
                if value is not None
            },
        )
        return [FlashcardDeckResponse.model_validate(item) for item in list(response or [])]

    async def update_flashcard_deck(
        self,
        deck_id: int,
        request_data: FlashcardDeckUpdateRequest,
    ) -> FlashcardDeckResponse:
        response = await self._request(
            "PATCH",
            f"/api/v1/flashcards/decks/{deck_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return FlashcardDeckResponse.model_validate(response)

    async def upload_flashcard_asset(self, file: tuple[str, bytes, str]) -> FlashcardAssetMetadata:
        response = await self._request(
            "POST",
            "/api/v1/flashcards/assets",
            files=[("file", file)],
        )
        return FlashcardAssetMetadata.model_validate(response)

    async def get_flashcard_asset_content(self, asset_uuid: str) -> ReadingExportResponse:
        return await self._binary_request("GET", f"/api/v1/flashcards/assets/{asset_uuid}/content")

    async def create_flashcard(
        self,
        request_data: FlashcardCreateRequest,
    ) -> FlashcardResponse:
        response = await self._request(
            "POST",
            "/api/v1/flashcards",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return FlashcardResponse.model_validate(response)

    async def create_flashcards_bulk(
        self,
        request_data: list[FlashcardCreateRequest],
    ) -> FlashcardListResponse:
        response = await self._request(
            "POST",
            "/api/v1/flashcards/bulk",
            json_data=[item.model_dump(exclude_none=True, mode="json") for item in request_data],
        )
        return FlashcardListResponse.model_validate(response)

    async def update_flashcards_bulk(
        self,
        request_data: list[FlashcardBulkUpdateItem],
    ) -> FlashcardBulkUpdateResponse:
        response = await self._request(
            "PATCH",
            "/api/v1/flashcards/bulk",
            json_data=[item.model_dump(exclude_none=True, mode="json") for item in request_data],
        )
        return FlashcardBulkUpdateResponse.model_validate(response)

    async def update_flashcard(
        self,
        card_uuid: str,
        request_data: FlashcardUpdateRequest,
    ) -> FlashcardResponse:
        response = await self._request(
            "PATCH",
            f"/api/v1/flashcards/{card_uuid}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return FlashcardResponse.model_validate(response)

    async def update_flashcards_bulk(
        self,
        request_data: list[FlashcardBulkUpdateItemRequest],
    ) -> FlashcardBulkUpdateResponse:
        response = await self._request(
            "PATCH",
            "/api/v1/flashcards/bulk",
            json_data=[
                item.model_dump(exclude_none=True, mode="json")
                for item in request_data
            ],
        )
        return FlashcardBulkUpdateResponse.model_validate(response)

    async def reset_flashcard_scheduling(
        self,
        card_uuid: str,
        request_data: FlashcardResetSchedulingRequest,
    ) -> FlashcardResponse:
        response = await self._request(
            "POST",
            f"/api/v1/flashcards/{card_uuid}/reset-scheduling",
            json_data=request_data.model_dump(mode="json"),
        )
        return FlashcardResponse.model_validate(response)

    async def set_flashcard_tags(
        self,
        card_uuid: str,
        request_data: FlashcardTagsUpdateRequest,
    ) -> FlashcardResponse:
        response = await self._request(
            "PUT",
            f"/api/v1/flashcards/{card_uuid}/tags",
            json_data=request_data.model_dump(mode="json"),
        )
        return FlashcardResponse.model_validate(response)

    async def get_flashcard_tags(self, card_uuid: str) -> FlashcardTagsResponse:
        response = await self._request("GET", f"/api/v1/flashcards/{card_uuid}/tags")
        return FlashcardTagsResponse.model_validate(response)

    async def list_flashcard_tag_suggestions(
        self,
        *,
        q: Optional[str] = None,
        limit: int = 50,
    ) -> FlashcardTagSuggestionsResponse:
        response = await self._request(
            "GET",
            "/api/v1/flashcards/tags",
            params={
                key: value
                for key, value in {"q": q, "limit": limit}.items()
                if value is not None
            },
        )
        return FlashcardTagSuggestionsResponse.model_validate(response)

    async def preview_structured_qa_import(
        self,
        request_data: StructuredQaImportPreviewRequest,
        *,
        max_lines: Optional[int] = None,
        max_line_length: Optional[int] = None,
        max_field_length: Optional[int] = None,
    ) -> StructuredQaImportPreviewResponse:
        response = await self._request(
            "POST",
            "/api/v1/flashcards/import/structured/preview",
            json_data=request_data.model_dump(mode="json"),
            params={
                key: value
                for key, value in {
                    "max_lines": max_lines,
                    "max_line_length": max_line_length,
                    "max_field_length": max_field_length,
                }.items()
                if value is not None
            },
        )
        return StructuredQaImportPreviewResponse.model_validate(response)

    async def import_flashcards_tsv(
        self,
        request_data: FlashcardsImportRequest,
        *,
        max_lines: Optional[int] = None,
        max_line_length: Optional[int] = None,
        max_field_length: Optional[int] = None,
    ) -> FlashcardsImportResponse:
        response = await self._request(
            "POST",
            "/api/v1/flashcards/import",
            json_data=request_data.model_dump(mode="json"),
            params={
                key: value
                for key, value in {
                    "max_lines": max_lines,
                    "max_line_length": max_line_length,
                    "max_field_length": max_field_length,
                }.items()
                if value is not None
            },
        )
        return FlashcardsImportResponse.model_validate(response)

    async def upload_flashcard_asset(self, file_path: Union[str, Path, tuple[str, bytes, str]]) -> FlashcardAssetMetadata:
        if isinstance(file_path, tuple):
            response = await self._request(
                "POST",
                "/api/v1/flashcards/assets",
                files=[("file", file_path)],
            )
            return FlashcardAssetMetadata.model_validate(response)
        httpx_files = prepare_files_for_httpx([file_path], upload_field_name="file")
        try:
            response = await self._request(
                "POST",
                "/api/v1/flashcards/assets",
                files=httpx_files,
            )
            return FlashcardAssetMetadata.model_validate(response)
        finally:
            cleanup_file_objects(httpx_files)

    async def get_flashcard_asset_content(self, asset_uuid: str) -> ReadingExportResponse:
        return await self._binary_request(
            "GET",
            f"/api/v1/flashcards/assets/{asset_uuid}/content",
        )

    async def import_flashcards_json_file(
        self,
        file_path: Union[str, Path],
        *,
        max_items: Optional[int] = None,
        max_field_length: Optional[int] = None,
    ) -> FlashcardsImportResponse:
        httpx_files = prepare_files_for_httpx([file_path], upload_field_name="file")
        try:
            response = await self._request(
                "POST",
                "/api/v1/flashcards/import/json",
                files=httpx_files,
                params={
                    key: value
                    for key, value in {
                        "max_items": max_items,
                        "max_field_length": max_field_length,
                    }.items()
                    if value is not None
                },
            )
            return FlashcardsImportResponse.model_validate(response)
        finally:
            cleanup_file_objects(httpx_files)

    async def import_flashcards_apkg(
        self,
        file_path: Union[str, Path],
        *,
        max_items: Optional[int] = None,
        max_field_length: Optional[int] = None,
    ) -> FlashcardsImportResponse:
        httpx_files = prepare_files_for_httpx([file_path], upload_field_name="file")
        try:
            response = await self._request(
                "POST",
                "/api/v1/flashcards/import/apkg",
                files=httpx_files,
                params={
                    key: value
                    for key, value in {
                        "max_items": max_items,
                        "max_field_length": max_field_length,
                    }.items()
                    if value is not None
                },
            )
            return FlashcardsImportResponse.model_validate(response)
        finally:
            cleanup_file_objects(httpx_files)

    async def get_flashcard_study_assistant_context(
        self,
        card_uuid: str,
    ) -> StudyAssistantContextResponse:
        response = await self._request(
            "GET",
            f"/api/v1/flashcards/{card_uuid}/assistant",
        )
        return StudyAssistantContextResponse.model_validate(response)

    async def respond_flashcard_study_assistant(
        self,
        card_uuid: str,
        request_data: StudyAssistantRespondRequest,
    ) -> StudyAssistantRespondResponse:
        response = await self._request(
            "POST",
            f"/api/v1/flashcards/{card_uuid}/assistant/respond",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return StudyAssistantRespondResponse.model_validate(response)

    async def export_flashcards(
        self,
        *,
        deck_id: Optional[int] = None,
        workspace_id: Optional[str] = None,
        include_workspace_items: bool = False,
        tag: Optional[str] = None,
        q: Optional[str] = None,
        export_format: str = "csv",
        include_reverse: bool = False,
        delimiter: str = "\t",
        include_header: bool = False,
        extended_header: bool = False,
    ) -> bytes:
        return await self._request_bytes(
            "GET",
            "/api/v1/flashcards/export",
            params={
                key: value
                for key, value in {
                    "deck_id": deck_id,
                    "workspace_id": workspace_id,
                    "include_workspace_items": include_workspace_items,
                    "tag": tag,
                    "q": q,
                    "format": export_format,
                    "include_reverse": include_reverse,
                    "delimiter": delimiter,
                    "include_header": include_header,
                    "extended_header": extended_header,
                }.items()
                if value is not None
            },
        )

    async def get_flashcard_analytics_summary(
        self,
        *,
        deck_id: Optional[int] = None,
        workspace_id: Optional[str] = None,
        include_workspace_items: bool = False,
    ) -> FlashcardAnalyticsSummaryResponse:
        response = await self._request(
            "GET",
            "/api/v1/flashcards/analytics/summary",
            params={
                key: value
                for key, value in {
                    "deck_id": deck_id,
                    "workspace_id": workspace_id,
                    "include_workspace_items": include_workspace_items,
                }.items()
                if value is not None
            },
        )
        return FlashcardAnalyticsSummaryResponse.model_validate(response)

    async def create_flashcard_template(
        self,
        request_data: FlashcardTemplateCreateRequest,
    ) -> FlashcardTemplateResponse:
        response = await self._request(
            "POST",
            "/api/v1/flashcards/templates",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return FlashcardTemplateResponse.model_validate(response)

    async def list_flashcard_templates(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> FlashcardTemplateListResponse:
        response = await self._request(
            "GET",
            "/api/v1/flashcards/templates",
            params={"limit": limit, "offset": offset},
        )
        return FlashcardTemplateListResponse.model_validate(response)

    async def get_flashcard_template(self, template_id: int) -> FlashcardTemplateResponse:
        response = await self._request("GET", f"/api/v1/flashcards/templates/{template_id}")
        return FlashcardTemplateResponse.model_validate(response)

    async def update_flashcard_template(
        self,
        template_id: int,
        request_data: FlashcardTemplateUpdateRequest,
    ) -> FlashcardTemplateResponse:
        response = await self._request(
            "PATCH",
            f"/api/v1/flashcards/templates/{template_id}",
            json_data=request_data.model_dump(exclude_unset=True, mode="json"),
        )
        return FlashcardTemplateResponse.model_validate(response)

    async def delete_flashcard_template(self, template_id: int, *, expected_version: int) -> dict[str, Any]:
        return await self._request(
            "DELETE",
            f"/api/v1/flashcards/templates/{template_id}",
            params={"expected_version": expected_version},
        )

    async def delete_flashcard(
        self,
        card_uuid: str,
        *,
        expected_version: int,
    ) -> Dict[str, Any]:
        return await self._request(
            "DELETE",
            f"/api/v1/flashcards/{card_uuid}",
            params={"expected_version": expected_version},
        )

    async def list_flashcards(
        self,
        *,
        deck_id: Optional[int] = None,
        workspace_id: Optional[str] = None,
        include_workspace_items: bool | None = None,
        tag: Optional[str] = None,
        due_status: Optional[str] = None,
        q: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        order_by: Optional[str] = None,
    ) -> FlashcardListResponse:
        response = await self._request(
            "GET",
            "/api/v1/flashcards",
            params={
                key: value
                for key, value in {
                    "deck_id": deck_id,
                    "workspace_id": workspace_id,
                    "include_workspace_items": include_workspace_items,
                    "tag": tag,
                    "due_status": due_status,
                    "q": q,
                    "limit": limit,
                    "offset": offset,
                    "order_by": order_by,
                }.items()
                if value is not None
            },
        )
        return FlashcardListResponse.model_validate(response)

    async def list_flashcard_tag_suggestions(
        self,
        *,
        q: Optional[str] = None,
        limit: int = 50,
    ) -> FlashcardTagSuggestionsResponse:
        response = await self._request(
            "GET",
            "/api/v1/flashcards/tags",
            params={key: value for key, value in {"q": q, "limit": limit}.items() if value is not None},
        )
        return FlashcardTagSuggestionsResponse.model_validate(response)

    async def get_flashcard_analytics_summary(
        self,
        *,
        deck_id: Optional[int] = None,
        workspace_id: Optional[str] = None,
        include_workspace_items: bool | None = None,
    ) -> FlashcardAnalyticsSummaryResponse:
        response = await self._request(
            "GET",
            "/api/v1/flashcards/analytics/summary",
            params={
                key: value
                for key, value in {
                    "deck_id": deck_id,
                    "workspace_id": workspace_id,
                    "include_workspace_items": include_workspace_items,
                }.items()
                if value is not None
            },
        )
        return FlashcardAnalyticsSummaryResponse.model_validate(response)

    async def get_flashcard(self, card_uuid: str) -> FlashcardResponse:
        response = await self._request("GET", f"/api/v1/flashcards/id/{card_uuid}")
        return FlashcardResponse.model_validate(response)

    async def reset_flashcard_scheduling(
        self,
        card_uuid: str,
        request_data: FlashcardResetSchedulingRequest,
    ) -> FlashcardResponse:
        response = await self._request(
            "POST",
            f"/api/v1/flashcards/{card_uuid}/reset-scheduling",
            json_data=request_data.model_dump(mode="json"),
        )
        return FlashcardResponse.model_validate(response)

    async def set_flashcard_tags(
        self,
        card_uuid: str,
        request_data: FlashcardTagsUpdate,
    ) -> FlashcardResponse:
        response = await self._request(
            "PUT",
            f"/api/v1/flashcards/{card_uuid}/tags",
            json_data=request_data.model_dump(mode="json"),
        )
        return FlashcardResponse.model_validate(response)

    async def get_flashcard_tags(self, card_uuid: str) -> Dict[str, Any]:
        return await self._request("GET", f"/api/v1/flashcards/{card_uuid}/tags")

    async def preview_structured_qa_import(
        self,
        request_data: StructuredQaImportPreviewRequest,
        *,
        max_lines: Optional[int] = None,
        max_line_length: Optional[int] = None,
        max_field_length: Optional[int] = None,
    ) -> StructuredQaImportPreviewResponse:
        params = {
            key: value
            for key, value in {
                "max_lines": max_lines,
                "max_line_length": max_line_length,
                "max_field_length": max_field_length,
            }.items()
            if value is not None
        }
        response = await self._request(
            "POST",
            "/api/v1/flashcards/import/structured/preview",
            params=params or None,
            json_data=request_data.model_dump(mode="json"),
        )
        return StructuredQaImportPreviewResponse.model_validate(response)

    async def import_flashcards(
        self,
        request_data: FlashcardsImportRequest,
        *,
        max_lines: Optional[int] = None,
        max_line_length: Optional[int] = None,
        max_field_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        params = {
            key: value
            for key, value in {
                "max_lines": max_lines,
                "max_line_length": max_line_length,
                "max_field_length": max_field_length,
            }.items()
            if value is not None
        }
        return await self._request(
            "POST",
            "/api/v1/flashcards/import",
            params=params or None,
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )

    async def import_flashcards_json(
        self,
        file: tuple[str, bytes, str],
        *,
        max_items: Optional[int] = None,
        max_field_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        params = {
            key: value for key, value in {"max_items": max_items, "max_field_length": max_field_length}.items()
            if value is not None
        }
        return await self._request(
            "POST",
            "/api/v1/flashcards/import/json",
            params=params or None,
            files=[("file", file)],
        )

    async def import_flashcards_apkg(
        self,
        file: tuple[str, bytes, str],
        *,
        max_items: Optional[int] = None,
        max_field_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        params = {
            key: value for key, value in {"max_items": max_items, "max_field_length": max_field_length}.items()
            if value is not None
        }
        return await self._request(
            "POST",
            "/api/v1/flashcards/import/apkg",
            params=params or None,
            files=[("file", file)],
        )

    async def list_flashcard_review_sessions(
        self,
        *,
        deck_id: Optional[int] = None,
        scope_key: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 20,
    ) -> list[FlashcardReviewSessionSummary]:
        response = await self._request(
            "GET",
            "/api/v1/flashcards/review-sessions",
            params={
                key: value
                for key, value in {
                    "deck_id": deck_id,
                    "scope_key": scope_key,
                    "status": status,
                    "limit": limit,
                }.items()
                if value is not None
            },
        )
        return [FlashcardReviewSessionSummary.model_validate(item) for item in response]

    async def get_next_flashcard_review(
        self,
        *,
        deck_id: Optional[int] = None,
        workspace_id: Optional[str] = None,
        include_workspace_items: bool | None = None,
    ) -> FlashcardNextReviewResponse:
        response = await self._request(
            "GET",
            "/api/v1/flashcards/review/next",
            params={
                key: value
                for key, value in {
                    "deck_id": deck_id,
                    "workspace_id": workspace_id,
                    "include_workspace_items": include_workspace_items,
                }.items()
                if value is not None
            },
        )
        return FlashcardNextReviewResponse.model_validate(response)

    async def review_flashcard(
        self,
        request_data: FlashcardReviewRequest,
    ) -> FlashcardReviewResponse:
        response = await self._request(
            "POST",
            "/api/v1/flashcards/review",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return FlashcardReviewResponse.model_validate(response)

    async def end_flashcard_review_session(self, review_session_id: int) -> FlashcardReviewSessionSummary:
        response = await self._request(
            "POST",
            "/api/v1/flashcards/review-sessions/end",
            json_data=FlashcardReviewSessionEndRequest(review_session_id=review_session_id).model_dump(mode="json"),
        )
        return FlashcardReviewSessionSummary.model_validate(response)

    async def get_flashcard_assistant(self, card_uuid: str) -> StudyAssistantContextResponse:
        response = await self._request("GET", f"/api/v1/flashcards/{card_uuid}/assistant")
        return StudyAssistantContextResponse.model_validate(response)

    async def respond_flashcard_assistant(
        self,
        card_uuid: str,
        request_data: StudyAssistantRespondRequest,
    ) -> StudyAssistantRespondResponse:
        response = await self._request(
            "POST",
            f"/api/v1/flashcards/{card_uuid}/assistant/respond",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return StudyAssistantRespondResponse.model_validate(response)

    async def generate_flashcards(self, request_data: FlashcardGenerateRequest) -> FlashcardGenerateResponse:
        response = await self._request(
            "POST",
            "/api/v1/flashcards/generate",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return FlashcardGenerateResponse.model_validate(response)

    async def export_flashcards(
        self,
        *,
        deck_id: Optional[int] = None,
        workspace_id: Optional[str] = None,
        include_workspace_items: bool | None = None,
        tag: Optional[str] = None,
        q: Optional[str] = None,
        format: str = "csv",
        include_reverse: bool | None = None,
        delimiter: Optional[str] = None,
        include_header: bool | None = None,
        extended_header: bool | None = None,
    ) -> ReadingExportResponse:
        params = {
            key: value
            for key, value in {
                "deck_id": deck_id,
                "workspace_id": workspace_id,
                "include_workspace_items": include_workspace_items,
                "tag": tag,
                "q": q,
                "format": format,
                "include_reverse": include_reverse,
                "delimiter": delimiter,
                "include_header": include_header,
                "extended_header": extended_header,
            }.items()
            if value is not None
        }
        return await self._binary_request("GET", "/api/v1/flashcards/export", params=params)

    async def create_flashcard_template(
        self,
        request_data: FlashcardTemplateCreateRequest,
    ) -> FlashcardTemplateResponse:
        response = await self._request(
            "POST",
            "/api/v1/flashcards/templates",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return FlashcardTemplateResponse.model_validate(response)

    async def list_flashcard_templates(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> FlashcardTemplateListResponse:
        response = await self._request(
            "GET",
            "/api/v1/flashcards/templates",
            params={"limit": limit, "offset": offset},
        )
        return FlashcardTemplateListResponse.model_validate(response)

    async def get_flashcard_template(self, template_id: int) -> FlashcardTemplateResponse:
        response = await self._request("GET", f"/api/v1/flashcards/templates/{template_id}")
        return FlashcardTemplateResponse.model_validate(response)

    async def update_flashcard_template(
        self,
        template_id: int,
        request_data: FlashcardTemplateUpdateRequest,
    ) -> FlashcardTemplateResponse:
        response = await self._request(
            "PATCH",
            f"/api/v1/flashcards/templates/{template_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return FlashcardTemplateResponse.model_validate(response)

    async def delete_flashcard_template(self, template_id: int, *, expected_version: int) -> Dict[str, Any]:
        return await self._request(
            "DELETE",
            f"/api/v1/flashcards/templates/{template_id}",
            params={"expected_version": expected_version},
        )

    async def create_study_pack_job(
        self,
        request_data: StudyPackCreateJobRequest,
    ) -> StudyPackJobAcceptedResponse:
        response = await self._request(
            "POST",
            "/api/v1/flashcards/study-packs/jobs",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return StudyPackJobAcceptedResponse.model_validate(response)

    async def get_study_pack_job_status(self, job_id: int) -> StudyPackJobStatusResponse:
        response = await self._request("GET", f"/api/v1/flashcards/study-packs/jobs/{job_id}")
        return StudyPackJobStatusResponse.model_validate(response)

    async def get_study_pack(self, pack_id: int) -> StudyPackSummaryResponse:
        response = await self._request("GET", f"/api/v1/flashcards/study-packs/{pack_id}")
        return StudyPackSummaryResponse.model_validate(response)

    async def regenerate_study_pack(self, pack_id: int) -> StudyPackJobAcceptedResponse:
        response = await self._request("POST", f"/api/v1/flashcards/study-packs/{pack_id}/regenerate")
        return StudyPackJobAcceptedResponse.model_validate(response)

    async def get_study_suggestion_status(
        self,
        anchor_type: str,
        anchor_id: int,
    ) -> SuggestionStatusResponse:
        response = await self._request(
            "GET",
            f"/api/v1/study-suggestions/anchors/{anchor_type}/{anchor_id}/status",
        )
        return SuggestionStatusResponse.model_validate(response)

    async def get_study_suggestion_snapshot(self, snapshot_id: int) -> SuggestionSnapshotResponse:
        response = await self._request("GET", f"/api/v1/study-suggestions/snapshots/{snapshot_id}")
        return SuggestionSnapshotResponse.model_validate(response)

    async def refresh_study_suggestion_snapshot(
        self,
        snapshot_id: int,
        request_data: SuggestionRefreshRequest,
    ) -> SuggestionJobAcceptedResponse:
        response = await self._request(
            "POST",
            f"/api/v1/study-suggestions/snapshots/{snapshot_id}/refresh",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return SuggestionJobAcceptedResponse.model_validate(response)

    async def trigger_study_suggestion_action(
        self,
        snapshot_id: int,
        request_data: SuggestionActionRequest,
    ) -> SuggestionActionResponse:
        response = await self._request(
            "POST",
            f"/api/v1/study-suggestions/snapshots/{snapshot_id}/actions",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return SuggestionActionResponse.model_validate(response)

    async def list_manuscript_projects(
        self,
        *,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> ManuscriptProjectListResponse:
        response = await self._request(
            "GET",
            "/api/v1/writing/manuscripts/projects",
            params={
                key: value
                for key, value in {"status": status, "limit": limit, "offset": offset}.items()
                if value is not None
            },
        )
        return ManuscriptProjectListResponse.model_validate(response)

    async def create_manuscript_project(
        self,
        request_data: ManuscriptProjectCreate,
    ) -> ManuscriptProjectResponse:
        response = await self._request(
            "POST",
            "/api/v1/writing/manuscripts/projects",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return ManuscriptProjectResponse.model_validate(response)

    async def get_manuscript_project(self, project_id: str) -> ManuscriptProjectResponse:
        response = await self._request("GET", f"/api/v1/writing/manuscripts/projects/{project_id}")
        return ManuscriptProjectResponse.model_validate(response)

    async def update_manuscript_project(
        self,
        project_id: str,
        request_data: ManuscriptProjectUpdate,
        *,
        expected_version: int,
    ) -> ManuscriptProjectResponse:
        response = await self._request(
            "PATCH",
            f"/api/v1/writing/manuscripts/projects/{project_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
            headers={"expected-version": str(expected_version)},
        )
        return ManuscriptProjectResponse.model_validate(response)

    async def delete_manuscript_project(self, project_id: str, *, expected_version: int) -> bool:
        await self._request(
            "DELETE",
            f"/api/v1/writing/manuscripts/projects/{project_id}",
            headers={"expected-version": str(expected_version)},
        )
        return True

    async def create_manuscript(
        self,
        project_id: str,
        request_data: ManuscriptPartCreate,
    ) -> ManuscriptPartResponse:
        response = await self._request(
            "POST",
            f"/api/v1/writing/manuscripts/projects/{project_id}/parts",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return ManuscriptPartResponse.model_validate(response)

    async def list_manuscripts(self, project_id: str) -> list[ManuscriptPartResponse]:
        response = await self._request("GET", f"/api/v1/writing/manuscripts/projects/{project_id}/parts")
        return [ManuscriptPartResponse.model_validate(item) for item in list(response or [])]

    async def get_manuscript(self, manuscript_id: str) -> ManuscriptPartResponse:
        response = await self._request("GET", f"/api/v1/writing/manuscripts/parts/{manuscript_id}")
        return ManuscriptPartResponse.model_validate(response)

    async def update_manuscript(
        self,
        manuscript_id: str,
        request_data: ManuscriptPartUpdate,
        *,
        expected_version: int,
    ) -> ManuscriptPartResponse:
        response = await self._request(
            "PATCH",
            f"/api/v1/writing/manuscripts/parts/{manuscript_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
            headers={"expected-version": str(expected_version)},
        )
        return ManuscriptPartResponse.model_validate(response)

    async def delete_manuscript(self, manuscript_id: str, *, expected_version: int) -> bool:
        await self._request(
            "DELETE",
            f"/api/v1/writing/manuscripts/parts/{manuscript_id}",
            headers={"expected-version": str(expected_version)},
        )
        return True

    async def create_manuscript_chapter(
        self,
        project_id: str,
        request_data: ManuscriptChapterCreate,
    ) -> ManuscriptChapterResponse:
        response = await self._request(
            "POST",
            f"/api/v1/writing/manuscripts/projects/{project_id}/chapters",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return ManuscriptChapterResponse.model_validate(response)

    async def list_manuscript_chapters(
        self,
        project_id: str,
        *,
        part_id: Optional[str] = None,
    ) -> list[ManuscriptChapterResponse]:
        response = await self._request(
            "GET",
            f"/api/v1/writing/manuscripts/projects/{project_id}/chapters",
            params={key: value for key, value in {"part_id": part_id}.items() if value is not None},
        )
        return [ManuscriptChapterResponse.model_validate(item) for item in list(response or [])]

    async def get_manuscript_chapter(self, chapter_id: str) -> ManuscriptChapterResponse:
        response = await self._request("GET", f"/api/v1/writing/manuscripts/chapters/{chapter_id}")
        return ManuscriptChapterResponse.model_validate(response)

    async def update_manuscript_chapter(
        self,
        chapter_id: str,
        request_data: ManuscriptChapterUpdate,
        *,
        expected_version: int,
    ) -> ManuscriptChapterResponse:
        response = await self._request(
            "PATCH",
            f"/api/v1/writing/manuscripts/chapters/{chapter_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
            headers={"expected-version": str(expected_version)},
        )
        return ManuscriptChapterResponse.model_validate(response)

    async def delete_manuscript_chapter(self, chapter_id: str, *, expected_version: int) -> bool:
        await self._request(
            "DELETE",
            f"/api/v1/writing/manuscripts/chapters/{chapter_id}",
            headers={"expected-version": str(expected_version)},
        )
        return True

    async def create_manuscript_scene(
        self,
        chapter_id: str,
        request_data: ManuscriptSceneCreate,
    ) -> ManuscriptSceneResponse:
        response = await self._request(
            "POST",
            f"/api/v1/writing/manuscripts/chapters/{chapter_id}/scenes",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return ManuscriptSceneResponse.model_validate(response)

    async def list_manuscript_scenes(self, chapter_id: str) -> list[ManuscriptSceneResponse]:
        response = await self._request("GET", f"/api/v1/writing/manuscripts/chapters/{chapter_id}/scenes")
        return [ManuscriptSceneResponse.model_validate(item) for item in list(response or [])]

    async def get_manuscript_scene(self, scene_id: str) -> ManuscriptSceneResponse:
        response = await self._request("GET", f"/api/v1/writing/manuscripts/scenes/{scene_id}")
        return ManuscriptSceneResponse.model_validate(response)

    async def update_manuscript_scene(
        self,
        scene_id: str,
        request_data: ManuscriptSceneUpdate,
        *,
        expected_version: int,
    ) -> ManuscriptSceneResponse:
        response = await self._request(
            "PATCH",
            f"/api/v1/writing/manuscripts/scenes/{scene_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
            headers={"expected-version": str(expected_version)},
        )
        return ManuscriptSceneResponse.model_validate(response)

    async def delete_manuscript_scene(self, scene_id: str, *, expected_version: int) -> bool:
        await self._request(
            "DELETE",
            f"/api/v1/writing/manuscripts/scenes/{scene_id}",
            headers={"expected-version": str(expected_version)},
        )
        return True

    async def get_manuscript_structure(self, project_id: str) -> ManuscriptStructureResponse:
        response = await self._request("GET", f"/api/v1/writing/manuscripts/projects/{project_id}/structure")
        return ManuscriptStructureResponse.model_validate(response)

    async def reorder_manuscript_entities(
        self,
        project_id: str,
        request_data: ReorderRequest,
    ) -> bool:
        await self._request(
            "POST",
            f"/api/v1/writing/manuscripts/projects/{project_id}/reorder",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return True

    async def create_manuscript_character(
        self,
        project_id: str,
        request_data: ManuscriptCharacterCreate,
    ) -> ManuscriptCharacterResponse:
        response = await self._request(
            "POST",
            f"/api/v1/writing/manuscripts/projects/{project_id}/characters",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return ManuscriptCharacterResponse.model_validate(response)

    async def list_manuscript_characters(
        self,
        project_id: str,
        *,
        role: Optional[str] = None,
        cast_group: Optional[str] = None,
    ) -> list[ManuscriptCharacterResponse]:
        response = await self._request(
            "GET",
            f"/api/v1/writing/manuscripts/projects/{project_id}/characters",
            params={key: value for key, value in {"role": role, "cast_group": cast_group}.items() if value is not None},
        )
        return [ManuscriptCharacterResponse.model_validate(item) for item in list(response or [])]

    async def get_manuscript_character(self, character_id: str) -> ManuscriptCharacterResponse:
        response = await self._request("GET", f"/api/v1/writing/manuscripts/characters/{character_id}")
        return ManuscriptCharacterResponse.model_validate(response)

    async def update_manuscript_character(
        self,
        character_id: str,
        request_data: ManuscriptCharacterUpdate,
        *,
        expected_version: int,
    ) -> ManuscriptCharacterResponse:
        response = await self._request(
            "PATCH",
            f"/api/v1/writing/manuscripts/characters/{character_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
            headers={"expected-version": str(expected_version)},
        )
        return ManuscriptCharacterResponse.model_validate(response)

    async def delete_manuscript_character(self, character_id: str, *, expected_version: int) -> bool:
        await self._request(
            "DELETE",
            f"/api/v1/writing/manuscripts/characters/{character_id}",
            headers={"expected-version": str(expected_version)},
        )
        return True

    async def create_manuscript_relationship(
        self,
        project_id: str,
        request_data: ManuscriptRelationshipCreate,
    ) -> ManuscriptRelationshipResponse:
        response = await self._request(
            "POST",
            f"/api/v1/writing/manuscripts/projects/{project_id}/characters/relationships",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return ManuscriptRelationshipResponse.model_validate(response)

    async def list_manuscript_relationships(self, project_id: str) -> list[ManuscriptRelationshipResponse]:
        response = await self._request(
            "GET",
            f"/api/v1/writing/manuscripts/projects/{project_id}/characters/relationships",
        )
        return [ManuscriptRelationshipResponse.model_validate(item) for item in list(response or [])]

    async def delete_manuscript_relationship(self, relationship_id: str, *, expected_version: int) -> bool:
        await self._request(
            "DELETE",
            f"/api/v1/writing/manuscripts/characters/relationships/{relationship_id}",
            headers={"expected-version": str(expected_version)},
        )
        return True

    async def create_manuscript_world_info(
        self,
        project_id: str,
        request_data: ManuscriptWorldInfoCreate,
    ) -> ManuscriptWorldInfoResponse:
        response = await self._request(
            "POST",
            f"/api/v1/writing/manuscripts/projects/{project_id}/world-info",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return ManuscriptWorldInfoResponse.model_validate(response)

    async def list_manuscript_world_info(
        self,
        project_id: str,
        *,
        kind: Optional[str] = None,
    ) -> list[ManuscriptWorldInfoResponse]:
        response = await self._request(
            "GET",
            f"/api/v1/writing/manuscripts/projects/{project_id}/world-info",
            params={key: value for key, value in {"kind": kind}.items() if value is not None},
        )
        return [ManuscriptWorldInfoResponse.model_validate(item) for item in list(response or [])]

    async def get_manuscript_world_info(self, item_id: str) -> ManuscriptWorldInfoResponse:
        response = await self._request("GET", f"/api/v1/writing/manuscripts/world-info/{item_id}")
        return ManuscriptWorldInfoResponse.model_validate(response)

    async def update_manuscript_world_info(
        self,
        item_id: str,
        request_data: ManuscriptWorldInfoUpdate,
        *,
        expected_version: int,
    ) -> ManuscriptWorldInfoResponse:
        response = await self._request(
            "PATCH",
            f"/api/v1/writing/manuscripts/world-info/{item_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
            headers={"expected-version": str(expected_version)},
        )
        return ManuscriptWorldInfoResponse.model_validate(response)

    async def delete_manuscript_world_info(self, item_id: str, *, expected_version: int) -> bool:
        await self._request(
            "DELETE",
            f"/api/v1/writing/manuscripts/world-info/{item_id}",
            headers={"expected-version": str(expected_version)},
        )
        return True

    async def create_manuscript_plot_line(
        self,
        project_id: str,
        request_data: ManuscriptPlotLineCreate,
    ) -> ManuscriptPlotLineResponse:
        response = await self._request(
            "POST",
            f"/api/v1/writing/manuscripts/projects/{project_id}/plot-lines",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return ManuscriptPlotLineResponse.model_validate(response)

    async def list_manuscript_plot_lines(self, project_id: str) -> list[ManuscriptPlotLineResponse]:
        response = await self._request("GET", f"/api/v1/writing/manuscripts/projects/{project_id}/plot-lines")
        return [ManuscriptPlotLineResponse.model_validate(item) for item in list(response or [])]

    async def update_manuscript_plot_line(
        self,
        plot_line_id: str,
        request_data: ManuscriptPlotLineUpdate,
        *,
        expected_version: int,
    ) -> ManuscriptPlotLineResponse:
        response = await self._request(
            "PATCH",
            f"/api/v1/writing/manuscripts/plot-lines/{plot_line_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
            headers={"expected-version": str(expected_version)},
        )
        return ManuscriptPlotLineResponse.model_validate(response)

    async def delete_manuscript_plot_line(self, plot_line_id: str, *, expected_version: int) -> bool:
        await self._request(
            "DELETE",
            f"/api/v1/writing/manuscripts/plot-lines/{plot_line_id}",
            headers={"expected-version": str(expected_version)},
        )
        return True

    async def create_manuscript_plot_event(
        self,
        plot_line_id: str,
        request_data: ManuscriptPlotEventCreate,
    ) -> ManuscriptPlotEventResponse:
        response = await self._request(
            "POST",
            f"/api/v1/writing/manuscripts/plot-lines/{plot_line_id}/events",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return ManuscriptPlotEventResponse.model_validate(response)

    async def list_manuscript_plot_events(self, plot_line_id: str) -> list[ManuscriptPlotEventResponse]:
        response = await self._request("GET", f"/api/v1/writing/manuscripts/plot-lines/{plot_line_id}/events")
        return [ManuscriptPlotEventResponse.model_validate(item) for item in list(response or [])]

    async def update_manuscript_plot_event(
        self,
        plot_event_id: str,
        request_data: ManuscriptPlotEventUpdate,
        *,
        expected_version: int,
    ) -> ManuscriptPlotEventResponse:
        response = await self._request(
            "PATCH",
            f"/api/v1/writing/manuscripts/plot-events/{plot_event_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
            headers={"expected-version": str(expected_version)},
        )
        return ManuscriptPlotEventResponse.model_validate(response)

    async def delete_manuscript_plot_event(self, plot_event_id: str, *, expected_version: int) -> bool:
        await self._request(
            "DELETE",
            f"/api/v1/writing/manuscripts/plot-events/{plot_event_id}",
            headers={"expected-version": str(expected_version)},
        )
        return True

    async def create_manuscript_plot_hole(
        self,
        project_id: str,
        request_data: ManuscriptPlotHoleCreate,
    ) -> ManuscriptPlotHoleResponse:
        response = await self._request(
            "POST",
            f"/api/v1/writing/manuscripts/projects/{project_id}/plot-holes",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return ManuscriptPlotHoleResponse.model_validate(response)

    async def list_manuscript_plot_holes(
        self,
        project_id: str,
        *,
        status: Optional[str] = None,
    ) -> list[ManuscriptPlotHoleResponse]:
        response = await self._request(
            "GET",
            f"/api/v1/writing/manuscripts/projects/{project_id}/plot-holes",
            params={key: value for key, value in {"status": status}.items() if value is not None},
        )
        return [ManuscriptPlotHoleResponse.model_validate(item) for item in list(response or [])]

    async def update_manuscript_plot_hole(
        self,
        plot_hole_id: str,
        request_data: ManuscriptPlotHoleUpdate,
        *,
        expected_version: int,
    ) -> ManuscriptPlotHoleResponse:
        response = await self._request(
            "PATCH",
            f"/api/v1/writing/manuscripts/plot-holes/{plot_hole_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
            headers={"expected-version": str(expected_version)},
        )
        return ManuscriptPlotHoleResponse.model_validate(response)

    async def delete_manuscript_plot_hole(self, plot_hole_id: str, *, expected_version: int) -> bool:
        await self._request(
            "DELETE",
            f"/api/v1/writing/manuscripts/plot-holes/{plot_hole_id}",
            headers={"expected-version": str(expected_version)},
        )
        return True

    async def link_manuscript_scene_character(
        self,
        scene_id: str,
        request_data: SceneCharacterLink,
    ) -> list[SceneCharacterLinkResponse]:
        response = await self._request(
            "POST",
            f"/api/v1/writing/manuscripts/scenes/{scene_id}/characters",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return [SceneCharacterLinkResponse.model_validate(item) for item in list(response or [])]

    async def list_manuscript_scene_characters(self, scene_id: str) -> list[SceneCharacterLinkResponse]:
        response = await self._request("GET", f"/api/v1/writing/manuscripts/scenes/{scene_id}/characters")
        return [SceneCharacterLinkResponse.model_validate(item) for item in list(response or [])]

    async def unlink_manuscript_scene_character(self, scene_id: str, character_id: str) -> bool:
        await self._request(
            "DELETE",
            f"/api/v1/writing/manuscripts/scenes/{scene_id}/characters/{character_id}",
        )
        return True

    async def link_manuscript_scene_world_info(
        self,
        scene_id: str,
        request_data: SceneWorldInfoLink,
    ) -> list[SceneWorldInfoLinkResponse]:
        response = await self._request(
            "POST",
            f"/api/v1/writing/manuscripts/scenes/{scene_id}/world-info",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return [SceneWorldInfoLinkResponse.model_validate(item) for item in list(response or [])]

    async def list_manuscript_scene_world_info(self, scene_id: str) -> list[SceneWorldInfoLinkResponse]:
        response = await self._request("GET", f"/api/v1/writing/manuscripts/scenes/{scene_id}/world-info")
        return [SceneWorldInfoLinkResponse.model_validate(item) for item in list(response or [])]

    async def unlink_manuscript_scene_world_info(self, scene_id: str, world_info_id: str) -> bool:
        await self._request(
            "DELETE",
            f"/api/v1/writing/manuscripts/scenes/{scene_id}/world-info/{world_info_id}",
        )
        return True

    async def create_manuscript_citation(
        self,
        scene_id: str,
        request_data: ManuscriptCitationCreate,
    ) -> ManuscriptCitationResponse:
        response = await self._request(
            "POST",
            f"/api/v1/writing/manuscripts/scenes/{scene_id}/citations",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return ManuscriptCitationResponse.model_validate(response)

    async def list_manuscript_citations(self, scene_id: str) -> list[ManuscriptCitationResponse]:
        response = await self._request("GET", f"/api/v1/writing/manuscripts/scenes/{scene_id}/citations")
        return [ManuscriptCitationResponse.model_validate(item) for item in list(response or [])]

    async def delete_manuscript_citation(self, citation_id: str, *, expected_version: int) -> bool:
        await self._request(
            "DELETE",
            f"/api/v1/writing/manuscripts/citations/{citation_id}",
            headers={"expected-version": str(expected_version)},
        )
        return True

    async def research_manuscript_scene(
        self,
        scene_id: str,
        request_data: ManuscriptResearchRequest,
    ) -> ManuscriptResearchResponse:
        response = await self._request(
            "POST",
            f"/api/v1/writing/manuscripts/scenes/{scene_id}/research",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return ManuscriptResearchResponse.model_validate(response)

    async def analyze_manuscript_scene(
        self,
        scene_id: str,
        request_data: ManuscriptAnalysisRequest,
    ) -> list[ManuscriptAnalysisResponse]:
        response = await self._request(
            "POST",
            f"/api/v1/writing/manuscripts/scenes/{scene_id}/analyze",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return [ManuscriptAnalysisResponse.model_validate(item) for item in list(response or [])]

    async def analyze_manuscript_chapter(
        self,
        chapter_id: str,
        request_data: ManuscriptAnalysisRequest,
    ) -> list[ManuscriptAnalysisResponse]:
        response = await self._request(
            "POST",
            f"/api/v1/writing/manuscripts/chapters/{chapter_id}/analyze",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return [ManuscriptAnalysisResponse.model_validate(item) for item in list(response or [])]

    async def analyze_manuscript_project_plot_holes(
        self,
        project_id: str,
        request_data: ManuscriptAnalysisRequest,
    ) -> list[ManuscriptAnalysisResponse]:
        response = await self._request(
            "POST",
            f"/api/v1/writing/manuscripts/projects/{project_id}/analyze/plot-holes",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return [ManuscriptAnalysisResponse.model_validate(item) for item in list(response or [])]

    async def analyze_manuscript_project_consistency(
        self,
        project_id: str,
        request_data: ManuscriptAnalysisRequest,
    ) -> list[ManuscriptAnalysisResponse]:
        response = await self._request(
            "POST",
            f"/api/v1/writing/manuscripts/projects/{project_id}/analyze/consistency",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return [ManuscriptAnalysisResponse.model_validate(item) for item in list(response or [])]

    async def list_manuscript_analyses(
        self,
        project_id: str,
        *,
        scope_type: Optional[str] = None,
        analysis_type: Optional[str] = None,
        include_stale: bool = False,
    ) -> ManuscriptAnalysisListResponse:
        response = await self._request(
            "GET",
            f"/api/v1/writing/manuscripts/projects/{project_id}/analyses",
            params={
                key: value
                for key, value in {
                    "scope_type": scope_type,
                    "analysis_type": analysis_type,
                    "include_stale": include_stale,
                }.items()
                if value is not None
            },
        )
        return ManuscriptAnalysisListResponse.model_validate(response)

    async def create_quiz(
        self,
        request_data: QuizCreateRequest,
    ) -> QuizResponse:
        response = await self._request(
            "POST",
            "/api/v1/quizzes",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return QuizResponse.model_validate(response)

    async def list_quizzes(
        self,
        *,
        q: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[QuizResponse]:
        response = await self._request(
            "GET",
            "/api/v1/quizzes",
            params={
                key: value
                for key, value in {
                    "q": q,
                    "limit": limit,
                    "offset": offset,
                }.items()
                if value is not None
            },
        )
        payload = QuizListResponse.model_validate(response)
        return payload.items

    async def get_quiz(self, quiz_id: int | str) -> QuizResponse:
        response = await self._request("GET", f"/api/v1/quizzes/{quiz_id}")
        return QuizResponse.model_validate(response)

    async def update_quiz(
        self,
        quiz_id: int | str,
        request_data: QuizUpdateRequest,
    ) -> QuizResponse:
        response = await self._request(
            "PATCH",
            f"/api/v1/quizzes/{quiz_id}",
            json_data=request_data.model_dump(exclude_none=True, exclude_unset=True, mode="json"),
        )
        return QuizResponse.model_validate(response)

    async def delete_quiz(
        self,
        quiz_id: int | str,
        *,
        expected_version: Optional[int] = None,
        hard: bool = False,
    ) -> Dict[str, Any]:
        return await self._request(
            "DELETE",
            f"/api/v1/quizzes/{quiz_id}",
            params={
                "expected_version": expected_version,
                "hard": hard,
            },
        )

    async def create_quiz_question(
        self,
        quiz_id: int | str,
        request_data: QuizQuestionCreateRequest,
    ) -> QuizQuestionResponse:
        response = await self._request(
            "POST",
            f"/api/v1/quizzes/{quiz_id}/questions",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return QuizQuestionResponse.model_validate(response)

    async def list_quiz_questions(
        self,
        quiz_id: int | str,
        *,
        q: Optional[str] = None,
        include_answers: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> QuizQuestionListResponse:
        response = await self._request(
            "GET",
            f"/api/v1/quizzes/{quiz_id}/questions",
            params={
                key: value
                for key, value in {
                    "q": q,
                    "include_answers": include_answers,
                    "limit": limit,
                    "offset": offset,
                }.items()
                if value is not None
            },
        )
        return QuizQuestionListResponse.model_validate(response)

    async def update_quiz_question(
        self,
        quiz_id: int | str,
        question_id: int | str,
        request_data: QuizQuestionUpdateRequest | Dict[str, Any],
    ) -> QuizQuestionResponse:
        payload = (
            request_data.model_dump(exclude_none=True, exclude_unset=True, mode="json")
            if hasattr(request_data, "model_dump")
            else dict(request_data)
        )
        response = await self._request(
            "PATCH",
            f"/api/v1/quizzes/{quiz_id}/questions/{question_id}",
            json_data=payload,
        )
        return QuizQuestionResponse.model_validate(response)

    async def delete_quiz_question(
        self,
        quiz_id: int | str,
        question_id: int | str,
        *,
        expected_version: Optional[int] = None,
        hard: bool = False,
    ) -> Dict[str, Any]:
        return await self._request(
            "DELETE",
            f"/api/v1/quizzes/{quiz_id}/questions/{question_id}",
            params={
                "expected_version": expected_version,
                "hard": hard,
            },
        )

    async def start_quiz_attempt(self, quiz_id: int | str) -> QuizAttemptResponse:
        response = await self._request(
            "POST",
            f"/api/v1/quizzes/{quiz_id}/attempts",
        )
        return QuizAttemptResponse.model_validate(response)

    async def submit_quiz_attempt(
        self,
        attempt_id: int | str,
        request_data: QuizAttemptSubmitRequest,
    ) -> QuizAttemptResponse:
        response = await self._request(
            "PUT",
            f"/api/v1/quizzes/attempts/{attempt_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return QuizAttemptResponse.model_validate(response)

    async def list_quiz_attempts(
        self,
        *,
        quiz_id: Optional[int | str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> QuizAttemptListResponse:
        response = await self._request(
            "GET",
            "/api/v1/quizzes/attempts",
            params={
                key: value
                for key, value in {
                    "quiz_id": quiz_id,
                    "limit": limit,
                    "offset": offset,
                }.items()
                if value is not None
            },
        )
        return QuizAttemptListResponse.model_validate(response)

    async def get_quiz_attempt(
        self,
        attempt_id: int | str,
        *,
        include_questions: bool = False,
        include_answers: bool = False,
    ) -> QuizAttemptResponse:
        response = await self._request(
            "GET",
            f"/api/v1/quizzes/attempts/{attempt_id}",
            params={
                "include_questions": include_questions,
                "include_answers": include_answers,
            },
        )
        return QuizAttemptResponse.model_validate(response)

    async def search_media_items(
        self,
        request_data: MediaSearchRequest,
        page: int = 1,
        results_per_page: int = 10,
    ) -> ServerMediaListResponse:
        response = await self._request(
            "POST",
            "/api/v1/media/search",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
            params={"page": page, "results_per_page": results_per_page},
        )
        return ServerMediaListResponse.model_validate(response)

    async def add_media(
        self,
        request_data: AddMediaRequest,
        file_paths: Optional[List[str]] = None,
    ) -> BatchMediaProcessResponse:
        form_data = model_to_form_data(request_data)
        if file_paths:
            form_data.update(
                {
                    key: form_data.get(key, value)
                    for key, value in {
                        "overwrite_existing": "false",
                        "perform_analysis": "true",
                        "use_cookies": "false",
                        "perform_rolling_summarization": "false",
                        "summarize_recursively": "false",
                        "perform_chunking": "true",
                        "use_adaptive_chunking": "false",
                        "use_multi_level_chunking": "false",
                        "chunk_size": "500",
                        "chunk_overlap": "200",
                    }.items()
                }
            )
        httpx_files = prepare_files_for_httpx(file_paths, upload_field_name="files")
        try:
            response_dict = await self._request(
                "POST", "/api/v1/media/add", data=form_data, files=httpx_files
            )
            return BatchMediaProcessResponse(**response_dict)
        finally:
            cleanup_file_objects(httpx_files)

    async def process_video(self, request_data: ProcessVideoRequest, file_paths: Optional[List[str]] = None) -> BatchMediaProcessResponse:
        form_data = model_to_form_data(request_data)
        httpx_files = prepare_files_for_httpx(file_paths, upload_field_name="files")
        try:
            response_dict = await self._request("POST", "/api/v1/media/process-videos", data=form_data, files=httpx_files)
            return BatchMediaProcessResponse(**response_dict)
        finally:
            cleanup_file_objects(httpx_files)

    async def process_audio(self, request_data: ProcessAudioRequest, file_paths: Optional[List[str]] = None) -> BatchMediaProcessResponse:
        form_data = model_to_form_data(request_data)
        httpx_files = prepare_files_for_httpx(file_paths, upload_field_name="files")
        try:
            response_dict = await self._request("POST", "/api/v1/media/process-audios", data=form_data, files=httpx_files)
            return BatchMediaProcessResponse(**response_dict)
        finally:
            cleanup_file_objects(httpx_files)

    async def process_pdf(self, request_data: ProcessPDFRequest, file_paths: Optional[List[str]] = None) -> BatchMediaProcessResponse:
        form_data = model_to_form_data(request_data)
        httpx_files = prepare_files_for_httpx(file_paths, upload_field_name="files")
        try:
            response_dict = await self._request("POST", "/api/v1/media/process-pdfs", data=form_data, files=httpx_files)
            return BatchMediaProcessResponse(**response_dict)
        finally:
            cleanup_file_objects(httpx_files)

    async def process_ebook(self, request_data: ProcessEbookRequest, file_paths: Optional[List[str]] = None) -> BatchMediaProcessResponse:
        form_data = model_to_form_data(request_data)
        httpx_files = prepare_files_for_httpx(file_paths, upload_field_name="files")
        try:
            response_dict = await self._request("POST", "/api/v1/media/process-ebooks", data=form_data, files=httpx_files)
            return BatchMediaProcessResponse(**response_dict)
        finally:
            cleanup_file_objects(httpx_files)

    async def process_document(self, request_data: ProcessDocumentRequest, file_paths: Optional[List[str]] = None) -> BatchMediaProcessResponse:
        form_data = model_to_form_data(request_data)
        httpx_files = prepare_files_for_httpx(file_paths, upload_field_name="files")
        try:
            response_dict = await self._request("POST", "/api/v1/media/process-documents", data=form_data, files=httpx_files)
            return BatchMediaProcessResponse(**response_dict)
        finally:
            cleanup_file_objects(httpx_files)

    async def process_code(self, request_data: ProcessCodeRequest, file_paths: Optional[List[str]] = None) -> BatchMediaProcessResponse:
        form_data = model_to_form_data(request_data)
        httpx_files = prepare_files_for_httpx(file_paths, upload_field_name="files")
        try:
            response_dict = await self._request("POST", "/api/v1/media/process-code", data=form_data, files=httpx_files)
            return BatchMediaProcessResponse(**response_dict)
        finally:
            cleanup_file_objects(httpx_files)

    async def process_email(self, request_data: ProcessEmailRequest, file_paths: Optional[List[str]] = None) -> BatchMediaProcessResponse:
        form_data = model_to_form_data(request_data)
        httpx_files = prepare_files_for_httpx(file_paths, upload_field_name="files")
        try:
            response_dict = await self._request("POST", "/api/v1/media/process-emails", data=form_data, files=httpx_files)
            return BatchMediaProcessResponse(**response_dict)
        finally:
            cleanup_file_objects(httpx_files)

    async def process_plaintext(self, request_data: ProcessPlaintextRequest, file_paths: Optional[List[str]] = None) -> BatchMediaProcessResponse:
        form_data = model_to_form_data(request_data)
        httpx_files = prepare_files_for_httpx(file_paths, upload_field_name="files")
        try:
            response_dict = await self._request("POST", "/api/v1/media/process-plaintext", data=form_data, files=httpx_files)
            return BatchMediaProcessResponse(**response_dict)
        finally:
            cleanup_file_objects(httpx_files)

    async def process_code(
        self,
        request_data: ProcessCodeRequest,
        file_paths: Optional[List[str]] = None,
    ) -> BatchMediaProcessResponse:
        form_data = model_to_form_data(request_data)
        httpx_files = prepare_files_for_httpx(file_paths, upload_field_name="files")
        try:
            response_dict = await self._request(
                "POST",
                "/api/v1/media/process-code",
                data=form_data,
                files=httpx_files,
            )
            return BatchMediaProcessResponse(**response_dict)
        finally:
            cleanup_file_objects(httpx_files)

    async def process_emails(
        self,
        request_data: ProcessEmailsRequest,
        file_paths: Optional[List[str]] = None,
    ) -> BatchMediaProcessResponse:
        form_data = model_to_form_data(request_data)
        httpx_files = prepare_files_for_httpx(file_paths, upload_field_name="files")
        try:
            response_dict = await self._request(
                "POST",
                "/api/v1/media/process-emails",
                data=form_data,
                files=httpx_files,
            )
            return BatchMediaProcessResponse(**response_dict)
        finally:
            cleanup_file_objects(httpx_files)

    async def process_web_scraping(
        self,
        request_data: ProcessWebScrapingRequest,
    ) -> IngestWebContentResponse:
        response = await self._request(
            "POST",
            "/api/v1/media/process-web-scraping",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return IngestWebContentResponse.model_validate(response)

    async def get_transcription_models(self) -> Dict[str, Any]:
        return await self._request("GET", "/api/v1/media/transcription-models")

    async def get_web_scraping_status(self) -> Dict[str, Any]:
        return await self._request("GET", "/api/v1/web-scraping/status")

    async def get_web_scraping_job_status(self, job_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/api/v1/web-scraping/job/{job_id}")

    async def cancel_web_scraping_job(self, job_id: str) -> Dict[str, Any]:
        return await self._request("DELETE", f"/api/v1/web-scraping/job/{job_id}")

    async def get_web_scraping_progress(self, task_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/api/v1/web-scraping/progress/{task_id}")

    async def get_web_scraping_cookies(self, domain: str) -> Dict[str, Any]:
        return await self._request("GET", f"/api/v1/web-scraping/cookies/{domain}")

    async def set_web_scraping_cookies(
        self,
        domain: str,
        cookies: list[dict[str, Any]],
    ) -> Dict[str, Any]:
        return await self._request(
            "POST",
            f"/api/v1/web-scraping/cookies/{domain}",
            json_data=cookies,
        )

    async def check_web_scraping_duplicate(self, url: str) -> Dict[str, Any]:
        return await self._request(
            "GET",
            "/api/v1/web-scraping/duplicates/check",
            params={"url": url},
        )

    async def process_xml(self, request_data: ProcessXMLRequest, file_path: str) -> BatchProcessXMLResponse: # XML expects single file
        form_data = model_to_form_data(request_data) # XMLIngestRequest becomes form data for 'payload'
        httpx_files = prepare_files_for_httpx([file_path], upload_field_name="file")
        try:
            # The XML endpoint expects 'payload' as a form field for the JSON data and 'file' for the file.
            # This might require custom request construction if httpx doesn't handle nested form data well.
            # Let's assume server expects payload fields flat, or adjust server.
            # For now, sending request_data fields as top-level form data alongside the file.
            response_dict = await self._request("POST", "/api/v1/media/process-xml", data=form_data, files=httpx_files) # Assuming route from Gradio
            # The actual response from /process-xml is a single item, not batch. Adjusting.
            # This is a placeholder, actual response structure for XML needs to be confirmed and modeled in schemas.py.
            # The Gradio endpoint returns a dict like {"status": "...", "media_id": "...", "title": "..."}.
            # For consistency, wrap it in BatchProcessXMLResponse structure.
            if response_dict and "status" in response_dict:
                 single_item_result = ProcessXMLResponseItem(
                    status=response_dict.get("status", "Error"),
                    input_ref=Path(file_path).name, # Use filename as input_ref
                    title=response_dict.get("title"),
                    # Populate other fields if process_xml_task returns them and they are in ProcessXMLResponseItem
                    author=request_data.author, # from input
                    keywords=request_data.keywords, # from input
                    content=response_dict.get("content"), # Assuming these might come from a more detailed response
                    summary=response_dict.get("summary"),
                    segments=response_dict.get("segments")
                )
                 return BatchProcessXMLResponse(
                    processed_count=1 if single_item_result.status not in ["Error"] else 0,
                    errors_count=1 if single_item_result.status == "Error" or single_item_result.error else 0,
                    errors=[single_item_result.error] if single_item_result.error else [],
                    results=[single_item_result]
                )
            raise APIResponseError(500, "Invalid response structure from XML processing", response_data=response_dict)
        finally:
            cleanup_file_objects(httpx_files)


    async def process_mediawiki_dump(
        self,
        request_data: ProcessMediaWikiRequest,
        dump_file_path: str
    ) -> AsyncGenerator[ProcessedMediaWikiPage, None]:
        form_data = model_to_form_data(request_data) # Handles wiki_name, namespaces_str etc.
        httpx_files = prepare_files_for_httpx([dump_file_path], upload_field_name="dump_file")

        try:
            async for item_dict in self._stream_request(
                "POST", "/api/v1/media/mediawiki/process-dump", data=form_data, files=httpx_files
            ):
                # Assuming each yielded item from the stream is a dict that can be parsed
                # into ProcessedMediaWikiPage or an error/progress event.
                # The client should decide how to handle non-page events (e.g. "summary", "error")
                if item_dict.get("type") == "item_result" and "data" in item_dict:
                    page_data = item_dict["data"]
                    page_data["input_ref"] = Path(dump_file_path).name # Add input_ref for client tracking
                    yield ProcessedMediaWikiPage(**page_data)
                elif "title" in item_dict and "content" in item_dict:
                    page_data = dict(item_dict)
                    page_data["input_ref"] = Path(dump_file_path).name
                    yield ProcessedMediaWikiPage(**page_data)
                elif item_dict.get("type") == "validation_error":
                    # Yield a ProcessedMediaWikiPage with error status for validation errors
                    yield ProcessedMediaWikiPage(
                        title=item_dict.get("title", "Unknown Page - Validation Error"),
                        content="", # No content on validation error
                        status="Error",
                        error_message=f"Validation Error: {item_dict.get('detail')}",
                        input_ref=Path(dump_file_path).name
                    )
                elif item_dict.get("type") == "error":
                     yield ProcessedMediaWikiPage(
                        title=item_dict.get("title", "Unknown Page - Processing Error"),
                        content="",
                        status="Error",
                        error_message=item_dict.get("message", "Unknown processing error"),
                        input_ref=Path(dump_file_path).name
                    )
                # Can add handling for "progress_total" and "summary" if needed by UI
                # For now, only yield processed pages or page-level errors
        finally:
            cleanup_file_objects(httpx_files)

    async def ingest_mediawiki_dump(
        self,
        request_data: ProcessMediaWikiRequest,
        dump_file_path: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        form_data = model_to_form_data(request_data)
        httpx_files = prepare_files_for_httpx([dump_file_path], upload_field_name="dump_file")

        try:
            async for item_dict in self._stream_request(
                "POST", "/api/v1/media/mediawiki/ingest-dump", data=form_data, files=httpx_files
            ):
                yield item_dict
        finally:
            cleanup_file_objects(httpx_files)

    async def list_prompts(self, include_deleted: bool = False) -> Dict[str, Any]:
        return await self._request(
            "GET",
            "/api/v1/prompts",
            params={
                "page": page,
                "per_page": per_page,
                "include_deleted": str(include_deleted).lower(),
                "sort_by": sort_by,
                "sort_order": sort_order,
            },
        )
        return PaginatedPromptsResponse.model_validate(response)

    async def preview_prompt(self, request_data: PromptPreviewRequest) -> Dict[str, Any]:
        return await self._request(
            "POST",
            "/api/v1/prompts/preview",
            json_data=request_data.model_dump(exclude_none=True),
        )

    async def create_prompt(self, request_data: PromptCreateRequest) -> PromptResponse:
        response = await self._request(
            "POST",
            "/api/v1/prompts",
            json_data=request_data.model_dump(exclude_none=True),
        )
        return PromptResponse.model_validate(response)

    async def get_prompt(self, prompt_identifier: Union[str, int], include_deleted: bool = False) -> PromptResponse:
        response = await self._request(
            "GET",
            f"/api/v1/prompts/{prompt_identifier}",
            params={"include_deleted": str(include_deleted).lower()},
        )
        return PromptResponse.model_validate(response)

    async def update_prompt(self, prompt_identifier: Union[str, int], request_data: PromptCreateRequest) -> PromptResponse:
        response = await self._request(
            "PUT",
            f"/api/v1/prompts/{prompt_identifier}",
            json_data=request_data.model_dump(exclude_none=True),
        )
        return PromptResponse.model_validate(response)

    async def record_prompt_usage(self, prompt_identifier: Union[str, int]) -> PromptResponse:
        response = await self._request(
            "POST",
            f"/api/v1/prompts/{prompt_identifier}/use",
        )
        return PromptResponse.model_validate(response)

    async def delete_prompt(self, prompt_identifier: Union[str, int]) -> Dict[str, Any]:
        await self._request(
            "DELETE",
            f"/api/v1/prompts/{prompt_identifier}",
        )
        return {}

    async def update_prompt(self, prompt_identifier: Union[str, int], request_data: PromptCreateRequest) -> Dict[str, Any]:
        return await self._request(
            "PUT",
            f"/api/v1/prompts/{prompt_identifier}",
            json_data=request_data.model_dump(exclude_none=True, exclude_unset=True),
        )

    async def delete_prompt(self, prompt_identifier: Union[str, int]) -> Dict[str, Any]:
        return await self._request(
            "DELETE",
            f"/api/v1/prompts/{prompt_identifier}",
        )

    async def get_prompts_health(self) -> Dict[str, Any]:
        return await self._request("GET", "/api/v1/prompts/health")

    async def get_prompt_sync_log(self, *, since_change_id: int = 0, limit: int = 100) -> Dict[str, Any]:
        return await self._request(
            "GET",
            "/api/v1/prompts/sync-log",
            params={"since_change_id": since_change_id, "limit": limit},
        )

    async def search_prompts(
        self,
        *,
        search_query: str,
        search_fields: Optional[List[str]] = None,
        page: int = 1,
        results_per_page: int = 20,
        include_deleted: bool = False,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "search_query": search_query,
            "page": page,
            "results_per_page": results_per_page,
            "include_deleted": str(include_deleted).lower(),
        }
        if search_fields is not None:
            params["search_fields"] = search_fields
        return await self._request("POST", "/api/v1/prompts/search", params=params)

    async def create_prompt_keyword(self, keyword_text: str) -> Dict[str, Any]:
        return await self._request(
            "POST",
            "/api/v1/prompts/keywords/",
            json_data={"keyword_text": keyword_text},
        )

    async def list_prompt_keywords(self) -> List[str]:
        return await self._request("GET", "/api/v1/prompts/keywords/")

    async def delete_prompt_keyword(self, keyword_text: str) -> Dict[str, Any]:
        return await self._request("DELETE", f"/api/v1/prompts/keywords/{quote(keyword_text, safe='')}")

    async def export_prompts(
        self,
        *,
        export_format: Literal["csv", "markdown"] = "csv",
        filter_keywords: Optional[List[str]] = None,
        include_system: bool = True,
        include_user: bool = True,
        include_details: bool = True,
        include_author: bool = True,
        include_associated_keywords: bool = True,
        markdown_template_name: Optional[str] = "Basic Template",
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "export_format": export_format,
            "include_system": str(include_system).lower(),
            "include_user": str(include_user).lower(),
            "include_details": str(include_details).lower(),
            "include_author": str(include_author).lower(),
            "include_associated_keywords": str(include_associated_keywords).lower(),
        }
        if filter_keywords is not None:
            params["filter_keywords"] = filter_keywords
        if markdown_template_name is not None:
            params["markdown_template_name"] = markdown_template_name
        return await self._request("GET", "/api/v1/prompts/export", params=params)

    async def export_prompt_keywords(self) -> Dict[str, Any]:
        return await self._request("GET", "/api/v1/prompts/keywords/export-csv")

    async def import_prompts(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self._request("POST", "/api/v1/prompts/import", json_data=payload)

    async def extract_prompt_template_variables(self, template: str) -> Dict[str, Any]:
        return await self._request(
            "POST",
            "/api/v1/prompts/templates/variables",
            json_data={"template": template},
        )

    async def render_prompt_template(self, template: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        return await self._request(
            "POST",
            "/api/v1/prompts/templates/render",
            json_data={"template": template, "variables": variables},
        )

    async def convert_prompt(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self._request("POST", "/api/v1/prompts/convert", json_data=payload)

    async def bulk_delete_prompts(self, prompt_ids: List[int]) -> Dict[str, Any]:
        return await self._request(
            "POST",
            "/api/v1/prompts/bulk/delete",
            json_data={"prompt_ids": prompt_ids},
        )

    async def bulk_update_prompt_keywords(
        self,
        prompt_ids: List[int],
        keywords: Optional[List[str]] = None,
        *,
        mode: Literal["add", "remove", "replace"] = "add",
        add_keywords: Optional[List[str]] = None,
        remove_keywords: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if add_keywords is None and remove_keywords is None:
            keyword_list = keywords or []
            if mode == "remove":
                add_keywords = []
                remove_keywords = keyword_list
            else:
                add_keywords = keyword_list
                remove_keywords = []
        return await self._request(
            "POST",
            "/api/v1/prompts/bulk/keywords",
            json_data={
                "prompt_ids": prompt_ids,
                "add_keywords": add_keywords or [],
                "remove_keywords": remove_keywords or [],
            },
        )

    async def record_prompt_usage(self, prompt_identifier: Union[str, int]) -> Dict[str, Any]:
        return await self._request("POST", f"/api/v1/prompts/{prompt_identifier}/use")

    async def create_prompt_collection(
        self,
        *,
        name: str,
        description: Optional[str] = None,
        prompt_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"name": name, "prompt_ids": prompt_ids or []}
        if description is not None:
            payload["description"] = description
        return await self._request("POST", "/api/v1/prompts/collections/create", json_data=payload)

    async def list_prompt_collections(self, *, limit: int = 200, offset: int = 0) -> Dict[str, Any]:
        return await self._request(
            "GET",
            "/api/v1/prompts/collections",
            params={"limit": limit, "offset": offset},
        )

    async def get_prompt_collection(self, collection_id: int) -> Dict[str, Any]:
        return await self._request("GET", f"/api/v1/prompts/collections/{collection_id}")

    async def update_prompt_collection(
        self,
        collection_id: int,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        prompt_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description
        if prompt_ids is not None:
            payload["prompt_ids"] = prompt_ids
        return await self._request("PUT", f"/api/v1/prompts/collections/{collection_id}", json_data=payload)

    async def call_server_characters_endpoint(
        self,
        method: str,
        endpoint: str,
        *,
        params: Dict[str, Any] | None = None,
        payload: Dict[str, Any] | list[Any] | None = None,
        data: Dict[str, Any] | None = None,
        files: list[tuple] | None = None,
        headers: Dict[str, str] | None = None,
    ) -> Any:
        return await self._call_server_api_namespace_endpoint(
            "characters",
            method,
            endpoint,
            params=params,
            payload=payload,
            data=data,
            files=files,
            headers=headers,
        )

    async def call_server_persona_endpoint(
        self,
        method: str,
        endpoint: str,
        *,
        params: Dict[str, Any] | None = None,
        payload: Dict[str, Any] | list[Any] | None = None,
        data: Dict[str, Any] | None = None,
        files: list[tuple] | None = None,
        headers: Dict[str, str] | None = None,
    ) -> Any:
        return await self._call_server_api_namespace_endpoint(
            "persona",
            method,
            endpoint,
            params=params,
            payload=payload,
            data=data,
            files=files,
            headers=headers,
        )

    async def query_characters(self, request_data: CharacterQueryRequest) -> Dict[str, Any]:
        return await self._request(
            "GET",
            "/api/v1/characters/query",
            params=request_data.model_dump(exclude_none=True, mode="json"),
        )

    async def list_characters(self, limit: int = 100, offset: int = 0) -> list[CharacterResponse]:
        return await self._request(
            "GET",
            "/api/v1/characters/",
            params={"limit": limit, "offset": offset},
        )

    async def search_characters(self, query: str, limit: int = 10) -> list[CharacterResponse]:
        return await self._request("GET", "/api/v1/characters/search/", params={"query": query, "limit": limit})

    async def get_character(self, character_id: int) -> Dict[str, Any]:
        return await self._request("GET", f"/api/v1/characters/{character_id}")

    async def create_character(self, request_data: CharacterCreateRequest) -> Dict[str, Any]:
        return await self._request(
            "POST",
            "/api/v1/characters/",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )

    async def update_character(self, character_id: int, request_data: CharacterUpdateRequest, expected_version: int) -> Dict[str, Any]:
        return await self._request(
            "PUT",
            f"/api/v1/characters/{character_id}",
            json_data=request_data.model_dump(exclude_unset=True, exclude_none=True, mode="json"),
            params={"expected_version": expected_version},
        )

    async def delete_character(self, character_id: int, expected_version: int) -> Dict[str, Any]:
        return await self._request(
            "DELETE",
            f"/api/v1/characters/{character_id}",
            params={"expected_version": expected_version},
        )

    async def restore_character(self, character_id: int, expected_version: int) -> Dict[str, Any]:
        return await self._request(
            "POST",
            f"/api/v1/characters/{character_id}/restore",
            params={"expected_version": expected_version},
        )

    async def get_character_exemplar(self, character_id: int, exemplar_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/api/v1/characters/{character_id}/exemplars/{exemplar_id}")

    async def create_character_exemplar(
        self,
        character_id: int,
        request_data: Union[CharacterExemplarCreate, List[CharacterExemplarCreate]],
    ) -> Dict[str, Any]:
        payload = (
            [item.model_dump(exclude_none=True, mode="json") for item in request_data]
            if isinstance(request_data, list)
            else request_data.model_dump(exclude_none=True, mode="json")
        )
        return await self._request(
            "POST",
            f"/api/v1/characters/{character_id}/exemplars",
            json_data=payload,
        )

    async def update_character_exemplar(
        self,
        character_id: int,
        exemplar_id: str,
        request_data: CharacterExemplarUpdate,
    ) -> Dict[str, Any]:
        return await self._request(
            "PUT",
            f"/api/v1/characters/{character_id}/exemplars/{exemplar_id}",
            json_data=request_data.model_dump(exclude_unset=True, exclude_none=True, mode="json"),
        )

    async def delete_character_exemplar(self, character_id: int, exemplar_id: str) -> Dict[str, Any]:
        return await self._request("DELETE", f"/api/v1/characters/{character_id}/exemplars/{exemplar_id}")

    async def search_character_exemplars(self, character_id: int, request_data: CharacterExemplarSearchRequest) -> Dict[str, Any]:
        return await self._request(
            "POST",
            f"/api/v1/characters/{character_id}/exemplars/search",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )

    async def select_character_exemplars_debug(
        self,
        character_id: int,
        request_data: CharacterExemplarSelectionDebugRequest,
    ) -> Dict[str, Any]:
        return await self._request(
            "POST",
            f"/api/v1/characters/{character_id}/exemplars/select/debug",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )

    async def list_persona_profiles(
        self,
        active_only: bool = False,
        include_deleted: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[PersonaProfileResponse]:
        params: Dict[str, Any] = {
            "active_only": str(active_only).lower(),
            "include_deleted": str(include_deleted).lower(),
            "limit": limit,
            "offset": offset,
        }
        return await self._request("GET", "/api/v1/persona/profiles", params=params)

    async def get_persona_profile(self, persona_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/api/v1/persona/profiles/{persona_id}")

    async def create_persona_profile(self, request_data: PersonaProfileCreate) -> Dict[str, Any]:
        return await self._request(
            "POST",
            "/api/v1/persona/profiles",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )

    async def update_persona_profile(
        self,
        persona_id: str,
        request_data: PersonaProfileUpdate,
        expected_version: Optional[int] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if expected_version is not None:
            params["expected_version"] = expected_version
        return await self._request(
            "PATCH",
            f"/api/v1/persona/profiles/{persona_id}",
            json_data=request_data.model_dump(exclude_unset=True, exclude_none=True, mode="json"),
            params=params,
        )

    async def delete_persona_profile(self, persona_id: str, expected_version: Optional[int] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if expected_version is not None:
            params["expected_version"] = expected_version
        return await self._request("DELETE", f"/api/v1/persona/profiles/{persona_id}", params=params)

    async def restore_persona_profile(self, persona_id: str, expected_version: int) -> Dict[str, Any]:
        return await self._request(
            "POST",
            f"/api/v1/persona/profiles/{persona_id}/restore",
            params={"expected_version": expected_version},
        )

    async def list_persona_archetypes(self) -> list[ArchetypeSummary]:
        response = await self._request("GET", "/api/v1/persona/archetypes")
        return [ArchetypeSummary.model_validate(item) for item in response]

    async def get_persona_archetype(self, key: str) -> ArchetypeTemplate:
        response = await self._request("GET", f"/api/v1/persona/archetypes/{quote(key, safe='')}")
        return ArchetypeTemplate.model_validate(response)

    async def preview_persona_archetype(self, key: str) -> ArchetypePreviewResponse:
        response = await self._request("GET", f"/api/v1/persona/archetypes/{quote(key, safe='')}/preview")
        return ArchetypePreviewResponse.model_validate(response)

    async def list_persona_exemplars(
        self,
        persona_id: str,
        include_disabled: bool = False,
        include_deleted: bool = False,
        include_deleted_personas: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[PersonaExemplarResponse]:
        params: Dict[str, Any] = {
            "include_disabled": str(include_disabled).lower(),
            "include_deleted": str(include_deleted).lower(),
            "include_deleted_personas": str(include_deleted_personas).lower(),
            "limit": limit,
            "offset": offset,
        }
        return await self._request("GET", f"/api/v1/persona/profiles/{persona_id}/exemplars", params=params)

    async def get_persona_exemplar(self, persona_id: str, exemplar_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/api/v1/persona/profiles/{persona_id}/exemplars/{exemplar_id}")

    async def create_persona_exemplar(self, persona_id: str, request_data: PersonaExemplarCreate) -> Dict[str, Any]:
        return await self._request(
            "POST",
            f"/api/v1/persona/profiles/{persona_id}/exemplars",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )

    async def import_persona_exemplars(self, persona_id: str, request_data: PersonaExemplarImportRequest) -> Dict[str, Any]:
        return await self._request(
            "POST",
            f"/api/v1/persona/profiles/{persona_id}/exemplars/import",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )

    async def update_persona_exemplar(
        self,
        persona_id: str,
        exemplar_id: str,
        request_data: PersonaExemplarUpdate,
    ) -> Dict[str, Any]:
        return await self._request(
            "PATCH",
            f"/api/v1/persona/profiles/{persona_id}/exemplars/{exemplar_id}",
            json_data=request_data.model_dump(exclude_unset=True, exclude_none=True, mode="json"),
        )

    async def review_persona_exemplar(
        self,
        persona_id: str,
        exemplar_id: str,
        request_data: PersonaExemplarReviewRequest,
    ) -> Dict[str, Any]:
        return await self._request(
            "POST",
            f"/api/v1/persona/profiles/{persona_id}/exemplars/{exemplar_id}/review",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )

    async def delete_persona_exemplar(self, persona_id: str, exemplar_id: str) -> Dict[str, Any]:
        return await self._request("DELETE", f"/api/v1/persona/profiles/{persona_id}/exemplars/{exemplar_id}")

    async def list_greetings(self, chat_id: str) -> GreetingListResponse:
        return await self._request("GET", f"/api/v1/chats/{chat_id}/greetings")

    async def select_greeting(self, chat_id: str, index: int) -> Dict[str, Any]:
        return await self._request(
            "PUT",
            f"/api/v1/chats/{chat_id}/greetings/select",
            json_data=GreetingSelectRequest(index=index).model_dump(mode="json"),
        )

    async def list_presets(self) -> PresetListResponse:
        return await self._request("GET", "/api/v1/chats/presets")

    async def create_preset(self, request_data: PresetCreate) -> Dict[str, Any]:
        return await self._request(
            "POST",
            "/api/v1/chats/presets",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )

    async def update_preset(self, preset_id: str, request_data: PresetUpdate) -> Dict[str, Any]:
        return await self._request(
            "PUT",
            f"/api/v1/chats/presets/{preset_id}",
            json_data=request_data.model_dump(exclude_unset=True, exclude_none=True, mode="json"),
        )

    async def delete_preset(self, preset_id: str) -> Dict[str, Any]:
        return await self._request("DELETE", f"/api/v1/chats/presets/{preset_id}")

    async def create_character_chat_session(
        self,
        request_data: CharacterChatSessionCreate | Dict[str, Any],
        seed_first_message: bool = False,
        greeting_strategy: Literal["default", "alternate_random", "alternate_index"] = "default",
        alternate_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "seed_first_message": str(seed_first_message).lower(),
            "greeting_strategy": greeting_strategy,
        }
        if alternate_index is not None:
            params["alternate_index"] = alternate_index
        return await self._request(
            "POST",
            "/api/v1/chats/",
            json_data=self._dump_request_payload(request_data, exclude_none=True),
            params=params,
        )

    async def list_character_chat_sessions(
        self,
        character_id: Optional[int] = None,
        character_scope: Literal["all", "character", "non_character"] = "all",
        limit: int = 50,
        offset: int = 0,
        include_deleted: bool = False,
        deleted_only: bool = False,
        include_settings: bool = False,
        scope_type: Optional[Literal["global", "workspace"]] = None,
        workspace_id: Optional[str] = None,
        include_message_counts: bool = True,
    ) -> Dict[str, Any]:
        scope_params = self._normalize_conversation_scope_params(scope_type=scope_type, workspace_id=workspace_id)
        params: Dict[str, Any] = {
            "character_scope": character_scope,
            "limit": limit,
            "offset": offset,
            "include_deleted": str(include_deleted).lower(),
            "deleted_only": str(deleted_only).lower(),
            "include_settings": str(include_settings).lower(),
            "include_message_counts": str(include_message_counts).lower(),
        }
        if character_id is not None:
            params["character_id"] = character_id
        if scope_params is not None:
            params.update(scope_params.model_dump(exclude_none=True, mode="json"))
        return await self._request("GET", "/api/v1/chats/", params=params)

    async def get_character_chat_session(
        self,
        chat_id: str,
        include_settings: bool = False,
        scope_type: Optional[Literal["global", "workspace"]] = None,
        workspace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        scope_params = self._normalize_conversation_scope_params(scope_type=scope_type, workspace_id=workspace_id)
        params: Dict[str, Any] = {"include_settings": str(include_settings).lower()}
        if scope_params is not None:
            params.update(scope_params.model_dump(exclude_none=True, mode="json"))
        return await self._request("GET", f"/api/v1/chats/{chat_id}", params=params)

    async def update_character_chat_session(
        self,
        chat_id: str,
        request_data: CharacterChatSessionUpdate | Dict[str, Any],
        expected_version: int,
        scope_type: Optional[Literal["global", "workspace"]] = None,
        workspace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        scope_params = self._normalize_conversation_scope_params(scope_type=scope_type, workspace_id=workspace_id)
        params: Dict[str, Any] = {"expected_version": expected_version}
        if scope_params is not None:
            params.update(scope_params.model_dump(exclude_none=True, mode="json"))
        return await self._request(
            "PUT",
            f"/api/v1/chats/{chat_id}",
            json_data=self._dump_request_payload(request_data, exclude_none=True, exclude_unset=True),
            params=params,
        )

    async def delete_character_chat_session(
        self,
        chat_id: str,
        expected_version: Optional[int] = None,
        hard_delete: bool = False,
        scope_type: Optional[Literal["global", "workspace"]] = None,
        workspace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        scope_params = self._normalize_conversation_scope_params(scope_type=scope_type, workspace_id=workspace_id)
        params: Dict[str, Any] = {"hard_delete": str(hard_delete).lower()}
        if expected_version is not None:
            params["expected_version"] = expected_version
        if scope_params is not None:
            params.update(scope_params.model_dump(exclude_none=True, mode="json"))
        return await self._request("DELETE", f"/api/v1/chats/{chat_id}", params=params)

    async def restore_character_chat_session(
        self,
        chat_id: str,
        expected_version: Optional[int] = None,
        scope_type: Optional[Literal["global", "workspace"]] = None,
        workspace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        scope_params = self._normalize_conversation_scope_params(scope_type=scope_type, workspace_id=workspace_id)
        params: Dict[str, Any] = {}
        if expected_version is not None:
            params["expected_version"] = expected_version
        if scope_params is not None:
            params.update(scope_params.model_dump(exclude_none=True, mode="json"))
        return await self._request("POST", f"/api/v1/chats/{chat_id}/restore", params=params or None)

    async def create_character_message(
        self,
        chat_id: str,
        request_data: CharacterMessageCreate | Dict[str, Any],
        scope_type: Optional[Literal["global", "workspace"]] = None,
        workspace_id: Optional[str] = None,
    ) -> CharacterMessageResponse:
        scope_params = self._normalize_conversation_scope_params(scope_type=scope_type, workspace_id=workspace_id)
        params = scope_params.model_dump(exclude_none=True, mode="json") if scope_params is not None else None
        response = await self._request(
            "POST",
            f"/api/v1/chats/{chat_id}/messages",
            json_data=self._dump_request_payload(request_data, exclude_none=True),
            params=params,
        )
        return CharacterMessageResponse.model_validate(response)

    async def list_character_messages(
        self,
        chat_id: str,
        limit: int = 50,
        offset: int = 0,
        include_deleted: bool = False,
        include_character_context: bool = False,
        format_for_completions: bool = False,
        include_tool_calls: bool = False,
        include_metadata: bool = False,
        include_message_ids: bool = False,
        scope_type: Optional[Literal["global", "workspace"]] = None,
        workspace_id: Optional[str] = None,
    ) -> CharacterMessageListResponse | Dict[str, Any]:
        scope_params = self._normalize_conversation_scope_params(scope_type=scope_type, workspace_id=workspace_id)
        params: Dict[str, Any] = {
            "limit": limit,
            "offset": offset,
            "include_deleted": str(include_deleted).lower(),
            "include_character_context": str(include_character_context).lower(),
            "format_for_completions": str(format_for_completions).lower(),
            "include_tool_calls": str(include_tool_calls).lower(),
            "include_metadata": str(include_metadata).lower(),
            "include_message_ids": str(include_message_ids).lower(),
        }
        if scope_params is not None:
            params.update(scope_params.model_dump(exclude_none=True, mode="json"))
        response = await self._request("GET", f"/api/v1/chats/{chat_id}/messages", params=params)
        if format_for_completions:
            return response
        return CharacterMessageListResponse.model_validate(response)

    async def get_character_message(
        self,
        message_id: str,
        include_tool_calls: bool = False,
        include_metadata: bool = False,
        scope_type: Optional[Literal["global", "workspace"]] = None,
        workspace_id: Optional[str] = None,
    ) -> CharacterMessageResponse:
        scope_params = self._normalize_conversation_scope_params(scope_type=scope_type, workspace_id=workspace_id)
        params: Dict[str, Any] = {
            "include_tool_calls": str(include_tool_calls).lower(),
            "include_metadata": str(include_metadata).lower(),
        }
        if scope_params is not None:
            params.update(scope_params.model_dump(exclude_none=True, mode="json"))
        response = await self._request("GET", f"/api/v1/messages/{message_id}", params=params)
        return CharacterMessageResponse.model_validate(response)

    async def update_character_message(
        self,
        message_id: str,
        request_data: CharacterMessageUpdate | Dict[str, Any],
        expected_version: int,
        scope_type: Optional[Literal["global", "workspace"]] = None,
        workspace_id: Optional[str] = None,
    ) -> CharacterMessageResponse:
        scope_params = self._normalize_conversation_scope_params(scope_type=scope_type, workspace_id=workspace_id)
        params: Dict[str, Any] = {"expected_version": expected_version}
        if scope_params is not None:
            params.update(scope_params.model_dump(exclude_none=True, mode="json"))
        response = await self._request(
            "PUT",
            f"/api/v1/messages/{message_id}",
            json_data=self._dump_request_payload(request_data, exclude_unset=True, exclude_none=True),
            params=params,
        )
        return CharacterMessageResponse.model_validate(response)

    async def delete_character_message(
        self,
        message_id: str,
        expected_version: int,
        scope_type: Optional[Literal["global", "workspace"]] = None,
        workspace_id: Optional[str] = None,
    ) -> None:
        scope_params = self._normalize_conversation_scope_params(scope_type=scope_type, workspace_id=workspace_id)
        params: Dict[str, Any] = {"expected_version": expected_version}
        if scope_params is not None:
            params.update(scope_params.model_dump(exclude_none=True, mode="json"))
        await self._request("DELETE", f"/api/v1/messages/{message_id}", params=params)

    async def search_character_messages(
        self,
        chat_id: str,
        query: str,
        limit: int = 50,
        scope_type: Optional[Literal["global", "workspace"]] = None,
        workspace_id: Optional[str] = None,
    ) -> CharacterMessageListResponse:
        scope_params = self._normalize_conversation_scope_params(scope_type=scope_type, workspace_id=workspace_id)
        params: Dict[str, Any] = {"query": query, "limit": limit}
        if scope_params is not None:
            params.update(scope_params.model_dump(exclude_none=True, mode="json"))
        response = await self._request("GET", f"/api/v1/chats/{chat_id}/messages/search", params=params)
        return CharacterMessageListResponse.model_validate(response)

    async def get_chat_settings(
        self,
        chat_id: str,
        scope_type: Optional[Literal["global", "workspace"]] = None,
        workspace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        scope_params = self._normalize_conversation_scope_params(scope_type=scope_type, workspace_id=workspace_id)
        params: Dict[str, Any] = {}
        if scope_params is not None:
            params.update(scope_params.model_dump(exclude_none=True, mode="json"))
        return await self._request("GET", f"/api/v1/chats/{chat_id}/settings", params=params or None)

    async def update_chat_settings(
        self,
        chat_id: str,
        request_data: ChatSettingsUpdate | Dict[str, Any],
        scope_type: Optional[Literal["global", "workspace"]] = None,
        workspace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        scope_params = self._normalize_conversation_scope_params(scope_type=scope_type, workspace_id=workspace_id)
        params: Dict[str, Any] = {}
        if scope_params is not None:
            params.update(scope_params.model_dump(exclude_none=True, mode="json"))
        return await self._request(
            "PUT",
            f"/api/v1/chats/{chat_id}/settings",
            json_data=self._dump_request_payload(request_data, exclude_none=True),
            params=params or None,
        )

    async def export_chat_history(
        self,
        chat_id: str,
        format: Literal["json", "markdown", "text"] = "json",
        include_metadata: bool = True,
        include_character: bool = True,
        page: int = 1,
        page_size: int = 1000,
    ) -> Dict[str, Any]:
        return await self._request(
            "GET",
            f"/api/v1/chats/{chat_id}/export",
            params={
                "format": format,
                "include_metadata": str(include_metadata).lower(),
                "include_character": str(include_character).lower(),
                "page": page,
                "page_size": page_size,
            },
        )

    async def get_author_note_info(self, chat_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/api/v1/chats/{chat_id}/author-note/info")

    async def export_lorebook_diagnostics(
        self,
        chat_id: str,
        page: int = 1,
        size: int = 50,
        order: Literal["asc", "desc"] = "asc",
    ) -> Dict[str, Any]:
        return await self._request(
            "GET",
            f"/api/v1/chats/{chat_id}/diagnostics/lorebook",
            params={"page": page, "size": size, "order": order},
        )

    async def list_chat_conversations(
        self,
        query: Optional[str] = None,
        state: Optional[str] = None,
        order_by: Literal["recency", "bm25", "hybrid", "topic"] = "recency",
        include_deleted: bool = False,
        deleted_only: bool = False,
        limit: int = 50,
        offset: int = 0,
        scope_type: Optional[Literal["global", "workspace"]] = None,
        workspace_id: Optional[str] = None,
        topic_label: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        cluster_id: Optional[str] = None,
        character_id: Optional[int] = None,
        character_scope: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        date_field: Literal["last_modified", "created_at"] = "last_modified",
    ) -> Dict[str, Any]:
        normalized_state = normalize_conversation_state(state)
        scope_params = self._normalize_conversation_scope_params(scope_type=scope_type, workspace_id=workspace_id)
        params: Dict[str, Any] = {
            "include_deleted": str(include_deleted).lower(),
            "deleted_only": str(deleted_only).lower(),
            "order_by": order_by,
            "limit": limit,
            "offset": offset,
            "date_field": date_field,
        }
        if query is not None:
            params["query"] = query
        if normalized_state is not None:
            params["state"] = normalized_state
        if scope_params is not None:
            params.update(scope_params.model_dump(exclude_none=True, mode="json"))
        if topic_label is not None:
            params["topic_label"] = topic_label
        if keywords is not None:
            params["keywords"] = keywords
        if cluster_id is not None:
            params["cluster_id"] = cluster_id
        if character_id is not None:
            params["character_id"] = character_id
        if character_scope is not None:
            params["character_scope"] = character_scope
        if start_date is not None:
            params["start_date"] = start_date
        if end_date is not None:
            params["end_date"] = end_date
        return await self._request("GET", "/api/v1/chat/conversations", params=params)

    async def get_chat_conversation(
        self,
        conversation_id: str,
        scope_type: Optional[Literal["global", "workspace"]] = None,
        workspace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        scope_params = self._normalize_conversation_scope_params(scope_type=scope_type, workspace_id=workspace_id)
        params: Dict[str, Any] = {}
        if scope_params is not None:
            params.update(scope_params.model_dump(exclude_none=True, mode="json"))
        return await self._request("GET", f"/api/v1/chat/conversations/{conversation_id}", params=params or None)

    async def update_chat_conversation(
        self,
        conversation_id: str,
        request_data: ConversationUpdateRequest,
        scope_type: Optional[Literal["global", "workspace"]] = None,
        workspace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        scope_params = self._normalize_conversation_scope_params(scope_type=scope_type, workspace_id=workspace_id)
        params: Dict[str, Any] = {}
        if scope_params is not None:
            params.update(scope_params.model_dump(exclude_none=True, mode="json"))
        return await self._request(
            "PATCH",
            f"/api/v1/chat/conversations/{conversation_id}",
            json_data=request_data.model_dump(exclude_none=True, exclude_unset=True, mode="json"),
            params=params or None,
        )

    async def get_chat_conversation_tree(
        self,
        conversation_id: str,
        limit: int = 50,
        offset: int = 0,
        max_depth: int = 4,
        scope_type: Optional[Literal["global", "workspace"]] = None,
        workspace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        scope_params = self._normalize_conversation_scope_params(scope_type=scope_type, workspace_id=workspace_id)
        params: Dict[str, Any] = {
            "limit": limit,
            "offset": offset,
            "max_depth": max_depth,
        }
        if scope_params is not None:
            params.update(scope_params.model_dump(exclude_none=True, mode="json"))
        return await self._request("GET", f"/api/v1/chat/conversations/{conversation_id}/tree", params=params)

    async def get_chat_conversation_messages_with_context(
        self,
        conversation_id: str,
        limit: int = 100,
        offset: int = 0,
        include_rag_context: bool = True,
        scope_type: Optional[Literal["global", "workspace"]] = None,
        workspace_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        scope_params = self._normalize_conversation_scope_params(scope_type=scope_type, workspace_id=workspace_id)
        params: Dict[str, Any] = {
            "limit": limit,
            "offset": offset,
            "include_rag_context": str(include_rag_context).lower(),
        }
        if scope_params is not None:
            params.update(scope_params.model_dump(exclude_none=True, mode="json"))
        return await self._request(
            "GET",
            f"/api/v1/chat/conversations/{conversation_id}/messages-with-context",
            params=params,
        )

    async def get_chat_conversation_citations(self, conversation_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/api/v1/chat/conversations/{conversation_id}/citations")

    @staticmethod
    def _chat_knowledge_save_request(
        request_data: ChatKnowledgeSaveRequest | Dict[str, Any],
    ) -> ChatKnowledgeSaveRequest:
        if isinstance(request_data, ChatKnowledgeSaveRequest):
            return request_data
        return ChatKnowledgeSaveRequest(**dict(request_data))

    @staticmethod
    def _conversation_share_link_create_request(
        request_data: ConversationShareLinkCreateRequest | Dict[str, Any],
    ) -> ConversationShareLinkCreateRequest:
        if isinstance(request_data, ConversationShareLinkCreateRequest):
            return request_data
        return ConversationShareLinkCreateRequest(**dict(request_data))

    async def list_chat_commands(self) -> ChatCommandsListResponse:
        response = await self._request("GET", "/api/v1/chat/commands")
        return ChatCommandsListResponse.model_validate(response)

    async def save_chat_knowledge(
        self,
        request_data: ChatKnowledgeSaveRequest | Dict[str, Any],
    ) -> ChatKnowledgeSaveResponse:
        payload = self._chat_knowledge_save_request(request_data)
        response = await self._request(
            "POST",
            "/api/v1/chat/knowledge/save",
            json_data=payload.model_dump(exclude_none=True, mode="json"),
        )
        return ChatKnowledgeSaveResponse.model_validate(response)

    async def create_chat_conversation_share_link(
        self,
        conversation_id: str,
        request_data: ConversationShareLinkCreateRequest | Dict[str, Any],
        scope_type: Optional[Literal["global", "workspace"]] = None,
        workspace_id: Optional[str] = None,
    ) -> ConversationShareLinkResponse:
        scope_params = self._normalize_conversation_scope_params(scope_type=scope_type, workspace_id=workspace_id)
        params: Dict[str, Any] = {}
        if scope_params is not None:
            params.update(scope_params.model_dump(exclude_none=True, mode="json"))
        payload = self._conversation_share_link_create_request(request_data)
        response = await self._request(
            "POST",
            f"/api/v1/chat/conversations/{conversation_id}/share-links",
            json_data=payload.model_dump(exclude_none=True, mode="json"),
            params=params or None,
        )
        return ConversationShareLinkResponse.model_validate(response)

    async def list_chat_conversation_share_links(
        self,
        conversation_id: str,
        scope_type: Optional[Literal["global", "workspace"]] = None,
        workspace_id: Optional[str] = None,
    ) -> ConversationShareLinksResponse:
        scope_params = self._normalize_conversation_scope_params(scope_type=scope_type, workspace_id=workspace_id)
        params: Dict[str, Any] = {}
        if scope_params is not None:
            params.update(scope_params.model_dump(exclude_none=True, mode="json"))
        response = await self._request(
            "GET",
            f"/api/v1/chat/conversations/{conversation_id}/share-links",
            params=params or None,
        )
        return ConversationShareLinksResponse.model_validate(response)

    async def revoke_chat_conversation_share_link(
        self,
        conversation_id: str,
        share_id: str,
        scope_type: Optional[Literal["global", "workspace"]] = None,
        workspace_id: Optional[str] = None,
    ) -> ConversationShareLinkRevokeResponse:
        scope_params = self._normalize_conversation_scope_params(scope_type=scope_type, workspace_id=workspace_id)
        params: Dict[str, Any] = {}
        if scope_params is not None:
            params.update(scope_params.model_dump(exclude_none=True, mode="json"))
        response = await self._request(
            "DELETE",
            f"/api/v1/chat/conversations/{conversation_id}/share-links/{share_id}",
            params=params or None,
        )
        return ConversationShareLinkRevokeResponse.model_validate(response)

    async def resolve_chat_conversation_share_token(
        self,
        share_token: str,
        *,
        limit: int = 200,
    ) -> SharedConversationResolveResponse:
        response = await self._request(
            "GET",
            f"/api/v1/chat/shared/conversations/{share_token}",
            params={"limit": limit},
        )
        return SharedConversationResolveResponse.model_validate(response)

    async def get_chat_analytics(
        self,
        *,
        start_date: str,
        end_date: str,
        bucket_granularity: Literal["day", "week"] = "day",
        limit: int = 100,
        offset: int = 0,
    ) -> ChatAnalyticsResponse:
        response = await self._request(
            "GET",
            "/api/v1/chat/analytics",
            params={
                "start_date": start_date,
                "end_date": end_date,
                "bucket_granularity": bucket_granularity,
                "limit": limit,
                "offset": offset,
            },
        )
        return ChatAnalyticsResponse.model_validate(response)

    async def start_chat_loop_run(
        self,
        request_data: ChatLoopStartRequest | Dict[str, Any],
    ) -> ChatLoopStartResponse:
        response = await self._request(
            "POST",
            "/api/v1/chat/loop/start",
            json_data=self._dump_request_payload(request_data),
        )
        return ChatLoopStartResponse.model_validate(response)

    async def list_chat_loop_events(
        self,
        run_id: str,
        *,
        after_seq: int = 0,
    ) -> ChatLoopEventsResponse:
        response = await self._request(
            "GET",
            f"/api/v1/chat/loop/{run_id}/events",
            params={"after_seq": after_seq},
        )
        return ChatLoopEventsResponse.model_validate(response)

    async def approve_chat_loop_call(
        self,
        run_id: str,
        request_data: ChatLoopApprovalDecisionRequest | Dict[str, Any] | str,
    ) -> ChatLoopActionResponse:
        payload = (
            ChatLoopApprovalDecisionRequest(approval_id=request_data, decision="approve")
            if isinstance(request_data, str)
            else request_data
        )
        response = await self._request(
            "POST",
            f"/api/v1/chat/loop/{run_id}/approve",
            json_data=self._dump_request_payload(payload),
        )
        return ChatLoopActionResponse.model_validate(response)

    async def reject_chat_loop_call(
        self,
        run_id: str,
        request_data: ChatLoopApprovalDecisionRequest | Dict[str, Any] | str,
    ) -> ChatLoopActionResponse:
        payload = (
            ChatLoopApprovalDecisionRequest(approval_id=request_data, decision="reject")
            if isinstance(request_data, str)
            else request_data
        )
        response = await self._request(
            "POST",
            f"/api/v1/chat/loop/{run_id}/reject",
            json_data=self._dump_request_payload(payload),
        )
        return ChatLoopActionResponse.model_validate(response)

    async def cancel_chat_loop_run(self, run_id: str) -> ChatLoopActionResponse:
        response = await self._request("POST", f"/api/v1/chat/loop/{run_id}/cancel")
        return ChatLoopActionResponse.model_validate(response)

    async def list_chat_dictionaries(
        self,
        *,
        include_inactive: bool = False,
        include_usage: bool = False,
    ) -> Dict[str, Any]:
        return await self._request(
            "GET",
            "/api/v1/chat/dictionaries",
            params={
                "include_inactive": str(include_inactive).lower(),
                "include_usage": str(include_usage).lower(),
            },
        )

    async def create_chat_dictionary(
        self,
        request_data: ChatDictionaryCreateRequest | Dict[str, Any],
    ) -> Dict[str, Any]:
        return await self._request(
            "POST",
            "/api/v1/chat/dictionaries",
            json_data=self._dump_request_payload(request_data),
        )

    async def get_chat_dictionary(self, dictionary_id: int) -> Dict[str, Any]:
        return await self._request("GET", f"/api/v1/chat/dictionaries/{dictionary_id}")

    async def update_chat_dictionary(
        self,
        dictionary_id: int,
        request_data: ChatDictionaryUpdateRequest | Dict[str, Any],
    ) -> Dict[str, Any]:
        return await self._request(
            "PUT",
            f"/api/v1/chat/dictionaries/{dictionary_id}",
            json_data=self._dump_request_payload(request_data, exclude_unset=True),
        )

    async def delete_chat_dictionary(
        self,
        dictionary_id: int,
        *,
        hard_delete: bool = False,
    ) -> Dict[str, Any]:
        return await self._request(
            "DELETE",
            f"/api/v1/chat/dictionaries/{dictionary_id}",
            params={"hard_delete": str(hard_delete).lower()},
        )

    async def add_chat_dictionary_entry(
        self,
        dictionary_id: int,
        request_data: DictionaryEntryCreateRequest | Dict[str, Any],
    ) -> Dict[str, Any]:
        return await self._request(
            "POST",
            f"/api/v1/chat/dictionaries/{dictionary_id}/entries",
            json_data=self._dump_request_payload(request_data),
        )

    async def list_chat_dictionary_entries(
        self,
        dictionary_id: int,
        *,
        group: str | None = None,
    ) -> Dict[str, Any]:
        params = {"group": group} if group is not None else None
        return await self._request(
            "GET",
            f"/api/v1/chat/dictionaries/{dictionary_id}/entries",
            params=params,
        )

    async def update_chat_dictionary_entry(
        self,
        entry_id: int,
        request_data: DictionaryEntryUpdateRequest | Dict[str, Any],
    ) -> Dict[str, Any]:
        return await self._request(
            "PUT",
            f"/api/v1/chat/dictionaries/entries/{entry_id}",
            json_data=self._dump_request_payload(request_data, exclude_unset=True),
        )

    async def delete_chat_dictionary_entry(self, entry_id: int) -> Dict[str, Any]:
        return await self._request("DELETE", f"/api/v1/chat/dictionaries/entries/{entry_id}")

    async def bulk_chat_dictionary_entry_operations(
        self,
        request_data: BulkDictionaryEntryOperationRequest | Dict[str, Any],
    ) -> Dict[str, Any]:
        return await self._request(
            "POST",
            "/api/v1/chat/dictionaries/entries/bulk",
            json_data=self._dump_request_payload(request_data),
        )

    async def reorder_chat_dictionary_entries(
        self,
        dictionary_id: int,
        request_data: DictionaryEntryReorderRequest | Dict[str, Any],
    ) -> Dict[str, Any]:
        return await self._request(
            "PUT",
            f"/api/v1/chat/dictionaries/{dictionary_id}/entries/reorder",
            json_data=self._dump_request_payload(request_data),
        )

    async def process_chat_dictionaries(
        self,
        request_data: ProcessChatDictionariesRequest | Dict[str, Any],
    ) -> Dict[str, Any]:
        return await self._request(
            "POST",
            "/api/v1/chat/dictionaries/process",
            json_data=self._dump_request_payload(request_data),
        )

    async def import_chat_dictionary_markdown(
        self,
        request_data: ImportDictionaryMarkdownRequest | Dict[str, Any],
    ) -> Dict[str, Any]:
        return await self._request(
            "POST",
            "/api/v1/chat/dictionaries/import",
            json_data=self._dump_request_payload(request_data),
        )

    async def export_chat_dictionary_markdown(self, dictionary_id: int) -> Dict[str, Any]:
        return await self._request("GET", f"/api/v1/chat/dictionaries/{dictionary_id}/export")

    async def export_chat_dictionary_json(self, dictionary_id: int) -> Dict[str, Any]:
        return await self._request("GET", f"/api/v1/chat/dictionaries/{dictionary_id}/export/json")

    async def import_chat_dictionary_json(
        self,
        request_data: ImportDictionaryJSONRequest | Dict[str, Any],
    ) -> Dict[str, Any]:
        return await self._request(
            "POST",
            "/api/v1/chat/dictionaries/import/json",
            json_data=self._dump_request_payload(request_data),
        )

    async def list_chat_dictionary_activity(
        self,
        dictionary_id: int,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> Dict[str, Any]:
        return await self._request(
            "GET",
            f"/api/v1/chat/dictionaries/{dictionary_id}/activity",
            params={"limit": limit, "offset": offset},
        )

    async def list_chat_dictionary_versions(
        self,
        dictionary_id: int,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> Dict[str, Any]:
        return await self._request(
            "GET",
            f"/api/v1/chat/dictionaries/{dictionary_id}/versions",
            params={"limit": limit, "offset": offset},
        )

    async def get_chat_dictionary_version(self, dictionary_id: int, revision: int) -> Dict[str, Any]:
        return await self._request("GET", f"/api/v1/chat/dictionaries/{dictionary_id}/versions/{revision}")

    async def revert_chat_dictionary_version(self, dictionary_id: int, revision: int) -> Dict[str, Any]:
        return await self._request("POST", f"/api/v1/chat/dictionaries/{dictionary_id}/versions/{revision}/revert")

    async def get_chat_dictionary_statistics(self, dictionary_id: int) -> Dict[str, Any]:
        return await self._request("GET", f"/api/v1/chat/dictionaries/{dictionary_id}/statistics")

    async def validate_chat_dictionary(
        self,
        request_data: ValidateDictionaryRequest | Dict[str, Any],
    ) -> Dict[str, Any]:
        return await self._request(
            "POST",
            "/api/v1/chat/dictionaries/validate",
            json_data=self._dump_request_payload(request_data),
        )

    async def list_prompt_versions(self, prompt_identifier: Union[str, int]) -> Dict[str, Any]:
        return await self._request(
            "GET",
            f"/api/v1/chat/messages/{message_id}/rag-context",
            params=params,
        )

    async def get_chat_conversation_messages_with_context(
        self,
        conversation_id: str,
        limit: int = 100,
        offset: int = 0,
        include_rag_context: bool = True,
        scope_type: Optional[Literal["global", "workspace"]] = None,
        workspace_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        scope_params = self._normalize_conversation_scope_params(scope_type=scope_type, workspace_id=workspace_id)
        params: Dict[str, Any] = {
            "limit": limit,
            "offset": offset,
            "include_rag_context": str(include_rag_context).lower(),
        }
        if scope_params is not None:
            params.update(scope_params.model_dump(exclude_none=True, mode="json"))
        return await self._request(
            "GET",
            f"/api/v1/chat/conversations/{conversation_id}/messages-with-context",
            params=params,
        )

    async def get_chat_conversation_citations(self, conversation_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/api/v1/chat/conversations/{conversation_id}/citations")

    async def list_chat_commands(self) -> ChatCommandsListResponse:
        response = await self._request("GET", "/api/v1/chat/commands")
        return ChatCommandsListResponse.model_validate(response)

    async def validate_chat_dictionary(
        self,
        request_data: ValidateDictionaryRequest,
    ) -> ValidateDictionaryResponse:
        response = await self._request(
            "POST",
            "/api/v1/chat/dictionaries/validate",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return ValidateDictionaryResponse.model_validate(response)

    async def get_chat_queue_status(self) -> ChatQueueStatusResponse:
        response = await self._request("GET", "/api/v1/chat/queue/status")
        return ChatQueueStatusResponse.model_validate(response)

    async def get_chat_queue_activity(self, limit: int = 50) -> ChatQueueActivityResponse:
        response = await self._request(
            "GET",
            "/api/v1/chat/queue/activity",
            params={"limit": limit},
        )
        return ChatQueueActivityResponse.model_validate(response)

    async def save_chat_knowledge(
        self,
        request_data: ChatKnowledgeSaveRequest | Dict[str, Any],
    ) -> ChatKnowledgeSaveResponse:
        payload = self._chat_knowledge_save_request(request_data)
        response = await self._request(
            "POST",
            "/api/v1/chat/knowledge/save",
            json_data=payload.model_dump(exclude_none=True, mode="json"),
        )
        return ChatKnowledgeSaveResponse.model_validate(response)

    async def get_chat_analytics(
        self,
        *,
        start_date: str,
        end_date: str,
        bucket_granularity: Literal["day", "week"] = "day",
        limit: int = 100,
        offset: int = 0,
    ) -> ChatAnalyticsResponse:
        response = await self._request(
            "GET",
            "/api/v1/chat/analytics",
            params={
                "start_date": start_date,
                "end_date": end_date,
                "bucket_granularity": bucket_granularity,
                "limit": limit,
                "offset": offset,
            },
        )
        return ChatAnalyticsResponse.model_validate(response)

    @staticmethod
    def _parse_chat_document_generation_response(
        response: Dict[str, Any],
    ) -> GenerateDocumentResponse | AsyncGenerationResponse:
        if "job_id" in response:
            return AsyncGenerationResponse.model_validate(response)
        return GenerateDocumentResponse.model_validate(response)

    async def generate_chat_document(
        self,
        request_data: GenerateDocumentRequest,
    ) -> GenerateDocumentResponse | AsyncGenerationResponse:
        if request_data.stream:
            raise ValueError(
                "Streaming chat document generation is not supported by the JSON API client; "
                "use async_generation=True and poll the job status instead."
            )
        response = await self._request(
            "POST",
            "/api/v1/chat/documents/generate",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return self._parse_chat_document_generation_response(response)

    async def get_chat_document_job_status(self, job_id: str) -> JobStatusResponse:
        response = await self._request("GET", f"/api/v1/chat/documents/jobs/{job_id}")
        return JobStatusResponse.model_validate(response)

    async def cancel_chat_document_job(self, job_id: str) -> Dict[str, str]:
        return await self._request("DELETE", f"/api/v1/chat/documents/jobs/{job_id}")

    async def list_chat_generated_documents(
        self,
        *,
        conversation_id: Optional[str] = None,
        document_type: DocumentType | str | None = None,
        limit: int = 50,
    ) -> DocumentListResponse:
        params: Dict[str, Any] = {"limit": limit}
        if conversation_id is not None:
            params["conversation_id"] = conversation_id
        if document_type is not None:
            params["document_type"] = document_type.value if isinstance(document_type, DocumentType) else document_type
        response = await self._request("GET", "/api/v1/chat/documents", params=params)
        return DocumentListResponse.model_validate(response)

    async def get_chat_generated_document(self, document_id: int) -> GeneratedDocument:
        response = await self._request("GET", f"/api/v1/chat/documents/{document_id}")
        return GeneratedDocument.model_validate(response)

    async def delete_chat_generated_document(self, document_id: int) -> Dict[str, str]:
        return await self._request("DELETE", f"/api/v1/chat/documents/{document_id}")

    async def save_chat_document_prompt_config(self, request_data: SavePromptConfigRequest) -> PromptConfigResponse:
        response = await self._request(
            "POST",
            "/api/v1/chat/documents/prompts",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return PromptConfigResponse.model_validate(response)

    async def get_chat_document_prompt_config(self, document_type: DocumentType | str) -> PromptConfigResponse:
        document_type_value = document_type.value if isinstance(document_type, DocumentType) else str(document_type)
        response = await self._request("GET", f"/api/v1/chat/documents/prompts/{document_type_value}")
        return PromptConfigResponse.model_validate(response)

    async def bulk_generate_chat_documents(self, request_data: BulkGenerateRequest) -> BulkGenerateResponse:
        response = await self._request(
            "POST",
            "/api/v1/chat/documents/bulk",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return BulkGenerateResponse.model_validate(response)

    async def get_chat_document_generation_statistics(self) -> GenerationStatistics:
        response = await self._request("GET", "/api/v1/chat/documents/statistics")
        return GenerationStatistics.model_validate(response)

    async def start_chat_loop_run(self, request_data: ChatLoopStartRequest | Dict[str, Any]) -> ChatLoopStartResponse:
        response = await self._request(
            "POST",
            "/api/v1/chat/loop/start",
            json_data=self._dump_request_payload(request_data),
        )
        return ChatLoopStartResponse.model_validate(response)

    async def list_chat_loop_events(self, run_id: str, after_seq: int = 0) -> ChatLoopEventsResponse:
        response = await self._request(
            "GET",
            f"/api/v1/chat/loop/{run_id}/events",
            params={"after_seq": after_seq},
        )
        return ChatLoopEventsResponse.model_validate(response)

    async def approve_chat_loop_call(
        self,
        run_id: str,
        request_data: ChatLoopApprovalDecisionRequest | Dict[str, Any] | str,
    ) -> ChatLoopActionResponse:
        payload = (
            ChatLoopApprovalDecisionRequest(approval_id=request_data, decision="approve")
            if isinstance(request_data, str)
            else request_data
        )
        response = await self._request(
            "POST",
            f"/api/v1/chat/loop/{run_id}/approve",
            json_data=self._dump_request_payload(payload),
        )
        return ChatLoopActionResponse.model_validate(response)

    async def reject_chat_loop_call(
        self,
        run_id: str,
        request_data: ChatLoopApprovalDecisionRequest | Dict[str, Any] | str,
    ) -> ChatLoopActionResponse:
        payload = (
            ChatLoopApprovalDecisionRequest(approval_id=request_data, decision="reject")
            if isinstance(request_data, str)
            else request_data
        )
        response = await self._request(
            "POST",
            f"/api/v1/chat/loop/{run_id}/reject",
            json_data=self._dump_request_payload(payload),
        )
        return ChatLoopActionResponse.model_validate(response)

    async def cancel_chat_loop_run(self, run_id: str) -> ChatLoopActionResponse:
        response = await self._request("POST", f"/api/v1/chat/loop/{run_id}/cancel")
        return ChatLoopActionResponse.model_validate(response)

    async def list_prompt_versions(self, prompt_identifier: Union[str, int]) -> List[PromptVersionResponse]:
        response = await self._request(
            "GET",
            f"/api/v1/prompts/{prompt_identifier}/versions",
        )
        return [PromptVersionResponse.model_validate(item) for item in response]

    async def restore_prompt_version(self, prompt_identifier: Union[str, int], version: int) -> PromptResponse:
        response = await self._request(
            "POST",
            f"/api/v1/prompts/{prompt_identifier}/versions/{version}/restore",
        )
        return PromptResponse.model_validate(response)

    async def create_prompt_collection(
        self,
        request_data: PromptCollectionCreateRequest | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        prompt_ids: List[int] | None = None,
    ) -> PromptCollectionCreateResponse | Dict[str, Any]:
        payload_model = request_data or PromptCollectionCreateRequest(
            name=name or "",
            description=description,
            prompt_ids=prompt_ids or [],
        )
        response = await self._request(
            "POST",
            "/api/v1/prompts/collections/create",
            json_data=payload_model.model_dump(exclude_none=True, mode="json"),
        )
        try:
            return PromptCollectionCreateResponse.model_validate(response)
        except Exception:
            return response

    async def list_prompt_collections(self, limit: int = 200, offset: int = 0) -> PromptCollectionListResponse | Dict[str, Any]:
        response = await self._request(
            "GET",
            "/api/v1/prompts/collections",
            params={"limit": limit, "offset": offset},
        )
        try:
            return PromptCollectionListResponse.model_validate(response)
        except Exception:
            return response

    async def get_prompt_collection(self, collection_id: int) -> PromptCollectionResponse | Dict[str, Any]:
        response = await self._request(
            "GET",
            f"/api/v1/prompts/collections/{collection_id}",
        )
        try:
            return PromptCollectionResponse.model_validate(response)
        except Exception:
            return response

    async def update_prompt_collection(
        self,
        collection_id: int,
        request_data: PromptCollectionUpdateRequest | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        prompt_ids: List[int] | None = None,
    ) -> PromptCollectionResponse | Dict[str, Any]:
        payload_model = request_data or PromptCollectionUpdateRequest(
            name=name,
            description=description,
            prompt_ids=prompt_ids,
        )
        response = await self._request(
            "PUT",
            f"/api/v1/prompts/collections/{collection_id}",
            json_data=payload_model.model_dump(exclude_none=True, mode="json"),
        )
        try:
            return PromptCollectionResponse.model_validate(response)
        except Exception:
            return response

    async def export_chatbook(self, request_data: ChatbookExportRequest) -> Dict[str, Any]:
        return await self._request(
            "POST",
            "/api/v1/chatbooks/export",
            json_data=request_data.model_dump(exclude_none=True),
        )

    async def continue_chatbook_export(self, request_data: ChatbookContinueExportRequest) -> Dict[str, Any]:
        return await self._request(
            "POST",
            "/api/v1/chatbooks/export/continue",
            json_data=request_data.model_dump(exclude_none=True),
        )

    async def preview_chatbook(self, chatbook_file_path: str) -> Dict[str, Any]:
        httpx_files = prepare_files_for_httpx([chatbook_file_path], upload_field_name="file")
        try:
            return await self._request(
                "POST",
                "/api/v1/chatbooks/preview",
                files=httpx_files,
            )
        finally:
            cleanup_file_objects(httpx_files)

    async def import_chatbook(self, chatbook_file_path: str, request_data: ChatbookImportRequest) -> Dict[str, Any]:
        httpx_files = prepare_files_for_httpx([chatbook_file_path], upload_field_name="file")
        form_data = model_to_form_data(request_data)
        if request_data.content_selections is not None:
            form_data["content_selections"] = json.dumps(request_data.content_selections)
        try:
            return await self._request(
                "POST",
                "/api/v1/chatbooks/import",
                data=form_data,
                files=httpx_files,
            )
        finally:
            cleanup_file_objects(httpx_files)

    async def list_chatbook_export_jobs(self, limit: int = 100, offset: int = 0) -> ChatbookExportJobListResponse:
        response = await self._request(
            "GET",
            "/api/v1/chatbooks/export/jobs",
            params={"limit": limit, "offset": offset},
        )
        return ChatbookExportJobListResponse.model_validate(response)

    async def list_chatbook_import_jobs(self, limit: int = 100, offset: int = 0) -> ChatbookImportJobListResponse:
        response = await self._request(
            "GET",
            "/api/v1/chatbooks/import/jobs",
            params={"limit": limit, "offset": offset},
        )
        return ChatbookImportJobListResponse.model_validate(response)

    async def get_chatbook_export_job(self, job_id: str) -> ChatbookExportJobResponse:
        response = await self._request(
            "GET",
            f"/api/v1/chatbooks/export/jobs/{job_id}",
        )
        return ChatbookExportJobResponse.model_validate(response)

    async def download_chatbook_export(
        self,
        job_id: str,
        *,
        token: str | None = None,
        exp: int | str | None = None,
    ) -> ReadingExportResponse:
        params = {key: value for key, value in {"token": token, "exp": exp}.items() if value is not None}
        return await self._binary_request(
            "GET",
            f"/api/v1/chatbooks/download/{job_id}",
            params=params or None,
        )

    async def get_chatbook_import_job(self, job_id: str) -> Dict[str, Any]:
        return await self._request(
            "GET",
            f"/api/v1/chatbooks/import/jobs/{job_id}",
        )
        return ChatbookImportJobResponse.model_validate(response)

    async def cancel_chatbook_export_job(self, job_id: str) -> ChatbookJobMutationResponse:
        response = await self._request(
            "DELETE",
            f"/api/v1/chatbooks/export/jobs/{job_id}",
        )
        return ChatbookJobMutationResponse.model_validate(response)

    async def cancel_chatbook_import_job(self, job_id: str) -> ChatbookJobMutationResponse:
        response = await self._request(
            "DELETE",
            f"/api/v1/chatbooks/import/jobs/{job_id}",
        )
        return ChatbookJobMutationResponse.model_validate(response)

    async def remove_chatbook_export_job(self, job_id: str) -> ChatbookJobMutationResponse:
        response = await self._request(
            "DELETE",
            f"/api/v1/chatbooks/export/jobs/{job_id}/remove",
        )
        return ChatbookJobMutationResponse.model_validate(response)

    async def cleanup_chatbook_exports(self) -> ChatbookCleanupResponse:
        response = await self._request(
            "POST",
            "/api/v1/chatbooks/cleanup",
        )
        return ChatbookCleanupResponse.model_validate(response)

    async def download_chatbook_export(
        self,
        job_id: str,
        *,
        token: str | None = None,
        exp: int | str | None = None,
    ) -> ReadingExportResponse:
        params = {key: value for key, value in {"token": token, "exp": exp}.items() if value is not None}
        return await self._binary_request(
            "GET",
            f"/api/v1/chatbooks/download/{job_id}",
            params=params or None,
        )

    async def remove_chatbook_import_job(self, job_id: str) -> ChatbookJobMutationResponse:
        response = await self._request(
            "DELETE",
            f"/api/v1/chatbooks/import/jobs/{job_id}/remove",
        )
        return ChatbookJobMutationResponse.model_validate(response)

    async def list_chatbook_export_jobs(self, *, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        return await self._request(
            "GET",
            "/api/v1/chatbooks/export/jobs",
            params={"limit": limit, "offset": offset},
        )

    async def list_chatbook_import_jobs(self, *, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        return await self._request(
            "GET",
            "/api/v1/chatbooks/import/jobs",
            params={"limit": limit, "offset": offset},
        )

    async def cancel_chatbook_export_job(self, job_id: str) -> Dict[str, Any]:
        return await self._request(
            "DELETE",
            f"/api/v1/chatbooks/export/jobs/{job_id}",
        )

    async def cancel_chatbook_import_job(self, job_id: str) -> Dict[str, Any]:
        return await self._request(
            "DELETE",
            f"/api/v1/chatbooks/import/jobs/{job_id}",
        )

    async def remove_chatbook_export_job(self, job_id: str) -> Dict[str, Any]:
        return await self._request(
            "DELETE",
            f"/api/v1/chatbooks/export/jobs/{job_id}/remove",
        )

    async def remove_chatbook_import_job(self, job_id: str) -> Dict[str, Any]:
        return await self._request(
            "DELETE",
            f"/api/v1/chatbooks/import/jobs/{job_id}/remove",
        )

    async def create_kanban_board(self, request_data: KanbanBoardCreate) -> KanbanBoardResponse:
        response = await self._request(
            "POST",
            "/api/v1/kanban/boards",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return KanbanBoardResponse.model_validate(response)

    async def list_kanban_boards(
        self,
        *,
        include_archived: bool = False,
        include_deleted: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> KanbanBoardListResponse:
        response = await self._request(
            "GET",
            "/api/v1/kanban/boards",
            params={
                "include_archived": include_archived,
                "include_deleted": include_deleted,
                "limit": limit,
                "offset": offset,
            },
        )
        return KanbanBoardListResponse.model_validate(response)

    async def get_kanban_board(
        self,
        board_id: int,
        *,
        include_lists: bool = True,
        include_cards: bool = True,
        include_archived: bool = False,
    ) -> KanbanBoardWithListsResponse:
        response = await self._request(
            "GET",
            f"/api/v1/kanban/boards/{board_id}",
            params={
                "include_lists": include_lists,
                "include_cards": include_cards,
                "include_archived": include_archived,
            },
        )
        return KanbanBoardWithListsResponse.model_validate(response)

    async def update_kanban_board(
        self,
        board_id: int,
        request_data: KanbanBoardUpdate,
        *,
        expected_version: int | None = None,
    ) -> KanbanBoardResponse:
        response = await self._request(
            "PATCH",
            f"/api/v1/kanban/boards/{board_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
            headers=self._expected_version_header(expected_version),
        )
        return KanbanBoardResponse.model_validate(response)

    async def archive_kanban_board(self, board_id: int) -> KanbanBoardResponse:
        response = await self._request("POST", f"/api/v1/kanban/boards/{board_id}/archive")
        return KanbanBoardResponse.model_validate(response)

    async def unarchive_kanban_board(self, board_id: int) -> KanbanBoardResponse:
        response = await self._request("POST", f"/api/v1/kanban/boards/{board_id}/unarchive")
        return KanbanBoardResponse.model_validate(response)

    async def delete_kanban_board(self, board_id: int) -> KanbanDetailResponse:
        response = await self._request("DELETE", f"/api/v1/kanban/boards/{board_id}")
        return KanbanDetailResponse.model_validate(response)

    async def restore_kanban_board(self, board_id: int) -> KanbanBoardResponse:
        response = await self._request("POST", f"/api/v1/kanban/boards/{board_id}/restore")
        return KanbanBoardResponse.model_validate(response)

    async def create_kanban_list(self, board_id: int, request_data: KanbanListCreate) -> KanbanListResponse:
        response = await self._request(
            "POST",
            f"/api/v1/kanban/boards/{board_id}/lists",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return KanbanListResponse.model_validate(response)

    async def list_kanban_lists(
        self,
        board_id: int,
        *,
        include_archived: bool = False,
        include_deleted: bool = False,
    ) -> KanbanListsListResponse:
        response = await self._request(
            "GET",
            f"/api/v1/kanban/boards/{board_id}/lists",
            params={"include_archived": include_archived, "include_deleted": include_deleted},
        )
        return KanbanListsListResponse.model_validate(response)

    async def reorder_kanban_lists(
        self,
        board_id: int,
        request_data: KanbanReorderRequest,
    ) -> KanbanReorderResponse:
        response = await self._request(
            "POST",
            f"/api/v1/kanban/boards/{board_id}/lists/reorder",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return KanbanReorderResponse.model_validate(response)

    async def get_kanban_list(self, list_id: int) -> KanbanListResponse:
        response = await self._request("GET", f"/api/v1/kanban/lists/{list_id}")
        return KanbanListResponse.model_validate(response)

    async def update_kanban_list(
        self,
        list_id: int,
        request_data: KanbanListUpdate,
        *,
        expected_version: int | None = None,
    ) -> KanbanListResponse:
        response = await self._request(
            "PATCH",
            f"/api/v1/kanban/lists/{list_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
            headers=self._expected_version_header(expected_version),
        )
        return KanbanListResponse.model_validate(response)

    async def archive_kanban_list(self, list_id: int) -> KanbanListResponse:
        response = await self._request("POST", f"/api/v1/kanban/lists/{list_id}/archive")
        return KanbanListResponse.model_validate(response)

    async def unarchive_kanban_list(self, list_id: int) -> KanbanListResponse:
        response = await self._request("POST", f"/api/v1/kanban/lists/{list_id}/unarchive")
        return KanbanListResponse.model_validate(response)

    async def delete_kanban_list(self, list_id: int) -> KanbanDetailResponse:
        response = await self._request("DELETE", f"/api/v1/kanban/lists/{list_id}")
        return KanbanDetailResponse.model_validate(response)

    async def restore_kanban_list(self, list_id: int) -> KanbanListResponse:
        response = await self._request("POST", f"/api/v1/kanban/lists/{list_id}/restore")
        return KanbanListResponse.model_validate(response)

    async def create_kanban_card(self, list_id: int, request_data: KanbanCardCreate) -> KanbanCardResponse:
        response = await self._request(
            "POST",
            f"/api/v1/kanban/lists/{list_id}/cards",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return KanbanCardResponse.model_validate(response)

    async def list_kanban_cards(
        self,
        list_id: int,
        *,
        include_archived: bool = False,
        include_deleted: bool = False,
    ) -> KanbanCardsListResponse:
        response = await self._request(
            "GET",
            f"/api/v1/kanban/lists/{list_id}/cards",
            params={"include_archived": include_archived, "include_deleted": include_deleted},
        )
        return KanbanCardsListResponse.model_validate(response)

    async def reorder_kanban_cards(
        self,
        list_id: int,
        request_data: KanbanReorderRequest,
    ) -> KanbanReorderResponse:
        response = await self._request(
            "POST",
            f"/api/v1/kanban/lists/{list_id}/cards/reorder",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return KanbanReorderResponse.model_validate(response)

    async def get_kanban_card(self, card_id: int) -> KanbanCardWithDetailsResponse:
        response = await self._request("GET", f"/api/v1/kanban/cards/{card_id}")
        return KanbanCardWithDetailsResponse.model_validate(response)

    async def update_kanban_card(
        self,
        card_id: int,
        request_data: KanbanCardUpdate,
        *,
        expected_version: int | None = None,
    ) -> KanbanCardResponse:
        response = await self._request(
            "PATCH",
            f"/api/v1/kanban/cards/{card_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
            headers=self._expected_version_header(expected_version),
        )
        return KanbanCardResponse.model_validate(response)

    async def move_kanban_card(self, card_id: int, request_data: KanbanCardMoveRequest) -> KanbanCardResponse:
        response = await self._request(
            "POST",
            f"/api/v1/kanban/cards/{card_id}/move",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return KanbanCardResponse.model_validate(response)

    async def copy_kanban_card(self, card_id: int, request_data: KanbanCardCopyRequest) -> KanbanCardResponse:
        response = await self._request(
            "POST",
            f"/api/v1/kanban/cards/{card_id}/copy",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return KanbanCardResponse.model_validate(response)

    async def archive_kanban_card(self, card_id: int) -> KanbanCardResponse:
        response = await self._request("POST", f"/api/v1/kanban/cards/{card_id}/archive")
        return KanbanCardResponse.model_validate(response)

    async def unarchive_kanban_card(self, card_id: int) -> KanbanCardResponse:
        response = await self._request("POST", f"/api/v1/kanban/cards/{card_id}/unarchive")
        return KanbanCardResponse.model_validate(response)

    async def delete_kanban_card(self, card_id: int) -> KanbanDetailResponse:
        response = await self._request("DELETE", f"/api/v1/kanban/cards/{card_id}")
        return KanbanDetailResponse.model_validate(response)

    async def restore_kanban_card(self, card_id: int) -> KanbanCardResponse:
        response = await self._request("POST", f"/api/v1/kanban/cards/{card_id}/restore")
        return KanbanCardResponse.model_validate(response)

    async def list_kanban_board_activities(
        self,
        board_id: int,
        *,
        list_id: int | None = None,
        card_id: int | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
        action_type: str | None = None,
        entity_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
        page: int | None = None,
        per_page: int | None = None,
    ) -> KanbanActivitiesListResponse:
        params = {
            key: value
            for key, value in {
                "list_id": list_id,
                "card_id": card_id,
                "created_after": created_after,
                "created_before": created_before,
                "action_type": action_type,
                "entity_type": entity_type,
                "limit": limit,
                "offset": offset,
                "page": page,
                "per_page": per_page,
            }.items()
            if value is not None
        }
        response = await self._request("GET", f"/api/v1/kanban/{board_id}/activities", params=params)
        return KanbanActivitiesListResponse.model_validate(response)

    async def list_kanban_card_activities(
        self,
        card_id: int,
        *,
        created_after: str | None = None,
        created_before: str | None = None,
        action_type: str | None = None,
        entity_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
        page: int | None = None,
        per_page: int | None = None,
    ) -> KanbanActivitiesListResponse:
        params = {
            key: value
            for key, value in {
                "created_after": created_after,
                "created_before": created_before,
                "action_type": action_type,
                "entity_type": entity_type,
                "limit": limit,
                "offset": offset,
                "page": page,
                "per_page": per_page,
            }.items()
            if value is not None
        }
        response = await self._request("GET", f"/api/v1/kanban/cards/{card_id}/activities", params=params)
        return KanbanActivitiesListResponse.model_validate(response)

    async def get_kanban_board_export(
        self,
        board_id: int,
        *,
        include_archived: bool = False,
        include_deleted: bool = False,
    ) -> KanbanBoardExportResponse:
        response = await self._request(
            "GET",
            f"/api/v1/kanban/{board_id}/export",
            params={"include_archived": include_archived, "include_deleted": include_deleted},
        )
        return KanbanBoardExportResponse.model_validate(response)

    async def export_kanban_board(
        self,
        board_id: int,
        request_data: KanbanBoardExportRequest,
    ) -> KanbanBoardExportResponse:
        response = await self._request(
            "POST",
            f"/api/v1/kanban/{board_id}/export",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return KanbanBoardExportResponse.model_validate(response)

    async def import_kanban_board(self, request_data: KanbanBoardImportRequest) -> KanbanBoardImportResponse:
        response = await self._request(
            "POST",
            "/api/v1/kanban/import",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return KanbanBoardImportResponse.model_validate(response)

    async def search_kanban_cards_basic(self, request_data: KanbanCardSearchRequest) -> KanbanCardSearchResponse:
        response = await self._request(
            "POST",
            "/api/v1/kanban/cards/search",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return KanbanCardSearchResponse.model_validate(response)

    async def search_kanban_cards_basic_get(
        self,
        query: str,
        *,
        board_id: int | None = None,
        limit: int = 50,
        offset: int = 0,
        page: int | None = None,
        per_page: int | None = None,
    ) -> KanbanCardSearchResponse:
        params = {
            key: value
            for key, value in {
                "q": query,
                "board_id": board_id,
                "limit": limit,
                "offset": offset,
                "page": page,
                "per_page": per_page,
            }.items()
            if value is not None
        }
        response = await self._request("GET", "/api/v1/kanban/cards/search", params=params)
        return KanbanCardSearchResponse.model_validate(response)

    async def bulk_move_kanban_cards(self, request_data: KanbanBulkMoveCardsRequest) -> KanbanBulkMoveCardsResponse:
        response = await self._request(
            "POST",
            "/api/v1/kanban/cards/bulk-move",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return KanbanBulkMoveCardsResponse.model_validate(response)

    async def bulk_archive_kanban_cards(self, request_data: KanbanBulkArchiveCardsRequest) -> KanbanBulkArchiveCardsResponse:
        response = await self._request(
            "POST",
            "/api/v1/kanban/cards/bulk-archive",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return KanbanBulkArchiveCardsResponse.model_validate(response)

    async def bulk_unarchive_kanban_cards(self, request_data: KanbanBulkArchiveCardsRequest) -> KanbanBulkUnarchiveCardsResponse:
        response = await self._request(
            "POST",
            "/api/v1/kanban/cards/bulk-unarchive",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return KanbanBulkUnarchiveCardsResponse.model_validate(response)

    async def bulk_delete_kanban_cards(self, request_data: KanbanBulkDeleteCardsRequest) -> KanbanBulkDeleteCardsResponse:
        response = await self._request(
            "POST",
            "/api/v1/kanban/cards/bulk-delete",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return KanbanBulkDeleteCardsResponse.model_validate(response)

    async def bulk_label_kanban_cards(self, request_data: KanbanBulkLabelCardsRequest) -> KanbanBulkLabelCardsResponse:
        response = await self._request(
            "POST",
            "/api/v1/kanban/cards/bulk-label",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return KanbanBulkLabelCardsResponse.model_validate(response)

    async def filter_kanban_board_cards(
        self,
        board_id: int,
        *,
        label_ids: list[int] | None = None,
        priority: str | None = None,
        due_before: str | None = None,
        due_after: str | None = None,
        overdue: bool | None = None,
        has_due_date: bool | None = None,
        has_checklist: bool | None = None,
        is_complete: bool | None = None,
        include_archived: bool = False,
        include_deleted: bool = False,
        limit: int = 50,
        offset: int = 0,
        page: int | None = None,
        per_page: int | None = None,
    ) -> KanbanFilteredCardsResponse:
        params = {
            key: value
            for key, value in {
                "label_ids": ",".join(str(label_id) for label_id in label_ids) if label_ids else None,
                "priority": priority,
                "due_before": due_before,
                "due_after": due_after,
                "overdue": overdue,
                "has_due_date": has_due_date,
                "has_checklist": has_checklist,
                "is_complete": is_complete,
                "include_archived": include_archived,
                "include_deleted": include_deleted,
                "limit": limit,
                "offset": offset,
                "page": page,
                "per_page": per_page,
            }.items()
            if value is not None
        }
        response = await self._request("GET", f"/api/v1/kanban/boards/{board_id}/cards", params=params)
        return KanbanFilteredCardsResponse.model_validate(response)

    async def copy_kanban_card_with_checklists(
        self,
        card_id: int,
        request_data: KanbanCardCopyWithChecklistsRequest,
    ) -> KanbanCardResponse:
        response = await self._request(
            "POST",
            f"/api/v1/kanban/cards/{card_id}/copy-with-checklists",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return KanbanCardResponse.model_validate(response)

    async def create_kanban_label(self, board_id: int, request_data: KanbanLabelCreate) -> KanbanLabelResponse:
        response = await self._request(
            "POST",
            f"/api/v1/kanban/boards/{board_id}/labels",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return KanbanLabelResponse.model_validate(response)

    async def list_kanban_labels(self, board_id: int) -> KanbanLabelsListResponse:
        response = await self._request("GET", f"/api/v1/kanban/boards/{board_id}/labels")
        return KanbanLabelsListResponse.model_validate(response)

    async def get_kanban_label(self, label_id: int) -> KanbanLabelResponse:
        response = await self._request("GET", f"/api/v1/kanban/labels/{label_id}")
        return KanbanLabelResponse.model_validate(response)

    async def update_kanban_label(self, label_id: int, request_data: KanbanLabelUpdate) -> KanbanLabelResponse:
        response = await self._request(
            "PATCH",
            f"/api/v1/kanban/labels/{label_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return KanbanLabelResponse.model_validate(response)

    async def delete_kanban_label(self, label_id: int) -> bool:
        await self._request("DELETE", f"/api/v1/kanban/labels/{label_id}")
        return True

    async def assign_kanban_label_to_card(self, card_id: int, label_id: int) -> KanbanDetailResponse:
        response = await self._request("POST", f"/api/v1/kanban/cards/{card_id}/labels/{label_id}")
        return KanbanDetailResponse.model_validate(response)

    async def remove_kanban_label_from_card(self, card_id: int, label_id: int) -> bool:
        await self._request("DELETE", f"/api/v1/kanban/cards/{card_id}/labels/{label_id}")
        return True

    async def list_kanban_card_labels(self, card_id: int) -> KanbanLabelsListResponse:
        response = await self._request("GET", f"/api/v1/kanban/cards/{card_id}/labels")
        return KanbanLabelsListResponse.model_validate(response)

    async def create_kanban_checklist(self, card_id: int, request_data: KanbanChecklistCreate) -> KanbanChecklistResponse:
        response = await self._request(
            "POST",
            f"/api/v1/kanban/cards/{card_id}/checklists",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return KanbanChecklistResponse.model_validate(response)

    async def list_kanban_checklists(self, card_id: int) -> KanbanChecklistsListResponse:
        response = await self._request("GET", f"/api/v1/kanban/cards/{card_id}/checklists")
        return KanbanChecklistsListResponse.model_validate(response)

    async def get_kanban_checklist(self, checklist_id: int) -> KanbanChecklistWithItemsResponse:
        response = await self._request("GET", f"/api/v1/kanban/checklists/{checklist_id}")
        return KanbanChecklistWithItemsResponse.model_validate(response)

    async def update_kanban_checklist(self, checklist_id: int, request_data: KanbanChecklistUpdate) -> KanbanChecklistResponse:
        response = await self._request(
            "PATCH",
            f"/api/v1/kanban/checklists/{checklist_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return KanbanChecklistResponse.model_validate(response)

    async def delete_kanban_checklist(self, checklist_id: int) -> bool:
        await self._request("DELETE", f"/api/v1/kanban/checklists/{checklist_id}")
        return True

    async def reorder_kanban_checklists(
        self,
        card_id: int,
        request_data: KanbanChecklistReorderRequest,
    ) -> KanbanChecklistsListResponse:
        response = await self._request(
            "POST",
            f"/api/v1/kanban/cards/{card_id}/checklists/reorder",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return KanbanChecklistsListResponse.model_validate(response)

    async def create_kanban_checklist_item(
        self,
        checklist_id: int,
        request_data: KanbanChecklistItemCreate,
    ) -> KanbanChecklistItemResponse:
        response = await self._request(
            "POST",
            f"/api/v1/kanban/checklists/{checklist_id}/items",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return KanbanChecklistItemResponse.model_validate(response)

    async def list_kanban_checklist_items(self, checklist_id: int) -> KanbanChecklistItemsListResponse:
        response = await self._request("GET", f"/api/v1/kanban/checklists/{checklist_id}/items")
        return KanbanChecklistItemsListResponse.model_validate(response)

    async def get_kanban_checklist_item(self, item_id: int) -> KanbanChecklistItemResponse:
        response = await self._request("GET", f"/api/v1/kanban/checklist-items/{item_id}")
        return KanbanChecklistItemResponse.model_validate(response)

    async def update_kanban_checklist_item(
        self,
        item_id: int,
        request_data: KanbanChecklistItemUpdate,
    ) -> KanbanChecklistItemResponse:
        response = await self._request(
            "PATCH",
            f"/api/v1/kanban/checklist-items/{item_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return KanbanChecklistItemResponse.model_validate(response)

    async def delete_kanban_checklist_item(self, item_id: int) -> bool:
        await self._request("DELETE", f"/api/v1/kanban/checklist-items/{item_id}")
        return True

    async def reorder_kanban_checklist_items(
        self,
        checklist_id: int,
        request_data: KanbanChecklistItemReorderRequest,
    ) -> KanbanChecklistItemsListResponse:
        response = await self._request(
            "POST",
            f"/api/v1/kanban/checklists/{checklist_id}/items/reorder",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return KanbanChecklistItemsListResponse.model_validate(response)

    async def check_kanban_checklist_item(self, item_id: int) -> KanbanChecklistItemResponse:
        response = await self._request("POST", f"/api/v1/kanban/checklist-items/{item_id}/check")
        return KanbanChecklistItemResponse.model_validate(response)

    async def uncheck_kanban_checklist_item(self, item_id: int) -> KanbanChecklistItemResponse:
        response = await self._request("POST", f"/api/v1/kanban/checklist-items/{item_id}/uncheck")
        return KanbanChecklistItemResponse.model_validate(response)

    async def toggle_all_kanban_checklist_items(
        self,
        checklist_id: int,
        request_data: KanbanToggleAllChecklistItemsRequest,
    ) -> KanbanChecklistWithItemsResponse:
        response = await self._request(
            "POST",
            f"/api/v1/kanban/checklists/{checklist_id}/toggle-all",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return KanbanChecklistWithItemsResponse.model_validate(response)

    async def create_kanban_comment(self, card_id: int, request_data: KanbanCommentCreate) -> KanbanCommentResponse:
        response = await self._request(
            "POST",
            f"/api/v1/kanban/cards/{card_id}/comments",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return KanbanCommentResponse.model_validate(response)

    async def list_kanban_comments(
        self,
        card_id: int,
        *,
        limit: int = 50,
        offset: int = 0,
        include_deleted: bool = False,
    ) -> KanbanCommentsListResponse:
        response = await self._request(
            "GET",
            f"/api/v1/kanban/cards/{card_id}/comments",
            params={"limit": limit, "offset": offset, "include_deleted": include_deleted},
        )
        return KanbanCommentsListResponse.model_validate(response)

    async def get_kanban_comment(self, comment_id: int, *, include_deleted: bool = False) -> KanbanCommentResponse:
        response = await self._request(
            "GET",
            f"/api/v1/kanban/comments/{comment_id}",
            params={"include_deleted": include_deleted},
        )
        return KanbanCommentResponse.model_validate(response)

    async def update_kanban_comment(self, comment_id: int, request_data: KanbanCommentUpdate) -> KanbanCommentResponse:
        response = await self._request(
            "PATCH",
            f"/api/v1/kanban/comments/{comment_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return KanbanCommentResponse.model_validate(response)

    async def delete_kanban_comment(self, comment_id: int, *, hard_delete: bool = False) -> bool:
        await self._request("DELETE", f"/api/v1/kanban/comments/{comment_id}", params={"hard_delete": hard_delete})
        return True

    async def search_kanban_cards_get(
        self,
        query: str,
        *,
        board_id: int | None = None,
        label_ids: list[int] | None = None,
        priority: str | None = None,
        include_archived: bool = False,
        search_mode: str = "fts",
        limit: int = 20,
        offset: int = 0,
        page: int | None = None,
        per_page: int | None = None,
    ) -> KanbanSearchResponse:
        params = {
            key: value
            for key, value in {
                "q": query,
                "board_id": board_id,
                "label_ids": ",".join(str(label_id) for label_id in label_ids) if label_ids else None,
                "priority": priority,
                "include_archived": include_archived,
                "search_mode": search_mode,
                "limit": limit,
                "offset": offset,
                "page": page,
                "per_page": per_page,
            }.items()
            if value is not None
        }
        response = await self._request("GET", "/api/v1/kanban/search", params=params)
        return KanbanSearchResponse.model_validate(response)

    async def search_kanban_cards(self, request_data: KanbanSearchRequest) -> KanbanSearchResponse:
        response = await self._request(
            "POST",
            "/api/v1/kanban/search",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return KanbanSearchResponse.model_validate(response)

    async def get_kanban_search_status(self) -> Dict[str, Any]:
        return await self._request("GET", "/api/v1/kanban/search/status")

    async def add_kanban_card_link(self, card_id: int, request_data: KanbanCardLinkCreate) -> KanbanCardLinkResponse:
        response = await self._request(
            "POST",
            f"/api/v1/kanban/cards/{card_id}/links",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return KanbanCardLinkResponse.model_validate(response)

    async def list_kanban_card_links(self, card_id: int, *, linked_type: str | None = None) -> KanbanCardLinksListResponse:
        params = {"linked_type": linked_type} if linked_type is not None else None
        response = await self._request("GET", f"/api/v1/kanban/cards/{card_id}/links", params=params)
        return KanbanCardLinksListResponse.model_validate(response)

    async def get_kanban_card_link_counts(self, card_id: int) -> KanbanCardLinkCountsResponse:
        response = await self._request("GET", f"/api/v1/kanban/cards/{card_id}/links/counts")
        return KanbanCardLinkCountsResponse.model_validate(response)

    async def remove_kanban_card_link(self, card_id: int, linked_type: str, linked_id: str) -> KanbanDetailResponse:
        response = await self._request("DELETE", f"/api/v1/kanban/cards/{card_id}/links/{linked_type}/{linked_id}")
        return KanbanDetailResponse.model_validate(response)

    async def remove_kanban_card_link_by_id_for_card(self, card_id: int, link_id: int) -> KanbanDetailResponse:
        response = await self._request("DELETE", f"/api/v1/kanban/cards/{card_id}/links/{link_id}")
        return KanbanDetailResponse.model_validate(response)

    async def remove_kanban_card_link_by_id(self, link_id: int) -> KanbanDetailResponse:
        response = await self._request("DELETE", f"/api/v1/kanban/links/{link_id}")
        return KanbanDetailResponse.model_validate(response)

    async def bulk_add_kanban_card_links(
        self,
        card_id: int,
        request_data: KanbanBulkCardLinksRequest,
    ) -> KanbanBulkCardLinksAddResponse:
        response = await self._request(
            "POST",
            f"/api/v1/kanban/cards/{card_id}/links/bulk-add",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return KanbanBulkCardLinksAddResponse.model_validate(response)

    async def bulk_remove_kanban_card_links(
        self,
        card_id: int,
        request_data: KanbanBulkCardLinksRequest,
    ) -> KanbanBulkCardLinksRemoveResponse:
        response = await self._request(
            "POST",
            f"/api/v1/kanban/cards/{card_id}/links/bulk-remove",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return KanbanBulkCardLinksRemoveResponse.model_validate(response)

    async def list_kanban_cards_by_linked_content(
        self,
        linked_type: str,
        linked_id: str,
        *,
        include_archived: bool = False,
        include_deleted: bool = False,
    ) -> KanbanLinkedCardsListResponse:
        response = await self._request(
            "GET",
            f"/api/v1/kanban/linked/{linked_type}/{linked_id}/cards",
            params={"include_archived": include_archived, "include_deleted": include_deleted},
        )
        return KanbanLinkedCardsListResponse.model_validate(response)

    async def list_connector_providers(self) -> List[ConnectorProvider]:
        response = await self._request("GET", "/api/v1/connectors/providers")
        return [ConnectorProvider.model_validate(item) for item in response]

    async def authorize_connector_provider(
        self,
        provider: str,
        *,
        state: str | None = None,
        scopes: list[str] | str | None = None,
    ) -> AuthorizeURLResponse:
        params: Dict[str, Any] = {}
        if state is not None:
            params["state"] = state
        if scopes is not None:
            params["scopes"] = ",".join(scopes) if isinstance(scopes, list) else scopes
        response = await self._request(
            "POST",
            f"/api/v1/connectors/providers/{provider}/authorize",
            params=params or None,
        )
        return AuthorizeURLResponse.model_validate(response)

    async def complete_connector_oauth_callback(
        self,
        provider: str,
        *,
        code: str | None = None,
        oauth_token: str | None = None,
        oauth_verifier: str | None = None,
        state: str | None = None,
    ) -> ConnectorAccount:
        params = {
            key: value
            for key, value in {
                "code": code,
                "oauth_token": oauth_token,
                "oauth_verifier": oauth_verifier,
                "state": state,
            }.items()
            if value is not None
        }
        response = await self._request(
            "GET",
            f"/api/v1/connectors/providers/{provider}/callback",
            params=params or None,
        )
        return ConnectorAccount.model_validate(response)

    async def list_connector_accounts(self) -> List[ConnectorAccount]:
        response = await self._request("GET", "/api/v1/connectors/accounts")
        return [ConnectorAccount.model_validate(item) for item in response]

    async def delete_connector_account(self, account_id: int) -> bool:
        await self._request("DELETE", f"/api/v1/connectors/accounts/{account_id}")
        return True

    async def browse_connector_sources(
        self,
        provider: str,
        *,
        account_id: int,
        parent_remote_id: str | None = None,
        page_size: int = 50,
        cursor: str | None = None,
    ) -> ConnectorBrowseResponse:
        params = {
            key: value
            for key, value in {
                "account_id": account_id,
                "parent_remote_id": parent_remote_id,
                "page_size": page_size,
                "cursor": cursor,
            }.items()
            if value is not None
        }
        response = await self._request(
            "GET",
            f"/api/v1/connectors/providers/{provider}/sources/browse",
            params=params,
        )
        return ConnectorBrowseResponse.model_validate(response)

    async def create_connector_source(self, request_data: ConnectorSourceCreateRequest) -> ConnectorSource:
        response = await self._request(
            "POST",
            "/api/v1/connectors/sources",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return ConnectorSource.model_validate(response)

    async def list_connector_sources(self) -> List[ConnectorSource]:
        response = await self._request("GET", "/api/v1/connectors/sources")
        return [ConnectorSource.model_validate(item) for item in response]

    async def update_connector_source(
        self,
        source_id: int,
        request_data: ConnectorSourcePatchRequest,
    ) -> ConnectorSource:
        response = await self._request(
            "PATCH",
            f"/api/v1/connectors/sources/{source_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return ConnectorSource.model_validate(response)

    async def import_connector_source(self, source_id: int) -> ConnectorImportJob:
        response = await self._request("POST", f"/api/v1/connectors/sources/{source_id}/import")
        return ConnectorImportJob.model_validate(response)

    async def get_connector_source_sync_status(self, source_id: int) -> ConnectorSourceSyncStatus:
        response = await self._request("GET", f"/api/v1/connectors/sources/{source_id}/sync")
        return ConnectorSourceSyncStatus.model_validate(response)

    async def trigger_connector_source_sync(self, source_id: int) -> ConnectorSourceSyncTriggerResponse:
        response = await self._request("POST", f"/api/v1/connectors/sources/{source_id}/sync")
        return ConnectorSourceSyncTriggerResponse.model_validate(response)

    async def get_connector_job_status(self, job_id: int) -> Dict[str, Any]:
        return await self._request("GET", f"/api/v1/connectors/jobs/{job_id}")

    async def create_chat_grammar(self, request_data: ChatGrammarCreate) -> ChatGrammarResponse:
        response = await self._request(
            "POST",
            "/api/v1/grammars",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return ChatGrammarResponse.model_validate(response)

    async def list_chat_grammars(
        self,
        *,
        include_archived: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> ChatGrammarListResponse:
        response = await self._request(
            "GET",
            "/api/v1/grammars",
            params={"include_archived": include_archived, "limit": limit, "offset": offset},
        )
        return ChatGrammarListResponse.model_validate(response)

    async def get_chat_grammar(self, grammar_id: str, *, include_archived: bool = False) -> ChatGrammarResponse:
        response = await self._request(
            "GET",
            f"/api/v1/grammars/{grammar_id}",
            params={"include_archived": include_archived},
        )
        return ChatGrammarResponse.model_validate(response)

    async def update_chat_grammar(
        self,
        grammar_id: str,
        request_data: ChatGrammarUpdate,
    ) -> ChatGrammarResponse:
        response = await self._request(
            "PATCH",
            f"/api/v1/grammars/{grammar_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return ChatGrammarResponse.model_validate(response)

    async def delete_chat_grammar(self, grammar_id: str, *, hard_delete: bool = False) -> bool:
        await self._request(
            "DELETE",
            f"/api/v1/grammars/{grammar_id}",
            params={"hard_delete": hard_delete},
        )
        return True

    async def submit_explicit_feedback(self, request_data: ExplicitFeedbackRequest) -> ExplicitFeedbackResponse:
        response = await self._request(
            "POST",
            "/api/v1/feedback/explicit",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return ExplicitFeedbackResponse.model_validate(response)

    async def list_feedback(self, conversation_id: str) -> FeedbackListResponse:
        response = await self._request(
            "GET",
            "/api/v1/feedback",
            params={"conversation_id": conversation_id},
        )
        return FeedbackListResponse.model_validate(response)

    async def update_feedback(
        self,
        feedback_id: str,
        request_data: FeedbackUpdateRequest,
    ) -> ExplicitFeedbackResponse:
        response = await self._request(
            "PATCH",
            f"/api/v1/feedback/{feedback_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return ExplicitFeedbackResponse.model_validate(response)

    async def delete_feedback(self, feedback_id: str) -> FeedbackDeleteResponse:
        response = await self._request("DELETE", f"/api/v1/feedback/{feedback_id}")
        return FeedbackDeleteResponse.model_validate(response)

    async def create_collections_feed(self, request_data: CollectionsFeedCreateRequest) -> CollectionsFeed:
        response = await self._request(
            "POST",
            "/api/v1/collections/feeds",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return CollectionsFeed.model_validate(response)

    async def list_collections_feeds(
        self,
        *,
        q: str | None = None,
        page: int = 1,
        size: int = 20,
    ) -> CollectionsFeedsListResponse:
        params = {key: value for key, value in {"q": q, "page": page, "size": size}.items() if value is not None}
        response = await self._request("GET", "/api/v1/collections/feeds", params=params)
        return CollectionsFeedsListResponse.model_validate(response)

    async def get_collections_feed(self, feed_id: int) -> CollectionsFeed:
        response = await self._request("GET", f"/api/v1/collections/feeds/{feed_id}")
        return CollectionsFeed.model_validate(response)

    async def update_collections_feed(
        self,
        feed_id: int,
        request_data: CollectionsFeedUpdateRequest,
    ) -> CollectionsFeed:
        response = await self._request(
            "PATCH",
            f"/api/v1/collections/feeds/{feed_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return CollectionsFeed.model_validate(response)

    async def delete_collections_feed(self, feed_id: int) -> bool:
        await self._request("DELETE", f"/api/v1/collections/feeds/{feed_id}")
        return True

    async def subscribe_collections_feed_websub(
        self,
        feed_id: int,
        request_data: CollectionsWebSubSubscribeRequest | None = None,
    ) -> CollectionsWebSubSubscriptionResponse:
        request = request_data or CollectionsWebSubSubscribeRequest()
        response = await self._request(
            "POST",
            f"/api/v1/collections/feeds/{feed_id}/websub/subscribe",
            json_data=request.model_dump(exclude_none=True, mode="json"),
        )
        return CollectionsWebSubSubscriptionResponse.model_validate(response)

    async def get_collections_feed_websub(self, feed_id: int) -> CollectionsWebSubSubscriptionResponse:
        response = await self._request("GET", f"/api/v1/collections/feeds/{feed_id}/websub")
        return CollectionsWebSubSubscriptionResponse.model_validate(response)

    async def unsubscribe_collections_feed_websub(self, feed_id: int) -> Dict[str, Any]:
        return await self._request("DELETE", f"/api/v1/collections/feeds/{feed_id}/websub")

    async def get_claims_status(self) -> Dict[str, Any]:
        return await self._request("GET", "/api/v1/claims/status")

    async def list_all_claims(
        self,
        *,
        media_id: int | None = None,
        review_status: str | None = None,
        reviewer_id: int | None = None,
        review_group: str | None = None,
        claim_cluster_id: int | None = None,
        limit: int = 100,
        offset: int = 0,
        include_deleted: bool = False,
        user_id: int | None = None,
    ) -> List[Dict[str, Any]]:
        params = {
            key: value
            for key, value in {
                "media_id": media_id,
                "review_status": review_status,
                "reviewer_id": reviewer_id,
                "review_group": review_group,
                "claim_cluster_id": claim_cluster_id,
                "limit": limit,
                "offset": offset,
                "include_deleted": include_deleted,
                "user_id": user_id,
            }.items()
            if value is not None
        }
        return await self._request("GET", "/api/v1/claims", params=params)

    async def get_claims_settings(self) -> ClaimsSettingsResponse:
        response = await self._request("GET", "/api/v1/claims/settings")
        return ClaimsSettingsResponse.model_validate(response)

    async def update_claims_settings(self, request_data: ClaimsSettingsUpdate) -> ClaimsSettingsResponse:
        response = await self._request(
            "PUT",
            "/api/v1/claims/settings",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return ClaimsSettingsResponse.model_validate(response)

    async def get_claims_monitoring_config(self) -> ClaimsMonitoringSettingsResponse:
        response = await self._request("GET", "/api/v1/claims/monitoring/config")
        return ClaimsMonitoringSettingsResponse.model_validate(response)

    async def update_claims_monitoring_config(
        self,
        request_data: ClaimsMonitoringSettingsUpdate,
    ) -> ClaimsMonitoringSettingsResponse:
        response = await self._request(
            "PATCH",
            "/api/v1/claims/monitoring/config",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return ClaimsMonitoringSettingsResponse.model_validate(response)

    async def list_claim_notifications(
        self,
        *,
        kind: str | None = None,
        target_user_id: str | None = None,
        target_review_group: str | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        delivered: bool | None = None,
        limit: int = 100,
        offset: int = 0,
        user_id: int | None = None,
    ) -> List[ClaimNotificationResponse]:
        params = {
            key: value
            for key, value in {
                "kind": kind,
                "target_user_id": target_user_id,
                "target_review_group": target_review_group,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "delivered": delivered,
                "limit": limit,
                "offset": offset,
                "user_id": user_id,
            }.items()
            if value is not None
        }
        response = await self._request("GET", "/api/v1/claims/notifications", params=params)
        return [ClaimNotificationResponse.model_validate(item) for item in response]

    async def get_claim_notifications_digest(
        self,
        *,
        kind: str | None = None,
        target_user_id: str | None = None,
        target_review_group: str | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        delivered: bool | None = None,
        include_items: bool = False,
        ack: bool = False,
        limit: int = 200,
        offset: int = 0,
        user_id: int | None = None,
    ) -> ClaimNotificationsDigestResponse:
        params = {
            key: value
            for key, value in {
                "kind": kind,
                "target_user_id": target_user_id,
                "target_review_group": target_review_group,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "delivered": delivered,
                "include_items": include_items,
                "ack": ack,
                "limit": limit,
                "offset": offset,
                "user_id": user_id,
            }.items()
            if value is not None
        }
        response = await self._request("GET", "/api/v1/claims/notifications/digest", params=params)
        return ClaimNotificationsDigestResponse.model_validate(response)

    async def ack_claim_notifications(self, request_data: ClaimNotificationsAckRequest) -> Dict[str, Any]:
        return await self._request(
            "POST",
            "/api/v1/claims/notifications/ack",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )

    async def evaluate_claim_watchlist_notifications(self, *, user_id: int | None = None) -> Dict[str, Any]:
        params = {"user_id": user_id} if user_id is not None else None
        return await self._request("POST", "/api/v1/claims/notifications/watchlists/evaluate", params=params)

    async def list_claim_alerts(self, *, user_id: int | None = None) -> List[ClaimsAlertConfigResponse]:
        params = {"user_id": user_id} if user_id is not None else None
        response = await self._request("GET", "/api/v1/claims/alerts", params=params)
        return [ClaimsAlertConfigResponse.model_validate(item) for item in response]

    async def create_claim_alert(
        self,
        request_data: ClaimsAlertConfigCreate,
        *,
        user_id: int | None = None,
    ) -> ClaimsAlertConfigResponse:
        response = await self._request(
            "POST",
            "/api/v1/claims/alerts",
            params={"user_id": user_id} if user_id is not None else None,
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return ClaimsAlertConfigResponse.model_validate(response)

    async def update_claim_alert(
        self,
        config_id: int,
        request_data: ClaimsAlertConfigUpdate,
    ) -> ClaimsAlertConfigResponse:
        response = await self._request(
            "PATCH",
            f"/api/v1/claims/alerts/{config_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return ClaimsAlertConfigResponse.model_validate(response)

    async def delete_claim_alert(self, config_id: int) -> Dict[str, Any]:
        return await self._request("DELETE", f"/api/v1/claims/alerts/{config_id}")

    async def evaluate_claim_alerts(
        self,
        *,
        window_sec: int = 3600,
        baseline_sec: int = 86400,
        user_id: int | None = None,
    ) -> Dict[str, Any]:
        params = {
            key: value
            for key, value in {"window_sec": window_sec, "baseline_sec": baseline_sec, "user_id": user_id}.items()
            if value is not None
        }
        return await self._request("POST", "/api/v1/claims/alerts/evaluate", params=params)

    async def get_claims_rebuild_health(self) -> Dict[str, Any]:
        return await self._request("GET", "/api/v1/claims/rebuild/health")

    async def get_claim_review_queue(
        self,
        *,
        status_filter: str | None = None,
        reviewer_id: int | None = None,
        review_group: str | None = None,
        media_id: int | None = None,
        extractor: str | None = None,
        limit: int = 100,
        offset: int = 0,
        include_deleted: bool = False,
        user_id: int | None = None,
    ) -> List[Dict[str, Any]]:
        params = {
            key: value
            for key, value in {
                "status_filter": status_filter,
                "reviewer_id": reviewer_id,
                "review_group": review_group,
                "media_id": media_id,
                "extractor": extractor,
                "limit": limit,
                "offset": offset,
                "include_deleted": include_deleted,
                "user_id": user_id,
            }.items()
            if value is not None
        }
        return await self._request("GET", "/api/v1/claims/review-queue", params=params)

    async def review_claim(
        self,
        claim_id: int,
        request_data: ClaimReviewRequest,
        *,
        user_id: int | None = None,
    ) -> Dict[str, Any]:
        return await self._request(
            "PATCH",
            f"/api/v1/claims/{claim_id}/review",
            params={"user_id": user_id} if user_id is not None else None,
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )

    async def get_claim_review_history(
        self,
        claim_id: int,
        *,
        user_id: int | None = None,
    ) -> List[Dict[str, Any]]:
        params = {"user_id": user_id} if user_id is not None else None
        return await self._request("GET", f"/api/v1/claims/{claim_id}/history", params=params)

    async def bulk_review_claims(
        self,
        request_data: ClaimReviewBulkRequest,
        *,
        user_id: int | None = None,
    ) -> Dict[str, Any]:
        return await self._request(
            "POST",
            "/api/v1/claims/review/bulk",
            params={"user_id": user_id} if user_id is not None else None,
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )

    async def list_claim_review_rules(
        self,
        *,
        user_id: int | None = None,
        active_only: bool = False,
    ) -> List[Dict[str, Any]]:
        params = {
            key: value for key, value in {"user_id": user_id, "active_only": active_only}.items() if value is not None
        }
        return await self._request("GET", "/api/v1/claims/review/rules", params=params)

    async def create_claim_review_rule(
        self,
        request_data: ClaimReviewRuleCreate,
        *,
        user_id: int | None = None,
    ) -> Dict[str, Any]:
        return await self._request(
            "POST",
            "/api/v1/claims/review/rules",
            params={"user_id": user_id} if user_id is not None else None,
            json_data=request_data.model_dump(exclude_unset=True, mode="json"),
        )

    async def update_claim_review_rule(
        self,
        rule_id: int,
        request_data: ClaimReviewRuleUpdate,
    ) -> Dict[str, Any]:
        return await self._request(
            "PATCH",
            f"/api/v1/claims/review/rules/{rule_id}",
            json_data=request_data.model_dump(exclude_unset=True, mode="json"),
        )

    async def delete_claim_review_rule(self, rule_id: int) -> Dict[str, Any]:
        return await self._request("DELETE", f"/api/v1/claims/review/rules/{rule_id}")

    async def get_claim_review_analytics(self) -> Dict[str, Any]:
        return await self._request("GET", "/api/v1/claims/review/analytics")

    async def list_claim_extractors(self) -> ClaimsExtractorCatalogResponse:
        response = await self._request("GET", "/api/v1/claims/extractors")
        return ClaimsExtractorCatalogResponse.model_validate(response)

    async def list_claim_review_metrics(
        self,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        extractor: str | None = None,
        extractor_version: str | None = None,
        user_id: int | None = None,
        limit: int = 500,
        offset: int = 0,
    ) -> ClaimsReviewExtractorMetricsResponse:
        params = {
            key: value
            for key, value in {
                "start_date": start_date,
                "end_date": end_date,
                "extractor": extractor,
                "extractor_version": extractor_version,
                "user_id": user_id,
                "limit": limit,
                "offset": offset,
            }.items()
            if value is not None
        }
        response = await self._request("GET", "/api/v1/claims/review/metrics", params=params)
        return ClaimsReviewExtractorMetricsResponse.model_validate(response)

    async def get_claims_analytics_dashboard(
        self,
        *,
        window_days: int = 7,
        window_sec: int = 3600,
        baseline_sec: int = 86400,
    ) -> ClaimsAnalyticsDashboardResponse:
        response = await self._request(
            "GET",
            "/api/v1/claims/analytics/dashboard",
            params={"window_days": window_days, "window_sec": window_sec, "baseline_sec": baseline_sec},
        )
        return ClaimsAnalyticsDashboardResponse.model_validate(response)

    async def export_claims_analytics(self, request_data: ClaimsAnalyticsExportRequest) -> ClaimsAnalyticsExportResponse:
        response = await self._request(
            "POST",
            "/api/v1/claims/analytics/export",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return ClaimsAnalyticsExportResponse.model_validate(response)

    async def list_claims_analytics_exports(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        status: str | None = None,
        format_filter: str | None = None,
        workspace_id: str | None = None,
    ) -> ClaimsAnalyticsExportListResponse:
        params = {
            key: value
            for key, value in {
                "limit": limit,
                "offset": offset,
                "status": status,
                "format": format_filter,
                "workspace_id": workspace_id,
            }.items()
            if value is not None
        }
        response = await self._request("GET", "/api/v1/claims/analytics/exports", params=params)
        return ClaimsAnalyticsExportListResponse.model_validate(response)

    async def download_claims_analytics_export(self, export_id: str) -> Any:
        return await self._request("GET", f"/api/v1/claims/analytics/export/{export_id}")

    async def download_claims_analytics_export_file(self, export_id: str) -> ReadingExportResponse:
        return await self._binary_request("GET", f"/api/v1/claims/analytics/export/{export_id}")

    async def list_claim_clusters(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        since: str | None = None,
        keyword: str | None = None,
        min_size: int | None = None,
        watchlisted: bool | None = None,
        user_id: int | None = None,
    ) -> List[Dict[str, Any]]:
        params = {
            key: value
            for key, value in {
                "limit": limit,
                "offset": offset,
                "since": since,
                "keyword": keyword,
                "min_size": min_size,
                "watchlisted": watchlisted,
                "user_id": user_id,
            }.items()
            if value is not None
        }
        return await self._request("GET", "/api/v1/claims/clusters", params=params)

    async def rebuild_claim_clusters(
        self,
        *,
        min_size: int = 2,
        method: str | None = None,
        similarity_threshold: float | None = None,
        user_id: int | None = None,
    ) -> Dict[str, Any]:
        params = {
            key: value
            for key, value in {
                "min_size": min_size,
                "method": method,
                "similarity_threshold": similarity_threshold,
                "user_id": user_id,
            }.items()
            if value is not None
        }
        return await self._request("POST", "/api/v1/claims/clusters/rebuild", params=params)

    async def get_claim_cluster(self, cluster_id: int) -> Dict[str, Any]:
        return await self._request("GET", f"/api/v1/claims/clusters/{cluster_id}")

    async def list_claim_cluster_links(
        self,
        cluster_id: int,
        *,
        direction: str = "both",
    ) -> List[ClaimsClusterLinkResponse]:
        response = await self._request(
            "GET",
            f"/api/v1/claims/clusters/{cluster_id}/links",
            params={"direction": direction},
        )
        return [ClaimsClusterLinkResponse.model_validate(item) for item in response]

    async def create_claim_cluster_link(
        self,
        cluster_id: int,
        request_data: ClaimsClusterLinkCreate,
    ) -> ClaimsClusterLinkResponse:
        response = await self._request(
            "POST",
            f"/api/v1/claims/clusters/{cluster_id}/links",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return ClaimsClusterLinkResponse.model_validate(response)

    async def delete_claim_cluster_link(self, cluster_id: int, child_cluster_id: int) -> Dict[str, Any]:
        return await self._request("DELETE", f"/api/v1/claims/clusters/{cluster_id}/links/{child_cluster_id}")

    async def list_claim_cluster_members(
        self,
        cluster_id: int,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        return await self._request(
            "GET",
            f"/api/v1/claims/clusters/{cluster_id}/members",
            params={"limit": limit, "offset": offset},
        )

    async def get_claim_cluster_timeline(
        self,
        cluster_id: int,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        return await self._request(
            "GET",
            f"/api/v1/claims/clusters/{cluster_id}/timeline",
            params={"limit": limit, "offset": offset},
        )

    async def get_claim_cluster_evidence(
        self,
        cluster_id: int,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        return await self._request(
            "GET",
            f"/api/v1/claims/clusters/{cluster_id}/evidence",
            params={"limit": limit, "offset": offset},
        )

    async def search_claims(
        self,
        q: str,
        *,
        limit: int = 50,
        offset: int = 0,
        group_by_cluster: bool = False,
        user_id: int | None = None,
    ) -> ClaimsSearchResponse:
        params = {
            key: value
            for key, value in {
                "q": q,
                "limit": limit,
                "offset": offset,
                "group_by_cluster": group_by_cluster,
                "user_id": user_id,
            }.items()
            if value is not None
        }
        response = await self._request("GET", "/api/v1/claims/search", params=params)
        return ClaimsSearchResponse.model_validate(response)

    async def list_claims_for_media(
        self,
        media_id: int,
        *,
        limit: int = 100,
        offset: int = 0,
        envelope: bool = False,
        absolute_links: bool = False,
        user_id: int | None = None,
    ) -> Any:
        params = {
            key: value
            for key, value in {
                "limit": limit,
                "offset": offset,
                "envelope": envelope,
                "absolute_links": absolute_links,
                "user_id": user_id,
            }.items()
            if value is not None
        }
        return await self._request("GET", f"/api/v1/claims/{media_id}", params=params)

    async def get_claim_item(
        self,
        claim_id: int,
        *,
        include_deleted: bool = False,
        user_id: int | None = None,
    ) -> Dict[str, Any]:
        params = {
            key: value
            for key, value in {"include_deleted": include_deleted, "user_id": user_id}.items()
            if value is not None
        }
        return await self._request("GET", f"/api/v1/claims/items/{claim_id}", params=params)

    async def update_claim_item(
        self,
        claim_id: int,
        request_data: ClaimUpdateRequest,
        *,
        user_id: int | None = None,
    ) -> Dict[str, Any]:
        return await self._request(
            "PATCH",
            f"/api/v1/claims/items/{claim_id}",
            params={"user_id": user_id} if user_id is not None else None,
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )

    async def rebuild_claims_for_media(self, media_id: int, *, user_id: int | None = None) -> Dict[str, Any]:
        params = {"user_id": user_id} if user_id is not None else None
        return await self._request("POST", f"/api/v1/claims/{media_id}/rebuild", params=params)

    async def rebuild_all_claims(
        self,
        *,
        policy: str = "missing",
        user_id: int | None = None,
    ) -> Dict[str, Any]:
        params = {key: value for key, value in {"policy": policy, "user_id": user_id}.items() if value is not None}
        return await self._request("POST", "/api/v1/claims/rebuild/all", params=params)

    async def rebuild_claims_fts(self, *, user_id: int | None = None) -> Dict[str, Any]:
        params = {"user_id": user_id} if user_id is not None else None
        return await self._request("POST", "/api/v1/claims/rebuild_fts", params=params)

    async def verify_claims_fva(
        self,
        request_data: FVAVerifyRequest,
        *,
        user_id: int | None = None,
    ) -> FVAVerifyResponse:
        response = await self._request(
            "POST",
            "/api/v1/claims/verify/fva",
            params={"user_id": user_id} if user_id is not None else None,
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return FVAVerifyResponse.model_validate(response)

    async def get_fva_settings(self) -> FVASettingsResponse:
        response = await self._request("GET", "/api/v1/claims/verify/fva/settings")
        return FVASettingsResponse.model_validate(response)

    async def list_skills(
        self,
        *,
        include_hidden: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> SkillsListResponse:
        response = await self._request(
            "GET",
            "/api/v1/skills/",
            params={"include_hidden": include_hidden, "limit": limit, "offset": offset},
        )
        return SkillsListResponse.model_validate(response)

    async def get_skills_context(self) -> SkillContextPayload:
        response = await self._request("GET", "/api/v1/skills/context")
        return SkillContextPayload.model_validate(response)

    async def get_skill(self, skill_name: str) -> SkillResponse:
        response = await self._request("GET", f"/api/v1/skills/{skill_name}")
        return SkillResponse.model_validate(response)

    async def create_skill(self, request_data: SkillCreate) -> SkillResponse:
        response = await self._request(
            "POST",
            "/api/v1/skills/",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return SkillResponse.model_validate(response)

    async def update_skill(
        self,
        skill_name: str,
        request_data: SkillUpdate,
        *,
        expected_version: int | None = None,
    ) -> SkillResponse:
        response = await self._request(
            "PUT",
            f"/api/v1/skills/{skill_name}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
            headers={"If-Match": str(expected_version)} if expected_version is not None else None,
        )
        return SkillResponse.model_validate(response)

    async def delete_skill(self, skill_name: str, *, expected_version: int | None = None) -> bool:
        await self._request(
            "DELETE",
            f"/api/v1/skills/{skill_name}",
            headers={"If-Match": str(expected_version)} if expected_version is not None else None,
        )
        return True

    async def import_skill(self, request_data: SkillImportRequest) -> SkillResponse:
        response = await self._request(
            "POST",
            "/api/v1/skills/import",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return SkillResponse.model_validate(response)

    async def import_skill_file(
        self,
        file_content: bytes,
        *,
        filename: str = "SKILL.md",
        content_type: str = "text/markdown",
        overwrite: bool = False,
    ) -> SkillResponse:
        response = await self._request(
            "POST",
            "/api/v1/skills/import/file",
            files=[("file", (filename, file_content, content_type))],
            params={"overwrite": overwrite},
        )
        return SkillResponse.model_validate(response)

    async def export_skill(self, skill_name: str) -> ReadingExportResponse:
        return await self._binary_request("GET", f"/api/v1/skills/{skill_name}/export")

    async def execute_skill(
        self,
        skill_name: str,
        request_data: SkillExecuteRequest | None = None,
    ) -> SkillExecutionResult:
        payload = request_data or SkillExecuteRequest()
        response = await self._request(
            "POST",
            f"/api/v1/skills/{skill_name}/execute",
            json_data=payload.model_dump(exclude_none=True, mode="json"),
        )
        return SkillExecutionResult.model_validate(response)

    async def seed_builtin_skills(self, *, overwrite: bool = False) -> Dict[str, Any]:
        return await self._request(
            "POST",
            "/api/v1/skills/seed",
            params={"overwrite": overwrite},
        )

    async def list_server_tools(self) -> ToolListResponse:
        response = await self._request("GET", "/api/v1/tools")
        return ToolListResponse.model_validate(response)

    async def execute_server_tool(self, request_data: ExecuteToolRequest) -> ExecuteToolResult:
        response = await self._request(
            "POST",
            "/api/v1/tools/execute",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )
        return ExecuteToolResult.model_validate(response)

    @staticmethod
    def _mcp_payload(request_data: Any, *, exclude_unset: bool = False) -> Dict[str, Any]:
        if hasattr(request_data, "model_dump"):
            return request_data.model_dump(
                exclude_none=True,
                exclude_unset=exclude_unset,
                mode="json",
            )
        if isinstance(request_data, dict):
            return {key: value for key, value in request_data.items() if value is not None}
        return dict(request_data or {})

    @staticmethod
    def _mcp_params(**kwargs: Any) -> Dict[str, Any]:
        return {key: value for key, value in kwargs.items() if value is not None}

    async def list_mcp_tool_registry(self) -> list[MCPGovernanceObject]:
        response = await self._request("GET", "/api/v1/mcp/hub/tool-registry")
        return [MCPGovernanceObject.model_validate(item) for item in response]

    async def list_mcp_tool_registry_modules(self) -> list[MCPGovernanceObject]:
        response = await self._request("GET", "/api/v1/mcp/hub/tool-registry/modules")
        return [MCPGovernanceObject.model_validate(item) for item in response]

    async def get_mcp_tool_registry_summary(self) -> MCPGovernanceSummary:
        response = await self._request("GET", "/api/v1/mcp/hub/tool-registry/summary")
        return MCPGovernanceSummary.model_validate(response)

    async def list_mcp_capability_mappings(
        self,
        *,
        owner_scope_type: str | None = None,
        owner_scope_id: int | None = None,
    ) -> list[MCPGovernanceObject]:
        response = await self._request(
            "GET",
            "/api/v1/mcp/hub/capability-mappings",
            params=self._mcp_params(owner_scope_type=owner_scope_type, owner_scope_id=owner_scope_id),
        )
        return [MCPGovernanceObject.model_validate(item) for item in response]

    async def preview_mcp_capability_mapping(
        self,
        request_data: MCPCapabilityMappingCreate | Dict[str, Any],
    ) -> MCPGovernanceObject:
        response = await self._request(
            "POST",
            "/api/v1/mcp/hub/capability-mappings/preview",
            json_data=self._mcp_payload(request_data),
        )
        return MCPGovernanceObject.model_validate(response)

    async def create_mcp_capability_mapping(
        self,
        request_data: MCPCapabilityMappingCreate | Dict[str, Any],
    ) -> MCPGovernanceObject:
        response = await self._request(
            "POST",
            "/api/v1/mcp/hub/capability-mappings",
            json_data=self._mcp_payload(request_data),
        )
        return MCPGovernanceObject.model_validate(response)

    async def update_mcp_capability_mapping(
        self,
        capability_adapter_mapping_id: int,
        request_data: MCPCapabilityMappingUpdate | Dict[str, Any],
    ) -> MCPGovernanceObject:
        response = await self._request(
            "PUT",
            f"/api/v1/mcp/hub/capability-mappings/{capability_adapter_mapping_id}",
            json_data=self._mcp_payload(request_data, exclude_unset=True),
        )
        return MCPGovernanceObject.model_validate(response)

    async def delete_mcp_capability_mapping(self, capability_adapter_mapping_id: int) -> Dict[str, Any]:
        return await self._request("DELETE", f"/api/v1/mcp/hub/capability-mappings/{capability_adapter_mapping_id}")

    async def list_mcp_external_servers(
        self,
        *,
        owner_scope_type: str | None = None,
        owner_scope_id: int | None = None,
    ) -> list[MCPGovernanceObject]:
        response = await self._request(
            "GET",
            "/api/v1/mcp/hub/external-servers",
            params=self._mcp_params(owner_scope_type=owner_scope_type, owner_scope_id=owner_scope_id),
        )
        return [MCPGovernanceObject.model_validate(item) for item in response]

    async def create_mcp_external_server(
        self,
        request_data: MCPExternalServerCreate | Dict[str, Any],
    ) -> MCPGovernanceObject:
        response = await self._request(
            "POST",
            "/api/v1/mcp/hub/external-servers",
            json_data=self._mcp_payload(request_data),
        )
        return MCPGovernanceObject.model_validate(response)

    async def import_mcp_external_server(self, server_id: str) -> MCPGovernanceObject:
        response = await self._request(
            "POST",
            f"/api/v1/mcp/hub/external-servers/{quote(server_id, safe='')}/import",
        )
        return MCPGovernanceObject.model_validate(response)

    async def update_mcp_external_server(
        self,
        server_id: str,
        request_data: MCPExternalServerUpdate | Dict[str, Any],
    ) -> MCPGovernanceObject:
        response = await self._request(
            "PUT",
            f"/api/v1/mcp/hub/external-servers/{quote(server_id, safe='')}",
            json_data=self._mcp_payload(request_data, exclude_unset=True),
        )
        return MCPGovernanceObject.model_validate(response)

    async def delete_mcp_external_server(self, server_id: str) -> Dict[str, Any]:
        return await self._request("DELETE", f"/api/v1/mcp/hub/external-servers/{quote(server_id, safe='')}")

    async def set_mcp_external_server_secret(
        self,
        server_id: str,
        request_data: MCPSecretSetRequest | Dict[str, Any],
    ) -> MCPGovernanceObject:
        response = await self._request(
            "POST",
            f"/api/v1/mcp/hub/external-servers/{quote(server_id, safe='')}/secret",
            json_data=self._mcp_payload(request_data),
        )
        return MCPGovernanceObject.model_validate(response)

    async def list_mcp_permission_profiles(
        self,
        *,
        owner_scope_type: str | None = None,
        owner_scope_id: int | None = None,
    ) -> list[MCPGovernanceObject]:
        response = await self._request(
            "GET",
            "/api/v1/mcp/hub/permission-profiles",
            params=self._mcp_params(owner_scope_type=owner_scope_type, owner_scope_id=owner_scope_id),
        )
        return [MCPGovernanceObject.model_validate(item) for item in response]

    async def create_mcp_permission_profile(
        self,
        request_data: MCPPermissionProfileCreate | Dict[str, Any],
    ) -> MCPGovernanceObject:
        response = await self._request(
            "POST",
            "/api/v1/mcp/hub/permission-profiles",
            json_data=self._mcp_payload(request_data),
        )
        return MCPGovernanceObject.model_validate(response)

    async def update_mcp_permission_profile(
        self,
        profile_id: int,
        request_data: MCPPermissionProfileUpdate | Dict[str, Any],
    ) -> MCPGovernanceObject:
        response = await self._request(
            "PUT",
            f"/api/v1/mcp/hub/permission-profiles/{profile_id}",
            json_data=self._mcp_payload(request_data, exclude_unset=True),
        )
        return MCPGovernanceObject.model_validate(response)

    async def delete_mcp_permission_profile(self, profile_id: int) -> Dict[str, Any]:
        return await self._request("DELETE", f"/api/v1/mcp/hub/permission-profiles/{profile_id}")

    async def list_mcp_policy_assignments(
        self,
        *,
        owner_scope_type: str | None = None,
        owner_scope_id: int | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
    ) -> list[MCPGovernanceObject]:
        response = await self._request(
            "GET",
            "/api/v1/mcp/hub/policy-assignments",
            params=self._mcp_params(
                owner_scope_type=owner_scope_type,
                owner_scope_id=owner_scope_id,
                target_type=target_type,
                target_id=target_id,
            ),
        )
        return [MCPGovernanceObject.model_validate(item) for item in response]

    async def create_mcp_policy_assignment(
        self,
        request_data: MCPPolicyAssignmentCreate | Dict[str, Any],
    ) -> MCPGovernanceObject:
        response = await self._request(
            "POST",
            "/api/v1/mcp/hub/policy-assignments",
            json_data=self._mcp_payload(request_data),
        )
        return MCPGovernanceObject.model_validate(response)

    async def update_mcp_policy_assignment(
        self,
        assignment_id: int,
        request_data: MCPPolicyAssignmentUpdate | Dict[str, Any],
    ) -> MCPGovernanceObject:
        response = await self._request(
            "PUT",
            f"/api/v1/mcp/hub/policy-assignments/{assignment_id}",
            json_data=self._mcp_payload(request_data, exclude_unset=True),
        )
        return MCPGovernanceObject.model_validate(response)

    async def delete_mcp_policy_assignment(self, assignment_id: int) -> Dict[str, Any]:
        return await self._request("DELETE", f"/api/v1/mcp/hub/policy-assignments/{assignment_id}")

    async def list_mcp_approval_policies(
        self,
        *,
        owner_scope_type: str | None = None,
        owner_scope_id: int | None = None,
    ) -> list[MCPGovernanceObject]:
        response = await self._request(
            "GET",
            "/api/v1/mcp/hub/approval-policies",
            params=self._mcp_params(owner_scope_type=owner_scope_type, owner_scope_id=owner_scope_id),
        )
        return [MCPGovernanceObject.model_validate(item) for item in response]

    async def create_mcp_approval_policy(
        self,
        request_data: MCPApprovalPolicyCreate | Dict[str, Any],
    ) -> MCPGovernanceObject:
        response = await self._request(
            "POST",
            "/api/v1/mcp/hub/approval-policies",
            json_data=self._mcp_payload(request_data),
        )
        return MCPGovernanceObject.model_validate(response)

    async def update_mcp_approval_policy(
        self,
        approval_policy_id: int,
        request_data: MCPApprovalPolicyUpdate | Dict[str, Any],
    ) -> MCPGovernanceObject:
        response = await self._request(
            "PUT",
            f"/api/v1/mcp/hub/approval-policies/{approval_policy_id}",
            json_data=self._mcp_payload(request_data, exclude_unset=True),
        )
        return MCPGovernanceObject.model_validate(response)

    async def delete_mcp_approval_policy(self, approval_policy_id: int) -> Dict[str, Any]:
        return await self._request("DELETE", f"/api/v1/mcp/hub/approval-policies/{approval_policy_id}")

    async def create_mcp_approval_decision(
        self,
        request_data: MCPApprovalDecisionCreate | Dict[str, Any],
    ) -> MCPGovernanceObject:
        response = await self._request(
            "POST",
            "/api/v1/mcp/hub/approval-decisions",
            json_data=self._mcp_payload(request_data),
        )
        return MCPGovernanceObject.model_validate(response)

    async def get_mcp_effective_policy(
        self,
        *,
        persona_id: str | None = None,
        group_id: str | None = None,
        org_id: int | None = None,
        team_id: int | None = None,
    ) -> MCPEffectivePolicyResponse:
        response = await self._request(
            "GET",
            "/api/v1/mcp/hub/effective-policy",
            params=self._mcp_params(persona_id=persona_id, group_id=group_id, org_id=org_id, team_id=team_id),
        )
        return MCPEffectivePolicyResponse.model_validate(response)

    async def list_mcp_org_tool_catalogs(self, org_id: int) -> list[MCPGovernanceObject]:
        response = await self._request("GET", f"/api/v1/orgs/{org_id}/mcp/tool_catalogs")
        return [MCPGovernanceObject.model_validate(item) for item in response]

    async def create_mcp_org_tool_catalog(
        self,
        org_id: int,
        request_data: MCPCatalogCreate | Dict[str, Any],
    ) -> MCPGovernanceObject:
        response = await self._request(
            "POST",
            f"/api/v1/orgs/{org_id}/mcp/tool_catalogs",
            json_data=self._mcp_payload(request_data),
        )
        return MCPGovernanceObject.model_validate(response)

    async def delete_mcp_org_tool_catalog(self, org_id: int, catalog_id: int) -> Dict[str, Any]:
        return await self._request("DELETE", f"/api/v1/orgs/{org_id}/mcp/tool_catalogs/{catalog_id}")

    async def add_mcp_org_catalog_entry(
        self,
        org_id: int,
        catalog_id: int,
        request_data: MCPCatalogEntryCreate | Dict[str, Any],
    ) -> MCPGovernanceObject:
        response = await self._request(
            "POST",
            f"/api/v1/orgs/{org_id}/mcp/tool_catalogs/{catalog_id}/entries",
            json_data=self._mcp_payload(request_data),
        )
        return MCPGovernanceObject.model_validate(response)

    async def delete_mcp_org_catalog_entry(self, org_id: int, catalog_id: int, tool_name: str) -> Dict[str, Any]:
        return await self._request(
            "DELETE",
            f"/api/v1/orgs/{org_id}/mcp/tool_catalogs/{catalog_id}/entries/{quote(tool_name, safe='')}",
        )

    async def list_mcp_team_tool_catalogs(self, team_id: int) -> list[MCPGovernanceObject]:
        response = await self._request("GET", f"/api/v1/teams/{team_id}/mcp/tool_catalogs")
        return [MCPGovernanceObject.model_validate(item) for item in response]

    async def create_mcp_team_tool_catalog(
        self,
        team_id: int,
        request_data: MCPCatalogCreate | Dict[str, Any],
    ) -> MCPGovernanceObject:
        response = await self._request(
            "POST",
            f"/api/v1/teams/{team_id}/mcp/tool_catalogs",
            json_data=self._mcp_payload(request_data),
        )
        return MCPGovernanceObject.model_validate(response)

    async def delete_mcp_team_tool_catalog(self, team_id: int, catalog_id: int) -> Dict[str, Any]:
        return await self._request("DELETE", f"/api/v1/teams/{team_id}/mcp/tool_catalogs/{catalog_id}")

    async def add_mcp_team_catalog_entry(
        self,
        team_id: int,
        catalog_id: int,
        request_data: MCPCatalogEntryCreate | Dict[str, Any],
    ) -> MCPGovernanceObject:
        response = await self._request(
            "POST",
            f"/api/v1/teams/{team_id}/mcp/tool_catalogs/{catalog_id}/entries",
            json_data=self._mcp_payload(request_data),
        )
        return MCPGovernanceObject.model_validate(response)

    async def delete_mcp_team_catalog_entry(self, team_id: int, catalog_id: int, tool_name: str) -> Dict[str, Any]:
        return await self._request(
            "DELETE",
            f"/api/v1/teams/{team_id}/mcp/tool_catalogs/{catalog_id}/entries/{quote(tool_name, safe='')}",
        )

    async def query_text2sql(self, request_data: Text2SQLRequest) -> Text2SQLResponse:
        response = await self._request(
            "POST",
            "/api/v1/text2sql/query",
            json_data=request_data.model_dump(mode="json"),
        )
        return Text2SQLResponse.model_validate(response)

    async def send_sync_changes(self, request_data: ClientChangesPayload) -> Dict[str, Any]:
        return await self._request(
            "POST",
            "/api/v1/sync/send",
            json_data=request_data.model_dump(mode="json"),
        )

    async def get_sync_changes(
        self,
        *,
        client_id: str,
        since_change_id: int = 0,
    ) -> ServerChangesResponse:
        response = await self._request(
            "GET",
            "/api/v1/sync/get",
            params={"client_id": client_id, "since_change_id": since_change_id},
        )
        return ServerChangesResponse.model_validate(response)

#
# End of client.py
########################################################################################################################
