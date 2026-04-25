# tldw_chatbook/tldw_api/client.py
#
#
from __future__ import annotations

# Imports
import json # For MediaWiki streaming
from pathlib import Path # For utils.prepare_files_for_httpx
from typing import Optional, Dict, Any, List, AsyncGenerator, Union, Literal
#
# 3rd-party Libraries
import httpx
#
# Local Imports
from .schemas import (
    ProcessVideoRequest, ProcessAudioRequest, ProcessPDFRequest,
    ProcessEbookRequest, ProcessDocumentRequest, ProcessXMLRequest, ProcessMediaWikiRequest,
    ProcessPlaintextRequest,
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
    DocumentAnnotationCreate,
    DocumentAnnotationListResponse,
    DocumentAnnotationResponse,
    DocumentAnnotationSyncRequest,
    DocumentAnnotationSyncResponse,
    DocumentAnnotationUpdate,
    DocumentFiguresResponse,
    DocumentInsightsRequest,
    DocumentInsightsResponse,
    DocumentOutlineResponse,
    DocumentReferencesResponse,
    ExportMode,
    FileArtifactsPurgeRequest,
    FileCreateRequest,
    IngestionSourceCreateRequest,
    IngestionSourceItemListResponse,
    IngestionSourceItemResponse,
    IngestionSourceListResponse,
    IngestionSourcePatchRequest,
    IngestionSourceResponse,
    IngestionSourceSyncTriggerResponse,
    ItemsBulkRequest,
    ItemsBulkResponse,
    MediaIngestBatchCancelResponse,
    MediaIngestJobCancelResponse,
    MediaIngestJobListResponse,
    MediaIngestJobStatus,
    MediaIngestSubmitRequest,
    MediaIngestSubmitResponse,
    MediaAdvancedVersionUpsertRequest,
    MediaMetadataPatchRequest,
    MediaVersionCreateRequest,
    MediaVersionDetail,
    MediaVersionRollbackRequest,
    ReadingExportResponse,
    ReadingHighlight,
    ReadingHighlightCreateRequest,
    ReadingHighlightUpdateRequest,
    ReadingImportJobResponse,
    ReadingImportJobStatus,
    ReadingImportJobsListResponse,
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
)
from .prompt_chatbook_schemas import (
    ChatbookContinueExportRequest,
    ChatbookExportRequest,
    ChatbookImportRequest,
    PromptCreateRequest,
    PromptPreviewRequest,
)
from .translation_schemas import (
    TranslateRequest,
    TranslateResponse,
)
from .ocr_vlm_schemas import (
    OCRBackendsResponse,
    OCRPointsPreloadResponse,
    VLMBackendsResponse,
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
from .rag_admin_schemas import (
    ChunkingTemplateApplyRequest,
    ChunkingTemplateApplyResponse,
    ChunkingTemplateCreateRequest,
    ChunkingTemplateDiagnosticsResponse,
    ChunkingTemplateListResponse,
    ChunkingTemplateResponse,
    ChunkingTemplateUpdateRequest,
    EmbeddingCollectionCreateRequest,
    EmbeddingCollectionListResponse,
    EmbeddingCollectionResponse,
    EmbeddingCollectionStatsResponse,
)
from .evaluations_schemas import (
    CreateEvaluationRequest,
    EvaluationDatasetCreateRequest,
    EvaluationDatasetListResponse,
    EvaluationDatasetResponse,
    EvaluationListResponse,
    EvaluationResponse,
    EvaluationRunCreateRequest,
    EvaluationRunListResponse,
    EvaluationRunResponse,
    UpdateEvaluationRequest,
)
from .flashcards_schemas import (
    FlashcardCreateRequest,
    FlashcardDeckCreateRequest,
    FlashcardDeckResponse,
    FlashcardListResponse,
    FlashcardNextReviewResponse,
    FlashcardResponse,
    FlashcardReviewRequest,
    FlashcardReviewResponse,
    FlashcardReviewSessionEndRequest,
    FlashcardReviewSessionSummary,
    FlashcardUpdateRequest,
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
    ManuscriptChapterCreate,
    ManuscriptChapterResponse,
    ManuscriptChapterUpdate,
    ManuscriptPartCreate,
    ManuscriptPartResponse,
    ManuscriptPartUpdate,
    ManuscriptProjectCreate,
    ManuscriptProjectListResponse,
    ManuscriptProjectResponse,
    ManuscriptProjectUpdate,
    ManuscriptSceneCreate,
    ManuscriptSceneResponse,
    ManuscriptSceneUpdate,
    ManuscriptStructureResponse,
    ReorderRequest,
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
from .chat_conversation_schemas import (
    ConversationScopeParams,
    ConversationUpdateRequest,
    normalize_conversation_state,
)
from .character_persona_schemas import (
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
    WatchlistAlertRuleCreateRequest,
    WatchlistAlertRuleDeleteResponse,
    WatchlistAlertRuleListResponse,
    WatchlistAlertRuleResponse,
    WatchlistAlertRuleUpdateRequest,
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
    ArxivSearchResponse,
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
        **kwargs: Any,
    ) -> Dict[str, Any]:
        request = NoteGraphRequest(**kwargs)
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
    ) -> Dict[str, Any]:
        return await self._request(
            "GET",
            "/api/v1/media/",
            params={
                "page": page,
                "results_per_page": results_per_page,
                "include_keywords": str(include_keywords).lower(),
            },
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

    async def list_ocr_backends(self) -> OCRBackendsResponse:
        response = await self._request("GET", "/api/v1/ocr/backends")
        return OCRBackendsResponse.model_validate(response)

    async def preload_ocr_points_transformers(self) -> OCRPointsPreloadResponse:
        response = await self._request("POST", "/api/v1/ocr/points/preload")
        return OCRPointsPreloadResponse.model_validate(response)

    async def list_vlm_backends(self) -> VLMBackendsResponse:
        response = await self._request("GET", "/api/v1/vlm/backends")
        return VLMBackendsResponse.model_validate(response)

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

    async def submit_media_ingest_jobs(
        self,
        request_data: MediaIngestSubmitRequest,
        file_paths: Optional[List[str]] = None,
    ) -> MediaIngestSubmitResponse:
        httpx_files = prepare_files_for_httpx(file_paths or []) if file_paths else None
        try:
            response = await self._request(
                "POST",
                "/api/v1/media/ingest/jobs",
                data=request_data.model_dump(exclude_none=True, mode="json"),
                files=httpx_files,
            )
            return MediaIngestSubmitResponse.model_validate(response)
        finally:
            if httpx_files:
                cleanup_file_objects(httpx_files)

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
    ) -> AsyncGenerator[Dict[str, Any], None]:
        params = {"batch_id": batch_id, "after_id": after_id}
        async for event in self._sse_request(
            "GET",
            "/api/v1/media/ingest/jobs/events/stream",
            params={key: value for key, value in params.items() if value is not None},
        ):
            yield event

    async def cancel_media_ingest_job(self, job_id: int, *, reason: str | None = None) -> MediaIngestJobCancelResponse:
        response = await self._request(
            "DELETE",
            f"/api/v1/media/ingest/jobs/{job_id}",
            params={key: value for key, value in {"reason": reason}.items() if value is not None},
        )
        return MediaIngestJobCancelResponse.model_validate(response)

    async def cancel_media_ingest_batch(
        self,
        *,
        batch_id: str | None = None,
        session_id: str | None = None,
        reason: str | None = None,
    ) -> MediaIngestBatchCancelResponse:
        response = await self._request(
            "POST",
            "/api/v1/media/ingest/jobs/cancel",
            params={
                key: value
                for key, value in {"batch_id": batch_id, "session_id": session_id, "reason": reason}.items()
                if value is not None
            },
        )
        return MediaIngestBatchCancelResponse.model_validate(response)

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

    async def save_reading_item(self, request_data: ReadingSaveRequest) -> Dict[str, Any]:
        return await self._request(
            "POST",
            "/api/v1/reading/save",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )

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

    async def bulk_update_reading_items(self, request_data: ItemsBulkRequest) -> ItemsBulkResponse:
        response = await self._request(
            "POST",
            "/api/v1/reading/items/bulk",
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
    ) -> ReadingTTSResponse:
        payload = request_data.model_dump(exclude_none=True, mode="json")
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
    ) -> ReadingExportResponse:
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

    async def delete_reading_highlight(self, highlight_id: int) -> Dict[str, Any]:
        return await self._request("DELETE", f"/api/v1/reading/highlights/{highlight_id}")

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
        limit: int = 100,
        offset: int = 0,
    ) -> list[FlashcardDeckResponse]:
        response = await self._request(
            "GET",
            "/api/v1/flashcards/decks",
            params={"limit": limit, "offset": offset},
        )
        return [FlashcardDeckResponse.model_validate(item) for item in list(response or [])]

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
        q: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> FlashcardListResponse:
        response = await self._request(
            "GET",
            "/api/v1/flashcards",
            params={
                key: value
                for key, value in {
                    "deck_id": deck_id,
                    "q": q,
                    "limit": limit,
                    "offset": offset,
                }.items()
                if value is not None
            },
        )
        return FlashcardListResponse.model_validate(response)

    async def get_next_flashcard_review(
        self,
        *,
        deck_id: Optional[int] = None,
    ) -> FlashcardNextReviewResponse:
        response = await self._request(
            "GET",
            "/api/v1/flashcards/review/next",
            params={
                key: value
                for key, value in {
                    "deck_id": deck_id,
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
        *,
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
    ) -> Dict[str, Any]:
        return await self._request(
            "POST",
            "/api/v1/media/search",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
            params={"page": page, "results_per_page": results_per_page},
        )

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

    async def process_plaintext(self, request_data: ProcessPlaintextRequest, file_paths: Optional[List[str]] = None) -> BatchMediaProcessResponse:
        form_data = model_to_form_data(request_data)
        httpx_files = prepare_files_for_httpx(file_paths, upload_field_name="files")
        try:
            response_dict = await self._request("POST", "/api/v1/media/process-plaintext", data=form_data, files=httpx_files)
            return BatchMediaProcessResponse(**response_dict)
        finally:
            cleanup_file_objects(httpx_files)

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
                "POST", "/api/v1/mediawiki/process-dump", data=form_data, files=httpx_files
            ):
                # Assuming each yielded item from the stream is a dict that can be parsed
                # into ProcessedMediaWikiPage or an error/progress event.
                # The client should decide how to handle non-page events (e.g. "summary", "error")
                if item_dict.get("type") == "item_result" and "data" in item_dict:
                    page_data = item_dict["data"]
                    page_data["input_ref"] = Path(dump_file_path).name # Add input_ref for client tracking
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

    async def list_prompts(self, include_deleted: bool = False) -> Dict[str, Any]:
        return await self._request(
            "GET",
            "/api/v1/prompts",
            params={"include_deleted": str(include_deleted).lower()},
        )

    async def preview_prompt(self, request_data: PromptPreviewRequest) -> Dict[str, Any]:
        return await self._request(
            "POST",
            "/api/v1/prompts/preview",
            json_data=request_data.model_dump(exclude_none=True),
        )

    async def create_prompt(self, request_data: PromptCreateRequest) -> Dict[str, Any]:
        return await self._request(
            "POST",
            "/api/v1/prompts",
            json_data=request_data.model_dump(exclude_none=True),
        )

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
            f"/api/v1/prompts/{prompt_identifier}/versions",
        )

    async def restore_prompt_version(self, prompt_identifier: Union[str, int], version: int) -> Dict[str, Any]:
        return await self._request(
            "POST",
            f"/api/v1/prompts/{prompt_identifier}/versions/{version}/restore",
        )

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

    async def get_chatbook_export_job(self, job_id: str) -> Dict[str, Any]:
        return await self._request(
            "GET",
            f"/api/v1/chatbooks/export/jobs/{job_id}",
        )

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

#
# End of client.py
########################################################################################################################
