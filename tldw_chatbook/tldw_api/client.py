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
    FileCreateRequest,
    IngestionSourceCreateRequest,
    IngestionSourcePatchRequest,
    ReadingProgressUpdate,
    ReadingUpdateRequest,
)
from .prompt_chatbook_schemas import (
    ChatbookExportRequest,
    ChatbookImportRequest,
    PromptCreateRequest,
    PromptPreviewRequest,
)
from .chat_conversation_schemas import (
    ConversationScopeParams,
    ConversationUpdateRequest,
    normalize_conversation_state,
)
from .character_persona_schemas import (
    CharacterCreateRequest,
    CharacterExemplarCreate,
    CharacterExemplarSearchRequest,
    CharacterExemplarSelectionDebugRequest,
    CharacterExemplarUpdate,
    CharacterQueryRequest,
    CharacterResponse,
    CharacterUpdateRequest,
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
                raise AuthenticationError(f"Authentication failed: {error_detail}")
            elif e.response.status_code == 422: # Unprocessable Entity (Pydantic validation error)
                raise APIRequestError(f"Validation Error: {error_detail}", response_data=response_data)
            raise APIResponseError(e.response.status_code, error_detail, response_data=response_data)
        except httpx.RequestError as e: # Covers ConnectError, TimeoutException, etc.
            raise APIConnectionError(f"Connection error to {url}: {e}")
        except json.JSONDecodeError:
            raise APIResponseError(response.status_code, "Failed to decode JSON response", response_data={"raw_text": response.text})


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
                pass
            if e.response.status_code == 401:
                raise AuthenticationError(f"Authentication failed: {error_detail}")
            raise APIResponseError(e.response.status_code, error_detail, response_data={"raw_text": response_text})
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

    async def delete_file_artifact(self, file_id: int, hard: bool = False, delete_file: bool = False) -> Dict[str, Any]:
        return await self._request(
            "DELETE",
            f"/api/v1/files/{file_id}",
            params={"hard": str(hard).lower(), "delete_file": str(delete_file).lower()},
        )

    async def create_ingestion_source(self, request_data: IngestionSourceCreateRequest) -> Dict[str, Any]:
        return await self._request(
            "POST",
            "/api/v1/ingestion-sources/",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )

    async def list_ingestion_sources(self) -> Dict[str, Any]:
        return await self._request("GET", "/api/v1/ingestion-sources/")

    async def get_ingestion_source(self, source_id: int) -> Dict[str, Any]:
        return await self._request("GET", f"/api/v1/ingestion-sources/{source_id}")

    async def patch_ingestion_source(
        self,
        source_id: int,
        request_data: IngestionSourcePatchRequest,
    ) -> Dict[str, Any]:
        return await self._request(
            "PATCH",
            f"/api/v1/ingestion-sources/{source_id}",
            json_data=request_data.model_dump(exclude_none=True, mode="json"),
        )

    async def list_ingestion_source_items(self, source_id: int) -> Dict[str, Any]:
        return await self._request("GET", f"/api/v1/ingestion-sources/{source_id}/items")

    async def trigger_ingestion_source_sync(self, source_id: int) -> Dict[str, Any]:
        return await self._request("POST", f"/api/v1/ingestion-sources/{source_id}/sync")

    async def upload_ingestion_source_archive(self, source_id: int, archive_path: str) -> Dict[str, Any]:
        httpx_files = prepare_files_for_httpx([archive_path], upload_field_name="archive")
        try:
            return await self._request(
                "POST",
                f"/api/v1/ingestion-sources/{source_id}/archive",
                files=httpx_files,
            )
        finally:
            cleanup_file_objects(httpx_files)

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
            "page": page,
            "size": size,
            "offset": offset,
            "limit": limit,
            "sort": sort,
        }
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

    async def get_chatbook_import_job(self, job_id: str) -> Dict[str, Any]:
        return await self._request(
            "GET",
            f"/api/v1/chatbooks/import/jobs/{job_id}",
        )

#
# End of client.py
########################################################################################################################
