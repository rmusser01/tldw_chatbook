# tldw_chatbook/tldw_api/__init__.py
from .client import TLDWAPIClient
from .exceptions import (
    TLDWAPIError, APIConnectionError, APIRequestError,
    APIResponseError, AuthenticationError
)
from .schemas import (
    ProcessVideoRequest, ProcessAudioRequest, ProcessPDFRequest,
    ProcessEbookRequest, ProcessDocumentRequest, ProcessXMLRequest, ProcessMediaWikiRequest,
    ProcessPlaintextRequest,
    MediaItemProcessResult, BatchMediaProcessResponse,
    BatchProcessXMLResponse, ProcessedMediaWikiPage,
    MediaType, ChunkMethod, PdfEngine, ScrapeMethod # Export Enums/Literals
)
from .notes_workspace_schemas import (
    MediaListItem,
    MediaListPagination,
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
from .prompt_chatbook_schemas import (
    PromptCreateRequest,
    PromptPreviewRequest,
    PromptResponse,
    PromptVersionResponse,
    ChatbookExportRequest,
    ChatbookImportRequest,
    ChatbookPreviewResponse,
    ChatbookExportJobResponse,
    ChatbookImportJobResponse,
)
from .chat_conversation_schemas import (
    ALLOWED_CONVERSATION_STATES,
    ConversationListItem,
    ConversationListPagination,
    ConversationListResponse,
    ConversationMetadata,
    ConversationScopeParams,
    ConversationState,
    ConversationTreeNode,
    ConversationTreePagination,
    ConversationTreeResponse,
    ConversationUpdateRequest,
    normalize_conversation_state,
)

__all__ = [
    "TLDWAPIClient",
    "TLDWAPIError", "APIConnectionError", "APIRequestError",
    "APIResponseError", "AuthenticationError",
    "ProcessVideoRequest", "ProcessAudioRequest", "ProcessPDFRequest",
    "ProcessEbookRequest", "ProcessDocumentRequest", "ProcessXMLRequest", "ProcessMediaWikiRequest",
    "ProcessPlaintextRequest",
    "MediaItemProcessResult", "BatchMediaProcessResponse",
    "BatchProcessXMLResponse", "ProcessedMediaWikiPage",
    "MediaType", "ChunkMethod", "PdfEngine", "ScrapeMethod",
    "MediaListItem", "MediaListPagination", "MediaListResponse", "MediaSearchRequest",
    "NoteCreateRequest", "NoteListResponse", "NoteResponse", "NoteUpdateRequest",
    "WorkspaceArtifactCreateRequest", "WorkspaceArtifactResponse", "WorkspaceArtifactUpdateRequest",
    "WorkspaceCreateRequest", "WorkspaceListResponse",
    "WorkspaceNoteCreateRequest", "WorkspaceNoteResponse", "WorkspaceNoteUpdateRequest",
    "WorkspaceResponse", "WorkspaceSourceCreateRequest", "WorkspaceSourceResponse",
    "WorkspaceSourceUpdateRequest", "WorkspaceUpdateRequest",
    "PromptCreateRequest", "PromptPreviewRequest", "PromptResponse", "PromptVersionResponse",
    "ChatbookExportRequest", "ChatbookImportRequest", "ChatbookPreviewResponse",
    "ChatbookExportJobResponse", "ChatbookImportJobResponse",
    "ALLOWED_CONVERSATION_STATES",
    "ConversationState", "normalize_conversation_state",
    "ConversationScopeParams", "ConversationListItem", "ConversationListPagination",
    "ConversationListResponse", "ConversationUpdateRequest", "ConversationMetadata",
    "ConversationTreeNode", "ConversationTreePagination", "ConversationTreeResponse",
]
