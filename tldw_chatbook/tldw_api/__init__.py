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

__all__ = [
    "TLDWAPIClient",
    "TLDWAPIError", "APIConnectionError", "APIRequestError",
    "APIResponseError", "AuthenticationError",
    "ProcessVideoRequest", "ProcessAudioRequest", "ProcessPDFRequest",
    "ProcessEbookRequest", "ProcessDocumentRequest", "ProcessXMLRequest", "ProcessMediaWikiRequest",
    "ProcessPlaintextRequest",
    "MediaItemProcessResult", "BatchMediaProcessResponse",
    "BatchProcessXMLResponse", "ProcessedMediaWikiPage",
    "MediaType", "ChunkMethod", "PdfEngine", "ScrapeMethod"
]
