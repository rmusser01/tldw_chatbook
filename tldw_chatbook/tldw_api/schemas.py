# tldw_chatbook/tldw_api/schemas.py
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, HttpUrl

# Enum-like Literals from API schema
MediaType = Literal['video', 'audio', 'document', 'pdf', 'ebook', 'xml', 'mediawiki_dump', 'plaintext']
ChunkMethod = Literal['semantic', 'tokens', 'paragraphs', 'sentences', 'words', 'ebook_chapters', 'json']
PdfEngine = Literal['pymupdf4llm', 'pymupdf', 'docling']
ScrapeMethod = Literal["individual", "sitemap", "url_level", "recursive_scraping"]


# --- Base Request Options (mirrors parts of AddMediaForm) ---
class BaseMediaRequest(BaseModel):
    urls: Optional[List[HttpUrl]] = None # FastAPI converts to list of strings, Pydantic can take HttpUrl
    title: Optional[str] = None
    author: Optional[str] = None
    keywords: Optional[List[str]] = None # Client sends list, will be joined to string by client method
    custom_prompt: Optional[str] = None
    system_prompt: Optional[str] = None
    overwrite_existing: bool = False # Though not used by "process-only" endpoints
    perform_analysis: bool = True
    api_name: Optional[str] = None
    api_key: Optional[str] = None # Client should handle securely
    use_cookies: bool = False
    cookies: Optional[str] = None
    summarize_recursively: bool = False
    perform_rolling_summarization: bool = False # from AddMediaForm

    # ChunkingOptions
    perform_chunking: bool = True
    chunk_method: Optional[ChunkMethod] = None
    use_adaptive_chunking: bool = False
    use_multi_level_chunking: bool = False
    chunk_language: Optional[str] = None
    chunk_size: int = 500
    chunk_overlap: int = 200
    custom_chapter_pattern: Optional[str] = None

    # AudioVideoOptions (subset relevant to multiple types or for base commonality)
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    transcription_model: Optional[str] = "deepdml/faster-whisper-large-v3-turbo-ct2" # default from video
    transcription_language: Optional[str] = "en"
    diarize: Optional[bool] = False
    timestamp_option: Optional[bool] = True
    vad_use: Optional[bool] = False
    perform_confabulation_check_of_analysis: Optional[bool] = False

    # PdfOptions
    pdf_parsing_engine: Optional[PdfEngine] = "pymupdf4llm"


# --- Specific Media Type Request Models ---
class ProcessVideoRequest(BaseMediaRequest):
    # Video specific defaults or overrides can go here if any
    # transcription_model is already in BaseMediaRequest with a video-like default
    pass

class ProcessAudioRequest(BaseMediaRequest):
    # Override transcription model default if different for audio
    transcription_model: Optional[str] = "deepdml/faster-distil-whisper-large-v3.5"
    pass

class ProcessPDFRequest(BaseMediaRequest):
    # pdf_parsing_engine is already in BaseMediaRequest
    pass

class ProcessEbookRequest(BaseMediaRequest):
    extraction_method: Literal['filtered', 'markdown', 'basic'] = 'filtered'
    chunk_method: Optional[ChunkMethod] = 'ebook_chapters' # Default for ebooks

class ProcessDocumentRequest(BaseMediaRequest):
    chunk_method: Optional[ChunkMethod] = 'sentences' # Default for documents
    chunk_size: int = 1000

class ProcessXMLRequest(BaseModel): # Based on XMLIngestRequest
    title: Optional[str] = None
    author: Optional[str] = None
    keywords: Optional[List[str]] = None
    system_prompt: Optional[str] = None
    custom_prompt: Optional[str] = None
    auto_summarize: bool = False
    api_name: Optional[str] = None
    api_key: Optional[str] = None
    # mode: str = "ephemeral" # For this client, we assume "process-only" which is ephemeral

class ProcessMediaWikiRequest(BaseModel): # Based on MediaWikiDumpOptionsForm
    wiki_name: str
    namespaces_str: Optional[str] = None # Comma-separated string of namespace IDs
    skip_redirects: bool = True
    chunk_max_size: int = 1000
    api_name_vector_db: Optional[str] = None # For potential embedding if API supports it without DB write
    api_key_vector_db: Optional[str] = None

class ProcessPlaintextRequest(BaseMediaRequest):
    # Plaintext specific options
    encoding: Literal['utf-8', 'ascii', 'latin-1', 'auto'] = 'utf-8'
    line_ending: Literal['auto', 'lf', 'crlf'] = 'auto'
    remove_extra_whitespace: bool = True
    convert_to_paragraphs: bool = False
    split_pattern: Optional[str] = None  # Regex pattern for custom splitting
    chunk_method: Optional[ChunkMethod] = 'paragraphs'  # Default for plaintext


# --- Response Models (from API specification) ---
class MediaItemProcessResult(BaseModel):
    status: Literal['Success', 'Error', 'Warning', 'Skipped'] # Added Skipped
    input_ref: str
    processing_source: Optional[str] = None
    media_type: str
    metadata: Optional[Dict[str, Any]] = None
    content: Optional[str] = None
    transcript: Optional[str] = None
    segments: Optional[List[Dict[str, Any]]] = None
    chunks: Optional[List[Any]] = None
    analysis: Optional[str] = None
    summary: Optional[str] = None
    analysis_details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    warnings: Optional[List[str]] = None
    db_id: Optional[int] = None # Will be None for these "process-only" endpoints
    db_message: Optional[str] = None # Will be specific message for these endpoints
    message: Optional[str] = None

class BatchMediaProcessResponse(BaseModel):
    processed_count: int
    errors_count: int
    errors: List[str]
    results: List[MediaItemProcessResult]
    confabulation_results: Optional[Any] = None

# Simplified XML response if it's different
class ProcessXMLResponseItem(BaseModel): # Based on your description of process_xml_task
    status: str
    input_ref: str # derived from filename
    title: Optional[str]
    author: Optional[str]
    content: Optional[str]
    summary: Optional[str] # if auto_summarize was true
    keywords: Optional[List[str]]
    segments: Optional[List[Dict[str, Any]]] # if XML structure leads to segments
    error: Optional[str] = None

class BatchProcessXMLResponse(BaseModel):
    processed_count: int
    errors_count: int
    errors: List[str]
    results: List[ProcessXMLResponseItem]


# MediaWiki streaming typically yields individual items or progress updates.
# The client will handle NDJSON stream. This schema is for a single processed page item.
class ProcessedMediaWikiPage(BaseModel): # From API spec
    title: str
    content: str
    namespace: Optional[int] = None
    page_id: Optional[int] = None
    revision_id: Optional[int] = None
    timestamp: Optional[str] = None
    chunks: List[Dict[str, Any]] = []
    media_id: Optional[int] = None
    message: Optional[str] = None
    status: str = "Pending"
    error_message: Optional[str] = None
    # Add input_ref for client-side tracking if server doesn't include it
    input_ref: Optional[str] = None # To be populated by client if needed

# For the client, we might not need a BatchMediaWikiResponse if handling stream directly.
# But if we were to collect all results, it might look like:
class BatchMediaWikiProcessResponse(BaseModel):
    processed_count: int
    errors_count: int
    errors: List[Dict[str, Any]] # Store full error events
    results: List[ProcessedMediaWikiPage] # List of successfully processed pages
    summary: Optional[Dict[str, Any]] = None # Final summary event from stream