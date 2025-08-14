"""Pydantic models for media ingestion form data."""

from typing import Optional, List
from pathlib import Path
from pydantic import BaseModel, Field, validator, HttpUrl


class BaseMediaFormData(BaseModel):
    """Base model for all media ingestion forms."""
    title: Optional[str] = Field(None, min_length=1, max_length=200, description="Media title")
    author: Optional[str] = Field(None, min_length=1, max_length=100, description="Media author")
    keywords: Optional[str] = Field(None, description="Comma-separated keywords")
    files: List[Path] = Field(default_factory=list, description="Selected local files")
    urls: List[str] = Field(default_factory=list, description="Media URLs")
    
    @validator('keywords')
    def clean_keywords(cls, v):
        if v:
            # Clean and normalize keywords
            keywords = [k.strip() for k in v.split(',') if k.strip()]
            return ', '.join(keywords)
        return v
    
    @validator('urls', each_item=True)
    def validate_url(cls, v):
        if v and not (v.startswith('http://') or v.startswith('https://')):
            raise ValueError('URL must start with http:// or https://')
        return v


class ProcessingStatus(BaseModel):
    """Status updates during media processing."""
    state: str = Field("idle", pattern="^(idle|processing|complete|error|cancelled)$")
    progress: float = Field(0.0, ge=0.0, le=1.0)
    current_file: Optional[str] = None
    current_operation: Optional[str] = None
    files_processed: int = Field(0, ge=0)
    total_files: int = Field(0, ge=0)
    message: str = ""
    error: Optional[str] = None
    elapsed_time: float = Field(0.0, ge=0.0)
    estimated_time: Optional[float] = None


class VideoFormData(BaseMediaFormData):
    """Form data specific to video ingestion."""
    # Download settings (from video_processing.py)
    download_video_flag: bool = Field(False, description="Download full video or extract audio only")
    use_cookies: bool = Field(False, description="Use cookies for download")
    cookies: Optional[str] = Field(None, description="Cookie string or file path")
    
    # Transcription settings (from transcription_service.py)
    transcription_enabled: bool = Field(True, description="Enable transcription")
    transcription_provider: Optional[str] = Field("faster-whisper", description="Transcription provider")
    transcription_model: Optional[str] = Field("base", description="Model to use")
    transcription_language: Optional[str] = Field("auto", description="Language code or 'auto'")
    vad_use: bool = Field(False, description="Use Voice Activity Detection")
    timestamp_option: bool = Field(True, description="Include timestamps in transcription")
    
    # Video processing settings (from video_processing.py)
    start_time: Optional[str] = Field(None, pattern="^\\d{2}:\\d{2}:\\d{2}$|^$", description="Start time (HH:MM:SS)")
    end_time: Optional[str] = Field(None, pattern="^\\d{2}:\\d{2}:\\d{2}$|^$", description="End time (HH:MM:SS)")
    
    # Analysis settings (from audio_processing.py)
    perform_analysis: bool = Field(True, description="Perform AI analysis/summarization")
    api_name: Optional[str] = Field(None, description="API for analysis")
    api_key: Optional[str] = Field(None, description="API key for analysis")
    custom_prompt: Optional[str] = Field(None, description="Custom prompt for analysis")
    summarize_recursively: bool = Field(False, description="Recursive summarization for long content")
    
    # Chunking settings (from chunking_service.py)
    perform_chunking: bool = Field(True, description="Enable chunking")
    chunk_method: Optional[str] = Field("time", description="Chunking method")
    max_chunk_size: int = Field(500, ge=100, le=10000, description="Max chunk size")
    chunk_overlap: int = Field(200, ge=0, le=500, description="Overlap between chunks")
    use_adaptive_chunking: bool = Field(False, description="Use adaptive chunking")
    use_multi_level_chunking: bool = Field(False, description="Use multi-level chunking")
    
    @validator('transcription_provider')
    def validate_provider(cls, v):
        valid_providers = ['faster-whisper', 'whisper', 'lightning-whisper-mlx', 
                          'parakeet-mlx', 'qwen2audio', 'nemo']
        if v and v not in valid_providers:
            raise ValueError(f'Invalid provider. Must be one of: {", ".join(valid_providers)}')
        return v


class AudioFormData(BaseMediaFormData):
    """Form data specific to audio ingestion."""
    # Download settings (from audio_processing.py)
    use_cookies: bool = Field(False, description="Use cookies for download")
    cookies: Optional[str] = Field(None, description="Cookie string or dict")
    
    # Transcription settings (from transcription_service.py and audio_processing.py)
    transcription_enabled: bool = Field(True, description="Enable transcription")
    transcription_provider: Optional[str] = Field("faster-whisper", description="Transcription provider")
    transcription_model: Optional[str] = Field("base", description="Model to use")
    transcription_language: Optional[str] = Field("en", description="Language code or 'auto'")
    translation_target_language: Optional[str] = Field(None, description="Translation target language")
    vad_use: bool = Field(False, description="Use Voice Activity Detection")
    timestamp_option: bool = Field(True, description="Include timestamps")
    
    # Audio time range (from audio_processing.py)
    start_time: Optional[str] = Field(None, pattern="^\\d{2}:\\d{2}:\\d{2}$|^$", description="Start time (HH:MM:SS)")
    end_time: Optional[str] = Field(None, pattern="^\\d{2}:\\d{2}:\\d{2}$|^$", description="End time (HH:MM:SS)")
    
    # Speaker diarization (from audio_processing.py)
    diarize: bool = Field(False, description="Enable speaker diarization")
    
    # Analysis settings (from audio_processing.py)
    perform_analysis: bool = Field(True, description="Perform AI analysis/summarization")
    api_name: Optional[str] = Field(None, description="API for analysis")
    api_key: Optional[str] = Field(None, description="API key for analysis")
    custom_prompt: Optional[str] = Field(None, description="Custom prompt for analysis")
    system_prompt: Optional[str] = Field(None, description="System prompt for analysis")
    summarize_recursively: bool = Field(False, description="Recursive summarization for long content")
    
    # Chunking settings (from audio_processing.py)
    perform_chunking: bool = Field(True, description="Enable chunking")
    chunk_method: Optional[str] = Field(None, description="Chunking method")
    max_chunk_size: int = Field(500, ge=100, le=10000, description="Max chunk size")
    chunk_overlap: int = Field(200, ge=0, le=500, description="Overlap between chunks")
    use_adaptive_chunking: bool = Field(False, description="Use adaptive chunking")
    use_multi_level_chunking: bool = Field(False, description="Use multi-level chunking")
    chunk_language: Optional[str] = Field(None, description="Language for chunking")


class PDFFormData(BaseMediaFormData):
    """Form data for PDF ingestion (from PDF_Processing_Lib.py)."""
    # PDF parser selection
    parser: str = Field("pymupdf4llm", description="PDF parser to use")
    
    # OCR settings (from OCR_Backends.py)
    ocr_enabled: bool = Field(False, description="Enable OCR for scanned PDFs")
    ocr_backend: str = Field("tesseract", description="OCR backend (tesseract/doctr/surya/marker)")
    ocr_language: str = Field("eng", description="OCR language code")
    
    # Analysis settings
    perform_analysis: bool = Field(False, description="Perform AI analysis/summarization")
    api_name: Optional[str] = Field(None, description="API for analysis")
    api_key: Optional[str] = Field(None, description="API key for analysis")
    custom_prompt: Optional[str] = Field(None, description="Custom prompt for analysis")
    system_prompt: Optional[str] = Field(None, description="System prompt for analysis")
    
    # Chunking settings
    perform_chunking: bool = Field(True, description="Enable chunking")
    chunk_method: str = Field("semantic", description="Chunking method")
    max_chunk_size: int = Field(1000, ge=100, le=10000, description="Max chunk size")
    chunk_overlap: int = Field(200, ge=0, le=500, description="Overlap between chunks")
    
    # Page range
    page_start: Optional[int] = Field(None, ge=1, description="Start page")
    page_end: Optional[int] = Field(None, ge=1, description="End page")
    
    @validator('parser')
    def validate_parser(cls, v):
        valid_parsers = ['pymupdf4llm', 'pymupdf', 'pdfplumber', 'docling', 'marker', 'unstructured']
        if v not in valid_parsers:
            raise ValueError(f'Invalid parser. Must be one of: {", ".join(valid_parsers)}')
        return v
    
    @validator('ocr_backend')
    def validate_ocr_backend(cls, v):
        valid_backends = ['tesseract', 'doctr', 'surya', 'marker', 'paddle', 'textract', 'gptocr']
        if v not in valid_backends:
            raise ValueError(f'Invalid OCR backend. Must be one of: {", ".join(valid_backends)}')
        return v


class DocumentFormData(BaseMediaFormData):
    """Form data for document ingestion (DOCX, ODT, RTF, etc from Document_Processing_Lib.py)."""
    # Processing options
    use_docling: bool = Field(False, description="Use Docling for advanced processing")
    
    # Analysis settings  
    auto_summarize: bool = Field(False, description="Auto-summarize document")
    api_name: Optional[str] = Field(None, description="API for analysis")
    api_key: Optional[str] = Field(None, description="API key for analysis")
    custom_prompt: Optional[str] = Field(None, description="Custom prompt for analysis")
    system_prompt: Optional[str] = Field(None, description="System prompt for analysis")
    
    # Chunking settings
    perform_chunking: bool = Field(True, description="Enable chunking")
    chunk_method: str = Field("semantic", description="Chunking method")
    max_chunk_size: int = Field(1000, ge=100, le=10000, description="Max chunk size")
    chunk_overlap: int = Field(200, ge=0, le=500, description="Overlap between chunks")
    
    # Output options
    preserve_tables: bool = Field(True, description="Preserve table structure")
    extract_metadata: bool = Field(True, description="Extract document metadata")


class EbookFormData(BaseMediaFormData):
    """Form data for ebook ingestion (EPUB, MOBI, FB2, etc from Book_Ingestion_Lib.py)."""
    # Extraction settings
    extraction_method: str = Field('filtered', description="Extraction method (filtered/full)")
    
    # Analysis settings
    perform_analysis: bool = Field(False, description="Perform AI analysis/summarization")
    api_name: Optional[str] = Field(None, description="API for analysis")
    api_key: Optional[str] = Field(None, description="API key for analysis")
    custom_prompt: Optional[str] = Field(None, description="Custom prompt for analysis")
    system_prompt: Optional[str] = Field(None, description="System prompt for analysis")
    summarize_recursively: bool = Field(False, description="Recursive summarization for long content")
    
    # Chunking settings
    perform_chunking: bool = Field(True, description="Enable chunking")
    chunk_method: str = Field("chapter", description="Chunking method (chapter/page/semantic)")
    max_chunk_size: int = Field(2000, ge=100, le=10000, description="Max chunk size")
    chunk_overlap: int = Field(200, ge=0, le=500, description="Overlap between chunks")
    
    # Format-specific options
    process_images: bool = Field(False, description="Process embedded images")
    extract_toc: bool = Field(True, description="Extract table of contents")
    preserve_formatting: bool = Field(False, description="Preserve text formatting")
    
    @validator('extraction_method')
    def validate_extraction_method(cls, v):
        if v not in ['filtered', 'full', 'basic']:
            raise ValueError('Extraction method must be: filtered, full, or basic')
        return v


class WebFormData(BaseMediaFormData):
    """Form data for web article ingestion (using TLDW API endpoints)."""
    # URL processing
    url_type: str = Field("article", description="Type of URL (article/youtube/generic)")
    
    # Content extraction options (from TLDW API)
    extract_text: bool = Field(True, description="Extract main text content")
    extract_metadata: bool = Field(True, description="Extract page metadata")
    use_readability: bool = Field(True, description="Use readability for clean extraction")
    
    # Fallback options
    use_archive_fallback: bool = Field(True, description="Use archive.org if URL fails")
    
    # Analysis settings
    perform_analysis: bool = Field(False, description="Perform AI analysis/summarization")
    api_name: Optional[str] = Field(None, description="API for analysis")
    api_key: Optional[str] = Field(None, description="API key for analysis")
    custom_prompt: Optional[str] = Field(None, description="Custom prompt for analysis")
    
    # Chunking settings
    perform_chunking: bool = Field(True, description="Enable chunking")
    chunk_method: str = Field("semantic", description="Chunking method")
    max_chunk_size: int = Field(1000, ge=100, le=10000, description="Max chunk size")