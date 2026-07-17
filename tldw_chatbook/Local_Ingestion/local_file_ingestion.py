# tldw_chatbook/Local_Ingestion/local_file_ingestion.py
"""
Programmatic interface for ingesting local files into the Media database.

This module provides functions to process and store various file types (PDFs, documents, 
e-books, etc.) without going through the UI, leveraging existing processing capabilities.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from loguru import logger

# Per-format processing libraries (process_pdf/process_document/process_ebook/
# LocalAudioProcessor/LocalVideoProcessor) are intentionally NOT imported at
# module scope here. This module is imported directly by app.py (for
# classify_ingest_source/persist_parsed_media), which bypasses
# Local_Ingestion's own lazy `__init__.py` (PEP 562 `__getattr__`) --
# standard Python import semantics run this file's module body regardless of
# which name was requested. Before this deferral, that meant pymupdf/
# onnxruntime (~170ms, via PDF_Processing_Lib) and the Document/ebook
# processing stack (~59ms) were paid on every app startup even when no PDF/
# document/ebook ingestion had happened (see task-257).
#
# Each processor is now imported lazily via the module-level placeholders +
# `_ensure_*()` helpers below (mirroring the pattern already used by
# `Embeddings/Chroma_Lib.py` and `Web_Scraping/Article_Extractor_Lib.py`),
# called at the top of `parse_local_file_for_ingest()`'s branch for that
# `file_type`, once per process (Python caches `sys.modules`, so repeated
# calls don't re-pay the cost). The module-level names are kept (rather than
# using plain local `from .X import Y` statements inside each branch) so
# that `unittest.mock.patch("...local_file_ingestion.process_pdf", ...)` /
# `monkeypatch.setattr("...local_file_ingestion.LocalVideoProcessor", ...)`
# -- used by existing tests -- continue to work: each `_ensure_*()` helper
# no-ops when its name is already bound to something other than the
# placeholder (e.g. a test's mock), so a patched value is never clobbered.
process_pdf = None
process_document = None
process_ebook = None
LocalAudioProcessor = None
LocalVideoProcessor = None


def _ensure_process_pdf():
    """Import PDF_Processing_Lib.process_pdf on first actual use (or return
    an already-bound value, e.g. a test's mock)."""
    global process_pdf
    if process_pdf is None:
        from .PDF_Processing_Lib import process_pdf as _process_pdf
        process_pdf = _process_pdf
    return process_pdf


def _ensure_process_document():
    """Import Document_Processing_Lib.process_document on first actual use
    (or return an already-bound value, e.g. a test's mock)."""
    global process_document
    if process_document is None:
        from .Document_Processing_Lib import process_document as _process_document
        process_document = _process_document
    return process_document


def _ensure_process_ebook():
    """Import Book_Ingestion_Lib.process_ebook on first actual use (or
    return an already-bound value, e.g. a test's mock)."""
    global process_ebook
    if process_ebook is None:
        from .Book_Ingestion_Lib import process_ebook as _process_ebook
        process_ebook = _process_ebook
    return process_ebook


def _ensure_local_audio_processor():
    """Import audio_processing.LocalAudioProcessor on first actual use (or
    return an already-bound value, e.g. a test's mock)."""
    global LocalAudioProcessor
    if LocalAudioProcessor is None:
        from .audio_processing import LocalAudioProcessor as _LocalAudioProcessor
        LocalAudioProcessor = _LocalAudioProcessor
    return LocalAudioProcessor


def _ensure_local_video_processor():
    """Import video_processing.LocalVideoProcessor on first actual use (or
    return an already-bound value, e.g. a test's mock)."""
    global LocalVideoProcessor
    if LocalVideoProcessor is None:
        from .video_processing import LocalVideoProcessor as _LocalVideoProcessor
        LocalVideoProcessor = _LocalVideoProcessor
    return LocalVideoProcessor


# Import database
from ..DB.Client_Media_DB_v2 import MediaDatabase

# Import metrics
from ..Metrics.metrics_logger import log_counter, log_histogram


class FileIngestionError(Exception):
    """Raised when file ingestion fails."""
    pass


class PermanentIngestError(FileIngestionError):
    """A parse/fetch failure that will fail identically on retry (bad URL,
    4xx, non-HTML content, empty extraction, missing extractor dependency).
    ``classify_parse_failure`` maps this to a permanent (non-retryable) job.
    """


_VIDEO_URL_HOSTS = ("youtube.com", "youtu.be", "vimeo.com", "dailymotion.com")
_VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".wmv", ".m4v", ".mpg", ".mpeg")
_AUDIO_EXTS = (".mp3", ".m4a", ".wav", ".flac", ".ogg", ".aac", ".wma", ".opus")
_TRACKING_PARAMS = frozenset({
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "fbclid", "igshid", "mc_cid", "mc_eid", "ref", "ref_src",
})


def _is_http_url(source: str) -> bool:
    from urllib.parse import urlparse
    try:
        return urlparse(source).scheme in ("http", "https")
    except Exception:
        return False


def classify_ingest_source(source: str) -> str:
    """Classify an ingest source into a media type.

    For an http/https URL: a known video host or a video-extension path ->
    ``"video"``; an audio-extension path -> ``"audio"``; otherwise
    ``"article"``. For any non-URL source, delegate to ``detect_file_type``.

    Args:
        source: A local file path or an http/https URL to classify.

    Returns:
        str: The media type -- ``"video"``, ``"audio"``, or ``"article"`` for
        a URL; for a file path, whatever ``detect_file_type`` returns
        (``"pdf"``, ``"document"``, ``"audio"``, ...).

    Raises:
        FileIngestionError: If ``source`` is a non-URL path whose extension is
            not a recognized ingestible type (propagated from
            ``detect_file_type``).
    """
    from urllib.parse import urlparse
    source = str(source)
    if _is_http_url(source):
        parsed = urlparse(source)
        host = (parsed.hostname or "").lower()
        path = parsed.path.lower()
        if any(host == h or host.endswith("." + h) for h in _VIDEO_URL_HOSTS) or path.endswith(_VIDEO_EXTS):
            return "video"
        if path.endswith(_AUDIO_EXTS):
            return "audio"
        return "article"
    return detect_file_type(source)


def canonicalize_url(url: str) -> str:
    """Canonicalize a URL to a stable, clean stored value.

    Lowercases the scheme and host, drops a default port (80/443) and the URL
    fragment, strips a trailing slash (except at the root), removes common
    tracking parameters (``utm_*``, ``gclid``, ``fbclid``, ...), and sorts the
    remaining query parameters so the same logical URL always canonicalizes to
    an identical string.

    Args:
        url: The URL to canonicalize -- typically the post-redirect
            ``resp.url`` from a successful fetch.

    Returns:
        str: The canonicalized URL.

    Raises:
        PermanentIngestError: If the URL carries a non-integer port (a
            malformed URL that fails identically on every retry).
    """
    from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
    parsed = urlparse(url)
    scheme = (parsed.scheme or "https").lower()
    host = (parsed.hostname or "").lower()
    netloc = host
    try:
        port = parsed.port
    except ValueError as exc:
        # urllib raises ValueError for a non-integer port (e.g. ":foo").
        raise PermanentIngestError(f"Invalid URL port: {url}") from exc
    if port and not ((scheme == "https" and port == 443) or (scheme == "http" and port == 80)):
        netloc = f"{host}:{port}"
    path = parsed.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    query = urlencode(sorted(
        (k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
        if k.lower() not in _TRACKING_PARAMS
    ))
    return urlunparse((scheme, netloc, path, "", query, ""))


def detect_file_type(file_path: Union[str, Path]) -> str:
    """
    Detect the type of file based on its extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File type as string: 'pdf', 'document', 'ebook', 'plaintext', 'html'
        
    Raises:
        FileIngestionError: If file type is not supported
    """
    file_path = Path(file_path)
    extension = file_path.suffix.lower()
    
    # PDF files
    if extension == '.pdf':
        return 'pdf'
    
    # Document files
    elif extension in ['.doc', '.docx', '.odt', '.rtf']:
        return 'document'
    
    # E-book files
    elif extension in ['.epub', '.mobi', '.azw', '.azw3', '.fb2']:
        return 'ebook'
    
    # HTML files (web articles)
    elif extension in ['.html', '.htm']:
        return 'html'
    
    # Plain text files
    elif extension in ['.txt', '.md', '.markdown', '.rst', '.log', '.csv']:
        return 'plaintext'
    
    # Audio files
    elif extension in ['.mp3', '.m4a', '.wav', '.flac', '.ogg', '.aac', '.wma', '.opus']:
        return 'audio'
    
    # Video files
    elif extension in ['.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv', '.wmv', '.m4v', '.mpg', '.mpeg']:
        return 'video'
    
    else:
        raise FileIngestionError(
            f"Unsupported file type: {extension}. "
            f"Supported types: PDF, DOCX, ODT, RTF, EPUB, MOBI, AZW, FB2, HTML, TXT, MD, "
            f"MP3, M4A, WAV, FLAC, OGG, AAC, MP4, AVI, MKV, MOV, WEBM"
        )


def get_supported_extensions() -> Dict[str, List[str]]:
    """
    Get all supported file extensions organized by media type.
    
    Returns:
        Dictionary mapping media types to their supported extensions
    """
    return {
        'pdf': ['.pdf'],
        'document': ['.doc', '.docx', '.odt', '.rtf'],
        'ebook': ['.epub', '.mobi', '.azw', '.azw3', '.fb2'],
        'html': ['.html', '.htm'],
        'plaintext': ['.txt', '.md', '.markdown', '.rst', '.log', '.csv'],
        'audio': ['.mp3', '.m4a', '.wav', '.flac', '.ogg', '.aac', '.wma', '.opus'],
        'video': ['.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv', '.wmv', '.m4v', '.mpg', '.mpeg']
    }


def parse_local_file_for_ingest(file_path: Union[str, Path], options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse a local file into a picklable payload, performing no database I/O.

    This is the pre-DB half of ``ingest_local_file`` (F3 parallel-parse
    split): file-type detection, per-type extraction (PDF/document/ebook/
    audio/video/HTML/plaintext), and result normalization. It never touches
    ``media_db`` -- audio/video extraction is routed through
    ``LocalAudioProcessor``/``LocalVideoProcessor`` with ``media_db=None``
    specifically so this holds for those types too, which otherwise support
    an internal "write while parsing" path. That also FIXES a live
    pre-split bug in the old single-function ``ingest_local_file`` (the
    exact mechanism verified empirically against the real DB methods): for
    audio/video, the processor's internal ``_store_in_database`` step wrote
    ONE degraded media row first -- bare-path URL (its ``input_ref``, not
    ``file://``-prefixed), processor-metadata title, no keywords. The
    pipeline's own ``add_media_with_keywords`` call (``url="file://..."``)
    then hit the CONTENT-HASH dedup fallback, matched that degraded row,
    took the "already exists, overwrite not enabled" branch, and returned
    ``(None, None, ...)`` -- so every real audio/video local ingest
    returned ``media_id=None``, and even ``app.py``'s
    ``get_media_by_url("file://...")`` recovery missed, because the
    surviving row's URL was the bare path. With the processors never
    handed a DB, the degraded row is never written, the pipeline's write
    is the first (and only) insert, and a real ``media_id`` comes back.
    Regression-locked by
    ``Tests/Local_Ingestion/test_ingest_parse_worker.py``'s
    ``test_ingest_local_file_audio_returns_real_media_id``.

    This makes the function safe to run inside a spawned worker process
    (see ``ingest_parse_worker.run_parse_job``): its return value is plain,
    picklable data, and pairing it with ``persist_parsed_media`` is the
    *only* place the resulting content is written to the database.

    Args:
        file_path: Path to the file to parse.
        options: Dict of ingestion options -- see ``ingest_parse_worker``'s
            module docstring for the full schema. Recognized keys (all
            optional): ``title``, ``author``, ``keywords``,
            ``custom_prompt``, ``system_prompt``, ``perform_analysis``,
            ``api_name``, ``api_key``, ``chunk_options``, ``metadata`` --
            mirroring ``ingest_local_file``'s keyword arguments of the
            same names.

    Returns:
        A payload dict consumed by ``persist_parsed_media``:
            - media_type: Media type string used for the DB write (e.g.
              'pdf', 'document', 'ebook', 'plaintext', 'html', 'audio',
              'video').
            - file_type: Same value as ``media_type`` -- kept as a
              separate key for parity with ``ingest_local_file``'s
              historical return dict.
            - title, author: Extracted (or override) title/author.
            - content: Extracted text content.
            - keywords: Combined list of caller-supplied and
              extracted keywords.
            - url: The ``file://`` URL passed to ``add_media_with_keywords``.
            - analysis_content: Analysis/summary text (empty string if
              ``perform_analysis`` was ``False`` or produced nothing).
            - chunks: Pre-computed chunks (``list[dict]``), or ``None`` when
              none were produced (chunking is then left to the DB layer).
            - chunk_options: The (possibly defaulted) chunking options
              dict, or ``None``.
            - metadata: Extra metadata dict (mirrors
              ``ingest_local_file``'s current behavior, where this is
              computed but not actually forwarded to
              ``add_media_with_keywords``, which has no such parameter --
              preserved as-is rather than silently starting to pass it).
            - file_path: ``str(file_path)`` (not absolutized -- the
              absolutized form only ever appears in ``url``).

    Raises:
        FileNotFoundError: If the file doesn't exist.
        FileIngestionError: If the file type is unsupported (message starts
            with "Unsupported file type"), or if processing fails for any
            other reason (message is
            ``f"Failed to ingest {file_type} file: {inner}"`` -- matching
            ``ingest_local_file``'s historical wrapping so composed callers
            see identical error text regardless of whether the failure
            originated during parsing or persistence).
    """
    raw_source = str(file_path)
    is_url = _is_http_url(raw_source)
    if is_url:
        # URL source: skip the file-path machinery entirely.
        file_type = classify_ingest_source(raw_source)   # "article" | "audio" | "video"
        source_url = raw_source                            # article branch overrides w/ canonical
        # keep file_path as the raw URL string so the audio/video branches'
        # `str(file_path)` passes the URL straight to the URL-accepting processor.
    else:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        try:
            file_type = detect_file_type(file_path)
        except FileIngestionError as e:
            logger.error(f"Unsupported file type: {file_path} - {e}")
            raise
        source_url = f"file://{file_path.absolute()}"

    title = options.get('title')
    author = options.get('author')
    keywords = options.get('keywords')
    custom_prompt = options.get('custom_prompt')
    system_prompt = options.get('system_prompt')
    perform_analysis = options.get('perform_analysis', False)
    api_name = options.get('api_name')
    api_key = options.get('api_key')
    chunk_options = options.get('chunk_options')
    metadata = options.get('metadata')

    # Set default values
    if title is None:
        title = raw_source if is_url else file_path.stem
    if keywords is None:
        keywords = []
    if chunk_options is None:
        chunk_options = {}

    # Prepare common parameters
    common_params = {
        'title': title,
        'author': author or 'Unknown',
        'keywords': ', '.join(keywords) if keywords else '',
        'custom_prompt': custom_prompt,
        'system_prompt': system_prompt,
        'summary': perform_analysis,
        'api_name': api_name,
        'api_key': api_key,
    }
    
    # Get media-specific defaults from config
    from ..config import get_media_ingestion_defaults
    
    # Map file types to media types used in config
    file_type_to_media_type = {
        'pdf': 'pdf',
        'document': 'document',
        'ebook': 'ebook',
        'xml': 'xml',
        'plaintext': 'plaintext',
        'html': 'web_article',  # HTML files map to web_article config
        'audio': 'audio',
        'video': 'video'
    }
    
    media_type = file_type_to_media_type.get(file_type, file_type)
    config_defaults = get_media_ingestion_defaults(media_type)
    
    # Extract defaults with proper key mapping
    defaults = {
        'method': config_defaults.get('chunk_method', 'paragraphs'),
        'size': config_defaults.get('chunk_size', 500),
        'overlap': config_defaults.get('chunk_overlap', 200)
    }
    common_params.update({
        'chunk_method': chunk_options.get('method', defaults.get('method', 'paragraphs')),
        'chunk_size': chunk_options.get('size', defaults.get('size', 500)),
        'chunk_overlap': chunk_options.get('overlap', defaults.get('overlap', 200)),
        'use_adaptive_chunking': chunk_options.get('adaptive', False),
        'use_multi_level_chunking': chunk_options.get('multi_level', False),
        'chunk_language': chunk_options.get('language', ''),
    })
    
    try:
        logger.info(f"Ingesting {file_type} file: {file_path}")
        
        # Process based on file type
        if file_type == 'pdf':
            result = _ensure_process_pdf()(
                file_input=str(file_path),
                filename=file_path.name,
                title_override=title,
                author_override=author,
                keywords=keywords,
                perform_chunking=True,
                chunk_options=chunk_options,
                perform_analysis=perform_analysis,
                api_name=api_name,
                api_key=api_key,
                custom_prompt=custom_prompt,
                system_prompt=system_prompt
            )
            
        elif file_type == 'document':
            result = _ensure_process_document()(
                file_path=str(file_path),
                title_override=title,
                author_override=author,
                keywords=keywords,
                custom_prompt=custom_prompt,
                system_prompt=system_prompt,
                auto_summarize=perform_analysis,
                api_name=api_name,
                api_key=api_key,
                chunk_options=chunk_options
            )
            
        elif file_type == 'ebook':
            result = _ensure_process_ebook()(
                file_path=str(file_path),
                title_override=title,
                author_override=author,
                keywords=keywords,
                custom_prompt=custom_prompt,
                system_prompt=system_prompt,
                perform_chunking=True,
                chunk_options=chunk_options,
                perform_analysis=perform_analysis,
                api_name=api_name,
                api_key=api_key
            )
            
        elif file_type == 'audio':
            # Initialize audio processor. media_db is intentionally None:
            # this function performs no database I/O (see docstring). A
            # real media_db here would make the processor write a degraded
            # row of its own first (bare-path URL, no keywords), which the
            # pipeline's later add_media_with_keywords call then matches
            # via content-hash dedup and no-ops against -- returning
            # media_id=None for every audio ingest (see the docstring's
            # bug-mechanism paragraph and the regression test named there).
            audio_processor = _ensure_local_audio_processor()(None)

            # Process single audio file
            results = audio_processor.process_audio_files(
                inputs=[str(file_path)],
                transcription_model=chunk_options.get('transcription_model', 'base'),
                transcription_language=chunk_options.get('transcription_language', 'en'),
                perform_chunking=True,
                chunk_method=chunk_options.get('method', 'sentences'),
                max_chunk_size=chunk_options.get('size', 500),
                chunk_overlap=chunk_options.get('overlap', 200),
                use_adaptive_chunking=chunk_options.get('adaptive', False),
                use_multi_level_chunking=chunk_options.get('multi_level', False),
                chunk_language=chunk_options.get('language', 'en'),
                diarize=chunk_options.get('diarize', False),
                vad_use=chunk_options.get('vad_filter', False),
                timestamp_option=True,
                perform_analysis=perform_analysis,
                api_name=api_name,
                api_key=api_key,
                custom_prompt=custom_prompt,
                system_prompt=system_prompt,
                summarize_recursively=chunk_options.get('recursive_summary', False),
                custom_title=title,
                author=author
            )
            
            # Extract first (and only) result
            if results['results']:
                result_data = results['results'][0]
                if result_data['status'] == 'Error':
                    raise FileIngestionError(f"Audio processing failed: {result_data.get('error', 'Unknown error')}")
                    
                result = {
                    'content': result_data.get('content', ''),
                    'title': result_data.get('metadata', {}).get('title', title),
                    'author': result_data.get('metadata', {}).get('author', author or 'Unknown'),
                    'keywords': keywords,
                    'chunks': result_data.get('chunks', []),
                    'analysis': result_data.get('analysis', ''),
                    'metadata': result_data.get('metadata', {})
                }
            else:
                raise FileIngestionError("Audio processing returned no results")
                
        elif file_type == 'video':
            # Initialize video processor. media_db is intentionally None --
            # see the matching comment in the 'audio' branch above.
            video_processor = _ensure_local_video_processor()(None)

            # Process single video file
            results = video_processor.process_videos(
                inputs=[str(file_path)],
                download_video_flag=False,  # Extract audio only for transcription
                transcription_model=chunk_options.get('transcription_model', 'base'),
                transcription_language=chunk_options.get('transcription_language', 'en'),
                perform_chunking=True,
                chunk_method=chunk_options.get('method', 'sentences'),
                max_chunk_size=chunk_options.get('size', 500),
                chunk_overlap=chunk_options.get('overlap', 200),
                use_adaptive_chunking=chunk_options.get('adaptive', False),
                use_multi_level_chunking=chunk_options.get('multi_level', False),
                chunk_language=chunk_options.get('language', 'en'),
                diarize=chunk_options.get('diarize', False),
                vad_use=chunk_options.get('vad_filter', False),
                timestamp_option=True,
                perform_analysis=perform_analysis,
                api_name=api_name,
                api_key=api_key,
                custom_prompt=custom_prompt,
                system_prompt=system_prompt,
                summarize_recursively=chunk_options.get('recursive_summary', False),
                custom_title=title,
                author=author
            )
            
            # Extract first (and only) result
            if results['results']:
                result_data = results['results'][0]
                if result_data['status'] == 'Error':
                    raise FileIngestionError(f"Video processing failed: {result_data.get('error', 'Unknown error')}")
                    
                result = {
                    'content': result_data.get('content', ''),
                    'title': result_data.get('metadata', {}).get('title', title),
                    'author': result_data.get('metadata', {}).get('author', author or 'Unknown'),
                    'keywords': keywords,
                    'chunks': result_data.get('chunks', []),
                    'analysis': result_data.get('analysis', ''),
                    'metadata': result_data.get('metadata', {})
                }
            else:
                raise FileIngestionError("Video processing returned no results")
            
        elif file_type == 'xml':
            # XML processing not yet implemented in the expected format
            raise FileIngestionError(f"XML file processing is not yet implemented")
            
        elif file_type == 'plaintext':
            # For plaintext files, we'll process them directly
            content = file_path.read_text(encoding='utf-8', errors='replace')
            
            # Simple result structure for plaintext
            result = {
                'content': content,
                'title': title,
                'author': author or 'Unknown',
                'keywords': keywords,
                'chunks': [],  # Chunking will be handled by the database
                'analysis': ''
            }
            
            # If chunking is requested, we'll let the database handle it
            # The MediaDatabase.add_media_with_keywords method will handle chunking
            
        elif file_type == 'html':
            # For HTML files, we'll extract text content
            from bs4 import BeautifulSoup
            
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Parse HTML and extract text
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract title if present
            title_tag = soup.find('title')
            if title_tag and not title:
                title = title_tag.text.strip()
            
            # Extract text content
            content = soup.get_text(separator='\n', strip=True)
            
            result = {
                'content': content,
                'title': title or file_path.stem,
                'author': author or 'Unknown',
                'keywords': keywords,
                'chunks': [],  # Chunking will be handled by the database
                'analysis': ''
            }

        elif file_type == 'article':
            from .web_article_ingestion import extract_article_for_ingest
            result = extract_article_for_ingest(raw_source, options)
            source_url = result.get('url', source_url)     # canonical post-redirect URL

        # Check if processing was successful. NOTE: some processors (e.g.
        # ``process_pdf``) always initialize an ``'error'`` key (defaulting
        # to ``None``) in their result dict, so ``'error' in result`` is
        # ALWAYS ``True`` -- even on a clean success -- and previously
        # raised ``FileIngestionError`` for every single PDF. Truthiness
        # (``result.get('error')``) is the correct check.
        if not result or result.get('error'):
            error_msg = result.get('error', 'Unknown error') if result else 'Processing returned no result'
            raise FileIngestionError(f"Failed to process {file_type} file: {error_msg}")

        # Extract content and metadata
        content = result.get('content', '')
        extracted_title = result.get('title', title)
        extracted_author = result.get('author', author or 'Unknown')
        extracted_keywords = result.get('keywords', [])
        chunks = result.get('chunks', [])
        analysis = result.get('analysis', '')

        # Combine keywords
        all_keywords = list(set(keywords + extracted_keywords))

        # Add custom metadata. NOTE: media_metadata is computed for parity
        # with ingest_local_file's pre-F3 behavior but is NOT forwarded to
        # add_media_with_keywords (persist_parsed_media), which has no such
        # parameter -- that was already true before this split (a
        # pre-existing no-op), preserved here rather than silently starting
        # to pass it.
        media_metadata = result.get('metadata', {})
        if metadata:
            media_metadata.update(metadata)
        # Preserve how the source was ingested so metadata consumers can tell
        # a web-article extraction from a local-file parse (don't clobber the
        # extractor's 'web_article' marker with 'local_file').
        if is_url:
            media_metadata['ingestion_method'] = 'web_article' if file_type == 'article' else 'url_download'
        else:
            media_metadata['ingestion_method'] = 'local_file'
        media_metadata['file_path'] = raw_source if is_url else str(file_path)
        media_metadata['file_type'] = file_type

        return {
            'media_type': file_type,
            'file_type': file_type,
            'title': extracted_title,
            'author': extracted_author,
            'content': content,
            'keywords': all_keywords,
            'url': source_url,
            'analysis_content': analysis,
            'chunks': chunks if chunks else None,
            'chunk_options': chunk_options if chunk_options else None,
            'metadata': media_metadata,
            'file_path': raw_source if is_url else str(file_path),
        }

    except PermanentIngestError:
        # keep the permanent classification intact for classify_parse_failure
        raise
    except Exception as e:
        logger.error(f"Error parsing {file_type} file {file_path}: {e}")
        raise FileIngestionError(f"Failed to ingest {file_type} file: {str(e)}")


def persist_parsed_media(
    payload: Dict[str, Any],
    media_db: MediaDatabase,
) -> tuple[Optional[int], Optional[str], str]:
    """
    Persist a payload produced by ``parse_local_file_for_ingest`` to the Media database.

    This is the post-parse half of ``ingest_local_file`` (F3 parallel-parse
    split): the single ``add_media_with_keywords`` call plus its logging
    and ingestion-date stamping. It is the *only* place in the ingest
    pipeline that writes to ``media_db``, and is meant to always run on the
    single Library ingest writer thread (never inside a parse worker
    process, which never receives a real ``media_db`` at all).

    Args:
        payload: The dict returned by ``parse_local_file_for_ingest``.
        media_db: The ``MediaDatabase`` instance to write to.

    Returns:
        ``(media_id, media_uuid, message)`` -- exactly
        ``MediaDatabase.add_media_with_keywords``'s return value.

    Raises:
        FileIngestionError: If the database write fails. The message is
            ``f"Failed to ingest {file_type} file: {inner}"``, matching
            ``ingest_local_file``'s historical wrapping (which, before this
            split, wrapped both the parse and the DB-write steps with this
            same message shape) so composed callers see identical error
            text regardless of which stage failed.
    """
    file_type = payload['file_type']
    try:
        logger.debug(f"Storing {file_type} content in database...")
        # Note: add_media_with_keywords returns tuple: (media_id, media_uuid, message)
        media_id, media_uuid, message = media_db.add_media_with_keywords(
            title=payload['title'],
            media_type=payload['media_type'],
            content=payload['content'],
            keywords=payload['keywords'],
            url=payload['url'],
            analysis_content=payload['analysis_content'],
            author=payload['author'],
            ingestion_date=datetime.now().strftime('%Y-%m-%d'),
            chunks=payload['chunks'],
            chunk_options=payload['chunk_options'],
        )
        logger.info(f"Successfully ingested {file_type} file with media_id: {media_id}")
        return media_id, media_uuid, message
    except Exception as e:
        logger.error(f"Error persisting {file_type} file {payload.get('file_path')}: {e}")
        raise FileIngestionError(f"Failed to ingest {file_type} file: {str(e)}")


def ingest_local_file(
    file_path: Union[str, Path],
    media_db: MediaDatabase,
    title: Optional[str] = None,
    author: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    custom_prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    perform_analysis: bool = False,
    api_name: Optional[str] = None,
    api_key: Optional[str] = None,
    chunk_options: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Ingest a local file into the Media database.

    Composes ``parse_local_file_for_ingest`` (extraction, no DB I/O) with
    ``persist_parsed_media`` (the single ``add_media_with_keywords``
    write) -- see those two functions for the pipeline split the F3
    parallel-parse worker pool relies on. This function's signature,
    return shape, and error behavior are unchanged by that split; it
    remains the single programmatic entry point used by
    ``batch_ingest_files``, ``quick_ingest``, the server ingest path, and
    (pre-F3-pool) the app's queue-runner.

    Args:
        file_path: Path to the file to ingest
        media_db: MediaDatabase instance to store the content
        title: Optional title override (defaults to filename)
        author: Optional author name
        keywords: Optional list of keywords
        custom_prompt: Optional custom prompt for analysis
        system_prompt: Optional system prompt for analysis
        perform_analysis: Whether to perform analysis (summarization)
        api_name: API provider name for analysis (if enabled)
        api_key: API key for analysis provider (if needed)
        chunk_options: Dictionary with chunking options:
            - method: 'semantic', 'tokens', 'paragraphs', 'sentences', 'words', 'ebook_chapters'
            - size: chunk size (default varies by method)
            - overlap: chunk overlap (default varies by method)
            - adaptive: use adaptive chunking (bool)
            - multi_level: use multi-level chunking (bool)
            - language: language code for semantic chunking
        metadata: Additional metadata to store with the media

    Returns:
        Dictionary with ingestion results:
            - media_id: ID of the created media entry
            - title: Title of the media
            - author: Author of the media
            - content_length: Length of extracted content
            - chunks_created: Number of chunks created
            - keywords: Keywords associated with the media
            - analysis: Analysis results (if performed)
            - error: Error message (if any)

    Raises:
        FileIngestionError: If ingestion fails
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(file_path)
    options = {
        'title': title,
        'author': author,
        'keywords': keywords,
        'custom_prompt': custom_prompt,
        'system_prompt': system_prompt,
        'perform_analysis': perform_analysis,
        'api_name': api_name,
        'api_key': api_key,
        'chunk_options': chunk_options,
        'metadata': metadata,
    }
    payload = parse_local_file_for_ingest(str(file_path), options)
    media_id, _media_uuid, _message = persist_parsed_media(payload, media_db)

    chunks = payload['chunks']
    return {
        'media_id': media_id,
        'title': payload['title'],
        'author': payload['author'],
        'content_length': len(payload['content']),
        'chunks_created': len(chunks) if chunks else 0,
        'keywords': payload['keywords'],
        'analysis': payload['analysis_content'],
        'file_type': payload['file_type'],
        'file_path': payload['file_path'],
    }


def batch_ingest_files(
    file_paths: List[Union[str, Path]],
    media_db: MediaDatabase,
    common_keywords: Optional[List[str]] = None,
    perform_analysis: bool = False,
    api_name: Optional[str] = None,
    api_key: Optional[str] = None,
    chunk_options: Optional[Dict[str, Any]] = None,
    stop_on_error: bool = False
) -> List[Dict[str, Any]]:
    """
    Ingest multiple files in batch.
    
    Args:
        file_paths: List of file paths to ingest
        media_db: MediaDatabase instance
        common_keywords: Keywords to apply to all files
        perform_analysis: Whether to perform analysis on all files
        api_name: API provider for analysis
        api_key: API key for analysis
        chunk_options: Chunking options for all files
        stop_on_error: Whether to stop on first error or continue
        
    Returns:
        List of ingestion results (one per file)
        
    Raises:
        FileIngestionError: If stop_on_error is True and an error occurs
    """
    start_time = time.time()
    total_files = len(file_paths)
    log_counter("local_file_ingestion_batch_start", labels={
        "total_files": str(total_files),
        "stop_on_error": str(stop_on_error)
    })
    
    results = []
    success_count = 0
    error_count = 0
    
    for file_path in file_paths:
        try:
            result = ingest_local_file(
                file_path=file_path,
                media_db=media_db,
                keywords=common_keywords,
                perform_analysis=perform_analysis,
                api_name=api_name,
                api_key=api_key,
                chunk_options=chunk_options
            )
            results.append(result)
            success_count += 1
            
        except Exception as e:
            error_count += 1
            error_result = {
                'file_path': str(file_path),
                'error': str(e),
                'success': False
            }
            results.append(error_result)
            
            if stop_on_error:
                log_counter("local_file_ingestion_batch_stopped_on_error")
                raise FileIngestionError(f"Batch ingestion stopped: {e}")
            else:
                logger.error(f"Error ingesting {file_path}, continuing with next file: {e}")
    
    # Log batch completion metrics
    duration = time.time() - start_time
    log_histogram("local_file_ingestion_batch_duration", duration, labels={
        "total_files": str(total_files),
        "success_count": str(success_count)
    })
    log_counter("local_file_ingestion_batch_complete", labels={
        "total_files": str(total_files),
        "success_count": str(success_count),
        "error_count": str(error_count)
    })
    
    return results


def ingest_directory(
    directory_path: Union[str, Path],
    media_db: MediaDatabase,
    recursive: bool = False,
    file_types: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Ingest all supported files in a directory.
    
    Args:
        directory_path: Path to directory
        media_db: MediaDatabase instance
        recursive: Whether to process subdirectories
        file_types: List of file types to process (e.g., ['pdf', 'document'])
                   If None, process all supported types
        exclude_patterns: List of filename patterns to exclude (e.g., ['*.tmp', 'draft_*'])
        **kwargs: Additional arguments passed to ingest_local_file
        
    Returns:
        List of ingestion results
    """
    directory_path = Path(directory_path)
    
    if not directory_path.is_dir():
        raise FileIngestionError(f"Not a directory: {directory_path}")
    
    # Get all files
    if recursive:
        files = list(directory_path.rglob('*'))
    else:
        files = list(directory_path.glob('*'))
    
    # Filter to only files (not directories)
    files = [f for f in files if f.is_file()]
    
    # Filter by file type if specified
    if file_types:
        filtered_files = []
        for file_path in files:
            try:
                if detect_file_type(file_path) in file_types:
                    filtered_files.append(file_path)
            except FileIngestionError:
                # Skip unsupported file types
                pass
        files = filtered_files
    else:
        # Filter to only supported file types
        supported_files = []
        for file_path in files:
            try:
                detect_file_type(file_path)
                supported_files.append(file_path)
            except FileIngestionError:
                # Skip unsupported file types
                pass
        files = supported_files
    
    # Apply exclude patterns
    if exclude_patterns:
        import fnmatch
        filtered_files = []
        for file_path in files:
            excluded = False
            for pattern in exclude_patterns:
                if fnmatch.fnmatch(file_path.name, pattern):
                    excluded = True
                    break
            if not excluded:
                filtered_files.append(file_path)
        files = filtered_files
    
    logger.info(f"Found {len(files)} files to ingest in {directory_path}")
    
    # Ingest files
    return batch_ingest_files(
        file_paths=files,
        media_db=media_db,
        **kwargs
    )


# Convenience function for use in scripts
def quick_ingest(file_path: Union[str, Path], db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Quick ingestion function for scripts and notebooks.
    
    Args:
        file_path: Path to file to ingest
        db_path: Optional path to media database (uses default if not provided)
        
    Returns:
        Ingestion result dictionary
    """
    from ..config import get_cli_setting
    
    if db_path is None:
        db_config = get_cli_setting("database", {})
        db_path = db_config.get("media_db_path", "~/.local/share/tldw_cli/tldw_cli_media_v2.db")
        db_path = Path(db_path).expanduser()
    
    # Initialize database
    media_db = MediaDatabase(str(db_path), client_id="quick_ingest")
    
    try:
        result = ingest_local_file(file_path, media_db)
        return result
    finally:
        media_db.close_connection()