# tldw_chatbook/Local_Ingestion/local_file_ingestion.py
"""
Programmatic interface for ingesting local files into the Media database.

This module provides functions to process and store various file types (PDFs, documents,
e-books, etc.) without going through the UI, leveraging existing processing capabilities.
"""

import json
import math
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

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
# `Web_Scraping/Article_Extractor_Lib.py`),
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
process_image = None
LocalAudioProcessor = None
LocalVideoProcessor = None


def _report_ingest_progress(
    progress_callback: Callable[[str, str, float | None], None] | None,
    phase: str,
    message: str,
    percent: float | None = None,
) -> None:
    """Report non-authoritative parse telemetry without affecting parsing."""
    if progress_callback is None:
        return
    try:
        progress_callback(phase, message, percent)
    except Exception:
        return


def _measured_transcription_percent(metadata: object) -> float | None:
    """Return progress only for a finite, bounded measured ratio."""
    if not isinstance(metadata, Mapping):
        return None

    try:
        for current_key, total_key in (
            ("current_time", "total_time"),
            ("chunk", "total_chunks"),
            ("current", "total"),
        ):
            if current_key not in metadata or total_key not in metadata:
                continue
            current_value = metadata[current_key]
            total_value = metadata[total_key]
            if (
                isinstance(current_value, bool)
                or isinstance(total_value, bool)
                or not isinstance(current_value, Real)
                or not isinstance(total_value, Real)
            ):
                return None
            current = float(current_value)
            total = float(total_value)
            if (
                not math.isfinite(current)
                or not math.isfinite(total)
                or total <= 0.0
                or current < 0.0
                or current > total
            ):
                return None
            return (current / total) * 100.0
    except Exception:
        # Provider metadata is non-authoritative telemetry and may be a custom
        # Mapping; hostile lookup/conversion behavior must not affect parsing.
        return None
    return None


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


def _ensure_process_image():
    """Import Image_Processing_Lib.process_image on first actual use (or
    return an already-bound value, e.g. a test's mock)."""
    global process_image
    if process_image is None:
        from .Image_Processing_Lib import process_image as _process_image

        process_image = _process_image
    return process_image


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
from ..DB.Client_Media_DB_v2 import MediaDatabase  # noqa: E402

# (task-21102) The engine-version pin comes from the stdlib-only
# ``chunking_engine_version`` module, NOT from ``Chunking.Chunk_Lib``: this
# module is on the app's boot-import path (app.py / Library.ingest_capabilities
# import it directly), and importing anything under ``tldw_chatbook.Chunking``
# executes the package init and with it the full shim + vendored engine
# (~15k LOC). ``Chunk_Lib.ENGINE_VERSION`` re-exports the same object, so the
# stamp cannot drift. Guarded by
# ``Tests/Packaging/test_chunking_import_closure.py``.
from ..chunking_engine_version import ENGINE_VERSION  # noqa: E402
from ..RAG_Search.ingestion_indexing import suppress_ingestion_indexing  # noqa: E402

# Import metrics
from ..Metrics.metrics_logger import log_counter, log_histogram  # noqa: E402


class FileIngestionError(Exception):
    """Raised when file ingestion fails."""

    pass


class PermanentIngestError(FileIngestionError):
    """A parse/fetch failure that will fail identically on retry (bad URL,
    4xx, non-HTML content, empty extraction, missing extractor dependency).
    ``classify_parse_failure`` maps this to a permanent (non-retryable) job.
    """


class NoContentExtractedError(FileIngestionError):
    """Extraction produced no text, so there was nothing to store.

    (task-14821) Raised by ``_reject_empty_extraction`` BEFORE any write
    is attempted. The Library ingest writer used to stamp every exception
    escaping that stage as ``category="write_error"``, which described a
    write that never happened -- and, being the category shown to the
    user, sent them looking for a database problem instead of the missing
    extractor the message itself named. ``ingest_error_category`` is the
    seam the writer reads instead of hard-coding one.
    """

    ingest_error_category = "no_content"


class EmptySourceIngestError(PermanentIngestError):
    """The source file is 0 bytes: there was nothing to ingest at all.

    (task-14821) Permanent, like every other 0-byte outcome, and reported
    as its own reason rather than as a write failure.
    """

    ingest_error_category = "empty_source"


class DirectLocalSTTIngestError(FileIngestionError):
    """A sanitized direct-local STT failure crossing the spawn boundary."""

    def __init__(
        self,
        message: str,
        *,
        error_detail: dict[str, Any],
        failed_attempt: dict[str, Any],
    ) -> None:
        self.error_detail = error_detail
        self.stt_failure_provenance = failed_attempt
        super().__init__(message)


_VIDEO_URL_HOSTS = ("youtube.com", "youtu.be", "vimeo.com", "dailymotion.com")
_VIDEO_EXTS = (
    ".mp4",
    ".avi",
    ".mkv",
    ".mov",
    ".webm",
    ".flv",
    ".wmv",
    ".m4v",
    ".mpg",
    ".mpeg",
)
_AUDIO_EXTS = (".mp3", ".m4a", ".wav", ".flac", ".ogg", ".aac", ".wma", ".opus")
_TRACKING_PARAMS = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "gclid",
        "fbclid",
        "igshid",
        "mc_cid",
        "mc_eid",
        "ref",
        "ref_src",
    }
)


def is_http_url(source: str) -> bool:
    """Return whether ``source`` is an http or https URL."""
    from urllib.parse import urlparse

    try:
        return urlparse(source).scheme in ("http", "https")
    except Exception:
        return False


# Backward-compatible alias retained for existing callers.
_is_http_url = is_http_url


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
        if any(
            host == h or host.endswith("." + h) for h in _VIDEO_URL_HOSTS
        ) or path.endswith(_VIDEO_EXTS):
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
    if port and not (
        (scheme == "https" and port == 443) or (scheme == "http" and port == 80)
    ):
        netloc = f"{host}:{port}"
    path = parsed.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    query = urlencode(
        sorted(
            (k, v)
            for k, v in parse_qsl(parsed.query, keep_blank_values=True)
            if k.lower() not in _TRACKING_PARAMS
        )
    )
    return urlunparse((scheme, netloc, path, "", query, ""))


def detect_file_type(file_path: Union[str, Path]) -> str:
    """
    Detect the type of file based on its extension.

    Args:
        file_path: Path to the file

    Returns:
        File type as string: 'pdf', 'document', 'ebook', 'plaintext',
        'html', 'image', 'audio', 'video'

    Raises:
        FileIngestionError: If file type is not supported
    """
    file_path = Path(file_path)
    extension = file_path.suffix.lower()

    # PDF files
    if extension == ".pdf":
        return "pdf"

    # Document files
    elif extension in [".doc", ".docx", ".odt", ".rtf"]:
        return "document"

    # E-book files
    elif extension in [".epub", ".mobi", ".azw", ".azw3", ".fb2"]:
        return "ebook"

    # HTML files (web articles)
    elif extension in [".html", ".htm"]:
        return "html"

    # Plain text files
    elif extension in [".txt", ".md", ".markdown", ".rst", ".log", ".csv"]:
        return "plaintext"

    # Image files (task-3307). Exactly the raster formats
    # ``Image_Processing_Lib``'s PIL loader opens on a plain Pillow
    # install: .svg is a vector document PIL cannot rasterize, .ico is an
    # icon container rather than content, and .heic/.heif need the
    # pillow_heif opener no install extra provides -- all three stay
    # honestly unsupported even though ``SUPPORTED_IMAGE_FORMATS`` lists
    # them.
    elif extension in [
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".bmp",
        ".tiff",
        ".tif",
    ]:
        return "image"

    # Audio files
    elif extension in [
        ".mp3",
        ".m4a",
        ".wav",
        ".flac",
        ".ogg",
        ".aac",
        ".wma",
        ".opus",
    ]:
        return "audio"

    # Video files
    elif extension in [
        ".mp4",
        ".avi",
        ".mkv",
        ".mov",
        ".webm",
        ".flv",
        ".wmv",
        ".m4v",
        ".mpg",
        ".mpeg",
    ]:
        return "video"

    else:
        raise FileIngestionError(
            f"Unsupported file type: {extension}. "
            f"Supported types: PDF, DOCX, ODT, RTF, EPUB, MOBI, AZW, FB2, HTML, TXT, MD, "
            f"PNG, JPG, GIF, WEBP, BMP, TIFF, "
            f"MP3, M4A, WAV, FLAC, OGG, AAC, MP4, AVI, MKV, MOV, WEBM"
        )


def get_supported_extensions() -> Dict[str, List[str]]:
    """
    Get all supported file extensions organized by media type.

    Returns:
        Dictionary mapping media types to their supported extensions
    """
    return {
        "pdf": [".pdf"],
        "document": [".doc", ".docx", ".odt", ".rtf"],
        "ebook": [".epub", ".mobi", ".azw", ".azw3", ".fb2"],
        "html": [".html", ".htm"],
        "plaintext": [".txt", ".md", ".markdown", ".rst", ".log", ".csv"],
        "image": [".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif"],
        "audio": [".mp3", ".m4a", ".wav", ".flac", ".ogg", ".aac", ".wma", ".opus"],
        "video": [
            ".mp4",
            ".avi",
            ".mkv",
            ".mov",
            ".webm",
            ".flv",
            ".wmv",
            ".m4v",
            ".mpg",
            ".mpeg",
        ],
    }


# (task-3301) Text-shaped types that produce no chunks of their own: when
# chunking is requested they are chunked here in the parse worker with the
# same ``improved_chunking_process`` the PDF/ebook processors use, and the
# chunks are stored through the same ``persist_parsed_media`` ->
# ``add_media_with_keywords(chunks=...)`` path. There is NO deferred
# chunking pass to defer to: ``Media.chunking_status`` is written but
# consumed nowhere, and the DB layer explicitly ignores ``chunk_options``
# as a placeholder -- so "the database will handle it" (the old comment on
# these branches) handled nothing.
# (task-3307 xhigh review round) ``image`` joined this set: the OCR text is
# text like any other, and routing it through the same tail is what makes
# the form's size/overlap govern it. The image branch calls
# ``process_image`` with ``chunk_options=None`` and clears its convenience
# single chunk so exactly one layer chunks.
_TEXT_CHUNK_TYPES = frozenset(
    {"plaintext", "html", "document", "article", "image"}
)

# Text-shaped types whose branch produces no analysis of its own. The
# ``document`` type is excluded: ``process_document`` runs its own
# ``analyze`` pass internally (``auto_summarize``), so analyzing it again
# here would double the LLM spend. ``image`` is INCLUDED (task-3307): the
# image branch deliberately calls ``process_image`` with
# ``perform_analysis=False`` -- its internal path routes through the
# legacy ``Summarization_General_Lib.analyze`` direct dispatch (the dead
# branch task-3301 documented) and cannot carry the ``[analysis_defaults]``
# call shape or keyless dispatch -- so the OCR text is analyzed by this
# module's shared chat_api_call tail like plaintext/html.
_TEXT_ANALYSIS_TYPES = frozenset({"plaintext", "html", "article", "image"})


def _decode_ingest_text(
    raw: bytes, encoding: Optional[str]
) -> tuple[str, list[str]]:
    """Decode raw text-file bytes per the ingest form's Encoding selection.

    (task-3301) The Encoding select (auto / utf-8 / utf-16 / latin-1 /
    cp1252) was defined in ``ingest_capabilities`` but consumed nowhere --
    plaintext reads hardcoded utf-8-with-replace and HTML reads hardcoded
    *strict* utf-8 (so a latin-1 HTML file failed the whole job).

    Args:
        raw: The file's raw bytes.
        encoding: The selection. ``None``/empty/``"auto"`` mean automatic:
            strict utf-8 first, then chardet detection when available (the
            repo's incumbent detector, already used by
            ``Utils/Utils.safe_read_file``), then utf-8-with-replace as the
            last resort. Any other value is used directly, with
            ``errors="replace"`` so an explicit wrong choice degrades to
            visible replacement characters rather than failing the job.
            (task-3301 xhigh review round 2, F13) That degrade-not-fail
            contract also covers an encoding NAME the codec registry does
            not know (a persisted/typed value like ``utf8-bom``): the
            explicit path used to let ``LookupError`` escape and fail the
            job while the auto path caught the same class.

    Returns:
        ``(text, warnings)`` -- the decoded text, plus warning lines for
        anything that had to be degraded (currently: the unknown-encoding
        fallback). Empty warnings on every clean decode.
    """
    choice = str(encoding or "auto").strip().lower()
    if choice and choice != "auto":
        try:
            return raw.decode(choice, errors="replace"), []
        except (LookupError, ValueError):
            # (F13) Unknown codec name: same fallback the auto path ends
            # on, surfaced as a warning instead of failing the job.
            return raw.decode("utf-8", errors="replace"), [
                f"Unknown text encoding '{choice}'; decoded as UTF-8 "
                "with replacement characters."
            ]
    try:
        return raw.decode("utf-8"), []
    except UnicodeDecodeError:
        pass
    try:  # optional dependency -- degrade quietly when absent
        import chardet

        detected = (chardet.detect(raw) or {}).get("encoding")
        if detected:
            try:
                return raw.decode(detected, errors="replace"), []
            except (LookupError, ValueError):
                pass
    except ImportError:
        pass
    return raw.decode("utf-8", errors="replace"), []


def _chunk_text_for_ingest(
    content: str,
    method: Any,
    max_size: Any,
    overlap: Any,
    template: Any = None,
) -> tuple[list[Dict[str, Any]], list[str]]:
    """Chunk extracted text with the repo's shared chunking service.

    Mirrors ``process_pdf``'s behavior at the seams: an empty result or a
    chunking failure degrades to one full-text chunk plus a warning, never
    a failed job.

    Args:
        content: The extracted text (non-empty).
        method: Chunking method name (``sentences``, ``words``, ...).
        max_size: Target chunk size (display strings are coerced).
        overlap: Chunk overlap (display strings are coerced).
        template: Optional pre-resolved template dict (spec §9.2 -- this
            fresh three-key dict used to be the fourth seam that dropped
            it). Unresolvable/invalid templates never reach here: the
            Library builder refuses them with named errors at option-build
            time (AC 37).

    Returns:
        ``(chunks, warnings)`` -- chunks in the ``{"text", "metadata"}``
        shape ``persist_parsed_media`` stores.
    """

    def _as_int(value: Any, fallback: int) -> int:
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return fallback

    warnings: list[str] = []
    try:
        from ..RAG_Search.chunking_service import improved_chunking_process

        chunk_options: Dict[str, Any] = {
            "method": str(method or "sentences"),
            "max_size": _as_int(max_size, 500),
            "overlap": _as_int(overlap, 100),
        }
        if template is not None:
            # The wrapper pops this key and forwards it as the Chunker
            # keyword -- the same pop-and-forward the pdf path relies on.
            chunk_options["template"] = template
        chunks = improved_chunking_process(content, chunk_options)
    except Exception as chunk_err:
        logger.opt(exception=True).error(f"Text chunking failed: {chunk_err}")
        warnings.append(f"Chunking failed: {chunk_err}")
        return (
            [{"text": content, "metadata": {"chunk_num": 0}}],
            warnings,
        )
    if not chunks:
        warnings.append("Chunking yielded no results; using full text.")
        return (
            [{"text": content, "metadata": {"chunk_num": 0}}],
            warnings,
        )
    return chunks, warnings


#: Default analysis instruction when the caller supplies no custom prompt.
_DEFAULT_ANALYSIS_PROMPT = (
    "Please provide a comprehensive summary of this document."
)

#: (task-3301 xhigh review round 2, F7) Sentinel distinguishing "caller never
#: passed chunk_options" from an explicit ``None``. When task-3301 made
#: ``chunk_options is None`` mean "do not chunk" at the parse seam (the
#: Library queue's Chunk-content OFF state), the public wrappers
#: (``ingest_local_file``/``batch_ingest_files``/``quick_ingest``) silently
#: inherited it through their ``None`` defaults -- out-of-tree callers that
#: previously got default chunking stopped chunking with no signal. The
#: wrappers now default to this sentinel, normalized to ``{}`` ("chunk with
#: defaults"); an EXPLICIT ``None`` still disables chunking. A sentinel
#: object (not a ``{}`` default) because ``process_pdf`` and friends
#: ``setdefault`` into the dict they receive -- a shared mutable default
#: would accrete state across calls.
_CHUNK_WITH_DEFAULTS: Any = object()


def _analysis_failure_reason(analysis: Any) -> Optional[str]:
    """Detect an in-band analysis failure string.

    (task-3301 xhigh review round, F4) ``analyze()``'s documented failure
    mode is RETURNING a string that starts with ``"Error:"`` -- the old
    ``isinstance(str) and strip()`` success check treated exactly that as
    success and persisted it as analysis content.

    Args:
        analysis: A candidate analysis value.

    Returns:
        The failure description (first line, capped) when ``analysis`` is
        an error string; ``None`` when it is not (including when it is
        simply empty -- absence is not failure).
    """
    if not isinstance(analysis, str):
        return None
    stripped = analysis.strip()
    if not stripped.lower().startswith("error:"):
        return None
    first_line = stripped.splitlines()[0].strip()
    reason = first_line[len("error:"):].strip() or first_line
    return reason[:200]


def _extract_chat_response_text(response: Any) -> str:
    """Extract the assistant text from a ``chat_api_call`` response.

    Mirrors the Media viewer's extraction (``UI/MediaWindow_v2.py``):
    plain strings pass through; OpenAI-shaped dicts yield
    ``choices[0].message.content`` (or ``choices[0].text``); bare
    ``{"content": ...}`` dicts yield their content.

    Args:
        response: Whatever the provider handler returned (streaming is
            never requested here, so generators are not expected).

    Returns:
        The extracted text, or ``""`` when no text could be extracted.
    """
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                message = choice.get("message")
                if isinstance(message, dict) and isinstance(
                    message.get("content"), str
                ):
                    return message["content"]
                if isinstance(choice.get("text"), str):
                    return choice["text"]
        if isinstance(response.get("content"), str):
            return response["content"]
    return ""


def _run_chat_analysis(
    *,
    api_name: str,
    api_key: Optional[str],
    content: str,
    custom_prompt: Optional[str],
    system_prompt: Optional[str],
    analysis_call: Optional[Dict[str, Any]],
) -> tuple[str, Optional[str]]:
    """Run one analysis call over the full content via the chat dispatcher.

    (task-3301 xhigh review round, F1+F10) This is the same call shape the
    Media viewer's analysis panel spends through (``app.chat_wrapper`` ->
    ``chat`` -> ``chat_api_call``): the unified dispatcher whose
    ``API_CALL_HANDLERS`` table the ingest seam's provider resolution is
    constrained to, carrying the full ``[analysis_defaults]`` settings
    (model/temperature/top_p/min_p/max_tokens/system prompt). The previous
    tail called ``Summarization_General_Lib.analyze``, whose direct
    dispatch sat in a dead ``else`` branch -- it returned
    ``'Error: Summarization failed unexpectedly.'`` without any API call
    on every normal install -- and it could carry neither model nor token
    settings.

    Args:
        api_name: A chat-dispatchable provider name (an
            ``API_CALL_HANDLERS`` key -- the job-option builder resolves
            display spellings before they get here).
        api_key: Credential, or ``None`` for sanctioned keyless dispatch.
        content: The extracted document text.
        custom_prompt: Analysis instruction; defaults to the shared
            summary prompt.
        system_prompt: System prompt for the call (the builder seeds it
            from ``[analysis_defaults] system_prompt``).
        analysis_call: Optional dict with ``model``/``temperature``/
            ``top_p``/``min_p``/``max_tokens`` from the resolution seam.

    Returns:
        ``(analysis_text, failure_reason)`` -- exactly one of the two is
        non-empty.
    """
    from tldw_chatbook.Library.ingest_analysis import (
        ANALYSIS_DEFAULT_MAX_TOKENS,
        ANALYSIS_DEFAULT_MIN_P,
        ANALYSIS_DEFAULT_TEMPERATURE,
        ANALYSIS_DEFAULT_TOP_P,
    )

    settings = analysis_call if isinstance(analysis_call, dict) else {}
    prompt = custom_prompt or _DEFAULT_ANALYSIS_PROMPT
    user_prompt = f"{prompt}\n\n---\n\nContent to analyze:\n\n{content}"

    try:
        from ..Chat.Chat_Functions import chat_api_call

        response = chat_api_call(
            api_endpoint=api_name,
            messages_payload=[{"role": "user", "content": user_prompt}],
            api_key=api_key,
            temp=settings.get("temperature", ANALYSIS_DEFAULT_TEMPERATURE),
            system_message=system_prompt,
            streaming=False,
            model=settings.get("model"),
            topp=settings.get("top_p", ANALYSIS_DEFAULT_TOP_P),
            minp=settings.get("min_p", ANALYSIS_DEFAULT_MIN_P),
            max_tokens=settings.get("max_tokens", ANALYSIS_DEFAULT_MAX_TOKENS),
        )
    except Exception as call_err:  # noqa: BLE001 - a failed analysis must
        # never fail the import itself; the reason travels as a warning.
        return "", str(call_err)[:200] or call_err.__class__.__name__

    text = _extract_chat_response_text(response)
    failure = _analysis_failure_reason(text)
    if failure:
        return "", failure
    if not text or not text.strip():
        return "", "provider returned an empty analysis"
    return text, None


def parse_local_file_for_ingest(
    file_path: Union[str, Path],
    options: Dict[str, Any],
    *,
    transcription_runner: Optional[Callable[..., Dict[str, Any]]] = None,
    progress_callback: Callable[[str, str, float | None], None] | None = None,
) -> Dict[str, Any]:
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
            ``api_name``, ``api_key``, ``analysis_keyless_ok`` (explicit
            opt-in for keyless analysis dispatch -- set only by the
            Library seam after readiness confirmed keyless-ready),
            ``analysis_call`` (model/temperature/top_p/min_p/max_tokens
            for the analysis call), ``chunk_options``, ``metadata`` --
            the first group mirroring ``ingest_local_file``'s keyword
            arguments of the same names.
        progress_callback: Optional best-effort callback receiving a controlled
            phase, user-facing message, and truthful stage percentage when one
            is observable.

    Returns:
        A payload dict consumed by ``persist_parsed_media``:
            - media_type: Media type string used for the DB write (e.g.
              'pdf', 'document', 'ebook', 'plaintext', 'html', 'image',
              'audio', 'video').
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
              Never an in-band ``"Error: ..."`` string -- those become a
              payload warning plus ``analysis_failed_reason`` instead
              (task-3301 xhigh review round, F4).
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
    _report_ingest_progress(
        progress_callback,
        "inspecting",
        "Inspecting source",
    )
    raw_source = str(file_path)
    is_url = _is_http_url(raw_source)
    if is_url:
        # URL source: skip the file-path machinery entirely.
        file_type = classify_ingest_source(raw_source)  # "article" | "audio" | "video"
        source_url = raw_source  # article branch overrides w/ canonical
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

    title = options.get("title")
    author = options.get("author")
    keywords = options.get("keywords")
    custom_prompt = options.get("custom_prompt")
    system_prompt = options.get("system_prompt")
    perform_analysis = options.get("perform_analysis", False)
    api_name = options.get("api_name")
    api_key = options.get("api_key")
    # (task-3301 xhigh review round, F8) Explicit keyless opt-in: only the
    # Library job-option builder sets this, strictly after the readiness
    # seam said the provider is keyless-ready. Direct callers that pass
    # ``api_name`` without a key keep the historical silent skip.
    keyless_ok = bool(options.get("analysis_keyless_ok"))
    # (task-3301 xhigh review round, F10) Full [analysis_defaults] call
    # shape (model/temperature/top_p/min_p/max_tokens) for the text tail.
    analysis_call = options.get("analysis_call")
    chunk_options = options.get("chunk_options")
    metadata = options.get("metadata")
    encoding = options.get("encoding")
    # (task-3301) ``chunk_options is None`` IS the Chunk-content toggle:
    # the Library queue passes a dict when the option is ON and ``None``
    # when it is OFF. The processors used to be called with a hardcoded
    # ``perform_chunking=True`` regardless, which made the OFF state a
    # silent no-op. Derived BEFORE the ``{}`` defaulting below erases the
    # distinction. (Programmatic callers that pass ``{}`` -- e.g.
    # ``local_media_reading_service`` -- keep their always-chunk behavior:
    # an empty dict is not ``None``.)
    perform_chunking = chunk_options is not None

    # Set default values
    if title is None:
        title = raw_source if is_url else file_path.stem
    if keywords is None:
        keywords = []
    if chunk_options is None:
        chunk_options = {}

    # (task 10, spec §9.1/§9.2) A resolved template travelling in
    # ``chunk_options["template"]`` (placed there by the Library job-option
    # builder) is materialized HERE, once, as this parse's chunk-stage
    # DEFAULTS. Every downstream seam re-injects its own defaults via
    # ``setdefault`` -- process_pdf (sentences/500/100), process_epub/
    # process_fb2 (ebook_chapters/1500/200), the audio/video key-by-key
    # re-projection, the shared text tail's fresh three-key dict -- and
    # those would arrive at the Chunker as EXPLICIT options that beat the
    # template (its merge order is defaults <- template <- explicit):
    # the inert-picker trap. Occupying the keys here makes each of those
    # re-injections a no-op; ``setdefault`` preserves any user-changed
    # value the builder kept, which is the other half of the ruling.
    ingest_template = chunk_options.get("template")
    # (task 4, auto-selection spec §4.3/§4.4) The Auto decision's travel
    # ticket (``{"tier": ..., "rationale": [...]}``, placed by the Library
    # job-option builder when the picker sentinel resolved) is extracted
    # HERE -- before any branch dispatch -- so no processor and never the
    # Chunker sees a non-chunking key, and the persist seam can record
    # ``mode``/``auto_tier``/``auto_rationale`` in ``Media.chunking_config``.
    auto_ticket = chunk_options.pop("auto", None)
    ingest_auto: Optional[Dict[str, Any]] = None
    if isinstance(auto_ticket, dict):
        ingest_auto = {
            "tier": str(auto_ticket.get("tier") or "").strip(),
            "rationale": [
                str(line)
                for line in (auto_ticket.get("rationale") or [])
                if str(line).strip()
            ],
        }
    # (task 11, spec §9.2 tail / AC 38) The template NAME is captured here
    # -- before any branch can consume the dict -- because the persist seam
    # needs it to fill the ``chunking_template``/``chunking_params`` columns
    # and ``Media.chunking_config``. It cannot read it back off
    # ``payload["chunk_options"]`` there: the pdf/document/ebook branches
    # hand the dict to ``improved_chunking_process``, which POPS the
    # ``template`` key (its documented contract), so by persist time the
    # key's presence depends on which branch ran. The resolved dict's
    # ``name`` is authoritative (``resolve_template`` sets it from the row's
    # UNIQUE column).
    ingest_template_name = ""
    if isinstance(ingest_template, dict):
        from ..Chunking.template_runtime import materialize_template_chunk_options

        materialize_template_chunk_options(chunk_options, ingest_template)
        template_name_value = ingest_template.get("name")
        if isinstance(template_name_value, str):
            ingest_template_name = template_name_value.strip()

    # Prepare common parameters
    common_params = {
        "title": title,
        "author": author or "Unknown",
        "keywords": ", ".join(keywords) if keywords else "",
        "custom_prompt": custom_prompt,
        "system_prompt": system_prompt,
        "summary": perform_analysis,
        "api_name": api_name,
        "api_key": api_key,
    }

    # Get media-specific defaults from config
    from ..config import get_media_ingestion_defaults

    # Map file types to media types used in config
    file_type_to_media_type = {
        "pdf": "pdf",
        "document": "document",
        "ebook": "ebook",
        "xml": "xml",
        "plaintext": "plaintext",
        "html": "web_article",  # HTML files map to web_article config
        "image": "image",
        "audio": "audio",
        "video": "video",
    }

    media_type = file_type_to_media_type.get(file_type, file_type)
    config_defaults = get_media_ingestion_defaults(media_type)

    # Extract defaults with proper key mapping
    defaults = {
        "method": config_defaults.get("chunk_method", "paragraphs"),
        "size": config_defaults.get("chunk_size", 500),
        "overlap": config_defaults.get("chunk_overlap", 200),
    }
    common_params.update(
        {
            "chunk_method": chunk_options.get(
                "method", defaults.get("method", "paragraphs")
            ),
            "chunk_size": chunk_options.get("size", defaults.get("size", 500)),
            "chunk_overlap": chunk_options.get("overlap", defaults.get("overlap", 200)),
            "use_adaptive_chunking": chunk_options.get("adaptive", False),
            "use_multi_level_chunking": chunk_options.get("multi_level", False),
            "chunk_language": chunk_options.get("language", ""),
        }
    )

    def transcription_progress(
        _percent: float,
        message: str,
        data: Any = None,
    ) -> None:
        _report_ingest_progress(
            progress_callback,
            "transcribing",
            str(message or "Transcribing audio"),
            _measured_transcription_percent(data),
        )

    try:
        logger.info(f"Ingesting {file_type} file: {file_path}")

        # Process based on file type
        if file_type == "pdf":
            _report_ingest_progress(progress_callback, "processing", "Processing PDF")
            result = _ensure_process_pdf()(
                file_input=str(file_path),
                filename=file_path.name,
                engine=options.get("pdf_engine"),
                page_range=options.get("page_range"),
                ocr=options.get("ocr", False),
                # (task-3303) OCR detail; the ``or`` fallbacks mirror
                # ``process_pdf``'s own declared defaults so a restored job
                # without these keys behaves identically.
                ocr_language=options.get("ocr_language") or "en",
                ocr_backend=options.get("ocr_backend") or "auto",
                extract_images=options.get("extract_images", False),
                title_override=title,
                author_override=author,
                keywords=keywords,
                # (task-3301) The real toggle, not a hardcoded True.
                perform_chunking=perform_chunking,
                chunk_options=chunk_options,
                perform_analysis=perform_analysis,
                api_name=api_name,
                api_key=api_key,
                keyless_ok=keyless_ok,
                custom_prompt=custom_prompt,
                system_prompt=system_prompt,
            )

        elif file_type == "document":
            _report_ingest_progress(
                progress_callback,
                "processing",
                "Processing document",
            )
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
                chunk_options=chunk_options,
                # (task-3303) The document group's own options; fallbacks
                # mirror ``process_document``'s declared defaults.
                processing_method=options.get("processing_method") or "auto",
                enable_ocr=options.get("enable_ocr", False),
                ocr_language=options.get("ocr_language") or "en",
            )

        elif file_type == "ebook":
            _report_ingest_progress(
                progress_callback,
                "processing",
                "Processing ebook",
            )
            result = _ensure_process_ebook()(
                file_path=str(file_path),
                method=options.get("extraction_method"),
                split_chapters=options.get("split_chapters", True),
                include_toc=options.get("include_toc", True),
                title_override=title,
                author_override=author,
                keywords=keywords,
                custom_prompt=custom_prompt,
                system_prompt=system_prompt,
                # (task-3301) The real toggle, not a hardcoded True.
                perform_chunking=perform_chunking,
                chunk_options=chunk_options,
                perform_analysis=perform_analysis,
                api_name=api_name,
                api_key=api_key,
                keyless_ok=keyless_ok,
            )

        elif file_type == "image":
            _report_ingest_progress(
                progress_callback,
                "processing",
                "Processing image",
            )
            # (task-3307, ship ruling in task-3310) The imported CONTENT is
            # the OCR text -- there is no other text in an image -- so the
            # OCR toggle defaults on (mirroring ``process_image``'s own
            # ``enable_ocr=True``) and a no-text parse fails honestly at
            # the persist seam rather than storing an empty row.
            #
            # ``extract_features`` is forced off: the visual-features dict
            # ``process_image`` would compute lands in ``result["visual_
            # features"]``, which this payload does not carry, and
            # ``persist_parsed_media`` forwards no metadata at all -- the
            # toggle would be paid-for compute whose output is dropped.
            # ``perform_analysis`` is forced off too: the processor's
            # internal analysis path is the legacy ``analyze()`` direct
            # dispatch (dead on a normal install, task-3301); the OCR text
            # is analyzed by the shared chat_api_call tail below instead
            # (``image`` is in ``_TEXT_ANALYSIS_TYPES``).
            result = _ensure_process_image()(
                file_path=str(file_path),
                title_override=title,
                author_override=author,
                keywords=keywords,
                enable_ocr=bool(options.get("ocr", True)),
                ocr_backend=options.get("ocr_backend") or "auto",
                ocr_language=options.get("ocr_language") or "en",
                extract_features=False,
                # (task-3307 xhigh review round) ALWAYS None: the shared
                # text-chunk tail below is the single chunking authority
                # for this type (``image`` is in ``_TEXT_CHUNK_TYPES``).
                # Delegating to the processor's own chunking left a hole --
                # it chunks only for a TRUTHY ``chunk_options``, while
                # "chunk ON with nothing typed" arrives as ``{}``, so the
                # OCR text persisted as one whole-text blob whatever size
                # the form asked for. Two chunking layers is how that
                # happened; there is now one. (task 10, spec §9.2) The
                # image branch is therefore template-unaffected BY DESIGN:
                # a resolved template governs the OCR text through the
                # shared tail's widened call, not through process_image.
                chunk_options=None,
                perform_analysis=False,
            )
            if isinstance(result, dict):
                # ``process_image`` returns a convenience single whole-text
                # "chunk" even when it did no chunking; dropping it is what
                # lets the tail's ``not chunks`` branch do the real work.
                result["chunks"] = []

        elif file_type == "audio":
            _report_ingest_progress(
                progress_callback,
                "processing",
                "Processing audio",
            )
            # Initialize audio processor. media_db is intentionally None:
            # this function performs no database I/O (see docstring). A
            # real media_db here would make the processor write a degraded
            # row of its own first (bare-path URL, no keywords), which the
            # pipeline's later add_media_with_keywords call then matches
            # via content-hash dedup and no-ops against -- returning
            # media_id=None for every audio ingest (see the docstring's
            # bug-mechanism paragraph and the regression test named there).
            audio_processor_class = _ensure_local_audio_processor()
            audio_processor = (
                audio_processor_class(None, transcription_runner=transcription_runner)
                if transcription_runner is not None
                else audio_processor_class(None)
            )

            # Process single audio file
            results = audio_processor.process_audio_files(
                inputs=[str(file_path)],
                transcription_provider=options.get(
                    "transcription_provider", "faster-whisper"
                ),
                transcription_model_dir=options.get("transcription_model_dir"),
                transcription_context=options.get("transcription_context"),
                transcription_model=options.get(
                    "transcription_model",
                    chunk_options.get("transcription_model", "base"),
                ),
                transcription_language=options.get(
                    "language",
                    chunk_options.get("transcription_language", "en"),
                ),
                translation_target_language=options.get("translation_target_language"),
                transcription_precision=options.get("transcription_precision"),
                transcription_local_files_only=options.get(
                    "transcription_local_files_only", False
                ),
                transcription_batch_route_resolved=options.get(
                    "transcription_batch_route_resolved", False
                ),
                # (task-3301) The real toggle, not a hardcoded True.
                perform_chunking=perform_chunking,
                chunk_method=chunk_options.get("method", "sentences"),
                max_chunk_size=chunk_options.get("size", 500),
                chunk_overlap=chunk_options.get("overlap", 200),
                use_adaptive_chunking=chunk_options.get("adaptive", False),
                use_multi_level_chunking=chunk_options.get("multi_level", False),
                chunk_language=chunk_options.get("language", "en"),
                # (task 10, spec §9.2) The key-by-key re-projection used to
                # drop any key it did not name -- a template travelling in
                # chunk_options died here. The scalars above now carry the
                # materialized template options; the template dict itself
                # rides this explicit kwarg so the chunk site's Chunker
                # template path is genuinely engaged.
                chunk_template=ingest_template,
                diarize=options.get("diarization", chunk_options.get("diarize", False)),
                # (task-3303) The panel's VAD toggle travels as its own
                # option; the chunk-options spelling stays as a fallback for
                # older callers that tucked it in there.
                vad_use=bool(
                    options.get("vad_filter", chunk_options.get("vad_filter", False))
                ),
                timestamp_option=options.get("timestamps", True),
                # (task-3306) Time-range trim: ffmpeg for local files,
                # yt-dlp postprocessor args for YouTube URLs. NOTE: the
                # panel's cookies_file is deliberately NOT forwarded here
                # -- ``download_audio_file`` parses a cookies string as a
                # JSON dict (raw Cookie header), so the video path's
                # cookiefile PATH would raise JSONDecodeError and fail the
                # job, and the audio YouTube path ignores cookies anyway.
                start_time=options.get("start_time"),
                end_time=options.get("end_time"),
                perform_analysis=perform_analysis,
                api_name=api_name,
                api_key=api_key,
                custom_prompt=custom_prompt,
                system_prompt=system_prompt,
                # (task-3306) The panel's own option travels first; the
                # chunk-options spelling stays as a fallback for older
                # callers that tucked it in there.
                summarize_recursively=bool(
                    options.get(
                        "summarize_recursively",
                        chunk_options.get("recursive_summary", False),
                    )
                ),
                custom_title=title,
                author=author,
                **(
                    {"transcription_progress_callback": transcription_progress}
                    if progress_callback is not None
                    else {}
                ),
            )

            # Extract first (and only) result
            if results["results"]:
                result_data = results["results"][0]
                if result_data["status"] == "Error":
                    if (
                        result_data.get("error_detail", {}).get("category")
                        == "stt_failure"
                    ):
                        raise DirectLocalSTTIngestError(
                            result_data.get("error", "Speech-to-text failed."),
                            error_detail=result_data["error_detail"],
                            failed_attempt=result_data["stt_failure_provenance"],
                        )
                    raise FileIngestionError(
                        f"Audio processing failed: {result_data.get('error', 'Unknown error')}"
                    )

                result = {
                    "content": result_data.get("content", ""),
                    "title": result_data.get("metadata", {}).get("title", title),
                    "author": result_data.get("metadata", {}).get(
                        "author", author or "Unknown"
                    ),
                    "keywords": keywords,
                    "chunks": result_data.get("chunks", []),
                    "analysis": result_data.get("analysis", ""),
                    "metadata": result_data.get("metadata", {}),
                    "transcription_model": result_data.get("transcription_model"),
                    "transcription_provenance": result_data.get(
                        "transcription_provenance"
                    ),
                }
            else:
                raise FileIngestionError("Audio processing returned no results")

        elif file_type == "video":
            _report_ingest_progress(
                progress_callback,
                "processing",
                "Processing video",
            )
            # Initialize video processor. media_db is intentionally None --
            # see the matching comment in the 'audio' branch above.
            video_processor_class = _ensure_local_video_processor()
            video_processor = (
                video_processor_class(None, transcription_runner=transcription_runner)
                if transcription_runner is not None
                else video_processor_class(None)
            )

            # Process single video file
            results = video_processor.process_videos(
                inputs=[str(file_path)],
                download_video_flag=False,  # Extract audio only for transcription
                transcription_provider=options.get(
                    "transcription_provider", "faster-whisper"
                ),
                transcription_model_dir=options.get("transcription_model_dir"),
                transcription_context=options.get("transcription_context"),
                transcription_model=options.get(
                    "transcription_model",
                    chunk_options.get("transcription_model", "base"),
                ),
                transcription_language=options.get(
                    "language",
                    chunk_options.get("transcription_language", "en"),
                ),
                translation_target_language=options.get("translation_target_language"),
                transcription_precision=options.get("transcription_precision"),
                transcription_local_files_only=options.get(
                    "transcription_local_files_only", False
                ),
                transcription_batch_route_resolved=options.get(
                    "transcription_batch_route_resolved", False
                ),
                # (task-3301) The real toggle, not a hardcoded True.
                perform_chunking=perform_chunking,
                chunk_method=chunk_options.get("method", "sentences"),
                max_chunk_size=chunk_options.get("size", 500),
                chunk_overlap=chunk_options.get("overlap", 200),
                use_adaptive_chunking=chunk_options.get("adaptive", False),
                use_multi_level_chunking=chunk_options.get("multi_level", False),
                chunk_language=chunk_options.get("language", "en"),
                # (task 10, spec §9.2) Same widened re-projection as the
                # audio branch: the template rides an explicit kwarg
                # (video's ``process_videos(**kwargs)`` forwards it into
                # the shared audio chunk site).
                chunk_template=ingest_template,
                diarize=options.get("diarization", chunk_options.get("diarize", False)),
                # (task-3303) The panel's VAD toggle travels as its own
                # option; the chunk-options spelling stays as a fallback for
                # older callers that tucked it in there.
                vad_use=bool(
                    options.get("vad_filter", chunk_options.get("vad_filter", False))
                ),
                timestamp_option=options.get("timestamps", True),
                # (task-3306) Time-range trim (applied once, at audio
                # extraction -- ``_process_single_video`` drops the bounds
                # before delegating so the audio stage cannot re-cut) and
                # gated-download cookies (a cookiefile PATH for yt-dlp;
                # never raw cookie text -- see ``_ingest_job_options``).
                start_time=options.get("start_time"),
                end_time=options.get("end_time"),
                use_cookies=bool(options.get("use_cookies", False)),
                cookies=options.get("cookies"),
                perform_analysis=perform_analysis,
                api_name=api_name,
                api_key=api_key,
                custom_prompt=custom_prompt,
                system_prompt=system_prompt,
                # (task-3306) Panel option first, chunk-options fallback.
                summarize_recursively=bool(
                    options.get(
                        "summarize_recursively",
                        chunk_options.get("recursive_summary", False),
                    )
                ),
                custom_title=title,
                author=author,
                **(
                    {"transcription_progress_callback": transcription_progress}
                    if progress_callback is not None
                    else {}
                ),
            )

            # Extract first (and only) result
            if results["results"]:
                result_data = results["results"][0]
                if result_data["status"] == "Error":
                    if (
                        result_data.get("error_detail", {}).get("category")
                        == "stt_failure"
                    ):
                        raise DirectLocalSTTIngestError(
                            result_data.get("error", "Speech-to-text failed."),
                            error_detail=result_data["error_detail"],
                            failed_attempt=result_data["stt_failure_provenance"],
                        )
                    raise FileIngestionError(
                        f"Video processing failed: {result_data.get('error', 'Unknown error')}"
                    )

                result = {
                    "content": result_data.get("content", ""),
                    "title": result_data.get("metadata", {}).get("title", title),
                    "author": result_data.get("metadata", {}).get(
                        "author", author or "Unknown"
                    ),
                    "keywords": keywords,
                    "chunks": result_data.get("chunks", []),
                    "analysis": result_data.get("analysis", ""),
                    "metadata": result_data.get("metadata", {}),
                    "transcription_model": result_data.get("transcription_model"),
                    "transcription_provenance": result_data.get(
                        "transcription_provenance"
                    ),
                }
            else:
                raise FileIngestionError("Video processing returned no results")

        elif file_type == "xml":
            # XML processing not yet implemented in the expected format
            raise FileIngestionError("XML file processing is not yet implemented")

        elif file_type == "plaintext":
            _report_ingest_progress(
                progress_callback,
                "processing",
                "Processing text file",
            )
            # (task-3301) Decoded per the form's Encoding selection; chunks
            # and analysis are produced in the shared text-type tail below --
            # the old "chunking will be handled by the database" comment
            # described a placeholder that never chunked anything.
            content, decode_warnings = _decode_ingest_text(
                file_path.read_bytes(), encoding
            )

            # Simple result structure for plaintext
            result = {
                "content": content,
                "title": title,
                "author": author or "Unknown",
                "keywords": keywords,
                "chunks": [],
                "analysis": "",
                "warnings": decode_warnings,
            }

        elif file_type == "html":
            _report_ingest_progress(
                progress_callback,
                "processing",
                "Processing HTML file",
            )
            # For HTML files, we'll extract text content
            from bs4 import BeautifulSoup

            # (task-3301) Decoded per the form's Encoding selection -- the
            # old strict-utf-8 open failed the whole job on latin-1 bytes.
            html_content, decode_warnings = _decode_ingest_text(
                file_path.read_bytes(), encoding
            )

            # Parse HTML and extract text
            soup = BeautifulSoup(html_content, "html.parser")

            # Extract title if present
            title_tag = soup.find("title")
            if title_tag and not title:
                title = title_tag.text.strip()

            # Extract text content
            content = soup.get_text(separator="\n", strip=True)

            result = {
                "content": content,
                "title": title or file_path.stem,
                "author": author or "Unknown",
                "keywords": keywords,
                "chunks": [],  # produced by the shared text-type tail below
                "analysis": "",
                "warnings": decode_warnings,
            }

        elif file_type == "article":
            _report_ingest_progress(
                progress_callback,
                "processing",
                "Processing web article",
            )
            from .web_article_ingestion import extract_article_for_ingest

            result = extract_article_for_ingest(raw_source, options)
            source_url = result.get("url", source_url)  # canonical post-redirect URL

        # Check if processing was successful. NOTE: some processors (e.g.
        # ``process_pdf``) always initialize an ``'error'`` key (defaulting
        # to ``None``) in their result dict, so ``'error' in result`` is
        # ALWAYS ``True`` -- even on a clean success -- and previously
        # raised ``FileIngestionError`` for every single PDF. Truthiness
        # (``result.get('error')``) is the correct check.
        if not result or result.get("error"):
            error_msg = (
                result.get("error", "Unknown error")
                if result
                else "Processing returned no result"
            )
            raise FileIngestionError(f"Failed to process {file_type} file: {error_msg}")

        # Extract content and metadata
        content = result.get("content", "")
        extracted_title = result.get("title", title)
        extracted_author = result.get("author", author or "Unknown")
        extracted_keywords = result.get("keywords", [])
        chunks = result.get("chunks", [])
        analysis = result.get("analysis", "")
        warnings = result.get("warnings", []) if result else []

        # (task-3301) ``process_document`` reports its internal analysis
        # under ``summary``; surface it as the payload's analysis rather
        # than dropping it on the floor.
        if not analysis and isinstance(result.get('summary'), str):
            analysis = result['summary']

        # (task-3301 xhigh review round, F4) A processor "analysis" that is
        # an in-band error string (``analyze()`` RETURNS its failures as
        # strings starting with 'Error:') must never persist as analysis
        # content -- it becomes a visible warning + done-row annotation,
        # and the import itself stays successful.
        analysis_failed_reason = _analysis_failure_reason(analysis)
        if analysis_failed_reason:
            warnings = list(warnings) + [
                f"Analysis failed: {analysis_failed_reason}"
            ]
            analysis = ""

        if not perform_chunking:
            # (task-3301) Chunk OFF means no chunk rows. Several processors
            # return a single full-text "chunk" as an internal convenience
            # even with chunking disabled (e.g. ``process_pdf``'s
            # consistency fallback); storing it would make the OFF state
            # indistinguishable from a one-chunk ON state in the DB.
            chunks = []
        elif (
            file_type in _TEXT_CHUNK_TYPES
            and not chunks
            and content
        ):
            # (task-3301) Chunk ON must chunk text types too. These
            # branches produce no chunks of their own, and no deferred
            # pass exists downstream (``add_media_with_keywords`` ignores
            # ``chunk_options`` as a placeholder), so the form's
            # size/overlap are applied right here with the same chunking
            # service the PDF path uses. An explicit method wins; the
            # default is the service's own word method, whose size unit
            # (words) is what the form's hint advertises -- the config's
            # per-media ``chunk_method`` is deliberately NOT consulted
            # here, because its methods size in different units
            # (sentences/paragraphs) than the form promises.
            _report_ingest_progress(
                progress_callback,
                "chunking",
                "Chunking extracted text",
            )
            chunks, chunk_warnings = _chunk_text_for_ingest(
                content,
                chunk_options.get("method") or "words",
                chunk_options.get("max_size", chunk_options.get("size", 500)),
                chunk_options.get("overlap", 100),
                template=ingest_template,
            )
            warnings = list(warnings) + chunk_warnings

        if (
            perform_analysis
            and api_name
            and (api_key or keyless_ok)
            and not analysis
            and not analysis_failed_reason
            and content
            and file_type in _TEXT_ANALYSIS_TYPES
        ):
            # (task-3301, reworked by the xhigh review round) Analyze-after-
            # import for the text types that hardcoded ``analysis: ""``.
            # One call over the full content through ``chat_api_call`` --
            # the Media viewer's own dispatch path -- with the full
            # ``[analysis_defaults]`` call shape (F1+F10); the credential
            # gate mirrors the processors' (F8). A failure is a warning on
            # the payload plus a done-row annotation, never a failed job.
            _report_ingest_progress(
                progress_callback,
                "analyzing",
                "Analyzing extracted text",
            )
            analysis_text, tail_failure = _run_chat_analysis(
                api_name=api_name,
                api_key=api_key,
                content=content,
                custom_prompt=custom_prompt,
                system_prompt=system_prompt,
                analysis_call=analysis_call,
            )
            if tail_failure:
                logger.warning(
                    f"Analysis failed for {file_path}: {tail_failure}"
                )
                warnings = list(warnings) + [f"Analysis failed: {tail_failure}"]
                analysis_failed_reason = tail_failure
            else:
                analysis = analysis_text

        # Combine keywords
        all_keywords = list(set(keywords + extracted_keywords))

        # Add custom metadata. NOTE: media_metadata is computed for parity
        # with ingest_local_file's pre-F3 behavior but is NOT forwarded to
        # add_media_with_keywords (persist_parsed_media), which has no such
        # parameter -- that was already true before this split (a
        # pre-existing no-op), preserved here rather than silently starting
        # to pass it.
        media_metadata = result.get("metadata", {})
        if metadata:
            media_metadata.update(metadata)
        # Preserve how the source was ingested so metadata consumers can tell
        # a web-article extraction from a local-file parse (don't clobber the
        # extractor's 'web_article' marker with 'local_file').
        if is_url:
            media_metadata["ingestion_method"] = (
                "web_article" if file_type == "article" else "url_download"
            )
        else:
            media_metadata["ingestion_method"] = "local_file"
        media_metadata["file_path"] = raw_source if is_url else str(file_path)
        media_metadata["file_type"] = file_type

        payload = {
            "media_type": file_type,
            "file_type": file_type,
            "title": extracted_title,
            "author": extracted_author,
            "content": content,
            "keywords": all_keywords,
            "url": source_url,
            "analysis_content": analysis,
            "chunks": chunks if chunks else None,
            "chunk_options": chunk_options if chunk_options else None,
            "metadata": media_metadata,
            "file_path": raw_source if is_url else str(file_path),
            "warnings": warnings,
            "transcription_model": result.get("transcription_model"),
            "transcription_provenance": result.get("transcription_provenance"),
        }
        # (task 11, AC 38) Travel ticket for the persist seam: which named
        # template governed this parse (empty/absent = plain options). See
        # the capture comment at the materialization site for why this is a
        # dedicated key rather than re-reading chunk_options["template"].
        if ingest_template_name:
            payload["chunking_template"] = ingest_template_name
        # (task 4, auto-selection spec §4.4) The Auto decision's ticket:
        # present exactly when the picker sentinel resolved for this parse
        # (any tier); the persist seam turns it into ``mode``/``auto_tier``/
        # ``auto_rationale`` on ``Media.chunking_config``.
        if ingest_auto is not None:
            payload["chunking_auto"] = ingest_auto
        # (task-3301) Analysis was requested but the job-option builder
        # found no callable provider: carry the reason through so the
        # queue's done row can say "analysis skipped: ..." instead of the
        # analysis being silently absent.
        skip_reason = str(options.get("analysis_skipped_reason") or "").strip()
        if perform_analysis and skip_reason:
            payload["analysis_skipped_reason"] = skip_reason
        # (task-3301 xhigh review round, F4) Analysis RAN and failed
        # (in-band error string or provider exception): carry the reason
        # so the done row can say "analysis failed: ..." -- same
        # annotation mechanism as the skip reason.
        if analysis_failed_reason:
            payload["analysis_failed_reason"] = analysis_failed_reason
        # (task-3306 xhigh review round) The option boundary rejected the
        # configured cookies path (missing/unsafe). Carry the reason so the
        # done row can say "cookies ignored: ..." -- the same annotation
        # mechanism as the analysis reasons -- AND surface it as a payload
        # warning, instead of the downloader logging "Invalid cookie
        # format" at debug and running un-authenticated.
        cookies_problem = str(options.get("cookies_problem") or "").strip()
        if cookies_problem:
            payload["cookies_problem"] = cookies_problem
            payload["warnings"] = list(payload["warnings"]) + [cookies_problem]
        return payload

    except DirectLocalSTTIngestError:
        raise
    except PermanentIngestError:
        # keep the permanent classification intact for classify_parse_failure
        raise
    except Exception as e:
        logger.error(f"Error parsing {file_type} file {file_path}: {e}")
        raise FileIngestionError(f"Failed to ingest {file_type} file: {str(e)}")


def _reject_empty_extraction(payload: Dict[str, Any], file_type: str) -> None:
    """Fail a parse that produced no text, rather than storing an empty row.

    An import that extracts nothing used to be written as a media row with
    empty content, reported as done in the queue and counted in the library
    total -- an entry that looks imported but silently returns nothing from
    search and RAG, with no signal to the user that anything went wrong.

    An empty *source* is reported differently from a failed extraction: the
    first is the file being what it is, the second means the content is there
    but this install could not read it (often missing optional tooling).

    Args:
        payload: The dict returned by ``parse_local_file_for_ingest``.
        file_type: Detected type, used in the message.

    Raises:
        FileIngestionError: When the payload carries no usable content.
    """
    if (payload.get("content") or "").strip():
        return

    source = payload.get("file_path") or ""
    name = Path(source).name or source or "this source"

    # URL sources never touch the filesystem, so only stat real local paths.
    if source and not is_http_url(source):
        try:
            if Path(source).stat().st_size == 0:
                # (task-2015) A zero-byte file fails identically on every
                # attempt -- permanent, so the queue row withholds Retry
                # (the retryable raise below stays retryable: installing
                # the missing tooling genuinely can fix an extraction miss).
                raise EmptySourceIngestError(
                    f"{name} is empty; there was nothing to ingest."
                )
        except OSError:
            # Unreadable/vanished: treat as an extraction failure below.
            pass

    if file_type == "image":
        # (task-3307) The generic copy below ("may be scanned images") reads
        # as nonsense for a file that IS an image. Retryable on purpose:
        # switching Extract text (OCR) on, or installing an OCR backend,
        # genuinely can fix the next attempt.
        raise NoContentExtractedError(
            f"No text was found in {name}. An image import stores the text "
            "OCR extracts; turn Extract text (OCR) on and install an OCR "
            "backend (docling, tesseract, easyocr, paddleocr, or docext)."
        )

    raise NoContentExtractedError(
        f"No text could be extracted from {name}. The {file_type} content may "
        "be scanned images, or the tooling for this file type may not be "
        "installed."
    )


def _effective_chunk_params(chunk_options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The flat chunk-stage parameters a template/persist run was governed by.

    Draws ``method`` / ``size`` / ``overlap`` out of the (post-materialization)
    chunk options, preferring the ``max_size`` spelling the template contract
    uses but storing the ``size`` spelling the flat chunk contract carries.
    Absent keys are omitted, never ``None`` -- the same present-but-``None``
    rule the template chunk contract itself follows.
    """
    opts = chunk_options if isinstance(chunk_options, dict) else {}
    params: Dict[str, Any] = {}
    if isinstance(opts.get("method"), str) and opts["method"]:
        params["method"] = opts["method"]
    size = opts.get("max_size", opts.get("size"))
    if size is not None:
        params["size"] = size
    if opts.get("overlap") is not None:
        params["overlap"] = opts["overlap"]
    return params


def _persist_chunking_template_columns(
    media_db: MediaDatabase,
    media_id: int,
    template_name: str,
    chunk_options: Optional[Dict[str, Any]],
    auto_decision: Optional[Dict[str, Any]] = None,
) -> None:
    """Record which template chunked ``media_id`` (task 11, spec §9.2 / AC
    38) -- and, when the Auto sentinel resolved the parse, the decision
    itself (task 4, auto-selection spec §4.4).

    Fills, in ONE transaction at the single Library ingest writer seam:

    * ``UnvectorizedMediaChunks.chunking_template`` / ``chunking_params`` --
      the columns migration v1->v2 added and nothing had ever written,
      alongside the ``chunk_engine_version`` stamp (template-tier ONLY,
      including a template-tier Auto win -- the winning template's
      name/params, exactly as a manual pick); and
    * ``Media.chunking_config`` -- the per-media stored choice the re-chunk
      resolution order (§9.1) reads first.

    The ``chunking_config`` JSON shape is dictated by BOTH existing readers
    and must round-trip them:

    * ``ChunkingTemplateLibrary.get_documents_using_template`` matches
      ``chunking_config LIKE '%"template": "<name>"%'`` -- so the JSON MUST
      keep ``json.dumps``' DEFAULT separators (``", "`` / ``": "``). A
      compact-separator dump would satisfy the ``json_extract`` reader while
      silently never matching the LIKE (a name that queries as unused).
    * ``ChunkingTemplateLibrary.get_template_statistics`` groups by
      ``json_extract(chunking_config, '$.template')`` -- so ``template`` must
      be a TOP-LEVEL string key.

    (task 4, auto-selection spec §4.4) When ``auto_decision`` is present
    (``{"tier": ..., "rationale": [...]}`` -- the parse seam's ticket), the
    config gains ``mode: "auto"``, ``auto_tier`` and ``auto_rationale``
    BEFORE the template key; the ``template`` key itself appears only on a
    template-tier win, so both #2 readers keep matching template-tier rows
    and never match plan/plain-tier rows. No schema change -- everything
    rides the existing JSON column. The method/chunk_size/chunk_overlap
    continuity keys ride for every recorded row (what actually governed).

    The column shape mirrors the dead ``MediaDetailsWidget`` writer's
    (``template`` / ``chunk_size`` / ``chunk_overlap`` / ``method``) for
    continuity with the only writer the JSON column has ever had.
    """
    params = _effective_chunk_params(chunk_options)
    # Key order matters for the chunking_params string only in that tests
    # pin the canonical spelling; the column is read as JSON, not matched.
    chunking_params_json = json.dumps(params)
    config: Dict[str, Any] = {}
    if auto_decision is not None:
        config["mode"] = "auto"
        config["auto_tier"] = str(auto_decision.get("tier") or "").strip()
        config["auto_rationale"] = list(
            auto_decision.get("rationale") or []
        )
    if template_name:
        config["template"] = template_name
    if "method" in params:
        config["method"] = params["method"]
    if "size" in params:
        config["chunk_size"] = params["size"]
    if "overlap" in params:
        config["chunk_overlap"] = params["overlap"]
    # DEFAULT separators are load-bearing (see docstring) -- never pass
    # ``separators=`` here. ``ensure_ascii=False`` is load-bearing the same
    # way: with the default escaping, a non-ASCII template name would be
    # stored as ``\uXXXX`` escapes, which the LIKE reader (matching the
    # literal name) silently never matches -- the name would query as
    # unused (task-14 carried minor from the task-11 review).
    chunking_config_json = json.dumps(config, ensure_ascii=False)
    # Sync-validation triggers on both tables require version to increment
    # by exactly 1 on UPDATE (and client_id/uuid to survive unchanged);
    # ``version = version + 1`` satisfies that without reading the rows
    # first. ``last_modified`` mirrors the DB layer's own UTC-ISO spelling.
    # No sync-log event is written for this bump: the rows were INSERTed
    # moments ago by this same writer thread with the template fields
    # already riding the insert-time sync payloads (the chunk dicts are
    # stamped before ``add_media_with_keywords``); the UPDATE fills the
    # local columns the INSERT statement does not carry.
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    with media_db.transaction() as conn:
        if template_name:
            # Template-tier only (spec §4.4): the winning template's
            # name/params on the chunk rows, exactly as #2 wrote them.
            conn.execute(
                "UPDATE UnvectorizedMediaChunks "
                "SET chunking_template = ?, chunking_params = ?, "
                "last_modified = ?, version = version + 1 "
                "WHERE media_id = ? AND deleted = 0",
                (template_name, chunking_params_json, now, media_id),
            )
        conn.execute(
            "UPDATE Media SET chunking_config = ?, last_modified = ?, "
            "version = version + 1 WHERE id = ?",
            (chunking_config_json, now, media_id),
        )


def persist_parsed_media(
    payload: Dict[str, Any],
    media_db: MediaDatabase,
    *,
    overwrite_existing: bool = False,
    generate_embeddings: bool = True,
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
        overwrite_existing: Whether a live matching row is updated in place.
            Defaults to the historical duplicate-skip behavior.
        generate_embeddings: Whether this write should enqueue best-effort
            semantic indexing. Source persistence is unaffected.

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

    task-4022 (review round 2): this is the ONE caller in the codebase
    that wants "re-importing this file un-trashes it" -- it's the real
    Library ingest writer, and the user re-importing a file they
    previously deleted is asking for exactly that. It passes
    ``restore_trashed=True`` explicitly. Every other
    ``add_media_with_keywords`` caller (chatbook SKIP-conflict imports,
    reading-list bulk imports, Console "save message as media", ...)
    leaves the flag at its default ``False`` and a trashed match is left
    untouched, same as before task-4022 ever existed.

    P1 re-critique finding 2: ``parse_local_file_for_ingest`` always
    normalizes ``payload["keywords"]`` to a list (``[]`` when the user
    typed none and nothing was auto-extracted -- see its own ``if
    keywords is None: keywords = []``), never ``None``. The DB layer now
    distinguishes "keywords argument omitted" (preserve existing curated
    keywords on a restore) from "keywords argument is an explicit empty
    list" (clear them) -- so passing ``payload["keywords"]`` through
    unchanged would silently WIPE a restored row's curated keywords on
    every plain re-import where the user didn't retype them, which is
    exactly the data loss task-4022 was written to prevent. ``or None``
    below restores the "nothing to contribute" signal this caller always
    means whenever the list is empty.
    """
    file_type = payload["file_type"]
    _reject_empty_extraction(payload, file_type)
    template_name = str(payload.get("chunking_template") or "").strip()
    try:
        logger.debug(f"Storing {file_type} content in database...")
        # task-12 (spec §8): stamp every chunk with the chunking engine
        # version at the ONE persist seam, so the DB writer
        # (``_persist_chunks``) persists it to the new
        # ``UnvectorizedMediaChunks.chunk_engine_version`` column. Top-level
        # key only (the chunker's ``metadata`` copy exists for in-memory
        # consumers); non-dict entries are skipped defensively -- the DB
        # writer already skips them, and pre-stamped chunks are not
        # overwritten (a future engine bump changes the value, not the rule).
        # (task 11, spec §9.2 / AC 38) The template stamp rides the same
        # setdefault pattern so the sync-event payload (which spreads the
        # whole chunk dict) carries the same truth the columns do; the DB
        # writer has no ``chunking_template``/``chunking_params`` columns in
        # its INSERT, so the actual column fill is the UPDATE below.
        chunking_params_json = json.dumps(
            _effective_chunk_params(payload.get("chunk_options"))
        )
        for chunk in payload.get("chunks") or []:
            if isinstance(chunk, dict):
                chunk.setdefault("chunk_engine_version", ENGINE_VERSION)
                if template_name:
                    chunk.setdefault("chunking_template", template_name)
                    chunk.setdefault("chunking_params", chunking_params_json)
        # Note: add_media_with_keywords returns tuple: (media_id, media_uuid, message)
        def _persist() -> tuple[Optional[int], Optional[str], str]:
            return media_db.add_media_with_keywords(
                title=payload["title"],
                media_type=payload["media_type"],
                content=payload["content"],
                keywords=payload["keywords"] or None,
                url=payload["url"],
                analysis_content=payload["analysis_content"],
                author=payload["author"],
                transcription_model=payload.get("transcription_model"),
                transcription_provenance=payload.get("transcription_provenance"),
                ingestion_date=datetime.now().strftime("%Y-%m-%d"),
                chunks=payload["chunks"],
                chunk_options=payload["chunk_options"],
                overwrite=overwrite_existing,
                restore_trashed=True,
            )

        if generate_embeddings:
            media_id, media_uuid, message = _persist()
        else:
            with suppress_ingestion_indexing():
                media_id, media_uuid, message = _persist()
        # (task 11, spec §9.2 tail / AC 38) Persisted chunks carry the
        # template columns alongside the engine-version stamp, and the
        # Media row records the per-media stored choice for the re-chunk
        # resolution order. Only when a template was actually used: the
        # no-template path writes nothing (byte-identical to today).
        # (task 4, auto-selection spec §4.4) An Auto-resolved parse records
        # its decision on EVERY tier -- template rows additionally carry
        # the winning template's name/params (the readers' shape), plan and
        # plain rows carry mode/auto_tier/auto_rationale with NO template
        # key. Still nothing when Auto was never chosen.
        auto_decision = payload.get("chunking_auto")
        if not isinstance(auto_decision, dict):
            auto_decision = None
        if (template_name or auto_decision) and media_id is not None:
            _persist_chunking_template_columns(
                media_db,
                media_id,
                template_name,
                payload.get("chunk_options"),
                auto_decision=auto_decision,
            )
        logger.info(f"Successfully ingested {file_type} file with media_id: {media_id}")
        return media_id, media_uuid, message
    except Exception as e:
        logger.error(
            f"Error persisting {file_type} file {payload.get('file_path')}: {e}"
        )
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
    chunk_options: Optional[Dict[str, Any]] = _CHUNK_WITH_DEFAULTS,
    metadata: Optional[Dict[str, Any]] = None,
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
        chunk_options: Dictionary with chunking options. (task-3301,
            amended by the xhigh round-2 review / F7) OMITTING this
            argument means "chunk with defaults" (``{}``) -- the public
            wrappers' historical behavior. An EXPLICIT ``None`` disables
            chunking for EVERY type (the Library queue's Chunk-content
            OFF contract at the parse seam). Pass ``{}`` or a populated
            dict to chunk with defaults/overrides. Keys:
            - method: 'semantic', 'tokens', 'paragraphs', 'sentences', 'words', 'ebook_chapters'
            - size / max_size: chunk size (default varies by method)
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
    if chunk_options is _CHUNK_WITH_DEFAULTS:
        # (F7) Omitted argument == the wrappers' historical default
        # chunking; a fresh dict per call because processors setdefault
        # into it.
        chunk_options = {}
    options = {
        "title": title,
        "author": author,
        "keywords": keywords,
        "custom_prompt": custom_prompt,
        "system_prompt": system_prompt,
        "perform_analysis": perform_analysis,
        "api_name": api_name,
        "api_key": api_key,
        "chunk_options": chunk_options,
        "metadata": metadata,
    }
    payload = parse_local_file_for_ingest(str(file_path), options)
    media_id, _media_uuid, _message = persist_parsed_media(payload, media_db)

    chunks = payload["chunks"]
    return {
        "media_id": media_id,
        "title": payload["title"],
        "author": payload["author"],
        "content_length": len(payload["content"]),
        "chunks_created": len(chunks) if chunks else 0,
        "keywords": payload["keywords"],
        "analysis": payload["analysis_content"],
        "file_type": payload["file_type"],
        "file_path": payload["file_path"],
    }


def batch_ingest_files(
    file_paths: List[Union[str, Path]],
    media_db: MediaDatabase,
    common_keywords: Optional[List[str]] = None,
    perform_analysis: bool = False,
    api_name: Optional[str] = None,
    api_key: Optional[str] = None,
    chunk_options: Optional[Dict[str, Any]] = _CHUNK_WITH_DEFAULTS,
    stop_on_error: bool = False,
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
        chunk_options: Chunking options for all files. Omitted means
            "chunk with defaults"; an explicit ``None`` disables chunking
            (F7 -- see ``ingest_local_file``).
        stop_on_error: Whether to stop on first error or continue

    Returns:
        List of ingestion results (one per file)

    Raises:
        FileIngestionError: If stop_on_error is True and an error occurs
    """
    start_time = time.time()
    total_files = len(file_paths)
    log_counter(
        "local_file_ingestion_batch_start",
        labels={"total_files": str(total_files), "stop_on_error": str(stop_on_error)},
    )

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
                chunk_options=chunk_options,
            )
            results.append(result)
            success_count += 1

        except Exception as e:
            error_count += 1
            error_result = {
                "file_path": str(file_path),
                "error": str(e),
                "success": False,
            }
            results.append(error_result)

            if stop_on_error:
                log_counter("local_file_ingestion_batch_stopped_on_error")
                raise FileIngestionError(f"Batch ingestion stopped: {e}")
            else:
                logger.error(
                    f"Error ingesting {file_path}, continuing with next file: {e}"
                )

    # Log batch completion metrics
    duration = time.time() - start_time
    log_histogram(
        "local_file_ingestion_batch_duration",
        duration,
        labels={"total_files": str(total_files), "success_count": str(success_count)},
    )
    log_counter(
        "local_file_ingestion_batch_complete",
        labels={
            "total_files": str(total_files),
            "success_count": str(success_count),
            "error_count": str(error_count),
        },
    )

    return results


def ingest_directory(
    directory_path: Union[str, Path],
    media_db: MediaDatabase,
    recursive: bool = False,
    file_types: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    **kwargs,
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
        files = list(directory_path.rglob("*"))
    else:
        files = list(directory_path.glob("*"))

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
    return batch_ingest_files(file_paths=files, media_db=media_db, **kwargs)


# Convenience function for use in scripts
def quick_ingest(
    file_path: Union[str, Path], db_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Quick ingestion function for scripts and notebooks.

    Args:
        file_path: Path to file to ingest
        db_path: Optional path to media database (uses default if not provided)

    Returns:
        Ingestion result dictionary
    """
    from ..config import get_media_db_path

    if db_path is None:
        db_path = get_media_db_path()

    # Initialize database
    media_db = MediaDatabase(str(db_path), client_id="quick_ingest")

    try:
        result = ingest_local_file(file_path, media_db)
        return result
    finally:
        media_db.close_connection()
