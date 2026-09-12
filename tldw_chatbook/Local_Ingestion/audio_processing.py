# audio_processing.py
"""
Local audio processing module for tldw_chatbook.
Handles audio file processing, transcription, chunking, and analysis.
Adapted from server implementation for local use.
"""

#
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from urllib.parse import urlparse
from loguru import logger

#
# External imports
import requests

# Optional numpy import
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None
    logger.warning(
        "numpy not available. Some audio processing features will be limited."
    )
#
# Local imports
from ..config import get_media_ingestion_defaults, get_cli_setting
from ..Chat.Chat_Functions import chat_api_call
from ..DB.Client_Media_DB_v2 import MediaDatabase
from ..Utils.text import sanitize_filename

# NOTE: `ChunkingService` (RAG_Search.chunking_service) is intentionally NOT
# imported at module level -- it transitively pulls in nltk (via
# Chunking.Chunk_Lib) and should only load when a LocalAudioProcessor is
# actually constructed, not just from importing this module. See
# LocalAudioProcessor.__init__ for the deferred import.
#
# Optional imports
try:
    import yt_dlp

    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False
    logger.warning("yt-dlp not available. YouTube/URL downloading will be disabled.")


#
# Using loguru logger imported above
################################################################################################################################
class AudioProcessingError(Exception):
    """Base exception for audio processing errors."""

    pass


class AudioDownloadError(AudioProcessingError):
    """Raised when audio download fails."""

    pass


class AudioTranscriptionError(AudioProcessingError):
    """Raised when audio transcription fails."""

    pass


#: Multipliers for the ``[[HH:]MM:]SS`` timecode fields, least significant
#: first -- the order :func:`parse_media_timecode` walks them in.
_TIMECODE_UNIT_SECONDS = (1, 60, 3600)


def parse_media_timecode(value: Optional[str]) -> Optional[float]:
    """Parse an ingest time-range field into seconds.

    Accepts the two spellings the "Start at"/"Stop at" fields advertise:
    ``HH:MM:SS`` (or the shorter ``MM:SS``) and a bare number of seconds.
    Fractional seconds are preserved.

    Args:
        value: The field's raw value; ``None``/blank means "no bound".

    Returns:
        The value in seconds, or ``None`` when there is no bound or the
        text is not a timecode this function understands. Callers must
        treat ``None`` as "unknown", never as zero.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    parts = text.split(":")
    if len(parts) > len(_TIMECODE_UNIT_SECONDS):
        return None
    total = 0.0
    try:
        for index, part in enumerate(reversed(parts)):
            total += float(part) * _TIMECODE_UNIT_SECONDS[index]
    except ValueError:
        return None
    return total if total >= 0 else None


def _format_seconds(seconds: float) -> str:
    """Render a duration for ffmpeg's ``-t``, without trailing zero noise."""
    text = f"{seconds:.3f}".rstrip("0").rstrip(".")
    return text or "0"


def build_ffmpeg_trim_args(
    start_time: Optional[str], end_time: Optional[str]
) -> tuple[list[str], list[str]]:
    """Build the ffmpeg arguments for an ABSOLUTE ``[start, stop]`` window.

    (task-3306 xhigh review round) This is the single authority for what
    the ingest form's "Start at"/"Stop at" mean, because the two media
    paths used to disagree. ``_extract_audio_from_video`` emitted ``-ss``
    BEFORE ``-i`` -- input seeking, which rebases the output's timestamps
    to zero -- and then ``-to`` as an OUTPUT option, so "Stop at 1:00" with
    "Start at 0:30" selected 0:30-1:30 (twice the requested span, including
    content the user had excluded). ``_extract_time_range`` put ``-ss``
    after ``-i``, where ``-to`` is absolute, and selected 0:30-1:00. Same
    two fields, same job, two different windows.

    Both callers now share this builder, and "Stop at" is absolute
    everywhere -- which is what the label promises.

    The tradeoff, chosen correctness-first and then speed: absolute stop
    could have been bought by moving ``-ss`` after ``-i`` on the video path
    (output seeking, where ``-to`` is already absolute), but output seeking
    decodes and discards everything before the start -- trimming the last
    minute of a two-hour recording would decode 119 minutes first. So the
    fast pre-input seek is kept and the absolute stop is converted into the
    duration it implies (``-t``), which is exact under input seeking. The
    conversion needs both bounds to parse as timecodes; when either does
    not (or the window is empty/inverted), the builder falls back to output
    seeking, which is slower but keeps the same absolute meaning rather
    than silently reinterpreting the user's numbers.

    Args:
        start_time: "Start at" value (``HH:MM:SS`` or seconds); blank/None
            means "from the beginning".
        end_time: "Stop at" value, absolute in the source's own timeline;
            blank/None means "to the end".

    Returns:
        ``(pre_input_args, post_input_args)`` -- arguments to place before
        the ``-i <input>`` pair and after it, respectively. Both are empty
        when neither bound is set.
    """
    start = str(start_time or "").strip()
    end = str(end_time or "").strip()

    if not start and not end:
        return [], []
    if not start:
        # No start bound: "stop at X" is already a duration from zero.
        return [], ["-t", end]
    if not end:
        return ["-ss", start], []

    start_seconds = parse_media_timecode(start)
    end_seconds = parse_media_timecode(end)
    if (
        start_seconds is not None
        and end_seconds is not None
        and end_seconds > start_seconds
    ):
        return ["-ss", start], ["-t", _format_seconds(end_seconds - start_seconds)]

    # Unparseable or non-positive window: keep the absolute meaning by
    # seeking on the output side, where -to is not rebased.
    logger.warning(
        "Time-range trim could not be converted to a duration; falling back "
        "to slower output-side seeking."
    )
    return [], ["-ss", start, "-to", end]


class LocalAudioProcessor:
    """Handles local audio processing including download, transcription, and analysis."""

    def __init__(
        self,
        media_db: Optional[MediaDatabase] = None,
        *,
        transcription_runner: Optional[Callable[..., Dict[str, Any]]] = None,
    ):
        """
        Initialize the audio processor.

        Args:
            media_db: Optional MediaDatabase instance for storage
        """
        from ..RAG_Search.chunking_service import ChunkingService

        self.media_db = media_db
        self._transcription_runner = transcription_runner
        self.config = get_media_ingestion_defaults("audio")
        self.chunking_service = ChunkingService()
        self._cancelled = False  # Flag to track cancellation

        # Get configuration settings
        self.max_file_size_mb = get_cli_setting(
            "media_processing.max_audio_file_size_mb", 500
        )
        if self.max_file_size_mb is None:
            self.max_file_size_mb = 500
        self.max_file_size = self.max_file_size_mb * 1024 * 1024

    def cancel(self):
        """Cancel the current processing operation."""
        logger.info("Cancellation requested for audio processing")
        self._cancelled = True

        # Clean up transcription service if it exists
        if hasattr(self, "transcription_service") and self.transcription_service:
            if hasattr(self.transcription_service, "cleanup"):
                logger.info("Cleaning up transcription service resources")
                self.transcription_service.cleanup()

    def is_cancelled(self) -> bool:
        """Check if processing has been cancelled."""
        return self._cancelled

    def reset_cancellation(self):
        """Reset the cancellation flag."""
        self._cancelled = False

    def download_audio_file(
        self,
        url: str,
        target_dir: str,
        use_cookies: bool = False,
        cookies: Optional[Dict] = None,
    ) -> str:
        """
        Download an audio file from a URL.

        Args:
            url: URL to download from
            target_dir: Directory to save the file
            use_cookies: Whether to use cookies for download
            cookies: Cookie dict if use_cookies is True

        Returns:
            Path to downloaded file

        Raises:
            AudioDownloadError: If download fails
        """
        from ..Utils.egress import (
            EgressBlockedError,
            EgressFetchError,
            guarded_fetch_requests,
            origin_set,
        )

        tmp_path: Optional[Path] = None
        try:
            logger.info(f"Downloading audio from: {url}")

            headers = {}
            if use_cookies and cookies:
                if isinstance(cookies, str):
                    cookie_dict = json.loads(cookies)
                else:
                    cookie_dict = cookies
                headers["Cookie"] = "; ".join(
                    [f"{k}={v}" for k, v in cookie_dict.items()]
                )

            trusted = origin_set(url)
            # Fast-fail on declared size, then enforce the REAL streamed size.
            save_path = None
            try:
                probe_headers = dict(headers)
                # single guarded fetch; filename needs response headers, so fetch
                # to a temp .part file then rename
                tmp_path = Path(target_dir) / (uuid.uuid4().hex + ".part")
                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                with open(tmp_path, "wb") as f:
                    response = guarded_fetch_requests(
                        url,
                        max_bytes=self.max_file_size,
                        trusted_origins=trusted,
                        timeout=120,
                        headers=probe_headers,
                        sink=f,
                    )
                response.raise_for_status()
                declared = int(response.headers.get("content-length", 0))
                if declared > self.max_file_size:
                    raise AudioDownloadError(
                        f"File size ({declared / (1024 * 1024):.2f} MB) exceeds limit"
                    )
                filename = self._get_filename_from_response(response, url)
                save_path = Path(target_dir) / filename
                save_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path.replace(save_path)
            except (EgressBlockedError, EgressFetchError) as e:
                if tmp_path is not None and tmp_path.exists():
                    tmp_path.unlink()
                raise AudioDownloadError(f"Download blocked or too large: {e}") from e
            except Exception:
                if tmp_path is not None and tmp_path.exists():
                    tmp_path.unlink()
                raise

            logger.info(f"Downloaded audio file: {save_path}")
            return str(save_path)

        except requests.RequestException as e:
            raise AudioDownloadError(f"Download failed: {str(e)}") from e
        except AudioDownloadError:
            raise
        except Exception as e:
            raise AudioDownloadError(f"Unexpected error: {str(e)}") from e

    def download_youtube_audio(
        self,
        url: str,
        output_dir: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> Optional[str]:
        """
        Download audio from YouTube using yt-dlp.

        Args:
            url: YouTube URL
            output_dir: Directory to save audio

        Returns:
            Path to downloaded audio file or None if failed
        """
        if not YT_DLP_AVAILABLE:
            raise AudioDownloadError("yt-dlp is not installed")

        try:
            # Configure yt-dlp
            ydl_opts = {
                "format": "bestaudio[ext=m4a]/bestaudio/best",
                "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
                "quiet": True,
                "no_warnings": True,
                "extract_audio": True,
                "audio_format": "mp3",
                "audio_quality": 192,
            }

            # Add time range options if specified
            if start_time or end_time:
                # yt-dlp uses postprocessor args for time ranges
                postprocessor_args = []
                if start_time:
                    postprocessor_args.extend(["-ss", start_time])
                if end_time:
                    if start_time:
                        postprocessor_args.extend(["-to", end_time])
                    else:
                        postprocessor_args.extend(["-t", end_time])

                ydl_opts["postprocessor_args"] = {"ffmpeg": postprocessor_args}

            # Add ffmpeg location if specified in config
            ffmpeg_path = get_cli_setting("media_processing.ffmpeg_path")
            if ffmpeg_path:
                ydl_opts["ffmpeg_location"] = ffmpeg_path

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                # yt-dlp might change extension after conversion
                audio_path = filename.rsplit(".", 1)[0] + ".mp3"

                if os.path.exists(audio_path):
                    return audio_path
                elif os.path.exists(filename):
                    return filename
                else:
                    raise AudioDownloadError("Downloaded file not found")

        except Exception as e:
            logger.error(f"YouTube download error: {str(e)}")
            raise AudioDownloadError(f"YouTube download failed: {str(e)}") from e

    def process_audio_files(
        self,
        inputs: List[str],
        transcription_provider: str = "faster-whisper",
        transcription_model: str = "base",
        transcription_model_dir: Optional[str] = None,
        transcription_language: Optional[str] = "en",
        translation_target_language: Optional[str] = None,
        perform_chunking: bool = True,
        chunk_method: Optional[str] = None,
        max_chunk_size: int = 500,
        chunk_overlap: int = 200,
        use_adaptive_chunking: bool = False,
        use_multi_level_chunking: bool = False,
        chunk_language: Optional[str] = None,
        chunk_template: Optional[Dict[str, Any]] = None,
        diarize: bool = False,
        vad_use: bool = False,
        timestamp_option: bool = True,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        perform_analysis: bool = True,
        api_name: Optional[str] = None,
        api_key: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        summarize_recursively: bool = False,
        use_cookies: bool = False,
        cookies: Optional[str] = None,
        keep_original: bool = False,
        custom_title: Optional[str] = None,
        author: Optional[str] = None,
        temp_dir: Optional[str] = None,
        transcription_progress_callback: Optional[
            Callable[[float, str, Optional[Dict]], None]
        ] = None,
        transcription_precision: Optional[str] = None,
        transcription_local_files_only: bool = False,
        transcription_batch_route_resolved: bool = False,
        transcription_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process multiple audio inputs from URLs or local files.

        Args:
            inputs: Audio URLs or local file paths to process.
            transcription_provider: Exact STT provider or semantic default.
            transcription_model: Provider-specific model identifier.
            transcription_model_dir: Optional local model directory.
            transcription_language: Language code or ``auto``.
            translation_target_language: Optional translation target language.
            perform_chunking: Whether to chunk the resulting transcript.
            chunk_method: Optional transcript chunking strategy.
            max_chunk_size: Maximum chunk size for the selected strategy.
            chunk_overlap: Requested overlap between adjacent chunks.
            use_adaptive_chunking: Whether to enable adaptive chunk sizing.
            use_multi_level_chunking: Whether to emit multiple chunk levels.
            chunk_language: Optional language hint for chunking.
            chunk_template: Optional pre-resolved chunking-template dict
                (task 10, spec §9.2 -- the widened audio/video ingest seam;
                the video path forwards the same name through ``**kwargs``).
                Its chunk-stage options merge under the scalar chunking
                arguments at the shared chunk site.
            diarize: Whether to request speaker diarization.
            vad_use: Whether to request voice activity detection.
            timestamp_option: Whether to request transcript timestamps.
            start_time: Optional media start-time bound.
            end_time: Optional media end-time bound.
            perform_analysis: Whether to analyze the transcript after STT.
            api_name: Optional analysis provider identifier.
            api_key: Optional analysis provider credential.
            custom_prompt: Optional analysis user prompt.
            system_prompt: Optional analysis system prompt.
            summarize_recursively: Whether to recursively summarize chunks.
            use_cookies: Whether media download may use configured cookies.
            cookies: Optional cookies source for media download.
            keep_original: Whether to retain the normalized audio artifact.
            custom_title: Optional title override.
            author: Optional author override.
            temp_dir: Optional caller-owned processing directory.
            transcription_progress_callback: Optional STT progress callback.
            transcription_precision: Optional normalized precision choice.
            transcription_local_files_only: Whether network model access is
                forbidden for this route.
            transcription_batch_route_resolved: Whether Library routing already
                resolved provider/model semantics.
            transcription_context: Optional worker-private direct-local model
                path and retry-lineage values.

        Returns:
            A dictionary containing per-input processing results and errors.
        """
        results = []
        errors = []

        with tempfile.TemporaryDirectory(prefix="audio_proc_") as default_temp:
            processing_dir = temp_dir or default_temp

            for input_item in inputs:
                # Check for cancellation before processing each file
                if self.is_cancelled():
                    logger.info("Processing cancelled by user")
                    if transcription_progress_callback:
                        transcription_progress_callback(
                            0,
                            "Processing cancelled. Already processed files have been saved.",
                            {"cancelled": True},
                        )
                    break

                try:
                    result = self._process_single_audio(
                        input_item=input_item,
                        processing_dir=processing_dir,
                        transcription_provider=transcription_provider,
                        transcription_model=transcription_model,
                        transcription_model_dir=transcription_model_dir,
                        transcription_language=transcription_language,
                        translation_target_language=translation_target_language,
                        transcription_precision=transcription_precision,
                        transcription_local_files_only=transcription_local_files_only,
                        transcription_batch_route_resolved=transcription_batch_route_resolved,
                        transcription_context=transcription_context,
                        perform_chunking=perform_chunking,
                        chunk_method=chunk_method,
                        max_chunk_size=max_chunk_size,
                        chunk_overlap=chunk_overlap,
                        use_adaptive_chunking=use_adaptive_chunking,
                        use_multi_level_chunking=use_multi_level_chunking,
                        chunk_language=chunk_language,
                        chunk_template=chunk_template,
                        diarize=diarize,
                        vad_use=vad_use,
                        timestamp_option=timestamp_option,
                        start_time=start_time,
                        end_time=end_time,
                        perform_analysis=perform_analysis,
                        api_name=api_name,
                        api_key=api_key,
                        custom_prompt=custom_prompt,
                        system_prompt=system_prompt,
                        summarize_recursively=summarize_recursively,
                        use_cookies=use_cookies,
                        cookies=cookies,
                        custom_title=custom_title,
                        author=author,
                        transcription_progress_callback=transcription_progress_callback,
                    )
                    results.append(result)

                except Exception as e:
                    logger.error(f"Error processing {input_item}: {str(e)}")
                    error_result = {
                        "status": "Error",
                        "input_ref": input_item,
                        "error": str(e),
                        "media_type": "audio",
                    }
                    results.append(error_result)
                    errors.append(str(e))

        # Calculate summary statistics
        processed_count = sum(1 for r in results if r.get("status") == "Success")
        errors_count = sum(1 for r in results if r.get("status") == "Error")

        return {
            "processed_count": processed_count,
            "errors_count": errors_count,
            "errors": errors,
            "results": results,
        }

    def _process_single_audio(
        self,
        input_item: str,
        processing_dir: str,
        transcription_progress_callback=None,
        media_type: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Process a single audio file or URL."""

        # Check if this is being called from video processing with an original URL
        original_url = kwargs.pop("original_url", None)

        result = {
            "status": "Pending",
            "input_ref": original_url or input_item,  # Use original URL if provided
            "processing_source": input_item,
            "media_type": media_type or "audio",
            "metadata": {
                "title": kwargs.get("custom_title"),
                "author": kwargs.get("author"),
            },
            "content": None,
            "segments": None,
            "chunks": None,
            "analysis": None,
            "analysis_details": {},
            "error": None,
            "warnings": [],
        }

        try:
            # Determine if input is URL or local file
            is_url = urlparse(input_item).scheme in ("http", "https")

            if is_url:
                # Download the file
                if "youtube.com" in input_item or "youtu.be" in input_item:
                    audio_path = self.download_youtube_audio(
                        input_item,
                        processing_dir,
                        kwargs.get("start_time"),
                        kwargs.get("end_time"),
                    )
                else:
                    audio_path = self.download_audio_file(
                        input_item,
                        processing_dir,
                        kwargs.get("use_cookies", False),
                        kwargs.get("cookies"),
                    )
                # Don't overwrite processing_source for URLs - keep the original URL
            else:
                # Local file
                if not os.path.exists(input_item):
                    raise FileNotFoundError(f"File not found: {input_item}")
                audio_path = input_item

            # Extract time range if specified
            # Note: For YouTube URLs, we should skip this as yt-dlp already handles it
            start_time = kwargs.get("start_time")
            end_time = kwargs.get("end_time")
            youtube_url = is_url and (
                "youtube.com" in input_item or "youtu.be" in input_item
            )

            if (start_time or end_time) and not youtube_url:
                logger.info(
                    f"Extracting time range: start={start_time}, end={end_time}"
                )
                audio_path = self._extract_time_range(
                    audio_path, processing_dir, start_time, end_time
                )

            # Update metadata
            if not result["metadata"]["title"]:
                result["metadata"]["title"] = Path(audio_path).stem

            # Transcribe audio
            provider = kwargs.get("transcription_provider", None)
            model = kwargs.get("transcription_model", None)
            language = kwargs.get("transcription_language", None)

            logger.info(
                f"[AUDIO] Starting transcription: provider={provider}, model={model}, language={language}"
            )
            logger.info(f"[AUDIO] Transcription audio file: {audio_path}")
            logger.info(
                f"[AUDIO] Audio file size: {Path(audio_path).stat().st_size / 1024 / 1024:.2f} MB"
            )
            logger.info(f"[AUDIO] Audio file exists: {os.path.exists(audio_path)}")

            transcription_start = time.time()
            try:
                logger.info("[AUDIO] Calling _transcribe_audio()")
                context = kwargs.get("transcription_context") or {}
                provenance_kwargs = (
                    {
                        "model_path": context.get("model_path"),
                        "attempt_id": context.get("attempt_id"),
                        "batch_id": context.get("batch_id"),
                        "job_id": context.get("job_id"),
                        "retry_of_attempt_id": context.get("retry_of_attempt_id"),
                        "retry_of_job_id": context.get("retry_of_job_id"),
                        "retry_source_failure_provenance": context.get(
                            "retry_source_failure_provenance"
                        ),
                        "timestamps": kwargs.get("timestamp_option", True),
                    }
                    if provider
                    in {"faster-whisper", "parakeet-onnx", "transcribe-cpp"}
                    else {}
                )
                transcription_result = self._transcribe_audio(
                    audio_path,
                    provider=provider,
                    model=model,
                    language=language,
                    model_dir=kwargs.get("transcription_model_dir"),
                    target_lang=kwargs.get("translation_target_language"),
                    compute_type=kwargs.get("transcription_precision"),
                    local_files_only=kwargs.get(
                        "transcription_local_files_only", False
                    ),
                    batch_route_resolved=kwargs.get(
                        "transcription_batch_route_resolved", False
                    ),
                    vad_filter=kwargs.get("vad_use", False),
                    diarize=kwargs.get("diarize", False),
                    progress_callback=transcription_progress_callback,
                    **provenance_kwargs,
                )
                logger.info("[AUDIO] _transcribe_audio() returned successfully")
            except Exception as e:
                error_detail = getattr(e, "error_detail", None)
                if isinstance(error_detail, dict):
                    logger.error(
                        "[AUDIO] Direct-local transcription failed: code={}",
                        error_detail.get("code", "inference_failed"),
                    )
                else:
                    logger.opt(exception=True).error(
                        f"[AUDIO] Transcription failed: {type(e).__name__}: {str(e)}"
                    )
                raise

            transcription_time = time.time() - transcription_start
            logger.info(
                f"[AUDIO] Transcription completed in {transcription_time:.2f} seconds"
            )

            # Log detailed transcription results
            if transcription_result:
                logger.info(
                    f"[AUDIO] Transcription result keys: {list(transcription_result.keys())}"
                )
                logger.info(
                    f"[AUDIO] Transcription text length: {len(transcription_result.get('text', ''))} characters"
                )
                logger.info(
                    f"[AUDIO] Number of segments: {len(transcription_result.get('segments', []))}"
                )

                if not transcription_result.get("text"):
                    logger.warning("[AUDIO] Transcription returned empty text!")
                else:
                    logger.info(
                        f"[AUDIO] First 100 chars of transcription: {transcription_result['text'][:100]}..."
                    )
            else:
                logger.error("[AUDIO] Transcription result is None!")

            result["segments"] = transcription_result.get("segments", [])
            result["content"] = transcription_result.get("text", "")
            result["transcription_model"] = transcription_result.get(
                "transcription_model"
            )
            result["transcription_provenance"] = transcription_result.get(
                "transcription_provenance"
            )

            logger.info(
                f"[AUDIO] Final result content length: {len(result['content'])} chars, segments: {len(result['segments'])}"
            )

            # Perform chunking if requested
            if kwargs.get("perform_chunking") and result["content"]:
                chunk_method = kwargs.get("chunk_method", "sentences")
                logger.info(
                    f"Starting text chunking: method={chunk_method}, max_size={kwargs.get('max_chunk_size', 500)}"
                )

                chunks = self._chunk_text(
                    result["content"],
                    method=chunk_method,
                    max_size=kwargs.get("max_chunk_size", 500),
                    overlap=kwargs.get("chunk_overlap", 200),
                    language=kwargs.get("chunk_language")
                    or kwargs.get("transcription_language", "en"),
                    # (task 10, spec §9.2) the widened audio/video seam: the
                    # pre-resolved template rides through to the chunking
                    # service (video's ``**kwargs`` path lands here too).
                    template=kwargs.get("chunk_template"),
                )
                result["chunks"] = chunks
                logger.debug(f"Chunking completed: {len(chunks)} chunks created")

            # Perform analysis if requested
            if (
                kwargs.get("perform_analysis")
                and kwargs.get("api_name")
                and result["content"]
            ):
                analysis = self._analyze_content(
                    content=result["content"],
                    chunks=result.get("chunks"),
                    api_name=kwargs["api_name"],
                    api_key=kwargs.get("api_key"),
                    custom_prompt=kwargs.get("custom_prompt"),
                    system_prompt=kwargs.get("system_prompt"),
                    summarize_recursively=kwargs.get("summarize_recursively", False),
                )
                result["analysis"] = analysis
                result["analysis_details"] = {
                    "api_name": kwargs["api_name"],
                    "custom_prompt": kwargs.get("custom_prompt"),
                    "recursive": kwargs.get("summarize_recursively", False),
                }

            # Store in database if available
            if self.media_db and result["content"]:
                db_result = self._store_in_database(result)
                result["db_id"] = db_result.get("id")
                result["db_message"] = db_result.get("message", "Stored successfully")

            result["status"] = "Success" if not result["warnings"] else "Warning"

            # Handle keep_original option - move audio file to Downloads folder
            if kwargs.get("keep_original", False) and is_url:
                try:
                    # Get user's Downloads folder
                    downloads_dir = Path.home() / "Downloads"
                    downloads_dir.mkdir(exist_ok=True)

                    # Generate a unique filename if needed
                    audio_filename = Path(audio_path).name
                    dest_path = downloads_dir / audio_filename

                    # Handle filename conflicts
                    if dest_path.exists():
                        base_name = dest_path.stem
                        extension = dest_path.suffix
                        counter = 1
                        while dest_path.exists():
                            dest_path = (
                                downloads_dir / f"{base_name}_{counter}{extension}"
                            )
                            counter += 1

                    # Move the file
                    shutil.move(audio_path, str(dest_path))
                    logger.info(f"Moved audio file to: {dest_path}")
                    result["saved_audio_path"] = str(dest_path)
                    result["warnings"].append(
                        f"Audio file saved to Downloads folder: {dest_path.name}"
                    )
                except Exception as e:
                    logger.error(f"Failed to move audio file to Downloads: {str(e)}")
                    result["warnings"].append(f"Could not save audio file: {str(e)}")

        except Exception as e:
            error_detail = getattr(e, "error_detail", None)
            if isinstance(error_detail, dict):
                logger.error(
                    "[AUDIO] Direct-local processing failed: code={}",
                    error_detail.get("code", "inference_failed"),
                )
            else:
                logger.opt(exception=True).error(f"Error processing audio: {str(e)}")
            result["status"] = "Error"
            result["error"] = str(e)
            if isinstance(error_detail, dict):
                result["error_detail"] = error_detail
            failed_attempt = getattr(e, "stt_failure_provenance", None)
            if isinstance(failed_attempt, dict):
                result["stt_failure_provenance"] = failed_attempt

        return result

    def _transcribe_audio(
        self, audio_path: str, progress_callback=None, **kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe audio using available transcription service.

        Args:
            audio_path: Path to audio file
            progress_callback: Optional callback for progress updates
            **kwargs: Additional transcription parameters
        """
        logger.info(f"[AUDIO] _transcribe_audio called with audio_path: {audio_path}")
        logger.info(
            f"[AUDIO] Transcription kwargs: provider={kwargs.get('provider')}, model={kwargs.get('model')}, language={kwargs.get('language')}"
        )

        def cancellable_progress_callback(progress, message, data=None):
            if self.is_cancelled():
                logger.info("[AUDIO] Transcription cancelled by user")
                raise AudioTranscriptionError("Transcription cancelled by user")
            if progress_callback:
                logger.debug(f"[AUDIO] Progress update: {progress}% - {message}")
                progress_callback(progress, message, data)

        if self._transcription_runner is not None:
            return self._transcription_runner(
                audio_path,
                progress_callback=cancellable_progress_callback,
                **kwargs,
            )

        if kwargs.get("provider") == "transcribe-cpp":
            from tldw_chatbook.STT.persistence import (
                build_transcription_provenance_document,
            )
            from tldw_chatbook.STT.transcribe_cpp import transcribe_file

            model_path = kwargs.get("model_path")
            attempt_id = kwargs.get("attempt_id")
            if not isinstance(attempt_id, str) or not attempt_id:
                attempt_id = f"direct-local-{uuid.uuid4().hex}"
            normalized = transcribe_file(
                audio_path=Path(audio_path),
                model_path=Path(model_path) if model_path else None,
                attempt_id=attempt_id,
                batch_id=kwargs.get("batch_id"),
                job_id=kwargs.get("job_id"),
                retry_of_attempt_id=kwargs.get("retry_of_attempt_id"),
                retry_of_job_id=kwargs.get("retry_of_job_id"),
                language=kwargs.get("language") or "en",
                timestamps=bool(kwargs.get("timestamps", True)),
                ffmpeg_path=get_cli_setting("media_processing.ffmpeg_path"),
            )
            provenance = build_transcription_provenance_document(
                normalized,
                failed_attempt=kwargs.get("retry_source_failure_provenance"),
            )
            return {
                "text": normalized.text,
                "segments": [
                    {
                        "start": segment.start_seconds,
                        "end": segment.end_seconds,
                        "text": segment.text,
                    }
                    for segment in normalized.segments
                ],
                "transcription_model": normalized.provenance.model_id,
                "transcription_provenance": provenance,
            }

        # Import transcription service when available
        try:
            logger.info("[AUDIO] Importing TranscriptionService")
            from .transcription_service import TranscriptionService

            service = TranscriptionService()
            logger.info("[AUDIO] TranscriptionService imported successfully")

            logger.info("[AUDIO] Calling TranscriptionService.transcribe()")
            result = service.transcribe(
                audio_path, progress_callback=cancellable_progress_callback, **kwargs
            )
            logger.info("[AUDIO] TranscriptionService.transcribe() completed")

            if result:
                logger.info(
                    f"[AUDIO] Transcription service returned: text_length={len(result.get('text', ''))}, segments={len(result.get('segments', []))}"
                )
            else:
                logger.error("[AUDIO] Transcription service returned None")

            if (
                result
                and kwargs.get("provider") == "faster-whisper"
                and isinstance(kwargs.get("attempt_id"), str)
            ):
                from tldw_chatbook.STT.contracts import (
                    ExecutionDevice,
                    ProducedCapabilities,
                    TimestampGranularity,
                    TranscriptionProvenance,
                    TranscriptionResult,
                    TranscriptionSegment,
                    TranscriptionTask,
                    TranscriptionTimings,
                )
                from tldw_chatbook.STT.persistence import (
                    build_transcription_provenance_document,
                )

                requested_language = str(kwargs.get("language") or "en").lower()
                observed_language = str(result.get("language") or "").lower() or None
                detected_language = (
                    observed_language if requested_language == "auto" else None
                )
                effective_language = (
                    detected_language or "auto"
                    if requested_language == "auto"
                    else requested_language
                )
                configured_device = str(service.config.get("device") or "auto").lower()
                try:
                    effective_device = ExecutionDevice(configured_device)
                except ValueError:
                    effective_device = ExecutionDevice.AUTO
                is_translation = result.get("task") == "translation" or (
                    str(kwargs.get("target_lang") or "").lower() == "en"
                    and requested_language != "en"
                )
                timestamps_requested = bool(kwargs.get("timestamps", True))
                normalized_segments = (
                    tuple(
                        TranscriptionSegment(
                            float(segment.get("start") or 0.0),
                            float(segment.get("end") or 0.0),
                            str(segment.get("text") or ""),
                            speaker=segment.get("speaker"),
                        )
                        for segment in result.get("segments") or []
                    )
                    if timestamps_requested
                    else ()
                )
                normalized = TranscriptionResult(
                    text=str(result.get("text") or ""),
                    segments=normalized_segments,
                    provenance=TranscriptionProvenance(
                        schema_version=1,
                        attempt_id=kwargs["attempt_id"],
                        batch_id=kwargs.get("batch_id"),
                        job_id=kwargs.get("job_id"),
                        retry_of_attempt_id=kwargs.get("retry_of_attempt_id"),
                        retry_of_job_id=kwargs.get("retry_of_job_id"),
                        provider_id="faster-whisper",
                        model_id=str(result.get("model") or kwargs.get("model") or "base"),
                        artifact_root=None,
                        artifact_dependencies=(),
                        precision=str(
                            kwargs.get("compute_type")
                            or service.config.get("compute_type")
                            or "int8"
                        ),
                        requested_device=effective_device,
                        effective_device=effective_device,
                        requested_language=requested_language,
                        effective_language=effective_language,
                        detected_language=detected_language,
                        task=(
                            TranscriptionTask.TRANSLATE
                            if is_translation
                            else TranscriptionTask.TRANSCRIBE
                        ),
                    ),
                    produced_capabilities=ProducedCapabilities(
                        timestamps=(
                            TimestampGranularity.SEGMENT
                            if normalized_segments
                            else TimestampGranularity.NONE
                        ),
                        punctuation=True,
                        capitalization=True,
                        vad=bool(kwargs.get("vad_filter", False)),
                        diarization=bool(result.get("diarization_performed", False)),
                    ),
                    duration_seconds=float(result.get("duration") or 0.0),
                    timings=TranscriptionTimings(),
                )
                result["transcription_model"] = normalized.provenance.model_id
                result["transcription_provenance"] = (
                    build_transcription_provenance_document(
                        normalized,
                        failed_attempt=kwargs.get("retry_source_failure_provenance"),
                    )
                )

            return result
        except ImportError as e:
            # Fallback for testing
            logger.error(f"[AUDIO] Failed to import TranscriptionService: {str(e)}")
            logger.warning(
                "[AUDIO] Transcription service not available, using placeholder"
            )
            return {
                "text": f"[Placeholder transcription for {Path(audio_path).name}]",
                "segments": [
                    {
                        "start": 0.0,
                        "end": 5.0,
                        "text": "This is a placeholder transcription.",
                    }
                ],
            }

    def _chunk_text(
        self,
        text: str,
        method: str = "sentences",
        max_size: int = 500,
        overlap: int = 200,
        language: str = "en",
        template: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Chunk text using the chunking service."""
        # ChunkingService.chunk_text takes flat keyword arguments
        # (content, chunk_size, chunk_overlap, method) -- NOT an options dict.
        # Passing one positionally put a dict where chunk_size is expected and
        # every audio/video ingest died in chunking with
        # "'<=' not supported between instances of 'dict' and 'int'" (task-840).
        # (task 10) The ``template`` kwarg is forwarded ONLY when set: the
        # no-template call stays byte-identical to today's four-kwarg shape,
        # so any duck-typed chunking service predating the kwarg (and the
        # task-840 characterization pin) keeps working when no template is
        # in play.
        chunk_kwargs: Dict[str, Any] = {
            "chunk_size": max_size,
            "chunk_overlap": overlap,
            "method": method,
        }
        if template is not None:
            chunk_kwargs["template"] = template
        chunks = self.chunking_service.chunk_text(text, **chunk_kwargs)
        # It returns dicts carrying 'text' plus real character offsets, not bare
        # strings; the previous wrapping nested the whole dict under another
        # "text" key. Carry the offsets through rather than dropping them: the
        # storage path otherwise re-derives them by summing chunk lengths, which
        # double-counts whenever chunks overlap and drifts whenever chunk text is
        # trimmed -- and overlap is on by default here.
        normalised = []
        for index, chunk in enumerate(chunks):
            if isinstance(chunk, dict):
                text = chunk.get("text", "")
                start_char = chunk.get("start_char")
                end_char = chunk.get("end_char")
                chunk_index = chunk.get("chunk_index", index)
            else:
                text, start_char, end_char, chunk_index = str(chunk), None, None, index
            entry = {
                "text": text,
                "metadata": {"method": method, "language": language},
                "chunk_index": chunk_index,
            }
            if start_char is not None:
                entry["start_char"] = start_char
            if end_char is not None:
                entry["end_char"] = end_char
            normalised.append(entry)
        return normalised

    def _analyze_content(
        self,
        content: str,
        chunks: Optional[List[Dict]],
        api_name: str,
        api_key: Optional[str],
        custom_prompt: Optional[str],
        system_prompt: Optional[str],
        summarize_recursively: bool = False,
    ) -> str:
        """Analyze/summarize content using LLM."""

        # Prepare prompt
        if custom_prompt:
            custom_prompt + "\n\n" + content
        else:
            pass

        # If chunking and recursive summarization
        if chunks and summarize_recursively and len(chunks) > 1:
            # Summarize each chunk first
            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                chunk_text = chunk.get("text", "")
                if chunk_text:
                    chunk_prompt = f"Summarize this section:\n\n{chunk_text}"
                    messages_payload = [{"role": "user", "content": chunk_prompt}]
                    summary = chat_api_call(
                        api_endpoint=api_name,
                        messages_payload=messages_payload,
                        api_key=api_key,
                        temp=0.7,
                        system_message=system_prompt,
                    )
                    chunk_summaries.append(summary)

            # Then summarize the summaries
            combined = "\n\n".join(chunk_summaries)
            final_prompt = (
                f"Combine and summarize these section summaries:\n\n{combined}"
            )
            messages_payload = [{"role": "user", "content": final_prompt}]
            return chat_api_call(
                api_endpoint=api_name,
                messages_payload=messages_payload,
                api_key=api_key,
                temp=0.7,
                system_message=system_prompt,
            )
        else:
            # Direct summarization
            prompt_text = custom_prompt or "Please summarize this audio transcription."
            messages_payload = [
                {"role": "user", "content": content + "\n\n" + prompt_text}
            ]
            return chat_api_call(
                api_endpoint=api_name,
                messages_payload=messages_payload,
                api_key=api_key,
                temp=0.7,
                system_message=system_prompt,
            )

    def _store_in_database(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Store processing results in the media database."""
        if not self.media_db:
            return {"message": "No database available"}

        try:
            # Prepare media data - store transcription in content field
            media_data = {
                "url": result.get("input_ref", ""),
                "title": result["metadata"].get("title", "Untitled"),
                "media_type": result.get("media_type", "audio"),
                "content": result.get("content", ""),  # Store transcription
                "author": result["metadata"].get("author", "Unknown"),
                "ingestion_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "analysis_content": result.get("analysis"),  # Store analysis separately
            }

            # Add media entry with analysis
            media_id, _, _ = self.media_db.add_media_with_keywords(**media_data)

            # Store chunks if available
            if result.get("chunks"):
                # Prepare chunks in the format expected by add_media_chunks_in_batches
                chunks_to_add = []
                for i, chunk in enumerate(result["chunks"]):
                    chunk_text = chunk.get("text", "")
                    # Prefer the chunker's real character offsets. Summing prior
                    # chunk lengths only happens to be right when chunks neither
                    # overlap nor get trimmed; with overlap on it double-counts.
                    start_index = chunk.get("start_char")
                    end_index = chunk.get("end_char")
                    if start_index is None or end_index is None:
                        start_index = sum(
                            len(c.get("text", "")) for c in result["chunks"][:i]
                        )
                        end_index = start_index + len(chunk_text)

                    chunks_to_add.append(
                        {
                            "text": chunk_text,
                            "start_index": start_index,
                            "end_index": end_index,
                        }
                    )

                # Use batch insert method
                self.media_db.add_media_chunks_in_batches(
                    media_id=media_id, chunks_to_add=chunks_to_add
                )

            return {"id": media_id, "message": "Stored successfully"}

        except Exception as e:
            logger.error(f"Database storage error: {str(e)}")
            return {"message": f"Storage failed: {str(e)}"}

    def _get_filename_from_response(self, response: requests.Response, url: str) -> str:
        """Extract filename from response headers or URL."""
        content_disposition = response.headers.get("content-disposition")
        if content_disposition:
            parts = content_disposition.split("filename=")
            if len(parts) > 1:
                filename = parts[1].strip("\"' ")
                if filename:
                    return sanitize_filename(filename)

        # Fallback to URL path
        path = Path(urlparse(url).path)
        if path.name:
            return sanitize_filename(path.name)

        # Generate unique filename
        return f"audio_{uuid.uuid4().hex[:8]}.mp3"

    def _extract_time_range(
        self,
        audio_path: str,
        output_dir: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> str:
        """
        Extract a time range from an audio file using ffmpeg.

        Args:
            audio_path: Path to input audio file
            output_dir: Directory to save the extracted audio
            start_time: Start time in format HH:MM:SS or seconds
            end_time: End time in format HH:MM:SS or seconds

        Returns:
            Path to the extracted audio file
        """
        # Find ffmpeg
        import shutil

        ffmpeg_cmd = shutil.which("ffmpeg")
        if not ffmpeg_cmd:
            # Try common locations
            for cmd in [
                "/usr/bin/ffmpeg",
                "/usr/local/bin/ffmpeg",
                "/opt/homebrew/bin/ffmpeg",
            ]:
                if os.path.exists(cmd):
                    ffmpeg_cmd = cmd
                    break

        if not ffmpeg_cmd:
            logger.warning("ffmpeg not found, skipping time range extraction")
            return audio_path

        # Generate output filename
        base_name = Path(audio_path).stem
        suffix = f"_trim_{start_time or '0'}_{end_time or 'end'}".replace(":", "-")
        output_path = os.path.join(output_dir, f"{base_name}{suffix}.mp3")

        # Build ffmpeg command. The trim arguments come from the shared
        # builder so this path and the video path cannot mean different
        # windows for the same Start/Stop pair (task-3306 review round).
        pre_input, post_input = build_ffmpeg_trim_args(start_time, end_time)
        command = [ffmpeg_cmd, *pre_input, "-i", audio_path, *post_input]

        # Output options
        command.extend(
            [
                "-acodec",
                "libmp3lame",
                "-ab",
                "192k",
                "-ar",
                "44100",
                "-y",  # Overwrite
                output_path,
            ]
        )

        try:
            logger.debug(f"Running ffmpeg command: {' '.join(command)}")
            subprocess.run(command, capture_output=True, text=True, check=True)
            logger.info(f"Extracted time range to: {output_path}")
            return output_path

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to extract time range: {e.stderr}")
            logger.warning("Continuing with full audio file")
            return audio_path


# Convenience function for backwards compatibility
def process_audio_files(**kwargs) -> Dict[str, Any]:
    """Process audio files using LocalAudioProcessor."""
    processor = LocalAudioProcessor()
    return processor.process_audio_files(**kwargs)


#
# End of audio_processing.py
#########################################################################################################################
