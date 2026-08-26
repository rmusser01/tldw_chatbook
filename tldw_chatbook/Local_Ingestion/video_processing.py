# video_processing.py
"""
Local video processing module for tldw_chatbook.
Handles video file processing by extracting audio and leveraging audio processing.
"""

import json
import os
import shutil
import tempfile
import logging
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from urllib.parse import urlparse
import subprocess
from loguru import logger

# Local imports
from .audio_processing import LocalAudioProcessor, build_ffmpeg_trim_args
from ..config import get_cli_setting
from ..DB.Client_Media_DB_v2 import MediaDatabase
from ..Metrics.metrics_logger import log_counter, log_histogram
from ..Utils.egress import EgressBlockedError, check_url_or_raise, origin_set

# Optional imports
try:
    import yt_dlp

    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False
    logging.warning("yt-dlp not available. Video downloading will be disabled.")

# Using loguru logger imported above


class VideoProcessingError(Exception):
    """Base exception for video processing errors."""

    pass


class VideoDownloadError(VideoProcessingError):
    """Raised when video download fails."""

    pass


def check_media_url_egress(url: str) -> None:
    """Apply the app's egress policy to a media URL before yt-dlp sees it.

    The media arm of ingest never consulted ``Utils/egress.py`` (TASK-19556
    (b)), while the article arm of the same entry point did, and
    ``audio_processing.download_audio_file`` guards its own plain-HTTP branch
    with ``guarded_fetch_requests(..., trusted_origins=origin_set(url))``.
    This is that same check, applied at the two yt-dlp seams, so both arms
    behave identically.

    The URL is its own trusted origin, exactly as in the audio arm: a media
    URL the user typed into the ingest form is an explicitly configured URL,
    which ``config.py``'s ``[web_security]`` contract permits to be private
    (an intranet media server is a legitimate source). What the check still
    refuses is what no configured URL may be: a cloud metadata endpoint
    (blocked regardless of trust) and anything that is not http(s) -- yt-dlp
    itself is happy to hand ``file://`` and dozens of other protocols to its
    extractors.

    WHAT THIS DOES NOT COVER, stated plainly: yt-dlp performs its own HTTP
    fetching. This is a pre-check on the entry URL only. It cannot
    re-validate yt-dlp's own redirect hops, the per-format media URLs an
    extractor discovers inside a page, or a DNS answer that changes between
    this call and yt-dlp's own resolution -- the TOCTOU window
    ``Utils/egress.py`` documents as a residual for every consumer of a
    resolve-then-connect check. Closing those would need a yt-dlp request
    hook and is not in this task's scope.

    Args:
        url: The media URL about to be handed to yt-dlp.

    Raises:
        VideoDownloadError: If the egress policy refuses the URL.
    """
    try:
        check_url_or_raise(url, trusted_origins=origin_set(url))
    except EgressBlockedError as exc:
        log_counter(
            "video_processing_download_error",
            labels={"error_type": "egress_blocked"},
        )
        raise VideoDownloadError(f"URL blocked by the egress policy: {exc}") from exc


class LocalVideoProcessor:
    """Handles local video processing including download, audio extraction, and analysis."""

    def __init__(
        self,
        media_db: Optional[MediaDatabase] = None,
        *,
        transcription_runner: Optional[Callable[..., Dict[str, Any]]] = None,
    ):
        """
        Initialize the video processor.

        Args:
            media_db: Optional MediaDatabase instance for storage
        """
        self.media_db = media_db
        self.audio_processor = LocalAudioProcessor(
            media_db,
            transcription_runner=transcription_runner,
        )
        self._cancelled = False  # Flag to track cancellation
        self.max_file_size_mb = get_cli_setting(
            "media_processing.max_video_file_size_mb", 2000
        )
        if self.max_file_size_mb is None:
            self.max_file_size_mb = 2000
        self.max_file_size = self.max_file_size_mb * 1024 * 1024

    def cancel(self):
        """Cancel the current processing operation."""
        logger.info("Cancellation requested for video processing")
        self._cancelled = True
        # Also cancel audio processor
        if self.audio_processor:
            self.audio_processor.cancel()

    def is_cancelled(self) -> bool:
        """Check if processing has been cancelled."""
        return self._cancelled

    def reset_cancellation(self):
        """Reset the cancellation flag."""
        self._cancelled = False
        # Also reset audio processor cancellation
        if self.audio_processor:
            self.audio_processor.reset_cancellation()

    @staticmethod
    def _write_temp_cookiefile(cookie_dict: Dict[str, Any]) -> str:
        """Write a ``name=value`` cookie mapping to a new temp file.

        Returns:
            The temp file's path. The caller OWNS this file and is
            responsible for removing it.
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for name, value in cookie_dict.items():
                f.write(f"{name}={value}\n")
            return f.name

    def _resolve_cookiefile(
        self, use_cookies: bool, cookies: Optional[Any]
    ) -> tuple[Optional[str], Optional[str]]:
        """Resolve yt-dlp's ``cookiefile``, tracking whether WE created it.

        (task-3306 xhigh review round) Ownership is returned explicitly
        instead of being inferred later from the path. ``download_video``
        used to clean up any cookiefile whose path started with
        ``tempfile.gettempdir()`` -- a heuristic that was only ever safe
        while the key could hold nothing but a temp file this class had
        written. Once the ingest form began routing a user's own cookies
        path into the same argument, a user who exported cookies to
        ``/tmp/cookies.txt`` lost the file on the first import, and the
        unlink failure was swallowed into a debug log.

        Args:
            use_cookies: Whether cookies were requested at all.
            cookies: Either a path to a cookies file, a JSON string of
                ``{name: value}`` pairs, or that mapping directly.

        Returns:
            ``(cookiefile, owned_temp_path)``. ``owned_temp_path`` is
            non-``None`` only for a file this call created, and is the ONLY
            thing a caller may delete.

        Raises:
            VideoDownloadError: When cookies were requested but the value is
                neither an existing file nor a cookie mapping. Cookies exist
                to get past an authentication gate; continuing without them
                only produces a later, unrelated-looking failure.
        """
        if not use_cookies or not cookies:
            return None, None

        if isinstance(cookies, str):
            candidate = os.path.expanduser(cookies.strip())
            if os.path.isfile(candidate):
                return candidate, None
            try:
                cookie_dict = json.loads(cookies)
            except json.JSONDecodeError as exc:
                raise VideoDownloadError(
                    f"Cookies file not found: {cookies}"
                ) from exc
            if not isinstance(cookie_dict, dict):
                raise VideoDownloadError(
                    f"Cookies file not found: {cookies}"
                )
            temp_path = self._write_temp_cookiefile(cookie_dict)
            return temp_path, temp_path

        if isinstance(cookies, dict):
            temp_path = self._write_temp_cookiefile(cookies)
            return temp_path, temp_path

        raise VideoDownloadError(
            f"Unsupported cookies value of type {type(cookies).__name__}"
        )

    @staticmethod
    def _discard_temp_cookiefile(path: Optional[str]) -> None:
        """Remove a temp cookiefile this class created, if any."""
        if not path:
            return
        try:
            os.unlink(path)
        except OSError as e:
            logger.debug(f"Failed to clean up temporary cookie file: {e}")

    def download_video(
        self,
        url: str,
        output_dir: str,
        download_video_flag: bool = False,
        use_cookies: bool = False,
        cookies: Optional[Dict] = None,
    ) -> Optional[str]:
        """
        Download video or just audio from URL using yt-dlp.

        Args:
            url: Video URL
            output_dir: Directory to save the file
            download_video_flag: If True, download full video; if False, extract audio only
            use_cookies: Whether to use cookies for download
            cookies: Cookie dict if use_cookies is True

        Returns:
            Path to downloaded file or None if failed
        """
        start_time = time.time()
        log_counter(
            "video_processing_download_attempt",
            labels={
                "download_type": "full_video" if download_video_flag else "audio_only",
                "use_cookies": str(use_cookies),
            },
        )

        if not YT_DLP_AVAILABLE:
            log_counter(
                "video_processing_download_error",
                labels={"error_type": "yt_dlp_not_available"},
            )
            raise VideoDownloadError("yt-dlp is not installed")

        # (TASK-19556) Before ANY yt-dlp work -- the probe below is itself a
        # fetch. Outside the try/except so a policy refusal keeps its own
        # reason instead of being rewrapped as "Download failed".
        check_media_url_egress(url)

        # Resolved before the try so an unusable cookies value fails with
        # its own reason instead of being wrapped as a download failure.
        cookiefile, owned_temp_cookiefile = self._resolve_cookiefile(
            use_cookies, cookies
        )

        try:
            # Base configuration
            ydl_opts = {
                "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
                "quiet": True,
                "no_warnings": True,
                "noplaylist": True,
            }

            if cookiefile:
                ydl_opts["cookiefile"] = cookiefile

            # Configure format based on download type
            if download_video_flag:
                # Download best quality video with audio
                ydl_opts["format"] = (
                    "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
                )
                ydl_opts["merge_output_format"] = "mp4"
            else:
                # Extract audio only
                ydl_opts["format"] = "bestaudio[ext=m4a]/bestaudio/best"
                ydl_opts["postprocessors"] = [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                        "preferredquality": "192",
                    }
                ]

            # Add ffmpeg location if specified
            ffmpeg_path = get_cli_setting("media_processing.ffmpeg_path")
            if ffmpeg_path:
                ydl_opts["ffmpeg_location"] = ffmpeg_path

            # Extract info first to check file size. (task-3306 review
            # round) The probe runs BEFORE the download, so an
            # authentication-gated URL -- the only reason the cookies
            # option exists -- failed here while the cookies sat unused in
            # the download's options.
            probe_opts: Dict[str, Any] = {"quiet": True}
            if cookiefile:
                probe_opts["cookiefile"] = cookiefile
            metadata_start = time.time()
            with yt_dlp.YoutubeDL(probe_opts) as ydl:
                info = ydl.extract_info(url, download=False)

                # Log metadata extraction
                metadata_duration = time.time() - metadata_start
                log_histogram(
                    "video_processing_metadata_extraction_duration", metadata_duration
                )

                # Check file size if available
                filesize = info.get("filesize") or info.get("filesize_approx", 0)
                if filesize:
                    log_histogram("video_processing_file_size_bytes", filesize)
                    if filesize > self.max_file_size:
                        log_counter(
                            "video_processing_download_error",
                            labels={"error_type": "file_too_large"},
                        )
                        raise VideoDownloadError(
                            f"File size ({filesize / (1024 * 1024):.2f} MB) exceeds limit"
                        )

            # Perform download
            download_start = time.time()
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)

                # Log download duration
                download_duration = time.time() - download_start
                log_histogram(
                    "video_processing_download_duration",
                    download_duration,
                    labels={
                        "download_type": "full_video"
                        if download_video_flag
                        else "audio_only"
                    },
                )

                # Get the actual filename
                filename = ydl.prepare_filename(info)

                # Handle different output formats
                if not download_video_flag:
                    # Audio extraction changes extension
                    base_name = os.path.splitext(filename)[0]
                    audio_path = base_name + ".mp3"
                    if os.path.exists(audio_path):
                        return audio_path

                if os.path.exists(filename):
                    # Log success
                    duration = time.time() - start_time
                    file_size = os.path.getsize(filename)
                    log_histogram(
                        "video_processing_download_total_duration",
                        duration,
                        labels={
                            "status": "success",
                            "download_type": "full_video"
                            if download_video_flag
                            else "audio_only",
                        },
                    )
                    log_histogram(
                        "video_processing_downloaded_file_size_bytes", file_size
                    )
                    log_counter(
                        "video_processing_download_success",
                        labels={
                            "download_type": "full_video"
                            if download_video_flag
                            else "audio_only"
                        },
                    )
                    return filename

                # Try to find the file with different extensions
                base_name = os.path.splitext(filename)[0]
                for ext in [".mp4", ".mp3", ".m4a", ".webm", ".mkv"]:
                    test_path = base_name + ext
                    if os.path.exists(test_path):
                        # Log success
                        duration = time.time() - start_time
                        file_size = os.path.getsize(test_path)
                        log_histogram(
                            "video_processing_download_total_duration",
                            duration,
                            labels={
                                "status": "success",
                                "download_type": "full_video"
                                if download_video_flag
                                else "audio_only",
                            },
                        )
                        log_histogram(
                            "video_processing_downloaded_file_size_bytes", file_size
                        )
                        log_counter(
                            "video_processing_download_success",
                            labels={
                                "download_type": "full_video"
                                if download_video_flag
                                else "audio_only",
                                "found_with_extension": ext,
                            },
                        )
                        return test_path

                # Log success
                duration = time.time() - start_time
                log_histogram(
                    "video_processing_download_total_duration",
                    duration,
                    labels={
                        "status": "not_found",
                        "download_type": "full_video"
                        if download_video_flag
                        else "audio_only",
                    },
                )
                log_counter(
                    "video_processing_download_error",
                    labels={"error_type": "file_not_found"},
                )
                raise VideoDownloadError("Downloaded file not found")

        except Exception as e:
            # Log error
            duration = time.time() - start_time
            log_histogram(
                "video_processing_download_total_duration",
                duration,
                labels={
                    "status": "error",
                    "download_type": "full_video"
                    if download_video_flag
                    else "audio_only",
                },
            )
            log_counter(
                "video_processing_download_error",
                labels={"error_type": type(e).__name__},
            )
            logger.error(f"Video download error: {str(e)}")
            raise VideoDownloadError(f"Download failed: {str(e)}") from e
        finally:
            # Clean up ONLY the temporary cookie file this call created.
            # A user-supplied path is never ours to delete, wherever it
            # happens to live.
            self._discard_temp_cookiefile(owned_temp_cookiefile)

    def extract_metadata(
        self, url: str, use_cookies: bool = False, cookies: Optional[Dict] = None
    ) -> Optional[Dict[str, Any]]:
        """Extract metadata from video URL without downloading.

        Args:
            url: Video URL.
            use_cookies: Whether to authenticate the request with cookies.
            cookies: A cookies-file path, a JSON string of ``{name: value}``
                pairs, or that mapping. (task-3306 review round) These two
                arguments were declared and then ignored, so metadata for a
                gated URL failed even when the caller had cookies.

        Returns:
            The metadata dict, or ``None`` on any failure.
        """
        start_time = time.time()
        log_counter("video_processing_metadata_attempt")

        if not YT_DLP_AVAILABLE:
            log_counter(
                "video_processing_metadata_error",
                labels={"error_type": "yt_dlp_not_available"},
            )
            return None

        # (TASK-19556) The metadata seam fetches too -- same guard as the
        # download seam, reported through this function's None contract.
        try:
            check_media_url_egress(url)
        except VideoDownloadError as exc:
            log_counter(
                "video_processing_metadata_error",
                labels={"error_type": "egress_blocked"},
            )
            logger.error(f"Metadata extraction refused: {exc}")
            return None

        owned_temp_cookiefile: Optional[str] = None
        try:
            cookiefile, owned_temp_cookiefile = self._resolve_cookiefile(
                use_cookies, cookies
            )
            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
                "extract_flat": False,  # Get full info
                "skip_download": True,
            }
            if cookiefile:
                ydl_opts["cookiefile"] = cookiefile

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)

                metadata = {
                    "title": info.get("title"),
                    "uploader": info.get("uploader"),
                    "upload_date": info.get("upload_date"),
                    "view_count": info.get("view_count"),
                    "like_count": info.get("like_count"),
                    "duration": info.get("duration"),
                    "tags": info.get("tags", []),
                    "description": info.get("description"),
                    "webpage_url": info.get("webpage_url", url),
                    "thumbnail": info.get("thumbnail"),
                }

                # Log success
                duration = time.time() - start_time
                log_histogram(
                    "video_processing_metadata_duration",
                    duration,
                    labels={"status": "success"},
                )
                log_counter(
                    "video_processing_metadata_success",
                    labels={
                        "has_duration": str(bool(metadata.get("duration"))),
                        "has_uploader": str(bool(metadata.get("uploader"))),
                    },
                )

                return metadata

        except Exception as e:
            # Log error
            duration = time.time() - start_time
            log_histogram(
                "video_processing_metadata_duration",
                duration,
                labels={"status": "error"},
            )
            log_counter(
                "video_processing_metadata_error",
                labels={"error_type": type(e).__name__},
            )
            logger.error(f"Error extracting metadata: {str(e)}")
            return None
        finally:
            self._discard_temp_cookiefile(owned_temp_cookiefile)

    def process_videos(
        self,
        inputs: List[str],
        download_video_flag: bool = False,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Process multiple video inputs.

        Args:
            inputs: List of video URLs or local file paths
            download_video_flag: If True, keep video file; if False, extract audio only
            start_time: Optional start time for video extraction (HH:MM:SS or seconds)
            end_time: Optional end time for video extraction (HH:MM:SS or seconds)
            **kwargs: Additional arguments passed to audio processing

        Returns:
            Dict with processing results
        """
        processing_start_time = time.time()
        total_inputs = len(inputs)
        log_counter(
            "video_processing_batch_start",
            labels={
                "total_inputs": str(total_inputs),
                "download_video": str(download_video_flag),
            },
        )

        results = []
        errors = []

        with tempfile.TemporaryDirectory(prefix="video_proc_") as temp_dir:
            for input_item in inputs:
                # Check for cancellation before processing each file
                if self.is_cancelled():
                    logger.info("Processing cancelled by user")
                    # Note: No progress callback available in this method signature
                    break

                try:
                    result = self._process_single_video(
                        input_item=input_item,
                        temp_dir=temp_dir,
                        download_video_flag=download_video_flag,
                        start_time=start_time,
                        end_time=end_time,
                        **kwargs,
                    )
                    results.append(result)

                except Exception as e:
                    logger.error(f"Error processing video {input_item}: {str(e)}")
                    error_result = {
                        "status": "Error",
                        "input_ref": input_item,
                        "error": str(e),
                        "media_type": "video",
                    }
                    results.append(error_result)
                    errors.append(str(e))

        # Calculate summary statistics
        processed_count = sum(1 for r in results if r.get("status") == "Success")
        errors_count = sum(1 for r in results if r.get("status") == "Error")

        # Log batch completion metrics
        duration = time.time() - processing_start_time
        log_histogram(
            "video_processing_batch_duration",
            duration,
            labels={
                "total_inputs": str(total_inputs),
                "success_count": str(processed_count),
            },
        )
        log_counter(
            "video_processing_batch_complete",
            labels={
                "total_inputs": str(total_inputs),
                "success_count": str(processed_count),
                "error_count": str(errors_count),
            },
        )

        return {
            "processed_count": processed_count,
            "errors_count": errors_count,
            "errors": errors,
            "results": results,
        }

    def _process_single_video(
        self,
        input_item: str,
        temp_dir: str,
        download_video_flag: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Process a single video file or URL."""
        start_time = time.time()
        logger.info(f"Starting single video processing for: {input_item}")
        logger.debug(f"Video processing kwargs: {kwargs}")
        logger.debug(f"Original input_item: '{input_item}'")

        result = {
            "status": "Pending",
            "input_ref": input_item,
            "processing_source": input_item,
            "media_type": "video",
            "metadata": {},
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
            # Check if it has a scheme or looks like a URL without scheme
            parsed = urlparse(input_item)
            is_url = parsed.scheme in ("http", "https")

            # If no scheme, check if it looks like a URL (e.g., www.youtube.com, youtube.com)
            if not is_url and not os.path.exists(input_item):
                # Common URL patterns without scheme
                url_patterns = [
                    "www.",
                    "youtube.com",
                    "youtu.be",
                    "vimeo.com",
                    "dailymotion.com",
                    "twitch.tv",
                    "twitter.com",
                    "x.com",
                    "instagram.com",
                    "facebook.com",
                ]
                # Check if it contains common domain patterns
                if (
                    any(pattern in input_item for pattern in url_patterns)
                    or "." in input_item
                    and "/" in input_item
                ):
                    # Add https:// prefix and update input_item
                    input_item = f"https://{input_item}"
                    is_url = True
                    logger.info(f"Added https:// prefix to URL: {input_item}")

            logger.debug(
                f"Input type for {input_item}: {'URL' if is_url else 'Local file'}"
            )

            log_counter(
                "video_processing_single_attempt",
                labels={
                    "is_url": str(is_url),
                    "download_video": str(download_video_flag),
                },
            )

            if is_url:
                # Update result with the corrected URL
                result["input_ref"] = input_item
                result["processing_source"] = input_item

                # Extract metadata
                logger.info(f"[VIDEO] Extracting metadata for URL: {input_item}")
                metadata = self.extract_metadata(
                    input_item, kwargs.get("use_cookies", False), kwargs.get("cookies")
                )
                if metadata:
                    result["metadata"] = metadata
                    logger.info(
                        f"[VIDEO] Metadata extracted successfully: title='{metadata.get('title', 'N/A')}', duration={metadata.get('duration', 'N/A')}s"
                    )
                else:
                    logger.warning(
                        f"[VIDEO] No metadata extracted for URL: {input_item}"
                    )

                # Get transcription progress callback if provided
                transcription_callback = kwargs.get("transcription_progress_callback")

                # Download video/audio
                logger.info(f"[VIDEO] Starting download from URL: {input_item}")
                logger.debug(
                    f"[VIDEO] Download settings: video={download_video_flag}, cookies={kwargs.get('use_cookies', False)}"
                )

                downloaded_path = self.download_video(
                    url=input_item,
                    output_dir=temp_dir,
                    download_video_flag=download_video_flag,
                    use_cookies=kwargs.get("use_cookies", False),
                    cookies=kwargs.get("cookies"),
                )

                if not downloaded_path:
                    logger.error(f"[VIDEO] Download failed for URL: {input_item}")
                    raise VideoDownloadError("Download failed")

                logger.info(
                    f"[VIDEO] Download completed successfully: {downloaded_path}"
                )
                logger.info(
                    f"[VIDEO] Downloaded file size: {os.path.getsize(downloaded_path) / (1024 * 1024):.2f} MB"
                )

                # Notify about download completion if we have a callback
                if transcription_callback:
                    transcription_callback(
                        0, "Download complete, preparing for transcription...", None
                    )

                processing_path = downloaded_path

            else:
                # Local file
                if not os.path.exists(input_item):
                    raise FileNotFoundError(f"File not found: {input_item}")

                processing_path = input_item
                result["metadata"]["title"] = Path(input_item).stem

            # Check if we have audio file or need to extract audio from video
            file_ext = Path(processing_path).suffix.lower()
            audio_extensions = {".mp3", ".m4a", ".wav", ".flac", ".ogg", ".aac"}

            if file_ext in audio_extensions:
                # Already audio file, process directly
                audio_path = processing_path
                logger.info(
                    f"[VIDEO] Input is already an audio file ({file_ext}), skipping extraction"
                )
            else:
                # Extract audio from video
                logger.info(
                    f"[VIDEO] Starting audio extraction from video: {processing_path}"
                )
                logger.info(f"[VIDEO] Video file extension: {file_ext}")
                try:
                    audio_path = self._extract_audio_from_video(
                        processing_path,
                        temp_dir,
                        kwargs.get("start_time"),
                        kwargs.get("end_time"),
                    )
                    logger.info(
                        f"[VIDEO] Audio extraction completed successfully: {audio_path}"
                    )
                    logger.info(
                        f"[VIDEO] Extracted audio file size: {os.path.getsize(audio_path) / (1024 * 1024):.2f} MB"
                    )
                    # (task-3306) The extraction above already applied the
                    # requested time range. The audio stage below re-cuts
                    # any local non-YouTube input carrying these bounds
                    # (``_process_single_audio`` -> ``_extract_time_range``),
                    # which would apply the trim TWICE -- a start of 60s
                    # shifting the transcript window to 120s. Drop the
                    # bounds so the trim runs exactly once. The
                    # already-audio path above keeps them: no extraction
                    # ran there, so the audio stage's trim is the only one.
                    kwargs.pop("start_time", None)
                    kwargs.pop("end_time", None)
                except Exception as e:
                    logger.opt(exception=True).error(
                        f"[VIDEO] Audio extraction failed: {type(e).__name__}: {str(e)}"
                    )
                    raise

            # Process audio using audio processor
            logger.info(f"Starting audio processing for extracted audio: {audio_path}")
            logger.info(
                f"Audio file details - exists: {os.path.exists(audio_path)}, size: {os.path.getsize(audio_path) if os.path.exists(audio_path) else 'N/A'} bytes"
            )
            logger.info(
                f"Audio processing parameters: provider={kwargs.get('transcription_provider')}, model={kwargs.get('transcription_model')}, language={kwargs.get('transcription_language')}"
            )

            # Get transcription progress callback if provided
            transcription_callback = kwargs.pop("transcription_progress_callback", None)

            # Notify about audio extraction completion if we have a callback
            if transcription_callback and file_ext not in audio_extensions:
                logger.info("Notifying callback about audio extraction completion")
                transcription_callback(
                    0, "Audio extracted, starting transcription...", None
                )

            logger.info("[VIDEO] Calling audio processor _process_single_audio()")
            logger.info(
                f"[VIDEO] Transcription parameters: provider={kwargs.get('transcription_provider')}, model={kwargs.get('transcription_model')}, language={kwargs.get('transcription_language')}"
            )

            try:
                audio_result = self.audio_processor._process_single_audio(
                    input_item=audio_path,
                    processing_dir=temp_dir,
                    transcription_progress_callback=transcription_callback,
                    media_type="video",
                    original_url=input_item
                    if is_url
                    else None,  # Pass original URL for proper storage
                    **kwargs,
                )
                logger.info(
                    "[VIDEO] Audio processor _process_single_audio() returned successfully"
                )
            except Exception as e:
                logger.opt(exception=True).error(
                    f"[VIDEO] Audio processing failed with exception: {type(e).__name__}: {str(e)}"
                )
                raise

            # Log detailed result information
            if audio_result:
                logger.info(
                    f"[VIDEO] Audio processing result status: {audio_result.get('status', 'Unknown')}"
                )
                logger.info(
                    f"[VIDEO] Transcription content length: {len(audio_result.get('content', '')) if audio_result.get('content') else 0} characters"
                )
                logger.info(
                    f"[VIDEO] Number of segments: {len(audio_result.get('segments', [])) if audio_result.get('segments') else 0}"
                )
                logger.info(
                    f"[VIDEO] Number of chunks: {len(audio_result.get('chunks', [])) if audio_result.get('chunks') else 0}"
                )
                logger.info(
                    f"[VIDEO] Has analysis: {bool(audio_result.get('analysis'))}"
                )

                if not audio_result.get("content"):
                    logger.warning("[VIDEO] No transcription content in audio result!")

                logger.debug(
                    f"[VIDEO] Full audio result keys: {list(audio_result.keys()) if audio_result else 'None'}"
                )
                if audio_result.get("status") == "Error":
                    result["status"] = "Error"
                    result["error"] = audio_result.get(
                        "error", "Speech-to-text failed."
                    )
                    if isinstance(audio_result.get("error_detail"), dict):
                        result["error_detail"] = audio_result["error_detail"]
                    if isinstance(audio_result.get("stt_failure_provenance"), dict):
                        result["stt_failure_provenance"] = audio_result[
                            "stt_failure_provenance"
                        ]
                    return result
            else:
                logger.error("[VIDEO] Audio processing failed - no result returned")
                raise Exception("Audio processing failed - no result returned")

            # Merge results
            result.update(
                {
                    # Don't overwrite processing_source for URLs - keep the original URL
                    "content": audio_result.get("content"),
                    "segments": audio_result.get("segments"),
                    "chunks": audio_result.get("chunks") or [],
                    "analysis": audio_result.get("analysis"),
                    "analysis_details": audio_result.get("analysis_details"),
                    "warnings": audio_result.get("warnings", []),
                    "transcription_model": audio_result.get("transcription_model"),
                    "transcription_provenance": audio_result.get(
                        "transcription_provenance"
                    ),
                }
            )

            # Update metadata if not already set
            if not result["metadata"].get("title") and audio_result.get(
                "metadata", {}
            ).get("title"):
                result["metadata"]["title"] = audio_result["metadata"]["title"]

            # Store in database if available
            logger.info(
                f"[VIDEO] Checking database save conditions: media_db={bool(self.media_db)}, has_content={bool(result.get('content'))}"
            )
            if self.media_db and result["content"]:
                logger.info(f"[VIDEO] Starting database save for: {input_item}")
                logger.info(
                    f"[VIDEO] Content to save length: {len(result['content'])} characters"
                )
                logger.debug(
                    f"[VIDEO] Full result before DB save: {json.dumps({k: str(v)[:100] + '...' if isinstance(v, str) and len(str(v)) > 100 else v for k, v in result.items()}, indent=2)}"
                )

                try:
                    db_result = self._store_in_database(result)
                    result["db_id"] = db_result.get("id")
                    result["db_message"] = db_result.get(
                        "message", "Stored successfully"
                    )
                    logger.info(
                        f"[VIDEO] Database storage successful: id={db_result.get('id')}, message={db_result.get('message')}"
                    )
                except Exception as e:
                    logger.opt(exception=True).error(
                        f"[VIDEO] Database storage failed: {type(e).__name__}: {str(e)}"
                    )
                    result["db_id"] = None
                    result["db_message"] = f"Database storage failed: {str(e)}"
                    raise
            else:
                if not self.media_db:
                    logger.warning(
                        "[VIDEO] No media database available - skipping save"
                    )
                    result["db_message"] = "No database available"
                elif not result.get("content"):
                    logger.warning(
                        "[VIDEO] No transcription content to save - skipping database save"
                    )
                    result["db_message"] = "No transcription content to save"

            result["status"] = "Success" if not result["warnings"] else "Warning"

            total_time = time.time() - start_time
            logger.info(
                f"Successfully completed video processing for {input_item} in {total_time:.2f} seconds"
            )

            # Log success metrics
            duration = time.time() - start_time
            log_histogram(
                "video_processing_single_duration",
                duration,
                labels={
                    "status": "success",
                    "is_url": str(is_url),
                    "has_analysis": str(bool(result.get("analysis"))),
                },
            )
            log_counter(
                "video_processing_single_success",
                labels={
                    "is_url": str(is_url),
                    "download_video": str(download_video_flag),
                    "chunks_created": str(len(result.get("chunks") or [])),
                },
            )

            # Handle keep_original option - move video/audio file to Downloads folder
            if kwargs.get("keep_original", False) and is_url and download_video_flag:
                try:
                    # Get user's Downloads folder
                    downloads_dir = Path.home() / "Downloads"
                    downloads_dir.mkdir(exist_ok=True)

                    # Generate a unique filename if needed
                    media_filename = Path(processing_path).name
                    dest_path = downloads_dir / media_filename

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
                    shutil.move(processing_path, str(dest_path))
                    logger.info(f"Moved video file to: {dest_path}")
                    result["saved_video_path"] = str(dest_path)
                    result["warnings"].append(
                        f"Video file saved to Downloads folder: {dest_path.name}"
                    )
                except Exception as e:
                    logger.error(f"Failed to move video file to Downloads: {str(e)}")
                    result["warnings"].append(f"Could not save video file: {str(e)}")

        except Exception as e:
            error_msg = str(e)
            # Check if this is a cancellation
            if "cancelled by user" in error_msg.lower():
                logger.info(f"Video processing cancelled by user for {input_item}")
                result["status"] = "Cancelled"
                result["error"] = "Processing cancelled by user"
            else:
                logger.opt(exception=True).error(f"Error processing video: {error_msg}")
                result["status"] = "Error"
                result["error"] = error_msg

            # Log error metrics
            duration = time.time() - start_time
            log_histogram(
                "video_processing_single_duration",
                duration,
                labels={"status": "error", "is_url": str(is_url)},
            )
            log_counter(
                "video_processing_single_error",
                labels={"is_url": str(is_url), "error_type": type(e).__name__},
            )

        return result

    def _extract_audio_from_video(
        self,
        video_path: str,
        output_dir: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> str:
        """Extract audio track from video file."""
        time.time()
        log_counter("video_processing_audio_extraction_attempt")

        logger.info(f"Extracting audio from video: {video_path}")
        logger.info(
            f"Video file exists: {os.path.exists(video_path)}, size: {os.path.getsize(video_path) if os.path.exists(video_path) else 'N/A'} bytes"
        )

        # Find ffmpeg
        try:
            ffmpeg_cmd = self._find_ffmpeg()
            logger.info(f"Found ffmpeg at: {ffmpeg_cmd}")
        except FileNotFoundError as e:
            logger.error(f"ffmpeg not found: {str(e)}")
            raise

        # Output path
        base_name = Path(video_path).stem
        if start_time or end_time:
            suffix = f"_trim_{start_time or '0'}_{end_time or 'end'}".replace(":", "-")
            audio_path = os.path.join(output_dir, f"{base_name}_audio{suffix}.mp3")
        else:
            audio_path = os.path.join(output_dir, f"{base_name}_audio.mp3")

        # Extract audio as MP3. The trim arguments come from the shared
        # builder in ``audio_processing`` -- this path used to emit ``-ss``
        # before ``-i`` (input seeking rebases the output's timestamps to
        # zero) and then ``-to`` as an OUTPUT option, which turned an
        # absolute "Stop at" into a duration measured from "Start at": the
        # same pair that selected 0:30-1:00 on an .mp3 selected 0:30-1:30
        # on an .mp4 (task-3306 review round).
        pre_input, post_input = build_ffmpeg_trim_args(start_time, end_time)
        command = [ffmpeg_cmd, *pre_input, "-i", video_path, *post_input]

        # Audio extraction options
        command.extend(
            [
                "-vn",  # No video
                "-acodec",
                "libmp3lame",
                "-ab",
                "192k",  # Audio bitrate
                "-ar",
                "44100",  # Sample rate
                "-y",  # Overwrite
                audio_path,
            ]
        )

        try:
            logger.info(f"Running ffmpeg command: {' '.join(command)}")
            logger.info(f"Extracting audio to: {audio_path}")

            subprocess.run(command, capture_output=True, text=True, check=True)

            logger.info("ffmpeg completed successfully")
            logger.info(
                f"Audio file created: {audio_path}, exists: {os.path.exists(audio_path)}, size: {os.path.getsize(audio_path) if os.path.exists(audio_path) else 'N/A'} bytes"
            )

            if not os.path.exists(audio_path):
                raise VideoProcessingError(
                    f"Audio extraction succeeded but output file not found: {audio_path}"
                )

            return audio_path

        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg failed with exit code {e.returncode}")
            logger.error(f"ffmpeg stderr: {e.stderr}")
            logger.error(f"ffmpeg stdout: {e.stdout}")
            raise VideoProcessingError(f"Failed to extract audio: {e.stderr}") from e

    def _find_ffmpeg(self) -> str:
        """Find ffmpeg executable."""
        # Check config first
        ffmpeg_path = get_cli_setting("media_processing.ffmpeg_path")
        if ffmpeg_path and os.path.exists(ffmpeg_path):
            return ffmpeg_path

        # Check common locations
        import shutil

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            return ffmpeg

        raise FileNotFoundError("ffmpeg not found")

    def _store_in_database(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Store processing results in the media database."""
        logger.info("[VIDEO] _store_in_database called")
        if not self.media_db:
            logger.warning("[VIDEO] No media database instance available")
            return {"message": "No database available"}

        try:
            # Prepare media data
            media_data = {
                "url": result.get("input_ref", ""),
                "title": result["metadata"].get("title", "Untitled"),
                "media_type": "video",
                "content": result.get("content", ""),  # Store transcription
                "author": result["metadata"].get("uploader", "Unknown"),
                "ingestion_date": None,  # Will use current time
                "analysis_content": result.get("analysis"),  # Store analysis separately
            }

            logger.info("[VIDEO] Prepared media data for DB save:")
            logger.info(f"[VIDEO]   - URL: {media_data['url']}")
            logger.info(f"[VIDEO]   - Title: {media_data['title']}")
            logger.info(
                f"[VIDEO]   - Content length: {len(media_data['content'])} chars"
            )
            logger.info(
                f"[VIDEO]   - Has analysis: {bool(media_data['analysis_content'])}"
            )

            # Add media entry with analysis
            logger.info("[VIDEO] Calling media_db.add_media_with_keywords()")
            media_id, _, _ = self.media_db.add_media_with_keywords(**media_data)
            logger.info(
                f"[VIDEO] Successfully saved to database with media_id: {media_id}"
            )

            # Store chunks if available
            if result.get("chunks"):
                logger.info(
                    f"[VIDEO] Storing {len(result['chunks'])} chunks for media_id: {media_id}"
                )

                # Prepare chunks in the format expected by add_media_chunks_in_batches
                chunks_to_add = []
                for i, chunk in enumerate(result["chunks"]):
                    chunk_text = chunk.get("text", "")
                    # Calculate start and end indices based on chunk position
                    # This is approximate since we don't have exact character positions
                    text_length = len(chunk_text)
                    start_index = sum(
                        len(c.get("text", "")) for c in result["chunks"][:i]
                    )
                    end_index = start_index + text_length

                    chunks_to_add.append(
                        {
                            "text": chunk_text,
                            "start_index": start_index,
                            "end_index": end_index,
                        }
                    )

                # Use batch insert method
                chunks_added = self.media_db.add_media_chunks_in_batches(
                    media_id=media_id, chunks_to_add=chunks_to_add
                )
                logger.info(f"[VIDEO] Successfully stored {chunks_added} chunks")
            else:
                logger.info("[VIDEO] No chunks to store")

            return {"id": media_id, "message": "Stored successfully"}

        except Exception as e:
            logger.opt(exception=True).error(
                f"[VIDEO] Database storage error: {type(e).__name__}: {str(e)}"
            )
            return {"message": f"Storage failed: {str(e)}"}


# Convenience function for backwards compatibility
def process_videos(**kwargs) -> Dict[str, Any]:
    """Process videos using LocalVideoProcessor."""
    processor = LocalVideoProcessor()
    return processor.process_videos(**kwargs)
