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
from typing import Optional, List, Dict, Any, Tuple
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
    logger.warning("numpy not available. Some audio processing features will be limited.")
#
# Local imports
from ..config import get_media_ingestion_defaults, get_cli_setting
from ..Chat.Chat_Functions import chat_api_call
from ..RAG_Search.chunking_service import ChunkingService
from ..DB.Client_Media_DB_v2 import MediaDatabase
from ..Utils.text import sanitize_filename
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


class LocalAudioProcessor:
    """Handles local audio processing including download, transcription, and analysis."""
    
    def __init__(self, media_db: Optional[MediaDatabase] = None):
        """
        Initialize the audio processor.
        
        Args:
            media_db: Optional MediaDatabase instance for storage
        """
        self.media_db = media_db
        self.config = get_media_ingestion_defaults("audio")
        self.chunking_service = ChunkingService()
        
        # Get configuration settings
        self.max_file_size_mb = get_cli_setting('media_processing.max_audio_file_size_mb', 500)
        if self.max_file_size_mb is None:
            self.max_file_size_mb = 500
        self.max_file_size = self.max_file_size_mb * 1024 * 1024
        
    def download_audio_file(self, url: str, target_dir: str, 
                          use_cookies: bool = False, 
                          cookies: Optional[Dict] = None) -> str:
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
        try:
            logger.info(f"Downloading audio from: {url}")
            
            headers = {}
            if use_cookies and cookies:
                if isinstance(cookies, str):
                    cookie_dict = json.loads(cookies)
                else:
                    cookie_dict = cookies
                headers['Cookie'] = '; '.join([f'{k}={v}' for k, v in cookie_dict.items()])
            
            response = requests.get(url, headers=headers, stream=True, timeout=120)
            response.raise_for_status()
            
            # Check file size
            file_size = int(response.headers.get('content-length', 0))
            if file_size > self.max_file_size:
                raise AudioDownloadError(
                    f"File size ({file_size / (1024*1024):.2f} MB) exceeds limit"
                )
            
            # Determine filename
            filename = self._get_filename_from_response(response, url)
            save_path = Path(target_dir) / filename
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download file
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"Downloaded audio file: {save_path}")
            return str(save_path)
            
        except requests.RequestException as e:
            raise AudioDownloadError(f"Download failed: {str(e)}") from e
        except Exception as e:
            raise AudioDownloadError(f"Unexpected error: {str(e)}") from e
    
    def download_youtube_audio(self, url: str, output_dir: str, 
                              start_time: Optional[str] = None, 
                              end_time: Optional[str] = None) -> Optional[str]:
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
                'format': 'bestaudio[ext=m4a]/bestaudio/best',
                'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
                'quiet': True,
                'no_warnings': True,
                'extract_audio': True,
                'audio_format': 'mp3',
                'audio_quality': 192,
            }
            
            # Add time range options if specified
            if start_time or end_time:
                # yt-dlp uses postprocessor args for time ranges
                postprocessor_args = []
                if start_time:
                    postprocessor_args.extend(['-ss', start_time])
                if end_time:
                    if start_time:
                        postprocessor_args.extend(['-to', end_time])
                    else:
                        postprocessor_args.extend(['-t', end_time])
                
                ydl_opts['postprocessor_args'] = {
                    'ffmpeg': postprocessor_args
                }
            
            # Add ffmpeg location if specified in config
            ffmpeg_path = get_cli_setting('media_processing.ffmpeg_path')
            if ffmpeg_path:
                ydl_opts['ffmpeg_location'] = ffmpeg_path
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                # yt-dlp might change extension after conversion
                audio_path = filename.rsplit('.', 1)[0] + '.mp3'
                
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
        transcription_language: Optional[str] = 'en',
        translation_target_language: Optional[str] = None,
        perform_chunking: bool = True,
        chunk_method: Optional[str] = None,
        max_chunk_size: int = 500,
        chunk_overlap: int = 200,
        use_adaptive_chunking: bool = False,
        use_multi_level_chunking: bool = False,
        chunk_language: Optional[str] = None,
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
        temp_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process multiple audio inputs (URLs or local files).
        
        Returns:
            Dict with processing results
        """
        results = []
        errors = []
        
        with tempfile.TemporaryDirectory(prefix="audio_proc_") as default_temp:
            processing_dir = temp_dir or default_temp
            
            for input_item in inputs:
                try:
                    result = self._process_single_audio(
                        input_item=input_item,
                        processing_dir=processing_dir,
                        transcription_provider=transcription_provider,
                        transcription_model=transcription_model,
                        transcription_language=transcription_language,
                        translation_target_language=translation_target_language,
                        perform_chunking=perform_chunking,
                        chunk_method=chunk_method,
                        max_chunk_size=max_chunk_size,
                        chunk_overlap=chunk_overlap,
                        use_adaptive_chunking=use_adaptive_chunking,
                        use_multi_level_chunking=use_multi_level_chunking,
                        chunk_language=chunk_language,
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
                        author=author
                    )
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing {input_item}: {str(e)}")
                    error_result = {
                        "status": "Error",
                        "input_ref": input_item,
                        "error": str(e),
                        "media_type": "audio"
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
            "results": results
        }
    
    def _process_single_audio(
        self,
        input_item: str,
        processing_dir: str,
        transcription_progress_callback=None,
        media_type: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Process a single audio file or URL."""
        
        # Check if this is being called from video processing with an original URL
        original_url = kwargs.pop('original_url', None)
        
        result = {
            "status": "Pending",
            "input_ref": original_url or input_item,  # Use original URL if provided
            "processing_source": input_item,
            "media_type": media_type or "audio",
            "metadata": {"title": kwargs.get("custom_title"), "author": kwargs.get("author")},
            "content": None,
            "segments": None,
            "chunks": None,
            "analysis": None,
            "analysis_details": {},
            "error": None,
            "warnings": []
        }
        
        try:
            # Determine if input is URL or local file
            is_url = urlparse(input_item).scheme in ('http', 'https')
            
            if is_url:
                # Download the file
                if 'youtube.com' in input_item or 'youtu.be' in input_item:
                    audio_path = self.download_youtube_audio(
                        input_item, 
                        processing_dir,
                        kwargs.get("start_time"),
                        kwargs.get("end_time")
                    )
                else:
                    audio_path = self.download_audio_file(
                        input_item, processing_dir, 
                        kwargs.get("use_cookies", False),
                        kwargs.get("cookies")
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
            youtube_url = is_url and ('youtube.com' in input_item or 'youtu.be' in input_item)
            
            if (start_time or end_time) and not youtube_url:
                logger.info(f"Extracting time range: start={start_time}, end={end_time}")
                audio_path = self._extract_time_range(
                    audio_path, 
                    processing_dir,
                    start_time,
                    end_time
                )
            
            # Update metadata
            if not result["metadata"]["title"]:
                result["metadata"]["title"] = Path(audio_path).stem
            
            # Transcribe audio
            provider = kwargs.get("transcription_provider", None)
            model = kwargs.get("transcription_model", None)
            language = kwargs.get("transcription_language", None)
            
            logger.info(f"Starting transcription: provider={provider}, model={model}, language={language}")
            logger.debug(f"Transcription audio file: {audio_path}, size: {Path(audio_path).stat().st_size / 1024 / 1024:.2f} MB")
            
            transcription_start = time.time()
            transcription_result = self._transcribe_audio(
                audio_path,
                provider=provider,
                model=model,
                language=language,
                target_lang=kwargs.get("translation_target_language"),
                vad_filter=kwargs.get("vad_use", False),
                diarize=kwargs.get("diarize", False),
                progress_callback=transcription_progress_callback
            )
            
            transcription_time = time.time() - transcription_start
            logger.info(f"Transcription completed in {transcription_time:.2f} seconds")
            
            result["segments"] = transcription_result.get("segments", [])
            result["content"] = transcription_result.get("text", "")
            
            logger.debug(f"Transcription result: {len(result['content'])} chars, {len(result['segments'])} segments")
            
            # Perform chunking if requested
            if kwargs.get("perform_chunking") and result["content"]:
                chunk_method = kwargs.get("chunk_method", "sentences")
                logger.info(f"Starting text chunking: method={chunk_method}, max_size={kwargs.get('max_chunk_size', 500)}")
                
                chunks = self._chunk_text(
                    result["content"],
                    method=chunk_method,
                    max_size=kwargs.get("max_chunk_size", 500),
                    overlap=kwargs.get("chunk_overlap", 200),
                    language=kwargs.get("chunk_language") or kwargs.get("transcription_language", "en")
                )
                result["chunks"] = chunks
                logger.debug(f"Chunking completed: {len(chunks)} chunks created")
            
            # Perform analysis if requested
            if kwargs.get("perform_analysis") and kwargs.get("api_name") and result["content"]:
                analysis = self._analyze_content(
                    content=result["content"],
                    chunks=result.get("chunks"),
                    api_name=kwargs["api_name"],
                    api_key=kwargs.get("api_key"),
                    custom_prompt=kwargs.get("custom_prompt"),
                    system_prompt=kwargs.get("system_prompt"),
                    summarize_recursively=kwargs.get("summarize_recursively", False)
                )
                result["analysis"] = analysis
                result["analysis_details"] = {
                    "api_name": kwargs["api_name"],
                    "custom_prompt": kwargs.get("custom_prompt"),
                    "recursive": kwargs.get("summarize_recursively", False)
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
                            dest_path = downloads_dir / f"{base_name}_{counter}{extension}"
                            counter += 1
                    
                    # Move the file
                    shutil.move(audio_path, str(dest_path))
                    logger.info(f"Moved audio file to: {dest_path}")
                    result["saved_audio_path"] = str(dest_path)
                    result["warnings"].append(f"Audio file saved to Downloads folder: {dest_path.name}")
                except Exception as e:
                    logger.error(f"Failed to move audio file to Downloads: {str(e)}")
                    result["warnings"].append(f"Could not save audio file: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}", exc_info=True)
            result["status"] = "Error"
            result["error"] = str(e)
        
        return result
    
    def _transcribe_audio(self, audio_path: str, progress_callback=None, **kwargs) -> Dict[str, Any]:
        """
        Transcribe audio using available transcription service.
        
        Args:
            audio_path: Path to audio file
            progress_callback: Optional callback for progress updates
            **kwargs: Additional transcription parameters
        """
        # Import transcription service when available
        try:
            from .transcription_service import TranscriptionService
            service = TranscriptionService()
            return service.transcribe(audio_path, progress_callback=progress_callback, **kwargs)
        except ImportError:
            # Fallback for testing
            logger.warning("Transcription service not available, using placeholder")
            return {
                "text": f"[Placeholder transcription for {Path(audio_path).name}]",
                "segments": [
                    {
                        "start": 0.0,
                        "end": 5.0,
                        "text": "This is a placeholder transcription."
                    }
                ]
            }
    
    def _chunk_text(self, text: str, method: str = "sentences", 
                   max_size: int = 500, overlap: int = 200,
                   language: str = "en") -> List[Dict[str, Any]]:
        """Chunk text using the chunking service."""
        chunk_options = {
            "method": method,
            "max_size": max_size,
            "overlap": overlap,
            "language": language
        }
        
        chunks = self.chunking_service.chunk_text(text, chunk_options)
        return [{"text": chunk, "metadata": {"method": method}} for chunk in chunks]
    
    def _analyze_content(self, content: str, chunks: Optional[List[Dict]], 
                        api_name: str, api_key: Optional[str],
                        custom_prompt: Optional[str], system_prompt: Optional[str],
                        summarize_recursively: bool = False) -> str:
        """Analyze/summarize content using LLM."""
        
        # Prepare prompt
        if custom_prompt:
            user_prompt = custom_prompt + "\n\n" + content
        else:
            user_prompt = f"Please summarize the following audio transcription:\n\n{content}"
        
        # If chunking and recursive summarization
        if chunks and summarize_recursively and len(chunks) > 1:
            # Summarize each chunk first
            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                chunk_text = chunk.get("text", "")
                if chunk_text:
                    chunk_prompt = f"Summarize this section:\n\n{chunk_text}"
                    summary = chat_api_call(
                        api_name=api_name,
                        input_data=chunk_prompt,
                        prompt="",
                        temp=0.7,
                        system_message=system_prompt
                    )
                    chunk_summaries.append(summary)
            
            # Then summarize the summaries
            combined = "\n\n".join(chunk_summaries)
            final_prompt = f"Combine and summarize these section summaries:\n\n{combined}"
            return chat_api_call(
                api_name=api_name,
                input_data=final_prompt,
                prompt="",
                temp=0.7,
                system_message=system_prompt
            )
        else:
            # Direct summarization
            return chat_api_call(
                api_name=api_name,
                input_data=content,
                prompt=custom_prompt or "Please summarize this audio transcription.",
                temp=0.7,
                system_message=system_prompt
            )
    
    def _store_in_database(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Store processing results in the media database."""
        if not self.media_db:
            return {"message": "No database available"}
        
        try:
            # Prepare media data
            media_data = {
                "url": result.get("input_ref", ""),
                "title": result["metadata"].get("title", "Untitled"),
                "media_type": result.get("media_type", "audio"),
                "content": result.get("analysis") or result.get("content", ""),
                "author": result["metadata"].get("author", "Unknown"),
                "ingestion_date": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Add media entry
            media_id, _, _ = self.media_db.add_media_with_keywords(**media_data)
            
            # Store chunks if available
            if result.get("chunks"):
                for i, chunk in enumerate(result["chunks"]):
                    self.media_db.add_media_chunk(
                        media_id=media_id,
                        chunk_text=chunk.get("text", ""),
                        chunk_index=i,
                        metadata=json.dumps(chunk.get("metadata", {}))
                    )
            
            return {"id": media_id, "message": "Stored successfully"}
            
        except Exception as e:
            logger.error(f"Database storage error: {str(e)}")
            return {"message": f"Storage failed: {str(e)}"}
    
    def _get_filename_from_response(self, response: requests.Response, url: str) -> str:
        """Extract filename from response headers or URL."""
        content_disposition = response.headers.get('content-disposition')
        if content_disposition:
            parts = content_disposition.split('filename=')
            if len(parts) > 1:
                filename = parts[1].strip('"\' ')
                if filename:
                    return sanitize_filename(filename)
        
        # Fallback to URL path
        path = Path(urlparse(url).path)
        if path.name:
            return sanitize_filename(path.name)
        
        # Generate unique filename
        return f"audio_{uuid.uuid4().hex[:8]}.mp3"
    
    def _extract_time_range(self, audio_path: str, output_dir: str, 
                           start_time: Optional[str] = None, 
                           end_time: Optional[str] = None) -> str:
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
        ffmpeg_cmd = shutil.which('ffmpeg')
        if not ffmpeg_cmd:
            # Try common locations
            for cmd in ['/usr/bin/ffmpeg', '/usr/local/bin/ffmpeg', '/opt/homebrew/bin/ffmpeg']:
                if os.path.exists(cmd):
                    ffmpeg_cmd = cmd
                    break
        
        if not ffmpeg_cmd:
            logger.warning("ffmpeg not found, skipping time range extraction")
            return audio_path
        
        # Generate output filename
        base_name = Path(audio_path).stem
        suffix = f"_trim_{start_time or '0'}_{end_time or 'end'}".replace(':', '-')
        output_path = os.path.join(output_dir, f"{base_name}{suffix}.mp3")
        
        # Build ffmpeg command
        command = [ffmpeg_cmd, '-i', audio_path]
        
        # Add start time if specified
        if start_time:
            command.extend(['-ss', start_time])
        
        # Add duration if end time is specified
        if end_time:
            if start_time:
                # Calculate duration from start to end
                # This is a simplified approach - ideally we'd parse the times properly
                command.extend(['-to', end_time])
            else:
                command.extend(['-t', end_time])
        
        # Output options
        command.extend([
            '-acodec', 'libmp3lame',
            '-ab', '192k',
            '-ar', '44100',
            '-y',  # Overwrite
            output_path
        ])
        
        try:
            logger.debug(f"Running ffmpeg command: {' '.join(command)}")
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True
            )
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
