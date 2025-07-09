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
from pathlib import Path
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse
import subprocess

# Local imports
from .audio_processing import LocalAudioProcessor, AudioProcessingError
from ..config import get_cli_setting
from ..DB.Client_Media_DB_v2 import MediaDatabase
from ..Utils.text import sanitize_filename

# Optional imports
try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False
    logging.warning("yt-dlp not available. Video downloading will be disabled.")

logger = logging.getLogger(__name__)


class VideoProcessingError(Exception):
    """Base exception for video processing errors."""
    pass


class VideoDownloadError(VideoProcessingError):
    """Raised when video download fails."""
    pass


class LocalVideoProcessor:
    """Handles local video processing including download, audio extraction, and analysis."""
    
    def __init__(self, media_db: Optional[MediaDatabase] = None):
        """
        Initialize the video processor.
        
        Args:
            media_db: Optional MediaDatabase instance for storage
        """
        self.media_db = media_db
        self.audio_processor = LocalAudioProcessor(media_db)
        self.max_file_size_mb = get_cli_setting('media_processing.max_video_file_size_mb', 2000)
        self.max_file_size = self.max_file_size_mb * 1024 * 1024
        
    def download_video(
        self,
        url: str,
        output_dir: str,
        download_video_flag: bool = False,
        use_cookies: bool = False,
        cookies: Optional[Dict] = None
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
        if not YT_DLP_AVAILABLE:
            raise VideoDownloadError("yt-dlp is not installed")
        
        try:
            # Base configuration
            ydl_opts = {
                'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
                'quiet': True,
                'no_warnings': True,
                'noplaylist': True,
            }
            
            # Add cookies if provided
            if use_cookies and cookies:
                if isinstance(cookies, str):
                    # If cookies is a file path
                    if os.path.exists(cookies):
                        ydl_opts['cookiefile'] = cookies
                    else:
                        # Try to parse as JSON
                        try:
                            cookie_dict = json.loads(cookies)
                            # Create temp cookie file
                            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                                for name, value in cookie_dict.items():
                                    f.write(f"{name}={value}\n")
                                ydl_opts['cookiefile'] = f.name
                        except json.JSONDecodeError:
                            logger.warning("Invalid cookie format")
                elif isinstance(cookies, dict):
                    # Create temp cookie file from dict
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                        for name, value in cookies.items():
                            f.write(f"{name}={value}\n")
                        ydl_opts['cookiefile'] = f.name
            
            # Configure format based on download type
            if download_video_flag:
                # Download best quality video with audio
                ydl_opts['format'] = 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'
                ydl_opts['merge_output_format'] = 'mp4'
            else:
                # Extract audio only
                ydl_opts['format'] = 'bestaudio[ext=m4a]/bestaudio/best'
                ydl_opts['postprocessors'] = [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }]
            
            # Add ffmpeg location if specified
            ffmpeg_path = get_cli_setting('media_processing.ffmpeg_path')
            if ffmpeg_path:
                ydl_opts['ffmpeg_location'] = ffmpeg_path
            
            # Extract info first to check file size
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                
                # Check file size if available
                filesize = info.get('filesize') or info.get('filesize_approx', 0)
                if filesize and filesize > self.max_file_size:
                    raise VideoDownloadError(
                        f"File size ({filesize / (1024*1024):.2f} MB) exceeds limit"
                    )
            
            # Perform download
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                
                # Get the actual filename
                filename = ydl.prepare_filename(info)
                
                # Handle different output formats
                if not download_video_flag:
                    # Audio extraction changes extension
                    base_name = os.path.splitext(filename)[0]
                    audio_path = base_name + '.mp3'
                    if os.path.exists(audio_path):
                        return audio_path
                
                if os.path.exists(filename):
                    return filename
                
                # Try to find the file with different extensions
                base_name = os.path.splitext(filename)[0]
                for ext in ['.mp4', '.mp3', '.m4a', '.webm', '.mkv']:
                    test_path = base_name + ext
                    if os.path.exists(test_path):
                        return test_path
                
                raise VideoDownloadError("Downloaded file not found")
                
        except Exception as e:
            logger.error(f"Video download error: {str(e)}")
            raise VideoDownloadError(f"Download failed: {str(e)}") from e
        finally:
            # Clean up temporary cookie file if created
            if 'cookiefile' in ydl_opts and ydl_opts['cookiefile'].startswith(tempfile.gettempdir()):
                try:
                    os.unlink(ydl_opts['cookiefile'])
                except:
                    pass
    
    def extract_metadata(self, url: str, use_cookies: bool = False, 
                        cookies: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """Extract metadata from video URL without downloading."""
        if not YT_DLP_AVAILABLE:
            return None
        
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,  # Get full info
                'skip_download': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                metadata = {
                    'title': info.get('title'),
                    'uploader': info.get('uploader'),
                    'upload_date': info.get('upload_date'),
                    'view_count': info.get('view_count'),
                    'like_count': info.get('like_count'),
                    'duration': info.get('duration'),
                    'tags': info.get('tags', []),
                    'description': info.get('description'),
                    'webpage_url': info.get('webpage_url', url),
                    'thumbnail': info.get('thumbnail'),
                }
                
                return metadata
                
        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            return None
    
    def process_videos(
        self,
        inputs: List[str],
        download_video_flag: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process multiple video inputs.
        
        Args:
            inputs: List of video URLs or local file paths
            download_video_flag: If True, keep video file; if False, extract audio only
            **kwargs: Additional arguments passed to audio processing
            
        Returns:
            Dict with processing results
        """
        results = []
        errors = []
        
        with tempfile.TemporaryDirectory(prefix="video_proc_") as temp_dir:
            for input_item in inputs:
                try:
                    result = self._process_single_video(
                        input_item=input_item,
                        temp_dir=temp_dir,
                        download_video_flag=download_video_flag,
                        **kwargs
                    )
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing video {input_item}: {str(e)}")
                    error_result = {
                        "status": "Error",
                        "input_ref": input_item,
                        "error": str(e),
                        "media_type": "video"
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
    
    def _process_single_video(
        self,
        input_item: str,
        temp_dir: str,
        download_video_flag: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Process a single video file or URL."""
        
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
            "warnings": []
        }
        
        try:
            # Determine if input is URL or local file
            is_url = urlparse(input_item).scheme in ('http', 'https')
            
            if is_url:
                # Extract metadata
                metadata = self.extract_metadata(
                    input_item,
                    kwargs.get("use_cookies", False),
                    kwargs.get("cookies")
                )
                if metadata:
                    result["metadata"] = metadata
                
                # Download video/audio
                logger.info(f"Downloading from URL: {input_item}")
                downloaded_path = self.download_video(
                    url=input_item,
                    output_dir=temp_dir,
                    download_video_flag=download_video_flag,
                    use_cookies=kwargs.get("use_cookies", False),
                    cookies=kwargs.get("cookies")
                )
                
                if not downloaded_path:
                    raise VideoDownloadError("Download failed")
                
                processing_path = downloaded_path
                
            else:
                # Local file
                if not os.path.exists(input_item):
                    raise FileNotFoundError(f"File not found: {input_item}")
                
                processing_path = input_item
                result["metadata"]["title"] = Path(input_item).stem
            
            # Check if we have audio file or need to extract audio from video
            file_ext = Path(processing_path).suffix.lower()
            audio_extensions = {'.mp3', '.m4a', '.wav', '.flac', '.ogg', '.aac'}
            
            if file_ext in audio_extensions:
                # Already audio file, process directly
                audio_path = processing_path
            else:
                # Extract audio from video
                logger.info(f"Extracting audio from video: {processing_path}")
                audio_path = self._extract_audio_from_video(processing_path, temp_dir)
            
            # Process audio using audio processor
            audio_result = self.audio_processor._process_single_audio(
                input_item=audio_path,
                processing_dir=temp_dir,
                **kwargs
            )
            
            # Merge results
            result.update({
                "processing_source": processing_path,
                "content": audio_result.get("content"),
                "segments": audio_result.get("segments"),
                "chunks": audio_result.get("chunks"),
                "analysis": audio_result.get("analysis"),
                "analysis_details": audio_result.get("analysis_details"),
                "warnings": audio_result.get("warnings", []),
            })
            
            # Update metadata if not already set
            if not result["metadata"].get("title") and audio_result.get("metadata", {}).get("title"):
                result["metadata"]["title"] = audio_result["metadata"]["title"]
            
            # Store in database if available
            if self.media_db and result["content"]:
                db_result = self._store_in_database(result)
                result["db_id"] = db_result.get("id")
                result["db_message"] = db_result.get("message", "Stored successfully")
            
            result["status"] = "Success" if not result["warnings"] else "Warning"
            
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
                            dest_path = downloads_dir / f"{base_name}_{counter}{extension}"
                            counter += 1
                    
                    # Move the file
                    shutil.move(processing_path, str(dest_path))
                    logger.info(f"Moved video file to: {dest_path}")
                    result["saved_video_path"] = str(dest_path)
                    result["warnings"].append(f"Video file saved to Downloads folder: {dest_path.name}")
                except Exception as e:
                    logger.error(f"Failed to move video file to Downloads: {str(e)}")
                    result["warnings"].append(f"Could not save video file: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}", exc_info=True)
            result["status"] = "Error"
            result["error"] = str(e)
        
        return result
    
    def _extract_audio_from_video(self, video_path: str, output_dir: str) -> str:
        """Extract audio track from video file."""
        
        # Find ffmpeg
        ffmpeg_cmd = self._find_ffmpeg()
        
        # Output path
        base_name = Path(video_path).stem
        audio_path = os.path.join(output_dir, f"{base_name}_audio.mp3")
        
        # Extract audio as MP3
        command = [
            ffmpeg_cmd,
            '-i', video_path,
            '-vn',  # No video
            '-acodec', 'libmp3lame',
            '-ab', '192k',  # Audio bitrate
            '-ar', '44100',  # Sample rate
            '-y',  # Overwrite
            audio_path
        ]
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"Extracted audio to: {audio_path}")
            return audio_path
            
        except subprocess.CalledProcessError as e:
            raise VideoProcessingError(
                f"Failed to extract audio: {e.stderr}"
            ) from e
    
    def _find_ffmpeg(self) -> str:
        """Find ffmpeg executable."""
        # Check config first
        ffmpeg_path = get_cli_setting('media_processing.ffmpeg_path')
        if ffmpeg_path and os.path.exists(ffmpeg_path):
            return ffmpeg_path
        
        # Check common locations
        import shutil
        ffmpeg = shutil.which('ffmpeg')
        if ffmpeg:
            return ffmpeg
        
        raise FileNotFoundError("ffmpeg not found")
    
    def _store_in_database(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Store processing results in the media database."""
        if not self.media_db:
            return {"message": "No database available"}
        
        try:
            # Prepare media data
            media_data = {
                "url": result.get("input_ref", ""),
                "title": result["metadata"].get("title", "Untitled"),
                "type": "video",
                "content": result.get("analysis") or result.get("content", ""),
                "author": result["metadata"].get("uploader", "Unknown"),
                "ingestion_date": None  # Will use current time
            }
            
            # Store additional metadata
            if result.get("metadata"):
                media_data["metadata"] = json.dumps(result["metadata"])
            
            # Add media entry
            media_id = self.media_db.add_media(**media_data)
            
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


# Convenience function for backwards compatibility
def process_videos(**kwargs) -> Dict[str, Any]:
    """Process videos using LocalVideoProcessor."""
    processor = LocalVideoProcessor()
    return processor.process_videos(**kwargs)