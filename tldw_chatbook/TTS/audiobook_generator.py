# audiobook_generator.py
# Description: AudioBook and Podcast generation functionality for the TTS module
#
# Imports
import asyncio
import re
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import json
from loguru import logger

# Third-party imports
from pydantic import BaseModel, Field

# Local imports
from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS import get_tts_service
from tldw_chatbook.TTS.audio_service import get_audio_service
# Text processing is handled by AdvancedTextProcessor
from tldw_chatbook.TTS.text_processor_advanced import AdvancedTextProcessor
from ..Metrics.metrics_logger import log_counter, log_histogram

#######################################################################################################################
#
# Data Models

@dataclass
class Chapter:
    """Represents a book chapter"""
    number: int
    title: str
    content: str
    start_position: int
    end_position: int
    voice: Optional[str] = None
    narrator_notes: Optional[str] = None
    estimated_duration: Optional[float] = None
    audio_file: Optional[Path] = None

@dataclass
class Character:
    """Represents a character with assigned voice"""
    name: str
    voice: str
    description: Optional[str] = None
    voice_settings: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AudioBookMetadata:
    """Metadata for the audiobook"""
    title: str
    author: str
    narrator: str
    language: str = "en"
    genre: Optional[str] = None
    publication_date: Optional[str] = None
    copyright: Optional[str] = None
    description: Optional[str] = None
    cover_image: Optional[Path] = None
    isbn: Optional[str] = None
    total_duration: Optional[float] = None
    chapter_count: int = 0

class AudioBookRequest(BaseModel):
    """Request model for audiobook generation"""
    content: str
    title: str
    author: str = "Unknown"
    narrator_voice: str = "alloy"
    provider: str = "openai"
    model: str = "tts-1"
    output_format: str = "mp3"
    chapter_detection: bool = True
    multi_voice: bool = False
    character_voices: Optional[Dict[str, str]] = Field(default_factory=dict)
    voice_settings: Optional[Dict[str, Any]] = Field(default_factory=dict)
    background_music: Optional[Path] = None
    music_volume: float = 0.1
    chapter_pause_duration: float = 2.0
    paragraph_pause_duration: float = 0.5
    sentence_pause_duration: float = 0.3
    max_chunk_size: int = 4000
    enable_ssml: bool = False
    normalize_audio: bool = True
    target_db: float = -20.0
    
class AudioBookProgress(BaseModel):
    """Progress tracking for audiobook generation"""
    total_chapters: int = 0
    completed_chapters: int = 0
    current_chapter: Optional[str] = None
    total_characters: int = 0
    processed_characters: int = 0
    estimated_duration: float = 0.0
    actual_duration: float = 0.0
    start_time: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    status: str = "idle"
    errors: List[str] = Field(default_factory=list)

#######################################################################################################################
#
# Chapter Detection

class ChapterDetector:
    """Detects and extracts chapters from text content"""
    
    # Common chapter patterns
    CHAPTER_PATTERNS = [
        r'^Chapter\s+(\d+)(?:\s*[:\-]\s*(.*))?$',  # Chapter 1: Title
        r'^CHAPTER\s+(\d+)(?:\s*[:\-]\s*(.*))?$',  # CHAPTER 1: Title
        r'^Ch\.?\s*(\d+)(?:\s*[:\-]\s*(.*))?$',    # Ch. 1: Title
        r'^(\d+)\.?\s+(.+)$',                       # 1. Title
        r'^Part\s+(\d+)(?:\s*[:\-]\s*(.*))?$',     # Part 1: Title
        r'^Section\s+(\d+)(?:\s*[:\-]\s*(.*))?$',  # Section 1: Title
    ]
    
    # Roman numeral patterns
    ROMAN_PATTERNS = [
        r'^Chapter\s+([IVXLCDM]+)(?:\s*[:\-]\s*(.*))?$',
        r'^([IVXLCDM]+)\.?\s+(.+)$',
    ]
    
    @classmethod
    def detect_chapters(cls, content: str) -> List[Chapter]:
        """
        Detect chapters in the content.
        
        Args:
            content: Full text content
            
        Returns:
            List of detected chapters
        """
        lines = content.split('\n')
        chapters = []
        current_chapter_content = []
        current_chapter_start = 0
        chapter_number = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                if current_chapter_content:
                    current_chapter_content.append("")
                continue
            
            # Check if line matches any chapter pattern
            chapter_match = cls._match_chapter_pattern(line)
            
            if chapter_match:
                # Save previous chapter if exists
                if current_chapter_content:
                    chapter_text = '\n'.join(current_chapter_content)
                    if chapter_text.strip():
                        chapters.append(Chapter(
                            number=chapter_number,
                            title=f"Chapter {chapter_number}",
                            content=chapter_text,
                            start_position=current_chapter_start,
                            end_position=i - 1
                        ))
                
                # Start new chapter
                chapter_number += 1
                current_chapter_content = []
                current_chapter_start = i
                
                # Extract chapter title if present
                if chapter_match.get('title'):
                    chapters.append(Chapter(
                        number=chapter_number,
                        title=chapter_match['title'],
                        content="",  # Will be filled later
                        start_position=i,
                        end_position=i
                    ))
            else:
                current_chapter_content.append(line)
        
        # Don't forget the last chapter
        if current_chapter_content:
            chapter_text = '\n'.join(current_chapter_content)
            if chapter_text.strip():
                if chapters and not chapters[-1].content:
                    # Fill the last chapter's content
                    chapters[-1].content = chapter_text
                    chapters[-1].end_position = len(lines) - 1
                else:
                    chapters.append(Chapter(
                        number=chapter_number + 1,
                        title=f"Chapter {chapter_number + 1}",
                        content=chapter_text,
                        start_position=current_chapter_start,
                        end_position=len(lines) - 1
                    ))
        
        # If no chapters detected, treat entire content as one chapter
        if not chapters:
            chapters.append(Chapter(
                number=1,
                title="Full Content",
                content=content,
                start_position=0,
                end_position=len(lines) - 1
            ))
        
        logger.info(f"Detected {len(chapters)} chapters")
        return chapters
    
    @classmethod
    def _match_chapter_pattern(cls, line: str) -> Optional[Dict[str, str]]:
        """Match line against chapter patterns"""
        # Try numeric patterns
        for pattern in cls.CHAPTER_PATTERNS:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                groups = match.groups()
                return {
                    'number': groups[0],
                    'title': groups[1] if len(groups) > 1 and groups[1] else f"Chapter {groups[0]}"
                }
        
        # Try Roman numeral patterns
        for pattern in cls.ROMAN_PATTERNS:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                groups = match.groups()
                return {
                    'number': groups[0],
                    'title': groups[1] if len(groups) > 1 and groups[1] else f"Chapter {groups[0]}"
                }
        
        return None

#######################################################################################################################
#
# Character Detection and Voice Assignment

class CharacterDetector:
    """Detects characters and dialogue in text"""
    
    # Dialogue patterns
    DIALOGUE_PATTERNS = [
        r'"([^"]+)"(?:\s+said\s+(\w+))?',  # "Hello," said John
        r'"([^"]+)"(?:\s+(\w+)\s+said)?',  # "Hello," John said
        r'(\w+)\s+said,?\s*"([^"]+)"',     # John said, "Hello"
        r'"([^"]+),"?\s+(\w+)\s+\w+ed',    # "Hello," John replied/asked/etc
    ]
    
    @classmethod
    def detect_characters(cls, content: str) -> List[str]:
        """
        Detect character names from dialogue.
        
        Args:
            content: Text content
            
        Returns:
            List of unique character names
        """
        characters = set()
        
        for pattern in cls.DIALOGUE_PATTERNS:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                # Extract character name from appropriate group
                for group in match.groups():
                    if group and group[0].isupper() and len(group) > 1:
                        # Likely a character name
                        characters.add(group)
        
        # Filter out common words that might be mistaken for names
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at'}
        characters = {c for c in characters if c.lower() not in common_words}
        
        logger.info(f"Detected {len(characters)} unique characters: {characters}")
        return sorted(list(characters))
    
    @classmethod
    def assign_dialogue_voices(
        cls, 
        content: str, 
        character_voices: Dict[str, str],
        narrator_voice: str
    ) -> List[Tuple[str, str]]:
        """
        Split content into segments with assigned voices.
        
        Args:
            content: Text content
            character_voices: Mapping of character names to voices
            narrator_voice: Default narrator voice
            
        Returns:
            List of (text, voice) tuples
        """
        segments = []
        last_end = 0
        
        # Find all dialogue sections
        dialogue_sections = []
        for pattern in cls.DIALOGUE_PATTERNS:
            matches = re.finditer(pattern, content)
            for match in matches:
                dialogue_sections.append({
                    'start': match.start(),
                    'end': match.end(),
                    'text': match.group(0),
                    'character': cls._extract_character_from_match(match)
                })
        
        # Sort by position
        dialogue_sections.sort(key=lambda x: x['start'])
        
        # Create segments
        for section in dialogue_sections:
            # Add narrator text before dialogue
            if section['start'] > last_end:
                narrator_text = content[last_end:section['start']].strip()
                if narrator_text:
                    segments.append((narrator_text, narrator_voice))
            
            # Add dialogue with character voice
            character = section['character']
            voice = character_voices.get(character, narrator_voice)
            segments.append((section['text'], voice))
            
            last_end = section['end']
        
        # Add remaining narrator text
        if last_end < len(content):
            narrator_text = content[last_end:].strip()
            if narrator_text:
                segments.append((narrator_text, narrator_voice))
        
        return segments
    
    @classmethod
    def _extract_character_from_match(cls, match: re.Match) -> Optional[str]:
        """Extract character name from regex match"""
        for group in match.groups():
            if group and group[0].isupper() and len(group) > 1:
                # Check if it's likely a character name
                if not group.lower() in {'the', 'a', 'an', 'and', 'or', 'but'}:
                    return group
        return None

#######################################################################################################################
#
# AudioBook Generator

class AudioBookGenerator:
    """Main audiobook generation engine"""
    
    def __init__(self, tts_service=None):
        self.tts_service = tts_service
        self.audio_service = get_audio_service()
        self.text_processor = AdvancedTextProcessor()
        self.progress = AudioBookProgress()
        self._generation_task = None
        self._cancel_event = asyncio.Event()
        
    async def initialize(self):
        """Initialize the generator with TTS service"""
        if not self.tts_service:
            self.tts_service = await get_tts_service()
    
    async def generate_audiobook(
        self,
        request: AudioBookRequest,
        progress_callback: Optional[callable] = None
    ) -> Path:
        """
        Generate a complete audiobook.
        
        Args:
            request: AudioBook generation request
            progress_callback: Optional callback for progress updates
            
        Returns:
            Path to the generated audiobook file
        """
        self.progress = AudioBookProgress(
            status="initializing",
            start_time=datetime.now()
        )
        
        try:
            # Detect chapters if enabled
            if request.chapter_detection:
                chapters = ChapterDetector.detect_chapters(request.content)
            else:
                chapters = [Chapter(
                    number=1,
                    title=request.title,
                    content=request.content,
                    start_position=0,
                    end_position=len(request.content.split('\n'))
                )]
            
            self.progress.total_chapters = len(chapters)
            
            # Detect characters if multi-voice is enabled
            characters = []
            if request.multi_voice:
                character_names = CharacterDetector.detect_characters(request.content)
                self.progress.total_characters = len(character_names)
                
                # Create character objects with assigned voices
                for i, name in enumerate(character_names):
                    voice = request.character_voices.get(
                        name, 
                        self._get_default_character_voice(i, request.provider)
                    )
                    characters.append(Character(name=name, voice=voice))
            
            # Estimate duration
            total_words = len(request.content.split())
            words_per_minute = 150  # Average speaking rate
            estimated_minutes = total_words / words_per_minute
            self.progress.estimated_duration = estimated_minutes * 60
            self.progress.estimated_completion = datetime.now() + timedelta(seconds=self.progress.estimated_duration)
            
            # Store chapters for M4B metadata
            self._chapters = chapters
            self._chapter_audio_files = []
            
            # Generate audio for each chapter
            chapter_audio_files = []
            
            for i, chapter in enumerate(chapters):
                if self._cancel_event.is_set():
                    raise asyncio.CancelledError("Generation cancelled by user")
                
                self.progress.current_chapter = chapter.title
                self.progress.status = f"generating_chapter_{i+1}"
                
                if progress_callback:
                    await progress_callback(self.progress)
                
                # Generate chapter audio
                chapter_audio = await self._generate_chapter_audio(
                    chapter, 
                    request,
                    characters
                )
                
                chapter_audio_files.append(chapter_audio)
                self._chapter_audio_files.append(chapter_audio)  # Store for M4B creation
                self.progress.completed_chapters += 1
                
                # Update actual duration
                duration = await self._get_audio_duration(chapter_audio)
                self.progress.actual_duration += duration
            
            # Combine all chapters into single audiobook
            self.progress.status = "combining_chapters"
            if progress_callback:
                await progress_callback(self.progress)
            
            output_path = await self._combine_chapters(
                chapter_audio_files,
                request,
                chapters
            )
            
            # Add metadata
            self.progress.status = "adding_metadata"
            if progress_callback:
                await progress_callback(self.progress)
            
            metadata = AudioBookMetadata(
                title=request.title,
                author=request.author,
                narrator=request.narrator_voice,
                language="en",
                total_duration=self.progress.actual_duration,
                chapter_count=len(chapters)
            )
            
            output_path = await self._add_metadata(output_path, metadata)
            
            # Clean up temporary files
            for chapter_file in chapter_audio_files:
                if chapter_file.exists():
                    chapter_file.unlink()
            
            self.progress.status = "completed"
            if progress_callback:
                await progress_callback(self.progress)
            
            logger.info(f"AudioBook generation completed: {output_path}")
            return output_path
            
        except Exception as e:
            self.progress.status = "failed"
            self.progress.errors.append(str(e))
            if progress_callback:
                await progress_callback(self.progress)
            logger.error(f"AudioBook generation failed: {e}")
            raise
    
    async def _generate_chapter_audio(
        self,
        chapter: Chapter,
        request: AudioBookRequest,
        characters: List[Character]
    ) -> Path:
        """Generate audio for a single chapter"""
        logger.info(f"Generating audio for {chapter.title}")
        
        # Create temporary file for chapter
        temp_file = tempfile.NamedTemporaryFile(
            suffix=f".{request.output_format}",
            prefix=f"chapter_{chapter.number}_",
            delete=False
        )
        chapter_path = Path(temp_file.name)
        temp_file.close()
        
        # Split chapter into segments if multi-voice
        if request.multi_voice and characters:
            character_voices = {c.name: c.voice for c in characters}
            segments = CharacterDetector.assign_dialogue_voices(
                chapter.content,
                character_voices,
                request.narrator_voice
            )
        else:
            segments = [(chapter.content, request.narrator_voice)]
        
        # Generate audio for each segment
        segment_audio_data = []
        
        for text, voice in segments:
            if not text.strip():
                continue
            
            # Process text with advanced features
            if request.enable_ssml and request.provider in ["elevenlabs"]:
                # Use SSML for providers that support it
                processed_text = self.text_processor.process_text(
                    text,
                    enable_ssml=True,
                    target_provider=request.provider,
                    expand_abbreviations=True,
                    apply_pronunciation=True,
                    detect_emotions=True,
                    smart_punctuation=True
                )
            else:
                # Use provider-optimized plain text
                processed_text = self.text_processor.format_for_tts(text, request.provider)
            
            # Create TTS request
            tts_request = OpenAISpeechRequest(
                model=request.model,
                input=processed_text,
                voice=voice,
                response_format=request.output_format,
                speed=1.0
            )
            
            # Map provider to internal model ID
            internal_model_id = self._get_internal_model_id(request.provider, request.model)
            
            # Generate audio
            audio_data = b""
            async for chunk in self.tts_service.generate_audio_stream(tts_request, internal_model_id):
                audio_data += chunk
            
            segment_audio_data.append(audio_data)
            
            # Track TTS generation metrics
            log_counter("tts_generation_total", labels={
                "provider": request.provider,
                "model": request.model,
                "voice": request.voice,
                "format": request.output_format
            })
            log_histogram("tts_character_count", len(normalized_text), labels={
                "provider": request.provider,
                "model": request.model
            })
        
        # Combine segments with appropriate pauses
        combined_audio = await self._combine_segments(
            segment_audio_data,
            request.output_format,
            request.sentence_pause_duration
        )
        
        # Apply post-processing if enabled
        if request.normalize_audio:
            combined_audio = await self._normalize_audio(
                combined_audio,
                request.output_format,
                request.target_db
            )
        
        # Write to file
        with open(chapter_path, 'wb') as f:
            f.write(combined_audio)
        
        chapter.audio_file = chapter_path
        return chapter_path
    
    async def _combine_segments(
        self,
        segments: List[bytes],
        format: str,
        pause_duration: float
    ) -> bytes:
        """Combine audio segments with pauses"""
        if len(segments) == 1:
            return segments[0]
        
        # Generate silence for pauses
        silence = await self._generate_silence(pause_duration, format)
        
        # Combine segments with silence between
        combined = b""
        for i, segment in enumerate(segments):
            combined += segment
            if i < len(segments) - 1:
                combined += silence
        
        return combined
    
    async def _combine_chapters(
        self,
        chapter_files: List[Path],
        request: AudioBookRequest,
        chapters: List[Chapter]
    ) -> Path:
        """Combine chapter audio files into final audiobook"""
        output_file = tempfile.NamedTemporaryFile(
            suffix=f".{request.output_format}",
            prefix=f"{request.title.replace(' ', '_')}_",
            delete=False
        )
        output_path = Path(output_file.name)
        output_file.close()
        
        # Generate chapter pause
        chapter_pause = await self._generate_silence(
            request.chapter_pause_duration,
            request.output_format
        )
        
        # Combine all chapters
        with open(output_path, 'wb') as out:
            for i, chapter_file in enumerate(chapter_files):
                with open(chapter_file, 'rb') as f:
                    out.write(f.read())
                
                # Add pause between chapters
                if i < len(chapter_files) - 1:
                    out.write(chapter_pause)
        
        return output_path
    
    async def _add_metadata(self, audio_path: Path, metadata: AudioBookMetadata) -> Path:
        """Add metadata to the audiobook file"""
        # Get file extension
        file_ext = audio_path.suffix[1:].lower()
        
        # For M4B, use the special chapter-aware method
        if file_ext == "m4b" and hasattr(self, '_chapter_audio_files'):
            # Create M4B with chapters
            final_name = f"{metadata.title} - {metadata.author}.m4b"
            final_path = audio_path.parent / final_name
            
            # Extract chapter titles from chapters
            chapter_titles = [ch.title for ch in self._chapters] if hasattr(self, '_chapters') else []
            
            # Create metadata dict
            metadata_dict = {
                "title": metadata.title,
                "artist": metadata.narrator,
                "album_artist": metadata.author,
                "album": metadata.title,
                "genre": metadata.genre or "Audiobook",
                "date": metadata.publication_date or "",
                "comment": metadata.description or "",
            }
            
            # Create M4B with chapters
            success = await self.audio_service.create_m4b_with_chapters(
                audio_files=[str(f) for f in self._chapter_audio_files],
                chapter_titles=chapter_titles,
                output_path=str(final_path),
                metadata=metadata_dict
            )
            
            if success:
                # Remove the temporary combined file
                audio_path.unlink()
                return final_path
        
        # For other formats, just rename with metadata in filename
        final_name = f"{metadata.title} - {metadata.author}.{file_ext}"
        final_path = audio_path.parent / final_name
        audio_path.rename(final_path)
        return final_path
    
    async def _generate_silence(self, duration: float, format: str) -> bytes:
        """Generate silence of specified duration"""
        try:
            # Try to use pydub for proper silence generation
            from pydub import AudioSegment
            import io
            
            # Create silence segment (duration in milliseconds)
            silence = AudioSegment.silent(duration=int(duration * 1000))
            
            # Export to requested format
            buffer = io.BytesIO()
            silence.export(buffer, format=format)
            return buffer.getvalue()
            
        except ImportError:
            logger.warning("pydub not installed. Install with: pip install pydub")
            # Fallback: return minimal valid audio data
            if format == "mp3":
                # Minimal MP3 frame header for silence
                # FF FB = sync word + MPEG-1 Layer 3
                # 90 = 128kbps, 44.1kHz, no padding, stereo
                # 00 = no data
                frame = b'\xff\xfb\x90\x00'
                # Approximate number of frames needed
                # MP3 frame is ~26ms at 128kbps
                frames_needed = int(duration * 38.46)  # 1000ms / 26ms
                return frame * frames_needed
            elif format == "wav":
                # Minimal WAV header + silence
                import struct
                sample_rate = 44100
                num_channels = 1
                bits_per_sample = 16
                num_samples = int(sample_rate * duration)
                
                # WAV header
                header = b'RIFF'
                header += struct.pack('<I', 36 + num_samples * 2)  # file size - 8
                header += b'WAVEfmt '
                header += struct.pack('<IHHIIHH', 16, 1, num_channels, sample_rate,
                                    sample_rate * num_channels * bits_per_sample // 8,
                                    num_channels * bits_per_sample // 8, bits_per_sample)
                header += b'data'
                header += struct.pack('<I', num_samples * 2)
                
                # Silent samples (16-bit zeros)
                silence_data = b'\x00\x00' * num_samples
                
                return header + silence_data
            else:
                # For other formats, return empty (will likely cause issues)
                logger.error(f"Cannot generate silence for format {format} without pydub")
                return b''
    
    async def _normalize_audio(self, audio_data: bytes, format: str, target_db: float) -> bytes:
        """Normalize audio to target dB level"""
        try:
            from pydub import AudioSegment
            import io
            import numpy as np
            
            # Load audio from bytes
            audio = AudioSegment.from_file(io.BytesIO(audio_data), format=format)
            
            # Calculate current dBFS (decibels relative to full scale)
            current_db = audio.dBFS
            
            # If audio is silent or very quiet, skip normalization
            if current_db == -float('inf') or current_db < -60:
                logger.warning("Audio is too quiet to normalize effectively")
                return audio_data
            
            # Calculate gain adjustment needed
            gain_db = target_db - current_db
            
            # Apply gain (limit to reasonable range to prevent distortion)
            gain_db = max(-20, min(20, gain_db))  # Limit gain adjustment to ±20dB
            
            # Apply the gain
            normalized = audio.apply_gain(gain_db)
            
            # Export back to bytes
            buffer = io.BytesIO()
            normalized.export(buffer, format=format)
            return buffer.getvalue()
            
        except ImportError:
            logger.warning("Audio normalization requires pydub and numpy. Install with: pip install pydub numpy")
            return audio_data  # Return unchanged
        except Exception as e:
            logger.error(f"Failed to normalize audio: {e}")
            return audio_data  # Return unchanged on error
    
    async def _get_audio_duration(self, audio_path: Path) -> float:
        """Get duration of audio file in seconds"""
        try:
            # Try mutagen first (lightweight, no dependencies)
            from mutagen import File
            
            audio_file = File(str(audio_path))
            if audio_file is not None and hasattr(audio_file.info, 'length'):
                return float(audio_file.info.length)
            else:
                raise ValueError("Could not determine duration with mutagen")
                
        except (ImportError, Exception) as e:
            # Fallback to pydub
            try:
                from pydub import AudioSegment
                
                audio = AudioSegment.from_file(audio_path)
                return len(audio) / 1000.0  # Convert milliseconds to seconds
                
            except ImportError:
                logger.warning("Cannot determine accurate duration. Install mutagen or pydub.")
                # Final fallback: estimate based on file size
                file_size = audio_path.stat().st_size
                
                # Estimate based on format and typical bitrates
                ext = audio_path.suffix.lower()
                if ext == '.mp3':
                    # Assume 128kbps for MP3
                    bitrate = 128 * 1000 / 8  # Convert to bytes per second
                    return file_size / bitrate
                elif ext == '.wav':
                    # Assume 44.1kHz, 16-bit, stereo
                    bitrate = 44100 * 2 * 2  # samples/sec * bytes/sample * channels
                    return file_size / bitrate
                elif ext == '.m4b' or ext == '.aac':
                    # Assume 64kbps for AAC/M4B
                    bitrate = 64 * 1000 / 8
                    return file_size / bitrate
                else:
                    # Generic estimate: 1MB = 1 minute
                    return (file_size / 1_000_000) * 60
            except Exception as e2:
                logger.error(f"Failed to determine audio duration: {e2}")
                # Last resort estimate
                return (audio_path.stat().st_size / 1_000_000) * 60
    
    def _get_internal_model_id(self, provider: str, model: str) -> str:
        """Map provider and model to internal model ID"""
        if provider == "openai":
            return f"openai_official_{model}"
        elif provider == "elevenlabs":
            return f"elevenlabs_{model}"
        elif provider == "kokoro":
            return "local_kokoro_default_onnx"
        elif provider == "chatterbox":
            return "local_chatterbox_default"
        else:
            return model
    
    def _get_default_character_voice(self, index: int, provider: str) -> str:
        """Get default voice for character based on index"""
        if provider == "openai":
            voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
            return voices[index % len(voices)]
        elif provider == "elevenlabs":
            # Use default ElevenLabs voices
            voices = ["21m00Tcm4TlvDq8ikWAM", "AZnzlk1XvdvUeBnXmlld"]
            return voices[index % len(voices)]
        else:
            return "default"
    
    async def cancel_generation(self):
        """Cancel ongoing generation"""
        self._cancel_event.set()
        if self._generation_task:
            self._generation_task.cancel()
    
    def get_progress(self) -> AudioBookProgress:
        """Get current generation progress"""
        return self.progress

#
# End of audiobook_generator.py
#######################################################################################################################