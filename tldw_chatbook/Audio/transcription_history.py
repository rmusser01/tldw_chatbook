# transcription_history.py
"""
Encrypted transcription history storage with privacy-first design.
Uses existing encryption utilities from the project.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from loguru import logger

from ..config import get_cli_setting, save_setting_to_cli_config
from ..Utils.config_encryption import ConfigEncryption


@dataclass
class TranscriptionEntry:
    """A single transcription history entry."""
    id: str
    timestamp: datetime
    transcript: str
    duration: float
    word_count: int
    language: str
    provider: str
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TranscriptionEntry':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class TranscriptionHistory:
    """
    Manages encrypted transcription history with privacy controls.
    """
    
    def __init__(self, history_file: Optional[Path] = None):
        """
        Initialize transcription history.
        
        Args:
            history_file: Path to history file, defaults to config directory
        """
        if history_file is None:
            config_dir = Path.home() / ".config" / "tldw_cli"
            config_dir.mkdir(parents=True, exist_ok=True)
            self.history_file = config_dir / "transcription_history.enc"
        else:
            self.history_file = history_file
        
        # Initialize encryption
        self.encryption = ConfigEncryption()
        self._password = None
        self._entries: List[TranscriptionEntry] = []
        self._loaded = False
        
        # Privacy settings
        self.max_entries = get_cli_setting('dictation.privacy.max_history_entries', 100)
        self.auto_delete_days = get_cli_setting('dictation.privacy.auto_delete_days', 30)
    
    def set_password(self, password: str):
        """Set encryption password."""
        self._password = password
        self._loaded = False  # Force reload with new password
    
    def is_encrypted(self) -> bool:
        """Check if history file is encrypted."""
        if not self.history_file.exists():
            return False
        
        try:
            with open(self.history_file, 'r') as f:
                data = json.load(f)
                return data.get('encrypted', False)
        except:
            return False
    
    def load(self, password: Optional[str] = None) -> bool:
        """
        Load history from file.
        
        Args:
            password: Decryption password if needed
            
        Returns:
            True if loaded successfully
        """
        if password:
            self._password = password
        
        if not self.history_file.exists():
            self._entries = []
            self._loaded = True
            return True
        
        try:
            with open(self.history_file, 'r') as f:
                data = json.load(f)
            
            if data.get('encrypted', False):
                if not self._password:
                    logger.warning("Encrypted history requires password")
                    return False
                
                # Decrypt data
                encrypted_content = data.get('content', '')
                decrypted = self.encryption.decrypt_value(
                    encrypted_content, 
                    self._password
                )
                
                if decrypted is None:
                    logger.error("Failed to decrypt history")
                    return False
                
                entries_data = json.loads(decrypted)
            else:
                entries_data = data.get('entries', [])
            
            # Parse entries
            self._entries = []
            for entry_data in entries_data:
                try:
                    entry = TranscriptionEntry.from_dict(entry_data)
                    self._entries.append(entry)
                except Exception as e:
                    logger.error(f"Failed to parse history entry: {e}")
            
            # Clean old entries
            self._clean_old_entries()
            
            self._loaded = True
            logger.info(f"Loaded {len(self._entries)} history entries")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load history: {e}")
            return False
    
    def save(self, encrypt: bool = True) -> bool:
        """
        Save history to file.
        
        Args:
            encrypt: Whether to encrypt the history
            
        Returns:
            True if saved successfully
        """
        try:
            # Convert entries to dict
            entries_data = [entry.to_dict() for entry in self._entries]
            
            if encrypt and self._password:
                # Encrypt content
                content_json = json.dumps(entries_data)
                encrypted = self.encryption.encrypt_value(
                    content_json,
                    self._password
                )
                
                data = {
                    'encrypted': True,
                    'version': '1.0',
                    'content': encrypted,
                    'updated': datetime.now().isoformat()
                }
            else:
                # Save unencrypted
                data = {
                    'encrypted': False,
                    'version': '1.0',
                    'entries': entries_data,
                    'updated': datetime.now().isoformat()
                }
            
            # Write to file
            with open(self.history_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(self._entries)} history entries")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save history: {e}")
            return False
    
    def add_entry(
        self,
        transcript: str,
        duration: float,
        language: str = 'en',
        provider: str = 'unknown',
        metadata: Optional[Dict[str, Any]] = None
    ) -> TranscriptionEntry:
        """Add a new transcription entry."""
        if not self._loaded:
            self.load()
        
        # Create entry
        entry = TranscriptionEntry(
            id=datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
            timestamp=datetime.now(),
            transcript=transcript,
            duration=duration,
            word_count=len(transcript.split()),
            language=language,
            provider=provider,
            metadata=metadata
        )
        
        # Add to list
        self._entries.insert(0, entry)
        
        # Enforce max entries
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[:self.max_entries]
        
        # Auto-save if encryption is set up
        if self._password or not self.is_encrypted():
            self.save(encrypt=bool(self._password))
        
        return entry
    
    def get_entries(
        self,
        limit: Optional[int] = None,
        search: Optional[str] = None,
        language: Optional[str] = None,
        provider: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[TranscriptionEntry]:
        """
        Get filtered history entries.
        
        Args:
            limit: Maximum number of entries
            search: Search text in transcripts
            language: Filter by language
            provider: Filter by provider
            start_date: Filter by start date
            end_date: Filter by end date
            
        Returns:
            List of matching entries
        """
        if not self._loaded:
            self.load()
        
        entries = self._entries.copy()
        
        # Apply filters
        if search:
            search_lower = search.lower()
            entries = [
                e for e in entries 
                if search_lower in e.transcript.lower()
            ]
        
        if language:
            entries = [e for e in entries if e.language == language]
        
        if provider:
            entries = [e for e in entries if e.provider == provider]
        
        if start_date:
            entries = [e for e in entries if e.timestamp >= start_date]
        
        if end_date:
            entries = [e for e in entries if e.timestamp <= end_date]
        
        # Apply limit
        if limit:
            entries = entries[:limit]
        
        return entries
    
    def delete_entry(self, entry_id: str) -> bool:
        """Delete a specific entry."""
        if not self._loaded:
            self.load()
        
        original_count = len(self._entries)
        self._entries = [e for e in self._entries if e.id != entry_id]
        
        if len(self._entries) < original_count:
            self.save(encrypt=bool(self._password))
            return True
        
        return False
    
    def clear_history(self) -> bool:
        """Clear all history entries."""
        self._entries = []
        return self.save(encrypt=bool(self._password))
    
    def _clean_old_entries(self):
        """Remove entries older than auto_delete_days."""
        if self.auto_delete_days <= 0:
            return
        
        cutoff_date = datetime.now().timestamp() - (self.auto_delete_days * 86400)
        original_count = len(self._entries)
        
        self._entries = [
            e for e in self._entries 
            if e.timestamp.timestamp() > cutoff_date
        ]
        
        if len(self._entries) < original_count:
            logger.info(f"Cleaned {original_count - len(self._entries)} old entries")
    
    def export_to_file(
        self,
        filepath: Path,
        format: str = 'txt',
        entries: Optional[List[TranscriptionEntry]] = None
    ) -> bool:
        """
        Export history to file.
        
        Args:
            filepath: Export file path
            format: Export format (txt, json, csv, md)
            entries: Specific entries to export, or all if None
            
        Returns:
            True if exported successfully
        """
        if entries is None:
            entries = self._entries
        
        try:
            if format == 'txt':
                content = self._export_as_text(entries)
            elif format == 'json':
                content = self._export_as_json(entries)
            elif format == 'csv':
                content = self._export_as_csv(entries)
            elif format == 'md':
                content = self._export_as_markdown(entries)
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
            
            filepath.write_text(content)
            logger.info(f"Exported {len(entries)} entries to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export history: {e}")
            return False
    
    def _export_as_text(self, entries: List[TranscriptionEntry]) -> str:
        """Export as plain text."""
        lines = []
        for entry in entries:
            lines.append(f"{'=' * 60}")
            lines.append(f"Date: {entry.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"Duration: {entry.duration:.1f}s | Words: {entry.word_count}")
            lines.append(f"Language: {entry.language} | Provider: {entry.provider}")
            lines.append("")
            lines.append(entry.transcript)
            lines.append("")
        
        return '\n'.join(lines)
    
    def _export_as_json(self, entries: List[TranscriptionEntry]) -> str:
        """Export as JSON."""
        data = {
            'exported': datetime.now().isoformat(),
            'count': len(entries),
            'entries': [entry.to_dict() for entry in entries]
        }
        return json.dumps(data, indent=2)
    
    def _export_as_csv(self, entries: List[TranscriptionEntry]) -> str:
        """Export as CSV."""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow([
            'Timestamp', 'Duration', 'Words', 'Language', 
            'Provider', 'Transcript'
        ])
        
        # Data
        for entry in entries:
            writer.writerow([
                entry.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                f"{entry.duration:.1f}",
                str(entry.word_count),
                entry.language,
                entry.provider,
                entry.transcript
            ])
        
        return output.getvalue()
    
    def _export_as_markdown(self, entries: List[TranscriptionEntry]) -> str:
        """Export as Markdown."""
        lines = ["# Transcription History", ""]
        
        for entry in entries:
            lines.append(f"## {entry.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append("")
            lines.append(f"- **Duration**: {entry.duration:.1f} seconds")
            lines.append(f"- **Words**: {entry.word_count}")
            lines.append(f"- **Language**: {entry.language}")
            lines.append(f"- **Provider**: {entry.provider}")
            lines.append("")
            lines.append("### Transcript")
            lines.append("")
            lines.append(entry.transcript)
            lines.append("")
            lines.append("---")
            lines.append("")
        
        return '\n'.join(lines)


# Global instance
_history_instance = None

def get_transcription_history() -> TranscriptionHistory:
    """Get the global transcription history instance."""
    global _history_instance
    if _history_instance is None:
        _history_instance = TranscriptionHistory()
    return _history_instance