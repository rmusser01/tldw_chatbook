# voice_manager_base.py
# Description: Base class for TTS voice profile managers
#
# Imports
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from loguru import logger

#######################################################################################################################
#
# Base Voice Manager Interface
#

class VoiceManagerBase(ABC):
    """
    Abstract base class for TTS voice profile managers.
    
    Provides a common interface for managing voice profiles across different TTS backends.
    Each backend (Higgs, Chatterbox, GPT-SoVITS, etc.) should implement this interface.
    """
    
    def __init__(self, voice_samples_dir: Path):
        """
        Initialize the voice manager.
        
        Args:
            voice_samples_dir: Directory for storing voice samples and profiles
        """
        self.voice_samples_dir = Path(voice_samples_dir)
        self.voice_samples_dir.mkdir(parents=True, exist_ok=True)
        self.backend_name = self.__class__.__name__.replace("VoiceManager", "").replace("VoiceProfileManager", "")
    
    @abstractmethod
    def create_profile(
        self,
        profile_name: str,
        reference_audio_path: str,
        display_name: Optional[str] = None,
        language: str = "en",
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """
        Create a new voice profile.
        
        Args:
            profile_name: Unique identifier for the profile
            reference_audio_path: Path to reference audio file
            display_name: Human-readable name
            language: Language code
            description: Profile description
            tags: Tags for categorization
            metadata: Additional backend-specific metadata
            
        Returns:
            (success, message) tuple
        """
        pass
    
    @abstractmethod
    def list_profiles(self, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        List all voice profiles.
        
        Args:
            tags: Filter by tags (if provided)
            
        Returns:
            List of profile summaries with at least:
            - name: Profile identifier
            - display_name: Human-readable name
            - language: Language code
            - description: Profile description
            - tags: List of tags
            - created_at: Creation timestamp
            - backend: Backend name
        """
        pass
    
    @abstractmethod
    def get_profile(self, profile_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific voice profile.
        
        Args:
            profile_name: Profile identifier
            
        Returns:
            Profile data or None if not found
        """
        pass
    
    @abstractmethod
    def delete_profile(self, profile_name: str) -> Tuple[bool, str]:
        """
        Delete a voice profile.
        
        Args:
            profile_name: Profile identifier
            
        Returns:
            (success, message) tuple
        """
        pass
    
    @abstractmethod
    def update_profile(
        self,
        profile_name: str,
        display_name: Optional[str] = None,
        language: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata_update: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """
        Update an existing voice profile.
        
        Args:
            profile_name: Profile identifier
            display_name: New display name (optional)
            language: New language code (optional)
            description: New description (optional)
            tags: New tags list (optional)
            metadata_update: Metadata fields to update (optional)
            
        Returns:
            (success, message) tuple
        """
        pass
    
    @abstractmethod
    def export_profile(self, profile_name: str, export_path: str) -> Tuple[bool, str]:
        """
        Export a voice profile with its reference audio.
        
        Args:
            profile_name: Profile to export
            export_path: Directory to export to
            
        Returns:
            (success, message) tuple
        """
        pass
    
    @abstractmethod
    def import_profile(
        self,
        import_path: str,
        profile_name: Optional[str] = None,
        overwrite: bool = False
    ) -> Tuple[bool, str]:
        """
        Import a voice profile from export package.
        
        Args:
            import_path: Path to profile package or profile.json
            profile_name: New name for profile (optional)
            overwrite: Whether to overwrite existing profile
            
        Returns:
            (success, message) tuple
        """
        pass
    
    def validate_audio_file(self, audio_path: Path) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate and get info about audio file.
        
        Default implementation checks file existence and extension.
        Subclasses can override for more detailed validation.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            (is_valid, info_dict) tuple
        """
        try:
            if not audio_path.exists():
                return False, {"error": "File not found"}
            
            valid_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
            if audio_path.suffix.lower() not in valid_extensions:
                return False, {"error": f"Unsupported format: {audio_path.suffix}"}
            
            info = {
                "path": str(audio_path),
                "size_mb": audio_path.stat().st_size / (1024 * 1024),
                "format": audio_path.suffix.lower()
            }
            
            return True, info
            
        except Exception as e:
            return False, {"error": str(e)}
    
    def get_backend_info(self) -> Dict[str, Any]:
        """
        Get information about this voice manager backend.
        
        Returns:
            Dictionary with backend information
        """
        return {
            "name": self.backend_name,
            "profiles_dir": str(self.voice_samples_dir),
            "supported_formats": ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'],
            "features": self.get_supported_features()
        }
    
    def get_supported_features(self) -> List[str]:
        """
        Get list of supported features for this backend.
        
        Subclasses should override to indicate their capabilities.
        
        Returns:
            List of feature strings
        """
        return ["basic_profiles", "import_export"]
    
    def search_profiles(
        self,
        query: str,
        search_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search profiles by text query.
        
        Default implementation searches in name, display_name, and description.
        
        Args:
            query: Search query
            search_fields: Fields to search in (default: name, display_name, description)
            
        Returns:
            List of matching profile summaries
        """
        if not search_fields:
            search_fields = ["name", "display_name", "description"]
        
        query_lower = query.lower()
        all_profiles = self.list_profiles()
        matches = []
        
        for profile in all_profiles:
            for field in search_fields:
                value = profile.get(field, "")
                if value and query_lower in str(value).lower():
                    matches.append(profile)
                    break
        
        return matches

#
# End of voice_manager_base.py
#######################################################################################################################