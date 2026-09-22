# voice_manager_base.py
# Description: Base class for TTS voice profile managers
#
# Imports
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from loguru import logger

from tldw_chatbook.TTS import loose_voice_lifetime as voice_files
from tldw_chatbook.Utils.timestamps import utc_now_iso

#######################################################################################################################
#
# Base Voice Manager Interface
#


class VoiceManagerBase(ABC):
    """
    Abstract base class for TTS voice profile managers.

    Provides a common interface for managing voice profiles across different TTS backends.
    Each backend (Higgs, Chatterbox, GPT-SoVITS, etc.) should implement this interface.

    The base also owns the shared single-JSON profile store mechanics (TASK-32863):
    the records file name, the load/save cache, and the update/delete/get CRUD
    that every backend implements identically. Subclasses customize storage via
    ``profiles_filename`` and log wording via the label attributes, keep
    timestamped backups by setting ``keep_backups`` and overriding
    ``_backup_before_save``, and implement the engine-specific
    create/list/export/import profile methods.
    """

    #: Records file name inside ``voice_samples_dir``.
    profiles_filename = "voice_profiles.json"
    #: Wording for store-level logs, e.g. "Failed to load {label} profiles".
    store_log_label = "voice"
    #: Prefix for CRUD error logs, e.g. "Error updating {label}profile".
    action_log_label = ""
    #: Create a ``backups`` directory and call ``_backup_before_save`` on save.
    keep_backups = False

    def __init__(self, voice_samples_dir: Path):
        """
        Initialize the voice manager and its profile store.

        Args:
            voice_samples_dir: Directory for storing voice samples and profiles;
                the records file, cache, and optional backups directory live
                inside it.
        """
        self.voice_samples_dir = Path(voice_samples_dir)
        voice_files.mkdir(self, self.voice_samples_dir, parents=True, exist_ok=True)
        self.backend_name = self.__class__.__name__.replace("VoiceManager", "").replace(
            "VoiceProfileManager", ""
        )
        self.profiles_file = self.voice_samples_dir / self.profiles_filename
        self._profiles_cache: Optional[Dict[str, Dict[str, Any]]] = None
        if self.keep_backups:
            self.backup_dir = self.voice_samples_dir / "backups"
            voice_files.mkdir(self, self.backup_dir, exist_ok=True)

    def _backup_before_save(self) -> None:
        """Hook: retain the previous records file before saving (no-op)."""

    @voice_files.call
    def load_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Load the profile records from the store file.

        Returns:
            The cached-or-loaded ``{name: record}`` mapping; empty when the
            store is missing or unreadable.
        """
        if self._profiles_cache is not None:
            return self._profiles_cache

        if self.profiles_file.exists():
            try:
                with voice_files.open_text(self, self.profiles_file, "r") as f:
                    self._profiles_cache = json.load(f)
                    return self._profiles_cache
            except Exception as e:
                logger.error(f"Failed to load {self.store_log_label} profiles: {e}")
                self._profiles_cache = {}
        else:
            self._profiles_cache = {}

        return self._profiles_cache

    @voice_files.call
    def save_profiles(self, profiles: Dict[str, Dict[str, Any]]) -> bool:
        """Persist the profile records, backing up first when enabled.

        Args:
            profiles: The complete ``{name: record}`` mapping to store.

        Returns:
            True on success; False (logged) on a write failure.
        """
        try:
            self._backup_before_save()

            with voice_files.open_text(self, self.profiles_file, "w") as f:
                json.dump(profiles, f, indent=2)

            self._profiles_cache = profiles
            return True
        except Exception as e:
            logger.error(f"Failed to save {self.store_log_label} profiles: {e}")
            return False

    @voice_files.call
    def get_profile(self, profile_name: str) -> Optional[Dict[str, Any]]:
        """Get one profile record.

        Args:
            profile_name: Profile identifier.

        Returns:
            The record, or None when absent.
        """
        profiles = self.load_profiles()
        return profiles.get(profile_name)

    @voice_files.call
    def update_profile(
        self,
        profile_name: str,
        display_name: Optional[str] = None,
        language: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata_update: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """Update an existing voice profile.

        Args:
            profile_name: Profile identifier.
            display_name: New display name (optional).
            language: New language code (optional).
            description: New description (optional).
            tags: New tags list (optional).
            metadata_update: Metadata fields to merge (optional).

        Returns:
            ``(success, message)``.
        """
        try:
            profiles = self.load_profiles()
            if profile_name not in profiles:
                return False, f"Profile '{profile_name}' not found"

            profile = profiles[profile_name]

            # Update fields if provided
            if display_name is not None:
                profile["display_name"] = display_name
            if language is not None:
                profile["language"] = language
            if description is not None:
                profile["description"] = description
            if tags is not None:
                profile["tags"] = tags
            if metadata_update:
                profile["metadata"].update(metadata_update)

            profile["updated_at"] = utc_now_iso()

            # Save updated profiles
            if self.save_profiles(profiles):
                return True, f"Successfully updated profile '{profile_name}'"
            else:
                return False, "Failed to save profile updates"

        except Exception as e:
            logger.error(
                f"Error updating {self.action_log_label}profile '{profile_name}': {e}"
            )
            return False, f"Error: {str(e)}"

    @voice_files.call
    def delete_profile(self, profile_name: str) -> Tuple[bool, str]:
        """Delete a voice profile and its reference directory.

        Args:
            profile_name: Profile identifier; validated to stay inside the
                samples root before any filesystem removal.

        Returns:
            ``(success, message)``.
        """
        try:
            profiles = self.load_profiles()
            if profile_name not in profiles:
                return False, f"Profile '{profile_name}' not found"

            # Remove profile directory. Profile names are path components:
            # the loose-voice wrapper refuses path-component characters in
            # profile_name before this call, and this resolution-free check
            # (validating via resolve() would desynchronize from the
            # admission layer's registered, unresolved roots) is the second
            # containment layer -- a traversal-shaped name never reaches
            # the filesystem.
            profile_dir = self.voice_samples_dir / profile_name
            if Path(profile_name).name != profile_name:
                return False, "Error: invalid profile name"
            if profile_dir.exists():
                voice_files.remove_tree(self, profile_dir)

            # Remove from profiles
            del profiles[profile_name]

            # Save updated profiles
            if self.save_profiles(profiles):
                logger.info(f"Deleted {self.store_log_label} profile '{profile_name}'")
                return True, f"Successfully deleted profile '{profile_name}'"
            else:
                return False, "Failed to save profile deletion"

        except Exception as e:
            logger.error(
                f"Error deleting {self.action_log_label}profile '{profile_name}': {e}"
            )
            return False, f"Error: {str(e)}"

    @abstractmethod
    def create_profile(
        self,
        profile_name: str,
        reference_audio_path: str,
        display_name: Optional[str] = None,
        language: str = "en",
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
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
    @abstractmethod
    @abstractmethod
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
        overwrite: bool = False,
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

            valid_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
            if audio_path.suffix.lower() not in valid_extensions:
                return False, {"error": f"Unsupported format: {audio_path.suffix}"}

            info = {
                "path": str(audio_path),
                "size_mb": audio_path.stat().st_size / (1024 * 1024),
                "format": audio_path.suffix.lower(),
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
            "supported_formats": [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"],
            "features": self.get_supported_features(),
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
        self, query: str, search_fields: Optional[List[str]] = None
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
