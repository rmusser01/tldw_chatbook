# voice_manager_base.py
# Description: Base class for TTS voice profile managers
#
# Imports
import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from loguru import logger

from tldw_chatbook.TTS import loose_voice_lifetime as voice_files
from tldw_chatbook.Utils.timestamps import utc_now, utc_now_iso

#######################################################################################################################
#
# Profile-store backup naming and ordering
#

#: Backup filename stamp: UTC, ``Z``-suffixed, and filename-safe -- ADR-173's
#: canonical stored shape carries ``:``, which a path component cannot.
#:
#: MICROSECONDS, not ADR-173's milliseconds, because this stamp is also the
#: backup's IDENTITY: at second precision every save inside the same second
#: resolved to one path, which ``voice_files.copy`` then replaced, so a burst
#: of edits (an import, a multi-select delete) kept ONE recovery point instead
#: of ``BACKUP_KEEP``. Copying the store takes far longer than a microsecond,
#: so consecutive saves are now collision-free without an existence-check loop.
BACKUP_STAMP_FORMAT = "%Y%m%dT%H%M%S.%fZ"

#: Second-precision stamps written before that fix -- parsed rather than
#: discarded, because a user upgrading mid-rotation has both shapes in one
#: backup directory and the older ones are still real recovery points.
_SUPERSEDED_BACKUP_STAMP_FORMATS = ("%Y%m%dT%H%M%SZ",)

#: How many rotated backups a profile store keeps.
BACKUP_KEEP = 10


def _backup_moment(path: Path) -> datetime:
    """When this backup was taken, as an aware UTC ``datetime``.

    (TASK-32893) Backups used to be named with ``datetime.now()`` -- naive
    LOCAL time -- and selected with ``sorted(glob(...))``, i.e. by filename
    bytes. Across a DST fall-back, or on a machine that changed timezone, the
    lexically-last name is not the most recent backup, so "restore the latest"
    restored an older file over the user's newer profiles.

    Args:
        path: A backup file named ``<stem>_backup_<stamp>.json``.

    Returns:
        The instant the backup was taken, as an aware UTC ``datetime``.
    """
    stamp = path.stem.rsplit("_backup_", 1)[-1]
    for fmt in (BACKUP_STAMP_FORMAT, *_SUPERSEDED_BACKUP_STAMP_FORMATS):
        try:
            return datetime.strptime(stamp, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    # A pre-TASK-32893 name carries a naive LOCAL stamp that cannot be placed
    # on a timeline without the writer's offset. The file's own mtime can:
    # ``shutil.copy`` does not preserve mtime, so the mtime IS the moment the
    # backup was taken.
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)


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
    timestamped backups by setting ``keep_backups``, and implement the
    engine-specific create/list/export/import profile methods.
    """

    #: Records file name inside ``voice_samples_dir``.
    profiles_filename = "voice_profiles.json"
    #: Wording for store-level logs, e.g. "Failed to load {label} profiles".
    store_log_label = "voice"
    #: Prefix for CRUD error logs, e.g. "Error updating {label}profile".
    action_log_label = ""
    #: Create a ``backups`` directory and call ``_backup_before_save`` on save.
    keep_backups = False
    #: Largest reference-audio file a profile may adopt. ``create_profile``
    #: copies the picked file whole through a single in-memory read
    #: (``loose_voice_lifetime.copy``), so an unbounded pick -- a multi-GB
    #: container, say -- is an OOM of the whole TUI. Matches the bound
    #: ``HiggsVoiceProfileManager`` already applies in its own validator.
    #: NOT ``sample_audio_validation.MAX_PLAYABLE_AUDIO_BYTES`` (8 MiB): that
    #: bounds audio this app GENERATED, and would reject an ordinary
    #: few-minute user recording.
    max_reference_audio_bytes = 100 * 1024 * 1024

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

    def _list_backups(self) -> List[Path]:
        """This store's backups, OLDEST FIRST, ordered by parsed timestamp.

        Returns:
            Chronologically ordered paths; empty when there are none, or when
            this backend does not keep backups.
        """
        if not self.keep_backups or not self.backup_dir.is_dir():
            return []
        return sorted(
            self.backup_dir.glob(f"{self.profiles_file.stem}_backup_*.json"),
            # The name breaks mtime ties so the prune order is deterministic.
            key=lambda path: (_backup_moment(path), path.name),
        )

    def _backup_before_save(self) -> None:
        """Copy the live records file aside before it is overwritten.

        (TASK-32893) Every save replaces the whole store, so this copy is the
        only thing between one bad profile edit and all of the user's
        profiles. Higgs had it; Chatterbox did not, and now both do via
        ``keep_backups``.

        A failure is logged and swallowed: an absent backup is not a reason to
        refuse the save the caller actually asked for.
        """
        if not self.keep_backups:
            return
        try:
            if not self.profiles_file.exists():
                return  # nothing stored yet, so nothing to preserve
            stamp = utc_now().strftime(BACKUP_STAMP_FORMAT)
            voice_files.copy(
                self,
                self.profiles_file,
                self.backup_dir / f"{self.profiles_file.stem}_backup_{stamp}.json",
            )
            # Prune by parsed timestamp, not by filename bytes.
            for stale in self._list_backups()[:-BACKUP_KEEP]:
                voice_files.unlink(self, stale)
        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")

    @voice_files.call
    def load_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Load the profile records from the store file.

        (TASK-32893) A read error used to be logged and answered with ``{}``
        -- the very same answer as "this user has no profiles" -- so the next
        ``save_profiles`` wrote an empty store over a file that was merely
        locked, permission-denied, or truncated, and every profile in it was
        gone. ABSENT and UNREADABLE are now different answers; the CRUD
        methods turn the raise into a refusal, which leaves the bytes on disk
        exactly as they were found.

        Returns:
            The cached-or-loaded ``{name: record}`` mapping; ``{}`` when the
            store does not exist yet.

        Raises:
            OSError: The store exists but could not be read.
            json.JSONDecodeError: The store exists but is not valid JSON.
            ValueError: The store exists but does not hold a JSON object.
        """
        if self._profiles_cache is not None:
            return self._profiles_cache

        if not self.profiles_file.exists():
            self._profiles_cache = {}
            return self._profiles_cache

        with voice_files.open_text(self, self.profiles_file, "r") as f:
            profiles = json.load(f)
        if not isinstance(profiles, dict):
            raise ValueError(
                f"Voice profile store {self.profiles_file} does not hold a JSON "
                f"object (found {type(profiles).__name__}); refusing to treat "
                f"it as empty."
            )
        self._profiles_cache = profiles
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

        Default implementation checks file existence, extension, and size
        against ``max_reference_audio_bytes``. Subclasses can override for
        more detailed validation.

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

            size_bytes = audio_path.stat().st_size
            if size_bytes > self.max_reference_audio_bytes:
                return False, {
                    "error": (
                        f"File too large: {size_bytes / (1024 * 1024):.1f}MB "
                        f"(max {self.max_reference_audio_bytes // (1024 * 1024)}MB)"
                    )
                }

            info = {
                "path": str(audio_path),
                "size_mb": size_bytes / (1024 * 1024),
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
