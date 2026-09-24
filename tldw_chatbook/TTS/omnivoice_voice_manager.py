# omnivoice_voice_manager.py
# Description: Voice profile management for the OmniVoice ONNX TTS backend
#
# Imports
import json
import re
import shutil
import wave
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from tldw_chatbook.TTS.backends.voice_manager_base import VoiceManagerBase
from tldw_chatbook.TTS.profile_errors import ProfileValidationError
from tldw_chatbook.TTS.profile_reference_types import (
    MAX_REFERENCE_TEXT_CHARACTERS,
    validate_reference_text,
)

# Optional imports
try:
    import soundfile as _sf

    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    _sf = None

#######################################################################################################################
#
# OmniVoice Voice Manager
#

_DEFAULT_VOICES_DIR = Path("~/.config/tldw_cli/omnivoice_voices")
_EXPORT_PREFIX = "omnivoice_voice_"
# Profile names become directory names; keep them to a safe, flat charset.
_PROFILE_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


class OmniVoiceVoiceManager(VoiceManagerBase):
    """Voice profile manager for the OmniVoice ONNX TTS backend.

    OmniVoice clones a voice from a reference clip plus its exact
    transcript, so every profile stores both: each profile lives in its
    own directory under ``voice_samples_dir`` as ``profile.json`` +
    ``reference.<ext>`` with a required ``reference_text`` field.

    Stored ``profile.json`` schema::

        {
          "name": "string",
          "display_name": "string|null",
          "language": "en",
          "description": "string|null",
          "reference_audio": "reference.wav",
          "reference_text": "bounded transcript (required — cloning needs it)",
          "created_at": "ISO-8601",
          "updated_at": "ISO-8601"
        }
    """

    def __init__(
        self,
        voice_samples_dir: Optional[Path] = None,
        max_reference_duration: float = 30.0,
    ):
        """
        Initialize the OmniVoice voice manager.

        Args:
            voice_samples_dir: Directory for storing voice profiles
                (default: ``~/.config/tldw_cli/omnivoice_voices``).
            max_reference_duration: Maximum reference clip duration in
                seconds; values <= 0 disable the limit (engine parity).
        """
        super().__init__(voice_samples_dir or _DEFAULT_VOICES_DIR.expanduser())
        self.max_reference_duration = float(max_reference_duration)

    # -- profile CRUD ----------------------------------------------------------------

    def create_profile(
        self,
        profile_name: str,
        reference_audio_path: str,
        display_name: Optional[str] = None,
        language: str = "en",
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        reference_text: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Create a new OmniVoice voice profile.

        ``reference_text`` — the transcript of the reference clip — is
        required for zero-shot cloning; calls following the base
        ``VoiceManagerBase`` signature without it are rejected with a
        message naming ``reference_text``. ``tags`` and ``metadata`` are
        accepted for signature compatibility but not stored (OmniVoice
        profiles carry only the cloning payload).
        """
        try:
            if tags or metadata:
                logger.debug(
                    "OmniVoice profiles do not store tags/metadata; ignoring"
                )
            valid, error = self._validate_profile_name(profile_name)
            if not valid:
                return False, error

            try:
                transcript = validate_reference_text(reference_text)
            except ProfileValidationError:
                return False, (
                    f"reference_text is required for OmniVoice voice cloning and "
                    f"must be non-empty text of at most "
                    f"{MAX_REFERENCE_TEXT_CHARACTERS} characters"
                )

            profile_dir = self._profile_dir(profile_name)
            if profile_dir.exists():
                return False, f"Profile '{profile_name}' already exists"

            ref_path = Path(reference_audio_path).expanduser()
            if not ref_path.is_file():
                return False, f"Reference audio not found: {reference_audio_path}"

            is_valid, audio_info = self.validate_audio_file(ref_path)
            if not is_valid:
                error = audio_info.get("error", "unknown")
                return False, f"Invalid audio file: {error}"

            duration = self._probe_duration(ref_path)
            if duration is None:
                return False, (
                    f"Could not determine the duration of {ref_path.name}; provide a "
                    f"WAV file (or install soundfile for other formats)"
                )
            if duration <= 0:
                return False, f"Reference audio has no audio frames: {ref_path.name}"
            limit = self.max_reference_duration
            if limit > 0 and duration > limit:
                return False, (
                    f"Reference audio duration {duration:.1f}s exceeds the "
                    f"{limit:.1f}s maximum for OmniVoice cloning"
                )

            profile_dir.mkdir(parents=True)
            dest_audio = profile_dir / f"reference{ref_path.suffix.lower()}"
            shutil.copy2(ref_path, dest_audio)

            now = datetime.now().isoformat()
            profile = {
                "name": profile_name,
                "display_name": display_name or profile_name,
                "language": language,
                "description": description,
                "reference_audio": dest_audio.name,
                "reference_text": transcript,
                "created_at": now,
                "updated_at": now,
            }
            try:
                self._write_profile(profile_dir, profile)
            except Exception:
                shutil.rmtree(profile_dir, ignore_errors=True)
                raise
            logger.info(f"Created OmniVoice voice profile '{profile_name}'")
            return True, f"Successfully created profile '{profile_name}'"

        except Exception as e:
            logger.error(f"Error creating OmniVoice profile: {e}")
            return False, f"Error: {str(e)}"

    def list_profiles(self, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """List all OmniVoice voice profiles.

        Args:
            tags: Filter by tags (accepted but not stored for OmniVoice
                profiles; providing it filters everything out).

        Returns:
            List of profile summaries.
        """
        result = []
        if not self.voice_samples_dir.is_dir():
            return result
        for entry in sorted(self.voice_samples_dir.iterdir()):
            if not entry.is_dir():
                continue
            stored = self._read_profile(entry)
            if stored is None:
                continue
            profile_tags = []
            if tags and not any(tag in profile_tags for tag in tags):
                continue
            result.append(
                {
                    "name": stored.get("name", entry.name),
                    "display_name": stored.get("display_name") or entry.name,
                    "language": stored.get("language", "unknown"),
                    "description": stored.get("description", ""),
                    "tags": profile_tags,
                    "created_at": stored.get("created_at", "unknown"),
                    "backend": "omnivoice",
                    "has_reference": bool(stored.get("reference_audio")),
                }
            )
        result.sort(key=lambda x: x["display_name"].lower())
        return result

    def get_profile(self, profile_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific OmniVoice voice profile (or None if not found)."""
        profile_dir = self._profile_dir(profile_name)
        if not profile_dir.is_dir():
            return None
        return self._read_profile(profile_dir)

    def update_profile(
        self,
        profile_name: str,
        display_name: Optional[str] = None,
        language: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata_update: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """Update an existing OmniVoice voice profile.

        ``tags`` and ``metadata_update`` are accepted for signature
        compatibility but not stored (OmniVoice profiles carry only the
        cloning payload).
        """
        try:
            if tags is not None or metadata_update:
                logger.debug(
                    "OmniVoice profiles do not store tags/metadata; ignoring"
                )
            profile = self.get_profile(profile_name)
            if profile is None:
                return False, f"Profile '{profile_name}' not found"

            if display_name is not None:
                profile["display_name"] = display_name
            if language is not None:
                profile["language"] = language
            if description is not None:
                profile["description"] = description

            profile["updated_at"] = datetime.now().isoformat()
            self._write_profile(self._profile_dir(profile_name), profile)
            return True, f"Successfully updated profile '{profile_name}'"

        except Exception as e:
            logger.error(f"Error updating OmniVoice profile: {e}")
            return False, f"Error: {str(e)}"

    def delete_profile(self, profile_name: str) -> Tuple[bool, str]:
        """Delete an OmniVoice voice profile and its reference audio."""
        try:
            profile_dir = self._profile_dir(profile_name)
            if not profile_dir.is_dir():
                return False, f"Profile '{profile_name}' not found"

            shutil.rmtree(profile_dir)
            logger.info(f"Deleted OmniVoice voice profile '{profile_name}'")
            return True, f"Successfully deleted profile '{profile_name}'"

        except Exception as e:
            logger.error(f"Error deleting OmniVoice profile: {e}")
            return False, f"Error: {str(e)}"

    # -- import/export ---------------------------------------------------------------

    def export_profile(self, profile_name: str, export_path: str) -> Tuple[bool, str]:
        """Export a voice profile with its reference audio.

        Args:
            profile_name: Profile to export
            export_path: Directory to export to

        Returns:
            (success, message) tuple
        """
        try:
            profile = self.get_profile(profile_name)
            if profile is None:
                return False, f"Profile '{profile_name}' not found"

            export_dir = Path(export_path)
            export_dir.mkdir(parents=True, exist_ok=True)
            package_dir = export_dir / f"{_EXPORT_PREFIX}{profile_name}"
            package_dir.mkdir(exist_ok=True)

            reference_name = profile.get("reference_audio")
            if reference_name:
                source_audio = self._profile_dir(profile_name) / reference_name
                if source_audio.is_file():
                    shutil.copy2(source_audio, package_dir / reference_name)
                else:
                    logger.warning(
                        f"Reference audio not found for export: {source_audio}"
                    )
                    profile = {
                        k: v for k, v in profile.items() if k != "reference_audio"
                    }

            with open(package_dir / "profile.json", "w") as f:
                json.dump(profile, f, indent=2)

            with open(package_dir / "README.txt", "w") as f:
                f.write(f"OmniVoice Voice Profile: {profile_name}\n")
                f.write(f"Display Name: {profile.get('display_name', profile_name)}\n")
                f.write(f"Language: {profile.get('language', 'unknown')}\n")
                f.write(f"Created: {profile.get('created_at', 'unknown')}\n")
                description = profile.get("description") or "No description"
                f.write(f"\nDescription:\n{description}\n")
                f.write(
                    f"\nReference transcript (reference_text, required for "
                    f"cloning):\n{profile.get('reference_text', '')}\n"
                )
                f.write("\nTo import this profile, use the import_profile function.\n")

            logger.info(f"Exported OmniVoice profile '{profile_name}' to {package_dir}")
            return True, f"Successfully exported to {package_dir}"

        except Exception as e:
            logger.error(f"Error exporting OmniVoice profile: {e}")
            return False, f"Error: {str(e)}"

    def import_profile(
        self,
        import_path: str,
        profile_name: Optional[str] = None,
        overwrite: bool = False,
    ) -> Tuple[bool, str]:
        """Import a voice profile from an export package.

        Args:
            import_path: Path to profile package or profile.json
            profile_name: New name for profile (optional)
            overwrite: Whether to overwrite existing profile

        Returns:
            (success, message) tuple
        """
        try:
            import_dir = Path(import_path).expanduser()
            if import_dir.is_file() and import_dir.name == "profile.json":
                package_dir = import_dir.parent
            elif import_dir.is_dir():
                package_dir = import_dir
            else:
                return (
                    False,
                    "Invalid import path. Expected profile.json or package directory",
                )

            profile_file = package_dir / "profile.json"
            if not profile_file.is_file():
                return False, "profile.json not found in import package"

            try:
                with open(profile_file, "r") as f:
                    imported = json.load(f)
            except Exception as e:
                return False, f"Could not read profile.json: {e}"

            try:
                transcript = validate_reference_text(imported.get("reference_text"))
            except ProfileValidationError:
                return False, (
                    "reference_text is required for OmniVoice voice cloning and is "
                    "missing or invalid in the import package"
                )

            name = profile_name or self._name_from_package(package_dir)
            valid, error = self._validate_profile_name(name)
            if not valid:
                return False, error

            profile_dir = self._profile_dir(name)
            if profile_dir.exists() and not overwrite:
                return (
                    False,
                    f"Profile '{name}' already exists. Use overwrite=True to replace",
                )

            reference_name = imported.get("reference_audio")
            if reference_name:
                source_audio = package_dir / Path(reference_name).name
                if not source_audio.is_file():
                    return (
                        False,
                        f"Reference audio not found in package: {reference_name}",
                    )

            now = datetime.now().isoformat()
            profile_dir.mkdir(parents=True, exist_ok=True)
            if reference_name:
                shutil.copy2(
                    package_dir / Path(reference_name).name,
                    profile_dir / Path(reference_name).name,
                )
            imported["name"] = name
            imported["reference_text"] = transcript
            imported["reference_audio"] = (
                Path(reference_name).name if reference_name else ""
            )
            imported.setdefault("created_at", now)
            imported["updated_at"] = now

            with open(profile_dir / "profile.json", "w") as f:
                json.dump(imported, f, indent=2)

            logger.info(f"Imported OmniVoice voice profile '{name}'")
            return True, f"Successfully imported profile '{name}'"

        except Exception as e:
            logger.error(f"Error importing OmniVoice profile: {e}")
            return False, f"Error: {str(e)}"

    # -- helpers ---------------------------------------------------------------------

    def get_reference_audio_path(self, profile_name: str) -> Optional[Path]:
        """Return the absolute path of a profile's reference clip, if any.

        Args:
            profile_name: Profile identifier

        Returns:
            Resolved path to the copied reference audio, or None if the
            profile or its clip is missing.
        """
        profile = self.get_profile(profile_name)
        if profile is None:
            return None
        reference_name = profile.get("reference_audio")
        if not reference_name:
            return None
        return self._profile_dir(profile_name) / Path(reference_name).name

    def get_supported_features(self) -> List[str]:
        """Capabilities advertised by this backend's profile manager."""
        return ["basic_profiles", "import_export", "zero_shot_cloning"]

    def _profile_dir(self, profile_name: str) -> Path:
        return self.voice_samples_dir / profile_name

    @staticmethod
    def _validate_profile_name(profile_name: str) -> Tuple[bool, str]:
        if not profile_name or not _PROFILE_NAME_PATTERN.fullmatch(profile_name):
            return (
                False,
                "Profile name must be non-empty letters, digits, '.', '_' or '-' "
                "(it becomes a directory name)",
            )
        return True, ""

    @staticmethod
    def _name_from_package(package_dir: Path) -> str:
        if package_dir.name.startswith(_EXPORT_PREFIX):
            return package_dir.name[len(_EXPORT_PREFIX) :]
        return f"imported_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    @staticmethod
    def _read_profile(profile_dir: Path) -> Optional[Dict[str, Any]]:
        profile_file = profile_dir / "profile.json"
        if not profile_file.is_file():
            return None
        try:
            with open(profile_file, "r") as f:
                stored = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load profile in {profile_dir}: {e}")
            return None
        if not isinstance(stored, dict):
            logger.error(f"Corrupt profile.json in {profile_dir}")
            return None
        return stored

    @staticmethod
    def _write_profile(profile_dir: Path, profile: Dict[str, Any]) -> None:
        with open(profile_dir / "profile.json", "w") as f:
            json.dump(profile, f, indent=2)

    @staticmethod
    def _probe_duration(audio_path: Path) -> Optional[float]:
        """Return the clip's duration in seconds, or None if unverifiable.

        WAV files are measured via the stdlib ``wave`` header; other
        containers need soundfile (the same library the OmniVoice engine
        uses to decode them).
        """
        if audio_path.suffix.lower() == ".wav":
            try:
                with wave.open(str(audio_path), "rb") as reader:
                    rate = reader.getframerate()
                    if rate <= 0:
                        return None
                    return reader.getnframes() / rate
            except (wave.Error, EOFError) as e:
                logger.error(f"Could not read WAV header of {audio_path}: {e}")
                return None
        if SOUNDFILE_AVAILABLE and _sf is not None:
            try:
                info = _sf.info(str(audio_path))
                if info.samplerate > 0:
                    return info.frames / info.samplerate
            except Exception as e:
                logger.error(f"Could not read audio info of {audio_path}: {e}")
        return None


#
# End of omnivoice_voice_manager.py
#######################################################################################################################
