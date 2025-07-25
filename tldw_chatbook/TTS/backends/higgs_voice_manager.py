# higgs_voice_manager.py
# Description: Voice profile management utilities for Higgs Audio TTS
#
# Imports
import os
import json
import shutil
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from loguru import logger

# Optional imports
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    librosa = None

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    sf = None

#######################################################################################################################
#
# Voice Profile Manager
#

class HiggsVoiceProfileManager:
    """
    Manages voice profiles for Higgs Audio TTS backend.
    
    Features:
    - Create, list, update, delete voice profiles
    - Import/export voice profiles
    - Analyze audio characteristics
    - Validate reference audio files
    - Backup and restore profiles
    """
    
    def __init__(self, voice_samples_dir: Path):
        """
        Initialize the voice profile manager.
        
        Args:
            voice_samples_dir: Directory for storing voice samples and profiles
        """
        self.voice_samples_dir = Path(voice_samples_dir)
        self.voice_samples_dir.mkdir(parents=True, exist_ok=True)
        
        self.profiles_file = self.voice_samples_dir / "voice_profiles.json"
        self.backup_dir = self.voice_samples_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        self._profiles_cache: Optional[Dict[str, Dict[str, Any]]] = None
    
    def load_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Load voice profiles from disk"""
        if self._profiles_cache is not None:
            return self._profiles_cache
        
        if self.profiles_file.exists():
            try:
                with open(self.profiles_file, 'r') as f:
                    self._profiles_cache = json.load(f)
                    return self._profiles_cache
            except Exception as e:
                logger.error(f"Failed to load voice profiles: {e}")
                self._profiles_cache = {}
        else:
            self._profiles_cache = {}
        
        return self._profiles_cache
    
    def save_profiles(self, profiles: Dict[str, Dict[str, Any]]) -> bool:
        """Save voice profiles to disk"""
        try:
            # Create backup before saving
            if self.profiles_file.exists():
                self._create_backup()
            
            with open(self.profiles_file, 'w') as f:
                json.dump(profiles, f, indent=2)
            
            self._profiles_cache = profiles
            return True
        except Exception as e:
            logger.error(f"Failed to save voice profiles: {e}")
            return False
    
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
            metadata: Additional metadata
            
        Returns:
            (success, message) tuple
        """
        try:
            # Validate inputs
            if not profile_name:
                return False, "Profile name cannot be empty"
            
            profiles = self.load_profiles()
            if profile_name in profiles:
                return False, f"Profile '{profile_name}' already exists"
            
            # Validate reference audio
            ref_path = Path(reference_audio_path)
            if not ref_path.exists():
                return False, f"Reference audio not found: {reference_audio_path}"
            
            # Validate audio file
            is_valid, audio_info = self._validate_audio_file(ref_path)
            if not is_valid:
                return False, f"Invalid audio file: {audio_info}"
            
            # Create profile directory
            profile_dir = self.voice_samples_dir / profile_name
            profile_dir.mkdir(exist_ok=True)
            
            # Copy reference audio
            dest_path = profile_dir / f"reference{ref_path.suffix}"
            shutil.copy2(ref_path, dest_path)
            
            # Create profile
            profile = {
                "display_name": display_name or profile_name,
                "reference_audio": str(dest_path),
                "language": language,
                "description": description or "",
                "tags": tags or [],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "metadata": metadata or {},
                "audio_info": audio_info
            }
            
            # Analyze audio characteristics if possible
            if LIBROSA_AVAILABLE:
                characteristics = self._analyze_audio_characteristics(ref_path)
                profile["metadata"].update(characteristics)
            
            # Save profile
            profiles[profile_name] = profile
            if self.save_profiles(profiles):
                logger.info(f"Created voice profile '{profile_name}'")
                return True, f"Successfully created profile '{profile_name}'"
            else:
                return False, "Failed to save profile"
            
        except Exception as e:
            logger.error(f"Error creating profile: {e}")
            return False, f"Error: {str(e)}"
    
    def update_profile(
        self,
        profile_name: str,
        display_name: Optional[str] = None,
        language: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata_update: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """Update an existing voice profile"""
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
            
            profile["updated_at"] = datetime.now().isoformat()
            
            # Save updated profiles
            if self.save_profiles(profiles):
                return True, f"Successfully updated profile '{profile_name}'"
            else:
                return False, "Failed to save profile updates"
            
        except Exception as e:
            logger.error(f"Error updating profile: {e}")
            return False, f"Error: {str(e)}"
    
    def delete_profile(self, profile_name: str) -> Tuple[bool, str]:
        """Delete a voice profile"""
        try:
            profiles = self.load_profiles()
            if profile_name not in profiles:
                return False, f"Profile '{profile_name}' not found"
            
            profile = profiles[profile_name]
            
            # Remove profile directory
            profile_dir = self.voice_samples_dir / profile_name
            if profile_dir.exists():
                shutil.rmtree(profile_dir)
            
            # Remove from profiles
            del profiles[profile_name]
            
            # Save updated profiles
            if self.save_profiles(profiles):
                logger.info(f"Deleted voice profile '{profile_name}'")
                return True, f"Successfully deleted profile '{profile_name}'"
            else:
                return False, "Failed to save profile deletion"
            
        except Exception as e:
            logger.error(f"Error deleting profile: {e}")
            return False, f"Error: {str(e)}"
    
    def list_profiles(self, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        List all voice profiles.
        
        Args:
            tags: Filter by tags (if provided)
            
        Returns:
            List of profile summaries
        """
        profiles = self.load_profiles()
        result = []
        
        for name, data in profiles.items():
            # Filter by tags if specified
            if tags:
                profile_tags = set(data.get("tags", []))
                if not any(tag in profile_tags for tag in tags):
                    continue
            
            summary = {
                "name": name,
                "display_name": data.get("display_name", name),
                "language": data.get("language", "unknown"),
                "description": data.get("description", ""),
                "tags": data.get("tags", []),
                "created_at": data.get("created_at", "unknown"),
                "has_reference": bool(data.get("reference_audio")),
                "audio_duration": data.get("audio_info", {}).get("duration", 0)
            }
            result.append(summary)
        
        # Sort by display name
        result.sort(key=lambda x: x["display_name"].lower())
        return result
    
    def get_profile(self, profile_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific voice profile"""
        profiles = self.load_profiles()
        return profiles.get(profile_name)
    
    def export_profile(self, profile_name: str, export_path: str) -> Tuple[bool, str]:
        """
        Export a voice profile with its reference audio.
        
        Args:
            profile_name: Profile to export
            export_path: Directory to export to
            
        Returns:
            (success, message) tuple
        """
        try:
            profiles = self.load_profiles()
            if profile_name not in profiles:
                return False, f"Profile '{profile_name}' not found"
            
            profile = profiles[profile_name]
            export_dir = Path(export_path)
            export_dir.mkdir(parents=True, exist_ok=True)
            
            # Create export package directory
            package_dir = export_dir / f"higgs_voice_{profile_name}"
            package_dir.mkdir(exist_ok=True)
            
            # Copy reference audio
            if "reference_audio" in profile:
                ref_path = Path(profile["reference_audio"])
                if ref_path.exists():
                    dest_audio = package_dir / ref_path.name
                    shutil.copy2(ref_path, dest_audio)
                    
                    # Update path in exported profile
                    export_profile = profile.copy()
                    export_profile["reference_audio"] = ref_path.name
                else:
                    export_profile = profile.copy()
                    export_profile.pop("reference_audio", None)
            else:
                export_profile = profile.copy()
            
            # Save profile metadata
            profile_file = package_dir / "profile.json"
            with open(profile_file, 'w') as f:
                json.dump(export_profile, f, indent=2)
            
            # Create README
            readme_path = package_dir / "README.txt"
            with open(readme_path, 'w') as f:
                f.write(f"Higgs Audio Voice Profile: {profile_name}\n")
                f.write(f"Display Name: {profile.get('display_name', profile_name)}\n")
                f.write(f"Language: {profile.get('language', 'unknown')}\n")
                f.write(f"Created: {profile.get('created_at', 'unknown')}\n")
                f.write(f"\nDescription:\n{profile.get('description', 'No description')}\n")
                f.write(f"\nTo import this profile, use the import_profile function.\n")
            
            logger.info(f"Exported profile '{profile_name}' to {package_dir}")
            return True, f"Successfully exported to {package_dir}"
            
        except Exception as e:
            logger.error(f"Error exporting profile: {e}")
            return False, f"Error: {str(e)}"
    
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
        try:
            import_path = Path(import_path)
            
            # Determine profile file location
            if import_path.is_file() and import_path.name == "profile.json":
                profile_file = import_path
                package_dir = import_path.parent
            elif import_path.is_dir():
                profile_file = import_path / "profile.json"
                package_dir = import_path
            else:
                return False, "Invalid import path. Expected profile.json or package directory"
            
            if not profile_file.exists():
                return False, "profile.json not found in import package"
            
            # Load profile data
            with open(profile_file, 'r') as f:
                import_profile = json.load(f)
            
            # Determine profile name
            if not profile_name:
                # Try to extract from directory name
                if package_dir.name.startswith("higgs_voice_"):
                    profile_name = package_dir.name[12:]  # Remove "higgs_voice_" prefix
                else:
                    profile_name = f"imported_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Check if profile exists
            profiles = self.load_profiles()
            if profile_name in profiles and not overwrite:
                return False, f"Profile '{profile_name}' already exists. Use overwrite=True to replace"
            
            # Create profile directory
            profile_dir = self.voice_samples_dir / profile_name
            profile_dir.mkdir(exist_ok=True)
            
            # Copy reference audio if exists
            if "reference_audio" in import_profile:
                ref_filename = import_profile["reference_audio"]
                source_audio = package_dir / ref_filename
                if source_audio.exists():
                    dest_audio = profile_dir / ref_filename
                    shutil.copy2(source_audio, dest_audio)
                    import_profile["reference_audio"] = str(dest_audio)
                else:
                    logger.warning(f"Reference audio not found: {ref_filename}")
                    import_profile.pop("reference_audio", None)
            
            # Update timestamps
            import_profile["imported_at"] = datetime.now().isoformat()
            if "updated_at" not in import_profile:
                import_profile["updated_at"] = import_profile.get("created_at", datetime.now().isoformat())
            
            # Save profile
            profiles[profile_name] = import_profile
            if self.save_profiles(profiles):
                logger.info(f"Imported voice profile '{profile_name}'")
                return True, f"Successfully imported profile '{profile_name}'"
            else:
                return False, "Failed to save imported profile"
            
        except Exception as e:
            logger.error(f"Error importing profile: {e}")
            return False, f"Error: {str(e)}"
    
    def _validate_audio_file(self, audio_path: Path) -> Tuple[bool, Dict[str, Any]]:
        """Validate and get info about audio file"""
        try:
            info = {"path": str(audio_path), "size_mb": audio_path.stat().st_size / (1024 * 1024)}
            
            if SOUNDFILE_AVAILABLE:
                # Get audio info using soundfile
                with sf.SoundFile(audio_path) as f:
                    info.update({
                        "duration": f.frames / f.samplerate,
                        "sample_rate": f.samplerate,
                        "channels": f.channels,
                        "format": f.format,
                        "subtype": f.subtype
                    })
                
                # Validate duration
                if info["duration"] > 300:  # 5 minutes max
                    return False, {"error": "Audio file too long (max 5 minutes)"}
                
                return True, info
            else:
                # Basic validation without soundfile
                valid_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
                if audio_path.suffix.lower() not in valid_extensions:
                    return False, {"error": f"Unsupported format: {audio_path.suffix}"}
                
                # Check file size
                if info["size_mb"] > 100:
                    return False, {"error": "Audio file too large (max 100MB)"}
                
                return True, info
                
        except Exception as e:
            return False, {"error": str(e)}
    
    def _analyze_audio_characteristics(self, audio_path: Path) -> Dict[str, Any]:
        """Analyze audio characteristics using librosa"""
        characteristics = {}
        
        if not LIBROSA_AVAILABLE:
            return characteristics
        
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            
            # Basic features
            characteristics["analyzed_sample_rate"] = sr
            characteristics["analyzed_duration"] = len(y) / sr
            
            # Pitch analysis
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                characteristics["mean_pitch_hz"] = float(np.mean(pitch_values))
                characteristics["pitch_std_hz"] = float(np.std(pitch_values))
            
            # Tempo analysis
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            characteristics["estimated_tempo_bpm"] = float(tempo)
            
            # Energy analysis
            rms = librosa.feature.rms(y=y)
            characteristics["mean_energy"] = float(np.mean(rms))
            characteristics["energy_variance"] = float(np.var(rms))
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            characteristics["mean_spectral_centroid"] = float(np.mean(spectral_centroids))
            
        except Exception as e:
            logger.warning(f"Audio analysis failed: {e}")
        
        return characteristics
    
    def _create_backup(self):
        """Create backup of current profiles"""
        try:
            if self.profiles_file.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_file = self.backup_dir / f"voice_profiles_backup_{timestamp}.json"
                shutil.copy2(self.profiles_file, backup_file)
                
                # Keep only last 10 backups
                backups = sorted(self.backup_dir.glob("voice_profiles_backup_*.json"))
                if len(backups) > 10:
                    for old_backup in backups[:-10]:
                        old_backup.unlink()
                        
        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")
    
    def restore_from_backup(self, backup_file: Optional[str] = None) -> Tuple[bool, str]:
        """Restore profiles from backup"""
        try:
            if backup_file:
                backup_path = Path(backup_file)
            else:
                # Use most recent backup
                backups = sorted(self.backup_dir.glob("voice_profiles_backup_*.json"))
                if not backups:
                    return False, "No backups found"
                backup_path = backups[-1]
            
            if not backup_path.exists():
                return False, f"Backup file not found: {backup_path}"
            
            # Load backup
            with open(backup_path, 'r') as f:
                backup_profiles = json.load(f)
            
            # Save as current profiles
            if self.save_profiles(backup_profiles):
                return True, f"Restored from backup: {backup_path.name}"
            else:
                return False, "Failed to save restored profiles"
                
        except Exception as e:
            logger.error(f"Error restoring backup: {e}")
            return False, f"Error: {str(e)}"


# Utility functions for command-line usage
async def create_voice_profile_cli(
    manager: HiggsVoiceProfileManager,
    profile_name: str,
    audio_path: str,
    display_name: Optional[str] = None,
    language: str = "en",
    description: Optional[str] = None
):
    """Command-line interface for creating voice profile"""
    success, message = manager.create_profile(
        profile_name=profile_name,
        reference_audio_path=audio_path,
        display_name=display_name,
        language=language,
        description=description
    )
    print(message)
    return success


async def list_voice_profiles_cli(manager: HiggsVoiceProfileManager, tags: Optional[List[str]] = None):
    """Command-line interface for listing voice profiles"""
    profiles = manager.list_profiles(tags=tags)
    
    if not profiles:
        print("No voice profiles found")
        return
    
    print(f"\nFound {len(profiles)} voice profile(s):\n")
    for profile in profiles:
        print(f"  {profile['name']} - {profile['display_name']}")
        print(f"    Language: {profile['language']}")
        if profile['description']:
            print(f"    Description: {profile['description']}")
        if profile['tags']:
            print(f"    Tags: {', '.join(profile['tags'])}")
        if profile['audio_duration'] > 0:
            print(f"    Duration: {profile['audio_duration']:.1f}s")
        print()


# Example usage
if __name__ == "__main__":
    # Example of using the voice manager
    async def main():
        manager = HiggsVoiceProfileManager(Path("~/.config/tldw_cli/higgs_voices").expanduser())
        
        # List profiles
        await list_voice_profiles_cli(manager)
        
        # Example of creating a profile (commented out)
        # await create_voice_profile_cli(
        #     manager,
        #     profile_name="my_voice",
        #     audio_path="/path/to/reference.wav",
        #     display_name="My Custom Voice",
        #     description="A friendly voice for general use"
        # )
    
    asyncio.run(main())

#
# End of higgs_voice_manager.py
#######################################################################################################################