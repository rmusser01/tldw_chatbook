# chatterbox_voice_manager.py
# Description: Voice profile management for Chatterbox TTS backend
#
# Imports
import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from loguru import logger

# Local imports
from .voice_manager_base import VoiceManagerBase

#######################################################################################################################
#
# Chatterbox Voice Manager
#

class ChatterboxVoiceManager(VoiceManagerBase):
    """
    Voice profile manager for Chatterbox TTS backend.
    
    Chatterbox uses simple reference audio files for zero-shot voice cloning.
    This manager provides a consistent interface for managing these voice references.
    """
    
    def __init__(self, voice_samples_dir: Path):
        """
        Initialize the Chatterbox voice manager.
        
        Args:
            voice_samples_dir: Directory for storing voice samples and profiles
        """
        super().__init__(voice_samples_dir)
        self.profiles_file = self.voice_samples_dir / "chatterbox_profiles.json"
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
                logger.error(f"Failed to load Chatterbox voice profiles: {e}")
                self._profiles_cache = {}
        else:
            self._profiles_cache = {}
        
        return self._profiles_cache
    
    def save_profiles(self, profiles: Dict[str, Dict[str, Any]]) -> bool:
        """Save voice profiles to disk"""
        try:
            with open(self.profiles_file, 'w') as f:
                json.dump(profiles, f, indent=2)
            
            self._profiles_cache = profiles
            return True
        except Exception as e:
            logger.error(f"Failed to save Chatterbox voice profiles: {e}")
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
        """Create a new voice profile for Chatterbox"""
        try:
            # Validate inputs
            if not profile_name:
                return False, "Profile name cannot be empty"
            
            profiles = self.load_profiles()
            if profile_name in profiles:
                return False, f"Profile '{profile_name}' already exists"
            
            # Validate reference audio
            ref_path = Path(reference_audio_path)
            is_valid, audio_info = self.validate_audio_file(ref_path)
            if not is_valid:
                return False, f"Invalid audio file: {audio_info.get('error', 'Unknown error')}"
            
            # Check audio duration recommendation for Chatterbox (7-20 seconds)
            if metadata and "duration" in metadata:
                duration = metadata["duration"]
                if duration < 7:
                    logger.warning(f"Audio duration ({duration}s) is less than recommended 7 seconds for Chatterbox")
                elif duration > 20:
                    logger.warning(f"Audio duration ({duration}s) exceeds recommended 20 seconds for Chatterbox")
            
            # Create profile directory
            profile_dir = self.voice_samples_dir / profile_name
            profile_dir.mkdir(exist_ok=True)
            
            # Copy reference audio
            dest_path = profile_dir / f"reference{ref_path.suffix}"
            shutil.copy2(ref_path, dest_path)
            
            # Create profile with Chatterbox-specific metadata
            profile = {
                "display_name": display_name or profile_name,
                "reference_audio": str(dest_path),
                "language": language,
                "description": description or "",
                "tags": tags or [],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "backend": "chatterbox",
                "metadata": metadata or {},
                "audio_info": audio_info,
                "chatterbox_settings": {
                    "recommended_duration": "7-20 seconds",
                    "watermarked": True,
                    "supports_emotion": True
                }
            }
            
            # Save profile
            profiles[profile_name] = profile
            if self.save_profiles(profiles):
                logger.info(f"Created Chatterbox voice profile '{profile_name}'")
                return True, f"Successfully created profile '{profile_name}'"
            else:
                return False, "Failed to save profile"
            
        except Exception as e:
            logger.error(f"Error creating Chatterbox profile: {e}")
            return False, f"Error: {str(e)}"
    
    def list_profiles(self, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """List all Chatterbox voice profiles"""
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
                "backend": "chatterbox",
                "has_reference": bool(data.get("reference_audio")),
                "audio_duration": data.get("audio_info", {}).get("duration", 0),
                "watermarked": True  # Chatterbox always watermarks
            }
            result.append(summary)
        
        # Sort by display name
        result.sort(key=lambda x: x["display_name"].lower())
        return result
    
    def get_profile(self, profile_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific Chatterbox voice profile"""
        profiles = self.load_profiles()
        profile = profiles.get(profile_name)
        if profile:
            # Ensure backend field is present
            profile["backend"] = "chatterbox"
        return profile
    
    def delete_profile(self, profile_name: str) -> Tuple[bool, str]:
        """Delete a Chatterbox voice profile"""
        try:
            profiles = self.load_profiles()
            if profile_name not in profiles:
                return False, f"Profile '{profile_name}' not found"
            
            # Remove profile directory
            profile_dir = self.voice_samples_dir / profile_name
            if profile_dir.exists():
                shutil.rmtree(profile_dir)
            
            # Remove from profiles
            del profiles[profile_name]
            
            # Save updated profiles
            if self.save_profiles(profiles):
                logger.info(f"Deleted Chatterbox voice profile '{profile_name}'")
                return True, f"Successfully deleted profile '{profile_name}'"
            else:
                return False, "Failed to save profile deletion"
            
        except Exception as e:
            logger.error(f"Error deleting Chatterbox profile: {e}")
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
        """Update an existing Chatterbox voice profile"""
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
            logger.error(f"Error updating Chatterbox profile: {e}")
            return False, f"Error: {str(e)}"
    
    def export_profile(self, profile_name: str, export_path: str) -> Tuple[bool, str]:
        """Export a Chatterbox voice profile"""
        try:
            profiles = self.load_profiles()
            if profile_name not in profiles:
                return False, f"Profile '{profile_name}' not found"
            
            profile = profiles[profile_name]
            export_dir = Path(export_path)
            export_dir.mkdir(parents=True, exist_ok=True)
            
            # Create export package directory
            package_dir = export_dir / f"chatterbox_voice_{profile_name}"
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
                f.write(f"Chatterbox Voice Profile: {profile_name}\n")
                f.write(f"Display Name: {profile.get('display_name', profile_name)}\n")
                f.write(f"Language: {profile.get('language', 'unknown')}\n")
                f.write(f"Created: {profile.get('created_at', 'unknown')}\n")
                f.write(f"\nDescription:\n{profile.get('description', 'No description')}\n")
                f.write(f"\nNote: Chatterbox audio includes watermarking for responsible AI use.\n")
                f.write(f"\nTo import this profile, use the import_profile function.\n")
            
            logger.info(f"Exported Chatterbox profile '{profile_name}' to {package_dir}")
            return True, f"Successfully exported to {package_dir}"
            
        except Exception as e:
            logger.error(f"Error exporting Chatterbox profile: {e}")
            return False, f"Error: {str(e)}"
    
    def import_profile(
        self,
        import_path: str,
        profile_name: Optional[str] = None,
        overwrite: bool = False
    ) -> Tuple[bool, str]:
        """Import a Chatterbox voice profile"""
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
            
            # Verify this is a Chatterbox profile
            if import_profile.get("backend") != "chatterbox" and "chatterbox_settings" not in import_profile:
                return False, "This does not appear to be a Chatterbox voice profile"
            
            # Determine profile name
            if not profile_name:
                # Try to extract from directory name
                if package_dir.name.startswith("chatterbox_voice_"):
                    profile_name = package_dir.name[17:]  # Remove "chatterbox_voice_" prefix
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
            
            # Update metadata
            import_profile["imported_at"] = datetime.now().isoformat()
            import_profile["backend"] = "chatterbox"
            if "updated_at" not in import_profile:
                import_profile["updated_at"] = import_profile.get("created_at", datetime.now().isoformat())
            
            # Save profile
            profiles[profile_name] = import_profile
            if self.save_profiles(profiles):
                logger.info(f"Imported Chatterbox voice profile '{profile_name}'")
                return True, f"Successfully imported profile '{profile_name}'"
            else:
                return False, "Failed to save imported profile"
            
        except Exception as e:
            logger.error(f"Error importing Chatterbox profile: {e}")
            return False, f"Error: {str(e)}"
    
    def get_supported_features(self) -> List[str]:
        """Get list of supported features for Chatterbox backend"""
        return [
            "basic_profiles",
            "import_export", 
            "zero_shot_cloning",
            "emotion_control",
            "watermarked_output",
            "7_20_second_samples"
        ]
    
    def get_reference_audio_path(self, profile_name: str) -> Optional[str]:
        """
        Get the reference audio path for a profile.
        
        This is a convenience method for Chatterbox integration.
        
        Args:
            profile_name: Profile identifier
            
        Returns:
            Path to reference audio file or None
        """
        profile = self.get_profile(profile_name)
        if profile:
            return profile.get("reference_audio")
        return None

#
# End of chatterbox_voice_manager.py
#######################################################################################################################