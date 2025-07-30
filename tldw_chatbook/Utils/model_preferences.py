# tldw_chatbook/Utils/model_preferences.py
# Management of recent and favorite models
#
# Imports
from __future__ import annotations
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from collections import deque

# Third-party imports
from loguru import logger

# Configure logger
logger = logger.bind(module="model_preferences")


@dataclass
class ModelUsage:
    """Track model usage statistics."""
    model_id: str
    last_used: datetime
    use_count: int = 1
    is_favorite: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_id": self.model_id,
            "last_used": self.last_used.isoformat(),
            "use_count": self.use_count,
            "is_favorite": self.is_favorite
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelUsage':
        """Create from dictionary."""
        return cls(
            model_id=data["model_id"],
            last_used=datetime.fromisoformat(data["last_used"]),
            use_count=data.get("use_count", 1),
            is_favorite=data.get("is_favorite", False)
        )


class ModelPreferencesManager:
    """Manage recent and favorite embedding models."""
    
    MAX_RECENT_MODELS = 10
    
    def __init__(self, preferences_dir: Optional[Path] = None):
        """Initialize preferences manager.
        
        Args:
            preferences_dir: Directory to store preferences file
        """
        if preferences_dir is None:
            preferences_dir = Path.home() / ".config" / "tldw_cli"
        
        self.preferences_dir = Path(preferences_dir)
        self.preferences_file = self.preferences_dir / "model_preferences.json"
        
        # Create directory if needed
        self.preferences_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing preferences
        self.model_usage: Dict[str, ModelUsage] = {}
        self.recent_models: deque[str] = deque(maxlen=self.MAX_RECENT_MODELS)
        self._load_preferences()
    
    def _load_preferences(self) -> None:
        """Load preferences from file."""
        if self.preferences_file.exists():
            try:
                with open(self.preferences_file, 'r') as f:
                    data = json.load(f)
                
                # Load model usage data
                for model_data in data.get("models", []):
                    usage = ModelUsage.from_dict(model_data)
                    self.model_usage[usage.model_id] = usage
                
                # Load recent models order
                recent = data.get("recent_models", [])
                self.recent_models = deque(recent, maxlen=self.MAX_RECENT_MODELS)
                
                logger.info(f"Loaded preferences for {len(self.model_usage)} models")
            except Exception as e:
                logger.error(f"Failed to load preferences: {e}")
    
    def _save_preferences(self) -> None:
        """Save preferences to file."""
        try:
            data = {
                "models": [usage.to_dict() for usage in self.model_usage.values()],
                "recent_models": list(self.recent_models),
                "version": "1.0"
            }
            
            with open(self.preferences_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.debug("Saved model preferences")
        except Exception as e:
            logger.error(f"Failed to save preferences: {e}")
    
    def record_model_use(self, model_id: str) -> None:
        """Record that a model was used."""
        now = datetime.now()
        
        if model_id in self.model_usage:
            # Update existing
            usage = self.model_usage[model_id]
            usage.last_used = now
            usage.use_count += 1
        else:
            # Create new
            self.model_usage[model_id] = ModelUsage(
                model_id=model_id,
                last_used=now
            )
        
        # Update recent models
        if model_id in self.recent_models:
            self.recent_models.remove(model_id)
        self.recent_models.append(model_id)
        
        self._save_preferences()
    
    def toggle_favorite(self, model_id: str) -> bool:
        """Toggle favorite status for a model.
        
        Returns:
            New favorite status
        """
        if model_id not in self.model_usage:
            # Create entry if doesn't exist
            self.model_usage[model_id] = ModelUsage(
                model_id=model_id,
                last_used=datetime.now(),
                use_count=0,
                is_favorite=True
            )
        else:
            usage = self.model_usage[model_id]
            usage.is_favorite = not usage.is_favorite
        
        self._save_preferences()
        return self.model_usage[model_id].is_favorite
    
    def is_favorite(self, model_id: str) -> bool:
        """Check if a model is marked as favorite."""
        return self.model_usage.get(model_id, ModelUsage(model_id, datetime.now())).is_favorite
    
    def get_recent_models(self, limit: int = 10) -> List[str]:
        """Get list of recently used models."""
        return list(self.recent_models)[-limit:][::-1]  # Most recent first
    
    def get_favorite_models(self) -> List[str]:
        """Get list of favorite models."""
        favorites = [
            model_id 
            for model_id, usage in self.model_usage.items()
            if usage.is_favorite
        ]
        # Sort by last used
        favorites.sort(
            key=lambda m: self.model_usage[m].last_used,
            reverse=True
        )
        return favorites
    
    def get_most_used_models(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get most frequently used models.
        
        Returns:
            List of (model_id, use_count) tuples
        """
        models = [
            (usage.model_id, usage.use_count)
            for usage in self.model_usage.values()
        ]
        models.sort(key=lambda x: x[1], reverse=True)
        return models[:limit]
    
    def get_model_stats(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get usage statistics for a model."""
        if model_id in self.model_usage:
            usage = self.model_usage[model_id]
            return {
                "last_used": usage.last_used.isoformat(),
                "use_count": usage.use_count,
                "is_favorite": usage.is_favorite,
                "days_since_last_use": (datetime.now() - usage.last_used).days
            }
        return None
    
    def clear_recent(self) -> None:
        """Clear recent models list."""
        self.recent_models.clear()
        self._save_preferences()
    
    def remove_model(self, model_id: str) -> None:
        """Remove all data for a model."""
        if model_id in self.model_usage:
            del self.model_usage[model_id]
        if model_id in self.recent_models:
            self.recent_models.remove(model_id)
        self._save_preferences()