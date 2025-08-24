"""
Storage Adapters for Tamagotchi State Persistence

Provides multiple storage backends for saving tamagotchi state with recovery support.
"""

from abc import ABC, abstractmethod
import json
import sqlite3
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import logging

# Import validators for state recovery
try:
    from .validators import StateValidator
except ImportError:
    # Fallback if validators module not available
    StateValidator = None

logger = logging.getLogger(__name__)


class StorageAdapter(ABC):
    """
    Abstract base class for storage implementations.
    
    All storage adapters must implement load, save, and delete methods.
    Includes state validation and recovery support.
    """
    
    def __init__(self, enable_recovery: bool = True):
        """
        Initialize storage adapter.
        
        Args:
            enable_recovery: Whether to enable automatic state recovery
        """
        self.enable_recovery = enable_recovery
    
    @abstractmethod
    def load(self, pet_id: str) -> Optional[Dict[str, Any]]:
        """
        Load pet state from storage.
        
        Args:
            pet_id: Unique identifier for the pet
        
        Returns:
            Dictionary of pet state or None if not found
        """
        pass
    
    @abstractmethod
    def save(self, pet_id: str, state: Dict[str, Any]) -> bool:
        """
        Save pet state to storage.
        
        Args:
            pet_id: Unique identifier for the pet
            state: Dictionary of pet state
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def delete(self, pet_id: str) -> bool:
        """
        Delete pet data from storage.
        
        Args:
            pet_id: Unique identifier for the pet
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    def list_pets(self) -> list[str]:
        """
        List all stored pet IDs.
        
        Returns:
            List of pet IDs
        """
        return []
    
    def load_with_recovery(self, pet_id: str, default_name: str = "Pet") -> Optional[Dict[str, Any]]:
        """
        Load pet state with automatic recovery on corruption.
        
        Args:
            pet_id: Unique identifier for the pet
            default_name: Default name for recovery
        
        Returns:
            Valid state dictionary or None
        """
        if not self.enable_recovery or not StateValidator:
            return self.load(pet_id)
        
        try:
            state = self.load(pet_id)
            if state is None:
                return None
            
            # Validate the loaded state
            is_valid, error = StateValidator.validate_state(state)
            
            if is_valid:
                return state
            else:
                logger.warning(f"Corrupted state for {pet_id}: {error}")
                
                # Try to repair the state
                repaired = StateValidator.repair_state(state, default_name)
                logger.info(f"State repaired for {pet_id}")
                
                # Save the repaired state
                if self.save(pet_id, repaired):
                    logger.info(f"Repaired state saved for {pet_id}")
                
                return repaired
                
        except Exception as e:
            logger.error(f"Error loading state for {pet_id}: {e}")
            
            # Create default state as fallback
            if StateValidator:
                return StateValidator.create_default_state(default_name)
            return None
    
    def save_with_backup(self, pet_id: str, state: Dict[str, Any]) -> bool:
        """
        Save pet state with backup of previous state.
        
        Args:
            pet_id: Unique identifier for the pet
            state: Dictionary of pet state
        
        Returns:
            True if successful, False otherwise
        """
        # This is overridden in concrete implementations
        return self.save(pet_id, state)


class MemoryStorage(StorageAdapter):
    """
    In-memory storage for testing and temporary use.
    
    Data is lost when the application closes.
    """
    
    def __init__(self, enable_recovery: bool = True):
        """Initialize empty in-memory storage."""
        super().__init__(enable_recovery)
        self.data: Dict[str, Dict[str, Any]] = {}
    
    def load(self, pet_id: str) -> Optional[Dict[str, Any]]:
        """Load pet state from memory."""
        return self.data.get(pet_id)
    
    def save(self, pet_id: str, state: Dict[str, Any]) -> bool:
        """Save pet state to memory."""
        self.data[pet_id] = state.copy()
        return True
    
    def delete(self, pet_id: str) -> bool:
        """Delete pet from memory."""
        if pet_id in self.data:
            del self.data[pet_id]
            return True
        return False
    
    def list_pets(self) -> list[str]:
        """List all pet IDs in memory."""
        return list(self.data.keys())


class JSONStorage(StorageAdapter):
    """
    JSON file storage implementation with backup support.
    
    Stores all pets in a single JSON file with automatic backups.
    """
    
    def __init__(self, filepath: str, enable_recovery: bool = True, max_backups: int = 3):
        """
        Initialize JSON storage.
        
        Args:
            filepath: Path to JSON file
            enable_recovery: Whether to enable state recovery
            max_backups: Maximum number of backup files to keep
        """
        super().__init__(enable_recovery)
        self.filepath = Path(filepath).expanduser()
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.max_backups = max_backups
        
        # Initialize file if it doesn't exist
        if not self.filepath.exists():
            self._write_data({})
    
    def _read_data(self) -> Dict[str, Dict[str, Any]]:
        """Read all data from JSON file."""
        try:
            if self.filepath.exists():
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading JSON storage: {e}")
        return {}
    
    def _write_data(self, data: Dict[str, Dict[str, Any]]) -> bool:
        """Write all data to JSON file."""
        try:
            # Write to temporary file first for safety
            temp_file = self.filepath.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Atomic replace
            temp_file.replace(self.filepath)
            return True
        except IOError as e:
            print(f"Error writing JSON storage: {e}")
            return False
    
    def load(self, pet_id: str) -> Optional[Dict[str, Any]]:
        """Load pet state from JSON file."""
        data = self._read_data()
        return data.get(pet_id)
    
    def _create_backup(self) -> None:
        """Create a backup of the current JSON file."""
        if not self.filepath.exists():
            return
        
        try:
            # Create backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.filepath.with_suffix(f'.backup_{timestamp}.json')
            
            # Copy current file to backup
            shutil.copy2(self.filepath, backup_path)
            
            # Clean up old backups
            self._cleanup_old_backups()
            
        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")
    
    def _cleanup_old_backups(self) -> None:
        """Remove old backup files exceeding max_backups limit."""
        try:
            # Find all backup files
            backup_pattern = f"{self.filepath.stem}.backup_*.json"
            backups = sorted(self.filepath.parent.glob(backup_pattern))
            
            # Remove oldest backups if exceeding limit
            while len(backups) > self.max_backups:
                oldest = backups.pop(0)
                oldest.unlink()
                logger.debug(f"Removed old backup: {oldest}")
                
        except Exception as e:
            logger.warning(f"Failed to cleanup backups: {e}")
    
    def save(self, pet_id: str, state: Dict[str, Any]) -> bool:
        """Save pet state to JSON file with backup."""
        # Create backup before saving
        if self.filepath.exists():
            self._create_backup()
        
        data = self._read_data()
        
        # Validate state before saving if recovery is enabled
        if self.enable_recovery and StateValidator:
            is_valid, error = StateValidator.validate_state(state)
            if not is_valid:
                logger.warning(f"Attempting to save invalid state: {error}")
                # Try to repair before saving
                state = StateValidator.repair_state(state, state.get('name', 'Pet'))
        
        # Add timestamp
        state_with_timestamp = state.copy()
        state_with_timestamp['last_saved'] = datetime.now().isoformat()
        
        data[pet_id] = state_with_timestamp
        return self._write_data(data)
    
    def delete(self, pet_id: str) -> bool:
        """Delete pet from JSON file."""
        data = self._read_data()
        if pet_id in data:
            del data[pet_id]
            return self._write_data(data)
        return False
    
    def list_pets(self) -> list[str]:
        """List all pet IDs in JSON file."""
        data = self._read_data()
        return list(data.keys())


class SQLiteStorage(StorageAdapter):
    """
    SQLite database storage implementation with recovery support.
    
    Provides robust storage with better performance for multiple pets.
    """
    
    def __init__(self, db_path: str, enable_recovery: bool = True):
        """
        Initialize SQLite storage.
        
        Args:
            db_path: Path to SQLite database file
            enable_recovery: Whether to enable state recovery
        """
        super().__init__(enable_recovery)
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tamagotchis (
                    pet_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    happiness REAL DEFAULT 50,
                    hunger REAL DEFAULT 50,
                    energy REAL DEFAULT 50,
                    health REAL DEFAULT 100,
                    age REAL DEFAULT 0,
                    personality TEXT DEFAULT 'balanced',
                    is_alive BOOLEAN DEFAULT 1,
                    total_interactions INTEGER DEFAULT 0,
                    sprite_theme TEXT DEFAULT 'emoji',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    extra_data TEXT  -- JSON field for additional data
                )
            """)
            
            # Create index for faster queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tamagotchis_updated 
                ON tamagotchis(updated_at)
            """)
    
    def load(self, pet_id: str) -> Optional[Dict[str, Any]]:
        """Load pet state from database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM tamagotchis WHERE pet_id = ?",
                (pet_id,)
            )
            row = cursor.fetchone()
            
            if row:
                state = dict(row)
                
                # Parse extra_data JSON if present
                if state.get('extra_data'):
                    try:
                        extra = json.loads(state['extra_data'])
                        state.update(extra)
                    except json.JSONDecodeError:
                        pass
                
                # Remove internal fields
                state.pop('extra_data', None)
                state.pop('created_at', None)
                state.pop('updated_at', None)
                
                return state
        
        return None
    
    def save(self, pet_id: str, state: Dict[str, Any]) -> bool:
        """Save pet state to database."""
        try:
            # Separate known fields from extra data
            known_fields = {
                'name', 'happiness', 'hunger', 'energy', 'health',
                'age', 'personality', 'is_alive', 'total_interactions',
                'sprite_theme'
            }
            
            # Extract known fields
            db_fields = {k: state[k] for k in known_fields if k in state}
            db_fields['pet_id'] = pet_id
            
            # Store remaining fields as JSON
            extra_data = {k: v for k, v in state.items() 
                         if k not in known_fields and k != 'pet_id'}
            
            if extra_data:
                db_fields['extra_data'] = json.dumps(extra_data)
            else:
                db_fields['extra_data'] = None
            
            with sqlite3.connect(self.db_path) as conn:
                # Build query dynamically based on available fields
                fields = list(db_fields.keys())
                placeholders = ['?' for _ in fields]
                
                # Use INSERT OR REPLACE for upsert
                query = f"""
                    INSERT OR REPLACE INTO tamagotchis 
                    ({', '.join(fields)}, updated_at)
                    VALUES ({', '.join(placeholders)}, CURRENT_TIMESTAMP)
                """
                
                conn.execute(query, list(db_fields.values()))
            
            return True
            
        except Exception as e:
            print(f"Error saving to SQLite: {e}")
            return False
    
    def delete(self, pet_id: str) -> bool:
        """Delete pet from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM tamagotchis WHERE pet_id = ?",
                    (pet_id,)
                )
                return cursor.rowcount > 0
        except Exception as e:
            print(f"Error deleting from SQLite: {e}")
            return False
    
    def list_pets(self) -> list[str]:
        """List all pet IDs in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT pet_id FROM tamagotchis ORDER BY updated_at DESC"
                )
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            print(f"Error listing pets: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about stored pets.
        
        Returns:
            Dictionary with statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                stats = {}
                
                # Total pets
                cursor = conn.execute("SELECT COUNT(*) FROM tamagotchis")
                stats['total_pets'] = cursor.fetchone()[0]
                
                # Alive pets
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM tamagotchis WHERE is_alive = 1"
                )
                stats['alive_pets'] = cursor.fetchone()[0]
                
                # Average stats
                cursor = conn.execute("""
                    SELECT 
                        AVG(happiness) as avg_happiness,
                        AVG(hunger) as avg_hunger,
                        AVG(energy) as avg_energy,
                        AVG(health) as avg_health,
                        AVG(age) as avg_age
                    FROM tamagotchis
                    WHERE is_alive = 1
                """)
                row = cursor.fetchone()
                if row[0] is not None:
                    stats['averages'] = {
                        'happiness': round(row[0], 1),
                        'hunger': round(row[1], 1),
                        'energy': round(row[2], 1),
                        'health': round(row[3], 1),
                        'age': round(row[4], 1)
                    }
                
                # Most common personality
                cursor = conn.execute("""
                    SELECT personality, COUNT(*) as count
                    FROM tamagotchis
                    GROUP BY personality
                    ORDER BY count DESC
                    LIMIT 1
                """)
                row = cursor.fetchone()
                if row:
                    stats['most_common_personality'] = row[0]
                
                return stats
                
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {}


class ConfigFileStorage(JSONStorage):
    """
    Storage adapter that uses the application's config directory.
    
    Automatically determines the appropriate config location based on OS.
    """
    
    def __init__(self, app_name: str = "tldw_chatbook"):
        """
        Initialize config file storage.
        
        Args:
            app_name: Application name for config directory
        """
        import os
        
        # Determine config directory based on OS
        if os.name == 'nt':  # Windows
            config_dir = Path(os.environ.get('APPDATA', '~')) / app_name
        else:  # Unix-like (Linux, macOS)
            config_dir = Path('~/.config') / app_name
        
        config_dir = config_dir.expanduser()
        filepath = config_dir / 'tamagotchi_pets.json'
        
        super().__init__(str(filepath))