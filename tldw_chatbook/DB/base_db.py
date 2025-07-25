# base_db.py
# Description: Base class for standardized database path handling
#
"""
base_db.py
----------

Base class that provides standardized path handling for all database modules.
This ensures consistent behavior across all DB classes for:
- Path type handling (str vs Path)
- Memory database special case (':memory:')
- Client ID handling
- Directory creation for file-based databases
"""

import sqlite3
from pathlib import Path
from typing import Union, Optional
from abc import ABC, abstractmethod
from loguru import logger


class BaseDB(ABC):
    """
    Base class for all database modules providing standardized path handling.
    
    This class ensures consistent handling of:
    - Union[str, Path] type for db_path
    - Special ':memory:' case for in-memory databases
    - Client ID for multi-client support
    - Automatic directory creation
    """
    
    def __init__(self, db_path: Union[str, Path], client_id: str = "default", 
                 check_integrity_on_startup: bool = False):
        """
        Initialize the base database with standardized path handling.
        
        Args:
            db_path: Path to the SQLite database file or ':memory:'
            client_id: Client identifier for multi-client support
            check_integrity_on_startup: Whether to run integrity check on startup
        """
        # Standardized path handling
        if isinstance(db_path, Path):
            self.is_memory_db = False
            self.db_path = db_path.resolve()
        else:
            self.is_memory_db = (db_path == ':memory:')
            if self.is_memory_db:
                self.db_path = Path(":memory:")  # Symbolic Path for consistency
            else:
                self.db_path = Path(db_path).resolve()
        
        # Store string representation for SQLite connection
        self.db_path_str = ':memory:' if self.is_memory_db else str(self.db_path)
        
        # Store client ID
        self.client_id = client_id
        
        # Create directory for file-based databases
        if not self.is_memory_db:
            try:
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.error(f"Failed to create database directory {self.db_path.parent}: {e}")
                raise
        
        # Initialize schema (implemented by subclasses)
        self._initialize_schema()
        
        # Run integrity check if requested
        if check_integrity_on_startup and not self.is_memory_db:
            logger.info(f"Running startup integrity check for {self.__class__.__name__}")
            if not self.check_integrity():
                logger.warning(f"Database integrity check failed for {self.db_path_str}. "
                              "Consider running repairs or restoring from backup.")
                # Note: We don't raise an exception here to allow the app to continue
                # with potentially degraded functionality. Subclasses can override
                # this behavior if they need stricter integrity enforcement.
        
        logger.info(f"{self.__class__.__name__} initialized with path: {self.db_path_str} [Client: {self.client_id}]")
    
    @abstractmethod
    def _initialize_schema(self):
        """
        Initialize the database schema.
        Must be implemented by subclasses.
        """
        pass
    
    def _get_connection(self) -> sqlite3.Connection:
        """
        Get a database connection with row factory.
        Can be overridden by subclasses for custom connection handling.
        """
        conn = sqlite3.connect(self.db_path_str)
        conn.row_factory = sqlite3.Row
        return conn
    
    def close(self):
        """
        Close database connections if needed.
        Can be overridden by subclasses.
        """
        pass
    
    def vacuum(self):
        """
        Vacuum the database to reclaim unused space and optimize performance.
        """
        if self.is_memory_db:
            logger.debug("Skipping vacuum for in-memory database")
            return
            
        try:
            conn = self._get_connection()
            conn.execute("VACUUM")
            conn.close()
            logger.info(f"Successfully vacuumed database: {self.db_path_str}")
        except Exception as e:
            logger.error(f"Failed to vacuum database: {e}")
            raise
    
    def check_integrity(self) -> bool:
        """
        Check the integrity of the database.
        
        Returns:
            bool: True if integrity check passes, False otherwise
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()
            conn.close()
            
            is_ok = result and result[0] == "ok"
            if is_ok:
                logger.info(f"Database integrity check passed: {self.db_path_str}")
            else:
                logger.error(f"Database integrity check failed: {self.db_path_str}")
            
            return is_ok
        except Exception as e:
            logger.error(f"Failed to check database integrity: {e}")
            return False