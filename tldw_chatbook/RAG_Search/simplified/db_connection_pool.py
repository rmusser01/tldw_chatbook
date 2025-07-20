"""
Simple database connection pool for FTS5 searches.
"""

import sqlite3
from typing import Dict, Any
from loguru import logger


# Global connection pools
_connection_pools: Dict[str, Any] = {}


def get_connection_pool(db_path: str, pool_size: int = 3) -> Any:
    """
    Get or create a connection pool for the given database.
    
    For now, this just returns a simple connection since the old pool
    implementation was removed. This maintains backward compatibility.
    """
    if db_path not in _connection_pools:
        logger.debug(f"Creating connection for database: {db_path}")
        # For backward compatibility, just store the path and pool size
        _connection_pools[db_path] = {
            'path': db_path,
            'pool_size': pool_size
        }
    
    return _connection_pools[db_path]


def close_all_pools():
    """Close all connection pools."""
    _connection_pools.clear()
    logger.debug("Cleared all connection pool references")