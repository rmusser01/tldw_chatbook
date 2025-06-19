"""
SQL identifier validation module for preventing SQL injection in dynamic queries.

This module provides validation for table names, column names, and other SQL identifiers
to ensure they match expected patterns and are safe to use in dynamic SQL construction.
"""

import re
from typing import Dict, Set, Optional, Union
from loguru import logger

# Define valid table names for each database
VALID_TABLES = {
    'chachanotes': {
        'character_cards', 'conversations', 'messages', 'notes', 
        'keywords', 'conversation_keywords', 'collection_keywords', 
        'note_keywords', 'sync_log'
    },
    'media': {
        'Media', 'Keywords', 'MediaKeywords', 'MediaVersion', 
        'MediaModifications', 'UnvectorizedMediaChunks', 'DocumentVersions',
        'IngestionTriggerTracking', 'sync_log', 'Media_fts', 
        'Keywords_fts', 'MediaChunks', 'MediaChunks_fts', 'Transcripts'
    },
    'prompts': {
        'Prompts', 'Keywords', 'PromptKeywords', 'sync_log',
        'Prompts_fts', 'Keywords_fts'
    }
}

# Define valid columns for each table (for most commonly used tables)
VALID_COLUMNS = {
    # ChaChaNotes DB
    'character_cards': {
        'id', 'uuid', 'name', 'alternate_greetings', 'description', 'personality',
        'post_history_instructions', 'first_mes', 'mes_example', 'scenario',
        'system_prompt', 'creator_notes', 'creator', 'character_version', 'avatar',
        'extensions', 'tags', 'created_at', 'last_modified', 'deleted', 'version',
        'deleted_at', 'client_id'
    },
    'conversations': {
        'id', 'uuid', 'character_id', 'title', 'deleted', 'created_at', 
        'last_modified', 'version', 'deleted_at', 'client_id'
    },
    'messages': {
        'id', 'uuid', 'conversation_id', 'sender', 'content', 'created_at',
        'last_modified', 'deleted', 'version', 'deleted_at', 'client_id'
    },
    'notes': {
        'id', 'uuid', 'title', 'content', 'created_at', 'last_modified',
        'deleted', 'version', 'deleted_at', 'client_id'
    },
    'keywords': {
        'id', 'uuid', 'keyword', 'deleted', 'created_at', 'last_modified',
        'version', 'deleted_at', 'client_id'
    },
    
    # Media DB
    'Media': {
        'id', 'uuid', 'title', 'type', 'url', 'content', 'author', 'ingestion_date',
        'last_modified', 'deleted', 'is_trash', 'trash_date', 'transcription_model',
        'vector_processing', 'vector_id', 'book_cover', 'file_hash', 'version',
        'deleted_at', 'client_id'
    },
    'Keywords': {
        'id', 'uuid', 'keyword', 'deleted', 'last_modified', 'version',
        'deleted_at', 'client_id'
    },
    
    # Prompts DB
    'Prompts': {
        'id', 'uuid', 'name', 'system_prompt', 'user_prompt', 'created_at',
        'last_modified', 'deleted', 'version', 'deleted_at', 'client_id'
    }
}

# Link table columns
LINK_TABLE_COLUMNS = {
    'conversation_keywords': {'conversation_id', 'keyword_id', 'created_at'},
    'collection_keywords': {'collection_id', 'keyword_id', 'created_at'},
    'note_keywords': {'note_id', 'keyword_id', 'created_at'},
    'MediaKeywords': {'media_id', 'keyword_id'},
    'PromptKeywords': {'prompt_id', 'keyword_id'}
}

# SQL identifier pattern - allows alphanumeric, underscore, and supports Unicode
# This pattern is designed to be safe while supporting non-English identifiers
SQL_IDENTIFIER_PATTERN = re.compile(r'^[\w\u0080-\uFFFF]+$', re.UNICODE)

# Reserved SQL keywords that should not be used as identifiers
SQL_RESERVED_KEYWORDS = {
    'SELECT', 'FROM', 'WHERE', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE',
    'TABLE', 'INDEX', 'VIEW', 'UNION', 'JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER',
    'ORDER', 'BY', 'GROUP', 'HAVING', 'LIMIT', 'OFFSET', 'AS', 'ON', 'AND', 'OR',
    'NOT', 'NULL', 'PRIMARY', 'KEY', 'FOREIGN', 'REFERENCES', 'CASCADE', 'SET',
    'VALUES', 'INTO', 'EXISTS', 'BETWEEN', 'LIKE', 'IN', 'IS', 'DISTINCT', 'ALL'
}


def validate_identifier(identifier: str, identifier_type: str = "identifier") -> bool:
    """
    Validates a SQL identifier (table name, column name, etc.) for safety.
    
    Args:
        identifier: The SQL identifier to validate
        identifier_type: Type of identifier for logging (e.g., "table", "column")
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not identifier:
        logger.warning(f"Empty {identifier_type} provided")
        return False
        
    # Check length limits
    if len(identifier) > 64:  # Common SQL identifier length limit
        logger.warning(f"{identifier_type} '{identifier}' exceeds maximum length")
        return False
        
    # Check against pattern
    if not SQL_IDENTIFIER_PATTERN.match(identifier):
        logger.warning(f"{identifier_type} '{identifier}' contains invalid characters")
        return False
        
    # Check against reserved keywords
    if identifier.upper() in SQL_RESERVED_KEYWORDS:
        logger.warning(f"{identifier_type} '{identifier}' is a reserved SQL keyword")
        return False
        
    return True


def validate_table_name(table_name: str, db_type: str) -> bool:
    """
    Validates a table name against the whitelist for a specific database type.
    
    Args:
        table_name: The table name to validate
        db_type: The database type ('chachanotes', 'media', or 'prompts')
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not validate_identifier(table_name, "table name"):
        return False
        
    valid_tables = VALID_TABLES.get(db_type, set())
    if table_name not in valid_tables:
        logger.warning(f"Table '{table_name}' not in whitelist for {db_type} database")
        return False
        
    return True


def validate_column_name(column_name: str, table_name: Optional[str] = None) -> bool:
    """
    Validates a column name, optionally against a specific table's schema.
    
    Args:
        column_name: The column name to validate
        table_name: Optional table name to validate against specific schema
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not validate_identifier(column_name, "column name"):
        return False
        
    # If table name provided and we have schema info, validate against it
    if table_name and table_name in VALID_COLUMNS:
        valid_columns = VALID_COLUMNS[table_name]
        if column_name not in valid_columns:
            logger.warning(f"Column '{column_name}' not in schema for table '{table_name}'")
            return False
            
    return True


def validate_column_list(columns: list[str], table_name: Optional[str] = None) -> bool:
    """
    Validates a list of column names.
    
    Args:
        columns: List of column names to validate
        table_name: Optional table name to validate against specific schema
        
    Returns:
        bool: True if all columns are valid, False otherwise
    """
    for column in columns:
        if not validate_column_name(column, table_name):
            return False
    return True


def validate_link_table(table_name: str, col1_name: str, col2_name: str) -> bool:
    """
    Validates a link table and its column names.
    
    Args:
        table_name: The link table name
        col1_name: First column name
        col2_name: Second column name
        
    Returns:
        bool: True if valid, False otherwise
    """
    if table_name not in LINK_TABLE_COLUMNS:
        logger.warning(f"Link table '{table_name}' not recognized")
        return False
        
    valid_columns = LINK_TABLE_COLUMNS[table_name]
    if col1_name not in valid_columns or col2_name not in valid_columns:
        logger.warning(f"Invalid columns for link table '{table_name}': {col1_name}, {col2_name}")
        return False
        
    return True


def get_safe_table_name(table_name: str, db_type: str) -> Optional[str]:
    """
    Returns a validated table name or None if invalid.
    
    Args:
        table_name: The table name to validate
        db_type: The database type
        
    Returns:
        Optional[str]: The table name if valid, None otherwise
    """
    if validate_table_name(table_name, db_type):
        return table_name
    return None


def get_safe_column_name(column_name: str, table_name: Optional[str] = None) -> Optional[str]:
    """
    Returns a validated column name or None if invalid.
    
    Args:
        column_name: The column name to validate
        table_name: Optional table name for schema validation
        
    Returns:
        Optional[str]: The column name if valid, None otherwise
    """
    if validate_column_name(column_name, table_name):
        return column_name
    return None


# Helper function to escape identifiers (as a last resort)
def escape_identifier(identifier: str) -> str:
    """
    Escapes a SQL identifier by wrapping it in double quotes.
    Note: This should only be used after validation, not as a replacement for validation.
    
    Args:
        identifier: The identifier to escape
        
    Returns:
        str: The escaped identifier
    """
    # Replace any existing double quotes with two double quotes (SQL escaping)
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'