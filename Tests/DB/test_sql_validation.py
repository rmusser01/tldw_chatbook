"""
Unit tests for SQL validation module.
"""

import pytest

from tldw_chatbook.DB.sql_validation import (
    validate_identifier, validate_table_name, validate_column_name,
    validate_column_list, validate_link_table, get_safe_table_name,
    get_safe_column_name, escape_identifier
)


class TestValidateIdentifier:
    """Test cases for validate_identifier function."""
    
    def test_valid_identifiers(self):
        """Test that valid SQL identifiers are accepted."""
        valid_identifiers = [
            "table_name",
            "column_name",
            "MyTable",
            "user_123",
            "таблица",  # Cyrillic
            "表",       # Chinese
            "column_name_with_underscores",
            "_starting_with_underscore",
            "CamelCaseTable"
        ]
        
        for identifier in valid_identifiers:
            assert validate_identifier(identifier) is True
    
    def test_invalid_identifiers(self):
        """Test that invalid SQL identifiers are rejected."""
        # Empty identifier
        assert validate_identifier("") is False
        
        # Too long (over 64 characters)
        assert validate_identifier("a" * 65) is False
        
        # Contains invalid characters
        assert validate_identifier("table-name") is False  # Hyphen
        assert validate_identifier("table name") is False  # Space
        assert validate_identifier("table;name") is False  # Semicolon
        assert validate_identifier("table'name") is False  # Quote
        assert validate_identifier("table\"name") is False  # Double quote
        
        # SQL keywords
        reserved_keywords = ["SELECT", "FROM", "WHERE", "DROP", "INSERT", "UPDATE", "DELETE"]
        for keyword in reserved_keywords:
            assert validate_identifier(keyword) is False
            assert validate_identifier(keyword.lower()) is False


class TestValidateTableName:
    """Test cases for validate_table_name function."""
    
    def test_valid_chachanotes_tables(self):
        """Test valid table names for chachanotes database."""
        valid_tables = [
            'character_cards', 'conversations', 'messages', 'notes',
            'keywords', 'conversation_keywords', 'collection_keywords',
            'note_keywords', 'sync_log'
        ]
        
        for table in valid_tables:
            assert validate_table_name(table, 'chachanotes') is True
    
    def test_valid_media_tables(self):
        """Test valid table names for media database."""
        valid_tables = [
            'Media', 'Keywords', 'MediaKeywords', 'MediaVersion',
            'MediaModifications', 'UnvectorizedMediaChunks', 'DocumentVersions',
            'sync_log', 'Media_fts', 'MediaChunks'
        ]
        
        for table in valid_tables:
            assert validate_table_name(table, 'media') is True
    
    def test_valid_prompts_tables(self):
        """Test valid table names for prompts database."""
        valid_tables = [
            'Prompts', 'Keywords', 'PromptKeywords', 'sync_log',
            'Prompts_fts', 'Keywords_fts'
        ]
        
        for table in valid_tables:
            assert validate_table_name(table, 'prompts') is True
    
    def test_invalid_table_names(self):
        """Test that invalid table names are rejected."""
        # Table not in whitelist
        assert validate_table_name('users', 'chachanotes') is False
        assert validate_table_name('invalid_table', 'media') is False
        
        # Invalid identifier
        assert validate_table_name('', 'chachanotes') is False
        assert validate_table_name('SELECT', 'media') is False
        assert validate_table_name('table;drop', 'prompts') is False
        
        # Unknown database type
        assert validate_table_name('Media', 'unknown_db') is False


class TestValidateColumnName:
    """Test cases for validate_column_name function."""
    
    def test_valid_column_names_with_table(self):
        """Test valid column names for specific tables."""
        # Character cards columns
        char_columns = ['id', 'uuid', 'name', 'description', 'personality', 'created_at']
        for col in char_columns:
            assert validate_column_name(col, 'character_cards') is True
        
        # Media columns
        media_columns = ['id', 'uuid', 'title', 'content', 'url', 'author']
        for col in media_columns:
            assert validate_column_name(col, 'Media') is True
        
        # Invalid column for table
        assert validate_column_name('invalid_column', 'character_cards') is False
    
    def test_valid_column_names_without_table(self):
        """Test column validation without table context."""
        valid_columns = [
            'id', 'uuid', 'name', 'created_at', 'last_modified',
            'column_name', 'user_id', 'is_active'
        ]
        
        for col in valid_columns:
            assert validate_column_name(col) is True
    
    def test_invalid_column_names(self):
        """Test that invalid column names are rejected."""
        assert validate_column_name('') is False
        assert validate_column_name('column-name') is False
        assert validate_column_name('DROP') is False
        assert validate_column_name('column;name') is False


class TestValidateColumnList:
    """Test cases for validate_column_list function."""
    
    def test_valid_column_list(self):
        """Test that valid column lists pass validation."""
        columns = ['id', 'name', 'created_at', 'last_modified']
        assert validate_column_list(columns) is True
        
        # With table context
        char_columns = ['id', 'name', 'personality']
        assert validate_column_list(char_columns, 'character_cards') is True
    
    def test_invalid_column_in_list(self):
        """Test that lists with invalid columns are rejected."""
        # One invalid column
        columns = ['id', 'name', 'DROP', 'created_at']
        assert validate_column_list(columns) is False
        
        # Invalid for specific table
        columns = ['id', 'invalid_column']
        assert validate_column_list(columns, 'character_cards') is False
    
    def test_empty_list(self):
        """Test that empty column list is handled."""
        assert validate_column_list([]) is True


class TestValidateLinkTable:
    """Test cases for validate_link_table function."""
    
    def test_valid_link_tables(self):
        """Test valid link table configurations."""
        assert validate_link_table('conversation_keywords', 'conversation_id', 'keyword_id') is True
        assert validate_link_table('note_keywords', 'note_id', 'keyword_id') is True
        assert validate_link_table('MediaKeywords', 'media_id', 'keyword_id') is True
        assert validate_link_table('PromptKeywords', 'prompt_id', 'keyword_id') is True
    
    def test_invalid_link_tables(self):
        """Test invalid link table configurations."""
        # Unknown link table
        assert validate_link_table('invalid_links', 'id1', 'id2') is False
        
        # Wrong column names
        assert validate_link_table('conversation_keywords', 'wrong_id', 'keyword_id') is False
        assert validate_link_table('MediaKeywords', 'media_id', 'wrong_id') is False
        
        # Swapped column names
        assert validate_link_table('conversation_keywords', 'keyword_id', 'conversation_id') is False


class TestGetSafeFunctions:
    """Test cases for get_safe_* functions."""
    
    def test_get_safe_table_name(self):
        """Test get_safe_table_name function."""
        # Valid table
        assert get_safe_table_name('character_cards', 'chachanotes') == 'character_cards'
        assert get_safe_table_name('Media', 'media') == 'Media'
        
        # Invalid table
        assert get_safe_table_name('invalid_table', 'chachanotes') is None
        assert get_safe_table_name('DROP', 'media') is None
    
    def test_get_safe_column_name(self):
        """Test get_safe_column_name function."""
        # Valid column
        assert get_safe_column_name('id') == 'id'
        assert get_safe_column_name('created_at') == 'created_at'
        
        # With table context
        assert get_safe_column_name('name', 'character_cards') == 'name'
        
        # Invalid column
        assert get_safe_column_name('DROP') is None
        assert get_safe_column_name('column;name') is None
        assert get_safe_column_name('invalid_col', 'character_cards') is None


class TestEscapeIdentifier:
    """Test cases for escape_identifier function."""
    
    def test_escape_simple_identifier(self):
        """Test escaping simple identifiers."""
        assert escape_identifier('table_name') == '"table_name"'
        assert escape_identifier('column') == '"column"'
    
    def test_escape_identifier_with_quotes(self):
        """Test escaping identifiers that contain quotes."""
        assert escape_identifier('table"name') == '"table""name"'
        assert escape_identifier('"quoted"') == '"""quoted"""'
    
    def test_escape_unicode_identifier(self):
        """Test escaping Unicode identifiers."""
        assert escape_identifier('таблица') == '"таблица"'
        assert escape_identifier('表') == '"表"'


class TestSQLValidationIntegration:
    """Integration tests for SQL validation."""
    
    def test_validate_dynamic_query_components(self):
        """Test validation of components for dynamic query construction."""
        # Simulate validating components for a dynamic query
        table = 'character_cards'
        columns = ['id', 'name', 'personality']
        pk_column = 'id'
        
        # Validate all components
        assert validate_table_name(table, 'chachanotes') is True
        assert validate_column_list(columns, table) is True
        assert validate_column_name(pk_column, table) is True
        
        # Try with invalid components
        assert validate_table_name('users; DROP TABLE--', 'chachanotes') is False
        assert validate_column_name('*', table) is False
    
    def test_unicode_support(self):
        """Test that Unicode identifiers are properly supported."""
        # Various Unicode scripts
        unicode_identifiers = [
            "用户表",      # Chinese
            "таблица",    # Cyrillic
            "テーブル",    # Japanese
            "사용자",      # Korean
            "πίνακας",    # Greek
            "جدول",       # Arabic
            "טבלה",       # Hebrew
        ]
        
        for identifier in unicode_identifiers:
            assert validate_identifier(identifier) is True