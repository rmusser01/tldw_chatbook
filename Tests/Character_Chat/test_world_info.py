"""
Tests for World Info/Lorebook functionality.
"""

import pytest
import json
from typing import Dict, Any, List

from tldw_chatbook.Character_Chat.world_info_processor import WorldInfoProcessor


class TestWorldInfoProcessor:
    """Test cases for the WorldInfoProcessor class."""
    
    @pytest.fixture
    def sample_character_data(self) -> Dict[str, Any]:
        """Create sample character data with world info."""
        return {
            'name': 'Test Character',
            'extensions': {
                'character_book': {
                    'scan_depth': 3,
                    'token_budget': 500,
                    'recursive_scanning': False,
                    'entries': [
                        {
                            'keys': ['Eldoria', 'eldoria'],
                            'content': 'Eldoria is a magical kingdom in the north.',
                            'enabled': True,
                            'position': 'before_char',
                            'insertion_order': 1,
                            'case_sensitive': False,
                            'selective': False
                        },
                        {
                            'keys': ['dragon', 'Dragon', 'dragons', 'Dragons'],
                            'content': 'Dragons are ancient creatures of immense power.',
                            'enabled': True,
                            'position': 'after_char',
                            'insertion_order': 2,
                            'case_sensitive': False,
                            'selective': False
                        },
                        {
                            'keys': ['magic sword'],
                            'content': 'The magic sword can only be wielded by the chosen one.',
                            'enabled': True,
                            'position': 'at_start',
                            'insertion_order': 3,
                            'case_sensitive': False,
                            'selective': True,
                            'secondary_keys': ['chosen one', 'prophecy']
                        },
                        {
                            'keys': ['disabled entry'],
                            'content': 'This should not appear.',
                            'enabled': False,
                            'position': 'before_char',
                            'insertion_order': 4,
                            'case_sensitive': False,
                            'selective': False
                        }
                    ]
                }
            }
        }
    
    @pytest.fixture
    def processor(self, sample_character_data) -> WorldInfoProcessor:
        """Create a WorldInfoProcessor instance."""
        return WorldInfoProcessor(sample_character_data)
    
    def test_initialization(self, processor):
        """Test processor initialization."""
        assert processor is not None
        assert len(processor.entries) == 3  # Only enabled entries
        assert processor.scan_depth == 3
        assert processor.token_budget == 500
        assert processor.recursive_scanning is False
    
    def test_simple_keyword_matching(self, processor):
        """Test basic keyword matching."""
        result = processor.process_messages(
            "I traveled to Eldoria yesterday.",
            []
        )
        
        assert len(result['matched_entries']) == 1
        assert result['matched_entries'][0]['content'] == 'Eldoria is a magical kingdom in the north.'
        assert 'before_char' in result['injections']
    
    def test_case_insensitive_matching(self, processor):
        """Test case-insensitive keyword matching."""
        result = processor.process_messages(
            "I saw a DRAGON flying overhead!",
            []
        )
        
        assert len(result['matched_entries']) == 1
        assert result['matched_entries'][0]['content'] == 'Dragons are ancient creatures of immense power.'
    
    def test_word_boundary_matching(self, processor):
        """Test that keywords match on word boundaries."""
        # Should not match "dragon" in "dragonfly"
        result = processor.process_messages(
            "A dragonfly landed on my hand.",
            []
        )
        
        assert len(result['matched_entries']) == 0
        assert 'tokens_used' in result
    
    def test_multiple_matches(self, processor):
        """Test multiple keyword matches in one message."""
        result = processor.process_messages(
            "In Eldoria, I encountered a dragon near the castle.",
            []
        )
        
        assert len(result['matched_entries']) == 2
        contents = [entry['content'] for entry in result['matched_entries']]
        assert 'Eldoria is a magical kingdom in the north.' in contents
        assert 'Dragons are ancient creatures of immense power.' in contents
    
    def test_selective_activation(self, processor):
        """Test selective activation with secondary keys."""
        # Primary key alone should not trigger selective entry
        result = processor.process_messages(
            "I found a magic sword in the cave.",
            []
        )
        assert len(result['matched_entries']) == 0
        
        # Primary + secondary key should trigger
        result = processor.process_messages(
            "The magic sword awaits the chosen one.",
            []
        )
        assert len(result['matched_entries']) == 1
        assert result['matched_entries'][0]['content'] == 'The magic sword can only be wielded by the chosen one.'
    
    def test_conversation_history_scanning(self, processor):
        """Test scanning conversation history."""
        history = [
            {'role': 'user', 'content': 'Tell me about your homeland.'},
            {'role': 'assistant', 'content': 'I come from Eldoria, a beautiful kingdom.'},
            {'role': 'user', 'content': 'What creatures live there?'}
        ]
        
        result = processor.process_messages(
            "Are there any dangerous ones?",
            history,
            scan_depth=3
        )
        
        assert len(result['matched_entries']) == 1
        assert result['matched_entries'][0]['content'] == 'Eldoria is a magical kingdom in the north.'
    
    def test_position_organization(self, processor):
        """Test organization of entries by position."""
        result = processor.process_messages(
            "In Eldoria, a dragon guards the magic sword of the chosen one.",
            []
        )
        
        injections = result['injections']
        
        # Debug: print what we actually got
        print(f"Matched entries: {len(result['matched_entries'])}")
        for entry in result['matched_entries']:
            print(f"  - {entry.get('content', '')[:50]}... at position: {entry.get('position', 'unknown')}")
        print(f"Injections: {injections}")
        
        # Verify we have the expected entries
        assert len(result['matched_entries']) == 3  # Eldoria, dragon, and magic sword
        
        # Check that injections are organized by position
        assert 'before_char' in injections
        assert len(injections['before_char']) > 0
        assert any('Eldoria' in content for content in injections['before_char'])
        
        assert 'after_char' in injections
        assert len(injections['after_char']) > 0
        assert any('Dragons' in content for content in injections['after_char'])
        
        assert 'at_start' in injections
        assert len(injections['at_start']) > 0
        assert any('magic sword' in content for content in injections['at_start'])
    
    def test_format_injections(self, processor):
        """Test formatting of injections."""
        result = processor.process_messages(
            "Tell me about Eldoria and its dragons.",
            []
        )
        
        formatted = processor.format_injections(result['injections'])
        
        # Only positions with content should be in the formatted result
        assert 'before_char' in formatted  # Eldoria entry
        assert 'after_char' in formatted  # Dragon entry
        assert isinstance(formatted['before_char'], str)
        assert isinstance(formatted['after_char'], str)
        
        # These positions had no matching entries, so they shouldn't be in formatted
        assert 'at_start' not in formatted
        assert 'at_end' not in formatted
    
    def test_empty_character_book(self):
        """Test handling of character without world info."""
        char_data = {'name': 'Test', 'extensions': {}}
        processor = WorldInfoProcessor(char_data)
        
        result = processor.process_messages("Any text here", [])
        assert len(result['matched_entries']) == 0
        assert len(result['injections']) == 0
    
    def test_malformed_entries(self):
        """Test handling of malformed world info entries."""
        char_data = {
            'name': 'Test',
            'extensions': {
                'character_book': {
                    'entries': [
                        {
                            # Missing keys
                            'content': 'This entry has no keys',
                            'enabled': True
                        },
                        {
                            'keys': [],  # Empty keys
                            'content': 'This entry has empty keys',
                            'enabled': True
                        },
                        {
                            'keys': ['valid'],
                            # Missing content
                            'enabled': True
                        }
                    ]
                }
            }
        }
        
        processor = WorldInfoProcessor(char_data)
        # Should handle gracefully without crashing
        assert len(processor.entries) == 1  # Only the valid entry
    
    def test_recursive_scanning(self):
        """Test recursive scanning feature."""
        char_data = {
            'name': 'Test',
            'extensions': {
                'character_book': {
                    'recursive_scanning': True,
                    'entries': [
                        {
                            'keys': ['castle'],
                            'content': 'The castle is protected by a dragon.',
                            'enabled': True,
                            'position': 'before_char'
                        },
                        {
                            'keys': ['dragon'],
                            'content': 'Dragons breathe fire.',
                            'enabled': True,
                            'position': 'before_char'
                        }
                    ]
                }
            }
        }
        
        processor = WorldInfoProcessor(char_data)
        result = processor.process_messages(
            "Tell me about the castle.",
            []
        )
        
        # With recursive scanning, mentioning castle should also trigger dragon
        assert len(result['matched_entries']) == 2
    
    def test_case_sensitive_matching(self):
        """Test case-sensitive keyword matching."""
        char_data = {
            'name': 'Test',
            'extensions': {
                'character_book': {
                    'entries': [
                        {
                            'keys': ['HTML'],
                            'content': 'HTML is a markup language.',
                            'enabled': True,
                            'case_sensitive': True
                        }
                    ]
                }
            }
        }
        
        processor = WorldInfoProcessor(char_data)
        
        # Should match exact case
        result = processor.process_messages("I'm learning HTML.", [])
        assert len(result['matched_entries']) == 1
        
        # Should not match different case
        result = processor.process_messages("I'm learning html.", [])
        assert len(result['matched_entries']) == 0


    def test_token_budget_management(self):
        """Test token budget limiting."""
        char_data = {
            'name': 'Test',
            'extensions': {
                'character_book': {
                    'token_budget': 50,  # Very small budget
                    'entries': [
                        {
                            'keys': ['test'],
                            'content': 'A' * 200,  # ~50 tokens
                            'enabled': True,
                            'position': 'before_char',
                            'insertion_order': 1
                        },
                        {
                            'keys': ['test'],
                            'content': 'B' * 200,  # ~50 tokens
                            'enabled': True,
                            'position': 'before_char',
                            'insertion_order': 2
                        }
                    ]
                }
            }
        }
        
        processor = WorldInfoProcessor(char_data)
        result = processor.process_messages("This is a test.", [], apply_token_budget=True)
        
        # Only first entry should be included due to budget
        assert len(result['matched_entries']) == 1
        assert result['matched_entries'][0]['content'] == 'A' * 200
        assert result['tokens_used'] <= 50


class TestWorldInfoIntegration:
    """Integration tests for world info with the chat system."""
    
    def test_message_injection_order(self):
        """Test the order of content injection."""
        processor = WorldInfoProcessor({
            'extensions': {
                'character_book': {
                    'entries': [
                        {
                            'keys': ['test'],
                            'content': 'START_CONTENT',
                            'enabled': True,
                            'position': 'at_start'
                        },
                        {
                            'keys': ['test'],
                            'content': 'BEFORE_CONTENT',
                            'enabled': True,
                            'position': 'before_char'
                        },
                        {
                            'keys': ['test'],
                            'content': 'AFTER_CONTENT',
                            'enabled': True,
                            'position': 'after_char'
                        },
                        {
                            'keys': ['test'],
                            'content': 'END_CONTENT',
                            'enabled': True,
                            'position': 'at_end'
                        }
                    ]
                }
            }
        })
        
        result = processor.process_messages("This is a test message.", [])
        formatted = processor.format_injections(result['injections'])
        
        # Build the final message as the chat system would
        parts = []
        if formatted.get('at_start'):
            parts.append(formatted['at_start'])
        if formatted.get('before_char'):
            parts.append(formatted['before_char'])
        parts.append("This is a test message.")
        if formatted.get('after_char'):
            parts.append(formatted['after_char'])
        if formatted.get('at_end'):
            parts.append(formatted['at_end'])
        
        final_message = '\n\n'.join(parts)
        
        # Check order
        assert final_message.index('START_CONTENT') < final_message.index('BEFORE_CONTENT')
        assert final_message.index('BEFORE_CONTENT') < final_message.index('This is a test message.')
        assert final_message.index('This is a test message.') < final_message.index('AFTER_CONTENT')
        assert final_message.index('AFTER_CONTENT') < final_message.index('END_CONTENT')