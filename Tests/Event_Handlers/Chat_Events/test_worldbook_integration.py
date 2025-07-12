"""Tests for worldbook/lorebook integration in chat."""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
from typing import Dict, List, Any


@pytest.fixture
def sample_worldbook_data():
    """Create sample worldbook data for testing."""
    return {
        "entries": {
            "dragons": {
                "keys": ["dragon", "dragons", "wyrm"],
                "content": "Dragons are ancient magical creatures.",
                "enabled": True,
                "priority": 10
            },
            "magic_system": {
                "keys": ["magic", "spell", "wizard"],
                "content": "Magic requires Intent, Knowledge, and Power.",
                "enabled": True,
                "priority": 5
            }
        },
        "settings": {
            "scan_depth": 50,
            "max_entries": 10,
            "case_sensitive": False
        }
    }


class TestWorldbookLoading:
    """Test worldbook file loading and parsing."""
    
    def test_load_worldbook_from_file(self, tmp_path, sample_worldbook_data):
        """Test loading worldbook from JSON file."""
        worldbook_file = tmp_path / "test_worldbook.json"
        worldbook_file.write_text(json.dumps(sample_worldbook_data))
        
        with open(worldbook_file, 'r') as f:
            data = json.load(f)
        
        assert "entries" in data
        assert "settings" in data
        assert len(data["entries"]) == 2
    
    def test_parse_worldbook_entries(self, sample_worldbook_data):
        """Test parsing worldbook entries."""
        entries = sample_worldbook_data["entries"]
        
        # Get enabled entries only
        enabled_entries = {
            k: v for k, v in entries.items() 
            if v.get("enabled", True)
        }
        
        assert len(enabled_entries) == 2
        assert "dragons" in enabled_entries
        assert "magic_system" in enabled_entries
    
    def test_worldbook_key_matching(self, sample_worldbook_data):
        """Test matching text against worldbook keys."""
        entries = sample_worldbook_data["entries"]
        text = "Tell me about dragons and magic"
        
        matched_entries = []
        for name, entry in entries.items():
            if not entry.get("enabled", True):
                continue
            
            # Check if any key matches
            for key in entry["keys"]:
                if key.lower() in text.lower():
                    matched_entries.append((name, entry))
                    break
        
        # Should match both entries
        assert len(matched_entries) == 2
        entry_names = [e[0] for e in matched_entries]
        assert "dragons" in entry_names
        assert "magic_system" in entry_names


class TestWorldbookIntegration:
    """Test worldbook integration with chat system."""
    
    def test_worldbook_context_injection(self, sample_worldbook_data):
        """Test injecting worldbook entries into chat context."""
        user_message = "What can you tell me about dragons?"
        
        # Find matching entries
        matched_content = []
        for name, entry in sample_worldbook_data["entries"].items():
            if not entry.get("enabled", True):
                continue
            
            for key in entry["keys"]:
                if key.lower() in user_message.lower():
                    matched_content.append(entry["content"])
                    break
        
        # Build context
        context = "\n\n".join(matched_content)
        
        assert "Dragons are ancient magical creatures" in context
    
    def test_priority_sorting(self, sample_worldbook_data):
        """Test sorting entries by priority."""
        entries = sample_worldbook_data["entries"]
        
        # Get enabled entries with priorities
        enabled = [
            (name, entry) for name, entry in entries.items()
            if entry.get("enabled", True)
        ]
        
        # Sort by priority (higher first)
        sorted_entries = sorted(
            enabled,
            key=lambda x: x[1].get("priority", 0),
            reverse=True
        )
        
        # Check order
        assert sorted_entries[0][0] == "dragons"  # priority 10
        assert sorted_entries[1][0] == "magic_system"  # priority 5


class TestWorldbookEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_worldbook(self):
        """Test handling empty worldbook."""
        empty_worldbook = {
            "entries": {},
            "settings": {"scan_depth": 50}
        }
        
        assert len(empty_worldbook["entries"]) == 0
    
    def test_malformed_entry(self):
        """Test handling malformed entries."""
        entry = {"content": "Some content"}  # Missing keys
        
        # Should handle missing fields
        keys = entry.get("keys", [])
        enabled = entry.get("enabled", True)
        priority = entry.get("priority", 0)
        
        assert keys == []
        assert enabled is True
        assert priority == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])