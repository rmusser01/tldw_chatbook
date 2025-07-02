"""Simple unit tests for Chat Window button tooltips."""
import pytest
import os
from pathlib import Path
from textual.widgets import Button

from tldw_chatbook.UI.Chat_Window import ChatWindow


class TestChatButtonTooltips:
    """Test suite for verifying tooltips on Chat Window buttons."""
    
    def test_button_tooltips_in_source(self):
        """Verify tooltips are defined in the source code."""
        # Get the correct path to the source file
        test_dir = Path(__file__).parent.parent.parent  # Go up to project root
        source_file = test_dir / "tldw_chatbook" / "UI" / "Chat_Window.py"
        
        # Read the source file
        with open(source_file, "r") as f:
            source = f.read()
        
        # Expected tooltips
        expected_tooltips = [
            ('tooltip="Toggle left sidebar (Ctrl+[)"', "Left sidebar toggle tooltip"),
            ('tooltip="Send message (Enter)"', "Send button tooltip"),
            ('tooltip="Suggest a response"', "Suggest button tooltip"),
            ('tooltip="Stop generation"', "Stop button tooltip"),
            ('tooltip="Toggle right sidebar (Ctrl+])"', "Right sidebar toggle tooltip"),
        ]
        
        # Verify each tooltip exists in source
        for tooltip_text, description in expected_tooltips:
            assert tooltip_text in source, f"{description} not found in source code"
    
    def test_suggest_button_improved(self):
        """Verify the suggest button has been improved with a tooltip."""
        # Get the correct path to the source file
        test_dir = Path(__file__).parent.parent.parent  # Go up to project root
        source_file = test_dir / "tldw_chatbook" / "UI" / "Chat_Window.py"
        
        with open(source_file, "r") as f:
            source = f.read()
        
        # Check that the suggest button has a tooltip
        assert 'id="respond-for-me-button"' in source, "Suggest button not found"
        assert 'tooltip="Suggest a response"' in source, "Suggest button tooltip not found"
        
        # Verify it still has the lightbulb emoji
        # Look for the emoji before the button definition
        import re
        # Find the Button definition with respond-for-me-button id
        pattern = r'Button\s*\(\s*"💡"[^)]*id="respond-for-me-button"'
        assert re.search(pattern, source, re.DOTALL), "Suggest button should have lightbulb emoji"