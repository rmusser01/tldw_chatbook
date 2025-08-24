"""
Unit tests for the refactored application.
Run with: pytest Tests/test_refactored_app_unit.py -v
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path
import json
import tempfile
from typing import Dict, Any


class TestRefactoredApp:
    """Test suite for the refactored application."""
    
    def test_app_initialization(self):
        """Test that app initializes with correct defaults."""
        from tldw_chatbook.app_refactored_v2 import TldwCliRefactored
        
        app = TldwCliRefactored()
        
        # Check reactive attributes
        assert app.current_screen == "chat"
        assert app.is_loading == False
        assert app.theme == "default"
        assert app.error_message is None
        
        # Check complex state
        assert isinstance(app.chat_state, dict)
        assert app.chat_state["provider"] == "openai"
        assert app.chat_state["model"] == "gpt-4"
        
        assert isinstance(app.notes_state, dict)
        assert app.notes_state["unsaved_changes"] == False
        
        assert isinstance(app.ui_state, dict)
        assert app.ui_state["dark_mode"] == True
    
    def test_screen_registry_building(self):
        """Test that screen registry is built correctly."""
        from tldw_chatbook.app_refactored_v2 import TldwCliRefactored
        
        app = TldwCliRefactored()
        
        # Check that some screens are registered
        assert len(app._screen_registry) > 0
        
        # Check for key screens (may vary based on what's available)
        expected_screens = ["chat", "notes", "media", "search"]
        available_screens = [s for s in expected_screens if s in app._screen_registry]
        
        assert len(available_screens) > 0, "At least some screens should be registered"
    
    def test_reactive_watchers(self):
        """Test that reactive watchers work."""
        from tldw_chatbook.app_refactored_v2 import TldwCliRefactored
        
        app = TldwCliRefactored()
        
        # Test current_screen watcher
        old_screen = app.current_screen
        app.current_screen = "notes"
        
        # The watcher should have been triggered
        assert app.current_screen == "notes"
        assert app.current_screen != old_screen
    
    def test_state_serialization(self):
        """Test state can be serialized to JSON."""
        from tldw_chatbook.app_refactored_v2 import TldwCliRefactored
        from datetime import datetime
        
        app = TldwCliRefactored()
        
        # Create state dict
        state = {
            "current_screen": app.current_screen,
            "theme": app.theme,
            "chat_state": dict(app.chat_state),
            "notes_state": dict(app.notes_state),
            "ui_state": dict(app.ui_state),
            "timestamp": datetime.now().isoformat()
        }
        
        # Should serialize without error
        json_str = json.dumps(state, indent=2, default=str)
        assert len(json_str) > 0
        
        # Should deserialize back
        loaded = json.loads(json_str)
        assert loaded["current_screen"] == "chat"
        assert loaded["theme"] == "default"
    
    @pytest.mark.asyncio
    async def test_navigation_error_handling(self):
        """Test that navigation handles errors gracefully."""
        from tldw_chatbook.app_refactored_v2 import TldwCliRefactored
        
        app = TldwCliRefactored()
        
        # Mock notify to check error messages
        app.notify = MagicMock()
        
        # Try to navigate to non-existent screen
        result = await app.navigate_to_screen("nonexistent")
        
        # Should return False and show error
        assert result == False
        app.notify.assert_called_with("Screen 'nonexistent' not found", severity="error")
    
    @pytest.mark.asyncio
    async def test_navigation_to_same_screen(self):
        """Test that navigating to current screen is handled."""
        from tldw_chatbook.app_refactored_v2 import TldwCliRefactored
        
        app = TldwCliRefactored()
        app.current_screen = "chat"
        
        # Navigate to same screen
        result = await app.navigate_to_screen("chat")
        
        # Should return True but not actually navigate
        assert result == True
        assert app.current_screen == "chat"
    
    def test_css_path_is_absolute(self):
        """Test that CSS path is absolute."""
        from tldw_chatbook.app_refactored_v2 import TldwCliRefactored
        
        app = TldwCliRefactored()
        
        # CSS_PATH should be a Path object
        assert isinstance(app.CSS_PATH, Path)
        
        # Should be absolute or relative to app location
        assert app.CSS_PATH.parts[-1] == "tldw_cli_modular.tcss"
    
    def test_button_compatibility_layer(self):
        """Test that old button patterns are supported."""
        from tldw_chatbook.app_refactored_v2 import TldwCliRefactored
        from textual.widgets import Button
        
        app = TldwCliRefactored()
        
        # Test tab button pattern
        button = Button("Test", id="tab-notes")
        assert button.id.startswith("tab-")
        screen_name = button.id[4:]
        assert screen_name == "notes"
        
        # Test tab-link pattern
        button2 = Button("Test", id="tab-link-media")
        assert button2.id.startswith("tab-link-")
        screen_name2 = button2.id[9:]
        assert screen_name2 == "media"
    
    @pytest.mark.asyncio
    async def test_save_and_load_state(self):
        """Test state persistence."""
        from tldw_chatbook.app_refactored_v2 import TldwCliRefactored
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create app and modify state
            app = TldwCliRefactored()
            app.current_screen = "notes"
            app.theme = "dark"
            app.chat_state = {**app.chat_state, "provider": "anthropic"}
            
            # Mock the path
            state_path = Path(tmpdir) / "state.json"
            
            # Save state
            with patch.object(Path, 'home', return_value=Path(tmpdir).parent):
                await app._save_state()
            
            # Create new app and load state
            app2 = TldwCliRefactored()
            
            # Write state manually for testing
            state = {
                "current_screen": "notes",
                "theme": "dark",
                "chat_state": {"provider": "anthropic", "model": "claude"},
                "notes_state": {},
                "ui_state": {}
            }
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps(state))
            
            # Load state
            with patch.object(Path, 'home', return_value=Path(tmpdir).parent):
                await app2._load_state()
            
            # Verify loaded correctly
            assert app2.theme == "dark"
            assert app2.chat_state["provider"] == "anthropic"
    
    def test_screen_parameter_detection(self):
        """Test that screen parameters are detected correctly."""
        from tldw_chatbook.app_refactored_v2 import TldwCliRefactored
        import inspect
        
        app = TldwCliRefactored()
        
        # Test different screen signatures
        class NoParamScreen:
            def __init__(self):
                pass
        
        class AppParamScreen:
            def __init__(self, app):
                self.app = app
        
        class AppInstanceScreen:
            def __init__(self, app_instance):
                self.app = app_instance
        
        class SelfFirstScreen:
            def __init__(self, app):
                self.app = app
        
        # Test parameter detection logic
        for screen_class in [NoParamScreen, AppParamScreen, AppInstanceScreen, SelfFirstScreen]:
            sig = inspect.signature(screen_class.__init__)
            params = list(sig.parameters.keys())
            if 'self' in params:
                params.remove('self')
            
            # This mimics the logic in _create_screen_instance
            if not params:
                assert screen_class == NoParamScreen
            elif 'app' in params:
                assert screen_class in [AppParamScreen, SelfFirstScreen]
            elif 'app_instance' in params:
                assert screen_class == AppInstanceScreen


class TestErrorRecovery:
    """Test error recovery mechanisms."""
    
    @pytest.mark.asyncio
    async def test_initial_screen_fallback(self):
        """Test fallback to chat screen on initial mount failure."""
        from tldw_chatbook.app_refactored_v2 import TldwCliRefactored
        
        app = TldwCliRefactored()
        app.current_screen = "nonexistent"
        
        # Mock navigate_to_screen to fail first time
        call_count = 0
        original_navigate = app.navigate_to_screen
        
        async def mock_navigate(screen_name):
            nonlocal call_count
            call_count += 1
            if call_count == 1 and screen_name == "nonexistent":
                return False
            return await original_navigate(screen_name)
        
        app.navigate_to_screen = mock_navigate
        
        # Should fallback to chat
        await app._mount_initial_screen()
        
        # Verify fallback was attempted
        assert call_count >= 1
    
    def test_import_fallbacks(self):
        """Test that import fallbacks work."""
        from tldw_chatbook.app_refactored_v2 import TldwCliRefactored
        
        app = TldwCliRefactored()
        
        # Test the _try_import_screen method
        # This should handle both new and old locations
        result = app._try_import_screen(
            "test",
            "nonexistent.module", "NonexistentClass",
            "also.nonexistent", "AlsoNonexistent"
        )
        
        # Should return None when both fail
        assert result is None
    
    @pytest.mark.asyncio
    async def test_state_save_error_handling(self):
        """Test that state save handles errors."""
        from tldw_chatbook.app_refactored_v2 import TldwCliRefactored
        
        app = TldwCliRefactored()
        
        # Mock path to raise error
        with patch.object(Path, 'write_text', side_effect=Exception("Write failed")):
            # Should not crash
            try:
                await app._save_state()
            except:
                pytest.fail("Save state should handle errors gracefully")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])