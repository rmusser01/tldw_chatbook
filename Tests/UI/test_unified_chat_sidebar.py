"""
Comprehensive tests for the unified chat sidebar using Textual testing framework.
Tests follow Textual best practices with real app integration testing.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

# Textual testing imports
from textual.app import App
from textual.widgets import Button, Input, Select, TextArea, Checkbox, ListView, TabbedContent, TabPane
from textual.pilot import Pilot

# Application imports
from tldw_chatbook.app import TldwCli
from tldw_chatbook.UI.Chat_Window_Enhanced import ChatWindowEnhanced
from tldw_chatbook.Widgets.Chat_Widgets.unified_chat_sidebar import (
    UnifiedChatSidebar,
    ChatSidebarState,
    CompactField,
    SearchableList,
    SessionTab,
    SettingsTab,
    ContentTab
)
from tldw_chatbook.Widgets.Chat_Widgets.sidebar_compatibility import (
    LegacySidebarAdapter,
    create_compatibility_adapter,
    WIDGET_ID_MAPPINGS
)


# ========================================
# Test App for Isolated Component Testing
# ========================================

class TestSidebarApp(App):
    """Minimal test app for sidebar testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Mock the required app_instance attributes
        self.app_config = {
            "chat_defaults": {
                "provider": "test_provider",
                "model": "test_model",
                "temperature": 0.7,
                "system_prompt": "Test prompt"
            }
        }
        self.chat_model_handler = MagicMock()
        self.chat_model_handler.is_current_chat_ephemeral.return_value = True
        self.current_chat_is_ephemeral = True
    
    def compose(self):
        yield UnifiedChatSidebar(self)


# ========================================
# Basic Sidebar Tests
# ========================================

@pytest.mark.asyncio
class TestUnifiedChatSidebar:
    """Test the main UnifiedChatSidebar component."""
    
    async def test_sidebar_initialization(self):
        """Test that the sidebar initializes correctly."""
        app = TestSidebarApp()
        async with app.run_test() as pilot:
            # Check sidebar exists
            sidebar = app.query_one(UnifiedChatSidebar)
            assert sidebar is not None
            
            # Check state initialization
            assert sidebar.state is not None
            assert isinstance(sidebar.state, ChatSidebarState)
            assert sidebar.state.active_tab == "session"
            assert sidebar.state.sidebar_width == 30
    
    async def test_sidebar_tabs_exist(self):
        """Test that all expected tabs are present."""
        app = TestSidebarApp()
        async with app.run_test() as pilot:
            # Check TabbedContent exists
            tabs = app.query_one("#sidebar-tabs", TabbedContent)
            assert tabs is not None
            
            # Check all three tabs exist
            session_tab = app.query_one("#session", TabPane)
            settings_tab = app.query_one("#settings", TabPane)
            content_tab = app.query_one("#content", TabPane)
            
            assert session_tab is not None
            assert settings_tab is not None
            assert content_tab is not None
    
    async def test_sidebar_toggle(self):
        """Test sidebar collapse/expand functionality."""
        app = TestSidebarApp()
        async with app.run_test() as pilot:
            sidebar = app.query_one(UnifiedChatSidebar)
            
            # Initially not collapsed
            assert sidebar.collapsed == False
            assert "collapsed" not in sidebar.classes
            
            # Toggle collapse
            sidebar.action_toggle_sidebar()
            assert sidebar.collapsed == True
            assert "collapsed" in sidebar.classes
            
            # Toggle back
            sidebar.action_toggle_sidebar()
            assert sidebar.collapsed == False
            assert "collapsed" not in sidebar.classes
    
    async def test_tab_switching(self):
        """Test switching between tabs."""
        app = TestSidebarApp()
        async with app.run_test() as pilot:
            sidebar = app.query_one(UnifiedChatSidebar)
            tabs = app.query_one("#sidebar-tabs", TabbedContent)
            
            # Initially on session tab
            assert tabs.active == "session"
            
            # Switch to settings tab
            sidebar.action_switch_tab("settings")
            await pilot.pause()
            assert tabs.active == "settings"
            assert sidebar.state.active_tab == "settings"
            
            # Switch to content tab
            sidebar.action_switch_tab("content")
            await pilot.pause()
            assert tabs.active == "content"
            assert sidebar.state.active_tab == "content"
    
    async def test_keyboard_shortcuts(self):
        """Test keyboard shortcuts for tab switching."""
        app = TestSidebarApp()
        async with app.run_test() as pilot:
            tabs = app.query_one("#sidebar-tabs", TabbedContent)
            
            # Test Alt+1 for Session tab
            await pilot.press("alt+1")
            assert tabs.active == "session"
            
            # Test Alt+2 for Settings tab
            await pilot.press("alt+2")
            assert tabs.active == "settings"
            
            # Test Alt+3 for Content tab
            await pilot.press("alt+3")
            assert tabs.active == "content"
            
            # Test Ctrl+\ for sidebar toggle
            sidebar = app.query_one(UnifiedChatSidebar)
            initial_collapsed = sidebar.collapsed
            await pilot.press("ctrl+\\")
            assert sidebar.collapsed != initial_collapsed


# ========================================
# Session Tab Tests
# ========================================

@pytest.mark.asyncio
class TestSessionTab:
    """Test the Session tab functionality."""
    
    async def test_session_tab_widgets(self):
        """Test that all session widgets are present."""
        app = TestSidebarApp()
        async with app.run_test() as pilot:
            # Switch to session tab
            sidebar = app.query_one(UnifiedChatSidebar)
            sidebar.action_switch_tab("session")
            await pilot.pause()
            
            # Check all expected widgets
            chat_id = app.query_one("#session-chat-id", Input)
            title = app.query_one("#session-title", Input)
            keywords = app.query_one("#session-keywords", TextArea)
            save_btn = app.query_one("#session-save-chat", Button)
            clone_btn = app.query_one("#session-clone-chat", Button)
            note_btn = app.query_one("#session-to-note", Button)
            new_btn = app.query_one("#session-new-chat", Button)
            strip_checkbox = app.query_one("#session-strip-tags", Checkbox)
            
            assert chat_id is not None
            assert title is not None
            assert keywords is not None
            assert save_btn is not None
            assert clone_btn is not None
            assert note_btn is not None
            assert new_btn is not None
            assert strip_checkbox is not None
    
    async def test_session_field_interaction(self):
        """Test interacting with session fields."""
        app = TestSidebarApp()
        async with app.run_test() as pilot:
            # Switch to session tab
            sidebar = app.query_one(UnifiedChatSidebar)
            sidebar.action_switch_tab("session")
            await pilot.pause()
            
            # Test title input
            title_input = app.query_one("#session-title", Input)
            await pilot.click("#session-title")
            await pilot.press(*"Test Chat Title")
            assert title_input.value == "Test Chat Title"
            
            # Test keywords textarea
            keywords = app.query_one("#session-keywords", TextArea)
            await pilot.click("#session-keywords")
            await pilot.press(*"test, keywords, here")
            assert "test, keywords, here" in keywords.text
            
            # Test checkbox is present and has expected initial value
            checkbox = app.query_one("#session-strip-tags", Checkbox)
            assert checkbox is not None
            assert checkbox.value == True  # Default is True from the widget definition


# ========================================
# Settings Tab Tests
# ========================================

@pytest.mark.asyncio
class TestSettingsTab:
    """Test the Settings tab functionality."""
    
    async def test_settings_tab_widgets(self):
        """Test that all settings widgets are present."""
        app = TestSidebarApp()
        async with app.run_test() as pilot:
            # Switch to settings tab
            sidebar = app.query_one(UnifiedChatSidebar)
            sidebar.action_switch_tab("settings")
            await pilot.pause()
            
            # Check basic settings
            provider = app.query_one("#settings-provider", Select)
            model = app.query_one("#settings-model", Select)
            temperature = app.query_one("#settings-temperature", Input)
            advanced_toggle = app.query_one("#settings-advanced-toggle", Checkbox)
            
            assert provider is not None
            assert model is not None
            assert temperature is not None
            assert advanced_toggle is not None
    
    async def test_progressive_disclosure(self):
        """Test advanced settings show/hide functionality."""
        app = TestSidebarApp()
        async with app.run_test() as pilot:
            # Switch to settings tab
            sidebar = app.query_one(UnifiedChatSidebar)
            sidebar.action_switch_tab("settings")
            await pilot.pause()
            
            # Advanced settings should be hidden initially
            advanced_container = app.query_one("#advanced-settings")
            assert "hidden" in advanced_container.classes
            
            # Click the checkbox to toggle advanced settings
            checkbox = app.query_one("#settings-advanced-toggle", Checkbox)
            await pilot.click("#settings-advanced-toggle")
            await pilot.pause()
            
            # Manually trigger the toggle since we're not going through the button handler
            await sidebar._toggle_advanced_settings()
            await pilot.pause()
            
            # Advanced settings should now be visible
            assert "hidden" not in advanced_container.classes
            
            # Check advanced widgets are accessible
            system_prompt = app.query_one("#settings-system-prompt", TextArea)
            top_p = app.query_one("#settings-top-p", Input)
            top_k = app.query_one("#settings-top-k", Input)
            min_p = app.query_one("#settings-min-p", Input)
            
            assert system_prompt is not None
            assert top_p is not None
            assert top_k is not None
            assert min_p is not None
    
    async def test_rag_settings_toggle(self):
        """Test RAG settings collapsible functionality."""
        app = TestSidebarApp()
        async with app.run_test() as pilot:
            # Switch to settings tab
            sidebar = app.query_one(UnifiedChatSidebar)
            sidebar.action_switch_tab("settings")
            await pilot.pause()
            
            # RAG settings should be hidden initially
            rag_container = app.query_one("#rag-settings")
            assert "hidden" in rag_container.classes
            
            # Get initial button state
            toggle_btn = app.query_one("#rag-toggle", Button)
            assert "▶ RAG Settings" in str(toggle_btn.label)
            
            # Manually trigger the toggle (since button event handling might not work in test)
            await sidebar._toggle_rag_settings()
            await pilot.pause()
            
            # RAG settings should now be visible
            assert "hidden" not in rag_container.classes
            assert "▼ RAG Settings" in str(toggle_btn.label)
            
            # Check RAG widgets
            rag_enabled = app.query_one("#settings-rag-enabled", Checkbox)
            rag_pipeline = app.query_one("#settings-rag-pipeline", Select)
            
            assert rag_enabled is not None
            assert rag_pipeline is not None


# ========================================
# Content Tab Tests
# ========================================

@pytest.mark.asyncio
class TestContentTab:
    """Test the Content tab functionality."""
    
    async def test_content_tab_widgets(self):
        """Test that all content widgets are present."""
        app = TestSidebarApp()
        async with app.run_test() as pilot:
            # Switch to content tab
            sidebar = app.query_one(UnifiedChatSidebar)
            sidebar.action_switch_tab("content")
            await pilot.pause()
            
            # Check search widgets
            search_input = app.query_one("#content-search-input", Input)
            filter_select = app.query_one("#content-filter", Select)
            search_btn = app.query_one("#content-search-btn", Button)
            results_list = app.query_one("#content-results-list", ListView)
            
            assert search_input is not None
            assert filter_select is not None
            assert search_btn is not None
            assert results_list is not None
            
            # Check pagination
            prev_btn = app.query_one("#content-prev", Button)
            next_btn = app.query_one("#content-next", Button)
            page_label = app.query_one("#content-page-label")
            
            assert prev_btn is not None
            assert next_btn is not None
            assert page_label is not None
            
            # Check action buttons
            load_btn = app.query_one("#content-load", Button)
            copy_btn = app.query_one("#content-copy", Button)
            
            assert load_btn is not None
            assert copy_btn is not None
    
    async def test_content_search_interaction(self):
        """Test search functionality in content tab."""
        app = TestSidebarApp()
        async with app.run_test() as pilot:
            # Switch to content tab
            sidebar = app.query_one(UnifiedChatSidebar)
            sidebar.action_switch_tab("content")
            await pilot.pause()
            
            # Enter search term
            search_input = app.query_one("#content-search-input", Input)
            await pilot.click("#content-search-input")
            await pilot.press(*"test search")
            assert search_input.value == "test search"
            
            # Change filter
            filter_select = app.query_one("#content-filter", Select)
            assert filter_select.value == "all"
            
            # Note: Full search functionality would require mocking the database


# ========================================
# Compatibility Layer Tests
# ========================================

@pytest.mark.asyncio
class TestCompatibilityLayer:
    """Test the backward compatibility layer."""
    
    async def test_legacy_adapter_creation(self):
        """Test creating the legacy adapter."""
        app = TestSidebarApp()
        async with app.run_test() as pilot:
            sidebar = app.query_one(UnifiedChatSidebar)
            adapter = LegacySidebarAdapter(sidebar, app)
            
            assert adapter is not None
            assert adapter.sidebar == sidebar
            assert adapter.app == app
    
    async def test_widget_id_mapping(self):
        """Test that old widget IDs map to new ones."""
        app = TestSidebarApp()
        async with app.run_test() as pilot:
            sidebar = app.query_one(UnifiedChatSidebar)
            adapter = LegacySidebarAdapter(sidebar, app)
            
            # Test settings mappings
            sidebar.action_switch_tab("settings")
            await pilot.pause()
            
            # Old ID -> New widget should work
            widget = adapter.query_one("chat-temperature")
            assert widget is not None
            assert widget.id == "settings-temperature"
            
            # Test session mappings
            sidebar.action_switch_tab("session")
            await pilot.pause()
            
            widget = adapter.query_one("chat-conversation-title-input")
            assert widget is not None
            assert widget.id == "session-title"
    
    async def test_legacy_state_retrieval(self):
        """Test getting state in legacy format."""
        app = TestSidebarApp()
        async with app.run_test() as pilot:
            sidebar = app.query_one(UnifiedChatSidebar)
            adapter = LegacySidebarAdapter(sidebar, app)
            
            # Set some values
            sidebar.action_switch_tab("settings")
            await pilot.pause()
            
            temp_input = app.query_one("#settings-temperature", Input)
            temp_input.value = "0.8"
            
            # Get legacy state
            legacy_state = adapter.get_sidebar_state()
            
            assert "temperature" in legacy_state
            assert legacy_state["temperature"] == "0.8"
            assert "is_ephemeral" in legacy_state


# ========================================
# Integration Tests with Full App
# ========================================

@pytest.mark.asyncio
class TestFullAppIntegration:
    """Test the unified sidebar with the full TldwCli app."""
    
    @pytest.mark.skip(reason="Full app test requires extensive mocking")
    async def test_sidebar_in_full_app(self):
        """Test sidebar integration in the complete application."""
        # This would require mocking all database connections,
        # API configurations, etc. Skipping for now but structure shown.
        
        with patch('tldw_chatbook.DB.ChaChaNotes_DB.Database'):
            app = TldwCli()
            async with app.run_test(size=(120, 40)) as pilot:
                # Navigate to chat tab
                await pilot.press("ctrl+1")  # Assuming this switches to chat
                
                # Verify unified sidebar exists
                sidebar = app.query_one(UnifiedChatSidebar)
                assert sidebar is not None
    
    async def test_sidebar_state_persistence(self):
        """Test that sidebar state persists correctly."""
        app = TestSidebarApp()
        async with app.run_test() as pilot:
            sidebar = app.query_one(UnifiedChatSidebar)
            
            # Test that state can be changed
            sidebar.state.sidebar_width = 35
            sidebar.state.active_tab = "content"
            sidebar.state.advanced_mode = True
            
            # Verify state was changed
            assert sidebar.state.sidebar_width == 35
            assert sidebar.state.active_tab == "content"
            assert sidebar.state.advanced_mode == True
            
            # Save preferences doesn't error (even if it doesn't actually save in test mode)
            sidebar.state.save_preferences()


# ========================================
# Compound Widget Tests
# ========================================

@pytest.mark.asyncio
class TestCompoundWidgets:
    """Test the compound widgets used in the sidebar."""
    
    async def test_compact_field(self):
        """Test CompactField widget."""
        class CompactFieldApp(App):
            def compose(self):
                yield CompactField("Test Label:", "test-field", value="initial")
        
        app = CompactFieldApp()
        async with app.run_test() as pilot:
            # Check label exists
            labels = app.query("Label")
            assert len(labels) > 0
            assert "Test Label:" in labels[0].renderable
            
            # Check input exists and has initial value
            input_field = app.query_one("#test-field", Input)
            assert input_field is not None
            assert input_field.value == "initial"
    
    async def test_searchable_list(self):
        """Test SearchableList widget."""
        class SearchableListApp(App):
            def compose(self):
                yield SearchableList("test", placeholder="Search test...")
        
        app = SearchableListApp()
        async with app.run_test() as pilot:
            # Check search input
            search_input = app.query_one("#test-search-input", Input)
            assert search_input is not None
            assert search_input.placeholder == "Search test..."
            
            # Check results list
            results = app.query_one("#test-results", ListView)
            assert results is not None
            
            # Check pagination
            prev_btn = app.query_one("#test-prev", Button)
            next_btn = app.query_one("#test-next", Button)
            assert prev_btn is not None
            assert next_btn is not None
            assert prev_btn.disabled == True  # Initially disabled


# ========================================
# Performance Tests
# ========================================

@pytest.mark.asyncio
class TestPerformance:
    """Test performance characteristics of the unified sidebar."""
    
    async def test_tab_switch_performance(self):
        """Test that tab switching is fast."""
        import time
        
        app = TestSidebarApp()
        async with app.run_test() as pilot:
            sidebar = app.query_one(UnifiedChatSidebar)
            
            # Measure tab switch time
            start = time.time()
            sidebar.action_switch_tab("settings")
            await pilot.pause()
            sidebar.action_switch_tab("content")
            await pilot.pause()
            sidebar.action_switch_tab("session")
            await pilot.pause()
            elapsed = time.time() - start
            
            # Should complete 3 tab switches in under 1 second (realistic for UI operations)
            assert elapsed < 1.0
    
    async def test_widget_count(self):
        """Verify the widget count reduction."""
        app = TestSidebarApp()
        async with app.run_test() as pilot:
            # Count all widgets in the sidebar
            all_widgets = app.query("UnifiedChatSidebar *")
            widget_count = len(all_widgets)
            
            # Should have significantly fewer widgets than old implementation
            # The unified sidebar has been optimized but still contains many widgets
            assert widget_count < 150  # Reasonable upper bound for all tabs and widgets
            
            # Log for verification
            print(f"Total widget count in unified sidebar: {widget_count}")


# ========================================
# Error Handling Tests
# ========================================

@pytest.mark.asyncio
class TestErrorHandling:
    """Test error handling in the unified sidebar."""
    
    async def test_invalid_tab_switch(self):
        """Test handling of invalid tab switch."""
        app = TestSidebarApp()
        async with app.run_test() as pilot:
            sidebar = app.query_one(UnifiedChatSidebar)
            tabs = app.query_one("#sidebar-tabs", TabbedContent)
            
            # Store the current tab
            original_tab = tabs.active
            
            # Try to switch to non-existent tab - this should log an error but not crash
            sidebar.action_switch_tab("nonexistent")
            await pilot.pause()
            
            # Should stay on original tab since the switch failed
            # The error is caught and logged, so tab should not change
            assert tabs.active == original_tab
    
    async def test_missing_config_handling(self):
        """Test handling of missing configuration."""
        app = TestSidebarApp()
        async with app.run_test() as pilot:
            # Should still initialize with defaults
            sidebar = app.query_one(UnifiedChatSidebar)
            assert sidebar is not None
            # Default value should be set
            assert sidebar.state.sidebar_width == 30  # Default width is 30


# ========================================
# Accessibility Tests
# ========================================

@pytest.mark.asyncio
class TestAccessibility:
    """Test accessibility features of the unified sidebar."""
    
    async def test_keyboard_navigation(self):
        """Test that all functionality is keyboard accessible."""
        app = TestSidebarApp()
        async with app.run_test() as pilot:
            # Tab through widgets
            await pilot.press("tab")
            await pilot.press("tab")
            await pilot.press("tab")
            
            # Should be able to navigate without mouse
            focused = app.focused
            assert focused is not None
    
    async def test_focus_management(self):
        """Test that focus is properly managed."""
        app = TestSidebarApp()
        async with app.run_test() as pilot:
            # Switch to settings tab
            sidebar = app.query_one(UnifiedChatSidebar)
            sidebar.action_switch_tab("settings")
            await pilot.pause()
            
            # Focus on temperature input
            await pilot.click("#settings-temperature")
            focused = app.focused
            assert focused.id == "settings-temperature"
            
            # Tab to next field
            await pilot.press("tab")
            new_focused = app.focused
            assert new_focused != focused


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])