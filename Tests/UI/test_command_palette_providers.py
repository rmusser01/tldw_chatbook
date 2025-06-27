# test_command_palette_providers.py
#
# Comprehensive tests for command palette providers
#
# Test Dependencies:
# - textual: Required for Provider and Hit classes
# - Core tldw_chatbook app must be importable
# - All command palette provider classes must be defined in app.py
#
# Imports
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from typing import List, AsyncIterator

# 3rd-party Libraries
from textual.app import App
from textual.command import Hit

# Local Imports  
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    # First try to import the base Provider class from textual
    from textual.command import Provider
    
    # Then try to import from the app
    from tldw_chatbook.app import (
        ThemeProvider,
        TabNavigationProvider,
        LLMProviderProvider,
        QuickActionsProvider,
        SettingsProvider,
        CharacterProvider,
        MediaProvider,
        DeveloperProvider
    )
    from tldw_chatbook.Constants import (
        TAB_CHAT, TAB_CCP, TAB_NOTES, TAB_MEDIA, TAB_SEARCH, 
        TAB_INGEST, TAB_TOOLS_SETTINGS, TAB_LLM, TAB_LOGS, 
        TAB_STATS, TAB_EVALS, TAB_CODING
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_AVAILABLE = False
    
    # Create dummy classes for testing
    class DummyProvider:
        def __init__(self):
            self.app = None
            self.matcher = None
        async def search(self, query): return []
        async def discover(self): return []
    
    ThemeProvider = DummyProvider
    TabNavigationProvider = DummyProvider
    LLMProviderProvider = DummyProvider
    QuickActionsProvider = DummyProvider
    SettingsProvider = DummyProvider
    CharacterProvider = DummyProvider
    MediaProvider = DummyProvider
    DeveloperProvider = DummyProvider
    
    # Constants
    TAB_CHAT = "chat"
    TAB_CCP = "conversations_characters_prompts"
    TAB_NOTES = "notes"
    TAB_MEDIA = "media"
    TAB_SEARCH = "search"
    TAB_INGEST = "ingest"
    TAB_TOOLS_SETTINGS = "tools_settings"
    TAB_LLM = "llm_management"
    TAB_LOGS = "logs"
    TAB_STATS = "stats"
    TAB_EVALS = "evals"
    TAB_CODING = "coding"

#######################################################################################################################
#
# --- Skip marker for tests requiring imports ---

requires_imports = pytest.mark.skipif(
    not IMPORTS_AVAILABLE, 
    reason="Command palette providers not available due to import errors"
)

#######################################################################################################################
#
# --- Fixtures ---

@pytest.fixture
def mock_app():
    """Create a mock app instance for testing providers."""
    app = MagicMock(spec=App)
    app.current_tab = "chat"
    app.theme = "textual-dark"
    app.notify = MagicMock()
    return app

@pytest.fixture
def mock_matcher():
    """Create a mock matcher for search testing."""
    matcher = MagicMock()
    matcher.match = MagicMock(return_value=1.0)
    matcher.highlight = MagicMock(side_effect=lambda x: x)
    return matcher

#######################################################################################################################
#
# --- ThemeProvider Tests ---

@requires_imports
class TestThemeProvider:
    """Test suite for ThemeProvider functionality."""
    
    @pytest.fixture
    def theme_provider(self, mock_app):
        """Create a ThemeProvider instance with mock app."""
        mock_screen = MagicMock()
        mock_screen.app = mock_app
        provider = ThemeProvider(screen=mock_screen)
        # Create a mock matcher object
        mock_matcher = MagicMock()
        mock_matcher.match = MagicMock(return_value=1.0)
        mock_matcher.highlight = MagicMock(side_effect=lambda x: x)
        # Set provider.matcher as a callable that returns the matcher object
        provider.matcher = MagicMock(return_value=mock_matcher)
        return provider
    
    @pytest.mark.asyncio
    async def test_discover_shows_single_change_theme_command(self, theme_provider):
        """Test that discover() shows only one 'Change Theme' command."""
        hits = []
        async for hit in theme_provider.discover():
            hits.append(hit)
        
        assert len(hits) == 1
        assert hits[0].text == "Theme: Change Theme"
        assert "Open theme selection menu" in hits[0].help
    
    @pytest.mark.asyncio
    async def test_search_shows_change_theme_command(self, theme_provider):
        """Test that search shows 'Change Theme' command for general queries."""
        hits = []
        async for hit in theme_provider.search("change"):
            hits.append(hit)
        
        # Should show at least the main command
        assert len(hits) >= 1
        change_theme_hits = [h for h in hits if "Theme: Change Theme" in h.text]
        assert len(change_theme_hits) == 1
    
    @pytest.mark.asyncio
    async def test_search_shows_specific_themes_for_theme_keywords(self, theme_provider):
        """Test that search shows specific themes when theme keywords are used."""
        with patch('tldw_chatbook.app.ALL_THEMES', []):  # Mock empty themes list
            hits = []
            async for hit in theme_provider.search("theme dark"):
                hits.append(hit)
            
            # Should show both main command and specific theme commands
            theme_specific_hits = [h for h in hits if "Switch to" in h.text]
            change_theme_hits = [h for h in hits if "Theme: Change Theme" in h.text]
            
            assert len(change_theme_hits) == 1
            # Should show built-in themes
            assert len(theme_specific_hits) >= 1
    
    @pytest.mark.asyncio
    async def test_search_filters_themes_correctly(self, theme_provider):
        """Test that theme search filters work correctly."""
        test_keywords = ["dark", "light", "gruvbox", "solarized", "dracula"]
        
        for keyword in test_keywords:
            hits = []
            async for hit in theme_provider.search(keyword):
                hits.append(hit)
            
            # Should show both main command and filtered themes
            assert len(hits) >= 1
            change_theme_hits = [h for h in hits if "Theme: Change Theme" in h.text]
            assert len(change_theme_hits) == 1
    
    def test_show_theme_submenu(self, theme_provider):
        """Test that show_theme_submenu provides helpful instruction."""
        theme_provider.show_theme_submenu()
        
        theme_provider.app.notify.assert_called_once()
        call_args = theme_provider.app.notify.call_args[0]
        assert "Type 'theme'" in call_args[0]
        assert "command palette" in call_args[0]
    
    def test_switch_theme_success(self, theme_provider):
        """Test successful theme switching."""
        with patch('tldw_chatbook.config.save_setting_to_cli_config') as mock_save:
            theme_provider.switch_theme("test-theme")
            
            assert theme_provider.app.theme == "test-theme"
            theme_provider.app.notify.assert_called_once_with(
                "Theme changed to test-theme", severity="information"
            )
            mock_save.assert_called_once_with("general", "default_theme", "test-theme")
    
    def test_switch_theme_failure(self, theme_provider):
        """Test theme switching with error handling."""
        theme_provider.app.theme = MagicMock(side_effect=Exception("Theme error"))
        
        theme_provider.switch_theme("invalid-theme")
        
        theme_provider.app.notify.assert_called_once()
        call_args = theme_provider.app.notify.call_args
        assert "Failed to apply theme" in call_args[0][0]
        assert call_args[1]['severity'] == "error"


#######################################################################################################################
#
# --- TabNavigationProvider Tests ---

@requires_imports
class TestTabNavigationProvider:
    """Test suite for TabNavigationProvider functionality."""
    
    @pytest.fixture
    def tab_provider(self, mock_app):
        """Create a TabNavigationProvider instance with mock app."""
        mock_screen = MagicMock()
        mock_screen.app = mock_app
        provider = TabNavigationProvider(screen=mock_screen)
        # Create a mock matcher object
        mock_matcher = MagicMock()
        mock_matcher.match = MagicMock(return_value=1.0)
        mock_matcher.highlight = MagicMock(side_effect=lambda x: x)
        # Set provider.matcher as a callable that returns the matcher object
        provider.matcher = MagicMock(return_value=mock_matcher)
        return provider
    
    @pytest.mark.asyncio
    async def test_discover_shows_popular_tabs(self, tab_provider):
        """Test that discover() shows popular tab navigation commands."""
        hits = []
        async for hit in tab_provider.discover():
            hits.append(hit)
        
        assert len(hits) == 5  # Popular tabs defined in discover method
        tab_names = [hit.text for hit in hits]
        assert any("Chat" in name for name in tab_names)
        assert any("Character Chat" in name for name in tab_names)
        assert any("Notes" in name for name in tab_names)
    
    @pytest.mark.asyncio
    async def test_search_shows_all_tabs(self, tab_provider):
        """Test that search shows all available tabs."""
        hits = []
        async for hit in tab_provider.search("tab"):
            hits.append(hit)
        
        # Should show all 12 tabs
        assert len(hits) == 12
        
        # Check that all major tabs are present
        tab_texts = [hit.text for hit in hits]
        assert any("Chat" in text for text in tab_texts)
        assert any("Character Chat" in text for text in tab_texts)
        assert any("Tools & Settings" in text for text in tab_texts)
        assert any("LLM Management" in text for text in tab_texts)
    
    def test_switch_tab_success(self, tab_provider):
        """Test successful tab switching."""
        tab_provider.switch_tab(TAB_NOTES)
        
        assert tab_provider.app.current_tab == TAB_NOTES
        tab_provider.app.notify.assert_called_once()
        call_args = tab_provider.app.notify.call_args[0]
        assert "Switched to" in call_args[0]
        assert "Notes" in call_args[0]
    
    def test_switch_tab_failure(self, tab_provider):
        """Test tab switching with error handling."""
        tab_provider.app.current_tab = MagicMock(side_effect=Exception("Tab error"))
        
        tab_provider.switch_tab(TAB_CHAT)
        
        tab_provider.app.notify.assert_called_once()
        call_args = tab_provider.app.notify.call_args
        assert "Failed to switch tab" in call_args[0][0]
        assert call_args[1]['severity'] == "error"
    
    @pytest.mark.parametrize("tab_id", [
        TAB_CHAT, TAB_CCP, TAB_NOTES, TAB_MEDIA, TAB_SEARCH,
        TAB_INGEST, TAB_TOOLS_SETTINGS, TAB_LLM, TAB_LOGS,
        TAB_STATS, TAB_EVALS, TAB_CODING
    ])
    def test_all_tabs_switchable(self, tab_provider, tab_id):
        """Test that all defined tabs can be switched to."""
        tab_provider.switch_tab(tab_id)
        assert tab_provider.app.current_tab == tab_id


#######################################################################################################################
#
# --- QuickActionsProvider Tests ---

@requires_imports
class TestQuickActionsProvider:
    """Test suite for QuickActionsProvider functionality."""
    
    @pytest.fixture
    def quick_actions_provider(self, mock_app):
        """Create a QuickActionsProvider instance with mock app."""
        mock_screen = MagicMock()
        mock_screen.app = mock_app
        provider = QuickActionsProvider(screen=mock_screen)
        # Create a mock matcher object
        mock_matcher = MagicMock()
        mock_matcher.match = MagicMock(return_value=1.0)
        mock_matcher.highlight = MagicMock(side_effect=lambda x: x)
        # Set provider.matcher as a callable that returns the matcher object
        provider.matcher = MagicMock(return_value=mock_matcher)
        return provider
    
    @pytest.mark.asyncio
    async def test_discover_shows_popular_actions(self, quick_actions_provider):
        """Test that discover() shows popular quick actions."""
        hits = []
        async for hit in quick_actions_provider.discover():
            hits.append(hit)
        
        assert len(hits) == 4  # Popular actions defined in discover method
        action_names = [hit.text for hit in hits]
        assert any("New Chat" in name for name in action_names)
        assert any("New Note" in name for name in action_names)
        assert any("Search All" in name for name in action_names)
    
    @pytest.mark.asyncio
    async def test_search_shows_all_actions(self, quick_actions_provider):
        """Test that search shows all available quick actions."""
        hits = []
        async for hit in quick_actions_provider.search("quick"):
            hits.append(hit)
        
        assert len(hits) == 8  # All actions defined in search method
        action_texts = [hit.text for hit in hits]
        assert any("New Chat Conversation" in text for text in action_texts)
        assert any("Export Chat" in text for text in action_texts)
        assert any("Import Media" in text for text in action_texts)
    
    def test_execute_new_chat_action(self, quick_actions_provider):
        """Test new chat action execution."""
        quick_actions_provider.execute_quick_action("new_chat")
        
        assert quick_actions_provider.app.current_tab == TAB_CHAT
        quick_actions_provider.app.notify.assert_called_once()
        call_args = quick_actions_provider.app.notify.call_args[0]
        assert "Chat tab" in call_args[0]
    
    def test_execute_new_note_action(self, quick_actions_provider):
        """Test new note action execution."""
        quick_actions_provider.execute_quick_action("new_note")
        
        assert quick_actions_provider.app.current_tab == TAB_NOTES
        quick_actions_provider.app.notify.assert_called_once()
    
    def test_execute_action_failure(self, quick_actions_provider):
        """Test quick action execution with error handling."""
        quick_actions_provider.app.current_tab = MagicMock(side_effect=Exception("Action error"))
        
        quick_actions_provider.execute_quick_action("new_chat")
        
        quick_actions_provider.app.notify.assert_called_once()
        call_args = quick_actions_provider.app.notify.call_args
        assert "Failed to execute quick action" in call_args[0][0]
        assert call_args[1]['severity'] == "error"


#######################################################################################################################
#
# --- LLMProviderProvider Tests ---

@requires_imports
class TestLLMProviderProvider:
    """Test suite for LLMProviderProvider functionality."""
    
    @pytest.fixture
    def llm_provider(self, mock_app):
        """Create an LLMProviderProvider instance with mock app."""
        mock_screen = MagicMock()
        mock_screen.app = mock_app
        provider = LLMProviderProvider(screen=mock_screen)
        # Create a mock matcher object
        mock_matcher = MagicMock()
        mock_matcher.match = MagicMock(return_value=1.0)
        mock_matcher.highlight = MagicMock(side_effect=lambda x: x)
        # Set provider.matcher as a callable that returns the matcher object
        provider.matcher = MagicMock(return_value=mock_matcher)
        return provider
    
    @pytest.mark.asyncio
    async def test_discover_shows_popular_providers(self, llm_provider):
        """Test that discover() shows popular LLM providers."""
        hits = []
        async for hit in llm_provider.discover():
            hits.append(hit)
        
        assert len(hits) == 6  # Show current + 5 popular providers
        provider_names = [hit.text for hit in hits]
        assert any("Show Current Provider" in name for name in provider_names)
        assert any("OpenAI" in name for name in provider_names)
        assert any("Anthropic" in name for name in provider_names)
    
    def test_show_current_provider(self, llm_provider):
        """Test showing current provider."""
        llm_provider.app.current_provider = "OpenAI"
        llm_provider.handle_llm_command(None, "show_current")
        
        llm_provider.app.notify.assert_called_once()
        call_args = llm_provider.app.notify.call_args[0]
        assert "Current LLM provider: OpenAI" in call_args[0]
    
    def test_provider_switch_request(self, llm_provider):
        """Test provider switch request."""
        llm_provider.handle_llm_command("Anthropic", "switch_Anthropic")
        
        llm_provider.app.notify.assert_called_once()
        call_args = llm_provider.app.notify.call_args[0]
        assert "Provider switch to Anthropic requested" in call_args[0]


#######################################################################################################################
#
# --- SettingsProvider Tests ---

@requires_imports
class TestSettingsProvider:
    """Test suite for SettingsProvider functionality."""
    
    @pytest.fixture
    def settings_provider(self, mock_app):
        """Create a SettingsProvider instance with mock app."""
        mock_screen = MagicMock()
        mock_screen.app = mock_app
        provider = SettingsProvider(screen=mock_screen)
        # Create a mock matcher object
        mock_matcher = MagicMock()
        mock_matcher.match = MagicMock(return_value=1.0)
        mock_matcher.highlight = MagicMock(side_effect=lambda x: x)
        # Set provider.matcher as a callable that returns the matcher object
        provider.matcher = MagicMock(return_value=mock_matcher)
        return provider
    
    @pytest.mark.asyncio
    async def test_discover_shows_popular_settings(self, settings_provider):
        """Test that discover() shows popular settings commands."""
        hits = []
        async for hit in settings_provider.discover():
            hits.append(hit)
        
        assert len(hits) == 4  # Popular settings defined in discover method
        setting_names = [hit.text for hit in hits]
        assert any("Open Settings Tab" in name for name in setting_names)
        assert any("Open Config File" in name for name in setting_names)
    
    def test_open_settings_tab(self, settings_provider):
        """Test opening settings tab."""
        settings_provider.handle_setting("open_settings")
        
        assert settings_provider.app.current_tab == TAB_TOOLS_SETTINGS
        settings_provider.app.notify.assert_called_once()
    
    def test_show_config_path(self, settings_provider):
        """Test showing config file path."""
        with patch('tldw_chatbook.config.DEFAULT_CONFIG_PATH', '/test/config.toml'):
            settings_provider.handle_setting("open_config")
            
            settings_provider.app.notify.assert_called_once()
            call_args = settings_provider.app.notify.call_args[0]
            assert "Config file location" in call_args[0]
    
    @pytest.mark.parametrize("temp_setting,expected_temp", [
        ("temp_low", "0.1"),
        ("temp_med", "0.7"), 
        ("temp_high", "1.0"),
    ])
    def test_temperature_settings(self, settings_provider, temp_setting, expected_temp):
        """Test temperature setting commands."""
        settings_provider.handle_setting(temp_setting)
        
        settings_provider.app.notify.assert_called_once()
        call_args = settings_provider.app.notify.call_args[0]
        assert f"Temperature set to {expected_temp}" in call_args[0]


#######################################################################################################################
#
# --- Integration Tests ---

@requires_imports
class TestCommandPaletteIntegration:
    """Integration tests for command palette providers."""
    
    def test_all_providers_have_required_methods(self):
        """Test that all providers implement required methods."""
        providers = [
            ThemeProvider, TabNavigationProvider, LLMProviderProvider,
            QuickActionsProvider, SettingsProvider, CharacterProvider,
            MediaProvider, DeveloperProvider
        ]
        
        # Create mock screen for provider initialization
        mock_screen = MagicMock()
        mock_screen.app = MagicMock()
        
        for provider_class in providers:
            provider = provider_class(screen=mock_screen)
            
            # Check required async methods
            assert hasattr(provider, 'search')
            assert hasattr(provider, 'discover')
            
            # Check methods are async
            import inspect
            assert inspect.iscoroutinefunction(provider.search)
            assert inspect.iscoroutinefunction(provider.discover)
    
    @pytest.mark.asyncio
    async def test_all_providers_return_hits_from_discover(self, mock_app):
        """Test that all providers return Hit objects from discover()."""
        mock_screen = MagicMock()
        mock_screen.app = mock_app
        
        providers = [
            ThemeProvider(screen=mock_screen), TabNavigationProvider(screen=mock_screen), 
            LLMProviderProvider(screen=mock_screen), QuickActionsProvider(screen=mock_screen),
            SettingsProvider(screen=mock_screen), CharacterProvider(screen=mock_screen),
            MediaProvider(screen=mock_screen), DeveloperProvider(screen=mock_screen)
        ]
        
        for provider in providers:
            # provider.app is accessed via provider.screen.app, no need to set directly
            # Create a mock matcher object
            mock_matcher = MagicMock()
            mock_matcher.match = MagicMock(return_value=1.0)
            mock_matcher.highlight = MagicMock(side_effect=lambda x: x)
            # Set provider.matcher as a callable that returns the matcher object
            provider.matcher = MagicMock(return_value=mock_matcher)
            
            hits = []
            async for hit in provider.discover():
                hits.append(hit)
            
            assert len(hits) > 0, f"{provider.__class__.__name__} should return at least one hit"
            
            for hit in hits:
                assert isinstance(hit, Hit), f"All items should be Hit objects"
                assert hasattr(hit, 'text'), f"Hit should have text attribute"
                assert hasattr(hit, 'help'), f"Hit should have help attribute"
    
    @pytest.mark.asyncio
    async def test_search_consistency_across_providers(self, mock_app):
        """Test that search behaves consistently across providers."""
        mock_screen = MagicMock()
        mock_screen.app = mock_app
        
        providers = [
            ThemeProvider(screen=mock_screen), TabNavigationProvider(screen=mock_screen), 
            LLMProviderProvider(screen=mock_screen), QuickActionsProvider(screen=mock_screen),
            SettingsProvider(screen=mock_screen), CharacterProvider(screen=mock_screen),
            MediaProvider(screen=mock_screen), DeveloperProvider(screen=mock_screen)
        ]
        
        test_queries = ["test", "switch", "open", "new"]
        
        for provider in providers:
            # provider.app is accessed via provider.screen.app, no need to set directly
            # Create a mock matcher object
            mock_matcher = MagicMock()
            mock_matcher.match = MagicMock(return_value=1.0)
            mock_matcher.highlight = MagicMock(side_effect=lambda x: x)
            # Set provider.matcher as a callable that returns the matcher object
            provider.matcher = MagicMock(return_value=mock_matcher)
            
            for query in test_queries:
                hits = []
                async for hit in provider.search(query):
                    hits.append(hit)
                
                # Each provider should handle search gracefully
                # (may return 0 or more hits, but shouldn't error)
                assert isinstance(hits, list), f"{provider.__class__.__name__} search should return list-like results"
    
    def test_provider_error_handling(self, mock_app):
        """Test that providers handle errors gracefully."""
        mock_screen = MagicMock()
        mock_screen.app = mock_app
        
        providers = [
            (ThemeProvider(screen=mock_screen), "switch_theme", ["test-theme"]),
            (TabNavigationProvider(screen=mock_screen), "switch_tab", [TAB_CHAT]),
            (QuickActionsProvider(screen=mock_screen), "execute_quick_action", ["new_chat"]),
            (SettingsProvider(screen=mock_screen), "handle_setting", ["open_settings"]),
        ]
        
        for provider, method_name, args in providers:
            provider.app = mock_app
            
            # Mock app to raise exception
            if hasattr(provider.app, 'current_tab'):
                provider.app.current_tab = MagicMock(side_effect=Exception("Test error"))
            if hasattr(provider.app, 'theme'):
                provider.app.theme = MagicMock(side_effect=Exception("Test error"))
            
            # Method should not raise exception
            method = getattr(provider, method_name)
            try:
                method(*args)
            except Exception as e:
                pytest.fail(f"{provider.__class__.__name__}.{method_name} should handle errors gracefully, but raised: {e}")
            
            # Should call notify with error
            provider.app.notify.assert_called()
            call_args = provider.app.notify.call_args
            if call_args[1].get('severity') == 'error':
                assert "Failed" in call_args[0][0] or "error" in call_args[0][0].lower()


#######################################################################################################################
#
# --- Performance Tests ---

@requires_imports
class TestCommandPalettePerformance:
    """Performance-related tests for command palette providers."""
    
    @pytest.mark.asyncio
    async def test_discover_performance(self, mock_app):
        """Test that discover() methods complete quickly."""
        import time
        
        mock_screen = MagicMock()
        mock_screen.app = mock_app
        
        providers = [
            ThemeProvider(screen=mock_screen), TabNavigationProvider(screen=mock_screen), 
            LLMProviderProvider(screen=mock_screen), QuickActionsProvider(screen=mock_screen),
            SettingsProvider(screen=mock_screen), CharacterProvider(screen=mock_screen),
            MediaProvider(screen=mock_screen), DeveloperProvider(screen=mock_screen)
        ]
        
        for provider in providers:
            # provider.app is accessed via provider.screen.app, no need to set directly
            # Create a mock matcher object
            mock_matcher = MagicMock()
            mock_matcher.match = MagicMock(return_value=1.0)
            mock_matcher.highlight = MagicMock(side_effect=lambda x: x)
            # Set provider.matcher as a callable that returns the matcher object
            provider.matcher = MagicMock(return_value=mock_matcher)
            
            start_time = time.time()
            hits = []
            async for hit in provider.discover():
                hits.append(hit)
            end_time = time.time()
            
            # Discover should complete in under 100ms
            assert (end_time - start_time) < 0.1, f"{provider.__class__.__name__}.discover() took too long"
    
    @pytest.mark.asyncio
    async def test_search_performance(self, mock_app):
        """Test that search() methods complete quickly."""
        import time
        
        mock_screen = MagicMock()
        mock_screen.app = mock_app
        
        providers = [
            TabNavigationProvider(screen=mock_screen), 
            QuickActionsProvider(screen=mock_screen), 
            SettingsProvider(screen=mock_screen)
        ]
        
        for provider in providers:
            # provider.app is accessed via provider.screen.app, no need to set directly
            # Create a mock matcher object
            mock_matcher = MagicMock()
            mock_matcher.match = MagicMock(return_value=1.0)
            mock_matcher.highlight = MagicMock(side_effect=lambda x: x)
            # Set provider.matcher as a callable that returns the matcher object
            provider.matcher = MagicMock(return_value=mock_matcher)
            
            start_time = time.time()
            hits = []
            async for hit in provider.search("test query"):
                hits.append(hit)
            end_time = time.time()
            
            # Search should complete in under 100ms
            assert (end_time - start_time) < 0.1, f"{provider.__class__.__name__}.search() took too long"


#######################################################################################################################
#
# --- Helper Functions for Testing ---

def extract_hit_texts(hits: List[Hit]) -> List[str]:
    """Helper function to extract text from Hit objects."""
    return [hit.text for hit in hits]

def assert_hit_contains_text(hits: List[Hit], expected_text: str):
    """Helper function to assert that hits contain expected text."""
    hit_texts = extract_hit_texts(hits)
    assert any(expected_text in text for text in hit_texts), f"Expected '{expected_text}' in hits: {hit_texts}"

# End of test_command_palette_providers.py