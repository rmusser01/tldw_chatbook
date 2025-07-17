# test_command_palette_basic.py
#
# Basic tests for command palette functionality without complex imports
#
import pytest
from unittest.mock import MagicMock, patch
from textual.command import Hit

@pytest.mark.unit
def test_command_palette_imports():
    """Test that command palette providers can be imported."""
    try:
        from tldw_chatbook.app import ThemeProvider
        assert ThemeProvider is not None
        
        from tldw_chatbook.app import TabNavigationProvider
        assert TabNavigationProvider is not None
        
        from tldw_chatbook.app import QuickActionsProvider
        assert QuickActionsProvider is not None
        
    except ImportError as e:
        pytest.skip(f"Command palette providers not available: {e}")

@pytest.mark.unit
def test_theme_provider_basic():
    """Basic test for ThemeProvider without app context."""
    try:
        from tldw_chatbook.app import ThemeProvider
        
        # Providers need a screen parameter
        mock_screen = MagicMock()
        provider = ThemeProvider(mock_screen)
        
        # Test that provider has required methods
        assert hasattr(provider, 'search')
        assert hasattr(provider, 'discover')
        assert hasattr(provider, 'switch_theme')
        assert hasattr(provider, 'show_theme_submenu')
        
        # Test methods are callable
        assert callable(provider.search)
        assert callable(provider.discover) 
        assert callable(provider.switch_theme)
        assert callable(provider.show_theme_submenu)
        
    except ImportError:
        pytest.skip("ThemeProvider not available")

@pytest.mark.unit
def test_tab_navigation_provider_basic():
    """Basic test for TabNavigationProvider without app context."""
    try:
        from tldw_chatbook.app import TabNavigationProvider
        
        mock_screen = MagicMock()
        provider = TabNavigationProvider(mock_screen)
        
        # Test that provider has required methods
        assert hasattr(provider, 'search')
        assert hasattr(provider, 'discover')
        assert hasattr(provider, 'switch_tab')
        
        # Test methods are callable
        assert callable(provider.search)
        assert callable(provider.discover)
        assert callable(provider.switch_tab)
        
    except ImportError:
        pytest.skip("TabNavigationProvider not available")

@pytest.mark.unit
def test_constants_available():
    """Test that tab constants are available."""
    try:
        from tldw_chatbook.Constants import TAB_CHAT, TAB_NOTES, TAB_SEARCH
        
        assert TAB_CHAT == "chat"
        assert TAB_NOTES == "notes" 
        assert TAB_SEARCH == "search"
        
    except ImportError:
        pytest.skip("Constants not available")

@pytest.mark.asyncio
@pytest.mark.unit
async def test_theme_provider_discover_structure():
    """Test that ThemeProvider.discover returns proper structure."""
    try:
        from tldw_chatbook.app import ThemeProvider
        
        # Create mock screen with app attached
        mock_app = MagicMock()
        mock_screen = MagicMock()
        mock_screen.app = mock_app
        
        provider = ThemeProvider(mock_screen)
        
        # Mock the matcher to return proper numeric values
        mock_matcher = MagicMock()
        mock_matcher.match = MagicMock(return_value=1.0)
        mock_matcher.highlight = MagicMock(side_effect=lambda x: x)
        provider.matcher = MagicMock(return_value=mock_matcher)
        
        hits = []
        async for hit in provider.discover():
            hits.append(hit)
        
        # Should return at least one hit
        assert len(hits) >= 1
        
        # First hit should be the main theme command
        assert "Theme: Change Theme" in hits[0].text
        
    except ImportError:
        pytest.skip("ThemeProvider not available")
    except Exception as e:
        pytest.fail(f"discover() method failed: {e}")

@pytest.mark.unit
def test_provider_error_handling_basic():
    """Test basic error handling in providers."""
    try:
        from tldw_chatbook.app import ThemeProvider
        
        # Create mock screen with app attached
        mock_app = MagicMock()
        mock_app.notify = MagicMock()
        # Mock the theme setter to raise an exception
        type(mock_app).theme = property(lambda self: None, lambda self, value: (_ for _ in ()).throw(Exception("Test error")))
        
        mock_screen = MagicMock()
        mock_screen.app = mock_app
        
        provider = ThemeProvider(mock_screen)
        
        # Should not raise exception
        provider.switch_theme("test-theme")
        
        # Should call notify with error
        mock_app.notify.assert_called_once()
        call_args = mock_app.notify.call_args
        assert call_args[1]['severity'] == "error"
        assert "Failed to apply theme" in call_args[0][0]
        
    except ImportError:
        pytest.skip("ThemeProvider not available")

def test_app_commands_registration():
    """Test that app has command providers registered.""" 
    try:
        from tldw_chatbook.app import TldwCli
        
        # Check that app has COMMANDS attribute
        assert hasattr(TldwCli, 'COMMANDS')
        
        # Should be a set or similar collection
        commands = TldwCli.COMMANDS
        assert commands is not None
        
        # Should have multiple providers
        assert len(commands) > 1
        
    except ImportError:
        pytest.skip("TldwCli not available")

def test_keybinding_registration():
    """Test that Ctrl+P binding is registered."""
    try:
        from tldw_chatbook.app import TldwCli
        
        # Check that app has BINDINGS
        assert hasattr(TldwCli, 'BINDINGS')
        
        bindings = TldwCli.BINDINGS
        assert bindings is not None
        
        # Check for Ctrl+P binding
        ctrl_p_bindings = [b for b in bindings if "ctrl+p" in str(b.key).lower()]
        assert len(ctrl_p_bindings) >= 1
        
        # Check for command_palette action
        palette_bindings = [b for b in bindings if "command_palette" in str(b.action)]
        assert len(palette_bindings) >= 1
        
    except ImportError:
        pytest.skip("TldwCli not available")

def test_theme_config_integration():
    """Test that theme config integration works."""
    try:
        with patch('tldw_chatbook.app.get_cli_setting') as mock_get_setting:
            with patch('tldw_chatbook.config.save_setting_to_cli_config') as mock_save_setting:
                from tldw_chatbook.app import ThemeProvider
                
                # Create mock screen with app attached
                mock_app = MagicMock()
                mock_screen = MagicMock()
                mock_screen.app = mock_app
                
                provider = ThemeProvider(mock_screen)
                
                # Test theme switching saves to config
                provider.switch_theme("test-theme")
                
                # Should save to config
                mock_save_setting.assert_called_once_with("general", "default_theme", "test-theme")
                
    except ImportError:
        pytest.skip("ThemeProvider or config functions not available")

# End of test_command_palette_basic.py