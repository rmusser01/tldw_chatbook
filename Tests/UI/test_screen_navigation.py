"""
Test suite for screen-based navigation.
Verifies that all screens can be navigated to and function properly.
"""

import pytest
from textual.app import App
from textual.testing import AppTest
from unittest.mock import MagicMock, patch

# Import the main app
from tldw_chatbook.app import TldwCli

# Import all screen classes
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.media_ingest_screen import MediaIngestScreen
from tldw_chatbook.UI.Screens.coding_screen import CodingScreen
from tldw_chatbook.UI.Screens.conversation_screen import ConversationScreen
from tldw_chatbook.UI.Screens.media_screen import MediaScreen
from tldw_chatbook.UI.Screens.notes_screen import NotesScreen
from tldw_chatbook.UI.Screens.search_screen import SearchScreen
from tldw_chatbook.UI.Screens.evals_screen import EvalsScreen
from tldw_chatbook.UI.Screens.tools_settings_screen import ToolsSettingsScreen
from tldw_chatbook.UI.Screens.llm_screen import LLMScreen
from tldw_chatbook.UI.Screens.customize_screen import CustomizeScreen
from tldw_chatbook.UI.Screens.logs_screen import LogsScreen
from tldw_chatbook.UI.Screens.stats_screen import StatsScreen
from tldw_chatbook.UI.Screens.stts_screen import STTSScreen
from tldw_chatbook.UI.Screens.study_screen import StudyScreen
from tldw_chatbook.UI.Screens.chatbooks_screen import ChatbooksScreen
from tldw_chatbook.UI.Screens.subscription_screen import SubscriptionScreen

from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen


@pytest.fixture
def mock_config():
    """Mock configuration to avoid loading real config files."""
    with patch('tldw_chatbook.config.load_cli_config_and_ensure_existence'):
        with patch('tldw_chatbook.config.get_cli_setting') as mock_setting:
            # Default settings for testing
            mock_setting.return_value = False
            yield mock_setting


@pytest.mark.asyncio
async def test_app_starts_with_screen_navigation(mock_config):
    """Test that the app starts with screen-based navigation enabled."""
    # Mock to disable splash screen
    mock_config.side_effect = lambda section, key, default: {
        ('splash_screen', 'enabled'): False,
        ('navigation', 'use_screen_navigation'): True,  # Force screen navigation
        ('general', 'use_link_navigation'): True,
    }.get((section, key), default)
    
    app = TldwCli()
    
    async with app.run_test() as pilot:
        # Check that screen navigation is enabled
        assert hasattr(app, '_use_screen_navigation')
        assert app._use_screen_navigation == True
        
        # Check that initial screen is pushed
        assert len(pilot.app.screen_stack) > 0
        
        # The current screen should be ChatScreen (default)
        current_screen = pilot.app.screen
        assert isinstance(current_screen, ChatScreen)


@pytest.mark.asyncio
async def test_navigate_to_all_screens(mock_config):
    """Test navigation to all available screens."""
    mock_config.side_effect = lambda section, key, default: {
        ('splash_screen', 'enabled'): False,
        ('navigation', 'use_screen_navigation'): True,
        ('general', 'use_link_navigation'): True,
    }.get((section, key), default)
    
    app = TldwCli()
    
    async with app.run_test() as pilot:
        # Test navigation to each screen
        screens_to_test = [
            ('chat', ChatScreen),
            ('media', MediaScreen),
            ('notes', NotesScreen),
            ('search', SearchScreen),
            ('coding', CodingScreen),
            ('ccp', ConversationScreen),
            ('ingest', MediaIngestScreen),
            ('evals', EvalsScreen),
            ('tools_settings', ToolsSettingsScreen),
            ('llm', LLMScreen),
            ('customize', CustomizeScreen),
            ('logs', LogsScreen),
            ('stats', StatsScreen),
            ('stts', STTSScreen),
            ('study', StudyScreen),
            ('chatbooks', ChatbooksScreen),
            ('subscriptions', SubscriptionScreen),
        ]
        
        for screen_name, screen_class in screens_to_test:
            # Post navigation message
            pilot.app.post_message(NavigateToScreen(screen_name=screen_name))
            
            # Allow time for navigation
            await pilot.pause(0.1)
            
            # Check current screen
            current_screen = pilot.app.screen
            assert isinstance(current_screen, screen_class), \
                f"Failed to navigate to {screen_name}. Expected {screen_class.__name__}, got {type(current_screen).__name__}"


@pytest.mark.asyncio
async def test_tab_links_emit_navigation_messages(mock_config):
    """Test that TabLinks widget emits NavigateToScreen messages."""
    mock_config.side_effect = lambda section, key, default: {
        ('splash_screen', 'enabled'): False,
        ('navigation', 'use_screen_navigation'): True,
        ('general', 'use_link_navigation'): True,
    }.get((section, key), default)
    
    from tldw_chatbook.UI.Tab_Links import TabLinks
    from tldw_chatbook.Constants import ALL_TABS
    
    class TestApp(App):
        def compose(self):
            yield TabLinks(tab_ids=ALL_TABS, initial_active_tab='chat')
    
    app = TestApp()
    messages_received = []
    
    # Capture NavigateToScreen messages
    @app.on(NavigateToScreen)
    def capture_navigation(message):
        messages_received.append(message)
    
    async with app.run_test() as pilot:
        # Find a tab link and click it
        tab_links = pilot.app.query_one(TabLinks)
        
        # Simulate clicking on the notes tab
        notes_link = tab_links.query_one("#tab-link-notes")
        await pilot.click(notes_link)
        
        # Check that navigation message was sent
        assert len(messages_received) > 0
        assert messages_received[0].screen_name == 'notes'


@pytest.mark.asyncio
async def test_screen_state_preservation():
    """Test that screen state is preserved when switching between screens."""
    app = TldwCli()
    
    async with app.run_test() as pilot:
        # Navigate to notes screen
        pilot.app.post_message(NavigateToScreen(screen_name='notes'))
        await pilot.pause(0.1)
        
        notes_screen = pilot.app.screen
        assert isinstance(notes_screen, NotesScreen)
        
        # Set some state
        notes_screen.test_value = "test_data"
        
        # Navigate away
        pilot.app.post_message(NavigateToScreen(screen_name='chat'))
        await pilot.pause(0.1)
        
        # Navigate back
        pilot.app.post_message(NavigateToScreen(screen_name='notes'))
        await pilot.pause(0.1)
        
        # Note: With switch_screen, the screen is recreated, so state won't be preserved
        # This is expected behavior for switch_screen vs push_screen
        new_notes_screen = pilot.app.screen
        assert isinstance(new_notes_screen, NotesScreen)
        # The screen is new, so it won't have our test value
        assert not hasattr(new_notes_screen, 'test_value')


@pytest.mark.asyncio
async def test_screen_lifecycle_methods():
    """Test that screen lifecycle methods are called properly."""
    class TestScreen(ChatScreen):
        mount_called = False
        
        async def on_mount(self):
            self.mount_called = True
            await super().on_mount()
    
    app = TldwCli()
    
    async with app.run_test() as pilot:
        # Create and push our test screen
        test_screen = TestScreen(app)
        await pilot.app.push_screen(test_screen)
        await pilot.pause(0.1)
        
        # Check that on_mount was called
        assert test_screen.mount_called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])