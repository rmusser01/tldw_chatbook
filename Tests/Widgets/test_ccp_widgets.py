"""
Unit tests for CCP widget components following Textual testing best practices.

This module tests all CCP widgets in isolation:
- CCPSidebarWidget
- CCPConversationViewWidget
- CCPCharacterCardWidget
- CCPCharacterEditorWidget
- CCPPromptEditorWidget
- CCPDictionaryEditorWidget
"""

import pytest
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from textual.app import App, ComposeResult
from textual.pilot import Pilot
from textual.widgets import Button, Input, TextArea, ListView, Select, Static, Label
from textual.containers import Container

from tldw_chatbook.Widgets.CCP_Widgets import (
    # Sidebar Widget
    CCPSidebarWidget,
    ConversationSearchRequested,
    ConversationLoadRequested,
    CharacterLoadRequested,
    PromptLoadRequested,
    DictionaryLoadRequested,
    ImportRequested,
    CreateRequested,
    RefreshRequested,
    
    # Conversation View Widget
    CCPConversationViewWidget,
    ConversationMessageWidget,
    MessageSelected,
    MessageEditRequested,
    MessageDeleteRequested,
    RegenerateRequested,
    ContinueConversationRequested,
    
    # Character Card Widget
    CCPCharacterCardWidget,
    EditCharacterRequested,
    CloneCharacterRequested,
    ExportCharacterRequested,
    DeleteCharacterRequested,
    StartChatRequested,
    
    # Character Editor Widget
    CCPCharacterEditorWidget,
    CharacterSaveRequested,
    CharacterFieldGenerateRequested,
    CharacterImageUploadRequested,
    CharacterImageGenerateRequested,
    CharacterEditorCancelled,
    AlternateGreetingAdded,
    AlternateGreetingRemoved,
    
    # Prompt Editor Widget
    CCPPromptEditorWidget,
    PromptSaveRequested,
    PromptDeleteRequested,
    PromptTestRequested,
    PromptEditorCancelled,
    PromptVariableAdded,
    PromptVariableRemoved,
    
    # Dictionary Editor Widget
    CCPDictionaryEditorWidget,
    DictionarySaveRequested,
    DictionaryDeleteRequested,
    DictionaryEntryAdded,
    DictionaryEntryRemoved,
    DictionaryEntryUpdated,
    DictionaryImportRequested,
    DictionaryExportRequested,
    DictionaryEditorCancelled,
)


# ========== Test Fixtures ==========

@pytest.fixture
def mock_parent_screen():
    """Create a mock parent screen with state."""
    from tldw_chatbook.UI.Screens.ccp_screen import CCPScreenState
    
    screen = Mock()
    screen.state = CCPScreenState()
    screen.app_instance = Mock()
    return screen


@pytest.fixture
def sample_character_data():
    """Sample character data for testing."""
    return {
        'id': 1,
        'name': 'Alice',
        'description': 'A helpful AI assistant',
        'personality': 'Friendly, knowledgeable, and patient',
        'scenario': 'You are chatting with Alice, an AI assistant',
        'first_message': 'Hello! How can I help you today?',
        'keywords': 'assistant,AI,helpful',
        'creator': 'TestUser',
        'version': '1.0',
        'alternate_greetings': [
            'Hi there! What can I do for you?',
            'Welcome! How may I assist you?'
        ],
        'tags': ['assistant', 'AI'],
        'system_prompt': 'You are a helpful assistant',
        'post_history_instructions': 'Remember to be helpful',
        'creator_notes': 'This is a test character'
    }


@pytest.fixture
def sample_prompt_data():
    """Sample prompt data for testing."""
    return {
        'id': 1,
        'name': 'Story Generator',
        'description': 'Generates creative stories',
        'content': 'Write a story about {{topic}} with {{characters}} characters',
        'category': 'creative',
        'is_system': False,
        'variables': [
            {'name': 'topic', 'type': 'text'},
            {'name': 'characters', 'type': 'number'}
        ]
    }


@pytest.fixture
def sample_dictionary_data():
    """Sample dictionary data for testing."""
    return {
        'id': 1,
        'name': 'Fantasy World',
        'description': 'A fantasy world dictionary',
        'strategy': 'sorted_evenly',
        'max_tokens': 1000,
        'entries': [
            {
                'key': 'Eldoria',
                'value': 'A magical kingdom in the north',
                'group': 'locations',
                'probability': 100
            },
            {
                'key': 'Dragon',
                'value': 'A mythical creature that breathes fire',
                'group': 'creatures',
                'probability': 80
            }
        ]
    }


# ========== CCPSidebarWidget Tests ==========

class TestCCPSidebarWidget:
    """Tests for CCPSidebarWidget."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, mock_parent_screen):
        """Test sidebar widget initialization."""
        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield CCPSidebarWidget(parent_screen=mock_parent_screen)
        
        app = TestApp()
        async with app.run_test() as pilot:
            sidebar = pilot.app.query_one(CCPSidebarWidget)
            
            # Check widget exists and has correct ID
            assert sidebar is not None
            assert sidebar.id == "ccp-sidebar"
            assert sidebar.has_class("ccp-sidebar")
            
            # Check state binding
            assert sidebar.state == mock_parent_screen.state
    
    @pytest.mark.asyncio
    async def test_search_input_posts_message(self, mock_parent_screen):
        """Test search input posts ConversationSearchRequested message."""
        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield CCPSidebarWidget(parent_screen=mock_parent_screen)
        
        app = TestApp()
        messages = []
        
        def on_conversation_search_requested(msg: ConversationSearchRequested):
            messages.append(msg)
        
        app.on_conversation_search_requested = on_conversation_search_requested
        
        async with app.run_test() as pilot:
            # Find search input and type
            search_input = pilot.app.query_one("#ccp-conversation-search-input", Input)
            search_input.value = "test search"
            
            # Trigger change event
            await pilot.pause()
            
            # Message should be posted
            assert len(messages) > 0
            assert messages[0].search_term == "test search"
            assert messages[0].search_type == "title"
    
    @pytest.mark.asyncio
    async def test_load_button_posts_message(self, mock_parent_screen):
        """Test load conversation button posts message."""
        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield CCPSidebarWidget(parent_screen=mock_parent_screen)
        
        app = TestApp()
        messages = []
        
        def on_conversation_load_requested(msg: ConversationLoadRequested):
            messages.append(msg)
        
        app.on_conversation_load_requested = on_conversation_load_requested
        
        async with app.run_test() as pilot:
            # Click load button
            await pilot.click("#ccp-load-conversation-button")
            await pilot.pause()
            
            # Message should be posted
            assert len(messages) > 0
            assert messages[0].conversation_id is None  # No specific ID
    
    @pytest.mark.asyncio
    async def test_character_section_interaction(self, mock_parent_screen):
        """Test character section interactions."""
        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield CCPSidebarWidget(parent_screen=mock_parent_screen)
        
        app = TestApp()
        messages = []
        
        def on_character_load_requested(msg: CharacterLoadRequested):
            messages.append(msg)
        
        app.on_character_load_requested = on_character_load_requested
        
        async with app.run_test() as pilot:
            # Click load character button
            await pilot.click("#ccp-load-character-button")
            await pilot.pause()
            
            # Message should be posted
            assert len(messages) > 0
    
    @pytest.mark.asyncio
    async def test_collapsible_sections(self, mock_parent_screen):
        """Test collapsible sections can be toggled."""
        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield CCPSidebarWidget(parent_screen=mock_parent_screen)
        
        app = TestApp()
        
        async with app.run_test() as pilot:
            # Find a collapsible section
            from textual.widgets import Collapsible
            collapsibles = pilot.app.query(Collapsible)
            
            if collapsibles:
                collapsible = collapsibles[0]
                initial_state = collapsible.collapsed
                
                # Toggle it
                await pilot.click(collapsible)
                await pilot.pause()
                
                # State should change
                assert collapsible.collapsed != initial_state


# ========== CCPConversationViewWidget Tests ==========

class TestCCPConversationViewWidget:
    """Tests for CCPConversationViewWidget."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, mock_parent_screen):
        """Test conversation view widget initialization."""
        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield CCPConversationViewWidget(parent_screen=mock_parent_screen)
        
        app = TestApp()
        async with app.run_test() as pilot:
            widget = pilot.app.query_one(CCPConversationViewWidget)
            
            assert widget is not None
            assert widget.id == "ccp-conversation-messages-view"
            assert widget.messages == []
    
    @pytest.mark.asyncio
    async def test_load_messages(self, mock_parent_screen):
        """Test loading messages into the widget."""
        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield CCPConversationViewWidget(parent_screen=mock_parent_screen)
        
        app = TestApp()
        
        async with app.run_test() as pilot:
            widget = pilot.app.query_one(CCPConversationViewWidget)
            
            # Load test messages
            test_messages = [
                {'id': 1, 'role': 'user', 'content': 'Hello'},
                {'id': 2, 'role': 'assistant', 'content': 'Hi there!'}
            ]
            
            widget.load_messages(test_messages)
            
            # Check messages loaded
            assert widget.messages == test_messages
            
            # Check message widgets created
            message_widgets = widget.query(ConversationMessageWidget)
            assert len(message_widgets) == 2
    
    @pytest.mark.asyncio
    async def test_clear_messages(self, mock_parent_screen):
        """Test clearing messages."""
        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield CCPConversationViewWidget(parent_screen=mock_parent_screen)
        
        app = TestApp()
        
        async with app.run_test() as pilot:
            widget = pilot.app.query_one(CCPConversationViewWidget)
            
            # Load then clear
            widget.load_messages([{'id': 1, 'role': 'user', 'content': 'Test'}])
            assert len(widget.messages) == 1
            
            widget.clear_messages()
            assert len(widget.messages) == 0
    
    @pytest.mark.asyncio
    async def test_message_selection_posts_event(self, mock_parent_screen):
        """Test selecting a message posts MessageSelected event."""
        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield CCPConversationViewWidget(parent_screen=mock_parent_screen)
        
        app = TestApp()
        messages = []
        
        def on_message_selected(msg: MessageSelected):
            messages.append(msg)
        
        app.on_message_selected = on_message_selected
        
        async with app.run_test() as pilot:
            widget = pilot.app.query_one(CCPConversationViewWidget)
            
            # Load a message
            widget.load_messages([{'id': 1, 'role': 'user', 'content': 'Test'}])
            await pilot.pause()
            
            # Click the message widget
            msg_widget = widget.query_one(ConversationMessageWidget)
            await pilot.click(msg_widget)
            await pilot.pause()
            
            # Event should be posted
            assert len(messages) > 0
            assert messages[0].message_id == 1


# ========== CCPCharacterCardWidget Tests ==========

class TestCCPCharacterCardWidget:
    """Tests for CCPCharacterCardWidget."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, mock_parent_screen):
        """Test character card widget initialization."""
        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield CCPCharacterCardWidget(parent_screen=mock_parent_screen)
        
        app = TestApp()
        async with app.run_test() as pilot:
            widget = pilot.app.query_one(CCPCharacterCardWidget)
            
            assert widget is not None
            assert widget.id == "ccp-character-card-view"
            assert widget.character_data == {}
    
    @pytest.mark.asyncio
    async def test_load_character(self, mock_parent_screen, sample_character_data):
        """Test loading character data."""
        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield CCPCharacterCardWidget(parent_screen=mock_parent_screen)
        
        app = TestApp()
        
        async with app.run_test() as pilot:
            widget = pilot.app.query_one(CCPCharacterCardWidget)
            
            # Load character
            widget.load_character(sample_character_data)
            
            # Check data loaded
            assert widget.character_data == sample_character_data
            
            # Check UI updated (name field)
            name_display = widget.query_one("#ccp-card-name-display", Static)
            assert sample_character_data['name'] in name_display.renderable
    
    @pytest.mark.asyncio
    async def test_edit_button_posts_message(self, mock_parent_screen, sample_character_data):
        """Test edit button posts EditCharacterRequested message."""
        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield CCPCharacterCardWidget(parent_screen=mock_parent_screen)
        
        app = TestApp()
        messages = []
        
        def on_edit_character_requested(msg: EditCharacterRequested):
            messages.append(msg)
        
        app.on_edit_character_requested = on_edit_character_requested
        
        async with app.run_test() as pilot:
            widget = pilot.app.query_one(CCPCharacterCardWidget)
            widget.load_character(sample_character_data)
            
            # Click edit button
            await pilot.click("#ccp-card-edit-button")
            await pilot.pause()
            
            # Message should be posted
            assert len(messages) > 0
            assert messages[0].character_id == sample_character_data['id']
    
    @pytest.mark.asyncio
    async def test_start_chat_button(self, mock_parent_screen, sample_character_data):
        """Test start chat button posts message."""
        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield CCPCharacterCardWidget(parent_screen=mock_parent_screen)
        
        app = TestApp()
        messages = []
        
        def on_start_chat_requested(msg: StartChatRequested):
            messages.append(msg)
        
        app.on_start_chat_requested = on_start_chat_requested
        
        async with app.run_test() as pilot:
            widget = pilot.app.query_one(CCPCharacterCardWidget)
            widget.load_character(sample_character_data)
            
            # Click start chat button
            await pilot.click("#ccp-card-start-chat-button")
            await pilot.pause()
            
            # Message should be posted
            assert len(messages) > 0
            assert messages[0].character_id == sample_character_data['id']


# ========== CCPCharacterEditorWidget Tests ==========

class TestCCPCharacterEditorWidget:
    """Tests for CCPCharacterEditorWidget."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, mock_parent_screen):
        """Test character editor widget initialization."""
        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield CCPCharacterEditorWidget(parent_screen=mock_parent_screen)
        
        app = TestApp()
        async with app.run_test() as pilot:
            widget = pilot.app.query_one(CCPCharacterEditorWidget)
            
            assert widget is not None
            assert widget.id == "ccp-character-editor-view"
            assert widget.character_data == {}
    
    @pytest.mark.asyncio
    async def test_load_character_for_editing(self, mock_parent_screen, sample_character_data):
        """Test loading character data into editor."""
        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield CCPCharacterEditorWidget(parent_screen=mock_parent_screen)
        
        app = TestApp()
        
        async with app.run_test() as pilot:
            widget = pilot.app.query_one(CCPCharacterEditorWidget)
            
            # Load character
            widget.load_character(sample_character_data)
            
            # Check fields populated
            name_input = widget.query_one("#ccp-char-name", Input)
            assert name_input.value == sample_character_data['name']
            
            desc_area = widget.query_one("#ccp-char-description", TextArea)
            assert desc_area.text == sample_character_data['description']
    
    @pytest.mark.asyncio
    async def test_save_button_validation(self, mock_parent_screen):
        """Test save button validates required fields."""
        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield CCPCharacterEditorWidget(parent_screen=mock_parent_screen)
        
        app = TestApp()
        messages = []
        
        def on_character_save_requested(msg: CharacterSaveRequested):
            messages.append(msg)
        
        app.on_character_save_requested = on_character_save_requested
        
        async with app.run_test() as pilot:
            widget = pilot.app.query_one(CCPCharacterEditorWidget)
            
            # Try to save without name
            await pilot.click("#save-character-btn")
            await pilot.pause()
            
            # No message should be posted (validation failed)
            assert len(messages) == 0
            
            # Set name and try again
            name_input = widget.query_one("#ccp-char-name", Input)
            name_input.value = "Test Character"
            
            await pilot.click("#save-character-btn")
            await pilot.pause()
            
            # Now message should be posted
            assert len(messages) > 0
            assert messages[0].character_data['name'] == "Test Character"
    
    @pytest.mark.asyncio
    async def test_alternate_greetings_management(self, mock_parent_screen):
        """Test adding and removing alternate greetings."""
        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield CCPCharacterEditorWidget(parent_screen=mock_parent_screen)
        
        app = TestApp()
        add_messages = []
        remove_messages = []
        
        def on_alternate_greeting_added(msg: AlternateGreetingAdded):
            add_messages.append(msg)
        
        def on_alternate_greeting_removed(msg: AlternateGreetingRemoved):
            remove_messages.append(msg)
        
        app.on_alternate_greeting_added = on_alternate_greeting_added
        app.on_alternate_greeting_removed = on_alternate_greeting_removed
        
        async with app.run_test() as pilot:
            widget = pilot.app.query_one(CCPCharacterEditorWidget)
            
            # Add a greeting
            greeting_input = widget.query_one("#alt-greeting-input", Input)
            greeting_input.value = "Hello there!"
            
            await pilot.click("#add-alt-greeting-btn")
            await pilot.pause()
            
            # Check greeting added
            assert len(widget.alternate_greetings) == 1
            assert widget.alternate_greetings[0] == "Hello there!"
            assert len(add_messages) > 0
            
            # Remove the greeting
            await pilot.click(".remove-greeting-btn")
            await pilot.pause()
            
            # Check greeting removed
            assert len(widget.alternate_greetings) == 0
            assert len(remove_messages) > 0


# ========== CCPPromptEditorWidget Tests ==========

class TestCCPPromptEditorWidget:
    """Tests for CCPPromptEditorWidget."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, mock_parent_screen):
        """Test prompt editor widget initialization."""
        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield CCPPromptEditorWidget(parent_screen=mock_parent_screen)
        
        app = TestApp()
        async with app.run_test() as pilot:
            widget = pilot.app.query_one(CCPPromptEditorWidget)
            
            assert widget is not None
            assert widget.id == "ccp-prompt-editor-view"
            assert widget.prompt_data == {}
            assert widget.variables == []
    
    @pytest.mark.asyncio
    async def test_load_prompt(self, mock_parent_screen, sample_prompt_data):
        """Test loading prompt data."""
        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield CCPPromptEditorWidget(parent_screen=mock_parent_screen)
        
        app = TestApp()
        
        async with app.run_test() as pilot:
            widget = pilot.app.query_one(CCPPromptEditorWidget)
            
            # Load prompt
            widget.load_prompt(sample_prompt_data)
            
            # Check data loaded
            assert widget.prompt_data == sample_prompt_data
            assert widget.variables == sample_prompt_data['variables']
            
            # Check fields populated
            name_input = widget.query_one("#ccp-prompt-name", Input)
            assert name_input.value == sample_prompt_data['name']
    
    @pytest.mark.asyncio
    async def test_variable_management(self, mock_parent_screen):
        """Test adding and removing variables."""
        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield CCPPromptEditorWidget(parent_screen=mock_parent_screen)
        
        app = TestApp()
        add_messages = []
        remove_messages = []
        
        def on_prompt_variable_added(msg: PromptVariableAdded):
            add_messages.append(msg)
        
        def on_prompt_variable_removed(msg: PromptVariableRemoved):
            remove_messages.append(msg)
        
        app.on_prompt_variable_added = on_prompt_variable_added
        app.on_prompt_variable_removed = on_prompt_variable_removed
        
        async with app.run_test() as pilot:
            widget = pilot.app.query_one(CCPPromptEditorWidget)
            
            # Add a variable
            var_input = widget.query_one("#ccp-variable-name-input", Input)
            var_input.value = "topic"
            
            await pilot.click("#add-variable-btn")
            await pilot.pause()
            
            # Check variable added
            assert len(widget.variables) == 1
            assert widget.variables[0]['name'] == "topic"
            assert len(add_messages) > 0
            
            # Remove the variable
            await pilot.click(".remove-var-btn")
            await pilot.pause()
            
            # Check variable removed
            assert len(widget.variables) == 0
            assert len(remove_messages) > 0
    
    @pytest.mark.asyncio
    async def test_preview_updates(self, mock_parent_screen):
        """Test preview updates when content changes."""
        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield CCPPromptEditorWidget(parent_screen=mock_parent_screen)
        
        app = TestApp()
        
        async with app.run_test() as pilot:
            widget = pilot.app.query_one(CCPPromptEditorWidget)
            
            # Add a variable
            widget.variables = [{'name': 'topic', 'type': 'text'}]
            
            # Set prompt content
            content_area = widget.query_one("#ccp-prompt-content", TextArea)
            content_area.text = "Write about {{topic}}"
            
            # Trigger preview update
            widget._update_preview()
            
            # Check preview shows variable highlighted
            preview_container = widget.query_one("#ccp-prompt-preview", Container)
            # Preview should have content
            assert preview_container.children
    
    @pytest.mark.asyncio
    async def test_test_prompt_button(self, mock_parent_screen, sample_prompt_data):
        """Test the test prompt button posts message."""
        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield CCPPromptEditorWidget(parent_screen=mock_parent_screen)
        
        app = TestApp()
        messages = []
        
        def on_prompt_test_requested(msg: PromptTestRequested):
            messages.append(msg)
        
        app.on_prompt_test_requested = on_prompt_test_requested
        
        async with app.run_test() as pilot:
            widget = pilot.app.query_one(CCPPromptEditorWidget)
            widget.load_prompt(sample_prompt_data)
            
            # Set test values for variables
            # (Would need to find and fill test inputs)
            
            # Click test button
            await pilot.click("#test-prompt-btn")
            await pilot.pause()
            
            # Message should be posted
            assert len(messages) > 0


# ========== CCPDictionaryEditorWidget Tests ==========

class TestCCPDictionaryEditorWidget:
    """Tests for CCPDictionaryEditorWidget."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, mock_parent_screen):
        """Test dictionary editor widget initialization."""
        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield CCPDictionaryEditorWidget(parent_screen=mock_parent_screen)
        
        app = TestApp()
        async with app.run_test() as pilot:
            widget = pilot.app.query_one(CCPDictionaryEditorWidget)
            
            assert widget is not None
            assert widget.id == "ccp-dictionary-editor-view"
            assert widget.dictionary_data == {}
            assert widget.entries == []
    
    @pytest.mark.asyncio
    async def test_load_dictionary(self, mock_parent_screen, sample_dictionary_data):
        """Test loading dictionary data."""
        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield CCPDictionaryEditorWidget(parent_screen=mock_parent_screen)
        
        app = TestApp()
        
        async with app.run_test() as pilot:
            widget = pilot.app.query_one(CCPDictionaryEditorWidget)
            
            # Load dictionary
            widget.load_dictionary(sample_dictionary_data)
            
            # Check data loaded
            assert widget.dictionary_data == sample_dictionary_data
            assert widget.entries == sample_dictionary_data['entries']
            
            # Check fields populated
            name_input = widget.query_one("#ccp-dict-name", Input)
            assert name_input.value == sample_dictionary_data['name']
    
    @pytest.mark.asyncio
    async def test_entry_management(self, mock_parent_screen):
        """Test adding and removing dictionary entries."""
        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield CCPDictionaryEditorWidget(parent_screen=mock_parent_screen)
        
        app = TestApp()
        add_messages = []
        remove_messages = []
        
        def on_dictionary_entry_added(msg: DictionaryEntryAdded):
            add_messages.append(msg)
        
        def on_dictionary_entry_removed(msg: DictionaryEntryRemoved):
            remove_messages.append(msg)
        
        app.on_dictionary_entry_added = on_dictionary_entry_added
        app.on_dictionary_entry_removed = on_dictionary_entry_removed
        
        async with app.run_test() as pilot:
            widget = pilot.app.query_one(CCPDictionaryEditorWidget)
            
            # Add an entry
            key_input = widget.query_one("#entry-key-input", Input)
            key_input.value = "TestKey"
            
            value_area = widget.query_one("#entry-value-textarea", TextArea)
            value_area.text = "Test value"
            
            await pilot.click("#add-entry-btn")
            await pilot.pause()
            
            # Check entry added
            assert len(widget.entries) == 1
            assert widget.entries[0]['key'] == "TestKey"
            assert len(add_messages) > 0
            
            # Select and remove the entry
            widget.selected_entry_index = 0
            await pilot.click("#remove-entry-btn")
            await pilot.pause()
            
            # Check entry removed
            assert len(widget.entries) == 0
            assert len(remove_messages) > 0
    
    @pytest.mark.asyncio
    async def test_import_export_buttons(self, mock_parent_screen):
        """Test import and export buttons post messages."""
        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield CCPDictionaryEditorWidget(parent_screen=mock_parent_screen)
        
        app = TestApp()
        import_messages = []
        export_messages = []
        
        def on_dictionary_import_requested(msg: DictionaryImportRequested):
            import_messages.append(msg)
        
        def on_dictionary_export_requested(msg: DictionaryExportRequested):
            export_messages.append(msg)
        
        app.on_dictionary_import_requested = on_dictionary_import_requested
        app.on_dictionary_export_requested = on_dictionary_export_requested
        
        async with app.run_test() as pilot:
            widget = pilot.app.query_one(CCPDictionaryEditorWidget)
            
            # Click import button
            await pilot.click("#import-dict-btn")
            await pilot.pause()
            
            # Import message should be posted
            assert len(import_messages) > 0
            
            # Click export button
            await pilot.click("#export-dict-btn")
            await pilot.pause()
            
            # Export message should be posted
            assert len(export_messages) > 0
    
    @pytest.mark.asyncio
    async def test_strategy_selection(self, mock_parent_screen):
        """Test strategy selection dropdown."""
        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield CCPDictionaryEditorWidget(parent_screen=mock_parent_screen)
        
        app = TestApp()
        
        async with app.run_test() as pilot:
            widget = pilot.app.query_one(CCPDictionaryEditorWidget)
            
            # Get strategy select
            strategy_select = widget.query_one("#ccp-dict-strategy", Select)
            
            # Check default value
            assert strategy_select.value == "sorted_evenly"
            
            # Change value
            strategy_select.value = "character_lore_first"
            
            # Get updated data
            data = widget.get_dictionary_data()
            assert data['strategy'] == "character_lore_first"


# ========== Message Tests ==========

class TestCCPMessages:
    """Test all CCP message types."""
    
    def test_sidebar_messages(self):
        """Test sidebar widget messages."""
        # ConversationSearchRequested
        msg = ConversationSearchRequested("search term", "content")
        assert msg.search_term == "search term"
        assert msg.search_type == "content"
        
        # ConversationLoadRequested
        msg = ConversationLoadRequested(conversation_id=123)
        assert msg.conversation_id == 123
        
        # CharacterLoadRequested
        msg = CharacterLoadRequested(character_id=456)
        assert msg.character_id == 456
        
        # ImportRequested
        msg = ImportRequested("character")
        assert msg.item_type == "character"
        
        # CreateRequested
        msg = CreateRequested("prompt")
        assert msg.item_type == "prompt"
        
        # RefreshRequested
        msg = RefreshRequested("dictionary")
        assert msg.list_type == "dictionary"
    
    def test_conversation_view_messages(self):
        """Test conversation view widget messages."""
        # MessageSelected
        msg = MessageSelected(1, {"content": "test"})
        assert msg.message_id == 1
        assert msg.message_data["content"] == "test"
        
        # MessageEditRequested
        msg = MessageEditRequested(2)
        assert msg.message_id == 2
        
        # MessageDeleteRequested
        msg = MessageDeleteRequested(3)
        assert msg.message_id == 3
        
        # RegenerateRequested
        msg = RegenerateRequested(4)
        assert msg.message_id == 4
        
        # ContinueConversationRequested
        msg = ContinueConversationRequested()
        assert msg is not None
    
    def test_character_messages(self):
        """Test character widget messages."""
        # EditCharacterRequested
        msg = EditCharacterRequested(1)
        assert msg.character_id == 1
        
        # CloneCharacterRequested
        msg = CloneCharacterRequested(2)
        assert msg.character_id == 2
        
        # ExportCharacterRequested
        msg = ExportCharacterRequested(3, "json")
        assert msg.character_id == 3
        assert msg.format == "json"
        
        # DeleteCharacterRequested
        msg = DeleteCharacterRequested(4)
        assert msg.character_id == 4
        
        # StartChatRequested
        msg = StartChatRequested(5)
        assert msg.character_id == 5
        
        # CharacterSaveRequested
        data = {"name": "Alice"}
        msg = CharacterSaveRequested(data)
        assert msg.character_data == data
        
        # CharacterFieldGenerateRequested
        msg = CharacterFieldGenerateRequested("description", "Alice")
        assert msg.field_name == "description"
        assert msg.character_name == "Alice"
    
    def test_prompt_messages(self):
        """Test prompt widget messages."""
        # PromptSaveRequested
        data = {"name": "Test Prompt"}
        msg = PromptSaveRequested(data)
        assert msg.prompt_data == data
        
        # PromptDeleteRequested
        msg = PromptDeleteRequested(1)
        assert msg.prompt_id == 1
        
        # PromptTestRequested
        test_data = {"prompt": "test", "values": {}}
        msg = PromptTestRequested(test_data)
        assert msg.prompt_data == test_data
        
        # PromptVariableAdded
        msg = PromptVariableAdded("var1", "text")
        assert msg.variable_name == "var1"
        assert msg.variable_type == "text"
        
        # PromptVariableRemoved
        msg = PromptVariableRemoved("var2")
        assert msg.variable_name == "var2"
    
    def test_dictionary_messages(self):
        """Test dictionary widget messages."""
        # DictionarySaveRequested
        data = {"name": "Test Dict"}
        msg = DictionarySaveRequested(data)
        assert msg.dictionary_data == data
        
        # DictionaryDeleteRequested
        msg = DictionaryDeleteRequested(1)
        assert msg.dictionary_id == 1
        
        # DictionaryEntryAdded
        entry = {"key": "test", "value": "value"}
        msg = DictionaryEntryAdded(entry)
        assert msg.entry == entry
        
        # DictionaryEntryRemoved
        msg = DictionaryEntryRemoved(5)
        assert msg.entry_index == 5
        
        # DictionaryEntryUpdated
        msg = DictionaryEntryUpdated(3, {"key": "updated"})
        assert msg.entry_index == 3
        assert msg.entry_data["key"] == "updated"
        
        # DictionaryImportRequested
        msg = DictionaryImportRequested("json")
        assert msg.format == "json"
        
        # DictionaryExportRequested
        msg = DictionaryExportRequested("csv")
        assert msg.format == "csv"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])