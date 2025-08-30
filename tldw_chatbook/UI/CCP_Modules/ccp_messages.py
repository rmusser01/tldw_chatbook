"""Message classes for CCP window inter-component communication.

Following Textual's message system for loose coupling between components.
"""

from typing import Optional, Dict, Any, List
from textual.message import Message


class CCPMessage(Message):
    """Base message class for all CCP-related messages."""
    
    def __init__(self, sender: Any = None) -> None:
        super().__init__()
        self.sender = sender


class ConversationMessage(CCPMessage):
    """Messages related to conversation operations."""
    
    class Selected(CCPMessage):
        """A conversation was selected."""
        def __init__(self, conversation_id: int, title: str, sender: Any = None) -> None:
            super().__init__(sender)
            self.conversation_id = conversation_id
            self.title = title
    
    class Loaded(CCPMessage):
        """A conversation was loaded."""
        def __init__(self, conversation_id: int, messages: List[Dict], sender: Any = None) -> None:
            super().__init__(sender)
            self.conversation_id = conversation_id
            self.messages = messages
    
    class Created(CCPMessage):
        """A new conversation was created."""
        def __init__(self, conversation_id: int, title: str, sender: Any = None) -> None:
            super().__init__(sender)
            self.conversation_id = conversation_id
            self.title = title
    
    class Updated(CCPMessage):
        """Conversation details were updated."""
        def __init__(self, conversation_id: int, title: str, keywords: str, sender: Any = None) -> None:
            super().__init__(sender)
            self.conversation_id = conversation_id
            self.title = title
            self.keywords = keywords
    
    class Deleted(CCPMessage):
        """A conversation was deleted."""
        def __init__(self, conversation_id: int, sender: Any = None) -> None:
            super().__init__(sender)
            self.conversation_id = conversation_id
    
    class SearchRequested(CCPMessage):
        """Search conversations requested."""
        def __init__(self, search_term: str, search_type: str = "title", sender: Any = None) -> None:
            super().__init__(sender)
            self.search_term = search_term
            self.search_type = search_type  # "title", "content", "tags"


class CharacterMessage(CCPMessage):
    """Messages related to character operations."""
    
    class Selected(CCPMessage):
        """A character was selected."""
        def __init__(self, character_id: int, name: str, sender: Any = None) -> None:
            super().__init__(sender)
            self.character_id = character_id
            self.name = name
    
    class Loaded(CCPMessage):
        """A character card was loaded."""
        def __init__(self, character_id: int, card_data: Dict[str, Any], sender: Any = None) -> None:
            super().__init__(sender)
            self.character_id = character_id
            self.card_data = card_data
    
    class Created(CCPMessage):
        """A new character was created."""
        def __init__(self, character_id: int, name: str, card_data: Dict[str, Any], sender: Any = None) -> None:
            super().__init__(sender)
            self.character_id = character_id
            self.name = name
            self.card_data = card_data
    
    class Updated(CCPMessage):
        """Character details were updated."""
        def __init__(self, character_id: int, card_data: Dict[str, Any], sender: Any = None) -> None:
            super().__init__(sender)
            self.character_id = character_id
            self.card_data = card_data
    
    class Deleted(CCPMessage):
        """A character was deleted."""
        def __init__(self, character_id: int, sender: Any = None) -> None:
            super().__init__(sender)
            self.character_id = character_id
    
    class ImportRequested(CCPMessage):
        """Import character card requested."""
        def __init__(self, file_path: str, sender: Any = None) -> None:
            super().__init__(sender)
            self.file_path = file_path
    
    class ExportRequested(CCPMessage):
        """Export character card requested."""
        def __init__(self, character_id: int, file_path: str, sender: Any = None) -> None:
            super().__init__(sender)
            self.character_id = character_id
            self.file_path = file_path
    
    class GenerateFieldRequested(CCPMessage):
        """Generate character field using AI requested."""
        def __init__(self, field_name: str, context: Dict[str, Any], sender: Any = None) -> None:
            super().__init__(sender)
            self.field_name = field_name
            self.context = context


class PromptMessage(CCPMessage):
    """Messages related to prompt operations."""
    
    class Selected(CCPMessage):
        """A prompt was selected."""
        def __init__(self, prompt_id: int, name: str, sender: Any = None) -> None:
            super().__init__(sender)
            self.prompt_id = prompt_id
            self.name = name
    
    class Loaded(CCPMessage):
        """A prompt was loaded."""
        def __init__(self, prompt_id: int, prompt_data: Dict[str, Any], sender: Any = None) -> None:
            super().__init__(sender)
            self.prompt_id = prompt_id
            self.prompt_data = prompt_data
    
    class Created(CCPMessage):
        """A new prompt was created."""
        def __init__(self, prompt_id: int, name: str, prompt_data: Dict[str, Any], sender: Any = None) -> None:
            super().__init__(sender)
            self.prompt_id = prompt_id
            self.name = name
            self.prompt_data = prompt_data
    
    class Updated(CCPMessage):
        """Prompt details were updated."""
        def __init__(self, prompt_id: int, prompt_data: Dict[str, Any], sender: Any = None) -> None:
            super().__init__(sender)
            self.prompt_id = prompt_id
            self.prompt_data = prompt_data
    
    class Deleted(CCPMessage):
        """A prompt was deleted."""
        def __init__(self, prompt_id: int, sender: Any = None) -> None:
            super().__init__(sender)
            self.prompt_id = prompt_id
    
    class SearchRequested(CCPMessage):
        """Search prompts requested."""
        def __init__(self, search_term: str, sender: Any = None) -> None:
            super().__init__(sender)
            self.search_term = search_term


class DictionaryMessage(CCPMessage):
    """Messages related to dictionary/world book operations."""
    
    class Selected(CCPMessage):
        """A dictionary was selected."""
        def __init__(self, dictionary_id: int, name: str, sender: Any = None) -> None:
            super().__init__(sender)
            self.dictionary_id = dictionary_id
            self.name = name
    
    class Loaded(CCPMessage):
        """A dictionary was loaded."""
        def __init__(self, dictionary_id: int, dictionary_data: Dict[str, Any], sender: Any = None) -> None:
            super().__init__(sender)
            self.dictionary_id = dictionary_id
            self.dictionary_data = dictionary_data
    
    class Created(CCPMessage):
        """A new dictionary was created."""
        def __init__(self, dictionary_id: int, name: str, dictionary_data: Dict[str, Any], sender: Any = None) -> None:
            super().__init__(sender)
            self.dictionary_id = dictionary_id
            self.name = name
            self.dictionary_data = dictionary_data
    
    class Updated(CCPMessage):
        """Dictionary details were updated."""
        def __init__(self, dictionary_id: int, dictionary_data: Dict[str, Any], sender: Any = None) -> None:
            super().__init__(sender)
            self.dictionary_id = dictionary_id
            self.dictionary_data = dictionary_data
    
    class Deleted(CCPMessage):
        """A dictionary was deleted."""
        def __init__(self, dictionary_id: int, sender: Any = None) -> None:
            super().__init__(sender)
            self.dictionary_id = dictionary_id
    
    class EntryAdded(CCPMessage):
        """A dictionary entry was added."""
        def __init__(self, dictionary_id: int, entry_data: Dict[str, Any], sender: Any = None) -> None:
            super().__init__(sender)
            self.dictionary_id = dictionary_id
            self.entry_data = entry_data
    
    class EntryRemoved(CCPMessage):
        """A dictionary entry was removed."""
        def __init__(self, dictionary_id: int, entry_key: str, sender: Any = None) -> None:
            super().__init__(sender)
            self.dictionary_id = dictionary_id
            self.entry_key = entry_key


class SidebarMessage(CCPMessage):
    """Messages related to sidebar operations."""
    
    class ToggleRequested(CCPMessage):
        """Sidebar toggle requested."""
        def __init__(self, sender: Any = None) -> None:
            super().__init__(sender)
    
    class CollapsibleToggled(CCPMessage):
        """A collapsible section was toggled."""
        def __init__(self, section_id: str, collapsed: bool, sender: Any = None) -> None:
            super().__init__(sender)
            self.section_id = section_id
            self.collapsed = collapsed
    
    class SearchFocused(CCPMessage):
        """Search input was focused."""
        def __init__(self, search_type: str, sender: Any = None) -> None:
            super().__init__(sender)
            self.search_type = search_type


class ViewChangeMessage(CCPMessage):
    """Messages for view changes in the main content area."""
    
    class Requested(CCPMessage):
        """View change requested."""
        def __init__(self, view_name: str, context: Optional[Dict[str, Any]] = None, sender: Any = None) -> None:
            super().__init__(sender)
            self.view_name = view_name  # "conversations", "character_card", "character_editor", "prompt_editor", etc.
            self.context = context or {}
    
    class Changed(CCPMessage):
        """View was changed."""
        def __init__(self, old_view: str, new_view: str, sender: Any = None) -> None:
            super().__init__(sender)
            self.old_view = old_view
            self.new_view = new_view