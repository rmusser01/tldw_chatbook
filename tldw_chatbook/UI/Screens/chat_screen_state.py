"""Chat screen state management following Textual best practices.

This module provides centralized state management for the chat screen,
ensuring that user conversations, typed messages, and UI state are
preserved when navigating between screens.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from loguru import logger

logger = logger.bind(module="ChatScreenState")


@dataclass
class MessageData:
    """Cached message data for quick restoration."""
    message_id: str
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: datetime
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_streaming: bool = False
    is_edited: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'message_id': self.message_id,
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'attachments': self.attachments,
            'metadata': self.metadata,
            'is_streaming': self.is_streaming,
            'is_edited': self.is_edited,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MessageData':
        """Create from dictionary."""
        timestamp = data.get('timestamp')
        if timestamp and isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        return cls(
            message_id=data.get('message_id', ''),
            role=data.get('role', 'user'),
            content=data.get('content', ''),
            timestamp=timestamp or datetime.now(),
            attachments=data.get('attachments', []),
            metadata=data.get('metadata', {}),
            is_streaming=data.get('is_streaming', False),
            is_edited=data.get('is_edited', False),
        )


@dataclass
class TabState:
    """State for a single chat tab."""
    tab_id: str
    title: str
    conversation_id: Optional[str] = None
    character_id: Optional[int] = None
    character_name: Optional[str] = None
    
    # Input state
    input_text: str = ""
    cursor_position: int = 0
    
    # UI state
    scroll_position: int = 0
    is_active: bool = False
    
    # Attachments
    pending_attachments: List[Dict[str, Any]] = field(default_factory=list)
    
    # Message cache
    messages: List[MessageData] = field(default_factory=list)
    
    # Session metadata
    is_ephemeral: bool = True
    has_unsaved_changes: bool = False
    created_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    
    # Settings overrides
    system_prompt_override: Optional[str] = None
    temperature_override: Optional[float] = None
    max_tokens_override: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'tab_id': self.tab_id,
            'title': self.title,
            'conversation_id': self.conversation_id,
            'character_id': self.character_id,
            'character_name': self.character_name,
            'input_text': self.input_text,
            'cursor_position': self.cursor_position,
            'scroll_position': self.scroll_position,
            'is_active': self.is_active,
            'pending_attachments': self.pending_attachments,
            'messages': [msg.to_dict() for msg in self.messages],
            'is_ephemeral': self.is_ephemeral,
            'has_unsaved_changes': self.has_unsaved_changes,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_activity': self.last_activity.isoformat() if self.last_activity else None,
            'system_prompt_override': self.system_prompt_override,
            'temperature_override': self.temperature_override,
            'max_tokens_override': self.max_tokens_override,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TabState':
        """Create from dictionary."""
        created_at = data.get('created_at')
        if created_at and isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        
        last_activity = data.get('last_activity')
        if last_activity and isinstance(last_activity, str):
            last_activity = datetime.fromisoformat(last_activity)
        
        messages = [MessageData.from_dict(msg) for msg in data.get('messages', [])]
        
        return cls(
            tab_id=data.get('tab_id', ''),
            title=data.get('title', 'New Chat'),
            conversation_id=data.get('conversation_id'),
            character_id=data.get('character_id'),
            character_name=data.get('character_name'),
            input_text=data.get('input_text', ''),
            cursor_position=data.get('cursor_position', 0),
            scroll_position=data.get('scroll_position', 0),
            is_active=data.get('is_active', False),
            pending_attachments=data.get('pending_attachments', []),
            messages=messages,
            is_ephemeral=data.get('is_ephemeral', True),
            has_unsaved_changes=data.get('has_unsaved_changes', False),
            created_at=created_at,
            last_activity=last_activity,
            system_prompt_override=data.get('system_prompt_override'),
            temperature_override=data.get('temperature_override'),
            max_tokens_override=data.get('max_tokens_override'),
        )


@dataclass
class ChatScreenState:
    """
    Complete state for the chat screen.
    
    This dataclass encapsulates all state needed to fully restore
    the chat screen when returning from another screen, following
    Textual's best practices for state management.
    """
    
    # Tab management
    tabs: List[TabState] = field(default_factory=list)
    active_tab_id: Optional[str] = None
    tab_order: List[str] = field(default_factory=list)  # Order of tabs in UI
    
    # UI state
    left_sidebar_collapsed: bool = False
    right_sidebar_collapsed: bool = False
    settings_sidebar_visible: bool = False
    
    # Voice input state
    voice_input_active: bool = False
    voice_input_language: str = "en-US"
    
    # Global attachments (shared across tabs)
    global_attachments: Dict[str, Any] = field(default_factory=dict)
    
    # Preferences
    show_timestamps: bool = True
    show_avatars: bool = True
    compact_mode: bool = False
    
    # Metadata
    last_saved: Optional[datetime] = None
    version: str = "1.0"
    
    def get_active_tab(self) -> Optional[TabState]:
        """Get the currently active tab."""
        if not self.active_tab_id:
            return None
        
        for tab in self.tabs:
            if tab.tab_id == self.active_tab_id:
                return tab
        return None
    
    def get_tab_by_id(self, tab_id: str) -> Optional[TabState]:
        """Get a tab by its ID."""
        for tab in self.tabs:
            if tab.tab_id == tab_id:
                return tab
        return None
    
    def add_tab(self, tab: TabState) -> None:
        """Add a new tab to the state."""
        self.tabs.append(tab)
        self.tab_order.append(tab.tab_id)
        logger.debug(f"Added tab {tab.tab_id} to state")
    
    def remove_tab(self, tab_id: str) -> bool:
        """Remove a tab from the state."""
        tab = self.get_tab_by_id(tab_id)
        if tab:
            self.tabs.remove(tab)
            if tab_id in self.tab_order:
                self.tab_order.remove(tab_id)
            if self.active_tab_id == tab_id:
                # Switch to next available tab
                self.active_tab_id = self.tab_order[0] if self.tab_order else None
            logger.debug(f"Removed tab {tab_id} from state")
            return True
        return False
    
    def update_tab_order(self, new_order: List[str]) -> None:
        """Update the order of tabs."""
        # Validate that all tab IDs exist
        existing_ids = {tab.tab_id for tab in self.tabs}
        if set(new_order) == existing_ids:
            self.tab_order = new_order
            logger.debug(f"Updated tab order: {new_order}")
    
    def validate(self) -> bool:
        """Validate the state for consistency."""
        # Check that active tab exists
        if self.active_tab_id and not self.get_tab_by_id(self.active_tab_id):
            logger.warning(f"Active tab {self.active_tab_id} not found in tabs")
            return False
        
        # Check tab order consistency (but allow empty tab_order for single tabs)
        tab_ids = {tab.tab_id for tab in self.tabs}
        order_ids = set(self.tab_order) if self.tab_order else set()
        
        # If tab_order is empty but we have tabs, populate it
        if not self.tab_order and self.tabs:
            self.tab_order = [tab.tab_id for tab in self.tabs]
            logger.debug(f"Auto-populated tab_order: {self.tab_order}")
            return True
        
        # Only fail if tab_order has entries but they don't match
        if self.tab_order and tab_ids != order_ids:
            logger.warning(f"Tab order doesn't match tab list. Tab IDs: {tab_ids}, Order IDs: {order_ids}")
            return False
        
        # Check for duplicate tab IDs
        seen_ids = set()
        for tab in self.tabs:
            if tab.tab_id in seen_ids:
                logger.warning(f"Duplicate tab ID: {tab.tab_id}")
                return False
            seen_ids.add(tab.tab_id)
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'tabs': [tab.to_dict() for tab in self.tabs],
            'active_tab_id': self.active_tab_id,
            'tab_order': self.tab_order,
            'left_sidebar_collapsed': self.left_sidebar_collapsed,
            'right_sidebar_collapsed': self.right_sidebar_collapsed,
            'settings_sidebar_visible': self.settings_sidebar_visible,
            'voice_input_active': self.voice_input_active,
            'voice_input_language': self.voice_input_language,
            'global_attachments': self.global_attachments,
            'show_timestamps': self.show_timestamps,
            'show_avatars': self.show_avatars,
            'compact_mode': self.compact_mode,
            'last_saved': self.last_saved.isoformat() if self.last_saved else None,
            'version': self.version,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatScreenState':
        """Create from dictionary."""
        last_saved = data.get('last_saved')
        if last_saved and isinstance(last_saved, str):
            last_saved = datetime.fromisoformat(last_saved)
        
        tabs = [TabState.from_dict(tab_data) for tab_data in data.get('tabs', [])]
        
        return cls(
            tabs=tabs,
            active_tab_id=data.get('active_tab_id'),
            tab_order=data.get('tab_order', []),
            left_sidebar_collapsed=data.get('left_sidebar_collapsed', False),
            right_sidebar_collapsed=data.get('right_sidebar_collapsed', False),
            settings_sidebar_visible=data.get('settings_sidebar_visible', False),
            voice_input_active=data.get('voice_input_active', False),
            voice_input_language=data.get('voice_input_language', 'en-US'),
            global_attachments=data.get('global_attachments', {}),
            show_timestamps=data.get('show_timestamps', True),
            show_avatars=data.get('show_avatars', True),
            compact_mode=data.get('compact_mode', False),
            last_saved=last_saved,
            version=data.get('version', '1.0'),
        )
    
    def create_snapshot(self) -> 'ChatScreenState':
        """Create a deep copy snapshot of the current state."""
        import copy
        return copy.deepcopy(self)