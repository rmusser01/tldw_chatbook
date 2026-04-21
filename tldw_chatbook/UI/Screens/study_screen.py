"""
Study Screen
Screen wrapper for Study functionality in screen-based navigation.
"""

from textual.screen import Screen
from textual.app import ComposeResult
from textual.reactive import reactive
from typing import Optional, List, Dict, Any
from loguru import logger

from ..Study_Window import StudyWindow


class StudyScreen(Screen):
    """Screen wrapper for Study functionality."""
    
    # Screen-specific state
    current_study_session: reactive[Optional[Dict[str, Any]]] = reactive(None)
    study_materials: reactive[List[str]] = reactive([])
    is_studying: reactive[bool] = reactive(False)
    current_topic: reactive[str] = reactive("")
    
    def compose(self) -> ComposeResult:
        """Compose the Study screen with the Study window."""
        logger.info("Composing Study screen")
        yield StudyWindow()
    
    async def on_mount(self) -> None:
        """Initialize Study features when screen is mounted."""
        logger.info("Study screen mounted")
        
        # Get the Study window
        study_window = self.query_one(StudyWindow)
        
        # Load any saved study sessions
        if hasattr(study_window, 'load_saved_sessions'):
            await study_window.load_saved_sessions()
        
        # Initialize study features
        if hasattr(study_window, 'initialize'):
            await study_window.initialize()
    
    async def on_screen_suspend(self) -> None:
        """Save state when screen is suspended (navigated away)."""
        logger.debug("Study screen suspended")
        
        # Save current study session if active
        if self.is_studying and self.current_study_session:
            study_window = self.query_one(StudyWindow)
            if hasattr(study_window, 'save_session'):
                await study_window.save_session(self.current_study_session)
        
        self.is_studying = False
    
    async def on_screen_resume(self) -> None:
        """Restore state when screen is resumed."""
        logger.debug("Study screen resumed")
        
        # Restore study session if it was active
        if self.current_study_session:
            study_window = self.query_one(StudyWindow)
            if hasattr(study_window, 'restore_session'):
                await study_window.restore_session(self.current_study_session)
    
    def update_study_materials(self, materials: List[str]) -> None:
        """Update the list of study materials."""
        self.study_materials = materials
        logger.debug(f"Updated study materials: {len(materials)} items")
    
    def start_study_session(self, topic: str) -> None:
        """Start a new study session."""
        self.current_topic = topic
        self.is_studying = True
        self.current_study_session = {
            "topic": topic,
            "start_time": None,  # Will be set by StudyWindow
            "materials": self.study_materials
        }
        logger.info(f"Started study session for topic: {topic}")