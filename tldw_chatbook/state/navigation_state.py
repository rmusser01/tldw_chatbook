"""
Navigation state management.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class NavigationState:
    """Manages navigation-related state."""
    
    # Current navigation
    current_screen: str = "chat"
    previous_screen: Optional[str] = None
    
    # Navigation history
    history: List[str] = field(default_factory=list)
    max_history: int = 50
    
    # Screen states
    splash_active: bool = False
    loading: bool = False
    
    def navigate_to(self, screen: str) -> None:
        """Navigate to a new screen."""
        if self.current_screen != screen:
            self.previous_screen = self.current_screen
            self.current_screen = screen
            
            # Maintain history
            self.history.append(screen)
            if len(self.history) > self.max_history:
                self.history.pop(0)
    
    def go_back(self) -> Optional[str]:
        """Navigate to previous screen."""
        if self.previous_screen:
            screen = self.previous_screen
            self.navigate_to(screen)
            return screen
        return None
    
    def clear_history(self) -> None:
        """Clear navigation history."""
        self.history.clear()
        self.previous_screen = None