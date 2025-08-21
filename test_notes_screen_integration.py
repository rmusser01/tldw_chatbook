#!/usr/bin/env python
"""Quick integration test for the refactored NotesScreen."""

import asyncio
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from textual.app import App
from textual import on
from textual.widgets import Button

from tldw_chatbook.UI.Screens.notes_screen import NotesScreen, NotesScreenState


class NotesScreenTestApp(App):
    """Test app to verify NotesScreen works properly."""
    
    CSS = """
    Screen {
        align: center middle;
    }
    """
    
    def __init__(self):
        super().__init__()
        self.notes_service = None  # Mock service
        
    async def on_mount(self):
        """Push the NotesScreen when app mounts."""
        logger.info("Test app mounted, pushing NotesScreen")
        await self.push_screen(NotesScreen(self))
    
    @on(Button.Pressed)
    async def handle_button(self, event):
        """Handle any button presses."""
        logger.info(f"Button pressed: {event.button.id}")


def test_notes_screen_state():
    """Test the NotesScreenState dataclass."""
    print("\n=== Testing NotesScreenState ===")
    
    # Create state
    state = NotesScreenState()
    print(f"✓ Created state with {len(state.__dict__)} attributes")
    
    # Check defaults
    assert state.auto_save_enabled == True
    print("✓ Auto-save enabled by default")
    
    assert state.has_unsaved_changes == False
    print("✓ No unsaved changes initially")
    
    assert state.sort_by == "date_created"
    print("✓ Default sort is by date_created")
    
    # Modify state
    state.selected_note_id = 123
    state.has_unsaved_changes = True
    assert state.selected_note_id == 123
    assert state.has_unsaved_changes == True
    print("✓ State modifications work correctly")
    
    print("\n=== All state tests passed! ===\n")


def test_notes_screen_import():
    """Test that NotesScreen and components import correctly."""
    print("\n=== Testing imports ===")
    
    # Import main screen
    from tldw_chatbook.UI.Screens.notes_screen import (
        NotesScreen, 
        NotesScreenState,
        NoteSelected,
        NoteSaved,
        NoteDeleted,
        AutoSaveTriggered,
        SyncRequested
    )
    print("✓ NotesScreen and messages import successfully")
    
    # Import new widgets
    from tldw_chatbook.Widgets.Note_Widgets.notes_editor_widget import NotesEditorWidget
    print("✓ NotesEditorWidget imports successfully")
    
    from tldw_chatbook.Widgets.Note_Widgets.notes_status_bar import NotesStatusBar
    print("✓ NotesStatusBar imports successfully")
    
    from tldw_chatbook.Widgets.Note_Widgets.notes_toolbar import NotesToolbar
    print("✓ NotesToolbar imports successfully")
    
    print("\n=== All import tests passed! ===\n")


async def test_notes_screen_lifecycle():
    """Test NotesScreen lifecycle methods."""
    print("\n=== Testing NotesScreen lifecycle ===")
    
    # Mock app
    class MockApp:
        def __init__(self):
            self.notes_service = None
        
        def notify(self, msg, severity='info'):
            print(f"  [{severity}] {msg}")
    
    app = MockApp()
    screen = NotesScreen(app)
    print("✓ NotesScreen created")
    
    # Test mount
    await screen.on_mount()
    print("✓ on_mount executed without errors")
    
    # Test state save/restore
    saved_state = screen.save_state()
    print(f"✓ State saved with {len(saved_state)} keys")
    
    screen.restore_state(saved_state)
    print("✓ State restored successfully")
    
    # Test unmount
    screen.on_unmount()
    print("✓ on_unmount executed without errors")
    
    print("\n=== All lifecycle tests passed! ===\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("NOTES SCREEN INTEGRATION TEST")
    print("=" * 60)
    
    # Run synchronous tests
    test_notes_screen_import()
    test_notes_screen_state()
    
    # Run async tests
    asyncio.run(test_notes_screen_lifecycle())
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! ✅")
    print("The refactored NotesScreen is working correctly.")
    print("=" * 60)
    
    # Optional: Run the test app
    if len(sys.argv) > 1 and sys.argv[1] == "--run-app":
        print("\nStarting test app... (Press Ctrl+C to exit)")
        app = NotesScreenTestApp()
        app.run()


if __name__ == "__main__":
    main()