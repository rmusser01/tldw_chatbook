#!/usr/bin/env python3

"""
Test the Tools Settings Window content visibility after fixes
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from textual.app import App
from tldw_chatbook.UI.Tools_Settings_Window import ToolsSettingsWindow


class ContentTestApp(App):
    """Test app for content visibility"""
    
    def compose(self):
        tools_window = ToolsSettingsWindow(self, id="tools-settings-content-test")
        yield tools_window


def main():
    """Run the content test app"""
    print("Testing Tools Settings content visibility after fixes...")
    print("Expected: Form elements should now be visible in General Settings")
    print("Look for: input fields, select dropdowns, checkboxes, labels")
    
    app = ContentTestApp()
    app.run()


if __name__ == "__main__":
    main()